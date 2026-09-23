#!/usr/bin/env python3
"""
Shared FITS/XISF -> displayable QPixmap conversion.

Consolidates what used to be several near-identical copies of the same
percentile-stretch pipeline scattered across DSOGallery.py and DSODetail.py
(one per call site, one per format). Both formats funnel through the same
stretch_array_to_qimage() once their pixel data has been read into a plain
numpy array - FITS via astropy.io.fits, XISF via XISFReader.read_xisf_pixels().
"""

import os
import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)

FITS_EXTENSIONS = ('.fits', '.fit', '.fts')
XISF_EXTENSIONS = ('.xisf',)


# Percentile limits are estimated from a strided sample of about this many
# pixels - sorting every pixel of a large stack takes seconds, and the sampled
# limits are visually identical
_PERCENTILE_SAMPLE_PIXELS = 1_000_000


def _stretch_channel_to_uint8(channel):
    """Linear-stretch a single 2D channel between its 0.25 and 99.75 percentiles
    (the central 99.5%, matching astropy's simple_norm(percent=99.5) used
    previously) and return it as uint8. Works in float32 to keep memory down."""
    data = channel.astype(np.float32)  # always a copy, so in-place ops are safe
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    step = max(1, int(np.sqrt(data.size / _PERCENTILE_SAMPLE_PIXELS)))
    lo, hi = np.percentile(data[::step, ::step], [0.25, 99.75])

    if hi > lo:
        data -= lo
        data *= 255.0 / (hi - lo)
    else:
        data *= 255.0
    np.clip(data, 0, 255, out=data)
    return data.astype(np.uint8)


def stretch_array_to_qimage(image_data):
    """Percentile-stretch a raw (h, w) grayscale or (h, w, 3) RGB numpy array
    (any numeric dtype/range - FITS and XISF both commonly use 16-bit integer
    or 32-bit float samples) into an 8-bit QImage.

    Returns a QImage detached from the numpy buffer (via .copy()) so it stays
    valid after the array that produced it goes out of scope or, for FITS,
    after the source file's memory-mapped HDU is closed. Safe to call from a
    worker thread.
    """
    is_rgb = image_data.ndim == 3 and image_data.shape[2] == 3

    if is_rgb:
        h, w = image_data.shape[:2]
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = _stretch_channel_to_uint8(image_data[:, :, c])
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    else:
        img8 = _stretch_channel_to_uint8(image_data)
        h, w = img8.shape
        qimage = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)

    return qimage.copy()


def _read_fits_array(file_path):
    """Open a FITS file and return its primary image data, normalized down to
    (h, w) or (h, w, 3) - the dimensionality handling FITS files have always
    needed (extra axes, or RGB planes stored channel-first instead of
    interleaved). The returned array is fully materialized (np.array(...))
    while the file is still open, since astropy may hand back a
    memory-mapped view that becomes invalid once the file is closed."""
    from astropy.io import fits

    with fits.open(file_path) as hdul:
        image_data = None
        for hdu in hdul:
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                image_data = hdu.data
                break

        if image_data is None:
            return None

        if image_data.ndim == 3 and image_data.shape[0] == 3:
            image_data = np.transpose(image_data, (1, 2, 0))
        elif image_data.ndim == 3 and image_data.shape[2] != 3:
            image_data = image_data[0]
        elif image_data.ndim == 4:
            image_data = image_data[0, 0]

        return np.array(image_data)


def load_astro_qimage(file_path):
    """Decode a FITS or XISF file into a displayable QImage. Unlike QPixmap,
    QImage is safe to create in a worker thread.

    Returns None on any failure (missing file, unsupported format/compression,
    corrupt data, etc.).
    """
    if not file_path or not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in FITS_EXTENSIONS:
            image_data = _read_fits_array(file_path)
        elif ext in XISF_EXTENSIONS:
            from XISFReader import read_xisf_pixels
            image_data = read_xisf_pixels(file_path)
        else:
            return None

        if image_data is None:
            return None

        qimage = stretch_array_to_qimage(image_data)
    except Exception as e:
        logger.debug(f"Could not load astro image {file_path}: {e}")
        return None

    return None if qimage.isNull() else qimage


def load_image_qimage(file_path):
    """Load any supported image (FITS/XISF or a standard format Qt can read)
    as a full-resolution QImage. Safe to call from a worker thread.
    Returns (qimage, None) on success or (None, error message) on failure."""
    if not file_path or not os.path.exists(file_path):
        return None, "File not found"

    ext = os.path.splitext(file_path)[1].lower()
    if ext in FITS_EXTENSIONS + XISF_EXTENSIONS:
        qimage = load_astro_qimage(file_path)
        return (qimage, None) if qimage is not None else (None, "Failed to load FITS/XISF file")

    from PySide6.QtGui import QImageReader
    QImageReader.setAllocationLimit(1024)  # MB - large stacked images exceed Qt's default
    reader = QImageReader(file_path)
    reader.setAutoTransform(True)
    qimage = reader.read()
    if qimage.isNull():
        return None, reader.errorString() or "Unknown error"
    return qimage, None


def load_astro_pixmap(file_path, max_dim=None):
    """Decode a FITS or XISF file into a displayable QPixmap.

    Returns None on any failure (missing file, unsupported format/compression,
    corrupt data, etc.) - callers already treat None as "no preview available".
    """
    qimage = load_astro_qimage(file_path)
    if qimage is None:
        return None

    # Scale the QImage before converting - cheaper than scaling the full pixmap
    if max_dim:
        qimage = qimage.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    pixmap = QPixmap.fromImage(qimage)
    return None if pixmap.isNull() else pixmap
