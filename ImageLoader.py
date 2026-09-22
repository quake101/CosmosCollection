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


def _normalize_channel(channel):
    """Percentile-stretch a single 2D channel to the 0-1 range."""
    from astropy.visualization import simple_norm
    try:
        norm = simple_norm(channel, stretch='linear', percent=99.5)
        return norm(channel)
    except Exception:
        lo, hi = np.percentile(channel, [0.5, 99.5])
        return (channel - lo) / (hi - lo) if hi > lo else channel


def stretch_array_to_qimage(image_data):
    """Percentile-stretch a raw (h, w) grayscale or (h, w, 3) RGB numpy array
    (any numeric dtype/range - FITS and XISF both commonly use 16-bit integer
    or 32-bit float samples) into an 8-bit QImage.

    Returns a QImage detached from the numpy buffer (via .copy()) so it stays
    valid after the array that produced it goes out of scope or, for FITS,
    after the source file's memory-mapped HDU is closed.
    """
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    is_rgb = image_data.ndim == 3 and image_data.shape[2] == 3

    if is_rgb:
        normalized = np.zeros(image_data.shape, dtype=float)
        for c in range(3):
            normalized[:, :, c] = _normalize_channel(image_data[:, :, c])
        rgb = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        if not rgb.flags['C_CONTIGUOUS']:
            rgb = np.ascontiguousarray(rgb)
        h, w, c = rgb.shape
        qimage = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
    else:
        normalized = np.clip(_normalize_channel(image_data), 0, 1)
        img8 = (normalized * 255).astype(np.uint8)
        if not img8.flags['C_CONTIGUOUS']:
            img8 = np.ascontiguousarray(img8)
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


def load_astro_pixmap(file_path, max_dim=None):
    """Decode a FITS or XISF file into a displayable QPixmap.

    Returns None on any failure (missing file, unsupported format/compression,
    corrupt data, etc.) - callers already treat None as "no preview available".
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
        pixmap = QPixmap.fromImage(qimage)
    except Exception as e:
        logger.debug(f"Could not load astro image {file_path}: {e}")
        return None

    if pixmap.isNull():
        return None

    if max_dim:
        pixmap = pixmap.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return pixmap
