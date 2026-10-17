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
import threading

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QImageReader, QPixmap, QTransform

logger = logging.getLogger(__name__)

# Qt image file decodes and encodes one at a time, app-wide - several at once
# on worker threads corrupted the heap now and then (a hard crash). The GIL is
# released while one runs, so the UI stays responsive meanwhile.
QT_IMAGE_IO_LOCK = threading.Lock()

FITS_EXTENSIONS = ('.fits', '.fit', '.fts')
XISF_EXTENSIONS = ('.xisf',)


# Percentile limits are estimated from a strided sample of about this many
# pixels - sorting every pixel of a large stack takes seconds, and the sampled
# limits are visually identical
_PERCENTILE_SAMPLE_PIXELS = 1_000_000

# Reads that report progress go in pieces this big. From the NAS, one read of
# a whole file is fastest (Windows keeps many requests in flight); each piece
# waits for the last, so small ones are slow: 4 MB pieces read at ~55% of one
# read's speed, 64 MB at ~87% - with a progress update every half second or so
_READ_CHUNK = 64 * 1024 * 1024


def read_file_bytes(file_path, progress=None, offset=0, size=None):
    """A file's bytes: size bytes from offset, or the rest of the file. With
    progress, they're read in pieces, calling progress(done, total) after each
    - it may raise to stop the read (e.g. a load that's no longer wanted).
    Without, in one go."""
    with open(file_path, 'rb') as f:
        if offset:
            f.seek(offset)
        if progress is None:
            return f.read() if size is None else f.read(size)
        total = os.fstat(f.fileno()).st_size - offset if size is None else size
        data = bytearray(total)
        done = 0
        progress(done, total)
        with memoryview(data) as view:
            while done < total:
                count = f.readinto(view[done:done + _READ_CHUNK])
                if not count:
                    break  # The file is shorter than it said
                done += count
                progress(done, total)
        return data if done == total else data[:done]


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
        # The channels at once - numpy lets go of the GIL for big arrays, so a
        # large stack stretches in about half the time
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(3) as pool:
            channels = pool.map(lambda c: _stretch_channel_to_uint8(image_data[:, :, c]), range(3))
            for c, channel in enumerate(channels):
                rgb[:, :, c] = channel
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    else:
        img8 = _stretch_channel_to_uint8(image_data)
        h, w = img8.shape
        qimage = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)

    return qimage.copy()


def _read_fits_array(file_path, progress=None):
    """Open a FITS file and return its primary image data, normalized down to
    (h, w) or (h, w, 3) - the dimensionality handling FITS files have always
    needed (extra axes, or RGB planes stored channel-first instead of
    interleaved). The returned array is fully materialized (np.array(...))
    while the file is still open, since astropy may hand back a
    memory-mapped view that becomes invalid once the file is closed.
    progress: read the file into memory first, reporting progress (see
    read_file_bytes) - otherwise astropy memory-maps it."""
    import io
    from astropy.io import fits

    source = io.BytesIO(read_file_bytes(file_path, progress)) if progress else file_path
    with fits.open(source) as hdul:
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


def read_astro_array(file_path, progress=None):
    """Raw pixel data of a FITS or XISF file as a numpy array - (h, w) or
    (h, w, 3), native dtype and range - or None for other extensions or a
    file with no image. Raises on unreadable/corrupt files. progress: called
    as the file is read (see read_file_bytes)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in FITS_EXTENSIONS:
        return _read_fits_array(file_path, progress)
    if ext in XISF_EXTENSIONS:
        from XISFReader import read_xisf_pixels
        return read_xisf_pixels(file_path, progress)
    return None


def load_astro_qimage(file_path, progress=None):
    """Decode a FITS or XISF file into a displayable QImage. Unlike QPixmap,
    QImage is safe to create in a worker thread. progress: called as the file
    is read (see read_file_bytes).

    Returns None on any failure (missing file, unsupported format/compression,
    corrupt data, etc.).
    """
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        image_data = read_astro_array(file_path, progress)
        if image_data is None:
            return None

        qimage = stretch_array_to_qimage(image_data)
    except Exception as e:
        logger.debug(f"Could not load astro image {file_path}: {e}")
        return None

    return None if qimage.isNull() else qimage


# Never hand Qt's image readers/writers a QIODevice made in Python (a QBuffer,
# say) on a worker thread: Qt holds its image-format plugin lock while it
# looks at the device, and a Python-made device takes the GIL to answer - so
# a main thread holding the GIL while loading any image (an icon, a pixmap)
# deadlocks the app. QImage.fromData() and file paths use devices Qt makes.

# EXIF orientation -> (clockwise rotation, then horizontal/vertical flip)
_EXIF_ORIENTATIONS = {2: (0, -1, 1), 3: (180, 1, 1), 4: (0, 1, -1), 5: (90, -1, 1),
                      6: (90, 1, 1), 7: (90, 1, -1), 8: (270, 1, 1)}


def header_exif(image):
    """An open Pillow image's EXIF, read from its header - never by decoding it.
    Pillow's own getexif() decodes a whole PNG first, looking for EXIF stored
    after the pixels: 16 s for a large PNG on the network drive, on whichever
    thread asked (it hung the Image Viewer). EXIF after a PNG's pixels is rare
    and against the spec, so it's left out."""
    if image.format == "PNG" and "exif" not in image.info:
        from PIL import Image
        return Image.Exif()
    return image.getexif()


def _exif_orientation(data):
    """The EXIF orientation (1-8) in an image file's bytes - 1 if none. Pillow
    only reads the header for this."""
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            return int(header_exif(image).get(0x0112, 1))
    except Exception:
        return 1


def _apply_orientation(qimage, orientation):
    rotation, flip_x, flip_y = _EXIF_ORIENTATIONS.get(orientation, (0, 1, 1))
    if rotation:
        qimage = qimage.transformed(QTransform().rotate(rotation))
    if flip_x < 0 or flip_y < 0:
        qimage = qimage.transformed(QTransform().scale(flip_x, flip_y))
    return qimage


def read_image_file(file_path, max_dim=None, auto_transform=True, progress=None):
    """Decode a standard image file (PNG, JPEG, TIFF...) into a QImage, scaled
    to fit max_dim if given. auto_transform applies the EXIF orientation.

    The whole file is read first (in large pieces, reporting to progress if
    given - see read_file_bytes): Qt's readers fetch a file in many small
    pieces, which took about 10x as long from a network drive. Safe to call
    from a worker thread. Returns (qimage, None) or (None, error message).
    """
    try:
        data = read_file_bytes(file_path, progress)
    except OSError as e:
        return None, f"Couldn't read the file: {e.strerror or e}"
    if not data:
        return None, "Empty file"

    # Like reading from the path: the format the extension names, then
    # whatever the content looks like (a PNG named .jpg, say)
    suffix = os.path.splitext(file_path)[1].lstrip('.').lower()
    qimage = QImage()
    with QT_IMAGE_IO_LOCK:
        QImageReader.setAllocationLimit(1024)  # MB - large stacked images exceed Qt's default
        for image_format in (suffix, None):
            qimage = QImage.fromData(data, image_format) if image_format else QImage.fromData(data)
            if not qimage.isNull():
                break
    if qimage.isNull():
        return None, "Unsupported or damaged image file"
    if auto_transform:
        qimage = _apply_orientation(qimage, _exif_orientation(data))
    if max_dim and max(qimage.width(), qimage.height()) > max_dim:
        qimage = qimage.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return qimage, None


def write_image_file(image, file_path, image_format, quality=-1):
    """Encode a QImage and write it to file_path - encoded into a local temp
    file, then written in one go, since Qt writes a file in small pieces too
    (slow on a network drive). Safe to call from a worker thread. Returns
    True on success."""
    import tempfile
    handle, temp_path = tempfile.mkstemp(suffix=f".{image_format.lower()}")
    os.close(handle)
    try:
        with QT_IMAGE_IO_LOCK:
            ok = image.save(temp_path, image_format, quality)
        if not ok:
            return False
        with open(temp_path, 'rb') as source:
            encoded = source.read()
        with open(file_path, 'wb') as f:
            f.write(encoded)
    except OSError as e:
        logger.warning(f"Could not write {file_path}: {e}")
        return False
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return True


def load_image_qimage(file_path, max_dim=None, progress=None):
    """Load any supported image (FITS/XISF or a standard format Qt can read)
    as a QImage - full resolution, or scaled to fit max_dim. Safe to call from
    a worker thread. progress(done, total) is called as the file is read; once
    done == total it's decoding, which can't report progress. Returns (qimage,
    None) on success or (None, error message) on failure."""
    if not file_path or not os.path.exists(file_path):
        return None, "File not found"

    ext = os.path.splitext(file_path)[1].lower()
    if ext in FITS_EXTENSIONS + XISF_EXTENSIONS:
        qimage = load_astro_qimage(file_path, progress)
        if qimage is None:
            return None, "Failed to load FITS/XISF file"
        if max_dim and max(qimage.width(), qimage.height()) > max_dim:
            qimage = qimage.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return qimage, None

    return read_image_file(file_path, max_dim=max_dim, progress=progress)


def load_screen_stretched_qimage(file_path):
    """A FITS/XISF sub at full resolution with an auto screen stretch (debayered
    on a color camera) - for reviewing raw light frames, which a plain linear
    stretch leaves nearly black. Safe to call from a worker thread.
    Returns (qimage, None) on success or (None, error message) on failure."""
    if not file_path or not os.path.exists(file_path):
        return None, "File not found"
    try:
        from FrameQuality import screen_stretched
        pixels = np.ascontiguousarray(screen_stretched(file_path))
    except Exception as e:
        logger.debug(f"Could not screen-stretch {file_path}: {e}")
        return None, f"Failed to load FITS/XISF file: {e}"
    height, width = pixels.shape[:2]
    if pixels.ndim == 3:
        qimage = QImage(pixels.data, width, height, width * 3, QImage.Format_RGB888)
    else:
        qimage = QImage(pixels.data, width, height, width, QImage.Format_Grayscale8)
    return qimage.copy(), None


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
