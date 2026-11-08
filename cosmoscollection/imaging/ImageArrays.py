"""
Reading FITS and XISF pixel data into numpy arrays - no Qt.

Used by ImageLoader.py (which turns the arrays into images for the windows and
re-exports these names), frame quality measurement, plate solving and mount
detection - so the command line can measure frames without loading Qt's GUI.
"""

import os

import numpy as np

FITS_EXTENSIONS = ('.fits', '.fit', '.fts')
XISF_EXTENSIONS = ('.xisf',)

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
        from cosmoscollection.imaging.XISFReader import read_xisf_pixels
        return read_xisf_pixels(file_path, progress)
    return None
