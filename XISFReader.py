#!/usr/bin/env python3
"""
XISF (PixInsight's native image format) pixel data reader.

Only handles the common case: pixel data physically attached to the same
monolithic .xisf file (location="attachment:offset:size"), stored either
uncompressed or zlib-compressed (with or without XISF's byte-shuffle filter).
Inline/embedded/external-file locations and LZ4/Zstandard compression raise a
clear error rather than silently failing - see the project's XISF support
plan for why those are out of scope (low real-world prevalence for the
former, no new dependencies for the latter).

Header/metadata extraction (OBJECT, TELESCOP, EXPTIME, etc.) lives in
SessionFileScanner.extract_xisf_header() - this module only decodes pixels.
"""

import zlib
import numpy as np

from SessionFileScanner import read_xisf_xml

_SAMPLE_FORMAT_DTYPES = {
    'UInt8': np.uint8,
    'UInt16': np.uint16,
    'UInt32': np.uint32,
    'UInt64': np.uint64,
    'Float32': np.float32,
    'Float64': np.float64,
}


def _unshuffle(data, item_size):
    """Reverse XISF's optional byte-shuffle filter: bytes are stored grouped
    by byte-position across all samples (all byte 0's, then all byte 1's, ...)
    rather than sample-by-sample - the same technique as HDF5's/Blosc's
    shuffle filter, used to improve compression ratio on smooth image data."""
    arr = np.frombuffer(data, dtype=np.uint8)
    sample_count = len(arr) // item_size
    arr = arr[:sample_count * item_size].reshape(item_size, sample_count)
    return arr.T.tobytes()


def _decompress(data, compression, expected_size):
    """compression is the raw XISF 'compression' attribute value, e.g.
    "zlib:1048576" or "zlib+sh:4:1048576" (codec[+sh]:[item-size:]uncompressed-size),
    or None/empty for uncompressed data. Returns the decompressed (and
    unshuffled, if applicable) bytes, verified against the sizes XISF itself
    records rather than trusted blindly."""
    if not compression:
        return data

    parts = compression.split(':')
    codec = parts[0]
    shuffled = codec.endswith('+sh')
    if shuffled:
        codec = codec[:-3]

    if codec != 'zlib':
        raise ValueError(
            f"Unsupported XISF compression codec: {codec!r} "
            "(only zlib and uncompressed data are supported)")

    if shuffled:
        if len(parts) != 3:
            raise ValueError(f"Unrecognized XISF compression attribute format: {compression!r}")
        item_size = int(parts[1])
        uncompressed_size = int(parts[2])
    else:
        if len(parts) != 2:
            raise ValueError(f"Unrecognized XISF compression attribute format: {compression!r}")
        uncompressed_size = int(parts[1])

    decompressed = zlib.decompress(data)
    if len(decompressed) != uncompressed_size:
        raise ValueError(
            f"XISF decompression size mismatch: expected {uncompressed_size} bytes, "
            f"got {len(decompressed)}")

    if shuffled:
        decompressed = _unshuffle(decompressed, item_size)

    if len(decompressed) != expected_size:
        raise ValueError(
            f"XISF pixel data size mismatch after decompression: expected {expected_size} "
            f"bytes, got {len(decompressed)}")

    return decompressed


def read_xisf_pixels(file_path):
    """Read and decode the pixel data of an XISF file's main <Image> element.

    Returns a numpy array shaped (height, width) for a single-channel image,
    or (height, width, channels) for a multi-channel one, with dtype matching
    the file's sampleFormat and values in their native (unnormalized) range -
    the same shape/range convention ImageLoader.stretch_array_to_qimage()
    already expects from the existing FITS decode path.
    """
    root, image_el, ns = read_xisf_xml(file_path)
    if image_el is None:
        raise ValueError(f"No <Image> element found in XISF header: {file_path}")

    geometry = image_el.get('geometry')
    sample_format = image_el.get('sampleFormat', 'UInt16')
    location = image_el.get('location')
    compression = image_el.get('compression')

    if not geometry:
        raise ValueError(f"XISF <Image> element has no geometry attribute: {file_path}")
    if not location:
        raise ValueError(f"XISF <Image> element has no location attribute: {file_path}")

    dims = [int(d) for d in geometry.split(':')]
    if len(dims) == 2:
        width, height = dims
        channels = 1
    elif len(dims) == 3:
        width, height, channels = dims
    else:
        raise ValueError(f"Unsupported XISF geometry (expected width:height[:channels]): {geometry!r}")

    dtype = _SAMPLE_FORMAT_DTYPES.get(sample_format)
    if dtype is None:
        raise ValueError(f"Unsupported XISF sampleFormat: {sample_format!r}")

    loc_parts = location.split(':')
    if loc_parts[0] != 'attachment' or len(loc_parts) != 3:
        raise NotImplementedError(
            f"Unsupported XISF pixel data location {location!r} - only "
            "attachment:offset:size (pixel data embedded in the same file) is "
            "currently supported")
    offset, block_size = int(loc_parts[1]), int(loc_parts[2])

    bytes_per_sample = np.dtype(dtype).itemsize
    expected_size = width * height * channels * bytes_per_sample

    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw = f.read(block_size)

    raw = _decompress(raw, compression, expected_size)

    arr = np.frombuffer(raw, dtype=dtype)
    if channels == 1:
        arr = arr.reshape(height, width)
    else:
        # XISF stores channels as separate contiguous planes (channel-major),
        # not interleaved like a typical RGB raster - reshape to (channels,
        # height, width) then move the channel axis last to match the
        # (height, width, channels) shape the shared stretch code expects.
        arr = arr.reshape(channels, height, width)
        arr = np.transpose(arr, (1, 2, 0))

    return arr
