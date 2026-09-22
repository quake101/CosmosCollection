#!/usr/bin/env python3
"""
Session File Scanner
Extracts capture metadata (DSO name, integration time, camera, filter, etc.)
from FITS and XISF sub-exposure files for the Session Manager.
"""

import os
import json
import struct
import logging
import xml.etree.ElementTree as ET
from collections import Counter

logger = logging.getLogger(__name__)

FITS_EXTENSIONS = {'.fits', '.fit', '.fts'}
XISF_EXTENSIONS = {'.xisf'}
SUPPORTED_EXTENSIONS = FITS_EXTENSIONS | XISF_EXTENSIONS

# Same keyword set ImageViewer._get_fits_info() displays, minus the ones that
# don't matter for session aggregation (EQUINOX, SWCREATE, SWMODIFY, etc. are
# still captured via header_json for reference, just not summarized on).
FITS_KEYWORDS = [
    'OBJECT', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'DATE-OBS', 'EXPTIME', 'FILTER',
    'FOCALLEN', 'APTDIA', 'APTAREA', 'FWHM', 'EQUINOX', 'RA', 'DEC', 'OBJCTRA',
    'OBJCTDEC', 'AIRMASS', 'GAIN', 'OFFSET', 'TEMP', 'CCD-TEMP', 'SET-TEMP',
    'XBINNING', 'YBINNING', 'IMAGETYP', 'FRAME', 'SWCREATE', 'SWMODIFY',
]

XISF_SIGNATURE = b"XISF0100"
XISF_NAMESPACE = "http://www.pixinsight.com/xisf"

# XISF Property elements that duplicate a FITS-equivalent keyword when a
# writer didn't also emit a FITSKeyword for it. Best-effort, not exhaustive.
XISF_PROPERTY_TO_FITS = {
    'Instrument:ExposureTime': 'EXPTIME',
    'Instrument:Camera:Name': 'INSTRUME',
    'Instrument:Telescope:Name': 'TELESCOP',
    'Instrument:Filter:Name': 'FILTER',
    'Observation:Time:Start': 'DATE-OBS',
    'Observation:Object:Name': 'OBJECT',
}

FRAME_TYPE_KEYWORDS = [
    ('LIGHT', 'Light'),
    ('DARK', 'Dark'),
    ('FLAT', 'Flat'),
    ('BIAS', 'Bias'),
]


def _coerce_fits_value(raw):
    """Coerce a raw FITS-card-style string (e.g. "'M31'" or "120.0") to a native type."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return raw[1:-1].strip()
    try:
        if '.' in raw or 'e' in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _clean_str(value):
    if value is None:
        return None
    return str(value).strip().strip("'").strip()


def extract_fits_header(file_path):
    """Read a FITS primary header and return the recognized keywords as native-typed values."""
    from astropy.io import fits

    with fits.open(file_path) as hdul:
        header = hdul[0].header
        return {kw: header[kw] for kw in FITS_KEYWORDS if kw in header}


def read_xisf_xml(file_path):
    """Read the signature + header-length-prefixed XML header block of an XISF
    file (never the pixel data) and return (root, image_el, ns) - the parsed XML
    root, its <Image> element (or None), and the namespace map to use for further
    xisf: lookups on either. Shared by extract_xisf_header() below and
    XISFReader.read_xisf_pixels(), which needs the same <Image> element's
    geometry/sampleFormat/location/compression attributes."""
    with open(file_path, 'rb') as f:
        signature = f.read(8)
        if signature != XISF_SIGNATURE:
            raise ValueError(f"Not a valid XISF file (bad signature): {file_path}")
        header_length = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # reserved
        xml_text = f.read(header_length).decode('utf-8')

    root = ET.fromstring(xml_text)
    ns = {'xisf': XISF_NAMESPACE}

    image_el = root.find('xisf:Image', ns)
    if image_el is None:
        image_el = root.find('Image')

    return root, image_el, ns


def extract_xisf_header(file_path):
    """Read only the XML header block of an XISF file (never the pixel data) and
    return the same shape as extract_fits_header() by mapping FITSKeyword/Property
    elements onto the equivalent FITS keywords."""
    root, image_el, ns = read_xisf_xml(file_path)

    result = {}

    if image_el is not None:
        keyword_els = image_el.findall('xisf:FITSKeyword', ns)
        if not keyword_els:
            keyword_els = image_el.findall('FITSKeyword')
        for kw_el in keyword_els:
            name = kw_el.get('name')
            value = kw_el.get('value')
            if not name or value is None:
                continue
            name = name.strip().upper()
            if name in FITS_KEYWORDS and name not in result:
                result[name] = _coerce_fits_value(value)

    property_els = []
    for scope in (image_el, root):
        if scope is None:
            continue
        found = scope.findall('xisf:Property', ns)
        if not found:
            found = scope.findall('Property')
        property_els.extend(found)

    for prop_el in property_els:
        fits_key = XISF_PROPERTY_TO_FITS.get(prop_el.get('id'))
        if not fits_key or fits_key in result:
            continue
        value = prop_el.get('value')
        if value is None:
            continue
        prop_type = prop_el.get('type', '')
        if prop_type.startswith('Float') or prop_type.startswith('Int') or prop_type.startswith('UInt'):
            try:
                result[fits_key] = float(value) if '.' in value else int(value)
            except ValueError:
                result[fits_key] = value
        else:
            result[fits_key] = value

    return result


def _normalize_frame_type(header):
    raw = str(header.get('IMAGETYP') or header.get('FRAME') or '').upper()
    for keyword, label in FRAME_TYPE_KEYWORDS:
        if keyword in raw:
            return label
    return 'Unknown'


def scan_file(file_path):
    """Extract header metadata from a single FITS/XISF file, or None if unreadable."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in FITS_EXTENSIONS:
            header = extract_fits_header(file_path)
            file_type = 'fits'
        elif ext in XISF_EXTENSIONS:
            header = extract_xisf_header(file_path)
            file_type = 'xisf'
        else:
            return None
    except Exception as e:
        logger.warning(f"Failed to read header from {file_path}: {e}")
        return None

    result = dict(header)
    result['_file_path'] = file_path
    result['_file_type'] = file_type
    result['_frame_type'] = _normalize_frame_type(header)
    return result


def scan_folder(folder_path, progress_callback=None):
    """Recursively scan a folder for FITS/XISF files and extract each one's header metadata."""
    matches = []
    for root, _dirs, files in os.walk(folder_path):
        for name in files:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                matches.append(os.path.join(root, name))

    results = []
    total = len(matches)
    for i, path in enumerate(matches):
        data = scan_file(path)
        if data is not None:
            results.append(data)
        if progress_callback:
            progress_callback(i + 1, total)
    return results


def summarize_files(file_dicts):
    """Aggregate a list of scan_file()/scan_folder() results into a session summary."""
    summary = {
        'sub_count': len(file_dicts),
        'integration_seconds': 0.0,
        'filters_used': [],
        'camera': None,
        'telescope': None,
        'dso_name': None,
        'earliest_sub_date': None,
        'latest_sub_date': None,
        'frame_type_counts': Counter(),
        'files': file_dicts,
    }
    if not file_dicts:
        return summary

    filters = set()
    cameras = Counter()
    telescopes = Counter()
    objects = Counter()
    dates = []

    for f in file_dicts:
        frame_type = f.get('_frame_type', 'Unknown')
        summary['frame_type_counts'][frame_type] += 1

        if frame_type == 'Light':
            exptime = f.get('EXPTIME')
            if isinstance(exptime, (int, float)):
                summary['integration_seconds'] += float(exptime)
            filt = _clean_str(f.get('FILTER'))
            if filt:
                filters.add(filt)

        camera = _clean_str(f.get('INSTRUME'))
        if camera:
            cameras[camera] += 1
        telescope = _clean_str(f.get('TELESCOP'))
        if telescope:
            telescopes[telescope] += 1
        obj = _clean_str(f.get('OBJECT'))
        if obj:
            objects[obj] += 1

        date_obs = f.get('DATE-OBS')
        if date_obs:
            dates.append(str(date_obs))

    summary['filters_used'] = sorted(filters)
    summary['camera'] = cameras.most_common(1)[0][0] if cameras else None
    summary['telescope'] = telescopes.most_common(1)[0][0] if telescopes else None
    summary['dso_name'] = objects.most_common(1)[0][0] if objects else None
    if dates:
        summary['earliest_sub_date'] = min(dates)
        summary['latest_sub_date'] = max(dates)

    return summary


def file_dict_to_row(f):
    """Map a scan_file() result onto usersessionfiles column names."""
    exptime = f.get('EXPTIME')
    gain = f.get('GAIN')
    offset = f.get('OFFSET')
    ccd_temp = f.get('CCD-TEMP')
    xbinning = f.get('XBINNING')
    ybinning = f.get('YBINNING')
    return {
        'file_path': f.get('_file_path'),
        'file_type': f.get('_file_type'),
        'frame_type': f.get('_frame_type'),
        'object_name': _clean_str(f.get('OBJECT')),
        'date_obs': f.get('DATE-OBS'),
        'exptime_seconds': float(exptime) if isinstance(exptime, (int, float)) else None,
        'filter_name': _clean_str(f.get('FILTER')),
        'camera': _clean_str(f.get('INSTRUME')),
        'telescope': _clean_str(f.get('TELESCOP')),
        'gain': float(gain) if isinstance(gain, (int, float)) else None,
        'offset_value': float(offset) if isinstance(offset, (int, float)) else None,
        'ccd_temp': float(ccd_temp) if isinstance(ccd_temp, (int, float)) else None,
        'xbinning': int(xbinning) if isinstance(xbinning, (int, float)) else None,
        'ybinning': int(ybinning) if isinstance(ybinning, (int, float)) else None,
        'header_json': json.dumps({k: v for k, v in f.items() if not k.startswith('_')}, default=str),
    }
