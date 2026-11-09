#!/usr/bin/env python3
"""
Session File Scanner
Extracts capture metadata (DSO name, integration time, camera, filter, etc.)
from FITS and XISF sub-exposure files for the Session Manager.
"""

import os
import re
import json
import math
import struct
import logging
import xml.etree.ElementTree as ET
from collections import Counter

logger = logging.getLogger(__name__)

# Subfolder the light quality review moves rejected subs into (skipped when scanning)
REJECTED_FOLDER = "Rejected"

FITS_EXTENSIONS = {'.fits', '.fit', '.fts'}
XISF_EXTENSIONS = {'.xisf'}
SUPPORTED_EXTENSIONS = FITS_EXTENSIONS | XISF_EXTENSIONS

# Header keywords captured for session aggregation (EQUINOX, SWCREATE, SWMODIFY,
# etc. are captured via header_json for reference, just not summarized on).
FITS_KEYWORDS = [
    'OBJECT', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'DATE-OBS', 'EXPTIME', 'FILTER',
    'FOCALLEN', 'APTDIA', 'APTAREA', 'FWHM', 'EQUINOX', 'RA', 'DEC', 'OBJCTRA',
    'OBJCTDEC', 'AIRMASS', 'GAIN', 'OFFSET', 'TEMP', 'CCD-TEMP', 'SET-TEMP',
    'XBINNING', 'YBINNING', 'IMAGETYP', 'FRAME', 'SWCREATE', 'SWMODIFY',
    'BAYERPAT',  # present on one-shot-colour subs - ProcessingHandoff debayers those
    'XPIXSZ',    # pixel size (microns) - FrameQuality's arcsec/pixel with FOCALLEN
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
    'PCL:CFASourcePattern': 'BAYERPAT',
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


def to_float(value):
    """A header value as a float, or None (missing, text, NaN or infinite)."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def exposure_of(header):
    """A sub's exposure in seconds (EXPTIME, else EXPOSURE), or None."""
    return to_float(header.get('EXPTIME', header.get('EXPOSURE'))) if header else None


def _first_value(header, names):
    return next((header[name] for name in names if header.get(name) not in (None, "")), None)


def _degrees(value):
    """Degrees from a number or a sexagesimal string ('+40 30 00'), or None."""
    result = to_float(value)
    if result is None and isinstance(value, str):
        result = _parse_sexagesimal(value)
    return result


def site_latitude(header):
    """Where a header says it was taken: latitude in degrees, or None."""
    return _degrees(_first_value(header or {}, ('SITELAT', 'OBSGEO-B', 'LAT-OBS')))


def site_longitude(header):
    """Where a header says it was taken: longitude in degrees, east positive
    (-180 to 180, also from headers that count 0-360), or None."""
    longitude = _degrees(_first_value(header or {}, ('SITELONG', 'OBSGEO-L', 'LONG-OBS')))
    if longitude is not None and longitude > 180:
        longitude -= 360
    return longitude


def site_of(header):
    """(lat, lon) a header says it was taken at, or (None, None) - also for a
    0, 0 placeholder."""
    lat, lon = site_latitude(header), site_longitude(header)
    if lat is None or lon is None or not (lat or lon):
        return None, None
    return lat, lon


def read_header(file_path):
    """Every keyword of a FITS or XISF file's header (COMMENT/HISTORY left out):
    FITS through the fast block reader, XISF from its XML (never the pixels).
    Raises ValueError for other files, OSError if it can't be read."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in FITS_EXTENSIONS:
        return read_fits_header_fast(file_path)
    if ext in XISF_EXTENSIONS:
        return extract_xisf_header(file_path, keywords=None)
    raise ValueError(f"Not a FITS or XISF file: {file_path}")


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


FITS_BLOCK = 2880
FITS_CARD = 80


def _fits_card_value(text):
    """The value of a FITS card's value field (after '= '), comment dropped:
    a string ('' is an escaped quote) or a native number/flag."""
    text = text.lstrip()
    if text.startswith("'"):
        chars, i = [], 1
        while i < len(text):
            if text[i] == "'":
                if text[i + 1:i + 2] == "'":
                    chars.append("'")
                    i += 2
                    continue
                break
            chars.append(text[i])
            i += 1
        return ''.join(chars).strip()
    return _coerce_fits_value(text.split('/', 1)[0])


def parse_fits_card(card):
    """(keyword, value) of one header card or .wcs file line - None for a card
    without a value (COMMENT, HISTORY, blank, END). The '=' belongs in column
    9, but lines a person wrote ('CRVAL1= 83.8') are read too."""
    equals = card.find('=', 1, 10)
    if equals < 0:
        return None
    key = card[:equals].strip()
    if not key or ' ' in key:
        return None
    try:
        return key, _fits_card_value(card[equals + 1:])
    except Exception:
        return None


def parse_fits_cards(cards):
    """{keyword: value} from header cards, stopping at END. A keyword given
    twice keeps its first value, as astropy reads it."""
    result = {}
    for card in cards:
        if card[:8].strip() == 'END':
            break
        parsed = parse_fits_card(card)
        if parsed and parsed[0] not in result:
            result[parsed[0]] = parsed[1]
    return result


def read_fits_header_fast(file_path, max_blocks=40):
    """Every keyword of a FITS primary header (COMMENT/HISTORY left out), read
    in 2880-byte blocks up to END - usually one small read, where astropy's
    fits.open makes several round trips (slow on a network drive). Values as
    parse_fits_card reads them. Raises ValueError if it isn't a FITS file."""
    cards = []
    with open(file_path, 'rb') as f:
        for block_index in range(max_blocks):
            block = f.read(FITS_BLOCK)
            if len(block) < FITS_BLOCK:
                break
            if block_index == 0 and not block.startswith(b'SIMPLE  ='):
                raise ValueError(f"Not a FITS file: {file_path}")
            block_cards = [block[start:start + FITS_CARD].decode('ascii', 'replace')
                           for start in range(0, FITS_BLOCK, FITS_CARD)]
            cards.extend(block_cards)
            if any(card[:8].strip() == 'END' for card in block_cards):
                break
    return parse_fits_cards(cards)


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


def extract_xisf_header(file_path, keywords=FITS_KEYWORDS):
    """Read only the XML header block of an XISF file (never the pixel data) and
    return the same shape as extract_fits_header() by mapping FITSKeyword/Property
    elements onto the equivalent FITS keywords. keywords=None keeps every
    FITS keyword (except COMMENT/HISTORY) and adds the <Image> element's
    imageType attribute (e.g. 'MasterLight') as '_XISF_IMAGETYPE'."""
    root, image_el, ns = read_xisf_xml(file_path)

    result = {}
    if keywords is None and image_el is not None and image_el.get('imageType'):
        result['_XISF_IMAGETYPE'] = image_el.get('imageType')

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
            if keywords is None:
                wanted = name not in ('COMMENT', 'HISTORY', '')
            else:
                wanted = name in keywords
            if wanted and name not in result:
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
    """Recursively scan a folder for FITS/XISF files and extract each one's header
    metadata. Folders named REJECTED_FOLDER (where the light quality review
    moves rejected subs) are skipped, so re-attaching a night doesn't bring them back."""
    matches = []
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d.lower() != REJECTED_FOLDER.lower()]
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


def normalize_object_name(name):
    """Reduce a free-text object name to a comparable key: drop parenthetical/
    bracketed common names, then everything but letters and digits, so
    'Sh2 129 (Squid Nebula)', 'SH2-129' and 'sh2_129' all become 'SH2129'."""
    if not name:
        return ""
    name = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', str(name))
    return re.sub(r'[^A-Z0-9]', '', name.upper())


def _parse_sexagesimal(value):
    """Parse '21 11 45' / '+59:57:41' style strings to a float in the same unit
    as the first field, or None."""
    parts = re.split(r'[\s:hdms°\'"]+', str(value).strip())
    parts = [p for p in parts if p]
    if not parts:
        return None
    try:
        sign = -1.0 if parts[0].startswith('-') else 1.0
        fields = [abs(float(p)) for p in parts[:3]]
    except ValueError:
        return None
    total = sum(f / (60.0 ** i) for i, f in enumerate(fields))
    return sign * total


def _file_coordinates(f):
    """(ra_deg, dec_deg) a sub was pointed at, from numeric RA/DEC (degrees)
    or the OBJCTRA (hours)/OBJCTDEC (degrees) strings, or None."""
    ra, dec = f.get('RA'), f.get('DEC')
    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
        return float(ra), float(dec)
    if f.get('OBJCTRA') is not None and f.get('OBJCTDEC') is not None:
        ra_h = _parse_sexagesimal(f.get('OBJCTRA'))
        dec_d = _parse_sexagesimal(f.get('OBJCTDEC'))
        if ra_h is not None and dec_d is not None:
            return ra_h * 15.0, dec_d
    return None


def _mean_coordinates(coords):
    """Mean of (ra_deg, dec_deg) pairs via unit vectors, so RA wrapping at
    0/360 doesn't skew the result."""
    x = y = z = 0.0
    for ra, dec in coords:
        ra_r, dec_r = math.radians(ra), math.radians(dec)
        x += math.cos(dec_r) * math.cos(ra_r)
        y += math.cos(dec_r) * math.sin(ra_r)
        z += math.sin(dec_r)
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.atan2(z, math.hypot(x, y)))
    return ra, dec


def angular_separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees between two RA/Dec points (degrees)."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((dec2 - dec1) / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


def angular_separations_deg(ra, dec, ras, decs):
    """Great-circle separations in degrees from one RA/Dec point to each of
    arrays ras, decs (degrees) - angular_separation_deg for many at once."""
    import numpy as np
    ra, dec = math.radians(ra), math.radians(dec)
    ras, decs = np.radians(ras), np.radians(decs)
    h = np.sin((decs - dec) / 2) ** 2 + math.cos(dec) * np.cos(decs) * np.sin((ras - ra) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def summarize_files(file_dicts):
    """Aggregate a list of scan_file()/scan_folder() results into a session summary."""
    summary = {
        'sub_count': len(file_dicts),
        'integration_seconds': 0.0,
        'filters_used': [],
        'camera': None,
        'telescope': None,
        'dso_name': None,
        'ra_deg': None,   # mean pointing of the Light frames, when their headers carry it
        'dec_deg': None,
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
    light_objects = Counter()  # calibration frames often carry OBJECT='FlatWizard' etc.
    light_coords = []
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
            light_obj = _clean_str(f.get('OBJECT'))
            if light_obj:
                light_objects[light_obj] += 1
            coords = _file_coordinates(f)
            if coords:
                light_coords.append(coords)

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
    best_objects = light_objects or objects
    summary['dso_name'] = best_objects.most_common(1)[0][0] if best_objects else None
    if light_coords:
        summary['ra_deg'], summary['dec_deg'] = _mean_coordinates(light_coords)
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
