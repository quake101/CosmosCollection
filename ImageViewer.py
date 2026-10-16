#!/usr/bin/env python3
"""
ImageViewer module for Cosmos Collection
Provides the ImageViewerWindow class for displaying images with zoom, pan, and annotation support.
"""

import json
import logging
import math
import platform
import ctypes
import re
import subprocess
import os
from collections import namedtuple
from html import escape
from pathlib import Path
from datetime import datetime, timezone

from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QSettings, QThread, QRectF
from PySide6.QtGui import QPixmap, QPainter, QImage
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QGroupBox, QScrollArea, QFileDialog, QMessageBox, QCheckBox, QProgressBar, QToolButton, QMenu
)

from Theme import COLORS, adapt_color, font_size, theme_manager, themed_style
from WindowPositionManager import WindowPositionManager
from ResourceManager import ResourceManager
from TimeFormatHelper import format_datetime
from SessionFileScanner import (FITS_EXTENSIONS, XISF_EXTENSIONS, read_xisf_xml,
                                _coerce_fits_value, _file_coordinates, _parse_sexagesimal)
from AnnotationOverlay import (ANNOTATION_LAYER_SETTINGS, ANNOTATION_LAYERS, AnnotationData, annotation_store,
                               image_pixel_size, paint_annotations, read_wcs_header, saved_annotation_layers,
                               screen_name_px)

# Set up logging
logger = logging.getLogger(__name__)


# Image load workers that outlived their viewer window. A QThread that's
# destroyed while still running crashes the app, so keep a reference until
# the load finishes instead of blocking the window close on it.
_orphaned_workers = set()


class ImageLoadWorker(QThread):
    """Decodes an image file (including FITS/XISF) into a QImage off the GUI thread"""
    image_loaded = Signal(object)  # QImage
    load_failed = Signal(str)  # error message

    def __init__(self, file_path, screen_stretch=False):
        super().__init__()
        self.file_path = file_path
        self.screen_stretch = screen_stretch

    def run(self):
        try:
            from ImageLoader import load_image_qimage, load_screen_stretched_qimage
            is_astro = os.path.splitext(self.file_path)[1].lower() in ('.fits', '.fit', '.fts', '.xisf')
            if self.screen_stretch and is_astro:
                qimage, error = load_screen_stretched_qimage(self.file_path)
            else:
                qimage, error = load_image_qimage(self.file_path)
            if qimage is not None:
                self.image_loaded.emit(qimage)
            else:
                self.load_failed.emit(error)
        except Exception as e:
            logger.error(f"Failed to load image {self.file_path}: {e}", exc_info=True)
            self.load_failed.emit(str(e))


class ImageSaveWorker(QThread):
    """Encodes and writes a QImage to disk off the GUI thread (PNG compression
    of a large image takes seconds)"""
    image_saved = Signal(bool, str)  # success, save path

    def __init__(self, image, save_path, image_format, quality=-1):
        super().__init__()
        self.image = image
        self.save_path = save_path
        self.image_format = image_format
        self.quality = quality

    def run(self):
        try:
            from ImageLoader import write_image_file
            ok = write_image_file(self.image, self.save_path, self.image_format, self.quality)
        except Exception as e:
            logger.error(f"Failed to save image to {self.save_path}: {e}", exc_info=True)
            ok = False
        self.image_saved.emit(ok, self.save_path)


class BackgroundSetterWorker(QThread):
    """Converts a FITS/XISF render to PNG (if needed) and sets it as the desktop
    background off the GUI thread"""
    background_set = Signal(str, str)  # level ('success'/'warning'/'error'), message

    def __init__(self, file_path, image=None):
        super().__init__()
        self.file_path = file_path
        self.image = image  # QImage to convert to PNG first, or None to use file_path directly

    def run(self):
        try:
            if self.image is not None:
                abs_path = ImageViewerWindow._export_background_png(self.image, self.file_path)
                if not abs_path:
                    self.background_set.emit("error", "Failed to convert image to PNG for desktop background")
                    return
            else:
                abs_path = str(Path(self.file_path).resolve())

            level, message = ImageViewerWindow._apply_desktop_background(abs_path)
            self.background_set.emit(level, message)
        except Exception as e:
            logger.error(f"Failed to set desktop background: {e}", exc_info=True)
            self.background_set.emit("error", f"Failed to set desktop background: {str(e)}")


# ------------------------------------------------------------------------------
# FITS/XISF header info for the File Information panel. Only the header is read,
# never the pixel data, and each section lists only what the file records.
# ------------------------------------------------------------------------------

# keywords: upper-case FITS keyword -> value (first occurrence wins)
# history: HISTORY card texts
# image: 'width'/'height'/'channels'/'sample_format'/'color_space'/'software' where known
HeaderData = namedtuple('HeaderData', ['keywords', 'history', 'image'])

# XISF Properties that hold the same data as a FITS keyword. Only used when the
# file doesn't also carry that FITS keyword.
XISF_PROPERTY_KEYWORDS = {
    'Observation:Object:Name': 'OBJECT',
    'Observation:Center:RA': 'RA',
    'Observation:Center:Dec': 'DEC',
    'Observation:Time:Start': 'DATE-OBS',
    'Observation:Location:Latitude': 'SITELAT',
    'Observation:Location:Longitude': 'SITELONG',
    'Observation:Location:Elevation': 'SITEELEV',
    'Instrument:Camera:Name': 'INSTRUME',
    'Instrument:Camera:XBinning': 'XBINNING',
    'Instrument:Camera:YBinning': 'YBINNING',
    'Instrument:ExposureTime': 'EXPTIME',
    'Instrument:Filter:Name': 'FILTER',
    'Instrument:Focuser:Position': 'FOCPOS',
    'Instrument:Sensor:Temperature': 'CCD-TEMP',
    'Instrument:Sensor:TargetTemperature': 'SET-TEMP',
    'Instrument:Sensor:XPixelSize': 'XPIXSZ',
    'Instrument:Sensor:YPixelSize': 'YPIXSZ',
    'Instrument:Telescope:Name': 'TELESCOP',
    'Instrument:Telescope:FocalLength': 'FOCALLEN',
    'Instrument:Telescope:Aperture': 'APTDIA',
    'PCL:CFASourcePattern': 'BAYERPAT',
}
# XISF stores these in meters, FITS in millimeters
XISF_METER_PROPERTIES = {'Instrument:Telescope:FocalLength', 'Instrument:Telescope:Aperture'}

BITPIX_FORMATS = {
    8: '8-bit unsigned integer',
    16: '16-bit integer',
    32: '32-bit integer',
    -32: '32-bit float',
    -64: '64-bit float',
}
XISF_SAMPLE_FORMATS = {
    'UInt8': '8-bit unsigned integer',
    'UInt16': '16-bit unsigned integer',
    'UInt32': '32-bit unsigned integer',
    'UInt64': '64-bit unsigned integer',
    'Float32': '32-bit float',
    'Float64': '64-bit float',
}

# Keywords that make up a WCS (plate solution), including SIP distortion terms
WCS_KEYWORD_PATTERN = re.compile(
    r'^(WCSAXES|CTYPE\d|CUNIT\d|CRVAL\d|CRPIX\d|CDELT\d|CROTA\d|CD\d_\d|PC\d_\d|'
    r'RADESYS|EQUINOX|LONPOLE|LATPOLE|(A|B|AP|BP)_(ORDER|\d+_\d+))$')


def read_image_header(file_path):
    """Read a FITS/XISF file's header into a HeaderData. Raises on unreadable files."""
    if Path(file_path).suffix.lower() in XISF_EXTENSIONS:
        return _read_xisf_header(file_path)
    return _read_fits_header(file_path)


def _read_fits_header(file_path):
    from astropy.io import fits

    keywords = {}
    history = []
    with fits.open(file_path) as hdul:
        headers = [hdul[0].header]
        # Compressed (fpack) files and some writers keep the image in an extension
        if hdul[0].header.get('NAXIS', 0) == 0:
            for hdu in hdul[1:]:
                if hdu.is_image and hdu.header.get('NAXIS', 0) >= 2:
                    headers.append(hdu.header)
                    break

        for header in headers:
            for card in header.cards:
                try:
                    key, value = card.keyword, card.value
                except Exception:
                    continue  # Malformed card
                if key == 'HISTORY':
                    history.append(str(value))
                elif key and key != 'COMMENT' and key not in keywords and _has_value(value):
                    keywords[key] = value

    image = {}
    image_header = headers[-1]
    naxis = image_header.get('NAXIS', 0)
    if naxis >= 2:
        image['width'] = image_header.get('NAXIS1')
        image['height'] = image_header.get('NAXIS2')
        image['channels'] = image_header.get('NAXIS3', 1) if naxis >= 3 else 1
    bitpix = image_header.get('BITPIX')
    if bitpix in BITPIX_FORMATS:
        sample_format = BITPIX_FORMATS[bitpix]
        # Unsigned 16/32-bit data is stored signed with a BZERO offset
        if bitpix in (16, 32) and image_header.get('BZERO') == 2 ** (bitpix - 1):
            sample_format = sample_format.replace('integer', 'unsigned integer')
        image['sample_format'] = sample_format

    return HeaderData(keywords, history, image)


def _read_xisf_header(file_path):
    root, image_el, ns = read_xisf_xml(file_path)

    def children(scope, tag):
        if scope is None:
            return []
        return scope.findall(f'xisf:{tag}', ns) or scope.findall(tag)

    keywords = {}
    history = []
    for kw_el in children(image_el, 'FITSKeyword'):
        name = (kw_el.get('name') or '').strip().upper()
        value = kw_el.get('value')
        if name == 'HISTORY':
            # PixInsight puts the history text in the comment attribute
            history.append(kw_el.get('comment') or _coerce_fits_value(value or ''))
        elif name and name != 'COMMENT' and name not in keywords and value is not None:
            value = _coerce_fits_value(value)
            if _has_value(value):
                keywords[name] = value

    metadata_el = root.find('xisf:Metadata', ns)
    if metadata_el is None:
        metadata_el = root.find('Metadata')

    properties = {}
    for scope in (image_el, root, metadata_el):
        for prop_el in children(scope, 'Property'):
            prop_id = prop_el.get('id')
            if not prop_id or prop_id in properties:
                continue
            value = prop_el.get('value')
            if value is None:
                value = (prop_el.text or '').strip()  # String properties can hold the text inline
            if prop_el.get('type', '').startswith(('Float', 'Int', 'UInt')):
                try:
                    value = float(value) if '.' in value or 'e' in value.lower() else int(value)
                except ValueError:
                    pass
            properties[prop_id] = value

    for prop_id, keyword in XISF_PROPERTY_KEYWORDS.items():
        value = properties.get(prop_id)
        if keyword in keywords or not _has_value(value):
            continue
        if prop_id in XISF_METER_PROPERTIES and isinstance(value, (int, float)):
            value = value * 1000.0
        keywords[keyword] = value

    image = {}
    if image_el is not None:
        dims = (image_el.get('geometry') or '').split(':')
        if len(dims) >= 2 and all(d.isdigit() for d in dims):
            image['width'], image['height'] = int(dims[0]), int(dims[1])
            image['channels'] = int(dims[2]) if len(dims) >= 3 else 1
        sample_format = image_el.get('sampleFormat')
        if sample_format:
            image['sample_format'] = XISF_SAMPLE_FORMATS.get(sample_format, sample_format)
        if image_el.get('colorSpace'):
            image['color_space'] = image_el.get('colorSpace')
    if _has_value(properties.get('XISF:CreatorApplication')):
        image['software'] = properties['XISF:CreatorApplication']

    return HeaderData(keywords, history, image)


def build_header_sections(header, image_size=None, fallback_wcs=None):
    """Organize a HeaderData into [(section title, [(label, value), ...]), ...].

    image_size: (width, height) of the decoded image, used when the header
        doesn't record its dimensions
    fallback_wcs: WCS keyword dict (e.g. a cached plate solve) to use when the
        file itself isn't plate solved
    """
    kw = header.keywords
    if header.image.get('width') and header.image.get('height'):
        image_size = (header.image['width'], header.image['height'])

    sections = [
        ("Target", _target_rows(kw)),
        ("Capture", _capture_rows(kw, header.history)),
        ("Equipment", _equipment_rows(kw, image_size)),
        ("Plate Solution", _plate_solution_rows(kw, image_size, fallback_wcs)),
        ("Site", _site_rows(kw)),
        ("Image Data", _image_rows(kw, header.image)),
    ]
    return [(title, rows) for title, rows in sections if rows]


def _target_rows(kw):
    rows = []
    _add_row(rows, "Object", _first_value(kw, 'OBJECT', 'OBJNAME'))
    coords = _file_coordinates(kw)
    if coords:
        rows.append(("RA", format_ra(coords[0])))
        rows.append(("Dec", format_dec(coords[1])))
    altitude = _to_float(_first_value(kw, 'OBJCTALT', 'CENTALT', 'ALTITUDE'))
    if altitude is not None:
        rows.append(("Altitude", f"{altitude:.1f}°"))
    azimuth = _to_float(_first_value(kw, 'OBJCTAZ', 'CENTAZ', 'AZIMUTH'))
    if azimuth is not None:
        rows.append(("Azimuth", f"{azimuth:.1f}°"))
    airmass = _to_float(kw.get('AIRMASS'))
    if airmass is not None:
        rows.append(("Airmass", f"{airmass:.3f}"))
    rotation = _to_float(kw.get('OBJCTROT'))
    if rotation is not None:
        rows.append(("Rotation", f"{_fmt_num(rotation, 2)}°"))
    _add_row(rows, "Pier Side", kw.get('PIERSIDE'))
    return rows


def _capture_rows(kw, history):
    rows = []
    _add_row(rows, "Frame Type", _first_value(kw, 'IMAGETYP', 'FRAME'))

    date_obs = kw.get('DATE-OBS')
    utc = _parse_iso_datetime(date_obs)
    if utc is not None:
        utc = utc.replace(tzinfo=timezone.utc) if utc.tzinfo is None else utc.astimezone(timezone.utc)
        rows.append(("Date (UTC)", format_datetime(utc, seconds=True)))
        # Prefer the capture site's local time when the file records it
        local = _parse_iso_datetime(kw.get('DATE-LOC'))
        if local is None:
            local = utc.astimezone()
        rows.append(("Date (Local)", format_datetime(local, seconds=True)))
    else:
        _add_row(rows, "Date", date_obs)

    exposure = _to_float(_first_value(kw, 'EXPTIME', 'EXPOSURE'))
    if exposure is not None:
        rows.append(("Exposure", format_duration(exposure)))

    frames = _to_float(_first_value(kw, 'NCOMBINE', 'STACKCNT'))
    if frames is None:
        # PixInsight records the integration count in the HISTORY cards
        for line in history:
            match = re.search(r'ImageIntegration\.numberOfImages:\s*(\d+)', str(line))
            if match:
                frames = float(match.group(1))
                break
    if frames is not None and frames > 1:
        rows.append(("Frames Stacked", f"{frames:.0f}"))
        live_time = _to_float(kw.get('LIVETIME'))
        if live_time is not None:
            rows.append(("Total Integration", format_duration(live_time)))

    _add_row(rows, "Filter", kw.get('FILTER'))
    for label, keyword in (("Gain", 'GAIN'), ("Offset", 'OFFSET')):
        number = _to_float(kw.get(keyword))
        _add_row(rows, label, _fmt_num(number, 2) if number is not None else kw.get(keyword))
    egain = _to_float(kw.get('EGAIN'))
    if egain is not None:
        rows.append(("e-/ADU", _fmt_num(egain, 3)))

    x_bin, y_bin = _to_float(kw.get('XBINNING')), _to_float(kw.get('YBINNING'))
    if x_bin is not None:
        rows.append(("Binning", f"{x_bin:.0f}x{(y_bin if y_bin is not None else x_bin):.0f}"))

    sensor_temp = _to_float(_first_value(kw, 'CCD-TEMP', 'CCD_TEMP', 'TEMPERAT'))
    set_temp = _to_float(_first_value(kw, 'SET-TEMP', 'SET_TEMP'))
    if sensor_temp is not None:
        text = f"{_fmt_num(sensor_temp, 1)} °C"
        if set_temp is not None:
            text += f" (set {_fmt_num(set_temp, 1)} °C)"
        rows.append(("Sensor Temp", text))
    elif set_temp is not None:
        rows.append(("Set Temp", f"{_fmt_num(set_temp, 1)} °C"))

    _add_row(rows, "Readout Mode", kw.get('READOUTM'))
    _add_row(rows, "Bayer Pattern", kw.get('BAYERPAT'))
    hfr = _to_float(kw.get('HFR'))
    if hfr is not None:
        rows.append(("HFR", f"{_fmt_num(hfr, 2)} px"))
    fwhm = _to_float(kw.get('FWHM'))
    if fwhm is not None:
        rows.append(("FWHM", _fmt_num(fwhm, 2)))
    return rows


def _equipment_rows(kw, image_size):
    rows = []
    _add_row(rows, "Telescope", kw.get('TELESCOP'))
    _add_row(rows, "Camera", kw.get('INSTRUME'))

    focal_length = _to_float(kw.get('FOCALLEN'))
    aperture = _to_float(kw.get('APTDIA'))
    if focal_length:
        rows.append(("Focal Length", f"{_fmt_num(focal_length, 1)} mm"))
    if aperture:
        rows.append(("Aperture", f"{_fmt_num(aperture, 1)} mm"))
    focal_ratio = _to_float(kw.get('FOCRATIO'))
    if not focal_ratio and focal_length and aperture:
        focal_ratio = focal_length / aperture
    if focal_ratio:
        rows.append(("Focal Ratio", f"f/{_fmt_num(focal_ratio, 1)}"))

    x_pixel, y_pixel = _to_float(kw.get('XPIXSZ')), _to_float(kw.get('YPIXSZ'))
    if x_pixel:
        text = f"{_fmt_num(x_pixel, 2)} µm"
        if y_pixel and abs(y_pixel - x_pixel) > 1e-6:
            text = f"{_fmt_num(x_pixel, 2)} x {_fmt_num(y_pixel, 2)} µm"
        rows.append(("Pixel Size", text))

    # Image scale as recorded, or from the optics (XPIXSZ already includes binning)
    scale = _to_float(_first_value(kw, 'PIXSCALE', 'SCALE'))
    if not scale and x_pixel and focal_length:
        scale = 206.265 * x_pixel / focal_length
    if scale:
        rows.append(("Image Scale", f"{scale:.2f}\"/px"))
        if image_size:
            rows.append(("Field of View", format_fov(scale, *image_size)))

    _add_row(rows, "Focuser Position", _first_value(kw, 'FOCPOS', 'FOCUSPOS'))
    focuser_temp = _to_float(_first_value(kw, 'FOCTEMP', 'FOCUSTEM'))
    if focuser_temp is not None:
        rows.append(("Focuser Temp", f"{_fmt_num(focuser_temp, 1)} °C"))
    rotator = _to_float(_first_value(kw, 'ROTATANG', 'ROTATOR'))
    if rotator is not None:
        rows.append(("Rotator Angle", f"{_fmt_num(rotator, 2)}°"))
    return rows


def _plate_solution_rows(kw, image_size, fallback_wcs):
    source = None
    wcs_keywords = {key: value for key, value in kw.items() if WCS_KEYWORD_PATTERN.match(key)}
    if _is_plate_solved(wcs_keywords):
        source = "File header"
    elif fallback_wcs:
        wcs_keywords = {key: value for key, value in fallback_wcs.items()
                        if WCS_KEYWORD_PATTERN.match(str(key))}
        if _is_plate_solved(wcs_keywords):
            source = "Cached plate solve"
    if source is None or not image_size:
        return []

    try:
        import numpy as np
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales

        wcs = WCS(wcs_keywords, naxis=2)
        width, height = image_size
        center = wcs.all_pix2world(np.array([[(width - 1) / 2.0, (height - 1) / 2.0]]), 0)[0]
        scale = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
        cd = wcs.pixel_scale_matrix
        rotation = math.degrees(math.atan2(-cd[0][1], cd[1][1]))
    except Exception as e:
        logger.debug(f"Could not evaluate plate solution: {e}")
        return []

    return [
        ("Center RA", format_ra(center[0])),
        ("Center Dec", format_dec(center[1])),
        ("Image Scale", f"{scale:.2f}\"/px"),
        ("Rotation", f"{rotation:.2f}°"),
        ("Field of View", format_fov(scale, width, height)),
        ("Source", source),
    ]


def _site_rows(kw):
    rows = []
    latitude = _to_degrees(_first_value(kw, 'SITELAT', 'OBSGEO-B', 'LAT-OBS'))
    if latitude is not None:
        rows.append(("Latitude", f"{abs(latitude):.4f}° {'N' if latitude >= 0 else 'S'}"))
    longitude = _to_degrees(_first_value(kw, 'SITELONG', 'OBSGEO-L', 'LONG-OBS'))
    if longitude is not None:
        if longitude > 180:
            longitude -= 360
        rows.append(("Longitude", f"{abs(longitude):.4f}° {'E' if longitude >= 0 else 'W'}"))
    elevation = _to_float(_first_value(kw, 'SITEELEV', 'OBSGEO-H', 'ALT-OBS'))
    if elevation is not None:
        rows.append(("Elevation", f"{elevation:.0f} m"))
    return rows


def _image_rows(kw, image):
    rows = []
    channels = image.get('channels')
    if channels:
        rows.append(("Channels", f"{channels} ({'mono' if channels == 1 else 'color'})"))
    _add_row(rows, "Sample Format", image.get('sample_format'))
    _add_row(rows, "Color Space", image.get('color_space'))
    _add_row(rows, "Created By", _first_value(kw, 'SWCREATE', 'CREATOR', 'PROGRAM') or image.get('software'))
    _add_row(rows, "Modified By", kw.get('SWMODIFY'))
    return rows


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    # astropy represents a keyword without a value as an Undefined instance
    return type(value).__name__ != 'Undefined'


def _first_value(kw, *names):
    """Value of the first keyword in names that the header has"""
    for name in names:
        if _has_value(kw.get(name)):
            return kw[name]
    return None


def _add_row(rows, label, value):
    """Append (label, value) if there's a value"""
    if _has_value(value) and not isinstance(value, bool):
        rows.append((label, str(value).strip()))


def _to_float(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _to_degrees(value):
    """Degrees from a number or a sexagesimal string like '+40 30 00'"""
    result = _to_float(value)
    if result is None and isinstance(value, str):
        result = _parse_sexagesimal(value)
    return result


def _fmt_num(value, decimals):
    """Number with up to `decimals` places, without trailing zeros"""
    return f"{value:.{decimals}f}".rstrip('0').rstrip('.')


def _parse_iso_datetime(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace('Z', '+00:00')
    # fromisoformat only takes up to 6 fractional digits
    text = re.sub(r'(\.\d{6})\d+', r'\1', text)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _is_plate_solved(wcs_keywords):
    return ('CRVAL1' in wcs_keywords and 'CRVAL2' in wcs_keywords and 'CRPIX1' in wcs_keywords
            and ('CD1_1' in wcs_keywords or 'CDELT1' in wcs_keywords))


def format_duration(seconds):
    """'300 s (5m)' style exposure/integration time"""
    text = f"{_fmt_num(seconds, 3)} s"
    if seconds < 60:
        return text
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours:.0f}h")
    if minutes:
        parts.append(f"{minutes:.0f}m")
    if secs >= 0.05:
        parts.append(f"{_fmt_num(secs, 1)}s")
    return f"{text} ({' '.join(parts)})"


def format_ra(ra_deg):
    """RA in degrees as '05h 35m 17.3s'"""
    total_seconds = round((ra_deg % 360.0) / 15.0 * 3600.0, 1) % 86400.0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02.0f}h {minutes:02.0f}m {seconds:04.1f}s"


def format_dec(dec_deg):
    """Dec in degrees as '+22° 00' 52"'"""
    sign = '-' if dec_deg < 0 else '+'
    total_arcsec = round(abs(dec_deg) * 3600.0)
    degrees, remainder = divmod(total_arcsec, 3600)
    arcmin, arcsec = divmod(remainder, 60)
    return f"{sign}{degrees:02d}° {arcmin:02d}' {arcsec:02d}\""


def format_fov(scale_arcsec, width, height):
    """Field of view for an image scale in arcsec/px, in arcmin or degrees"""
    width_arcmin = scale_arcsec * width / 60.0
    height_arcmin = scale_arcsec * height / 60.0
    if max(width_arcmin, height_arcmin) >= 120:
        return f"{width_arcmin / 60.0:.2f}° x {height_arcmin / 60.0:.2f}°"
    return f"{width_arcmin:.1f}' x {height_arcmin:.1f}'"


class ImageViewerWindow(QDialog):
    """Window to display an image in full size with enhanced controls"""
    zoom_changed = Signal(float)  # Signal for zoom level changes

    def __init__(self, pixmap: QPixmap, title: str, file_path: str = None, parent=None,
                 dso_ra: float = None, dso_dec: float = None, screen_stretch: bool = False):
        """pixmap may be None when file_path is given - the window then opens
        straight away and loads the image in a background thread. screen_stretch
        loads a FITS/XISF raw sub with an auto screen stretch (debayered on a color
        camera) instead of the plain linear one - for reviewing light frames."""
        super().__init__(parent)
        self.screen_stretch = screen_stretch
        self.setWindowTitle(f"{title} - Image Viewer - Cosmos Collection")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(800, 600)
        WindowPositionManager.restore_window_position(self, "ImageViewer")

        self.setMinimumSize(300, 300)

        # Store the original pixmap and file path
        self.original_pixmap = pixmap
        self.file_path = file_path
        self.zoom_factor = 1.0
        self.initial_zoom_factor = 1.0
        self.image_position = [0, 0]
        self.last_mouse_pos = None
        self.is_panning = False
        self._scaled_cache = None  # (zoom_factor, downscaled pixmap) for zoom < 100%
        self.image_load_worker = None

        # Store DSO coordinates for plate solving hints
        self.dso_ra = dso_ra  # RA in degrees
        self.dso_dec = dso_dec  # Dec in degrees

        # Create main layout
        main_layout = QVBoxLayout()

        # Create toolbar for controls
        toolbar = QHBoxLayout()

        # Add zoom controls
        zoom_out_button = QPushButton("-")
        themed_style(zoom_out_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
        zoom_out_button.setToolTip("Zoom out")
        zoom_out_button.clicked.connect(self._zoom_out)
        toolbar.addWidget(zoom_out_button)

        zoom_in_button = QPushButton("+")
        themed_style(zoom_in_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
        zoom_in_button.setToolTip("Zoom in")
        zoom_in_button.clicked.connect(self._zoom_in)
        toolbar.addWidget(zoom_in_button)

        reset_button = QPushButton("Reset")
        reset_button.setFixedHeight(30)
        reset_button.setToolTip("Reset zoom to 100%")
        reset_button.clicked.connect(self._reset_zoom)
        toolbar.addWidget(reset_button)

        # Add file-specific buttons if file path is available
        if self.file_path:
            open_location_button = QPushButton("Open File Location")
            open_location_button.setFixedHeight(30)
            open_location_button.setToolTip("Open the folder containing this image")
            open_location_button.clicked.connect(self._open_file_location)
            toolbar.addWidget(open_location_button)

            self.set_bg_button = QPushButton("Set as Background")
            self.set_bg_button.setFixedHeight(30)
            self.set_bg_button.setToolTip("Set this image as your desktop background")
            self.set_bg_button.clicked.connect(self._set_as_background)
            toolbar.addWidget(self.set_bg_button)

            # Annotations, as in the gallery: on/off, which layers, plate solving
            toolbar.addSpacing(6)
            self.annotations_check = QCheckBox("Show Annotations")
            self.annotations_check.setToolTip("Mark the objects in the image (it needs a plate solve)")
            toolbar.addWidget(self.annotations_check)
            self.layers_button = QToolButton()
            self.layers_button.setText("Layers")
            self.layers_button.setToolTip("What the annotations show")
            self.layers_button.setPopupMode(QToolButton.InstantPopup)
            layers_menu = QMenu(self.layers_button)
            layers = self._annotation_layers()
            self.layer_actions = {}
            for key, label in ANNOTATION_LAYERS:
                action = layers_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(layers[key])
                action.toggled.connect(self._update_annotation_settings)
                self.layer_actions[key] = action
            # Dark nebulae are among the DSOs
            self.layer_actions["dark"].setEnabled(layers["dsos"])
            self.layer_actions["dsos"].toggled.connect(self.layer_actions["dark"].setEnabled)
            self.layers_button.setMenu(layers_menu)
            toolbar.addWidget(self.layers_button)
            self.solve_button = QPushButton("Plate Solve")
            self.solve_button.setFixedHeight(30)
            self.solve_button.clicked.connect(self._start_plate_solve)
            toolbar.addWidget(self.solve_button)

            self.save_annotated_button = QPushButton("Save with Annotations")
            self.save_annotated_button.setFixedHeight(30)
            self.save_annotated_button.setToolTip("Save image with annotations overlaid")
            self.save_annotated_button.clicked.connect(self._save_with_annotations)
            toolbar.addWidget(self.save_annotated_button)

        toolbar.addStretch()

        # Add file info toggle button on the right side
        if self.file_path:
            self.info_toggle_button = QPushButton("Show File Info")
            self.info_toggle_button.setFixedHeight(30)
            self.info_toggle_button.setToolTip("Show/hide EXIF and file information")
            self.info_toggle_button.setCheckable(True)
            self.info_toggle_button.setChecked(False)
            self.info_toggle_button.clicked.connect(self._toggle_file_info)
            toolbar.addWidget(self.info_toggle_button)

        # Annotations come from the shared store (loaded in the background from
        # the image's .wcs), or from a solve that couldn't be saved next to it
        self._unsaved_annotations = None
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.annotations_enabled = settings.value("annotation_enabled", False, type=bool)
        if self.file_path:
            annotation_store().ready.connect(self._on_annotations_ready)
            self.annotations_check.setChecked(self.annotations_enabled)
            self.annotations_check.toggled.connect(self._toggle_annotations)
        self.plate_solve_result = None
        self.plate_solve_worker = None
        self._solve_message = None       # Plate solving's progress or result, for the status bar
        self._solving = self._stopping = False
        self._solve_message_timer = QTimer(self)  # Clears a result a moment after it shows
        self._solve_message_timer.setSingleShot(True)
        self._solve_message_timer.timeout.connect(self._clear_solve_message)
        self.background_worker = None
        self.save_worker = None

        main_layout.addLayout(toolbar)

        # Create horizontal layout for image and file info panel
        content_layout = QHBoxLayout()

        # Create a container widget for the image
        self.image_container = QWidget()
        self.image_container.setLayout(QVBoxLayout())
        self.image_container.setStyleSheet("background-color: black;")

        # Create image label
        self.image_label = QLabel()
        if pixmap is not None:
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText(f"Loading {Path(file_path).name}..." if file_path else "No image")
            themed_style(self.image_label, lambda: f"color: {adapt_color('#aaaaaa', on_dark=True)}; font-size: {font_size(12)};")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        # Add image label to container
        self.image_container.layout().addWidget(self.image_label)
        content_layout.addWidget(self.image_container)

        # Create file information panel as a right-side groupbox (hidden by default)
        self.file_info_panel = QGroupBox("File Information")
        self.file_info_panel.setVisible(False)
        self.file_info_panel.setFixedWidth(350)

        # Dark mode styling for the groupbox
        themed_style(self.file_info_panel, lambda: f"""
            QGroupBox {{
                font-weight: bold;
                font-size: {font_size(12)};
                color: {COLORS['text']};
                background-color: {COLORS['background']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: {COLORS['text']};
                background-color: {COLORS['background']};
            }}
        """)

        # Create file info layout
        file_info_layout = QVBoxLayout()
        file_info_layout.setContentsMargins(15, 10, 15, 15)

        # Create scrollable area for file info content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Values wrap to the panel width instead
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Dark mode styling for scroll area
        themed_style(scroll_area, lambda: f"""
            QScrollArea {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                background-color: {COLORS['background']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['border']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['border_light']};
            }}
        """)

        self.file_info_content = QLabel()
        # Dark mode styling for the content label
        themed_style(self.file_info_content, lambda: f"""
            QLabel {{
                font-size: {font_size(10)};
                color: {COLORS['text_secondary']};
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                padding: 10px;
                background-color: {COLORS['background_light']};
                border: none;
            }}
        """)
        self.file_info_content.setWordWrap(True)
        self.file_info_content.setAlignment(Qt.AlignTop)
        self.file_info_content.setTextFormat(Qt.RichText)
        # The section colors are baked into the rich text, so re-render on theme changes
        theme_manager().theme_changed.connect(self._refresh_file_info)

        scroll_area.setWidget(self.file_info_content)
        file_info_layout.addWidget(scroll_area)

        self.file_info_panel.setLayout(file_info_layout)
        content_layout.addWidget(self.file_info_panel)

        # Set stretch ratios - image container gets most space, info panel gets fixed width
        content_layout.setStretch(0, 1)  # image_container
        content_layout.setStretch(1, 0)  # file_info_panel

        main_layout.addLayout(content_layout)

        # Add status bar - with a busy bar and Stop while plate solving
        status_row = QHBoxLayout()
        self.status_bar = QLabel()
        self.status_bar.setMinimumWidth(1)  # Long messages are cut short, not widening the window
        themed_style(self.status_bar, lambda: f"font-size: {font_size(10)};")
        status_row.addWidget(self.status_bar, 1)
        self.solve_busy = QProgressBar()
        self.solve_busy.setRange(0, 0)  # Busy, no percentage
        self.solve_busy.setTextVisible(False)
        self.solve_busy.setFixedSize(110, 12)
        self.solve_busy.hide()
        status_row.addWidget(self.solve_busy)
        self.solve_stop = QPushButton("Stop")
        self.solve_stop.setToolTip("Stop plate solving (after the solver that's running)")
        self.solve_stop.clicked.connect(self._stop_plate_solve)
        self.solve_stop.hide()
        status_row.addWidget(self.solve_stop)
        main_layout.addLayout(status_row)

        # Set stretch to ensure content takes most space and status bar stays at bottom
        main_layout.setStretch(0, 0)  # toolbar
        main_layout.setStretch(1, 1)  # content_layout
        main_layout.setStretch(2, 0)  # status_bar

        self.setLayout(main_layout)

        # Flag to track if initial fit has been done
        self.initial_fit_done = False

        # No pixmap given - load the file in the background so the UI stays responsive
        if self.original_pixmap is None and self.file_path:
            self._start_image_load()

        # Reopen the file info panel if it was open last time
        if self.file_path and QSettings("CosmosCollection", "CosmosCollection").value(
                self.INFO_PANEL_SETTING, False, type=bool):
            self._set_file_info_visible(True)

        # Update status
        self._update_annotation_controls()
        self._update_status()

    def _image_buttons(self):
        """Toolbar buttons that need the image to be loaded"""
        return [getattr(self, name) for name in
                ('set_bg_button', 'save_annotated_button')
                if hasattr(self, name)]

    def _start_image_load(self):
        """Load self.file_path in a worker thread"""
        for button in self._image_buttons():
            button.setEnabled(False)

        self.image_load_worker = ImageLoadWorker(self.file_path, screen_stretch=self.screen_stretch)
        self.image_load_worker.image_loaded.connect(self._on_image_loaded)
        self.image_load_worker.load_failed.connect(self._on_image_load_failed)
        self.image_load_worker.start()

    def _on_image_loaded(self, qimage):
        """Show the image once the worker has decoded it"""
        # QPixmap must be created on the GUI thread
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            self._on_image_load_failed("Failed to convert image")
            return

        self.original_pixmap = pixmap
        self._scaled_cache = None
        self.image_label.setStyleSheet("")
        self.image_label.setText("")
        for button in self._image_buttons():
            button.setEnabled(True)
        # A panel opened before the image arrived is missing the image's size
        if self.file_info_panel.isVisible():
            self._load_file_information()

        if self.isVisible():
            self._do_initial_fit()
        # Otherwise showEvent does the initial fit

    def _on_image_load_failed(self, error_message):
        """Show why the image couldn't be loaded"""
        logger.error(f"Image viewer failed to load {self.file_path}: {error_message}")
        self.image_label.setText(f"Failed to load image:\n{Path(self.file_path).name}\n\n{error_message}")

    def showEvent(self, event):
        """Handle window show event - fit image to window on first show"""
        super().showEvent(event)
        if not self.initial_fit_done:
            # Defer the initial fit to ensure the window is properly sized
            QTimer.singleShot(50, self._do_initial_fit)

    def _do_initial_fit(self):
        """Perform initial fit to window and set up initial zoom factor"""
        # Still loading - _on_image_loaded calls this again when the image arrives
        if self.original_pixmap is None:
            return

        if not self.initial_fit_done:
            self._fit_to_window()
            self.initial_zoom_factor = self.zoom_factor
            self.initial_fit_done = True

    def eventFilter(self, obj, event):
        """Handle mouse events for zooming and panning"""
        if obj == self.image_label:
            if event.type() == QEvent.Wheel:
                if event.angleDelta().y() > 0:
                    self._zoom_in()
                else:
                    self._zoom_out()
                return True
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.is_panning = True
                self.last_mouse_pos = event.position()
                self.setCursor(Qt.ClosedHandCursor)
                return True
            elif event.type() == QEvent.MouseMove:
                if self.is_panning and self.last_mouse_pos is not None:
                    # Calculate the drag distance
                    dx = event.position().x() - self.last_mouse_pos.x()
                    dy = event.position().y() - self.last_mouse_pos.y()

                    # Update image position
                    self.image_position[0] += dx
                    self.image_position[1] += dy

                    # Update the display
                    self._update_zoom()

                    self.last_mouse_pos = event.position()
                    return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.is_panning = False
                self.last_mouse_pos = None
                self.setCursor(Qt.ArrowCursor)
                return True
        return super().eventFilter(obj, event)

    def _update_zoom(self):
        """Update the image display with current zoom level and position"""
        if self.original_pixmap is None:
            return

        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()

        # Size of the whole image at the current zoom
        new_width = int(img_w * self.zoom_factor)
        new_height = int(img_h * self.zoom_factor)

        # Create a new pixmap with the same size as the label
        label_size = self.image_label.size()
        final_pixmap = QPixmap(label_size)
        final_pixmap.fill(Qt.transparent)

        # Create a painter to draw the scaled image at the correct position
        painter = QPainter(final_pixmap)

        # Calculate the position to draw the image
        x = (label_size.width() - new_width) // 2 + self.image_position[0]
        y = (label_size.height() - new_height) // 2 + self.image_position[1]

        # Draw only the part of the image that's visible in the label, rather
        # than scaling the whole image on every pan/zoom (at high zoom on a large
        # image that meant building a pixmap of a gigabyte or more each time).
        # Zoomed out, use a downscaled copy cached per zoom level - Qt's smooth
        # scaling averages pixels properly when shrinking, whereas the painter's
        # bilinear filtering would alias. Zoomed in, sample the original directly.
        if new_width < 1 or new_height < 1:
            source_pixmap = None  # Window not sized yet - nothing to draw
        elif self.zoom_factor < 1.0:
            if self._scaled_cache is None or self._scaled_cache[0] != self.zoom_factor:
                self._scaled_cache = (self.zoom_factor, self.original_pixmap.scaled(
                    new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            source_pixmap = self._scaled_cache[1]
        else:
            self._scaled_cache = None
            source_pixmap = self.original_pixmap

        # Visible region in image coordinates, clamped to the image bounds
        if source_pixmap is not None:
            left = max(0.0, -x / self.zoom_factor)
            top = max(0.0, -y / self.zoom_factor)
            right = min(float(img_w), (label_size.width() - x) / self.zoom_factor)
            bottom = min(float(img_h), (label_size.height() - y) / self.zoom_factor)

        if source_pixmap is not None and right > left and bottom > top:
            target = QRectF(x + left * self.zoom_factor, y + top * self.zoom_factor,
                            (right - left) * self.zoom_factor, (bottom - top) * self.zoom_factor)
            # Source pixmap may be the downscaled copy - map image coords onto it
            sx = source_pixmap.width() / img_w
            sy = source_pixmap.height() / img_h
            source = QRectF(left * sx, top * sy, (right - left) * sx, (bottom - top) * sy)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawPixmap(target, source_pixmap, source)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        # Annotations over the whole image (clipped to it), names a readable size on screen
        data = self._annotations() if self.annotations_enabled else None
        if data is not None and new_width >= 1 and new_height >= 1:
            whole = QRectF(x, y, img_w * self.zoom_factor, img_h * self.zoom_factor)
            paint_annotations(painter, data, self._annotation_layers(), img_w, QRectF(0, 0, img_w, img_h),
                              whole, font_px=screen_name_px(whole))

        painter.end()

        # Update the display
        self.image_label.setPixmap(final_pixmap)

        # Update status
        self._update_status()
        self.zoom_changed.emit(self.zoom_factor)  # Emit zoom level change signal

    def _fit_to_window(self):
        """Fit the image to the window size"""
        if self.original_pixmap is None:
            return

        # Get available size
        available_size = self.image_container.size()

        # Calculate zoom factor to fit
        width_ratio = available_size.width() / self.original_pixmap.width()
        height_ratio = available_size.height() / self.original_pixmap.height()
        self.zoom_factor = min(width_ratio, height_ratio)

        # Reset position
        self.image_position = [0, 0]

        # Update display
        self._update_zoom()

    def resizeEvent(self, event):
        """Handle window resize event"""
        super().resizeEvent(event)

        if not self.initial_fit_done:
            # If initial fit hasn't been done yet, just update the display
            self._update_zoom()
            return

        # Store the current relative position of the image center before resize
        old_size = event.oldSize() if event.oldSize().isValid() else self.size()
        new_size = event.size()

        # Only auto-fit if we're at the initial zoom level (user hasn't manually zoomed)
        if abs(self.zoom_factor - self.initial_zoom_factor) < 0.001:
            # User is at initial zoom - maintain fit to window behavior
            self._fit_to_window()
            self.initial_zoom_factor = self.zoom_factor
        else:
            # User has manually zoomed - preserve zoom level and try to maintain image position
            if old_size.isValid() and old_size.width() > 0 and old_size.height() > 0:
                # Calculate the ratio of size change
                width_ratio = new_size.width() / old_size.width()
                height_ratio = new_size.height() / old_size.height()

                # Adjust image position proportionally to maintain relative position
                self.image_position[0] = int(self.image_position[0] * width_ratio)
                self.image_position[1] = int(self.image_position[1] * height_ratio)

            # Update the display with preserved zoom and adjusted position
            self._update_zoom()

    def _zoom_in(self):
        """Zoom in on the image"""
        if self.original_pixmap is None:
            return
        self.zoom_factor = min(self.zoom_factor * 1.2, 8.0)
        self._update_zoom()

    def _zoom_out(self):
        """Zoom out on the image"""
        if self.original_pixmap is None:
            return
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self._update_zoom()

    def _reset_zoom(self):
        """Reset zoom to 100%"""
        if self.original_pixmap is None:
            return
        self.zoom_factor = 1.0
        self.image_position = [0, 0]
        self._update_zoom()

    def _update_status(self, annotation_status=None):
        """Update the status bar with current zoom level and image size"""
        if self.original_pixmap is None:
            self.status_bar.setText("Loading image...")
            return

        zoom_percent = int(self.zoom_factor * 100)
        image_size = f"{self.original_pixmap.width()}x{self.original_pixmap.height()}"
        base_status = f"Zoom: {zoom_percent}% | Image Size: {image_size} pixels"

        annotation_status = annotation_status or self._solve_message or self._annotation_status()
        if annotation_status:
            self.status_bar.setText(f"{base_status} | {annotation_status}")
        else:
            self.status_bar.setText(base_status)

    def _open_file_location(self):
        """Open the file location in the system's file explorer"""
        if self.file_path:
            success = ResourceManager.open_file_manager(self.file_path)
            if not success:
                QMessageBox.critical(self, "Error", "Failed to open file location")

    def _set_as_background(self):
        """Set the current image as the desktop background. The PNG conversion and
        OS call run in a worker thread so the window doesn't hang on large images."""
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "No file path available")
            return

        if self.background_worker and self.background_worker.isRunning():
            return

        # FITS/XISF can't be used as a wallpaper directly - convert the stretched
        # render already shown in the viewer to PNG. QPixmap isn't safe to use off
        # the GUI thread, so hand the worker a QImage copy.
        image = None
        if Path(self.file_path).suffix.lower() in ['.fits', '.fit', '.fts', '.xisf']:
            if self.original_pixmap is None or self.original_pixmap.isNull():
                QMessageBox.critical(self, "Error", "No image available to convert for desktop background")
                return
            image = self.original_pixmap.toImage()

        self.set_bg_button.setEnabled(False)
        self.setCursor(Qt.BusyCursor)
        self._update_status("Converting image to PNG..." if image is not None
                            else "Setting desktop background...")

        self.background_worker = BackgroundSetterWorker(self.file_path, image)
        self.background_worker.background_set.connect(self._on_background_set)
        self.background_worker.start()

    def _on_background_set(self, level, message):
        """Show the result of the background worker"""
        self.set_bg_button.setEnabled(True)
        self.unsetCursor()
        self._update_status()

        if level == "success":
            QMessageBox.information(self, "Success", message)
        elif level == "warning":
            QMessageBox.warning(self, "Warning", message)
        else:
            QMessageBox.critical(self, "Error", message)

    @staticmethod
    def _apply_desktop_background(abs_path):
        """Set abs_path as the desktop wallpaper. Safe to call from a worker thread.
        Returns (level, message) where level is 'success', 'warning' or 'error'."""
        success = ("success", "Desktop background updated successfully!")

        if platform.system() == "Windows":
            # Windows implementation using ctypes
            SPI_SETDESKWALLPAPER = 20
            SPIF_UPDATEINIFILE = 0x01
            SPIF_SENDCHANGE = 0x02

            # Call SystemParametersInfoW to set the wallpaper
            result = ctypes.windll.user32.SystemParametersInfoW(
                SPI_SETDESKWALLPAPER,
                0,
                abs_path,
                SPIF_UPDATEINIFILE | SPIF_SENDCHANGE
            )
            return success if result else ("error", "Failed to set desktop background")

        if platform.system() == "Darwin":
            # macOS implementation
            script = f'''
            tell application "Finder"
                set desktop picture to POSIX file "{abs_path}"
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return success

        if platform.system() == "Linux":
            desktop = ImageViewerWindow._linux_desktop_name()
            if ImageViewerWindow._set_linux_background(abs_path, desktop):
                return success
            return ("warning",
                    f"Could not set background on this desktop environment "
                    f"({desktop or 'unknown'}).\n"
                    f"Supported: KDE Plasma, GNOME, Cinnamon, MATE, XFCE.")

        return ("warning", f"Setting desktop background is not supported on {platform.system()}")

    @staticmethod
    def _linux_desktop_name():
        """Return the current desktop environment in lowercase, e.g. 'kde:plasma'
        or 'ubuntu:gnome' (XDG_CURRENT_DESKTOP can hold several names separated by ':')"""
        return (os.environ.get('XDG_CURRENT_DESKTOP')
                or os.environ.get('DESKTOP_SESSION')
                or '').lower()

    @staticmethod
    def _set_linux_background(abs_path, desktop):
        """Set the wallpaper using the right tool for the Linux desktop environment.
        Returns True on success, False if the desktop isn't supported or the command failed."""
        uri = Path(abs_path).as_uri()

        def run(*args):
            subprocess.run(list(args), check=True, capture_output=True, timeout=15)

        try:
            if 'kde' in desktop or 'plasma' in desktop:
                # Plasma 5.24+ ships a CLI tool for this
                try:
                    run("plasma-apply-wallpaperimage", abs_path)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

                # Older Plasma - set it through a plasmashell script over D-Bus
                script = (
                    "var allDesktops = desktops();"
                    "for (var i = 0; i < allDesktops.length; i++) {"
                    "  var d = allDesktops[i];"
                    "  d.wallpaperPlugin = 'org.kde.image';"
                    "  d.currentConfigGroup = ['Wallpaper', 'org.kde.image', 'General'];"
                    f"  d.writeConfig('Image', {json.dumps(uri)});"
                    "}"
                )
                for qdbus in ("qdbus6", "qdbus", "qdbus-qt5"):
                    try:
                        run(qdbus, "org.kde.plasmashell", "/PlasmaShell",
                            "org.kde.PlasmaShell.evaluateScript", script)
                        return True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                return False

            if 'xfce' in desktop:
                # Every monitor/workspace has its own last-image property
                result = subprocess.run(["xfconf-query", "-c", "xfce4-desktop", "-l"],
                                        check=True, capture_output=True, text=True, timeout=15)
                props = [p for p in result.stdout.split() if p.endswith('/last-image')]
                if not props:
                    return False
                for prop in props:
                    run("xfconf-query", "-c", "xfce4-desktop", "-p", prop, "-s", abs_path)
                return True

            if 'cinnamon' in desktop:
                run("gsettings", "set", "org.cinnamon.desktop.background", "picture-uri", uri)
                return True

            if 'mate' in desktop:
                run("gsettings", "set", "org.mate.background", "picture-filename", abs_path)
                return True

            if any(name in desktop for name in ('gnome', 'unity', 'budgie', 'pantheon')):
                run("gsettings", "set", "org.gnome.desktop.background", "picture-uri", uri)
                # GNOME 42+ uses a separate key in dark mode; older versions lack it
                try:
                    run("gsettings", "set", "org.gnome.desktop.background", "picture-uri-dark", uri)
                except subprocess.CalledProcessError:
                    pass
                return True

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to set background on desktop '{desktop}': {e}")
            return False

        logger.warning(f"Setting background not supported on desktop '{desktop}'")
        return False

    @staticmethod
    def _export_background_png(image, file_path):
        """Save image (a QImage) as a PNG next to the original file for use as a
        desktop background. The file has to persist since macOS/GNOME reference it
        directly. Safe to call from a worker thread. Returns the absolute path, or
        None on failure."""
        # "_background" suffix so an existing PNG of the same name isn't overwritten
        source = Path(file_path).resolve()
        png_path = source.with_name(f"{source.stem}_background.png")
        from ImageLoader import write_image_file
        if not write_image_file(image, str(png_path), "PNG"):
            logger.error(f"Failed to save background PNG to {png_path}")
            return None

        logger.info(f"Converted {file_path} to {png_path} for desktop background")
        return str(png_path)

    def _save_with_annotations(self):
        """Save the current image with annotations overlaid"""
        if not self.file_path or not self.original_pixmap:
            QMessageBox.warning(self, "Warning", "No image available to save")
            return

        data = self._annotations()
        if data is None:
            if annotation_store().known(self.file_path) or self._unsaved_annotations is not None:
                message = ("This image isn't plate solved yet, so there's nothing to annotate.\n\n"
                           "Use 'Show Annotations' to plate solve it first.")
            else:
                message = "The annotations are still loading - try again in a moment."
            QMessageBox.information(self, "No Annotations", message)
            return

        if self.save_worker and self.save_worker.isRunning():
            return

        try:
            # Draw onto a QImage copy rather than a QPixmap - the encoding and
            # writing happen in a worker thread, where QPixmap isn't safe to use.
            annotated_image = self.original_pixmap.toImage().convertToFormat(QImage.Format_RGB32)

            # Full resolution, with names sized relative to the image (as in a collage)
            painter = QPainter(annotated_image)
            whole = QRectF(annotated_image.rect())
            paint_annotations(painter, data, self._annotation_layers(), annotated_image.width(), whole, whole)
            painter.end()

            # Generate default filename - keep JPEG as JPEG, everything else
            # (including FITS/XISF, which can only be saved as PNG/JPEG here) as PNG
            original_path = Path(self.file_path)
            default_ext = '.jpg' if original_path.suffix.lower() in ('.jpg', '.jpeg') else '.png'
            default_name = f"{original_path.stem}_annotated{default_ext}"
            default_path = str(original_path.parent / default_name)

            # Open save dialog
            png_filter = "PNG Image (*.png)"
            jpeg_filter = "JPEG Image (*.jpg *.jpeg)"
            save_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save Annotated Image",
                default_path,
                f"{png_filter};;{jpeg_filter}",
                jpeg_filter if default_ext == '.jpg' else png_filter
            )

            if save_path:
                # Only PNG and JPEG can be written - if the name has any other
                # extension (e.g. .xisf), use the chosen filter's extension so the
                # file's name matches its contents
                save_ext = Path(save_path).suffix.lower()
                if save_ext not in ('.png', '.jpg', '.jpeg'):
                    save_ext = '.jpg' if selected_filter == jpeg_filter else '.png'
                    save_path = str(Path(save_path).with_suffix(save_ext))
                    if Path(save_path).exists() and QMessageBox.question(
                            self, "Overwrite File",
                            f"{Path(save_path).name} already exists.\nDo you want to replace it?"
                    ) != QMessageBox.Yes:
                        return

                if save_ext in ('.jpg', '.jpeg'):
                    self.save_worker = ImageSaveWorker(annotated_image, save_path, "JPEG", 95)
                else:
                    self.save_worker = ImageSaveWorker(annotated_image, save_path, "PNG")

                self.save_annotated_button.setEnabled(False)
                self.setCursor(Qt.BusyCursor)
                self._update_status(f"Saving {Path(save_path).name}...")
                self.save_worker.image_saved.connect(self._on_annotated_image_saved)
                self.save_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotated image: {str(e)}")

    def _on_annotated_image_saved(self, success, save_path):
        """Show the result of the save worker"""
        self.save_annotated_button.setEnabled(True)
        self.unsetCursor()
        self._update_status()

        if success:
            QMessageBox.information(self, "Success",
                                  f"Annotated image saved successfully!\n{save_path}")
        else:
            QMessageBox.critical(self, "Error", f"Failed to save annotated image:\n{save_path}")

    INFO_PANEL_SETTING = "image_viewer/file_info_visible"

    def _toggle_file_info(self):
        """Toggle the file information panel, remembering the choice for next time"""
        if hasattr(self, 'file_info_panel'):
            visible = not self.file_info_panel.isVisible()
            self._set_file_info_visible(visible)
            QSettings("CosmosCollection", "CosmosCollection").setValue(self.INFO_PANEL_SETTING, visible)

    def _set_file_info_visible(self, visible):
        self.file_info_panel.setVisible(visible)
        if hasattr(self, 'info_toggle_button'):
            self.info_toggle_button.setChecked(visible)
            self.info_toggle_button.setText("Hide File Info" if visible else "Show File Info")
        if visible:
            # Load and display file information when showing
            self._load_file_information()

    def _load_file_information(self):
        """Load and display file information"""
        if not self.file_path or not hasattr(self, 'file_info_content'):
            return

        try:
            # Get file stats
            file_stats = os.stat(self.file_path)
            file_path_obj = Path(self.file_path)

            # Basic file information
            file_name = file_path_obj.name
            file_dir = str(file_path_obj.parent)
            file_size_bytes = file_stats.st_size

            # Format file size
            if file_size_bytes < 1024:
                file_size = f"{file_size_bytes} bytes"
            elif file_size_bytes < 1024 * 1024:
                file_size = f"{file_size_bytes / 1024:.1f} KB"
            elif file_size_bytes < 1024 * 1024 * 1024:
                file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
            else:
                file_size = f"{file_size_bytes / (1024 * 1024 * 1024):.1f} GB"

            # Dates
            created_time = format_datetime(datetime.fromtimestamp(file_stats.st_ctime), seconds=True)
            modified_time = format_datetime(datetime.fromtimestamp(file_stats.st_mtime), seconds=True)

            # FITS/XISF header - read before the dimensions so they're known
            # even while the image itself is still loading
            header = None
            if file_path_obj.suffix.lower() in FITS_EXTENSIONS | XISF_EXTENSIONS:
                try:
                    header = read_image_header(self.file_path)
                except Exception as e:
                    logger.warning(f"Failed to read header from {self.file_path}: {e}")

            # Image dimensions
            if self.original_pixmap:
                image_size = (self.original_pixmap.width(), self.original_pixmap.height())
            elif header and header.image.get('width') and header.image.get('height'):
                image_size = (header.image['width'], header.image['height'])
            else:
                image_size = None

            if image_size:
                width, height = image_size
                megapixels = (width * height) / 1000000
                if megapixels >= 1:
                    dimensions = f"{width} x {height} pixels ({megapixels:.1f} MP)"
                else:
                    dimensions = f"{width} x {height} pixels ({megapixels * 1000:.0f}K pixels)"
            else:
                dimensions = "Unknown"

            sections = [("File", [
                ("Filename", file_name),
                ("Location", file_dir),
                ("File Size", file_size),
                ("Dimensions", dimensions),
                ("Created", created_time),
                ("Modified", modified_time),
            ])]

            if header is not None:
                cached_wcs = self._cached_wcs_header()
                if cached_wcs and image_size:
                    cached_wcs = self._wcs_for_display(cached_wcs, image_size[1])
                sections.extend(build_header_sections(header, image_size, cached_wcs))
            else:
                # Try to get EXIF data if available
                exif_info = self._get_exif_info()
                if exif_info:
                    sections.append(("EXIF", list(exif_info.items())))

            self.file_info_content.setText(self._file_info_html(sections))

        except Exception as e:
            self.file_info_content.setText(f"Error loading file information:<br>{escape(str(e))}")

    @staticmethod
    def _file_info_html(sections):
        """Render [(title, [(label, value), ...]), ...] as rich text for the
        File Information panel - one label/value table with a heading row per
        section, so the value column lines up across sections"""
        parts = ["<table width='100%' cellspacing='0' cellpadding='1'>"]
        for index, (title, rows) in enumerate(sections):
            padding = 0 if index == 0 else 10
            parts.append(f"<tr><td colspan='2' style='font-weight: bold; color: {COLORS['info']}; "
                         f"padding-top: {padding}px;'>{escape(title)}</td></tr>")
            for label, value in rows:
                # Zero-width spaces let long paths wrap at their separators
                value_html = escape(str(value)).replace('\\', '\\​').replace('/', '/​')
                parts.append(
                    f"<tr><td style='color: {COLORS['text_disabled']}; padding-right: 8px; "
                    f"white-space: nowrap;' valign='top'>{escape(label)}</td>"
                    f"<td style='color: {COLORS['text']};'>{value_html}</td></tr>")
        parts.append("</table>")
        return "".join(parts)

    def _wcs_for_display(self, wcs_header, image_height):
        """A plate solve's WCS with y running up the image as displayed.

        ASTAP solves a FITS file with its rows in stored order (first row =
        y 1, ignoring ROWORDER), and this viewer shows the first stored row at
        the top, so FITS solutions are flipped. Solves of PNG/JPG (and XISF,
        converted bottom-up before solving) already match the display.
        """
        if self.file_path and Path(self.file_path).suffix.lower() in FITS_EXTENSIONS:
            from AnnotationOverlay import flip_wcs_rows
            return flip_wcs_rows(wcs_header, image_height)
        return wcs_header

    def _cached_wcs_header(self):
        """WCS keywords from this image's plate solve - the loaded result, or
        the cached .wcs file next to the image - or None"""
        if self.plate_solve_result and self.plate_solve_result.success and self.plate_solve_result.wcs_header:
            return self.plate_solve_result.wcs_header
        wcs_file = Path(self.file_path).with_suffix('.wcs')
        return read_wcs_header(self.file_path) if wcs_file.exists() else None

    def _refresh_file_info(self):
        """Re-render the File Information panel (e.g. with the new theme's colors)"""
        if self.file_info_panel.isVisible():
            self._load_file_information()

    def _get_exif_info(self):
        """Extract basic EXIF information from the image file"""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            # Open image with PIL to read EXIF
            with Image.open(self.file_path) as img:
                exif_data = img.getexif()

                if not exif_data:
                    return None

                # Extract useful EXIF information
                exif_info = {}

                # Common EXIF tags we want to show
                useful_tags = {
                    'Make': 'Camera Make',
                    'Model': 'Camera Model',
                    'DateTime': 'Date Taken',
                    'ExposureTime': 'Exposure Time',
                    'FNumber': 'F-Number',
                    'ISO': 'ISO Speed',
                    'FocalLength': 'Focal Length',
                    'Flash': 'Flash',
                    'WhiteBalance': 'White Balance',
                    'ExposureProgram': 'Exposure Program',
                    'MeteringMode': 'Metering Mode'
                }

                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name in useful_tags:
                        # Format specific values
                        if tag_name == 'ExposureTime' and isinstance(value, tuple):
                            value = f"{value[0]}/{value[1]} sec"
                        elif tag_name == 'FNumber' and isinstance(value, tuple):
                            value = f"f/{value[0]/value[1]:.1f}"
                        elif tag_name == 'FocalLength' and isinstance(value, tuple):
                            value = f"{value[0]/value[1]:.1f}mm"

                        exif_info[useful_tags[tag_name]] = str(value)

                return exif_info if exif_info else None

        except ImportError:
            # PIL not available
            return None
        except Exception as e:
            # Any other error reading EXIF
            return None

    def _update_annotation_controls(self):
        """As in the gallery: Layers and the solve button show while the
        annotations do, and the button reads Re-Plate Solve once solved."""
        if not hasattr(self, 'annotations_check'):
            return
        on = self.annotations_enabled
        solved = self._unsaved_annotations is not None or bool(annotation_store().is_solved(self.file_path))
        self.layers_button.setVisible(on)
        self.solve_button.setVisible(on)
        self.solve_button.setEnabled(not self._solving)
        self.solve_button.setText("Re-Plate Solve" if solved else "Plate Solve")
        self.solve_button.setToolTip("Solve the image again, replacing its saved solution" if solved
                                     else "Find where the image is in the sky, so its objects can be marked")

    def _start_plate_solve(self):
        """Plate solve the image in the background - progress in the status bar."""
        if not self.file_path or self._solving:
            return
        from PlateSolver import PlateSolverWorker
        hints = None
        if self.dso_ra is not None and self.dso_dec is not None:
            hints = {'ra': self.dso_ra, 'dec': self.dso_dec, 'radius': 15}  # Near the image's object
        # Re-solving replaces the image's saved solution
        worker = PlateSolverWorker(self.file_path, hints, fresh=read_wcs_header(self.file_path) is not None)
        worker.progress.connect(lambda message: self._show_solving(f"Plate solving - {message}"))
        worker.solve_finished.connect(self._on_plate_solve_finished)
        self.plate_solve_worker = worker
        self._solving, self._stopping = True, False
        self._solve_message_timer.stop()
        self._show_solving("Plate solving...")
        worker.start()

    def _show_solving(self, message):
        """Plate solving's progress in the status bar, with the busy bar and Stop."""
        self._solve_message = message if not self._stopping else "Stopping plate solving..."
        self.status_bar.setToolTip(self._solve_message)
        self.solve_busy.setVisible(self._solving)
        self.solve_stop.setVisible(self._solving)
        self.solve_stop.setEnabled(not self._stopping)
        self._update_annotation_controls()
        self._update_status()

    def _stop_plate_solve(self):
        """Stop once the solver that's running finishes (they can't be interrupted)."""
        if self._solving and self.plate_solve_worker is not None:
            self._stopping = True
            self.plate_solve_worker.cancel()
            self._show_solving("")

    def _on_plate_solve_finished(self, result):
        """The plate solve's result in the status bar for a moment, and its annotations."""
        self._solving = False
        self.plate_solve_result = result
        if result.success:
            scale = f", {result.pixel_scale:.2f}\"/px" if result.pixel_scale else ""
            solver = "the built-in solver" if result.solver_used == "Built-in" else result.solver_used
            message, details = f"Plate solved with {solver}{scale} ✓", None
            self._use_new_solution(result)
        elif self._stopping:
            message, details = "Plate solving stopped", None
        else:
            details = (result.error_message or "No solution").strip()
            message = f"Couldn't plate solve - {details.splitlines()[0]}"
        self._stopping = False
        self._show_solving(message)
        self.status_bar.setToolTip(details or message)  # Every solver's reason when it failed
        self._solve_message_timer.start(6000 if result.success else 12000)

    def _clear_solve_message(self):
        self._solve_message = None
        self.status_bar.setToolTip("")
        self._update_status()

    _annotation_layers = staticmethod(saved_annotation_layers)

    def _annotations(self):
        """This image's AnnotationData - None if it isn't plate solved, or while
        it loads in the background (_on_annotations_ready repaints then)."""
        if not self.file_path:
            return None
        if self._unsaved_annotations is not None:
            return self._unsaved_annotations
        return annotation_store().get(self.file_path, want_stars=self._annotation_layers()["stars"])

    def _on_annotations_ready(self, path):
        if path == self.file_path:
            self._update_annotation_controls()
            self._update_status()
            if self.annotations_enabled:
                self._update_zoom()

    def _annotation_status(self):
        """Status bar text about the annotations, or None when they're off."""
        if not (self.annotations_enabled and self.file_path):
            return None
        data = self._annotations()
        if data is None:
            if not annotation_store().known(self.file_path):
                return "Loading annotations..."
            return "Not plate solved - use Plate Solve to find its objects"
        count = len(data.dsos)
        text = f"Annotations: {count} DSO{'s' if count != 1 else ''}"
        if self._annotation_layers()["stars"] and data is not self._unsaved_annotations:
            stars = data.stars
            text += (", looking up stars..." if stars is None
                     else f", {len(stars)} star{'s' if len(stars) != 1 else ''}")
        return text

    def _use_new_solution(self, result):
        """Show the annotations of a plate solve that just finished"""
        self._unsaved_annotations = None
        annotation_store().forget(self.file_path)
        if read_wcs_header(self.file_path) is None and result.wcs_header:
            # It couldn't be saved next to the image (e.g. a read-only folder) - keep it here
            try:
                width, height = image_pixel_size(self.file_path)
                self._unsaved_annotations = AnnotationData.from_header(
                    result.wcs_header, width, height,
                    fits_rows=Path(self.file_path).suffix.lower() in FITS_EXTENSIONS)
            except Exception as e:
                logger.warning(f"Could not use the plate solve for annotations: {e}")
        self._update_zoom()

    def _update_annotation_settings(self):
        """Save the layer choices and redraw"""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        for key, setting in ANNOTATION_LAYER_SETTINGS.items():
            settings.setValue(setting, self.layer_actions[key].isChecked())
        self._update_zoom()

    def _toggle_annotations(self, enabled):
        """Turn the annotations overlay on or off"""
        self.annotations_enabled = enabled
        QSettings("CosmosCollection", "CosmosCollection").setValue("annotation_enabled", enabled)
        self._update_annotation_controls()
        self._update_zoom()

    def closeEvent(self, event):
        """Save window position when closing"""
        # Cancel any running plate solve
        if self.plate_solve_worker and self.plate_solve_worker.isRunning():
            self.plate_solve_worker.cancel()
            self.plate_solve_worker.wait(1000)

        # Don't block the close on a slow load - drop its result and keep the
        # thread object alive until it finishes
        if self.image_load_worker and self.image_load_worker.isRunning():
            worker = self.image_load_worker
            worker.image_loaded.disconnect()
            worker.load_failed.disconnect()
            _orphaned_workers.add(worker)
            worker.finished.connect(lambda: _orphaned_workers.discard(worker))

        # Let a save carry on in the background after the window closes, so the
        # file still gets written
        if self.save_worker and self.save_worker.isRunning():
            worker = self.save_worker
            worker.image_saved.disconnect()
            _orphaned_workers.add(worker)
            worker.finished.connect(lambda: _orphaned_workers.discard(worker))

        # Let a background change finish - destroying a running QThread crashes
        if self.background_worker and self.background_worker.isRunning():
            self.background_worker.background_set.disconnect()
            self.background_worker.wait()

        WindowPositionManager.save_window_position(self, "ImageViewer")
        event.accept()
