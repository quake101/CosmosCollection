#!/usr/bin/env python3
"""
Image Details for Cosmos Collection
The capture details the Image Viewer shows on an image and saves with it -
object, dates, integration (in total and by filter), filters, telescope,
camera, optics, image scale, mount, location and camera settings.

They come from the image's library entry and the session it's linked to,
with the file's header filling what those don't record. Each line can be
edited for an image (kept in userimagedetails, by file path, so subs that
aren't in the library keep theirs too), and the look - which lines, over
the image or on a banner above or below it, size, font, colors - is one
style for every image (QSettings, with named presets).

Everything is painted in image pixels, so what the viewer shows over the
image is what's saved.
"""

import copy
import json
import logging
import os
import sqlite3
import statistics
from collections import Counter
from datetime import date, datetime

from PySide6.QtCore import Qt, QObject, QRectF, QRunnable, QSettings, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QWidget, QLabel, QPushButton,
                               QComboBox, QCheckBox, QLineEdit, QListWidget, QListWidgetItem, QAbstractItemView,
                               QToolButton, QSlider, QFontComboBox, QColorDialog, QInputDialog, QMessageBox)

from Theme import COLORS, contrast_text, font_size, themed_style
from WindowPositionManager import WindowPositionMixin

logger = logging.getLogger(__name__)


# (key, name in the dialog, prefix on the image - None for none, on by default)
FIELDS = [
    ("title", "Object", None, True),
    ("dates", "Dates", "Date", True),
    ("integration", "Integration", "Integration", True),
    ("integration_filters", "Integration by filter", None, False),
    ("filters", "Filters", "Filters", True),
    ("telescope", "Telescope", "Telescope", True),
    ("camera", "Camera", "Camera", True),
    ("optics", "Optics", "Optics", False),
    ("scale", "Image scale", "Scale", False),
    ("mount", "Mount", "Mount", False),
    ("location", "Location", "Location", False),
    ("camera_settings", "Camera settings", None, False),
    ("custom", "Custom text", None, False),
]
FIELD_NAMES = {key: name for key, name, _prefix, _on in FIELDS}
FIELD_PREFIXES = {key: prefix for key, _name, prefix, _on in FIELDS}

LAYOUTS = [("overlay", "Over the image"), ("banner_below", "Banner below"), ("banner_above", "Banner above")]
POSITIONS = ["Bottom Left", "Bottom Center", "Bottom Right", "Top Left", "Top Center", "Top Right",
             "Center Left", "Center", "Center Right"]
BANNER_ALIGNMENTS = ["Left", "Center", "Right"]
# Body text height, as a share of the image's short side
SIZES = [("small", "Small", 0.016), ("medium", "Medium", 0.021), ("large", "Large", 0.028)]
SIZE_SCALES = {key: scale for key, _name, scale in SIZES}
TITLE_SCALE = 1.45     # the title line, with title emphasis, to the body text
SEPARATOR = " · "

STYLE_SETTING = "image_viewer/details"
PRESETS_SETTING = "image_viewer/details_presets"
DIALOG_SETTING = "image_viewer/details_dialog_open"

DEFAULT_STYLE = {
    "enabled": False,
    "layout": "overlay",
    "position": "Bottom Left",
    "banner_align": "Center",
    "size": "medium",
    "fields": [[key, on] for key, _name, _prefix, on in FIELDS],  # in the order shown
    "labels": True,              # "Telescope: ..." rather than just the name
    "title_emphasis": True,      # the object's name larger and bold
    "font_family": "",           # the app's font
    "text_color": "#ffffff",
    "background_color": "#000000",
    "background_opacity": 55,    # percent, behind the text over the image (a banner is solid)
    "banner_style": "lines",     # "lines": one per line; "inline": run together after the title
    "custom_text": "",           # the custom line - the same on every image
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def normalize_style(style):
    """A complete style from a saved one: defaults for what's missing or
    unusable, the fields in its order with any new ones added."""
    result = copy.deepcopy(DEFAULT_STYLE)
    if not isinstance(style, dict):
        return result
    for key, default in DEFAULT_STYLE.items():
        value = style.get(key)
        if key == "fields" or value is None or type(value) is not type(default):
            continue
        result[key] = value
    if result["layout"] not in dict(LAYOUTS):
        result["layout"] = DEFAULT_STYLE["layout"]
    if result["position"] not in POSITIONS:
        result["position"] = DEFAULT_STYLE["position"]
    if result["banner_align"] not in BANNER_ALIGNMENTS:
        result["banner_align"] = DEFAULT_STYLE["banner_align"]
    if result["size"] not in SIZE_SCALES:
        result["size"] = DEFAULT_STYLE["size"]
    if result["banner_style"] not in ("lines", "inline"):
        result["banner_style"] = DEFAULT_STYLE["banner_style"]
    result["background_opacity"] = max(0, min(100, int(result["background_opacity"])))
    for key in ("text_color", "background_color"):
        if not QColor(result[key]).isValid():
            result[key] = DEFAULT_STYLE[key]

    fields, seen = [], set()
    for entry in style.get("fields") or []:
        if (isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] in FIELD_NAMES
                and entry[0] not in seen):
            fields.append([entry[0], bool(entry[1])])
            seen.add(entry[0])
    defaults = dict((key, on) for key, on in DEFAULT_STYLE["fields"])
    fields.extend([key, defaults[key]] for key, _name, _prefix, _on in FIELDS if key not in seen)
    result["fields"] = fields
    return result


def _json_value(text):
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


def load_style():
    settings = QSettings("CosmosCollection", "CosmosCollection")
    return normalize_style(_json_value(settings.value(STYLE_SETTING, "", type=str)))


def save_style(style):
    QSettings("CosmosCollection", "CosmosCollection").setValue(STYLE_SETTING, json.dumps(style))


def load_presets():
    """{name: style} - each without "enabled" (a preset is a look, not on/off)."""
    value = _json_value(QSettings("CosmosCollection", "CosmosCollection").value(PRESETS_SETTING, "", type=str))
    if not isinstance(value, dict):
        return {}
    return {str(name): normalize_style(style) for name, style in value.items() if isinstance(style, dict)}


def save_presets(presets):
    QSettings("CosmosCollection", "CosmosCollection").setValue(PRESETS_SETTING, json.dumps(presets))


def is_banner(style):
    return style["layout"] in ("banner_below", "banner_above")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _num(value, decimals):
    """A number with up to `decimals` places, without trailing zeros."""
    return f"{value:.{decimals}f}".rstrip('0').rstrip('.')


def format_total(seconds):
    """'12h 30m', '45m', '30s' - an integration time."""
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{_num(seconds, 1)}s"
    hours, minutes = divmod(round(seconds / 60.0), 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    return f"{hours}h" if hours else f"{minutes}m"


def format_exposure(seconds):
    """'300s', '0.5s' - one sub's exposure."""
    return f"{_num(float(seconds or 0), 2)}s"


def _subs(count):
    return f"{count} sub{'s' if count != 1 else ''}"


def format_dates(dates):
    """'Oct 4, 2025', 'Oct 4 – 12, 2025 (5 nights)', 'Dec 30, 2024 – Jan 2, 2025 (3 nights)'."""
    dates = sorted(set(dates))
    if not dates:
        return ""
    first, last = dates[0], dates[-1]
    if first == last:
        return f"{first:%b} {first.day}, {first.year}"
    if first.year != last.year:
        text = f"{first:%b} {first.day}, {first.year} – {last:%b} {last.day}, {last.year}"
    elif first.month != last.month:
        text = f"{first:%b} {first.day} – {last:%b} {last.day}, {last.year}"
    else:
        text = f"{first:%b} {first.day} – {last.day}, {last.year}"
    return f"{text} ({len(dates)} nights)" if len(dates) > 1 else text


def _parse_date(value):
    """A date from '2025-10-04', '2025-10-04T21:30:00' or a datetime - None if it isn't one."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for length, pattern in ((19, "%Y-%m-%dT%H:%M:%S"), (19, "%Y-%m-%d %H:%M:%S"), (10, "%Y-%m-%d"),
                            (10, "%Y/%m/%d")):
        try:
            return datetime.strptime(text[:length], pattern).date()
        except ValueError:
            continue
    return None


def format_scale(scale_arcsec, image_size):
    """'1.93"/px, 2.21° x 1.47°' - the field of view when the image size is known."""
    from ImageViewer import format_fov
    text = f"{scale_arcsec:.2f}\"/px"
    if image_size and image_size[0] and image_size[1]:
        text += ", " + format_fov(scale_arcsec, *image_size)
    return text


def format_optics(focal_length, aperture=None, focal_ratio=None):
    """'400 mm f/5'."""
    if not focal_length:
        return ""
    text = f"{_num(focal_length, 1)} mm"
    if not focal_ratio and aperture:
        focal_ratio = focal_length / aperture
    if focal_ratio:
        text += f" f/{_num(focal_ratio, 1)}"
    return text


def format_camera_settings(gain=None, offset=None, temperature=None, binning=None):
    """'Gain 100, Offset 50, -10 °C, Bin 2x2' - what's known of them."""
    parts = []
    if gain is not None:
        parts.append(f"Gain {_num(gain, 2)}")
    if offset is not None:
        parts.append(f"Offset {_num(offset, 2)}")
    if temperature is not None:
        parts.append(f"{_num(round(temperature, 1), 1)} °C")
    if binning and binning > 1:
        parts.append(f"Bin {binning:.0f}x{binning:.0f}")
    return ", ".join(parts)


def format_filter_breakdown(groups):
    """'Ha 40×300s (3h 20m) · OIII 30×300s + 12×600s (4h 30m)' from
    [(filter name, exposure seconds, subs, seconds)], in filter order - each
    filter once, with its exposures."""
    from SessionObservations import normalize_filter
    filters = {}  # filter key -> [name, [(subs, exposure)], seconds]
    for name, exposure, subs, seconds in groups:
        if subs <= 0:
            continue
        entry = filters.setdefault(normalize_filter(name), [name or "No filter", [], 0.0])
        entry[1].append((subs, exposure))
        entry[2] += seconds or 0
    parts = []
    for name, exposures, seconds in filters.values():
        counts = " + ".join(f"{subs}×{format_exposure(exposure)}" for subs, exposure in exposures)
        parts.append(f"{name} {counts} ({format_total(seconds)})")
    return SEPARATOR.join(parts)


# ---------------------------------------------------------------------------
# Gathering (worker thread - no Qt widgets)
# ---------------------------------------------------------------------------

def path_key(path):
    """How a file path is matched: case and separators as Windows treats them."""
    return os.path.normcase(os.path.normpath(path or ""))


def _like_escape(text):
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _library_rows(conn, image_path):
    """The library entries of the file (one per object it shows), session-linked first."""
    import EquipmentMapper
    name = os.path.basename(image_path)
    if not name:
        return []
    rows = conn.execute(f"""
        SELECT ui.image_path, ui.dsodetailid, ui.session_id, ui.date_taken, ui.integration_time,
               {EquipmentMapper.select_sql('ui')}
        FROM userimages ui
        WHERE ui.image_path LIKE ? ESCAPE '\\'
        ORDER BY ui.session_id IS NULL, ui.id
    """, ("%" + _like_escape(name),)).fetchall()
    key = path_key(image_path)
    result = []
    for row in rows:
        if path_key(row[0]) != key:
            continue
        entry = {"dsodetailid": row[1], "session_id": row[2], "date_taken": row[3],
                 "integration_time": row[4]}
        entry.update(EquipmentMapper.image_equipment(tuple(row[5:11])))
        result.append(entry)
    return result


def session_id_for(conn, image_path):
    """The session the image's library entry is linked to, or None."""
    return next((row["session_id"] for row in _library_rows(conn, image_path) if row["session_id"]), None)


def _object_names(conn, dsodetailids):
    """'M 81 (Bode's Galaxy) & M 82 (Cigar Galaxy)' for the objects."""
    from DSONames import load_common_names, preferred_designation, with_common_name
    ids = list(dict.fromkeys(i for i in dsodetailids if i))
    if not ids:
        return ""
    common = load_common_names(conn, ids)
    names = []
    for dso in ids:
        designations = [(c, d) for c, d in conn.execute(
            "SELECT catalogue, designation FROM cataloguenr WHERE dsodetailid = ? AND catalogue != 'NAME'",
            (dso,)).fetchall() if c and d]
        name = preferred_designation(designations) or common.get(str(dso)) or ""
        names.append(with_common_name(name, common.get(str(dso))))
    return " & ".join(name for name in names if name)


def _mode(values):
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else None


def _session_details(conn, session_id, details, numbers):
    """Fill details from the session: its nights, integration by filter, gear,
    location, mount and camera settings."""
    import EquipmentMapper
    import MountDetection
    import SessionObservations as so
    row = conn.execute(f"""
        SELECT s.dso_name, s.location_name, s.filters_used, s.sub_count, s.integration_seconds,
               s.earliest_sub_date, s.latest_sub_date, s.telescope_id, s.camera_id,
               {EquipmentMapper.session_camera_sql('s')}, tel.name, tel.focal_length, tel.aperture, tel.mount_type,
               s.mount_override, {MountDetection.detected_sql('s')}
        FROM usersessions s LEFT JOIN usertelescopes tel ON tel.id = s.telescope_id
        WHERE s.id = ?
    """, (session_id,)).fetchone()
    if row is None:
        return
    (dso_name, location, filters_used, sub_count, integration_seconds, earliest, latest, telescope_id,
     camera_id, camera, telescope, focal_length, aperture, telescope_mount, mount_override, detected) = row
    numbers.setdefault("session_name", dso_name)
    numbers.setdefault("telescope_id", telescope_id)
    numbers.setdefault("camera_id", camera_id)

    # Each night's counted subs (the files or the hand-logged rows), plus Light
    # files that couldn't be put on a night
    observations = conn.execute("SELECT id, night_date FROM usersessionobservations WHERE session_id = ?",
                                (session_id,)).fetchall()
    breakdowns = so.load_breakdowns(conn, [obs_id for obs_id, _night in observations])
    groups = {}  # (filter key, exposure bucket) -> [name, exposure, subs, seconds]
    nights = []
    for obs_id, night in observations:
        rows = breakdowns.get(obs_id) or []
        if any(r["counted_subs"] > 0 for r in rows) and _parse_date(night):
            nights.append(_parse_date(night))
        for r in rows:
            group = groups.setdefault((so.normalize_filter(r["filter_name"]), so.exposure_bucket(r["exposure_seconds"])),
                                      [r["filter_name"] or "", r["exposure_seconds"] or 0, 0, 0.0])
            group[2] += r["counted_subs"]
            group[3] += r["counted_seconds"] or 0
    for filter_name, exposure, count, seconds in conn.execute("""
            SELECT filter_name, exptime_seconds, COUNT(*), COALESCE(SUM(exptime_seconds), 0) FROM usersessionfiles
            WHERE session_id = ? AND observation_id IS NULL AND frame_type = 'Light'
            GROUP BY filter_name, exptime_seconds""", (session_id,)).fetchall():
        group = groups.setdefault((so.normalize_filter(filter_name), so.exposure_bucket(exposure)),
                                  [filter_name or "", exposure or 0, 0, 0.0])
        group[2] += count
        group[3] += seconds
    ordered = sorted(groups.values(), key=lambda g: (so.normalize_filter(g[0]), g[1]))

    if not nights:
        nights = [d for d in (_parse_date(earliest), _parse_date(latest)) if d]
    if nights:
        details["dates"] = format_dates(nights)
    if integration_seconds:
        details["integration"] = f"{format_total(integration_seconds)} ({_subs(sub_count or 0)})" \
            if sub_count else format_total(integration_seconds)
    breakdown = format_filter_breakdown(ordered)
    if breakdown:
        details["integration_filters"] = breakdown
    names = {}
    for name, _exposure, subs, _seconds in ordered:
        if subs > 0 and name:
            names.setdefault(so.normalize_filter(name), name)
    filters = sorted(names.values(), key=str.lower)
    if filters:
        details["filters"] = ", ".join(filters)
    elif filters_used:
        details["filters"] = filters_used
    if telescope:
        details.setdefault("telescope", telescope)
    if camera:
        details.setdefault("camera", camera)
    if focal_length:
        numbers.setdefault("focal_length", focal_length)
        numbers.setdefault("aperture", aperture)
    if location:
        details["location"] = location
    mount, _how = MountDetection.summarize(detected, mount_override, telescope_mount)
    if mount:
        details["mount"] = mount

    lights = conn.execute("""
        SELECT gain, offset_value, ccd_temp, xbinning FROM usersessionfiles
        WHERE session_id = ? AND frame_type = 'Light'""", (session_id,)).fetchall()
    if lights:
        temperatures = [r[2] for r in lights if r[2] is not None]
        settings = format_camera_settings(_mode(r[0] for r in lights), _mode(r[1] for r in lights),
                                          statistics.median(temperatures) if temperatures else None,
                                          _mode(r[3] for r in lights))
        if settings:
            details["camera_settings"] = settings


def _equipment_numbers(conn, numbers):
    """Focal length and aperture of the telescope, pixel size of the camera."""
    if numbers.get("telescope_id") and not numbers.get("focal_length"):
        row = conn.execute("SELECT focal_length, aperture FROM usertelescopes WHERE id = ?",
                           (numbers["telescope_id"],)).fetchone()
        if row and row[0]:
            numbers["focal_length"], numbers["aperture"] = row[0], row[1]
    if numbers.get("camera_id"):
        try:
            row = conn.execute("SELECT pixel_size FROM userequipment WHERE id = ?",
                               (numbers["camera_id"],)).fetchone()
        except sqlite3.OperationalError:
            row = None  # A database from before pixel sizes
        if row and row[0]:
            numbers["pixel_size"] = row[0]


def _header_details(header, details, numbers):
    """Fill what's still missing from the file's FITS/XISF header."""
    from ImageViewer import _first_value, _parse_iso_datetime, _to_float
    kw = header.keywords

    def fill(key, value):
        if value and not details.get(key):
            details[key] = str(value).strip()

    fill("title", _first_value(kw, 'OBJECT', 'OBJNAME'))
    when = _parse_iso_datetime(kw.get('DATE-LOC')) or _parse_iso_datetime(kw.get('DATE-OBS'))
    if when is not None:
        fill("dates", format_dates([when.date()]))

    frames = _to_float(_first_value(kw, 'NCOMBINE', 'STACKCNT'))
    live_time = _to_float(kw.get('LIVETIME'))
    exposure = _to_float(_first_value(kw, 'EXPTIME', 'EXPOSURE'))
    if live_time:
        fill("integration", format_total(live_time) + (f" ({_subs(int(frames))})" if frames and frames > 1 else ""))
    elif frames and frames > 1:
        fill("integration", _subs(int(frames)))
    elif exposure:
        fill("integration", format_exposure(exposure))
    fill("filters", kw.get('FILTER'))
    fill("telescope", kw.get('TELESCOP'))
    fill("camera", kw.get('INSTRUME'))

    focal_length = _to_float(kw.get('FOCALLEN'))
    if focal_length and not details.get("optics"):
        fill("optics", format_optics(focal_length, _to_float(kw.get('APTDIA')), _to_float(kw.get('FOCRATIO'))))
    scale = _to_float(_first_value(kw, 'PIXSCALE', 'SCALE'))
    pixel = _to_float(kw.get('XPIXSZ'))  # Includes the binning
    if not scale and pixel and focal_length:
        scale = 206.265 * pixel / focal_length
    if scale:
        numbers.setdefault("header_scale", scale)

    eqmode = _to_float(kw.get('EQMODE'))
    if eqmode is not None:
        fill("mount", "Equatorial" if eqmode == 1 else "Alt-Az")
    elif str(kw.get('PIERSIDE') or "").strip().lower() in ("east", "west"):
        fill("mount", "Equatorial")

    temperature = _to_float(_first_value(kw, 'CCD-TEMP', 'CCD_TEMP', 'TEMPERAT'))
    if temperature is None:
        temperature = _to_float(_first_value(kw, 'SET-TEMP', 'SET_TEMP'))
    fill("camera_settings", format_camera_settings(_to_float(kw.get('GAIN')), _to_float(kw.get('OFFSET')),
                                                   temperature, _to_float(kw.get('XBINNING'))))


def _solved_scale(wcs_header):
    """Image scale in arcsec/px from a plate solution's keywords, or None."""
    if not wcs_header:
        return None
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        from ImageViewer import WCS_KEYWORD_PATTERN, _is_plate_solved
        keywords = {key: value for key, value in wcs_header.items() if WCS_KEYWORD_PATTERN.match(str(key))}
        if not _is_plate_solved(keywords):
            return None
        return float(proj_plane_pixel_scales(WCS(keywords, naxis=2)).mean()) * 3600.0
    except Exception as e:
        logger.debug(f"Could not read the plate solution's scale: {e}")
        return None


def gather(conn, image_path, header=None, image_size=None, wcs_header=None, fallback_title=None):
    """{field key: text} for an image - '' for what isn't known. conn: an open
    connection (any thread's own); header: the file's HeaderData, or None;
    wcs_header: its plate solution's keywords, or None."""
    details = {key: "" for key in FIELD_NAMES}
    numbers = {}
    try:
        rows = _library_rows(conn, image_path)
    except sqlite3.Error as e:
        logger.warning(f"Couldn't look up the image in the library: {e}")
        rows = []
    if rows:
        first = rows[0]
        try:
            details["title"] = _object_names(conn, [r["dsodetailid"] for r in rows])
        except sqlite3.Error as e:
            logger.debug(f"Couldn't name the image's objects: {e}")
        details["telescope"] = first["telescope"]
        details["camera"] = first["camera"]
        numbers["telescope_id"] = first["telescope_id"]
        numbers["camera_id"] = first["camera_id"]
        session_id = next((r["session_id"] for r in rows if r["session_id"]), None)
        if session_id:
            try:
                _session_details(conn, session_id, details, numbers)
            except sqlite3.Error as e:
                logger.warning(f"Couldn't read the image's session: {e}")
        if not details["dates"]:
            taken = _parse_date(first["date_taken"])
            details["dates"] = format_dates([taken]) if taken else (first["date_taken"] or "").strip()
        if not details["integration"]:
            details["integration"] = (first["integration_time"] or "").strip()
    try:
        _equipment_numbers(conn, numbers)
    except sqlite3.Error as e:
        logger.debug(f"Couldn't read the image's equipment: {e}")

    # The session's name when it adds the common name the catalog doesn't know
    # ("Sh2 129 (Squid Nebula)")
    session_name = (numbers.get("session_name") or "").strip()
    if session_name and details["title"] and "(" not in details["title"] and "(" in session_name:
        from DSONames import base_name
        if base_name(session_name).lower() == details["title"].lower():
            details["title"] = session_name
    if header is not None:
        _header_details(header, details, numbers)
    if not details["title"]:
        details["title"] = session_name or fallback_title or ""
    if not details["optics"] and numbers.get("focal_length"):
        details["optics"] = format_optics(numbers["focal_length"], numbers.get("aperture"))

    # The image scale as measured by a plate solve, else as recorded, else from the equipment
    scale = _solved_scale(wcs_header) or numbers.get("header_scale")
    if not scale and numbers.get("pixel_size") and numbers.get("focal_length"):
        scale = 206.265 * numbers["pixel_size"] / numbers["focal_length"]
    if scale:
        details["scale"] = format_scale(scale, image_size)
    return details


# ---------------------------------------------------------------------------
# Per-image edits
# ---------------------------------------------------------------------------

def ensure_schema(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS userimagedetails ("
                 "image_path TEXT PRIMARY KEY COLLATE NOCASE, fields_json TEXT)")


def load_edits(conn, image_path):
    """{field key: text} the user typed for this image."""
    try:
        row = conn.execute("SELECT fields_json FROM userimagedetails WHERE image_path = ?",
                           (path_key(image_path),)).fetchone()
    except sqlite3.OperationalError:
        return {}  # None saved yet
    value = _json_value(row[0]) if row else None
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if k in FIELD_NAMES and k != "custom"}


def save_edits(conn, image_path, edits):
    """Keep the image's edits (none: forget them). Commits."""
    ensure_schema(conn)
    if edits:
        conn.execute("INSERT OR REPLACE INTO userimagedetails (image_path, fields_json) VALUES (?, ?)",
                     (path_key(image_path), json.dumps(edits)))
    else:
        conn.execute("DELETE FROM userimagedetails WHERE image_path = ?", (path_key(image_path),))
    conn.commit()


class DetailsSignals(QObject):
    ready = Signal(str, object)  # image path, {"auto", "edits", "info"} (or the exception)


class DetailsTask(QRunnable):
    """Gathers an image's details in the background, with its own database
    connection: the library and session, then the file's header (read here
    unless file_info - read_file_info()'s result - is given)."""

    def __init__(self, image_path, signals, image_size=None, file_info=None, wcs_header=None,
                 fallback_title=None):
        super().__init__()
        self.image_path, self.signals = image_path, signals
        self.image_size, self.file_info = image_size, file_info
        self.wcs_header, self.fallback_title = wcs_header, fallback_title

    def run(self):
        try:
            from ImageViewer import read_file_info
            from ResourceManager import ResourceManager, attach_update_catalogs
            from AnnotationOverlay import read_wcs_header
            info = self.file_info
            if info is None:
                try:
                    info = read_file_info(self.image_path)
                except Exception as e:
                    logger.debug(f"Couldn't read {self.image_path} for its details: {e}")
            header = info.get("header") if isinstance(info, dict) else None
            wcs_header = self.wcs_header or read_wcs_header(self.image_path)
            conn = sqlite3.connect(str(ResourceManager.get_database_path()))
            try:
                attach_update_catalogs(conn)
                auto = gather(conn, self.image_path, header, self.image_size, wcs_header, self.fallback_title)
                edits = load_edits(conn, self.image_path)
            finally:
                conn.close()
            result = {"auto": auto, "edits": edits, "info": info}
        except Exception as e:
            logger.error(f"Couldn't gather the details of {self.image_path}: {e}", exc_info=True)
            result = e
        try:
            self.signals.ready.emit(self.image_path, result)
        except RuntimeError:
            pass  # The viewer closed meanwhile


def detail_lines(auto, edits, style):
    """[(key, text)] to draw: the style's lines that are on, in its order,
    with what the user typed in place of what was found, empty ones left out."""
    lines = []
    for key, on in style["fields"]:
        if not on:
            continue
        if key == "custom":
            text = style.get("custom_text", "")
        else:
            text = edits.get(key, auto.get(key, "")) if edits is not None else auto.get(key, "")
        text = " ".join(str(text or "").split())
        if not text:
            continue
        prefix = FIELD_PREFIXES.get(key)
        if style["labels"] and prefix:
            text = f"{prefix}: {text}"
        lines.append((key, text))
    return lines


def description_text(auto, edits):
    """The image's capture details as plain text, a line each with its name -
    every one that's known, whatever the image shows (e.g. for an upload's
    description)."""
    auto, edits = auto or {}, edits or {}
    lines = []
    for key, name, _prefix, _on in FIELDS:
        if key == "custom":
            continue
        text = " ".join(str(edits.get(key, auto.get(key, "")) or "").split())
        if text:
            lines.append(text if key == "title" else f"{name}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Painting (image pixels)
# ---------------------------------------------------------------------------

def body_px(short_side, style):
    return max(9.0, short_side * SIZE_SCALES[style["size"]])


def _font(style, px, bold=False):
    font = QFont()
    if style["font_family"]:
        font.setFamily(style["font_family"])
    font.setPixelSize(max(5, round(px)))
    font.setBold(bold)
    return font


def _wrap(text, metrics, width):
    """Lines of text no wider than width: broken between ' · ' parts first,
    then between words, and mid-word only for a word wider than the line."""
    def fits(candidate):
        return metrics.horizontalAdvance(candidate) <= width

    def words(part):
        result, line = [], ""
        for word in part.split(" "):
            candidate = f"{line} {word}" if line else word
            if fits(candidate) or not line:
                line = candidate
            else:
                result.append(line)
                line = word
            while not fits(line) and len(line) > 1:
                cut = len(line)
                while cut > 1 and not fits(line[:cut]):
                    cut -= 1
                result.append(line[:cut])
                line = line[cut:]
        return result + ([line] if line else [])

    if fits(text):
        return [text]
    lines, line = [], ""
    for part in text.split(SEPARATOR):
        candidate = f"{line}{SEPARATOR}{part}" if line else part
        if fits(candidate):
            line = candidate
            continue
        if line:
            lines.append(line)
        pieces = words(part)
        lines.extend(pieces[:-1])
        line = pieces[-1] if pieces else ""
    if line:
        lines.append(line)
    return lines


def _layout(lines, style, size, width, inline=False):
    """[(font, metrics, text)] - each line wrapped to width, the title larger
    with title emphasis. inline runs every line after the title together."""
    if inline:
        title = [(key, text) for key, text in lines[:1] if key == "title"]
        rest = lines[len(title):]
        lines = title + ([("rest", SEPARATOR.join(text for _key, text in rest))] if rest else [])
    body = _font(style, size)
    body_metrics = QFontMetricsF(body)
    title_font = _font(style, size * TITLE_SCALE, bold=True) if style["title_emphasis"] else body
    title_metrics = QFontMetricsF(title_font)
    result = []
    for key, text in lines:
        font, metrics = (title_font, title_metrics) if key == "title" else (body, body_metrics)
        for line in _wrap(text, metrics, max(1.0, width)):
            result.append((font, metrics, line))
    return result


def _block_size(laid_out, size):
    width = max((metrics.horizontalAdvance(text) for _font, metrics, text in laid_out), default=0)
    height = sum(metrics.height() for _font, metrics, _text in laid_out) + size * 0.15 * max(0, len(laid_out) - 1)
    return width, height


def _draw_lines(painter, laid_out, x, y, width, align, size, color, outline=None):
    """The lines from (x, y) down, each aligned in width - outlined when
    outline is a color (text over the image)."""
    pen = None
    if outline is not None:
        pen = QPen(outline, max(1.0, size * 0.1))
        pen.setJoinStyle(Qt.RoundJoin)
    top = y
    for font, metrics, text in laid_out:
        advance = metrics.horizontalAdvance(text)
        left = x if align == "Left" else (x + width - advance if align == "Right" else x + (width - advance) / 2)
        path = QPainterPath()
        path.addText(left, top + metrics.ascent(), font, text)
        if pen is not None:
            painter.strokePath(path, pen)
        painter.fillPath(path, color)
        top += metrics.height() + size * 0.15


def _alignment(position):
    if position.endswith("Left"):
        return "Left"
    if position.endswith("Right"):
        return "Right"
    return "Center"


def paint_overlay(painter, lines, image_rect, style):
    """The lines in a translucent box at the style's position on image_rect
    (image pixels)."""
    if not lines:
        return
    short_side = min(image_rect.width(), image_rect.height())
    size = body_px(short_side, style)
    pad = size * 0.6
    margin = size * 1.0
    align = _alignment(style["position"])
    share = 0.85 if align == "Center" else 0.6
    room = max(size * 4, image_rect.width() * share - 2 * pad)
    room = min(room, image_rect.width() - 2 * (margin + pad))
    laid_out = _layout(lines, style, size, room)
    block_width, block_height = _block_size(laid_out, size)
    box_width, box_height = block_width + 2 * pad, block_height + 2 * pad

    position = style["position"]
    if align == "Left":
        x = image_rect.left() + margin
    elif align == "Right":
        x = image_rect.right() - margin - box_width
    else:
        x = image_rect.center().x() - box_width / 2
    if position.startswith("Top"):
        y = image_rect.top() + margin
    elif position.startswith("Bottom"):
        y = image_rect.bottom() - margin - box_height
    else:
        y = image_rect.center().y() - box_height / 2

    text_color = QColor(style["text_color"])
    background = QColor(style["background_color"])
    background.setAlpha(round(255 * style["background_opacity"] / 100))
    outline = QColor(0, 0, 0, 200) if text_color.lightnessF() >= 0.5 else QColor(255, 255, 255, 200)

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setClipRect(image_rect, Qt.IntersectClip)
    if background.alpha():
        painter.setPen(Qt.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(QRectF(x, y, box_width, box_height), size * 0.35, size * 0.35)
    # The outline keeps text readable over the sky when the box is faint
    _draw_lines(painter, laid_out, x + pad, y + pad, block_width, align, size, text_color,
                outline if style["background_opacity"] < 70 else None)
    painter.restore()


def _banner_layout(lines, image_size, style):
    width, height = image_size
    size = body_px(min(width, height), style)
    pad_x, pad_y = size * 1.2, size * 0.8
    laid_out = _layout(lines, style, size, width - 2 * pad_x, inline=style["banner_style"] == "inline")
    return laid_out, size, pad_x, pad_y


def banner_height(lines, image_size, style):
    """Height in image pixels of the banner for an image of image_size
    (width, height) - 0 when it has nothing to show or the style isn't a banner."""
    if not lines or not is_banner(style) or not image_size[0] or not image_size[1]:
        return 0
    laid_out, size, _pad_x, pad_y = _banner_layout(lines, image_size, style)
    return int(round(_block_size(laid_out, size)[1] + 2 * pad_y))


def paint_banner(painter, lines, rect, image_size, style):
    """The banner in rect (image pixels: the image's width, banner_height() high)."""
    if not lines:
        return
    laid_out, size, pad_x, pad_y = _banner_layout(lines, image_size, style)
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setClipRect(rect, Qt.IntersectClip)
    painter.fillRect(rect, QColor(style["background_color"]))
    _draw_lines(painter, laid_out, rect.left() + pad_x, rect.top() + pad_y, rect.width() - 2 * pad_x,
                style["banner_align"], size, QColor(style["text_color"]))
    painter.restore()


def compose(image, lines, style):
    """The image with its details: painted over it (in place), or a new,
    taller QImage with the banner above or below. QImage only - safe off the
    GUI thread."""
    if not lines:
        return image
    size = (image.width(), image.height())
    if not is_banner(style):
        painter = QPainter(image)
        paint_overlay(painter, lines, QRectF(image.rect()), style)
        painter.end()
        return image
    height = banner_height(lines, size, style)
    result = QImage(image.width(), image.height() + height, QImage.Format_RGB32)
    result.fill(QColor(style["background_color"]))
    painter = QPainter(result)
    above = style["layout"] == "banner_above"
    painter.drawImage(0, height if above else 0, image)
    banner = QRectF(0, 0 if above else image.height(), image.width(), height)
    paint_banner(painter, lines, banner, size, style)
    painter.end()
    return result


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ImageDetailsDialog(WindowPositionMixin, QDialog):
    """Which details show on the image and how - non-modal beside the viewer,
    every change shown on the image straight away."""
    WINDOW_POSITION_KEY = "ImageDetailsDialog"
    style_changed = Signal(dict)
    edits_changed = Signal(str, dict)   # image path, its edits
    save_requested = Signal()

    def __init__(self, style, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Details - Cosmos Collection")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.resize(420, 640)
        self.setup_window_position()
        self.style = normalize_style(style)
        self.image_path = None
        self.auto, self.edits = {}, {}
        self._loading = True   # The image's details are still being gathered
        self._updating = False

        layout = QVBoxLayout(self)
        intro = QLabel("Choose the details to show on the image. They're saved with it by Export.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.layout_combo = QComboBox()
        for key, name in LAYOUTS:
            self.layout_combo.addItem(name, key)
        self.layout_combo.setToolTip("A banner is added to the image, so nothing covers the photo")
        self.layout_combo.currentIndexChanged.connect(self._on_controls_changed)
        form.addRow("Show:", self.layout_combo)
        self.position_combo = QComboBox()
        self.position_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.position_label = QLabel("Position:")
        form.addRow(self.position_label, self.position_combo)
        self.size_combo = QComboBox()
        for key, name, _scale in SIZES:
            self.size_combo.addItem(name, key)
        self.size_combo.setToolTip("Text size, relative to the image")
        self.size_combo.currentIndexChanged.connect(self._on_controls_changed)
        form.addRow("Size:", self.size_combo)
        layout.addLayout(form)

        layout.addWidget(QLabel("Details (drag to reorder):"))
        self.field_list = QListWidget()
        self.field_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.field_list.setDefaultDropAction(Qt.MoveAction)
        self.field_list.itemChanged.connect(self._on_item_changed)
        self.field_list.model().rowsMoved.connect(self._on_rows_moved)
        self.field_list.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.field_list, 1)

        edit_row = QHBoxLayout()
        self.value_edit = QLineEdit()
        self.value_edit.textEdited.connect(self._on_value_edited)
        edit_row.addWidget(self.value_edit, 1)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Use the details found for this image again")
        self.reset_button.clicked.connect(self._reset_value)
        edit_row.addWidget(self.reset_button)
        layout.addLayout(edit_row)
        self.value_hint = QLabel()
        self.value_hint.setWordWrap(True)
        themed_style(self.value_hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        layout.addWidget(self.value_hint)

        # More options: tucked away, most people never need them
        self.more_toggle = QToolButton()
        self.more_toggle.setText("More options")
        self.more_toggle.setCheckable(True)
        self.more_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.more_toggle.setArrowType(Qt.RightArrow)
        self.more_toggle.setStyleSheet("QToolButton { border: none; }")
        self.more_toggle.toggled.connect(self._toggle_more)
        layout.addWidget(self.more_toggle)
        self.more = QWidget()
        more_form = QFormLayout(self.more)
        more_form.setContentsMargins(20, 0, 0, 0)
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(lambda _font: self._on_controls_changed())
        font_row = QHBoxLayout()
        font_row.addWidget(self.font_combo, 1)
        self.default_font_button = QPushButton("Default")
        self.default_font_button.setToolTip("Use the app's font")
        self.default_font_button.clicked.connect(self._use_default_font)
        font_row.addWidget(self.default_font_button)
        more_form.addRow("Font:", font_row)
        self.text_color_button = QPushButton()
        self.text_color_button.clicked.connect(lambda: self._choose_color("text_color", "Text Color"))
        more_form.addRow("Text color:", self.text_color_button)
        self.background_button = QPushButton()
        self.background_button.clicked.connect(lambda: self._choose_color("background_color", "Background Color"))
        more_form.addRow("Background:", self.background_button)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setToolTip("How solid the box behind the text over the image is")
        self.opacity_slider.valueChanged.connect(self._on_controls_changed)
        self.opacity_label = QLabel("Background opacity:")
        more_form.addRow(self.opacity_label, self.opacity_slider)
        self.title_check = QCheckBox("Object name larger and bold")
        self.title_check.toggled.connect(self._on_controls_changed)
        more_form.addRow(self.title_check)
        self.labels_check = QCheckBox("Name each detail (\"Telescope: ...\")")
        self.labels_check.toggled.connect(self._on_controls_changed)
        more_form.addRow(self.labels_check)
        self.banner_style_combo = QComboBox()
        self.banner_style_combo.addItem("One per line", "lines")
        self.banner_style_combo.addItem("Run together", "inline")
        self.banner_style_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.banner_style_label = QLabel("Banner text:")
        more_form.addRow(self.banner_style_label, self.banner_style_combo)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Saved looks - choose one to use it")
        self.preset_combo.activated.connect(self._load_preset)
        preset_row.addWidget(self.preset_combo, 1)
        save_preset = QPushButton("Save As...")
        save_preset.setToolTip("Save this look to use again")
        save_preset.clicked.connect(self._save_preset)
        preset_row.addWidget(save_preset)
        self.delete_preset_button = QPushButton("Delete")
        self.delete_preset_button.clicked.connect(self._delete_preset)
        preset_row.addWidget(self.delete_preset_button)
        more_form.addRow("Presets:", preset_row)
        self.more.setVisible(False)
        layout.addWidget(self.more)

        buttons = QHBoxLayout()
        reset_style = QPushButton("Reset Look")
        reset_style.setToolTip("Go back to the standard details and look")
        reset_style.clicked.connect(self._reset_style)
        buttons.addWidget(reset_style)
        buttons.addStretch()
        save_button = QPushButton("Export...")
        save_button.setToolTip("Save, copy, set as background or share the image")
        save_button.clicked.connect(self.save_requested.emit)
        buttons.addWidget(save_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self._refresh_presets()
        self._show_style()

    # Showing

    def set_details(self, image_path, auto, edits):
        """The image whose details are being edited (auto None: still gathering them)."""
        self.image_path = image_path
        self.auto = dict(auto or {})
        self.edits = dict(edits or {})
        self._loading = auto is None
        self._show_fields()

    def _show_style(self):
        self._updating = True
        try:
            style = self.style
            self.layout_combo.setCurrentIndex(self.layout_combo.findData(style["layout"]))
            self.size_combo.setCurrentIndex(self.size_combo.findData(style["size"]))
            self._fill_positions()
            if style["font_family"]:
                self.font_combo.setCurrentFont(QFont(style["font_family"]))
            else:
                self.font_combo.setCurrentFont(QFont())
            self._color_button(self.text_color_button, style["text_color"])
            self._color_button(self.background_button, style["background_color"])
            self.opacity_slider.setValue(style["background_opacity"])
            self.title_check.setChecked(style["title_emphasis"])
            self.labels_check.setChecked(style["labels"])
            self.banner_style_combo.setCurrentIndex(self.banner_style_combo.findData(style["banner_style"]))
            self._update_visibility()
        finally:
            self._updating = False
        self._show_fields()

    def _fill_positions(self):
        """Position over the image, or the banner's text alignment."""
        banner = is_banner(self.style)
        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        self.position_combo.addItems(BANNER_ALIGNMENTS if banner else POSITIONS)
        self.position_combo.setCurrentText(self.style["banner_align"] if banner else self.style["position"])
        self.position_combo.blockSignals(False)
        self.position_label.setText("Align:" if banner else "Position:")

    def _update_visibility(self):
        banner = is_banner(self.style)
        self.opacity_label.setVisible(not banner)
        self.opacity_slider.setVisible(not banner)
        self.banner_style_label.setVisible(banner)
        self.banner_style_combo.setVisible(banner)

    def _value(self, key):
        if key == "custom":
            return self.style["custom_text"]
        return self.edits.get(key, self.auto.get(key, ""))

    def _show_fields(self):
        current = self.field_list.currentItem()
        current_key = current.data(Qt.UserRole) if current else "title"
        self._updating = True
        try:
            self.field_list.clear()
            for key, on in self.style["fields"]:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, key)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)
                self._label_item(item)
                self.field_list.addItem(item)
                if key == current_key:
                    self.field_list.setCurrentItem(item)
        finally:
            self._updating = False
        self._show_value()

    def _label_item(self, item):
        key = item.data(Qt.UserRole)
        value = " ".join(str(self._value(key) or "").split())
        if self._loading and key != "custom":
            shown = "..."
        elif value:
            shown = value if len(value) <= 60 else value[:57] + "..."
        else:
            shown = "(type your text)" if key == "custom" else "(not recorded)"
        edited = " ✎" if key in self.edits else ""
        item.setText(f"{FIELD_NAMES[key]}{edited}:  {shown}")
        item.setToolTip(value or "")

    def _show_value(self):
        item = self.field_list.currentItem()
        key = item.data(Qt.UserRole) if item else None
        # Typing waits for what was found (the found edits would replace it)
        self.value_edit.setEnabled(key == "custom" or (key is not None and not self._loading))
        if key is None:
            self.value_edit.clear()
            self.reset_button.setEnabled(False)
            self.value_hint.clear()
            return
        self.value_edit.setText(self._value(key))
        if key == "custom":
            self.value_edit.setPlaceholderText("e.g. your name, © 2025, processed in PixInsight")
            self.value_hint.setText("The custom text is the same on every image.")
            self.reset_button.setEnabled(bool(self.style["custom_text"]))
        else:
            found = self.auto.get(key, "")
            self.value_edit.setPlaceholderText(found or "Not recorded - type it here")
            if key in self.edits:
                self.value_hint.setText(f"Edited for this image. Found: {found or 'nothing'}")
            else:
                self.value_hint.setText("Edit it to change it for this image only.")
            self.reset_button.setEnabled(key in self.edits)

    # Changes

    def _emit_style(self):
        self.style_changed.emit(copy.deepcopy(self.style))

    def _on_controls_changed(self, *_args):
        if self._updating:
            return
        style = self.style
        old_banner = is_banner(style)
        style["layout"] = self.layout_combo.currentData()
        style["size"] = self.size_combo.currentData()
        if is_banner(style) == old_banner and self.position_combo.currentText():
            style["banner_align" if old_banner else "position"] = self.position_combo.currentText()
        style["background_opacity"] = self.opacity_slider.value()
        style["title_emphasis"] = self.title_check.isChecked()
        style["labels"] = self.labels_check.isChecked()
        style["banner_style"] = self.banner_style_combo.currentData()
        family = self.font_combo.currentFont().family()
        style["font_family"] = "" if family == QFont().family() else family
        if is_banner(style) != old_banner:
            self._fill_positions()
            self._update_visibility()
        self._emit_style()

    def _use_default_font(self):
        self.style["font_family"] = ""
        self._show_style()
        self._emit_style()

    def _choose_color(self, key, title):
        color = QColorDialog.getColor(QColor(self.style[key]), self, title)
        if color.isValid():
            self.style[key] = color.name()
            self._show_style()
            self._emit_style()

    @staticmethod
    def _color_button(button, color):
        button.setText(color.upper())
        button.setStyleSheet(f"QPushButton {{ background-color: {color}; color: {contrast_text(color)}; }}")

    def _on_item_changed(self, item):
        if self._updating:
            return
        key = item.data(Qt.UserRole)
        for entry in self.style["fields"]:
            if entry[0] == key:
                entry[1] = item.checkState() == Qt.Checked
        self._emit_style()

    def _on_rows_moved(self, *_args):
        """Dragged into a new order - read it back from the list."""
        if self._updating:
            return
        on = dict((key, value) for key, value in self.style["fields"])
        order = [self.field_list.item(row).data(Qt.UserRole) for row in range(self.field_list.count())]
        self.style["fields"] = [[key, on[key]] for key in order if key in on]
        self._emit_style()

    def _on_current_changed(self, *_args):
        if not self._updating:
            self._show_value()

    def _on_value_edited(self, text):
        item = self.field_list.currentItem()
        if item is None:
            return
        key = item.data(Qt.UserRole)
        if key == "custom":
            self.style["custom_text"] = text
            if text.strip():
                self._turn_on(item)
            self._emit_style()
        elif self.image_path is not None:
            if text == self.auto.get(key, ""):
                self.edits.pop(key, None)
            else:
                self.edits[key] = text
                if text.strip():
                    self._turn_on(item)
            self.edits_changed.emit(self.image_path, dict(self.edits))
        self._updating = True
        self._label_item(item)
        self._updating = False
        self.reset_button.setEnabled(key in self.edits or (key == "custom" and bool(text)))

    def _turn_on(self, item):
        """Typing a line's text shows it."""
        if item.checkState() != Qt.Checked:
            self._updating = True
            item.setCheckState(Qt.Checked)
            self._updating = False
            key = item.data(Qt.UserRole)
            for entry in self.style["fields"]:
                if entry[0] == key:
                    entry[1] = True
            self._emit_style()

    def _reset_value(self):
        item = self.field_list.currentItem()
        if item is None:
            return
        key = item.data(Qt.UserRole)
        if key == "custom":
            self.style["custom_text"] = ""
            self._emit_style()
        elif key in self.edits:
            del self.edits[key]
            self.edits_changed.emit(self.image_path, dict(self.edits))
        self._updating = True
        self._label_item(item)
        self._updating = False
        self._show_value()

    def _reset_style(self):
        custom = self.style["custom_text"]
        self.style = normalize_style({"custom_text": custom})
        self._show_style()
        self._emit_style()

    def _toggle_more(self, shown):
        self.more.setVisible(shown)
        self.more_toggle.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)

    # Presets

    def _refresh_presets(self, selected=None):
        presets = load_presets()
        self.preset_combo.clear()
        if not presets:
            self.preset_combo.addItem("No presets yet")
            self.preset_combo.setEnabled(False)
        else:
            self.preset_combo.setEnabled(True)
            for name in sorted(presets, key=str.lower):
                self.preset_combo.addItem(name)
            if selected:
                self.preset_combo.setCurrentText(selected)
        self.delete_preset_button.setEnabled(bool(presets))

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Name for this look:",
                                        text=self.preset_combo.currentText() if self.preset_combo.isEnabled() else "")
        name = name.strip()
        if not ok or not name:
            return
        presets = load_presets()
        if name in presets and QMessageBox.question(
                self, "Replace Preset", f"Replace the preset \"{name}\"?") != QMessageBox.Yes:
            return
        preset = copy.deepcopy(self.style)
        preset.pop("enabled", None)
        presets[name] = preset
        save_presets(presets)
        self._refresh_presets(name)

    def _load_preset(self, _index=None):
        presets = load_presets()
        preset = presets.get(self.preset_combo.currentText())
        if preset is None:
            return
        enabled = self.style["enabled"]
        self.style = normalize_style(preset)
        self.style["enabled"] = enabled
        self._show_style()
        self._emit_style()

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        presets = load_presets()
        if name not in presets:
            return
        if QMessageBox.question(self, "Delete Preset", f"Delete the preset \"{name}\"?") != QMessageBox.Yes:
            return
        del presets[name]
        save_presets(presets)
        self._refresh_presets()
