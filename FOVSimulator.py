import json
import logging
import math
import re
import sqlite3
import threading
from datetime import datetime, timedelta
import urllib.parse
import numpy as np
from PySide6.QtCore import Qt, QUrl, QTimer, QSettings, QDate, QRectF, QPointF, Signal, QEvent
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout,
    QComboBox, QCheckBox, QPushButton, QMessageBox,
    QToolButton, QMenu, QSpinBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QDateEdit, QToolTip, QLineEdit, QCompleter
)
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from DatabaseManager import DatabaseManager
from HorizonProfile import load_active_horizon
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS, contrast_text, font_px, font_size, themed_style, theme_manager
from UrlOpener import open_url
from TimeFormatHelper import format_time, get_time_format_24h
from NINAIntegration import NINAIntegration

logger = logging.getLogger(__name__)

# Aladin Lite v3, hosted in our own page (PAGE_TEMPLATE) with all of Aladin's
# own controls turned off. Pinned, so a new Aladin release can't change it.
ALADIN_SCRIPT = "https://aladin.cds.unistra.fr/AladinLite/api/v3/3.8.2/aladin.js"
ALADIN_BASE_URL = "https://aladin.cds.unistra.fr/"
ALADIN_BROWSER_URL = "https://aladin.cds.unistra.fr/AladinLite/"
SURVEY = "P/DSS2/color"

CAMERA_COLOR = "#ff8800"
EYEPIECE_COLOR = "#00ff00"
HORIZON_COLOR = "#ffaa00"
HORIZON_DIM_COLOR = "rgba(255, 170, 0, 0.45)"   # The flat 0° line, when a custom horizon is drawn too
BLOCKED_FILL = "rgba(255, 140, 0, 0.28)"        # Sky hidden by the custom horizon
MERIDIAN_COLOR = "#4fc3f7"
SKY_REFRESH_MS = 5 * 60 * 1000                  # The sky turns: redraw horizon/meridian every 5 minutes
PATH_COLOR = "rgba(255, 255, 255, 0.85)"        # Night View: the target's path for the night
NIGHT_VIEW_FOV = 110.0                          # Night View: a wide look at the sky
COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


# Camera presets: (heading, [(name as shown, sensor width mm, height mm, pixel size µm or None)]).
# Sizes are resolution x pixel size. The FOV Simulator lists them; the Equipment window
# offers them (named without the size, see preset_camera_name) when adding a camera.
CAMERA_PRESET_GROUPS = (
    ("DSLR CAMERAS", (
        ("Canon Full Frame (36x24mm)", 36, 24, None),
        ("Canon APS-C (22.3x14.9mm)", 22.3, 14.9, None),
        ("Canon APS-H (28.7x19mm)", 28.7, 19.0, None),
        ("Nikon Full Frame (35.9x24mm)", 35.9, 24.0, None),
        ("Nikon APS-C (23.5x15.6mm)", 23.5, 15.6, None),
        ("Sony Full Frame (35.8x23.8mm)", 35.8, 23.8, None),
        ("Sony APS-C (23.5x15.6mm)", 23.5, 15.6, None),
    )),
    ("ZWO ASI CAMERAS", (
        ("ASI6200MM Pro (36x24mm)", 36.0, 24.0, 3.76),
        ("ASI2600MM Pro (23.5x15.7mm)", 23.5, 15.7, 3.76),
        ("ASI533MM Pro (11.3x11.3mm)", 11.3, 11.3, 3.76),
        ("ASI294MM Pro (19.1x13.0mm)", 19.1, 13.0, 4.63),
        ("ASI183MM Pro (13.2x8.8mm)", 13.2, 8.8, 2.4),
        ("ASI585MC (11.1x6.3mm)", 11.1, 6.3, 2.9),
        ("ASI662MC (5.6x3.2mm) (Seestar S30)", 5.6, 3.2, 2.9),
        ("IMX585 (11.1x6.3mm) (Seestar S30 Pro)", 11.1, 6.3, 2.9),
        ("ASI385MC (7.3x4.1mm)", 7.3, 4.1, 3.75),
        ("ASI462MC (5.6x3.2mm) (Seestar S50)", 5.6, 3.2, 2.9),
        ("OS08B10 (11.1x6.3mm) (Seestar S50 Pro)", 11.1, 6.3, 2.9),
        ("ASI224MC (4.9x3.7mm)", 4.9, 3.7, 3.75),
        ("ASI120MM (4.8x3.6mm)", 4.8, 3.6, 3.75),
    )),
    ("QHY CAMERAS", (
        ("QHY600M (36x24mm)", 36.0, 24.0, 3.76),
        ("QHY268M (23.5x15.7mm)", 23.5, 15.7, 3.76),
        ("QHY294M (19.1x13.0mm)", 19.1, 13.0, 4.63),
        ("QHY183M (13.2x8.8mm)", 13.2, 8.8, 2.4),
        ("QHY174M (11.3x7.1mm)", 11.3, 7.1, 5.86),
    )),
    ("SBIG CAMERAS", (
        ("SBIG STF-8300M (17.96x13.52mm)", 17.96, 13.52, 5.4),
        ("SBIG ST-2000XM (11.8x8.9mm)", 11.8, 8.9, 7.4),
    )),
    ("ATIK CAMERAS", (
        ("Atik 460EX (12.5x10.0mm)", 12.5, 10.0, 4.54),
        ("Atik 383L+ (17.6x13.5mm)", 17.6, 13.5, 5.4),
    )),
    ("VAONIS CAMERAS", (
        ("IMX585 (11.1x6.3mm) (Vaonis Vespera II)", 11.1, 6.3, 2.9),
    )),
    ("CELESTRON CAMERAS", (
        ("IMX178 (7.4x4.9mm) (Celestron Origin)", 7.4, 4.9, 2.4),
        ("IMX678 (7.7x4.3mm) (Celestron Origin Mark II)", 7.7, 4.3, 2.0),
    )),
    ("DWARFLAB CAMERAS", (
        ("IMX678 (7.7x4.3mm) (Dwarf 3)", 7.7, 4.3, 2.0),
        ("IMX662 (5.6x3.2mm) (Dwarf Mini)", 5.6, 3.2, 2.9),
        ("OV50Q40 (9.8x7.4mm) (Draco)", 9.8, 7.4, 1.2),
    )),
)


def preset_camera_name(text):
    """A preset's name without its size: 'ASI462MC (5.6x3.2mm) (Seestar S50)' -> 'ASI462MC (Seestar S50)'."""
    return re.sub(r"\s*\([\d.]+x[\d.]+mm\)", "", text).strip()


def camera_presets():
    """{name without size: (sensor width, height, pixel size or None, name as listed)}."""
    return {preset_camera_name(text): (width, height, pixel, text)
            for _heading, cameras in CAMERA_PRESET_GROUPS for text, width, height, pixel in cameras}


def tangent_to_radec(ra0, dec0, x, y):
    """Sky position at tangent-plane coordinates x (east) and y (north) around
    ra0/dec0 - the inverse gnomonic projection, so frames stay true
    rectangles at any declination."""
    dec0_r = math.radians(dec0)
    rho = math.hypot(x, y)
    if rho == 0:
        return ra0 % 360.0, dec0
    c = math.atan(rho)
    sin_c, cos_c = math.sin(c), math.cos(c)
    dec = math.asin(cos_c * math.sin(dec0_r) + y * sin_c * math.cos(dec0_r) / rho)
    ra = ra0 + math.degrees(math.atan2(x * sin_c,
                                       rho * math.cos(dec0_r) * cos_c - y * math.sin(dec0_r) * sin_c))
    return ra % 360.0, math.degrees(dec)


def offset_to_radec(ra0, dec0, east, north):
    """Sky position at an offset of 'east' and 'north' degrees from ra0/dec0.
    Offsets are angles along each axis, as a sensor sees them (tan(angle)
    on the tangent plane)."""
    return tangent_to_radec(ra0, dec0, math.tan(math.radians(east)), math.tan(math.radians(north)))


def frame_to_radec(ra0, dec0, right, up, rotation=0.0):
    """Sky position of a point 'right' and 'up' degrees from the center of a
    camera frame at ra0/dec0, as the frame shows it. rotation is the camera
    angle: where the frame's top points, in degrees from north through east
    (the position angle a plate solve reports). At 0, up is north and right
    is west."""
    x, y = math.tan(math.radians(right)), math.tan(math.radians(up))
    angle = math.radians(rotation)
    east = -x * math.cos(angle) + y * math.sin(angle)
    north = x * math.sin(angle) + y * math.cos(angle)
    return tangent_to_radec(ra0, dec0, east, north)


def frame_corners(ra, dec, width, height, rotation=0.0, steps=8):
    """Outline (RA, Dec points) of a camera frame width x height degrees
    centered on ra/dec, turned by rotation (see frame_to_radec) - each edge
    in steps, so wide fields stay true. Starts at the frame's top left."""
    hw, hh = width / 2.0, height / 2.0
    corners = ((-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh))
    points = []
    for (r1, u1), (r2, u2) in zip(corners, corners[1:] + corners[:1]):
        for i in range(steps):
            t = i / steps
            points.append([round(v, 6) for v in
                           frame_to_radec(ra, dec, r1 + (r2 - r1) * t, u1 + (u2 - u1) * t, rotation)])
    return points


def top_marker(ra, dec, width, height, rotation=0.0):
    """A small triangle on the middle of a frame's top edge, pointing out, so a
    turned frame shows which way is up."""
    size = min(width, height) * 0.08
    corners = ((-size, height / 2.0), (size, height / 2.0), (0.0, height / 2.0 + size * 1.4))
    return [[round(v, 6) for v in frame_to_radec(ra, dec, right, up, rotation)] for right, up in corners]


def rotated_size(width, height, rotation):
    """Width and height (degrees) of the box around a frame turned by rotation."""
    angle = math.radians(rotation)
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    return width * c + height * s, width * s + height * c


def mosaic_size(width, height, columns, rows, overlap):
    """Total width and height (degrees) of a columns x rows mosaic whose
    panels overlap by overlap (0-1), along the frame's own axes."""
    return (width * (1 + (columns - 1) * (1 - overlap)),
            height * (1 + (rows - 1) * (1 - overlap)))


def mosaic_panels(ra, dec, width, height, columns, rows, overlap, rotation=0.0):
    """Mosaic panels around ra/dec: [{"number", "center", "outline"}], laid
    out along the frame's axes (turned by rotation, see frame_to_radec) and
    numbered in reading order as the frame shows them."""
    step_x, step_y = width * (1 - overlap), height * (1 - overlap)
    panels = []
    for row in range(rows):
        for column in range(columns):
            right = (column - (columns - 1) / 2.0) * step_x
            up = ((rows - 1) / 2.0 - row) * step_y
            center = frame_to_radec(ra, dec, right, up, rotation)
            panels.append({"number": len(panels) + 1, "center": [round(v, 6) for v in center],
                           "outline": frame_corners(center[0], center[1], width, height, rotation)})
    return panels


def default_view_fov(data):
    """View width (degrees) for an object without a frame: 3x its size, at least 0.5°."""
    try:
        size_min = data.get('size_min', 30) or 30  # Arcminutes - 30 if unknown
        size_max = data.get('size_max', 30) or 30
        return max(max(size_min, size_max) / 60.0 * 3.0, 0.5)
    except (TypeError, ValueError):
        return 1.0


def coordinates_target(text):
    """A target for typed coordinates - '10.68 41.27' (degrees) or
    '00h42m44s +41d16m09s' / '00:42:44 +41:16:09' (RA in hours) - or None."""
    if not re.search(r"\d", text) or re.search(r"[a-ce-gi-ln-rt-z]", text, re.IGNORECASE):
        return None  # Names have letters besides h/m/s/d
    parts = text.replace(",", " ").split()
    try:
        if len(parts) == 2 and all(re.fullmatch(r"[+-]?\d+(\.\d+)?", part) for part in parts):
            ra, dec = float(parts[0]), float(parts[1])
        else:
            coord = SkyCoord(text, unit=(u.hourangle, u.deg))
            ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    except (ValueError, TypeError):
        return None
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        return None
    return {"name": f"{format_ra(ra)} {format_dec(dec)}", "ra_deg": ra, "dec_deg": dec,
            "size_min": 30, "size_max": 30}


def compass_point(azimuth):
    """'N', 'NE', ... for an azimuth in degrees."""
    return COMPASS[int((azimuth % 360) / 45.0 + 0.5) % 8]


def position_angle(ra1, dec1, ra2, dec2):
    """Direction from the first sky position to the second, in degrees from north through east."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    d_ra = ra2 - ra1
    return math.degrees(math.atan2(math.sin(d_ra) * math.cos(dec2),
                                   math.cos(dec1) * math.sin(dec2) - math.sin(dec1) * math.cos(dec2) * math.cos(d_ra))) % 360


def _observer_frame(lat, lon, when):
    return AltAz(obstime=when, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg))


def altaz_to_radec(lat, lon, when, alts, azs):
    """RA and Dec arrays (degrees) of altitudes/azimuths seen from lat/lon at 'when' (astropy Time)."""
    sky = SkyCoord(alt=np.asarray(alts, dtype=float) * u.deg, az=np.asarray(azs, dtype=float) * u.deg,
                   frame=_observer_frame(lat, lon, when)).icrs
    return sky.ra.deg, sky.dec.deg


def radec_to_altaz(lat, lon, when, ra, dec):
    """(altitude, azimuth) in degrees of a sky position seen from lat/lon at 'when'."""
    seen = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(_observer_frame(lat, lon, when))
    return float(seen.alt.deg), float(seen.az.deg)


def zenith_up_rotation(lat, lon, when, ra, dec):
    """The viewer rotation that puts the zenith straight up at a view centered on
    ra/dec (Aladin's rotation turns north that many degrees counterclockwise)."""
    zenith_ra, zenith_dec = altaz_to_radec(lat, lon, when, [90.0], [0.0])
    return -position_angle(ra, dec, float(zenith_ra[0]), float(zenith_dec[0]))


# The time slider: one night, local noon to noon, in 5-minute steps
NIGHT_STEP_MINUTES = 5
NIGHT_STEPS = 24 * 60 // NIGHT_STEP_MINUTES
SUNRISE_ALT = -0.833                  # The sun's top at the horizon, with refraction
ASTRONOMICAL_DARK_ALT = -18.0


def night_profile(lat, lon, noon, ra=None, dec=None, horizon=None):
    """The 24 hours from 'noon' (an aware local datetime) in NIGHT_STEP_MINUTES steps:
    {"times": aware datetimes, "sun_alt", "target_alt", "target_az" (numpy arrays,
    the target ones None without a position), "target_up" (above the horizon - the
    custom one if given), "transit" (index of the meridian crossing - the target's
    highest - or None), "sunset", "dusk", "dawn", "sunrise" (indexes or None)}."""
    from astropy.coordinates import get_sun
    steps = np.arange(NIGHT_STEPS + 1)
    times = Time(noon) + steps * NIGHT_STEP_MINUTES * u.min
    frame = AltAz(obstime=times, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg))
    sun_alt = get_sun(times).transform_to(frame).alt.deg
    local_times = [noon + timedelta(minutes=int(i) * NIGHT_STEP_MINUTES) for i in steps]
    profile = {"times": local_times, "sun_alt": sun_alt, "target_alt": None, "target_az": None,
               "target_up": None, "transit": None}
    if ra is not None and dec is not None:
        seen = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(frame)
        alt, az = seen.alt.deg, seen.az.deg
        floor = np.maximum(horizon.altitude_at(az), 0.0) if horizon is not None else 0.0
        profile.update(target_alt=alt, target_az=az, target_up=alt > floor)
        highest = int(np.argmax(alt))
        if 0 < highest < NIGHT_STEPS:  # A real crossing, not the window's edge
            profile["transit"] = highest

    def crossing(threshold, going_down):
        below = sun_alt < threshold
        for i in range(1, len(below)):
            if below[i] != below[i - 1] and below[i] == going_down:
                return i
        return None
    profile.update(sunset=crossing(SUNRISE_ALT, True), dusk=crossing(ASTRONOMICAL_DARK_ALT, True),
                   dawn=crossing(ASTRONOMICAL_DARK_ALT, False), sunrise=crossing(SUNRISE_ALT, False))
    return profile


def target_path(lat, lon, when, night):
    """The target's track across the night as it'll look from the ground, drawn in
    the sky of 'when' (so it stays put on a ground-fixed view): ([[ra, dec], ...]
    segments while it's up between sunset and sunrise, [(index, ra, dec)] on the hour)."""
    if night.get("target_alt") is None:
        return [], []
    sun_down = night["sun_alt"] < SUNRISE_ALT
    if not sun_down.any():
        sun_down[:] = True  # No night at all (midsummer far north): the whole day
    keep = sun_down & (night["target_alt"] > -1.0)
    if not keep.any():
        return [], []
    indexes = np.nonzero(keep)[0]
    ras, decs = altaz_to_radec(lat, lon, when, night["target_alt"][indexes], night["target_az"][indexes])
    segments, current, previous = [], [], None
    hours = []
    for index, ra, dec in zip(indexes, ras, decs):
        if previous is not None and index != previous + 1:
            segments.append(current)
            current = []
        current.append([round(float(ra), 5), round(float(dec), 5)])
        previous = index
        if night["times"][index].minute == 0 and night["target_alt"][index] > 0:
            hours.append((int(index), float(ra), float(dec)))
    segments.append(current)
    return [segment for segment in segments if len(segment) > 1], hours


def sky_state(sun_alt):
    """'Daylight', 'Civil twilight', ... 'Dark' for the sun's altitude."""
    if sun_alt > SUNRISE_ALT:
        return "Daylight"
    if sun_alt > -6:
        return "Civil twilight"
    if sun_alt > -12:
        return "Nautical twilight"
    if sun_alt > ASTRONOMICAL_DARK_ALT:
        return "Astronomical twilight"
    return "Dark"


def hour_label(moment):
    """'9 PM' or '21h' - an hour mark, in the user's time format."""
    if get_time_format_24h():
        return f"{moment.hour}h"
    hour = moment.hour % 12 or 12
    return f"{hour} {'AM' if moment.hour < 12 else 'PM'}"


def sky_tint(sun_alt):
    """How strongly to tint the sky blue for the sun's altitude: 0 when dark."""
    return max(0.0, min(1.0, (sun_alt - ASTRONOMICAL_DARK_ALT) / 24.0)) * 0.55


def angular_distance(ra1, dec1, ra2, dec2):
    """Angle between two sky positions, in degrees."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((dec2 - dec1) / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


# --- A framing saved to a session ---
# One per session, in usersessions.framing as JSON: {"version": 1, "ra", "dec"
# (the frame or mosaic center), "rotation", "columns", "rows", "overlap" (0-1),
# "panel_width", "panel_height" (degrees), "telescope", "focal_length",
# "camera", "barlow", "saved"}.

PANEL_IN_NAME = re.compile(r"\bpanel\s*(\d+)\b", re.IGNORECASE)


def short_object_name(name):
    """'Sh2 129 (Squid Nebula)' -> 'Sh2 129' - panel names stay short, like NINA's."""
    return (name or "Target").split(" (")[0].strip() or "Target"


def framing_panels(framing, name):
    """[{"number", "name", "ra", "dec"}] for a saved framing; a single frame is
    one entry named after the object, mosaic panels are '<object> Panel <n>'."""
    columns, rows = framing.get("columns", 1), framing.get("rows", 1)
    panels = mosaic_panels(framing["ra"], framing["dec"], framing["panel_width"], framing["panel_height"],
                           columns, rows, framing.get("overlap", 0.0), framing.get("rotation", 0))
    short = short_object_name(name)
    return [{"number": p["number"], "name": short if columns * rows == 1 else f"{short} Panel {p['number']}",
             "ra": p["center"][0], "dec": p["center"][1]} for p in panels]


def framing_summary(framing):
    """'3×2 mosaic, 20% overlap, rotated 30°' (or 'Single frame')."""
    columns, rows = framing.get("columns", 1), framing.get("rows", 1)
    parts = [f"{columns}×{rows} mosaic, {framing.get('overlap', 0):.0%} overlap" if columns * rows > 1
             else "Single frame"]
    if framing.get("rotation"):
        parts.append(f"rotated {framing['rotation']}°")
    return ", ".join(parts)


def match_panels(framing, subs):
    """Light subs per panel. subs: [{"ra", "dec", "object", "seconds"}].
    A sub belongs to the panel nearest where it was taken (when its header has
    a position within a panel's size of it), else to 'Panel <n>' in its object
    name. Returns ({panel number: {"subs", "seconds"}}, subs not matched)."""
    panels = framing_panels(framing, "")
    numbers = {p["number"] for p in panels}
    reach = max(framing["panel_width"], framing["panel_height"])
    counts = {n: {"subs": 0, "seconds": 0.0} for n in numbers}
    unmatched = 0
    for sub in subs:
        number = None
        if len(panels) == 1:
            number = 1
        elif sub.get("ra") is not None and sub.get("dec") is not None:
            distance, nearest = min((angular_distance(sub["ra"], sub["dec"], p["ra"], p["dec"]), p["number"])
                                    for p in panels)
            if distance <= reach:
                number = nearest
        if number is None:
            found = PANEL_IN_NAME.search(sub.get("object") or "")
            if found and int(found.group(1)) in numbers:
                number = int(found.group(1))
        if number is None:
            unmatched += 1
            continue
        counts[number]["subs"] += 1
        counts[number]["seconds"] += sub.get("seconds") or 0.0
    return counts, unmatched


def format_ra(ra):
    """'00h 42m 44.3s'"""
    total = (ra % 360.0) / 15.0 * 3600.0
    hours, rest = divmod(round(total, 1), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{int(hours) % 24:02d}h {int(minutes):02d}m {seconds:04.1f}s"


def format_dec(dec):
    """'+41° 16' 08"'"""
    sign = "-" if dec < 0 else "+"
    total = round(abs(dec) * 3600.0)
    degrees, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{sign}{int(degrees):02d}° {int(minutes):02d}' {int(seconds):02d}\""


def load_session_framing(conn, session_id):
    """The session's saved framing (dict), or None."""
    try:
        row = conn.execute("SELECT framing FROM usersessions WHERE id = ?", (session_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None
    except (sqlite3.Error, ValueError, TypeError) as e:
        logger.warning(f"Could not read the framing of session {session_id}: {e}")
        return None


def save_session_framing(conn, session_id, framing):
    """Store (or with None, remove) a session's framing. Does not commit."""
    conn.execute("UPDATE usersessions SET framing = ? WHERE id = ?",
                 (json.dumps(framing) if framing else None, session_id))


def sky_lines(lat, lon, when, horizon=True, meridian=False, profile=None, ra=None, dec=None):
    """Horizon and meridian for an observer, as RA/Dec point lists for 'when'.

    Returns {"horizon": the 0° line, "custom": the custom horizon line (or None),
    "blocked": polygons of sky the custom horizon hides, "meridian": north -
    zenith - south (or None), "labels": [{"text", "ra", "dec", "color"}],
    "target_alt", "target_az"}. All in one coordinate transform."""
    frame = AltAz(obstime=when, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg))
    parts = {}      # name -> (start, end) into the arrays below
    alts, azs = [], []

    def add(name, part_alts, part_azs):
        parts[name] = (len(alts), len(alts) + len(part_alts))
        alts.extend(part_alts)
        azs.extend(part_azs)

    around = np.arange(0, 361, 1.0)
    blocked_names = []
    labels = []
    if horizon:
        add("horizon", [0.0] * len(around), around)
        if profile is not None:
            heights = np.clip(profile.altitude_at(around), 0.0, 90.0)
            if heights.max() > 0.05:
                add("custom", list(heights), around)
                # The blocked sky in 10° slices (one polygon around the whole sky
                # would wrap): along the custom horizon, then back along 0°
                for start in range(0, 360, 10):
                    slice_az = np.arange(start, start + 11, 1.0)
                    slice_alt = np.clip(profile.altitude_at(slice_az), 0.0, 90.0)
                    if slice_alt.max() <= 0.05:
                        continue
                    name = f"blocked{start}"
                    add(name, list(slice_alt) + [0.0] * len(slice_az), list(slice_az) + list(slice_az[::-1]))
                    blocked_names.append(name)
        for i, label in enumerate(COMPASS):
            add(f"label:{label}", [2.0], [i * 45.0])
            labels.append((f"label:{label}", label, HORIZON_COLOR))
    if meridian:
        up = np.arange(0, 91, 1.0)
        add("meridian", list(up) + list(up[::-1][1:]), [180.0] * len(up) + [0.0] * (len(up) - 1))
        # Named on the side objects cross highest (south in the northern hemisphere)
        add("label:meridian", [55.0], [180.0 if lat >= 0 else 0.0])
        labels.append(("label:meridian", "Meridian", MERIDIAN_COLOR))
        add("label:zenith", [90.0], [0.0])
        labels.append(("label:zenith", "Zenith", MERIDIAN_COLOR))

    sky = SkyCoord(alt=np.asarray(alts) * u.deg, az=np.asarray(azs) * u.deg, frame=frame).icrs
    points = np.column_stack([sky.ra.deg, sky.dec.deg]).round(5).tolist()

    def part(name):
        if name not in parts:
            return None
        start, end = parts[name]
        return points[start:end]

    target_alt = target_az = None
    if ra is not None and dec is not None:
        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(frame)
        target_alt, target_az = float(target.alt.deg), float(target.az.deg)
    return {
        "horizon": part("horizon"),
        "custom": part("custom"),
        "blocked": [part(name) for name in blocked_names],
        "meridian": part("meridian"),
        "labels": [{"text": text, "ra": part(key)[0][0], "dec": part(key)[0][1], "color": color}
                   for key, text, color in labels],
        "target_alt": target_alt,
        "target_az": target_az,
    }


# The viewer page: Aladin Lite with none of its own controls (no toolbar /
# left bar, menus or status bar) - our Qt controls drive it through the cc*
# functions, and it reports back by setting document.title to "cc:{json}".
PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); }
  #aladin { position: absolute; left: 0; top: 0; right: 0; bottom: 0; }
  #labels { position: absolute; left: 0; top: 0; right: 0; bottom: 0; pointer-events: none; overflow: hidden; z-index: 20; }
  #knob { position: absolute; width: 14px; height: 14px; border-radius: 50%; transform: translate(-50%, -50%);
          display: none; z-index: 22; cursor: grab; box-sizing: border-box; touch-action: none;
          background: rgba(0, 0, 0, 0.55); border: 2px solid var(--frame); box-shadow: 0 0 3px #000; }
  #knob:hover { background: var(--frame); }
  #tint { position: absolute; left: 0; top: 0; right: 0; bottom: 0; pointer-events: none; z-index: 15; }
  .sky-label.small { font-size: 11px; font-weight: 600; }
  .sky-label { position: absolute; transform: translate(-50%, -50%); white-space: nowrap;
               font: bold 13px "Segoe UI", Arial, sans-serif; text-shadow: 0 0 3px #000, 0 0 3px #000, 0 0 2px #000; }
  #indicator { position: absolute; left: 0; right: 0; text-align: center; display: none; pointer-events: none; z-index: 21;
               font: bold 13px "Segoe UI", Arial, sans-serif; text-shadow: 0 0 3px #000, 0 0 3px #000; }
  #message { position: absolute; left: 0; top: 0; right: 0; bottom: 0; display: none; align-items: center; justify-content: center;
             text-align: center; padding: 20px; z-index: 30; background: var(--bg); color: var(--text);
             font: 14px "Segoe UI", Arial, sans-serif; }
</style>
</head>
<body>
<div id="aladin"></div><div id="labels"></div><div id="knob" title="Drag to rotate the frame (Shift: 15° steps)"></div>
<div id="tint"></div><div id="indicator"></div><div id="message"></div>
<script>
var config = __CONFIG__;
function applyColors(colors) {
    document.documentElement.style.setProperty('--bg', colors.background);
    document.documentElement.style.setProperty('--text', colors.text);
}
applyColors(config.colors);
document.documentElement.style.setProperty('--frame', config.colors.frame);
var reported = {};
function report(data) {
    for (var key in data) reported[key] = data[key];
    document.title = 'cc:' + JSON.stringify(reported);
}
function showMessage(text) {
    var m = document.getElementById('message');
    m.textContent = text;
    m.style.display = 'flex';
    report({error: text});
}
</script>
<script src="__SCRIPT__" charset="utf-8"
        onerror="showMessage('Could not load the sky viewer. Check your internet connection.')"></script>
<script>
var aladin = null, frameLayer = null, skyLayer = null;
var state = {frames: null, sky: null};
var labels = [];

function makeShape(s) {
    // Hover/selection keep the same color - these outlines aren't clickable objects
    var options = {color: s.color, lineWidth: s.width || 2, opacity: 1,
                   selectionColor: s.color, hoverColor: s.color};
    if (s.fill) { options.fill = true; options.fillColor = s.fill; }
    if (s.type === 'circle') return A.circle(s.ra, s.dec, s.radius, options);
    if (s.type === 'polygon') return A.polygon(s.points.slice(), options);
    return A.polyline(s.points, options);
}

function setLayer(layer, shapes) {
    layer.removeAll();
    (shapes || []).forEach(function (s) {
        try { layer.add(makeShape(s)); } catch (e) { console.error('Shape failed', e); }
    });
}

function inView(p, w, h) {
    return p && !isNaN(p[0]) && !isNaN(p[1]) && p[0] >= 0 && p[1] >= 0 && p[0] <= w && p[1] <= h;
}

function pixel(ra, dec) {
    try { return aladin.world2pix(ra, dec); } catch (e) { return null; }
}

function placeLabels() {
    var host = document.getElementById('labels');
    host.innerHTML = '';
    var w = host.clientWidth, h = host.clientHeight;
    var degreesPerPixel = aladin.getFov()[0] / Math.max(w, 1);
    labels.forEach(function (l) {
        if (l.size && l.size / degreesPerPixel < 30) return;
        var p = pixel(l.ra, l.dec);
        if (!inView(p, w, h)) return;
        var d = document.createElement('div');
        d.className = l.small ? 'sky-label small' : 'sky-label';
        d.textContent = l.text;
        d.style.color = l.color;
        d.style.left = p[0] + 'px';
        d.style.top = p[1] + 'px';
        host.appendChild(d);
    });
    // Point to the horizon when none of it is in view
    var indicator = document.getElementById('indicator');
    var line = state.sky && state.sky.horizon;
    var seen = !line || line.some(function (pt) { return inView(pixel(pt[0], pt[1]), w, h); });
    if (seen) {
        indicator.style.display = 'none';
    } else {
        var below = state.sky.targetAlt === null || state.sky.targetAlt >= 0;
        indicator.textContent = below ? '▼ Horizon (zoom out to see it)' : '▲ Horizon (zoom out to see it)';
        indicator.style.color = state.sky.color;
        indicator.style.top = below ? '' : '10px';
        indicator.style.bottom = below ? '10px' : '';
        indicator.style.display = 'block';
    }
    placeKnob(w, h);
}

// Rotation handle: drag it around the frame's center to turn the frame
var dragging = false;
function placeKnob(w, h) {
    if (dragging) return;  // It follows the mouse
    var knob = document.getElementById('knob');
    var handle = state.frames && state.frames.handle;
    var p = handle ? pixel(handle.knob[0], handle.knob[1]) : null;
    if (!inView(p, w, h)) { knob.style.display = 'none'; return; }
    knob.style.left = p[0] + 'px';
    knob.style.top = p[1] + 'px';
    knob.style.display = 'block';
}

function positionAngle(x, y) {
    // Where (x, y) lies from the frame's center, in degrees from north through east
    var c = state.frames.handle.center;
    var p0 = pixel(c[0], c[1]);
    if (!p0) return null;
    var step = aladin.getFov()[0] / 50;
    var flip = c[1] + step > 89.99;  // Near the pole: north from a point south of the center
    var pn = pixel(c[0], flip ? c[1] - step : c[1] + step);
    var pe = pixel(c[0] + step / Math.max(Math.cos(c[1] * Math.PI / 180), 0.01), c[1]);
    if (!pn || !pe) return null;
    var nx = (pn[0] - p0[0]) * (flip ? -1 : 1), ny = (pn[1] - p0[1]) * (flip ? -1 : 1);
    var ex = pe[0] - p0[0], ey = pe[1] - p0[1];
    var nl = Math.hypot(nx, ny) || 1, el = Math.hypot(ex, ey) || 1;
    var mx = x - p0[0], my = y - p0[1];
    var angle = Math.atan2((mx * ex + my * ey) / el, (mx * nx + my * ny) / nl) * 180 / Math.PI;
    return (angle + 360) % 360;
}

var rotateSeq = 0;
(function () {
    var knob = document.getElementById('knob');
    knob.addEventListener('pointerdown', function (e) {
        if (!aladin || !state.frames || !state.frames.handle) return;
        dragging = true;
        try { knob.setPointerCapture(e.pointerId); } catch (err) {}
        knob.style.cursor = 'grabbing';
        e.preventDefault();
        e.stopPropagation();
    });
    knob.addEventListener('pointermove', function (e) {
        if (!dragging) return;
        var r = document.getElementById('aladin').getBoundingClientRect();
        var x = e.clientX - r.left, y = e.clientY - r.top;
        knob.style.left = x + 'px';
        knob.style.top = y + 'px';
        var angle = positionAngle(x, y);
        if (angle !== null) report({rotate: angle, snap: e.shiftKey, rotating: true, rotateSeq: ++rotateSeq});
        e.preventDefault();
        e.stopPropagation();
    });
    function endDrag() {
        if (!dragging) return;
        dragging = false;
        knob.style.cursor = 'grab';
        report({rotating: false, rotateSeq: ++rotateSeq});
        refresh();
    }
    knob.addEventListener('pointerup', endDrag);
    knob.addEventListener('pointercancel', endDrag);
})();

// Keep the HTML overlays (labels, the rotation handle) on the sky: check the view every
// frame and re-place them when it moved. Aladin doesn't announce zooms set from code, and
// its wheel zooms and coasting pans keep animating after their events.
var lastView = '', overlaysDirty = true, reportedView = null, reportedAt = 0;
function refresh() { overlaysDirty = true; }
function watchView() {
    if (aladin) {
        var c = aladin.getRaDec(), f = aladin.getFov(), host = document.getElementById('labels');
        // Where the sky lands on screen - also changes when Aladin's canvas settles its size
        var p0 = pixel(c[0], c[1]), p1 = pixel(c[0], c[1] + (c[1] > 0 ? -1 : 1) * f[0] / 8);
        var view = [c[0], c[1], f[0], f[1], host.clientWidth, host.clientHeight,
                    p0 && p0[0], p0 && p0[1], p1 && p1[0], p1 && p1[1]].join(',');
        if (overlaysDirty || view !== lastView) {
            overlaysDirty = false;
            lastView = view;
            placeLabels();
        }
        var now = Date.now(), seen = [c[0], c[1], f[0], aladin.getRotation()].join(',');
        if (seen !== reportedView && now - reportedAt > 150) {
            reportedView = seen;
            reportedAt = now;
            report({fov: f[0], ra: c[0], dec: c[1], rot: aladin.getRotation()});
        }
    }
    requestAnimationFrame(watchView);
}

function applyState() {
    if (!aladin) return;
    setLayer(skyLayer, state.sky ? state.sky.shapes : []);
    setLayer(frameLayer, state.frames ? state.frames.shapes : []);
    labels = [].concat((state.sky && state.sky.labels) || [], (state.frames && state.frames.labels) || []);
    refresh();
}

// Called from Qt
window.ccSetFrames = function (frames) { state.frames = frames; applyState(); };
window.ccSetSky = function (sky) { state.sky = sky; applyState(); };
window.ccZoom = function (fov) { if (aladin) aladin.setFoV(fov); else config.fov = fov; };
window.ccCenter = function (fov) {
    if (!aladin) return;
    if (config.ra !== null && config.dec !== null) aladin.gotoRaDec(config.ra, config.dec);
    else if (config.target) aladin.gotoObject(config.target);
    aladin.setFoV(fov);
};
window.ccView = function () {
    if (!aladin) return null;
    var c = aladin.getRaDec();
    return JSON.stringify({ra: c[0], dec: c[1], fov: aladin.getFov()[0], rot: aladin.getRotation()});
};
function turnView(rotation) {
    // aladin.setRotation() ignores 0 (it rejects any falsy angle) - its view's own doesn't
    if (aladin.view && typeof aladin.view.setRotation === 'function') aladin.view.setRotation(rotation);
    else aladin.setRotation(rotation || 1e-9);
}
window.ccSetView = function (ra, dec, rotation, fov) {
    if (!aladin) return;
    aladin.gotoRaDec(ra, dec);
    turnView(rotation);
    if (fov) aladin.setFoV(fov);
};
window.ccRotate = function (rotation) { if (aladin) turnView(rotation); };
window.ccSetTarget = function (ra, dec, target) { config.ra = ra; config.dec = dec; config.target = target; };
window.ccTint = function (alpha) {
    document.getElementById('tint').style.background = alpha > 0 ? 'rgba(70, 110, 170, ' + alpha + ')' : 'none';
};
window.ccColors = function (colors) {
    applyColors(colors);
    if (aladin) { try { aladin.setBackgroundColor(colors.background); } catch (e) {} }
};

if (typeof A !== 'undefined') {
    A.init.then(function () {
        var options = {
            survey: config.survey, fov: config.fov, projection: 'SIN', cooFrame: 'ICRS',
            showZoomControl: false, showLayersControl: false, showFullscreenControl: false,
            showSimbadPointerControl: false, showCooGridControl: false, showSettingsControl: false,
            showColorPickerControl: false, showShareControl: false, showFrame: false, showFov: false,
            showCooLocation: false, showProjectionControl: false, showContextMenu: false,
            showStatusBar: false, showReticle: true, reticleColor: config.colors.reticle, reticleSize: 20,
            backgroundColor: config.colors.background, log: false, samp: false
        };
        if (config.target) options.target = config.target;
        aladin = A.aladin('#aladin', options);
        skyLayer = A.graphicOverlay({name: 'sky'});
        aladin.addOverlay(skyLayer);
        frameLayer = A.graphicOverlay({name: 'frames'});
        aladin.addOverlay(frameLayer);
        applyState();
        report({ready: true, fov: aladin.getFov()[0]});
        requestAnimationFrame(watchView);
    }).catch(function (e) {
        showMessage('The sky viewer could not start (' + e + ').');
    });
}
</script>
</body>
</html>
"""


def viewer_page(config):
    """The viewer page's HTML for config ({"ra", "dec", "target", "fov",
    "survey", "colors"})."""
    return (PAGE_TEMPLATE.replace("__SCRIPT__", ALADIN_SCRIPT)
            .replace("__CONFIG__", json.dumps(config).replace("</", "<\\/")))


def page_colors():
    """Theme colors for the viewer page."""
    return {"background": COLORS["background"], "text": COLORS["text"], "reticle": COLORS["accent"],
            "frame": CAMERA_COLOR}


class AladinLiteWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "AladinLite"

    SMART_TELESCOPES = [
        {'name': 'ZWO Seestar S30',         'aperture': 30,  'focal_length': 150, 'camera_text': 'ASI662MC (5.6x3.2mm) (Seestar S30)',            'smart_telescope': True},
        {'name': 'ZWO Seestar S30 Pro',     'aperture': 30,  'focal_length': 160, 'camera_text': 'IMX585 (11.1x6.3mm) (Seestar S30 Pro)',         'smart_telescope': True},
        {'name': 'ZWO Seestar S50',         'aperture': 50,  'focal_length': 250, 'camera_text': 'ASI462MC (5.6x3.2mm) (Seestar S50)',            'smart_telescope': True},
        {'name': 'ZWO Seestar S50 Pro',     'aperture': 50,  'focal_length': 260, 'camera_text': 'OS08B10 (11.1x6.3mm) (Seestar S50 Pro)',        'smart_telescope': True},
        {'name': 'Vaonis Vespera II',        'aperture': 50,  'focal_length': 250, 'camera_text': 'IMX585 (11.1x6.3mm) (Vaonis Vespera II)',       'smart_telescope': True},
        {'name': 'Celestron Origin',         'aperture': 152, 'focal_length': 335, 'camera_text': 'IMX178 (7.4x4.9mm) (Celestron Origin)',         'smart_telescope': True},
        {'name': 'Celestron Origin Mark II', 'aperture': 152, 'focal_length': 335, 'camera_text': 'IMX678 (7.7x4.3mm) (Celestron Origin Mark II)', 'smart_telescope': True},
        {'name': 'Dwarf 3',                  'aperture': 35,  'focal_length': 150, 'camera_text': 'IMX678 (7.7x4.3mm) (Dwarf 3)',                  'smart_telescope': True},
        {'name': 'Dwarf Mini',               'aperture': 30,  'focal_length': 150, 'camera_text': 'IMX662 (5.6x3.2mm) (Dwarf Mini)',               'smart_telescope': True},
        {'name': 'DWARFLAB Draco',           'aperture': 90,  'focal_length': 340, 'camera_text': 'OV50Q40 (9.8x7.4mm) (Draco)',                   'smart_telescope': True},
    ]

    def __init__(self, data: dict, parent=None, framing=None, session_id=None):
        """framing: a session's saved framing to show (see load_session_framing);
        session_id: the session it was opened from - Save to Session offers it first."""
        super().__init__(parent)
        self.setWindowTitle(f"{data['name']} - FOV Simulator - Cosmos Collection")
        self.resize(1200, 800)
        self.setup_window_position()

        self.data = data
        self.telescopes = []
        self.selected_telescope = None
        self.current_fov = None
        self.view_fov = None          # The viewer's current width (degrees), as the page reports it
        self.page_loaded = False      # The viewer page has loaded (Aladin itself may still be starting)
        self.observer_lat = None
        self.observer_lon = None
        self.observer_name = None
        self.horizon_profile = None   # The active location's custom horizon, if it has one
        self.target_altaz = None      # (alt, az) of the target when the horizon/meridian was last drawn
        self.frame_center = None      # (RA, Dec) the frame was moved to - None keeps it on the target
        self._rotating = False        # The rotation handle is being dragged
        self.observer_tz = None       # The active location's time zone
        self.night = None             # night_profile() of the night on the time slider
        self.night_noon = None        # Local noon starting that night (the slider's left end)
        self.live = True              # The time follows the clock (else it's the slider's)
        self._shown_time = None       # The time the horizon/meridian (and Night View) show
        self._reanchoring = False     # Night View: moving the view with the sky, waiting on the page
        self._reanchor_again = False
        self._rotate_seq = None       # The last rotation report handled
        self._page_error = False
        self.session_id = session_id
        self.framing_note = None      # Why a saved framing couldn't be restored exactly

        self.default_fov = default_view_fov(data)
        self.current_fov = self.default_fov

        # Create main layout
        layout = QVBoxLayout()

        # Equipment row
        telescope_layout = QHBoxLayout()

        # Telescope selection
        telescope_label = QLabel("Telescope:")
        self.telescope_combo = QComboBox()
        self.telescope_combo.setMinimumWidth(280)
        self.telescope_combo.addItem("Default View", None)
        self.telescope_combo.currentTextChanged.connect(self._on_telescope_changed)

        self.show_smart_telescopes = QCheckBox("Show Smart Telescopes")
        self.show_smart_telescopes.setChecked(False)
        self.show_smart_telescopes.toggled.connect(self._on_show_smart_telescopes_toggled)

        # Load telescopes
        self._load_telescopes()

        # What to draw on the sky
        self.show_telescope_fov = QCheckBox("Telescope FOV")
        self.show_telescope_fov.setToolTip("Outline what the selected telescope and camera or eyepiece see")
        self.show_telescope_fov.setChecked(False)
        self.show_telescope_fov.toggled.connect(self._on_show_fov_toggled)

        self.show_horizon = QCheckBox("Virtual Horizon")
        self.show_horizon.setToolTip(
            "Your horizon right now at the active location, with compass points.\n"
            "If the location has a custom horizon, the sky it blocks is shaded.")
        self.show_horizon.setChecked(False)
        self.show_horizon.toggled.connect(self._on_show_horizon_toggled)

        self.show_meridian = QCheckBox("Meridian Line")
        self.show_meridian.setToolTip(
            "The line from north through the zenith to south, where objects are highest -\n"
            "and where a German equatorial mount does its meridian flip")
        self.show_meridian.setChecked(False)
        self.show_meridian.toggled.connect(self._on_show_meridian_toggled)

        # Camera/Eyepiece selection (for different FOVs)
        camera_label = QLabel("Camera/Eyepiece:")
        self.camera_combo = QComboBox()

        # Load user equipment first
        self._load_user_cameras_and_eyepieces()

        # Visual eyepieces with typical apparent FOV values
        self.camera_combo.addItem("--- EYEPIECES ---", None)
        self.camera_combo.addItem("32mm Eyepiece (52° AFOV)", {"type": "eyepiece", "focal_length": 32, "apparent_fov": 52})
        self.camera_combo.addItem("25mm Eyepiece (52° AFOV)", {"type": "eyepiece", "focal_length": 25, "apparent_fov": 52})
        self.camera_combo.addItem("20mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 20, "apparent_fov": 50})
        self.camera_combo.addItem("15mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 15, "apparent_fov": 50})
        self.camera_combo.addItem("10mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 10, "apparent_fov": 50})
        self.camera_combo.addItem("6mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 6, "apparent_fov": 50})

        # Camera presets
        for heading, cameras in CAMERA_PRESET_GROUPS:
            self.camera_combo.addItem(f"--- {heading} ---", None)
            for text, width, height, pixel_size in cameras:
                self.camera_combo.addItem(text, {"type": "camera", "sensor_width": width, "sensor_height": height,
                                                 "pixel_size": pixel_size})

        self.camera_combo.currentTextChanged.connect(self._on_camera_changed)

        # Barlow/Reducer selection
        barlow_label = QLabel("Barlow/Reducer:")
        self.barlow_combo = QComboBox()

        # Optical accessories
        self.barlow_combo.addItem("None (1.0x)", {"factor": 1.0, "type": "none"})

        # Load user barlows/reducers
        self._load_user_barlows()

        self.barlow_combo.addItem("--- BARLOWS ---", None)
        self.barlow_combo.addItem("1.25x Barlow", {"factor": 1.25, "type": "barlow"})
        self.barlow_combo.addItem("1.5x Barlow", {"factor": 1.5, "type": "barlow"})
        self.barlow_combo.addItem("2x Barlow", {"factor": 2.0, "type": "barlow"})
        self.barlow_combo.addItem("2.5x Barlow", {"factor": 2.5, "type": "barlow"})
        self.barlow_combo.addItem("3x Barlow", {"factor": 3.0, "type": "barlow"})
        self.barlow_combo.addItem("4x Barlow", {"factor": 4.0, "type": "barlow"})
        self.barlow_combo.addItem("5x Barlow", {"factor": 5.0, "type": "barlow"})
        self.barlow_combo.addItem("--- REDUCERS ---", None)
        self.barlow_combo.addItem("0.5x Reducer", {"factor": 0.5, "type": "reducer"})
        self.barlow_combo.addItem("0.6x Reducer", {"factor": 0.6, "type": "reducer"})
        self.barlow_combo.addItem("0.63x Reducer", {"factor": 0.63, "type": "reducer"})
        self.barlow_combo.addItem("0.67x Reducer", {"factor": 0.67, "type": "reducer"})
        self.barlow_combo.addItem("0.7x Reducer", {"factor": 0.7, "type": "reducer"})
        self.barlow_combo.addItem("0.75x Reducer", {"factor": 0.75, "type": "reducer"})
        self.barlow_combo.addItem("0.8x Reducer", {"factor": 0.8, "type": "reducer"})

        self.barlow_combo.currentTextChanged.connect(self._on_barlow_changed)

        # Mosaic panels (cameras only): columns x rows, overlapping
        self.mosaic_label = QLabel("Mosaic:")
        self.mosaic_columns = QSpinBox()
        self.mosaic_columns.setRange(1, 8)
        self.mosaic_columns.setToolTip("Panels across")
        self.mosaic_rows = QSpinBox()
        self.mosaic_rows.setRange(1, 8)
        self.mosaic_rows.setToolTip("Panels down")
        self.mosaic_times = QLabel("×")
        self.mosaic_overlap_label = QLabel("Overlap:")
        self.mosaic_overlap = QSpinBox()
        self.mosaic_overlap.setRange(0, 60)
        self.mosaic_overlap.setSuffix("%")
        self.mosaic_overlap.setValue(20)
        self.mosaic_overlap.setToolTip("How much neighboring panels overlap, for stitching")
        for spin in (self.mosaic_columns, self.mosaic_rows, self.mosaic_overlap):
            spin.valueChanged.connect(self._on_mosaic_changed)

        # Camera rotation (cameras only)
        self.rotation_label = QLabel("Rotation:")
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 359)
        self.rotation_spin.setWrapping(True)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.valueChanged.connect(self._on_mosaic_changed)

        # Arrange telescope controls
        telescope_layout.addWidget(telescope_label)
        telescope_layout.addWidget(self.telescope_combo)
        telescope_layout.addWidget(self.show_smart_telescopes)
        telescope_layout.addWidget(camera_label)
        telescope_layout.addWidget(self.camera_combo)
        telescope_layout.addWidget(barlow_label)
        telescope_layout.addWidget(self.barlow_combo)
        telescope_layout.addStretch()

        # Target List button
        target_menu = QMenu(self)

        self.add_target_action = QAction("Add to Target List", self)
        self.add_target_action.triggered.connect(self._add_to_target_list)
        target_menu.addAction(self.add_target_action)

        self.remove_target_action = QAction("Remove from Target List", self)
        self.remove_target_action.triggered.connect(self._remove_from_target_list)
        self.remove_target_action.setVisible(False)
        target_menu.addAction(self.remove_target_action)

        self.open_target_action = QAction("Open in Target List", self)
        self.open_target_action.triggered.connect(self._open_from_target_list)
        self.open_target_action.setVisible(False)
        target_menu.addAction(self.open_target_action)

        # Search for another target to frame (catalogs, then SIMBAD)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search target...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(190)
        self.search_edit.setToolTip("Another object to frame: a catalog name (M 31, NGC 7000, Sh2-129),\n"
                                    "a common name (Andromeda Galaxy) or coordinates (RA Dec).\n"
                                    "Not in the catalogs? It's looked up on SIMBAD.")
        self.search_edit.returnPressed.connect(lambda: self._search_target(self.search_edit.text()))
        self._search_names = None  # Suggestion text -> (catalog, designation), loaded on first use
        self.search_edit.installEventFilter(self)
        telescope_layout.addWidget(self.search_edit)

        target_button = QToolButton()
        target_button.setText("Target List")
        target_button.setMenu(target_menu)
        target_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        telescope_layout.addWidget(target_button)

        # Add NINA button if integration is enabled
        if NINAIntegration.is_enabled():
            # Built when it opens: the frame as shown, or each mosaic panel
            self.nina_menu = QMenu(self)
            self.nina_menu.aboutToShow.connect(self._fill_nina_menu)

            nina_button = QToolButton()
            nina_button.setText("NINA")
            nina_button.setToolTip("Send to NINA")
            nina_button.setMenu(self.nina_menu)
            nina_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            telescope_layout.addWidget(nina_button)

        layout.addLayout(telescope_layout)

        # Overlay row: what to show, mosaic panels, and back to the target
        overlay_layout = QHBoxLayout()
        overlay_layout.addWidget(QLabel("Show:"))
        overlay_layout.addWidget(self.show_telescope_fov)
        overlay_layout.addWidget(self.show_horizon)
        overlay_layout.addWidget(self.show_meridian)
        overlay_layout.addSpacing(24)
        overlay_layout.addWidget(self.mosaic_label)
        overlay_layout.addWidget(self.mosaic_columns)
        overlay_layout.addWidget(self.mosaic_times)
        overlay_layout.addWidget(self.mosaic_rows)
        overlay_layout.addSpacing(8)
        overlay_layout.addWidget(self.mosaic_overlap_label)
        overlay_layout.addWidget(self.mosaic_overlap)
        overlay_layout.addSpacing(8)
        overlay_layout.addWidget(self.rotation_label)
        overlay_layout.addWidget(self.rotation_spin)
        self.reset_framing_button = QPushButton("Reset")
        self.reset_framing_button.setToolTip("Back to a single frame on the target, not rotated")
        self.reset_framing_button.clicked.connect(self._reset_framing)
        overlay_layout.addWidget(self.reset_framing_button)
        overlay_layout.addStretch()
        self.move_frame_button = QPushButton("Move Frame Here")
        self.move_frame_button.setToolTip("Put the frame (or mosaic) where the view's crosshair is -\n"
                                          "pan the sky to where you want it first")
        self.move_frame_button.clicked.connect(self._move_frame_here)
        overlay_layout.addWidget(self.move_frame_button)
        center_button = QPushButton("Center on Target")
        center_button.setToolTip("Go back to the target, with the frame on it again, zoomed to fit")
        center_button.clicked.connect(self._center_on_target)
        overlay_layout.addWidget(center_button)
        self.save_session_button = QPushButton("Save to Session...")
        self.save_session_button.setToolTip("Keep this frame or mosaic with one of this object's sessions\n"
                                            "(or a new planned one) - its panels, progress and NINA actions\n"
                                            "are then in the session's details")
        self.save_session_button.clicked.connect(self._save_to_session)
        overlay_layout.addWidget(self.save_session_button)
        layout.addLayout(overlay_layout)

        # Time: any night, any time - the horizon, meridian and Night View follow it
        time_layout = QHBoxLayout()
        self.night_view_check = QCheckBox("Night View")
        self.night_view_check.setToolTip(
            "See the sky from the ground: the horizon stays put (zenith up) and the target\n"
            "and stars move across it as you change the time - with the target's path for the night")
        self.night_view_check.toggled.connect(self._on_night_view_toggled)
        time_layout.addWidget(self.night_view_check)
        time_layout.addSpacing(12)
        self.night_label = QLabel("Night of:")
        time_layout.addWidget(self.night_label)
        self.night_date = QDateEdit()
        self.night_date.setCalendarPopup(True)
        self.night_date.setDisplayFormat("ddd MMM d, yyyy")
        self.night_date.setToolTip("The night to show (the evening's date)")
        self.night_date.dateChanged.connect(self._on_night_date_changed)
        time_layout.addWidget(self.night_date)
        self.time_slider = NightSlider()
        self.time_slider.moved.connect(self._on_time_slider)
        time_layout.addWidget(self.time_slider, 1)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(150)
        time_layout.addWidget(self.time_label)
        self.now_button = QPushButton("Now")
        self.now_button.setToolTip("Back to the current time, following the clock")
        self.now_button.clicked.connect(self._go_now)
        time_layout.addWidget(self.now_button)
        self.play_button = QPushButton("▶ Play")
        self.play_button.setToolTip("Watch the night go by (an hour in about 2 seconds)")
        self.play_button.clicked.connect(self._toggle_play)
        time_layout.addWidget(self.play_button)
        layout.addLayout(time_layout)
        self.time_widgets = (self.night_view_check, self.night_label, self.night_date, self.time_slider,
                             self.time_label, self.now_button, self.play_button)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(150)
        self.play_timer.timeout.connect(self._play_step)

        # Initialize web view as None initially - we'll create it safely later
        self.web_view = None
        self.web_view_error = None

        # Create a placeholder widget for the web view
        self.web_placeholder = QLabel("Loading Aladin Lite...")
        self.web_placeholder.setAlignment(Qt.AlignCenter)
        themed_style(self.web_placeholder, lambda: f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['text']}; font-size: {font_px(14)}; }}")
        self.web_placeholder.setMinimumSize(400, 300)

        layout.addWidget(self.web_placeholder, 1)

        # Create a horizontal layout for the bottom controls
        bottom_layout = QHBoxLayout()

        # Add FOV information display
        self.fov_info_label = QLabel()
        self.fov_info_label.setWordWrap(True)
        themed_style(self.fov_info_label, lambda: f"font-size: {font_size(10)};")
        bottom_layout.addWidget(self.fov_info_label, 1)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom_layout.addWidget(close_button)

        # Add the bottom layout to the main layout
        layout.addLayout(bottom_layout)

        # Create central widget and set layout for QMainWindow
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.fallback_button = None  # Track fallback button to avoid duplicates
        self._renderer_crashed = False  # Set by _on_render_process_terminated; suppresses the generic load-failed message

        # Add loading timeout
        self.loading_timeout = QTimer()
        self.loading_timeout.timeout.connect(self._handle_loading_timeout)
        self.loading_timeout.setSingleShot(True)

        # The horizon and meridian move across the sky; redraw them now and then
        self.sky_timer = QTimer(self)
        self.sky_timer.setInterval(SKY_REFRESH_MS)
        self.sky_timer.timeout.connect(self._on_sky_timer)

        # Load persistent settings before creating web view
        self._load_aladin_settings()
        if framing:
            self._apply_framing(framing)
        if (self.show_horizon.isChecked() or self.show_meridian.isChecked()) and not self._load_observer_location():
            # Saved on, but there's no location (any more)
            for check in (self.show_horizon, self.show_meridian):
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
        self._update_mosaic_controls()
        for widget in self.time_widgets:
            widget.setEnabled(False)
        QTimer.singleShot(0, self._setup_time)  # Once the window shows: the night takes a moment to work out
        self._update_fov_info()

        # Update target list button state
        self._update_target_list_button()

        theme_manager().theme_changed.connect(self._on_theme_changed)

        # Defer web view creation to avoid initialization crashes
        QTimer.singleShot(100, self._create_web_view_safely)

        logger.debug(f"Opened FOV Simulator window with default FOV: {self.default_fov:.2f}°")

    def _create_web_view_safely(self):
        """Safely create the web view with error handling"""
        try:
            logger.debug("Creating web view safely...")

            # Try to create the web view
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
            except ImportError as ie:
                raise Exception(f"QWebEngineView not available: {ie}")

            self.web_view = QWebEngineView()
            self.web_view.setMinimumSize(400, 300)
            # No browser menu (Back, Reload, Inspect...) - it isn't a web page to the user
            self.web_view.setContextMenuPolicy(Qt.NoContextMenu)

            # Enable WebGL and hardware acceleration for Aladin Lite
            try:
                from PySide6.QtWebEngineCore import QWebEngineSettings
                settings = self.web_view.settings()
                # Enable WebGL - critical for Aladin Lite rendering
                settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
                # Our page is set with an https base URL, but loads Aladin and its sky tiles remotely
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
                logger.debug("WebGL, JavaScript, and hardware acceleration enabled for Aladin window")
            except Exception as e:
                logger.warning(f"Could not enable WebGL settings: {e}")

            # Add load progress and error handling
            self.web_view.loadStarted.connect(self._on_load_started)
            self.web_view.loadProgress.connect(self._on_load_progress)
            self.web_view.loadFinished.connect(self._on_load_finished)
            self.web_view.page().renderProcessTerminated.connect(self._on_render_process_terminated)
            self.web_view.page().titleChanged.connect(self._on_page_report)

            # Load the viewer — placeholder stays visible until _on_load_finished swaps it out
            self._load_viewer_page()
            logger.debug("Web view created successfully")

        except Exception as e:
            logger.error(f"Failed to create web view safely: {e}")
            self.web_view_error = str(e)

            # Update placeholder to show error and offer browser fallback
            if self.web_placeholder:
                self.web_placeholder.setText(f"Failed to load Aladin Lite\nError: {str(e)}\n\nClick below to open in browser instead.")
                themed_style(self.web_placeholder, lambda: f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: {font_px(12)}; }}")

                # Add a button to open in browser as fallback
                self._add_browser_fallback_button()

    def _add_browser_fallback_button(self):
        """Add a button to open Aladin Lite in the default browser"""
        try:
            # Don't add button if it already exists
            if self.fallback_button is not None:
                return

            # Find the central widget and its layout
            central_widget = self.centralWidget()
            if central_widget and central_widget.layout():
                main_layout = central_widget.layout()

                # Create a fallback button
                self.fallback_button = QPushButton("Open Aladin Lite in Browser")
                themed_style(self.fallback_button, lambda: f"QPushButton {{ background-color: {COLORS['success']}; color: {contrast_text(COLORS['success'])}; font-weight: bold; margin: 10px; padding: 8px; }}")
                self.fallback_button.clicked.connect(self._open_in_browser)

                # Insert before the bottom controls (last item should be the bottom layout)
                main_layout.insertWidget(main_layout.count() - 1, self.fallback_button)
                logger.debug("Added browser fallback button")
        except Exception as e:
            logger.error(f"Failed to add browser fallback button: {e}")

    def _open_in_browser(self):
        """Open the current view in Aladin Lite in the default browser"""
        if self.web_view is not None and self.page_loaded:
            self.web_view.page().runJavaScript("ccView()", self._open_view_in_browser)
        else:
            self._open_view_in_browser(None)

    def _open_view_in_browser(self, view):
        """view: the page's ccView() JSON ({"ra", "dec", "fov"}), or None for the target."""
        try:
            ra, dec = self._target_position()
            fov = self.view_fov or self.current_fov or self.default_fov
            if view:
                current = json.loads(view)
                ra, dec, fov = current["ra"], current["dec"], current["fov"]
            target = f"{ra} {dec}" if ra is not None and dec is not None else self.data.get('name', 'M1')
            browser_url = ALADIN_BROWSER_URL + "?" + urllib.parse.urlencode(
                {"target": target, "fov": f"{fov:.4f}", "survey": SURVEY})

            logger.debug(f"Opening Aladin Lite in browser: {browser_url}")
            open_url(browser_url)

            # Show a message to the user
            QMessageBox.information(self, "Opened in Browser",
                                  f"Aladin Lite has been opened in your default browser for {self.data.get('name', 'the selected object')}.")

        except Exception as e:
            logger.error(f"Failed to open Aladin Lite in browser: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open Aladin Lite in browser: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event with proper cleanup"""
        try:
            logger.debug("Cleaning up Aladin Lite window")
            self.sky_timer.stop()
            self.play_timer.stop()
            # Stop any pending JavaScript operations
            if hasattr(self, 'web_view') and self.web_view:
                self.web_view.stop()
                # Clear the web view content
                self.web_view.setHtml("")
            event.accept()
        except Exception as e:
            logger.warning(f"Error during Aladin window cleanup: {e}")
            event.accept()  # Always accept to prevent hanging

    def _load_telescopes(self):
        """Load active user telescopes from database"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length, is_active
                    FROM usertelescopes
                    WHERE focal_length IS NOT NULL AND focal_length > 0 AND is_active = 1
                    ORDER BY name ASC
                """)

                telescopes = cursor.fetchall()
                self.telescopes = []

                for telescope_id, name, aperture, focal_length, is_active in telescopes:
                    telescope_data = {
                        'id': telescope_id,
                        'name': name,
                        'aperture': aperture,
                        'focal_length': focal_length,
                        'is_active': is_active
                    }
                    self.telescopes.append(telescope_data)

                    # Add to combo box
                    display_name = f"{name} ({focal_length}mm f/{focal_length/aperture:.1f})" if aperture else f"{name} ({focal_length}mm)"
                    self.telescope_combo.addItem(display_name, telescope_data)

                logger.debug(f"Loaded {len(telescopes)} active telescopes with focal length data")

        except Exception as e:
            logger.error(f"Error loading telescopes: {str(e)}")

    def _load_user_cameras_and_eyepieces(self):
        """Load user cameras and eyepieces from database to populate camera combo"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Load user cameras
                cursor.execute("""
                    SELECT id, name, sensor_width, sensor_height, pixel_size
                    FROM userequipment
                    WHERE equipment_type = 'camera' AND COALESCE(is_active, 1) = 1
                    ORDER BY name ASC
                """)
                user_cameras = cursor.fetchall()

                # Load user eyepieces
                cursor.execute("""
                    SELECT id, name, focal_length, apparent_fov
                    FROM userequipment
                    WHERE equipment_type = 'eyepiece' AND COALESCE(is_active, 1) = 1
                    ORDER BY name ASC
                """)
                user_eyepieces = cursor.fetchall()

                # Add user cameras section if there are any
                if user_cameras:
                    self.camera_combo.addItem("--- YOUR CAMERAS ---", None)
                    for eq_id, name, sensor_width, sensor_height, pixel_size in user_cameras:
                        if sensor_width and sensor_height:
                            display_text = f"{name} ({sensor_width}x{sensor_height}mm)"
                            self.camera_combo.addItem(display_text, {
                                "type": "camera",
                                "sensor_width": sensor_width,
                                "sensor_height": sensor_height,
                                "pixel_size": pixel_size,
                                "source": "user",
                                "id": eq_id
                            })

                # Add user eyepieces section if there are any
                if user_eyepieces:
                    self.camera_combo.addItem("--- YOUR EYEPIECES ---", None)
                    for eq_id, name, focal_length, apparent_fov in user_eyepieces:
                        if focal_length and apparent_fov:
                            display_text = f"{name} ({focal_length}mm, {apparent_fov}\u00b0 AFOV)"
                            self.camera_combo.addItem(display_text, {
                                "type": "eyepiece",
                                "focal_length": focal_length,
                                "apparent_fov": apparent_fov,
                                "source": "user",
                                "id": eq_id
                            })

                logger.debug(f"Loaded {len(user_cameras)} user cameras and {len(user_eyepieces)} user eyepieces")

        except Exception as e:
            logger.error(f"Error loading user cameras/eyepieces: {str(e)}")

    def _load_user_barlows(self):
        """Load user barlows and reducers from database to populate barlow combo"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Load user barlows and reducers
                cursor.execute("""
                    SELECT id, name, factor, equipment_type
                    FROM userequipment
                    WHERE equipment_type IN ('barlow', 'reducer') AND COALESCE(is_active, 1) = 1
                    ORDER BY equipment_type, factor DESC
                """)
                user_barlows = cursor.fetchall()

                # Add user barlows/reducers section if there are any
                if user_barlows:
                    self.barlow_combo.addItem("--- YOUR EQUIPMENT ---", None)
                    for eq_id, name, factor, eq_type in user_barlows:
                        if factor:
                            display_text = f"{name} ({factor}x)"
                            self.barlow_combo.addItem(display_text, {
                                "type": eq_type,
                                "factor": factor,
                                "source": "user",
                                "id": eq_id
                            })

                logger.debug(f"Loaded {len(user_barlows)} user barlows/reducers")

        except Exception as e:
            logger.error(f"Error loading user barlows/reducers: {str(e)}")

    def _settings_widgets(self):
        return (self.telescope_combo, self.show_telescope_fov, self.show_smart_telescopes,
                self.camera_combo, self.barlow_combo, self.show_horizon, self.show_meridian,
                self.mosaic_columns, self.mosaic_rows, self.mosaic_overlap, self.rotation_spin)

    def _load_aladin_settings(self):
        """Load persistent Aladin Lite settings from QSettings"""
        # Block signals during loading to prevent save being triggered
        for widget in self._settings_widgets():
            widget.blockSignals(True)
        try:
            settings = QSettings("CosmosCollection", "AladinLite")

            # Load show smart telescopes checkbox and populate combo if needed
            # (must happen before telescope name restore so smart entries are in the combo)
            show_smart = settings.value("show_smart_telescopes", False, type=bool)
            self.show_smart_telescopes.setChecked(show_smart)
            if show_smart:
                self._add_smart_telescopes()
            logger.debug(f"Restored show smart telescopes: {show_smart}")

            # Load telescope selection
            saved_telescope_name = settings.value("telescope_name", None)
            if saved_telescope_name:
                # Find the telescope in the combo box
                for i in range(self.telescope_combo.count()):
                    data = self.telescope_combo.itemData(i)
                    if data and data.get('name') == saved_telescope_name:
                        self.telescope_combo.setCurrentIndex(i)
                        self.selected_telescope = data
                        logger.debug(f"Restored telescope selection: {saved_telescope_name}")
                        break

            # Load show telescope FOV checkbox
            self.show_telescope_fov.setChecked(settings.value("show_telescope_fov", False, type=bool))

            # Load camera/eyepiece selection
            saved_camera_text = settings.value("camera_eyepiece", None)
            if saved_camera_text:
                index = self.camera_combo.findText(saved_camera_text)
                if index < 0:
                    # Saved before its sensor size was corrected: the same camera, new size
                    key = self._camera_key(saved_camera_text)
                    index = next((i for i in range(self.camera_combo.count())
                                  if self.camera_combo.itemData(i) is not None
                                  and self._camera_key(self.camera_combo.itemText(i)) == key), -1)
                if index >= 0:
                    self.camera_combo.setCurrentIndex(index)
                    logger.debug(f"Restored camera/eyepiece selection: {saved_camera_text}")

            # Load barlow/reducer selection
            saved_barlow_text = settings.value("barlow_reducer", None)
            if saved_barlow_text:
                index = self.barlow_combo.findText(saved_barlow_text)
                if index >= 0:
                    self.barlow_combo.setCurrentIndex(index)
                    logger.debug(f"Restored barlow/reducer selection: {saved_barlow_text}")

            # Sky overlays
            self.show_horizon.setChecked(settings.value("show_horizon", False, type=bool))
            self.show_meridian.setChecked(settings.value("show_meridian", False, type=bool))

            # Mosaic
            self.mosaic_columns.setValue(settings.value("mosaic_columns", 1, type=int))
            self.mosaic_rows.setValue(settings.value("mosaic_rows", 1, type=int))
            self.mosaic_overlap.setValue(settings.value("mosaic_overlap", 20, type=int))
            self.rotation_spin.setValue(settings.value("camera_rotation", 0, type=int) % 360)

            logger.debug("Aladin Lite settings loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Aladin settings: {str(e)}")
        finally:
            for widget in self._settings_widgets():
                widget.blockSignals(False)

    @staticmethod
    def _camera_key(text):
        """A camera's name without its size: 'ASI462MC (5.6x3.2mm) (Seestar S50)'
        -> ('ASI462MC', ('Seestar S50)',))."""
        parts = text.split(" (")
        return parts[0], tuple(parts[2:])

    def _save_aladin_settings(self):
        """Save persistent Aladin Lite settings to QSettings"""
        try:
            settings = QSettings("CosmosCollection", "AladinLite")

            # Save telescope selection
            telescope_data = self.telescope_combo.currentData()
            if telescope_data:
                settings.setValue("telescope_name", telescope_data.get('name'))
            else:
                settings.setValue("telescope_name", None)

            settings.setValue("show_telescope_fov", self.show_telescope_fov.isChecked())
            settings.setValue("show_smart_telescopes", self.show_smart_telescopes.isChecked())
            settings.setValue("camera_eyepiece", self.camera_combo.currentText())
            settings.setValue("barlow_reducer", self.barlow_combo.currentText())
            settings.setValue("show_horizon", self.show_horizon.isChecked())
            settings.setValue("show_meridian", self.show_meridian.isChecked())
            settings.setValue("mosaic_columns", self.mosaic_columns.value())
            settings.setValue("mosaic_rows", self.mosaic_rows.value())
            settings.setValue("mosaic_overlap", self.mosaic_overlap.value())
            settings.setValue("camera_rotation", self.rotation_spin.value())

            logger.debug("Aladin Lite settings saved successfully")

        except Exception as e:
            logger.error(f"Error saving Aladin settings: {str(e)}")

    def _add_smart_telescopes(self):
        """Append smart telescope entries to the telescope combo box."""
        self.telescope_combo.addItem("--- SMART TELESCOPES ---", None)
        for st in self.SMART_TELESCOPES:
            display = f"{st['name']} ({st['focal_length']}mm f/{st['focal_length']/st['aperture']:.1f})"
            self.telescope_combo.addItem(display, dict(st))

    def _remove_smart_telescopes(self):
        """Remove all smart telescope entries (and separator) from the telescope combo box."""
        i = 0
        while i < self.telescope_combo.count():
            text = self.telescope_combo.itemText(i)
            data = self.telescope_combo.itemData(i)
            if text == "--- SMART TELESCOPES ---" or (data and data.get('smart_telescope')):
                self.telescope_combo.removeItem(i)
            else:
                i += 1

    def _on_show_smart_telescopes_toggled(self, checked):
        """Handle Show Smart Telescopes checkbox toggle."""
        if checked:
            self._add_smart_telescopes()
        else:
            # If a smart telescope is currently selected, reset to Default View first
            current_data = self.telescope_combo.currentData()
            if current_data and current_data.get('smart_telescope'):
                self.telescope_combo.setCurrentIndex(0)
            self._remove_smart_telescopes()
        self._save_aladin_settings()

    def _on_telescope_changed(self):
        """Handle telescope selection change"""
        current_data = self.telescope_combo.currentData()
        if current_data:
            self.selected_telescope = current_data
            logger.debug(f"Selected telescope: {current_data['name']} ({current_data['focal_length']}mm)")

            if current_data.get('smart_telescope'):
                camera_text = current_data.get('camera_text')
                if camera_text:
                    idx = self.camera_combo.findText(camera_text)
                    if idx >= 0:
                        self.camera_combo.blockSignals(True)
                        self.camera_combo.setCurrentIndex(idx)
                        self.camera_combo.blockSignals(False)
                self.barlow_combo.blockSignals(True)
                self.barlow_combo.setCurrentIndex(0)  # "None (1.0x)"
                self.barlow_combo.blockSignals(False)
            else:
                # Auto-select equipment associated with this telescope
                self._select_telescope_equipment(current_data.get('id'))
        else:
            self.selected_telescope = None
            logger.debug("Selected default view")

        self._save_aladin_settings()
        self._update_aladin_view()

    def _select_telescope_equipment(self, telescope_id):
        """Auto-select equipment associated with the given telescope"""
        if not telescope_id:
            return

        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Get all equipment IDs associated with this telescope
                cursor.execute("""
                    SELECT e.id, e.equipment_type
                    FROM userequipment e
                    JOIN telescope_equipment te ON e.id = te.equipment_id
                    WHERE te.telescope_id = ? AND COALESCE(e.is_active, 1) = 1
                    ORDER BY e.name ASC
                """, (telescope_id,))

                equipment = cursor.fetchall()

                # Find first camera or eyepiece and select it
                camera_selected = False
                barlow_selected = False

                for eq_id, eq_type in equipment:
                    # Select first camera/eyepiece in the camera combo
                    if not camera_selected and eq_type in ('camera', 'eyepiece'):
                        for i in range(self.camera_combo.count()):
                            data = self.camera_combo.itemData(i)
                            if data and data.get('source') == 'user' and data.get('id') == eq_id:
                                self.camera_combo.blockSignals(True)
                                self.camera_combo.setCurrentIndex(i)
                                self.camera_combo.blockSignals(False)
                                camera_selected = True
                                logger.debug(f"Auto-selected camera/eyepiece ID {eq_id} for telescope")
                                break

                    # Select first barlow/reducer in the barlow combo
                    if not barlow_selected and eq_type in ('barlow', 'reducer'):
                        for i in range(self.barlow_combo.count()):
                            data = self.barlow_combo.itemData(i)
                            if data and data.get('source') == 'user' and data.get('id') == eq_id:
                                self.barlow_combo.blockSignals(True)
                                self.barlow_combo.setCurrentIndex(i)
                                self.barlow_combo.blockSignals(False)
                                barlow_selected = True
                                logger.debug(f"Auto-selected barlow/reducer ID {eq_id} for telescope")
                                break

        except Exception as e:
            logger.error(f"Error selecting telescope equipment: {str(e)}")

    def _on_camera_changed(self):
        """Handle camera/sensor selection change"""
        self._save_aladin_settings()
        self._update_aladin_view()

    def _on_barlow_changed(self):
        """Handle barlow/reducer selection change"""
        self._save_aladin_settings()
        self._update_aladin_view()

    def _on_show_fov_toggled(self):
        """Handle show telescope FOV checkbox toggle"""
        self._save_aladin_settings()
        self._update_aladin_view()

    def _calculate_telescope_fov(self):
        """Calculate telescope FOV based on selected telescope and camera/eyepiece"""
        if not self.selected_telescope:
            return None

        telescope_fl = self.selected_telescope['focal_length']  # mm
        telescope_aperture = self.selected_telescope.get('aperture', 100)  # mm

        # Get barlow/reducer factor
        barlow_data = self.barlow_combo.currentData()
        barlow_factor = 1.0  # Default no change
        if barlow_data and 'factor' in barlow_data:
            barlow_factor = barlow_data['factor']

        # Apply barlow/reducer to effective focal length
        effective_fl = telescope_fl * barlow_factor

        camera_data = self.camera_combo.currentData()

        if not camera_data or camera_data is None:
            return None

        if camera_data.get('type') == 'eyepiece':
            # Visual observation with eyepiece
            eyepiece_fl = camera_data['focal_length']  # mm
            apparent_fov = camera_data['apparent_fov']  # degrees

            # Calculate magnification using effective focal length
            magnification = effective_fl / eyepiece_fl

            # True FOV = Apparent FOV / Magnification
            true_fov_deg = apparent_fov / magnification
            true_fov_arcmin = true_fov_deg * 60

            barlow_text = f" with {barlow_factor}x" if barlow_factor != 1.0 else ""
            logger.debug(f"Eyepiece FOV calculation: {eyepiece_fl}mm eyepiece{barlow_text}, {apparent_fov}° AFOV, {magnification:.1f}x mag, {true_fov_arcmin:.1f}' true FOV")

            barlow_details = f" + {barlow_factor}x" if barlow_factor != 1.0 else ""
            return {
                'width_arcmin': true_fov_arcmin,
                'height_arcmin': true_fov_arcmin,
                'type': 'visual',
                'details': f"{eyepiece_fl}mm eyepiece{barlow_details}, {magnification:.0f}x mag"
            }

        elif camera_data.get('type') == 'camera':
            # Camera sensor
            sensor_width = camera_data['sensor_width']  # mm
            sensor_height = camera_data['sensor_height']  # mm

            # FOV = 2 * arctan(sensor_size / (2 * effective_focal_length)) * (180/π) * 60 (arcmin)
            fov_width_rad = 2 * math.atan(sensor_width / (2 * effective_fl))
            fov_height_rad = 2 * math.atan(sensor_height / (2 * effective_fl))

            fov_width_arcmin = fov_width_rad * (180 / math.pi) * 60
            fov_height_arcmin = fov_height_rad * (180 / math.pi) * 60

            barlow_text = f" with {barlow_factor}x" if barlow_factor != 1.0 else ""
            logger.debug(f"Camera FOV calculation: {sensor_width}x{sensor_height}mm sensor, {effective_fl}mm effective FL{barlow_text}, FOV={fov_width_arcmin:.1f}'x{fov_height_arcmin:.1f}'")

            barlow_details = f" + {barlow_factor}x" if barlow_factor != 1.0 else ""
            pixel_size = camera_data.get('pixel_size')
            return {
                'width_arcmin': fov_width_arcmin,
                'height_arcmin': fov_height_arcmin,
                'type': 'camera',
                'details': f"{sensor_width}×{sensor_height}mm sensor{barlow_details}",
                # Arcseconds per pixel, when the camera's pixel size is known
                'pixel_scale': 206.265 * pixel_size / effective_fl if pixel_size else None,
            }

        return None

    # ------------------------------------------------------------------
    # The viewer page and what's drawn on it
    # ------------------------------------------------------------------

    def _target_position(self):
        """(RA, Dec) of the target in degrees, or (None, None) when only its name is known."""
        ra, dec = self.data.get('ra_deg'), self.data.get('dec_deg')
        if ra is None or dec is None:
            return None, None
        return float(ra), float(dec)

    def _mosaic(self):
        """(columns, rows, overlap 0-1) - 1 x 1 unless a camera frame is shown."""
        camera = self.camera_combo.currentData()
        if not (camera and camera.get('type') == 'camera'):
            return 1, 1, 0.0
        return self.mosaic_columns.value(), self.mosaic_rows.value(), self.mosaic_overlap.value() / 100.0

    def _rotation(self):
        """Camera angle in degrees (north through east) - 0 unless a camera frame is shown."""
        camera = self.camera_combo.currentData()
        if not (camera and camera.get('type') == 'camera'):
            return 0
        return self.rotation_spin.value()

    def _frame_overlay(self):
        """The telescope's field of view (and mosaic panels) for the page:
        {"shapes", "labels", "size": (width, height) degrees}, or None."""
        if not (self.selected_telescope and self.show_telescope_fov.isChecked()):
            return None
        fov = self._calculate_telescope_fov()
        ra, dec = self.frame_center or self._target_position()
        if not fov or ra is None:
            return None
        width, height = fov['width_arcmin'] / 60.0, fov['height_arcmin'] / 60.0
        if fov['type'] == 'visual':
            return {"shapes": [{"type": "circle", "ra": ra, "dec": dec, "radius": width / 2.0,
                                "color": EYEPIECE_COLOR, "width": 2}],
                    "labels": [], "size": (width, height), "extent": (width / 2.0, width / 2.0)}
        columns, rows, overlap = self._mosaic()
        rotation = self._rotation()
        total_width, total_height = mosaic_size(width, height, columns, rows, overlap)
        shapes, labels = [], []
        if columns * rows == 1:
            shapes.append({"type": "polygon", "points": frame_corners(ra, dec, width, height, rotation),
                           "color": CAMERA_COLOR, "width": 2})
        else:
            for panel in mosaic_panels(ra, dec, width, height, columns, rows, overlap, rotation):
                shapes.append({"type": "polygon", "points": panel["outline"], "color": CAMERA_COLOR, "width": 2})
                labels.append({"text": str(panel["number"]), "ra": panel["center"][0],
                               "dec": panel["center"][1], "color": CAMERA_COLOR,
                               "size": min(width, height)})  # Hidden when the panel is tiny on screen
        if rotation:
            # Which way is up, once turned
            shapes.append({"type": "polygon", "points": top_marker(ra, dec, total_width, total_height, rotation),
                           "color": CAMERA_COLOR, "fill": CAMERA_COLOR, "width": 1})
        # A handle on a stem above the top edge - drag it to turn the frame
        stem = total_height / 2.0 + total_height * 0.22
        base = frame_to_radec(ra, dec, 0.0, total_height / 2.0, rotation)
        knob = frame_to_radec(ra, dec, 0.0, stem, rotation)
        shapes.append({"type": "polyline", "points": [list(base), list(knob)], "color": CAMERA_COLOR, "width": 1})
        # Half extents on the sky (east-west, north-south) of the turned frame and its handle
        angle = math.radians(rotation)
        points = [(right, up) for right in (-total_width / 2, total_width / 2)
                  for up in (-total_height / 2, total_height / 2)] + [(0.0, stem)]
        extent = (max(abs(-r * math.cos(angle) + u * math.sin(angle)) for r, u in points),
                  max(abs(r * math.sin(angle) + u * math.cos(angle)) for r, u in points))
        return {"shapes": shapes, "labels": labels,
                "handle": {"center": [ra, dec], "knob": [round(knob[0], 6), round(knob[1], 6)]},
                "size": rotated_size(total_width, total_height, rotation), "extent": extent}

    def _fit_fov(self, overlay=None):
        """View width (degrees) that fits the field of view (and the rotation handle)
        with some sky around it, in the view's shape - or the default view for the object."""
        overlay = overlay if overlay is not None else self._frame_overlay()
        if not overlay:
            return self.default_fov
        # Until the page shows, the placeholder holds its place in the window
        view = self.web_view if self.web_view is not None and self.web_view.isVisible() else self.web_placeholder
        aspect = view.width() / max(view.height(), 1) if view is not None and view.height() > 50 else 16 / 9
        half_x, half_y = overlay["extent"]
        return 2.0 * max(half_x, half_y * aspect) * 1.2

    def _load_viewer_page(self):
        """Load our Aladin Lite page centered on the target (or a moved frame)."""
        ra, dec = self._target_position()
        name = self.data.get('name') or self.data.get('dsodetailid') or "M1"
        view_ra, view_dec = self.frame_center or (ra, dec)
        self.current_fov = self._fit_fov()
        config = {
            "ra": ra, "dec": dec,
            "target": f"{view_ra} {view_dec}" if view_ra is not None else str(name),
            "fov": self.current_fov,
            "survey": SURVEY,
            "colors": page_colors(),
        }
        self.page_loaded = False
        self.web_view.setHtml(viewer_page(config), QUrl(ALADIN_BASE_URL))
        self._test_connectivity_async()

    def _run_js(self, code, callback=None):
        """Run JavaScript in the viewer page once it has loaded (before that,
        _on_load_finished pushes everything anyway)."""
        if self.web_view is None or not self.page_loaded:
            return
        if callback is None:
            self.web_view.page().runJavaScript(code)
        else:
            self.web_view.page().runJavaScript(code, callback)

    def _push_frames(self, zoom=False):
        """Draw the telescope's field of view; zoom=True also fits the view to it."""
        overlay = self._frame_overlay()
        payload = ({"shapes": overlay["shapes"], "labels": overlay["labels"], "handle": overlay.get("handle")}
                   if overlay else None)
        self._run_js(f"ccSetFrames({json.dumps(payload)})")
        if zoom and overlay:
            self.current_fov = self._fit_fov(overlay)
            self._run_js(f"ccZoom({self.current_fov})")
        self._update_mosaic_controls()
        self._update_fov_info()

    def _update_aladin_view(self, preserve_target=True):
        """Redraw the field of view after an equipment or overlay change,
        zooming to fit it (the view keeps its position)."""
        self._push_frames(zoom=True)

    def _on_mosaic_changed(self, *_args):
        if self._rotating:
            self._push_frames()  # Turning with the handle: no zooming while dragging
            return
        self._save_aladin_settings()
        self._push_frames(zoom=True)

    def _reset_framing(self):
        """Back to one frame on the target, not turned, zoomed to fit."""
        for spin, value in ((self.mosaic_columns, 1), (self.mosaic_rows, 1),
                            (self.mosaic_overlap, 20), (self.rotation_spin, 0)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.frame_center = None
        self._save_aladin_settings()
        self._push_frames()
        self.current_fov = self._fit_fov()
        self._go_to_target(self.current_fov)

    def _update_mosaic_controls(self):
        """Mosaic panels need a telescope and a camera, with the field of view shown."""
        camera = self.camera_combo.currentData()
        enabled = bool(self.selected_telescope and self.show_telescope_fov.isChecked()
                       and camera and camera.get('type') == 'camera')
        for widget in (self.mosaic_label, self.mosaic_columns, self.mosaic_times, self.mosaic_rows):
            widget.setEnabled(enabled)
            widget.setToolTip("" if enabled else
                              "Choose a telescope and a camera, and show the Telescope FOV, to plan a mosaic")
        if enabled:
            self.mosaic_columns.setToolTip("Panels across")
            self.mosaic_rows.setToolTip("Panels down")
        several = enabled and self.mosaic_columns.value() * self.mosaic_rows.value() > 1
        self.mosaic_overlap_label.setEnabled(several)
        self.mosaic_overlap.setEnabled(several)
        self.rotation_label.setEnabled(enabled)
        self.rotation_spin.setEnabled(enabled)
        self.save_session_button.setEnabled(enabled)
        self.reset_framing_button.setEnabled(enabled)
        self.move_frame_button.setEnabled(bool(self.selected_telescope and self.show_telescope_fov.isChecked()
                                               and camera is not None))
        self.rotation_spin.setToolTip(
            "Camera angle: where the top of the frame points, from north through east\n"
            "(the rotation or position angle a plate solve reports).\n"
            "Or drag the round handle above the frame (Shift: 15° steps)." if enabled else
            "Choose a telescope and a camera, and show the Telescope FOV, to turn the frame")

    def _center_on_target(self):
        """Back to the target - with the frame on it again - zoomed to fit."""
        if self.frame_center is not None:
            self.frame_center = None
            self._push_frames()
        self.current_fov = self._fit_fov()
        self._go_to_target(self.current_fov)

    def _go_to_target(self, fov):
        """Center the view on the target at fov - level with the ground in Night View."""
        ra, dec = self._target_position()
        if ra is None:
            self._run_js(f"ccCenter({fov})")  # Only its name: Aladin looks it up
            return
        self._run_js(f"ccSetView({ra}, {dec}, {self._view_rotation(ra, dec)}, {fov})")

    def _view_rotation(self, ra, dec):
        """0 (north up) - or in Night View, whatever puts the zenith up there."""
        if not (self.night_view_check.isChecked() and self.observer_lat is not None):
            return 0.0
        return zenith_up_rotation(self.observer_lat, self.observer_lon, Time(self._sky_time()), ra, dec)

    def _move_frame_here(self):
        """Put the frame where the view's crosshair is."""
        self._run_js("ccView()", self._on_move_frame_view)

    def _on_move_frame_view(self, view):
        if not view:
            return
        try:
            current = json.loads(view)
            ra, dec = float(current["ra"]) % 360.0, float(current["dec"])
        except (ValueError, KeyError, TypeError):
            return
        target_ra, target_dec = self._target_position()
        # Back on the target when it's (nearly) there
        near = target_ra is not None and angular_distance(ra, dec, target_ra, target_dec) < 1 / 3600
        self.frame_center = None if near else (ra, dec)
        self._push_frames()

    # ------------------------------------------------------------------
    # Framing saved to a session
    # ------------------------------------------------------------------

    def _current_framing(self):
        """The frame or mosaic as shown, for a session (see load_session_framing) - or None
        when no camera frame is shown."""
        camera = self.camera_combo.currentData()
        if not (self.selected_telescope and self.show_telescope_fov.isChecked()
                and camera and camera.get('type') == 'camera'):
            return None
        fov = self._calculate_telescope_fov()
        ra, dec = self.frame_center or self._target_position()
        if not fov or ra is None:
            return None
        columns, rows, overlap = self._mosaic()
        return {
            "version": 1, "ra": round(ra, 6), "dec": round(dec, 6), "rotation": self._rotation(),
            "columns": columns, "rows": rows, "overlap": overlap,
            "panel_width": fov['width_arcmin'] / 60.0, "panel_height": fov['height_arcmin'] / 60.0,
            "telescope": self.selected_telescope.get('name'),
            "focal_length": self.selected_telescope.get('focal_length'),
            "camera": self.camera_combo.currentText(), "barlow": self.barlow_combo.currentText(),
            "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _apply_framing(self, framing):
        """Show a session's saved framing: its equipment (if it's still here),
        mosaic, rotation and center."""
        missing = []
        name = framing.get("telescope")
        index = next((i for i in range(self.telescope_combo.count())
                      if (self.telescope_combo.itemData(i) or {}).get('name') == name), -1)
        if index < 0 and any(st['name'] == name for st in self.SMART_TELESCOPES):
            self.show_smart_telescopes.setChecked(True)  # Adds the smart telescopes
            index = next((i for i in range(self.telescope_combo.count())
                          if (self.telescope_combo.itemData(i) or {}).get('name') == name), -1)
        if index >= 0:
            self.telescope_combo.setCurrentIndex(index)
        elif name:
            missing.append(name)
        camera = framing.get("camera") or ""
        index = self.camera_combo.findText(camera)
        if index < 0 and camera:
            key = self._camera_key(camera)
            index = next((i for i in range(self.camera_combo.count())
                          if self.camera_combo.itemData(i) is not None
                          and self._camera_key(self.camera_combo.itemText(i)) == key), -1)
        if index >= 0:
            self.camera_combo.setCurrentIndex(index)
        elif camera:
            missing.append(camera)
        index = self.barlow_combo.findText(framing.get("barlow") or "")
        if index >= 0:
            self.barlow_combo.setCurrentIndex(index)
        self.show_telescope_fov.setChecked(True)
        self.mosaic_columns.setValue(framing.get("columns", 1))
        self.mosaic_rows.setValue(framing.get("rows", 1))
        self.mosaic_overlap.setValue(round(framing.get("overlap", 0.2) * 100))
        self.rotation_spin.setValue(int(framing.get("rotation", 0)) % 360)
        ra, dec = framing.get("ra"), framing.get("dec")
        target_ra, target_dec = self._target_position()
        if ra is not None and dec is not None and (
                target_ra is None or angular_distance(ra, dec, target_ra, target_dec) > 1 / 3600):
            self.frame_center = (ra, dec)
        self.framing_note = (f"Saved with {' and '.join(missing)}, which "
                             f"{'are' if len(missing) > 1 else 'is'}n't in your equipment now" if missing else None)

    def _sessions_for_object(self, conn):
        """This object's sessions, newest first: [(id, name, date, status, has framing)] - by
        position when both have one, else by name."""
        name = (self.data.get('name') or "").strip().lower()
        ra, dec = self._target_position()
        found = []
        for session_id, session_name, date, status, s_ra, s_dec, framing in conn.execute(
                "SELECT id, dso_name, session_date, status, ra_deg, dec_deg, framing FROM usersessions "
                "ORDER BY session_date DESC, id DESC"):
            if ra is not None and s_ra is not None and s_dec is not None:
                same = angular_distance(ra, dec, s_ra, s_dec) < 0.5
            else:
                plain = (session_name or "").strip().lower()
                same = bool(name) and (plain == name or plain.startswith(name + " ("))
            if same:
                found.append((session_id, session_name, date, status, bool(framing)))
        return found

    def _save_to_session(self):
        """Keep the frame or mosaic with one of this object's sessions (or a new planned one)."""
        framing = self._current_framing()
        if framing is None:
            return
        try:
            with DatabaseManager().get_connection() as conn:
                sessions = self._sessions_for_object(conn)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't load the sessions: {e}")
            return
        dialog = SaveFramingDialog(self.data.get('name') or "this object", framing, sessions,
                                   self.session_id, self)
        if dialog.exec() != QDialog.Accepted:
            return
        session_id = dialog.chosen_session
        try:
            with DatabaseManager().get_connection() as conn:
                if session_id is None:
                    session_id = self._create_planned_session(conn)
                save_session_framing(conn, session_id, framing)
                conn.commit()
                row = conn.execute("SELECT dso_name, session_date FROM usersessions WHERE id = ?",
                                   (session_id,)).fetchone()
        except Exception as e:
            logger.error(f"Error saving the framing: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't save the framing: {e}")
            return
        self.session_id = session_id
        _refresh_session_manager()
        QMessageBox.information(
            self, "Framing Saved",
            f"Saved to the {row[0]} session on {row[1]}.\n\n"
            "Its panels, progress and NINA actions are under Framing in the session's details.")

    def _create_planned_session(self, conn):
        """A new planned session for tonight at the active location. Returns its id."""
        name = self.data.get('name') or "Target"
        ra, dec = self._target_position()
        location = conn.execute("SELECT location_lat, location_lon, location_name, timezone FROM usersettings "
                                "WHERE is_active = 1 LIMIT 1").fetchone() or (None, None, None, None)
        target_id = None
        try:
            from DSOTargetList import find_existing_target
            entry = find_existing_target(conn, name, ra, dec)
            target_id = entry["id"] if entry else None
        except Exception as e:
            logger.debug(f"No target list entry for {name}: {e}")
        telescope = self.selected_telescope or {}
        camera = self.camera_combo.currentData() or {}
        camera_name = self.camera_combo.currentText().split(" (")[0] if camera.get('source') == 'user' else ""
        cursor = conn.execute("""
            INSERT INTO usersessions (target_id, dso_name, ra_deg, dec_deg, status, session_date,
                                      location_lat, location_lon, location_name, location_timezone,
                                      location_source, telescope_id, camera, modified_date)
            VALUES (?, ?, ?, ?, 'Planned', ?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """, (target_id, name, ra, dec, datetime.now().strftime("%Y-%m-%d"), location[0], location[1],
              location[2], location[3], None if telescope.get('smart_telescope') else telescope.get('id'),
              camera_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return cursor.lastrowid

    def _load_observer_location(self):
        """Load the active location (and its custom horizon) from the DB. Returns True on success."""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon, location_name, timezone FROM usersettings "
                               "WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if row and row[0] is not None and row[1] is not None:
                    from DSOTargetList import local_zone
                    self.observer_lat = float(row[0])
                    self.observer_lon = float(row[1])
                    self.observer_name = row[2]
                    self.observer_tz = local_zone(row[3])
                    self.horizon_profile = load_active_horizon(conn)
                    return True
        except Exception as e:
            logger.error(f"Error loading observer location: {e}")
        return False

    def _sky_overlay(self):
        """Horizon (with the custom horizon's blocked sky), meridian and - in Night
        View - the target's path, for the page at the chosen time; or None when
        none of them is shown."""
        horizon, meridian = self.show_horizon.isChecked(), self.show_meridian.isChecked()
        self.target_altaz = None
        if not (horizon or meridian) or self.observer_lat is None:
            return None
        ra, dec = self._target_position()
        when = Time(self._sky_time())
        try:
            lines = sky_lines(self.observer_lat, self.observer_lon, when, horizon=horizon,
                              meridian=meridian, profile=self.horizon_profile, ra=ra, dec=dec)
        except Exception as e:
            logger.error(f"Error calculating the horizon/meridian: {e}", exc_info=True)
            return None
        if lines["target_alt"] is not None:
            self.target_altaz = (lines["target_alt"], lines["target_az"])
        shapes = []
        if horizon:
            for polygon in lines["blocked"]:
                shapes.append({"type": "polygon", "points": polygon, "color": "rgba(0, 0, 0, 0)",
                               "fill": BLOCKED_FILL, "width": 1})
            shapes.append({"type": "polyline", "points": lines["horizon"], "width": 2,
                           "color": HORIZON_DIM_COLOR if lines["custom"] else HORIZON_COLOR})
            if lines["custom"]:
                shapes.append({"type": "polyline", "points": lines["custom"], "color": HORIZON_COLOR, "width": 2})
        if meridian:
            shapes.append({"type": "polyline", "points": lines["meridian"], "color": MERIDIAN_COLOR, "width": 2})
        labels = list(lines["labels"])
        if self.night_view_check.isChecked() and self.night is not None and ra is not None:
            # Where the target goes tonight, as it'll look from here (it moves along this as time passes)
            segments, hours = target_path(self.observer_lat, self.observer_lon, when, self.night)
            for segment in segments:
                shapes.append({"type": "polyline", "points": segment, "color": PATH_COLOR, "width": 2})
            for index, hour_ra, hour_dec in hours:
                labels.append({"text": hour_label(self.night["times"][index]), "ra": hour_ra, "dec": hour_dec,
                               "color": PATH_COLOR, "small": True})
            shapes.append({"type": "circle", "ra": ra, "dec": dec, "radius": 0.8, "color": PATH_COLOR, "width": 2})
        return {"shapes": shapes, "labels": labels, "horizon": lines["horizon"],
                "targetAlt": lines["target_alt"], "color": HORIZON_COLOR}

    def _push_sky(self):
        """Draw the horizon/meridian (and Night View's path and twilight) for the chosen time."""
        self._shown_time = self._sky_time()
        payload = self._sky_overlay()
        self._run_js(f"ccSetSky({json.dumps(payload)})")
        tint = 0.0
        if self.night_view_check.isChecked() and self.night is not None:
            tint = sky_tint(float(self.night["sun_alt"][self._time_index()]))
        self._run_js(f"ccTint({tint:.3f})")
        if payload and self.live:
            self.sky_timer.start()
        else:
            self.sky_timer.stop()
        self._update_fov_info()

    # ------------------------------------------------------------------
    # Time: any night, any time
    # ------------------------------------------------------------------

    def _setup_time(self):
        """The time row for the active location - tonight, now."""
        has_location = self.observer_lat is not None or self._load_observer_location()
        for widget in self.time_widgets:
            widget.setEnabled(has_location)
            if not has_location:
                widget.setToolTip("Set your observing location in Settings to use the time and Night View")
        if has_location:
            self._go_now()

    def _tonight_noon(self, now):
        """Local noon starting the night 'now' is in (before noon: last night's)."""
        noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
        return noon if now >= noon else noon - timedelta(days=1)

    def _load_night(self, noon):
        """The night starting at 'noon' on the slider."""
        self.night_noon = noon
        ra, dec = self._target_position()
        try:
            self.night = night_profile(self.observer_lat, self.observer_lon, noon, ra, dec, self.horizon_profile)
        except Exception as e:
            logger.error(f"Error calculating the night: {e}", exc_info=True)
            self.night = None
        self.time_slider.set_night(self.night, self.data.get('name') or "Target")
        self.night_date.blockSignals(True)
        self.night_date.setDate(QDate(noon.year, noon.month, noon.day))
        self.night_date.blockSignals(False)

    def _sky_time(self):
        """The time the sky is shown for: the clock's, or the slider's."""
        if self.observer_tz is None:
            return datetime.now().astimezone()
        if self.live or self.night_noon is None:
            return datetime.now(self.observer_tz)
        return self.night_noon + timedelta(minutes=self.time_slider.value() * NIGHT_STEP_MINUTES)

    def _time_index(self):
        """The slider step nearest the shown time."""
        if self.night_noon is None:
            return 0
        minutes = (self._sky_time() - self.night_noon).total_seconds() / 60.0
        return max(0, min(NIGHT_STEPS, round(minutes / NIGHT_STEP_MINUTES)))

    def _go_now(self):
        """Back to the clock."""
        self.play_timer.stop()
        self.play_button.setText("▶ Play")
        self.live = True
        if self.observer_lat is None or self.observer_tz is None:
            return
        now = datetime.now(self.observer_tz)
        noon = self._tonight_noon(now)
        if self.night is None or noon != self.night_noon:
            self._load_night(noon)
        self.time_slider.setValue(self._time_index())
        self.time_slider.set_now(self._time_index())
        self._time_changed()

    def _on_time_slider(self, value):
        self.live = False
        self._time_changed()

    def _on_night_date_changed(self, date):
        if self.observer_tz is None:
            return
        self.live = False
        self._load_night(datetime(date.year(), date.month(), date.day(), 12, 0, tzinfo=self.observer_tz))
        self._update_now_mark()
        self._time_changed()

    def _update_now_mark(self):
        now = datetime.now(self.observer_tz)
        minutes = (now - self.night_noon).total_seconds() / 60.0 if self.night_noon else -1
        self.time_slider.set_now(round(minutes / NIGHT_STEP_MINUTES) if 0 <= minutes <= 24 * 60 else None)

    def _toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("▶ Play")
            return
        if self.time_slider.value() >= NIGHT_STEPS:
            start = self.night["sunset"] if self.night and self.night["sunset"] is not None else 0
            self.time_slider.setValue(start)
        self.live = False
        self.play_button.setText("⏸ Pause")
        self.play_timer.start()

    def _play_step(self):
        if self._reanchoring:
            return  # Let the view catch up
        value = self.time_slider.value() + 1
        if value > NIGHT_STEPS:
            self._toggle_play()
            return
        self.time_slider.setValue(value)
        self._on_time_slider(value)

    def _on_sky_timer(self):
        """Following the clock: keep up with the sky."""
        if self.live:
            self._go_now()
        else:
            self._push_sky()

    def _time_changed(self):
        """Show the sky for the chosen time."""
        when = self._sky_time()
        text = format_time(when)
        if self.night_noon is not None and when.date() != self.night_noon.date():
            text += f" {when:%b} {when.day}"  # After midnight
        if self.night is not None:
            text += f" · {sky_state(float(self.night['sun_alt'][self._time_index()]))}"
        self.time_label.setText(("Now: " if self.live else "") + text)
        if self.night_view_check.isChecked() and self.page_loaded:
            self._reanchor()
        else:
            self._push_sky()

    def _reanchor(self):
        """Night View: keep the view on the same piece of ground as the sky turns."""
        if self._reanchoring:
            self._reanchor_again = True
            return
        self._reanchoring = True
        self._run_js("ccView()", self._on_reanchor_view)

    def _on_reanchor_view(self, view):
        try:
            if (view and self._shown_time is not None and self.observer_lat is not None
                    and self.night_view_check.isChecked()):  # (It may have been turned off meanwhile)
                current = json.loads(view)
                alt, az = radec_to_altaz(self.observer_lat, self.observer_lon, Time(self._shown_time),
                                         float(current["ra"]), float(current["dec"]))
                when = Time(self._sky_time())
                ras, decs = altaz_to_radec(self.observer_lat, self.observer_lon, when, [alt], [az])
                ra, dec = float(ras[0]), float(decs[0])
                rotation = zenith_up_rotation(self.observer_lat, self.observer_lon, when, ra, dec)
                self._run_js(f"ccSetView({ra}, {dec}, {rotation}, null)")
        except Exception as e:
            logger.error(f"Error moving the Night View: {e}", exc_info=True)
        finally:
            self._push_sky()
            self._reanchoring = False
            if self._reanchor_again:
                self._reanchor_again = False
                self._reanchor()

    def _on_night_view_toggled(self, checked):
        """Night View: look at the target's part of the sky from the ground."""
        if checked and self.observer_lat is None and not self._load_observer_location():
            QMessageBox.warning(self, "No Location Set",
                                "Please set your observing location in Settings to use the Night View.")
            self.night_view_check.blockSignals(True)
            self.night_view_check.setChecked(False)
            self.night_view_check.blockSignals(False)
            return
        if checked and not self.show_horizon.isChecked():
            self.show_horizon.setChecked(True)  # The ground is what it's about
        when = self._sky_time()
        ra, dec = self._target_position()
        if checked:
            # Toward the target, a wide view a little below it: low targets get the
            # horizon at the bottom, high ones stay in view (the page points to the horizon)
            if ra is not None:
                target_alt, az = radec_to_altaz(self.observer_lat, self.observer_lon, Time(when), ra, dec)
            else:
                target_alt, az = 37.0, 180.0
            center_alt = min(max(target_alt - 12.0, 25.0), 75.0)
            ras, decs = altaz_to_radec(self.observer_lat, self.observer_lon, Time(when), [center_alt], [az])
            center_ra, center_dec = float(ras[0]), float(decs[0])
            rotation = zenith_up_rotation(self.observer_lat, self.observer_lon, Time(when), center_ra, center_dec)
            self._run_js(f"ccSetView({center_ra}, {center_dec}, {rotation}, {NIGHT_VIEW_FOV})")
        else:
            center_ra, center_dec = self.frame_center or (ra, dec)
            self.current_fov = self._fit_fov()
            if center_ra is not None:
                self._run_js(f"ccSetView({center_ra}, {center_dec}, 0, {self.current_fov})")
        self._push_sky()

    def _level_view(self, report):
        """Night View: keep the zenith up as the view is panned."""
        if not (self.night_view_check.isChecked() and self.observer_lat is not None) or self._reanchoring:
            return
        try:
            ra, dec = float(report["ra"]), float(report["dec"])
            rotation = zenith_up_rotation(self.observer_lat, self.observer_lon,
                                          Time(self._shown_time or self._sky_time()), ra, dec)
        except (KeyError, TypeError, ValueError):
            return
        if abs((rotation - float(report.get("rot", 0)) + 180) % 360 - 180) > 0.5:
            self._run_js(f"ccRotate({rotation})")

    def _on_show_horizon_toggled(self):
        """Handle Virtual Horizon checkbox toggle."""
        self._on_sky_toggled(self.show_horizon)

    def _on_show_meridian_toggled(self):
        """Handle Meridian Line checkbox toggle."""
        self._on_sky_toggled(self.show_meridian)

    def _on_sky_toggled(self, check):
        if check.isChecked() and not self._load_observer_location():
            QMessageBox.warning(self, "No Location Set",
                f"Please set your observing location in Settings to use the {check.text()}.")
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
        self._save_aladin_settings()
        self._push_sky()

    def _on_page_report(self, title):
        """The page reports through its title: "cc:{json}" with the view's
        width ("fov") or an error it showed ("error")."""
        if not title.startswith("cc:"):
            return
        try:
            report = json.loads(title[3:])
        except ValueError:
            return
        if report.get("error") and not self._page_error:
            self._page_error = True
            logger.error(f"Sky viewer: {report['error']}")
            self._add_browser_fallback_button()
        if report.get("rotateSeq") not in (None, self._rotate_seq):
            self._rotate_seq = report["rotateSeq"]
            if report.get("rotating") and report.get("rotate") is not None:
                angle = float(report["rotate"])
                value = round(angle / 15.0) * 15 if report.get("snap") else round(angle)
                self._rotating = True
                self.rotation_spin.setValue(int(value) % 360)
            elif self._rotating:
                # Let go: keep the angle (the zoom stays as it is)
                self._rotating = False
                self._save_aladin_settings()
                self._update_fov_info()
        if report.get("fov") and float(report["fov"]) != self.view_fov:
            self.view_fov = float(report["fov"])
            self._update_fov_info()
        if "ra" in report:
            self._level_view(report)

    def _on_theme_changed(self):
        self._run_js(f"ccColors({json.dumps(page_colors())})")

    def _update_fov_info(self):
        """Update the FOV information display"""
        view = self.view_fov or self.current_fov
        info_parts = [f"View FOV: {view:.2f}° ({view*60:.1f}')"]

        # Object size info
        obj_size_min = self.data.get('size_min') or 0
        obj_size_max = self.data.get('size_max') or 0
        if obj_size_min > 0 and obj_size_max > 0:
            if abs(obj_size_min - obj_size_max) < 0.1:
                info_parts.append(f"Object: {obj_size_min:.1f}'")
            else:
                info_parts.append(f"Object: {obj_size_min:.1f}'–{obj_size_max:.1f}'")

        # Telescope FOV info
        if self.selected_telescope:
            telescope_fov_data = self._calculate_telescope_fov()
            if telescope_fov_data:
                telescope_name = self.selected_telescope['name']
                telescope_fl = self.selected_telescope['focal_length']

                if telescope_fov_data['type'] == 'visual':
                    fov_str = f"{telescope_fov_data['width_arcmin']:.1f}'"
                else:
                    fov_str = f"{telescope_fov_data['width_arcmin']:.1f}'×{telescope_fov_data['height_arcmin']:.1f}'"
                info_parts.append(f"{telescope_name} ({telescope_fl}mm): {fov_str} {telescope_fov_data['details']}")
                if telescope_fov_data.get('pixel_scale'):
                    info_parts.append(f"Pixel scale: {telescope_fov_data['pixel_scale']:.2f}\"/px")

                if self.show_telescope_fov.isChecked() and self._rotation():
                    info_parts.append(f"Rotation {self._rotation()}°")
                columns, rows, overlap = self._mosaic()
                if self.show_telescope_fov.isChecked() and columns * rows > 1:
                    width, height = mosaic_size(telescope_fov_data['width_arcmin'] / 60.0,
                                                telescope_fov_data['height_arcmin'] / 60.0, columns, rows, overlap)
                    info_parts.append(f"Mosaic {columns}×{rows} ({overlap:.0%} overlap): {width:.2f}°×{height:.2f}°")

        if self.frame_center is not None:
            target_ra, target_dec = self._target_position()
            if target_ra is not None:
                offset = angular_distance(*self.frame_center, target_ra, target_dec)
                info_parts.append(f"Frame moved {offset:.2f}° from {self.data.get('name', 'the target')}")
        if self.framing_note:
            info_parts.append(self.framing_note)

        # Where the target is at the chosen time
        if self.target_altaz is not None:
            alt, az = self.target_altaz
            place = self.observer_name or "your location"
            when = f"Now at {place}" if self.live else f"At {format_time(self._sky_time())} from {place}"
            where = f"{when}: alt {alt:.1f}°, az {az:.0f}° ({compass_point(az)})"
            if alt < 0:
                where += " - below the horizon"
            elif self.horizon_profile is not None and alt < self.horizon_profile.altitude_at(az):
                where += " - behind your custom horizon"
            info_parts.append(where)
        if self.target_altaz is not None and self.night is not None and self.night["transit"] is not None:
            transit = self.night["transit"]
            info_parts.append(f"Crosses the meridian at {format_time(self.night['times'][transit])} "
                              f"(alt {float(self.night['target_alt'][transit]):.0f}°)")

        self.fov_info_label.setText(" | ".join(info_parts))

    def _on_load_started(self):
        """Handle web page load started"""
        logger.debug("Aladin Lite: Load started")
        self._renderer_crashed = False
        if self.web_placeholder:
            self.web_placeholder.setText("Loading Aladin Lite...")
        # Start timeout timer (30 seconds)
        self.loading_timeout.start(30000)

    def _on_load_progress(self, progress):
        """Handle web page load progress"""
        logger.debug(f"Aladin Lite: Load progress {progress}%")
        # Chromium still reports load progress after a renderer crash; don't stomp the crash message.
        if self.web_placeholder and not self._renderer_crashed:
            self.web_placeholder.setText(f"Loading Aladin Lite... {progress}%")

    def _on_render_process_terminated(self, termination_status, exit_code):
        """Handle GPU/renderer process crash without crashing the whole app"""
        from PySide6.QtWebEngineCore import QWebEnginePage
        status_names = {
            QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus: "Normal",
            QWebEnginePage.RenderProcessTerminationStatus.AbnormalTerminationStatus: "Abnormal",
            QWebEnginePage.RenderProcessTerminationStatus.CrashedTerminationStatus: "Crashed",
            QWebEnginePage.RenderProcessTerminationStatus.KilledTerminationStatus: "Killed",
        }
        status_name = status_names.get(termination_status, f"Unknown({termination_status})")
        logger.error(f"Aladin Lite renderer process terminated: {status_name}, exit code: {exit_code}")
        self.loading_timeout.stop()
        self._renderer_crashed = True

        if self.web_placeholder is None:
            # Placeholder was already removed; re-create it so we can show the error
            central_widget = self.centralWidget()
            if central_widget and central_widget.layout():
                self.web_placeholder = QLabel("Renderer crashed")
                central_widget.layout().insertWidget(1, self.web_placeholder)

        if self.web_placeholder:
            self.web_placeholder.setText(
                "Aladin Lite renderer crashed (GPU issue)\n\n"
                "This is usually caused by a GPU driver incompatibility.\n"
                "Click below to open in your browser instead."
            )
            themed_style(self.web_placeholder, lambda: f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: {font_px(12)}; }}")
            self.web_placeholder.show()
        self._add_browser_fallback_button()
        # Stop any in-progress load to break the automatic crash-and-retry cycle
        try:
            if self.web_view:
                self.web_view.stop()
        except Exception:
            pass

    def _on_load_finished(self, success):
        """Handle web page load finished"""
        logger.debug(f"Aladin Lite: Load finished, success={success}")
        self.loading_timeout.stop()

        if not success:
            logger.error("Failed to load Aladin Lite")
            # If the renderer already crashed, _on_render_process_terminated set a more
            # accurate message — don't stomp it with the generic network-error text.
            if self.web_placeholder and not self._renderer_crashed:
                self.web_placeholder.setText("Failed to load Aladin Lite\nCheck your internet connection\n\nClick below to open in browser instead.")
                themed_style(self.web_placeholder, lambda: f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: {font_px(12)}; }}")
            self._add_browser_fallback_button()
        else:
            logger.debug("Aladin Lite loaded successfully")
            # Ensure the web view is visible and placeholder is hidden
            self._ensure_web_view_visible()

            # Check WebGL availability for diagnostics
            self._check_webgl_support()

            # Hand the page what to draw - it waits for Aladin to start if need be
            self.page_loaded = True
            self._push_frames()
            self._push_sky()
            if self.night_view_check.isChecked():
                self._on_night_view_toggled(True)

    def _check_webgl_support(self):
        """Check if WebGL is available in the browser context"""
        if not self.web_view:
            return

        # JavaScript to check WebGL support
        js_code = """
        (function() {
            var canvas = document.createElement('canvas');
            var gl = canvas.getContext('webgl') || canvas.getContext('webgl2') || canvas.getContext('experimental-webgl');
            if (gl) {
                var debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                var vendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown';
                var renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown';
                return JSON.stringify({
                    available: true,
                    version: gl instanceof WebGL2RenderingContext ? 'WebGL2' : 'WebGL1',
                    vendor: vendor,
                    renderer: renderer
                });
            } else {
                return JSON.stringify({available: false, error: 'WebGL not available'});
            }
        })();
        """

        def handle_webgl_result(result):
            try:
                import json
                webgl_info = json.loads(result)
                if webgl_info.get('available'):
                    logger.info(f"WebGL is available: {webgl_info.get('version', 'Unknown')} - Vendor: {webgl_info.get('vendor', 'Unknown')}, Renderer: {webgl_info.get('renderer', 'Unknown')}")
                else:
                    logger.error(f"WebGL is NOT available: {webgl_info.get('error', 'Unknown error')}")
                    logger.error("Aladin Lite will not render properly without WebGL support")
            except Exception as e:
                logger.error(f"Failed to parse WebGL info: {e}, result: {result}")

        try:
            self.web_view.page().runJavaScript(js_code, handle_webgl_result)
        except Exception as e:
            logger.error(f"Failed to check WebGL support: {e}")

    def _handle_loading_timeout(self):
        """Handle loading timeout"""
        logger.warning("Aladin Lite loading timed out after 30 seconds")
        if self.web_placeholder:
            self.web_placeholder.setText("Loading timed out\nAladin Lite may be slow to respond\n\nClick below to open in browser instead.")
            themed_style(self.web_placeholder, lambda: f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['warning']}; font-size: {font_px(12)}; }}")
        self._add_browser_fallback_button()

    def _ensure_web_view_visible(self):
        """Ensure the web view is visible and replace placeholder if needed"""
        try:
            if self.web_view and self.web_placeholder:
                logger.debug("Replacing placeholder with web view after successful load")

                # Find the central widget and its layout
                central_widget = self.centralWidget()
                if central_widget and central_widget.layout():
                    main_layout = central_widget.layout()

                    # Find the placeholder in the layout and replace it
                    for i in range(main_layout.count()):
                        item = main_layout.itemAt(i)
                        if item and item.widget() == self.web_placeholder:
                            # Remove placeholder
                            main_layout.removeWidget(self.web_placeholder)
                            self.web_placeholder.hide()
                            self.web_placeholder.deleteLater()
                            self.web_placeholder = None

                            # Add web view in the same position
                            main_layout.insertWidget(i, self.web_view, 1)
                            self.web_view.show()
                            logger.debug("Successfully replaced placeholder with web view")
                            break

            elif self.web_view:
                # Just make sure web view is visible
                self.web_view.show()
                logger.debug("Made web view visible")

        except Exception as e:
            logger.error(f"Error ensuring web view visibility: {e}")

    def _test_connectivity_async(self):
        """Test connectivity to Aladin Lite server asynchronously"""
        def test_connection():
            try:
                # Use a plain TCP connection instead of urllib/ssl to avoid an
                # OpenSSL conflict with PySide6.QtNetwork (both link OpenSSL;
                # using Python's ssl on a background thread alongside Qt's
                # network stack can trigger an abort() in the C layer).
                import socket
                with socket.create_connection(("aladin.cds.unistra.fr", 443), timeout=10):
                    logger.debug("Aladin server connectivity test successful")
            except Exception as e:
                logger.warning(f"Aladin server connectivity test failed: {e}")

        threading.Thread(target=test_connection, daemon=True).start()

    # ------------------------------------------------------------------
    # Target List helpers
    # ------------------------------------------------------------------

    def _check_if_in_target_list(self):
        """Check if the current DSO is already in the target list."""
        return self._target_list_entry() is not None

    def _target_list_entry(self):
        """The current DSO's Target List entry - same name, another designation of
        the same object, or the same position - or None."""
        try:
            from DSOTargetList import find_existing_target, delete_target
            dso_name = self.data.get('name', '').strip()
            if not dso_name:
                return None
            with DatabaseManager().get_connection() as conn:
                return find_existing_target(
                    conn, dso_name, self.data.get('ra_deg'), self.data.get('dec_deg'))
        except Exception as e:
            logger.error(f"Error checking target list status: {e}")
            return None

    def _find_target_list_name(self):
        """Return the stored target list name for the current DSO, or None."""
        entry = self._target_list_entry()
        return entry["name"] if entry else None

    def _update_target_list_button(self):
        """Show/hide target list actions based on current list membership."""
        try:
            in_list = self._check_if_in_target_list()
            self.add_target_action.setVisible(not in_list)
            self.remove_target_action.setVisible(in_list)
            self.open_target_action.setVisible(in_list)
        except Exception as e:
            logger.error(f"Error updating target list button: {e}")
            self.add_target_action.setVisible(True)
            self.remove_target_action.setVisible(False)
            self.open_target_action.setVisible(False)

    def _add_to_target_list(self):
        """Open the Add Target dialog for the current DSO."""
        try:
            from DSOTargetList import AddTargetDialog
            dialog = AddTargetDialog(dso_data=self.data.copy(), parent=self)
            if dialog.exec():
                self._update_target_list_button()
        except ImportError as e:
            logger.error(f"Could not import DSOTargetList: {e}")
            QMessageBox.warning(self, "Import Error", f"Could not load Target List feature: {e}")
        except Exception as e:
            logger.error(f"Error adding to target list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {e}")

    def _remove_from_target_list(self):
        """Remove the current DSO from the target list after confirmation."""
        try:
            dso_name = self.data.get('name', 'this DSO')
            reply = QMessageBox.question(
                self, "Remove from Target List",
                f"Remove '{dso_name}' from the target list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            from DSOTargetList import find_existing_target, delete_target
            entry = self._target_list_entry()
            if entry:
                # Just this object's entry (this used to delete every entry with a
                # matching name or position)
                with DatabaseManager().get_connection() as conn:
                    delete_target(conn, entry["id"])
            self._update_target_list_button()
            QMessageBox.information(self, "Success", f"'{dso_name}' removed from target list.")
        except Exception as e:
            logger.error(f"Error removing from target list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove from target list: {e}")

    def _open_from_target_list(self):
        """Open the Target List window and highlight the current DSO."""
        try:
            target_name = self._find_target_list_name()
            if not target_name:
                QMessageBox.warning(self, "Not Found",
                    f"Could not find '{self.data.get('name', '')}' in your target list.")
                return
            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, '_target_list_window') or not self._target_list_window.isVisible():
                self._target_list_window = DSOTargetListWindow()
            self._target_list_window.show()
            self._target_list_window.raise_()
            success = self._target_list_window.open_and_select_target(target_name)
            if not success:
                QMessageBox.warning(self, "Not Found",
                    f"Could not navigate to '{target_name}' in the target list.")
        except Exception as e:
            logger.error(f"Error opening from target list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open target list: {e}")

    # ------------------------------------------------------------------
    # NINA: the frame as shown, or each mosaic panel - only when clicked
    # ------------------------------------------------------------------

    def _nina_targets(self):
        """(rotation or None, [(label, name, ra, dec)]): the mosaic's panels, or the
        one frame (where it was moved to) - the target itself without a frame."""
        name = short_object_name(self.data.get('name'))
        framing = self._current_framing()
        if framing is not None:
            panels = framing_panels(framing, self.data.get('name'))
            return framing["rotation"], [(f"Panel {p['number']}", p["name"], p["ra"], p["dec"]) for p in panels]
        ra, dec = self.frame_center or self._target_position()
        return None, [(name, name, ra, dec)]

    def _fill_nina_menu(self):
        menu = self.nina_menu
        menu.clear()
        rotation, targets = self._nina_targets()
        if len(targets) > 1:
            # Kept on self: a submenu only Python refers to can be deleted under the menu
            send = QMenu("Send Panel to Framing Assistant", menu)
            slew = QMenu("Slew to Panel", menu)
            self._nina_submenus = (send, slew)
            menu.addMenu(send)
            menu.addMenu(slew)
            for label, name, ra, dec in targets:
                send.addAction(label).triggered.connect(
                    lambda _=False, n=name, r=ra, d=dec: self._send_to_nina(n, r, d, rotation))
                slew.addAction(label).triggered.connect(
                    lambda _=False, n=name, r=ra, d=dec: self._slew_to_nina_target(n, r, d))
            note = menu.addAction(f"{len(targets)} panels, each sent with its rotation ({rotation}°)")
            note.setEnabled(False)
        else:
            _label, name, ra, dec = targets[0]
            menu.addAction("Send to Framing Assistant").triggered.connect(
                lambda: self._send_to_nina(name, ra, dec, rotation))
            menu.addAction("Slew to Target" if self.frame_center is None else "Slew to Frame").triggered.connect(
                lambda: self._slew_to_nina_target(name, ra, dec))

    def _send_to_nina(self, name, ra, dec, rotation=None):
        """A frame or panel to NINA's Framing Assistant, with the camera rotation."""
        NINAIntegration.frame_with_rotation(ra, dec, name, rotation, self)

    def _slew_to_nina_target(self, name, ra, dec):
        """Slew the mount there via NINA (NINA's own confirmation first)."""
        NINAIntegration.slew_to_coordinates(ra, dec, name, self)

    # ------------------------------------------------------------------
    # Search: frame another target
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):
        if watched is getattr(self, "search_edit", None) and event.type() == QEvent.FocusIn \
                and self._search_names is None:
            self._load_search_names()
        return super().eventFilter(watched, event)

    def _load_search_names(self):
        """Suggestions for the search box: every catalog designation, common names as they're said."""
        self._search_names = {}
        try:
            with DatabaseManager().get_connection() as conn:
                for catalog, designation in conn.execute("SELECT catalogue, designation FROM cataloguenr"):
                    if not catalog or not designation:
                        continue
                    text = designation if catalog.upper() == "NAME" else f"{catalog} {designation}"
                    self._search_names.setdefault(text, (catalog, designation))
        except Exception as e:
            logger.error(f"Error loading search names: {e}")
        completer = QCompleter(sorted(self._search_names, key=str.lower), self.search_edit)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.setMaxVisibleItems(12)
        completer.activated.connect(self._search_target)
        self.search_edit.setCompleter(completer)

    def _search_target(self, text):
        """Frame another object: from the catalogs, typed coordinates, or SIMBAD."""
        text = (text or "").strip()
        if not text:
            return
        if self._search_names is None:
            self._load_search_names()
        data = self._find_catalog_target(text)
        if data is None:
            data = coordinates_target(text)
        if data is not None:
            self._set_target(data)
            return
        settings = QSettings("CosmosCollection", "CosmosCollection")
        if not settings.value("simbad_lookup_on_search_failure", True, type=bool):
            QMessageBox.information(self, "Not Found",
                                    f"{text} isn't in the catalogs.\n\nLooking it up on SIMBAD is turned off "
                                    "(Settings: \"Offer SIMBAD lookup when search finds no results\").")
            return
        self._search_simbad(text)

    def _find_catalog_target(self, text):
        """The object's catalog data for a name typed or picked, or None."""
        from DSOTargetList import find_full_dso_data
        database = DatabaseManager()
        exact = self._search_names.get(text) or next(
            (value for key, value in self._search_names.items() if key.lower() == text.lower()), None)
        if exact is not None:
            catalog, designation = exact
            return find_full_dso_data(database, f"{catalog} {designation}", {})
        return find_full_dso_data(database, text, {})

    def _search_simbad(self, text):
        from main import SimbadQueryWorker
        self.search_edit.setEnabled(False)
        self.search_edit.setPlaceholderText(f"Looking up {text} on SIMBAD...")
        worker = SimbadQueryWorker(text, self)
        self._simbad_worker = worker

        def done():
            self.search_edit.setEnabled(True)
            self.search_edit.setPlaceholderText("Search target...")

        def found(result):
            done()
            size_max = result.get("size_maj") or result.get("size_min") or 0
            self._set_target({
                "name": " ".join(str(result.get("main_id") or text).split()),
                "ra_deg": result["ra_deg"], "dec_deg": result["dec_deg"],
                "size_min": result.get("size_min") or size_max, "size_max": size_max,
                "dso_type": result.get("dso_type"),
            })

        def not_found():
            done()
            QMessageBox.information(self, "Not Found", f"{text} isn't in the catalogs or on SIMBAD.")

        def failed(message):
            done()
            QMessageBox.warning(self, "Search Failed", message)

        worker.object_found.connect(found)
        worker.object_not_found.connect(not_found)
        worker.error_occurred.connect(failed)
        worker.start()

    def _set_target(self, data):
        """Frame a new object here: a fresh single frame on it (the equipment stays)."""
        self.data = data
        name = data.get('name') or "Target"
        self.setWindowTitle(f"{name} - FOV Simulator - Cosmos Collection")
        self.default_fov = default_view_fov(data)
        self.session_id = None
        self.framing_note = None
        self.search_edit.clear()
        self.search_edit.clearFocus()
        for spin, value in ((self.mosaic_columns, 1), (self.mosaic_rows, 1),
                            (self.mosaic_overlap, 20), (self.rotation_spin, 0)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.frame_center = None
        self._save_aladin_settings()
        self._update_target_list_button()
        ra, dec = self._target_position()
        self._run_js(f"ccSetTarget({json.dumps(ra)}, {json.dumps(dec)}, "
                     f"{json.dumps(f'{ra} {dec}' if ra is not None else name)})")
        if self.observer_lat is not None and self.night_noon is not None:
            self._load_night(self.night_noon)  # The new target's night
            self._update_now_mark()
        self._push_frames()
        if self.night_view_check.isChecked():
            self._on_night_view_toggled(True)
        else:
            self.current_fov = self._fit_fov()
            self._go_to_target(self.current_fov)
            self._push_sky()


def _refresh_session_manager():
    """Reload an open Session Manager's list (a new session, or a framing saved)."""
    from PySide6.QtWidgets import QApplication
    for widget in QApplication.topLevelWidgets():
        if type(widget).__name__ == "SessionManagerWindow" and hasattr(widget, "_load_sessions"):
            try:
                widget._load_sessions()
            except Exception as e:
                logger.debug(f"Could not refresh the Session Manager: {e}")


class SaveFramingDialog(QDialog):
    """Pick the session to keep a framing with: one of the object's sessions or
    a new planned one. chosen_session is the session id, or None for a new one."""

    def __init__(self, name, framing, sessions, preferred=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save to Session")
        self.chosen_session = None
        layout = QVBoxLayout(self)
        summary = framing_summary(framing)
        layout.addWidget(QLabel(f"Save this framing of {name} ({summary[0].lower() + summary[1:]}) to:"))
        self.list = QListWidget()
        new_item = QListWidgetItem("A new planned session for tonight")
        new_item.setData(Qt.UserRole, None)
        self.list.addItem(new_item)
        select = new_item
        open_sessions = []
        for session_id, session_name, date, status, has_framing in sessions:
            text = f"{session_name} - {date} - {status}"
            if has_framing:
                text += " (has a framing - it will be replaced)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, session_id)
            self.list.addItem(item)
            if session_id == preferred:
                select = item
            elif status in ("Planned", "In Progress"):
                open_sessions.append(item)
        if select is new_item and open_sessions:
            select = open_sessions[0]
        self.list.setCurrentItem(select)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 300)

    def accept(self):
        item = self.list.currentItem()
        self.chosen_session = item.data(Qt.UserRole) if item else None
        super().accept()


class NightSlider(QWidget):
    """One night, local noon to noon, in NIGHT_STEP_MINUTES steps - painted with
    how dark the sky is, when the target is up (a green strip), when it crosses
    the meridian, and now. Emits moved(step) when the user changes it."""

    moved = Signal(int)
    SKY_COLORS = ((SUNRISE_ALT, "#6f93bd"), (-6.0, "#4a6488"), (-12.0, "#30425f"),
                  (ASTRONOMICAL_DARK_ALT, "#1f2a3f"), (-90.0, "#0b0f17"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._now = None
        self.night = None
        self.target_name = "Target"
        self.setMinimumHeight(40)
        self.setMinimumWidth(240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = max(0, min(NIGHT_STEPS, int(value)))
        self.update()

    def set_now(self, index):
        self._now = index
        self.update()

    def set_night(self, night, target_name):
        self.night = night
        self.target_name = target_name
        self.update()

    # Geometry
    def _band(self):
        return QRectF(6, 4, self.width() - 12, 16)

    def _x(self, index):
        band = self._band()
        return band.left() + band.width() * index / NIGHT_STEPS

    def _index(self, x):
        band = self._band()
        return round((x - band.left()) / max(band.width(), 1) * NIGHT_STEPS)

    @classmethod
    def _sky_color(cls, sun_alt):
        for limit, color in cls.SKY_COLORS:
            if sun_alt > limit:
                return QColor(color)
        return QColor(cls.SKY_COLORS[-1][1])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        band = self._band()
        night = self.night
        if night is not None:
            # The sky's darkness through the night
            for i in range(NIGHT_STEPS):
                left, right = self._x(i), self._x(i + 1)
                painter.fillRect(QRectF(left, band.top(), right - left + 0.6, band.height()),
                                 self._sky_color(float(night["sun_alt"][i])))
            # When the target is up
            if night.get("target_up") is not None:
                for i in range(NIGHT_STEPS):
                    if night["target_up"][i]:
                        left, right = self._x(i), self._x(i + 1)
                        painter.fillRect(QRectF(left, band.bottom() - 4, right - left + 0.6, 4), QColor(80, 190, 110))
            # Dusk and dawn
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            for key in ("dusk", "dawn"):
                if night.get(key) is not None:
                    x = self._x(night[key])
                    painter.drawLine(QPointF(x, band.top()), QPointF(x, band.bottom()))
            # The meridian crossing
            if night.get("transit") is not None:
                x = self._x(night["transit"])
                painter.setPen(QPen(QColor(MERIDIAN_COLOR), 2))
                painter.drawLine(QPointF(x, band.top() - 2), QPointF(x, band.bottom() + 2))
        else:
            painter.fillRect(band, QColor(COLORS["background_light"]))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(band)
        # Hours below
        painter.setPen(QColor(COLORS["text_secondary"]))
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.8))
        painter.setFont(font)
        if night is not None:
            for i in range(0, NIGHT_STEPS + 1, 3 * 60 // NIGHT_STEP_MINUTES):
                x = self._x(i)
                painter.drawLine(QPointF(x, band.bottom() + 1), QPointF(x, band.bottom() + 4))
                text = hour_label(night["times"][i])
                width = painter.fontMetrics().horizontalAdvance(text)
                left = min(max(x - width / 2, 0), self.width() - width)
                painter.drawText(QPointF(left, band.bottom() + 15), text)
        # Now
        if self._now is not None:
            x = self._x(self._now)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(COLORS["accent"]))
            painter.drawPolygon(QPolygonF([QPointF(x - 4, 0), QPointF(x + 4, 0), QPointF(x, 5)]))
        # The handle
        x = self._x(self._value)
        painter.setPen(QPen(QColor(COLORS["text"]), 2))
        painter.drawLine(QPointF(x, band.top() - 3), QPointF(x, band.bottom() + 3))
        painter.setBrush(QColor(COLORS["accent"]))
        painter.setPen(QPen(QColor(COLORS["text"]), 1))
        painter.drawEllipse(QPointF(x, band.center().y()), 5, 5)
        if self.hasFocus():
            painter.setPen(QPen(QColor(COLORS["accent"]), 1, Qt.DotLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _describe(self, index):
        night = self.night
        if night is None:
            return ""
        index = max(0, min(NIGHT_STEPS, index))
        parts = [format_time(night["times"][index]), sky_state(float(night["sun_alt"][index]))]
        if night.get("target_alt") is not None:
            parts.append(f"{self.target_name} alt {float(night['target_alt'][index]):.0f}°")
        if night.get("transit") == index:
            parts.append("crosses the meridian")
        return " · ".join(parts)

    def _move_to(self, index):
        index = max(0, min(NIGHT_STEPS, index))
        if index != self._value:
            self.setValue(index)
            self.moved.emit(index)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._move_to(self._index(event.position().x()))

    def mouseMoveEvent(self, event):
        index = self._index(event.position().x())
        if event.buttons() & Qt.LeftButton:
            self._move_to(index)
        QToolTip.showText(event.globalPosition().toPoint(), self._describe(index), self)

    def wheelEvent(self, event):
        steps = 1 if event.angleDelta().y() > 0 else -1
        self._move_to(self._value + steps)
        event.accept()

    def keyPressEvent(self, event):
        moves = {Qt.Key_Left: -1, Qt.Key_Right: 1, Qt.Key_PageDown: -12, Qt.Key_PageUp: 12,
                 Qt.Key_Home: -NIGHT_STEPS, Qt.Key_End: NIGHT_STEPS}
        if event.key() in moves:
            self._move_to(self._value + moves[event.key()])
        else:
            super().keyPressEvent(event)
