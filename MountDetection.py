#!/usr/bin/env python3
"""
Mount Detection
Tells whether each night's lights were taken on an equatorial or an alt-az
mount, for the Import Wizard and the Session Manager. No Qt - it runs on
worker threads, or from the command line to check folders of subs:

    python MountDetection.py "H:/Astrophotography/M 31/2025-10-04/lights" [...]

A night is told by the first of these that settles it - measurements before
what the capture app wrote, which an unofficial app can get wrong:
  1. How far the field turned. An alt-az mount (with no field derotator) turns
     the sky in its frames by the change in the target's parallactic angle; an
     equatorial mount holds it still, or turns it 180° at a meridian flip. Stars
     are matched between subs of the night (NativePlateSolver's quads - no
     catalog, offline) and the turn compared with both. Rough polar alignment
     lets an equatorial night drift several degrees, so a night is only judged
     when the two would differ by MIN_SEPARATION or more.
  2. A fixed camera's frame angle (fixed_camera(): a Seestar S50). Its camera
     is fixed to the arm, so a plate solved frame sits at 0°/180° on the sky in
     equatorial mode and at the target's parallactic angle in alt-az (either
     mod 180, whichever way the frame was mirrored) - which tells even a short
     night, unless the target was near the meridian. Checked against the turn
     and the header on 138 nights: 122 agreed, none disagreed, 16 too near the
     meridian. The built-in solver first, then the user's other solvers.
  3. Its header: a Seestar's EQMODE (1 = equatorial mode, 0 = alt-az). Older
     Seestar firmware leaves it out in either mode, so a missing key says
     nothing. When a measurement disagrees, it wins and the evidence says so.
  4. PIERSIDE (East/West, written by N.I.N.A. and other ASCOM apps): only an
     equatorial mount has a pier side.
  5. What the same camera and telescope showed that night on another target.
Otherwise the night stays unknown.

Detecting the mount by field rotation was inspired by a script from wileecyte
(https://github.com/wileecyte), which plate solved a folder's subs and
judged the mount by their rotation.

Each night's verdict is a usersessionmounts row its light files point to
(usersessionfiles.mount_id), so it follows the files when nights are rebuilt
(a time zone change, files removed). usersessions.mount_override is the
user's own choice, which wins over what was detected.
"""

import json
import logging
import math
import os
import sqlite3
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

import SessionFileScanner
from SessionObservations import make_sub_key, night_date_for, parse_obs_datetime

logger = logging.getLogger(__name__)

EQUATORIAL, ALT_AZ, MIXED = "Equatorial", "Alt-Az", "Mixed"
HEADER, ROTATION, ANGLE, PIER_SIDE, SAME_NIGHT = "header", "rotation", "frame angle", "pier side", "same night"

# Equipment's mount types (EquipmentManager.MOUNT_TYPES) as one of the two
EQUIPMENT_MOUNTS = {"Alt-Az": ALT_AZ, "Dobsonian": ALT_AZ,
                    "Equatorial (German)": EQUATORIAL, "Equatorial (Fork)": EQUATORIAL}

MIN_SEPARATION = 12.0   # Degrees apart the turns of the two mounts must be to judge a night
TOLERANCE = 0.4         # A measured turn within this share of that separation of one mount's is it
STEP = 30.0             # Predicted alt-az turn measured at a time, so enough stars stay in both frames
SAME_POINTING = 0.25    # Subs pointed this close (degrees) are one field - a mosaic's panels aren't
SAME_SITE = 0.5         # ...and taken this close (degrees of latitude/longitude) one site
MIN_MATCHED = 10        # Stars that must match between two subs
CODE_TOLERANCE = 0.01   # How alike two quads' shapes must be (two subs from one camera: very)
MATCH_PX = 3.0          # A star matches within this many analysis pixels
VERIFY_STARS = 80
WORKERS = 4             # Nights checked at once: reads from a network drive wait more than they compute
ANGLE_MIN_SEPARATION = 20.0     # Degrees apart the frame angles of the two mounts must be to judge by it
ANGLE_TOLERANCE = 0.4           # A frame angle within this share of that separation of one mount's is it
ANGLE_TRIES = 3                 # Subs tried for a frame angle, the most telling first
SOLVE_MIN_STARS = 15            # A built-in solve trusted for a frame angle: this many stars matched...
SOLVE_MAX_RMS_PX = 1.5          # ...this close (pixels) - good subs solve to about 0.3

# How strongly a verdict was shown - the strongest wins when several cover a night.
# Measurements outrank headers; each measurement by how far apart its two answers were.
ROTATION_STRENGTH = 200.0   # + the separation (12-180)
ANGLE_STRENGTH = 100.0      # + the separation (20-90)
HEADER_STRENGTH = 50.0
PIER_SIDE_STRENGTH = 5.0
SAME_NIGHT_STRENGTH = 1.0

HINT_KEYWORDS = ("EQMODE", "PIERSIDE", "SITELAT", "SITELONG", "LAT-OBS", "LONG-OBS", "NAXIS1", "NAXIS2")
HINT_VERSION = 2            # Bump when HINT_KEYWORDS changes: headers are read again
CACHE_VERSION = 1           # Bump when the measuring changes: turns and angles are measured again

J2000 = datetime(2000, 1, 1, 12)


@dataclass(eq=False)
class Sub:
    """A light frame: where it is, when it was taken (UTC, naive), where it
    pointed and (when its header says) where it was taken."""
    path: str
    when: datetime
    ra: Optional[float] = None
    dec: Optional[float] = None
    size: int = 0
    mtime_ns: int = 0
    sub_key: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


@dataclass(eq=False)
class NightMount:
    """The verdict for one night's subs of one rig (camera and telescope)."""
    night: date
    rig: str
    subs: list = field(default_factory=list)
    mount: Optional[str] = None     # EQUATORIAL, ALT_AZ or None (couldn't tell)
    source: Optional[str] = None    # HEADER, ROTATION, PIER_SIDE or SAME_NIGHT
    evidence: str = ""              # How it was told (or why it couldn't be), for people
    strength: float = 0.0


# ---------------------------------------------------------------------------
# Building the nights
# ---------------------------------------------------------------------------

def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rig_of(header):
    """The camera and telescope as the files name them - the key nights share a verdict by."""
    header = header or {}
    return "|".join(str(header.get(k) or "").strip().lower() for k in ("INSTRUME", "TELESCOP"))


def utc_of(header):
    """A header's DATE-OBS as a naive UTC datetime, or None."""
    when = parse_obs_datetime((header or {}).get("DATE-OBS"))
    if when is not None and when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when


def night_of(when, longitude=None):
    """The observing night of a UTC time: local solar time from the site
    longitude, else this computer's time zone (as LibraryScanner splits nights)."""
    if longitude is not None:
        local = when + timedelta(hours=longitude / 15.0)
    else:
        local = when.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    return night_date_for(local)


def sub_from_header(path, header, size=0, mtime_ns=0, frame_type="Light"):
    """A Sub from a light's header, or None without a capture time."""
    when = utc_of(header)
    if when is None:
        return None
    coords = SessionFileScanner._file_coordinates(header or {})
    exposure = _number(header.get("EXPTIME", header.get("EXPOSURE")))
    key = make_sub_key(header.get("DATE-OBS"), exposure, SessionFileScanner._clean_str(header.get("FILTER")),
                       frame_type, SessionFileScanner._clean_str(header.get("INSTRUME")))
    lat, lon = site_of(header)
    return Sub(path, when, coords[0] if coords else None, coords[1] if coords else None, size, mtime_ns, key,
               lat, lon)


def site_of(header):
    """(lat, lon) a header says it was taken at, or (None, None)."""
    header = header or {}
    lat = _number(header.get("SITELAT", header.get("LAT-OBS")))
    lon = _number(header.get("SITELONG", header.get("LONG-OBS")))
    if lat is None or lon is None or not (lat or lon):
        return None, None
    return lat, lon


def nights_from_file_dicts(files, longitude=None):
    """[(night, rig, [Sub])] of SessionFileScanner.scan_file() dicts' lights."""
    groups = defaultdict(list)
    for f in files:
        if f.get("_frame_type") != "Light" or not f.get("_file_path"):
            continue
        sub = sub_from_header(f["_file_path"], f)
        if sub is None:
            continue
        site_lon = _number(f.get("SITELONG", f.get("LONG-OBS")))
        groups[(night_of(sub.when, site_lon if site_lon is not None else longitude), rig_of(f))].append(sub)
    return [(night, rig, subs) for (night, rig), subs in sorted(groups.items())]


# ---------------------------------------------------------------------------
# Sky geometry
# ---------------------------------------------------------------------------

def _gmst_deg(when):
    """Greenwich mean sidereal time (degrees) at a naive UTC datetime."""
    days = (when - J2000).total_seconds() / 86400.0
    return (280.46061837 + 360.98564736629 * days) % 360.0


def parallactic_angle(ra, dec, lat, lon, when):
    """The target's parallactic angle (degrees) seen from lat/lon at a UTC time:
    the angle between north and the zenith at it - how an alt-az frame is turned."""
    hour_angle = math.radians(_gmst_deg(when) + lon - ra)
    phi, delta = math.radians(lat), math.radians(dec)
    return math.degrees(math.atan2(math.sin(hour_angle),
                                   math.tan(phi) * math.cos(delta) - math.sin(delta) * math.cos(hour_angle)))


def _wrap(angle):
    """An angle in -180..180."""
    return (angle + 180.0) % 360.0 - 180.0


def _apart(a, b):
    """How far apart two angles are, 0..180."""
    return abs(_wrap(a - b))


def separation(predicted):
    """How far apart the turns an equatorial mount (0, or 180 after a flip) and
    an alt-az mount (the predicted turn, either way round - a frame can be
    mirrored) would show."""
    turn = abs(_wrap(predicted))
    return min(turn, 180.0 - turn)


def _fields(located):
    """[(sub, lat, lon)] grouped by where they pointed (a mosaic's panels apart)
    and where they were taken (subs others shared from their own sites apart)."""
    groups = []
    for item in located:
        sub, lat, lon = item
        for group in groups:
            first, first_lat, first_lon = group[0]
            if (SessionFileScanner.angular_separation_deg(sub.ra, sub.dec, first.ra, first.dec) <= SAME_POINTING
                    and abs(lat - first_lat) <= SAME_SITE and _apart(lon, first_lon) <= SAME_SITE):
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


# ---------------------------------------------------------------------------
# Measuring the field's turn between two subs
# ---------------------------------------------------------------------------

def _loaded(path, memo=None):
    """NativePlateSolver.load_stars(path), read once per night: memo (a dict)
    keeps what a night's checks have read - stars and header, not pixels."""
    from NativePlateSolver import load_stars
    if memo is None:
        return load_stars(path)
    if path not in memo:
        memo[path] = load_stars(path)
    return memo[path]


def _stars(path, memo=None):
    """A sub's stars as complex x + iy, two rankings (see NativePlateSolver.find_stars)."""
    return [stars[:, 0] + 1j * stars[:, 1] for stars in _loaded(path, memo)[4]]


def _match(za, zb, brightest):
    """(turn in degrees, matched stars) taking stars za onto zb, or None."""
    from NativePlateSolver import _quads, _similarity, _matches, close_pairs
    top_a, top_b = za[:brightest], zb[:brightest]
    if top_a.size < 6 or top_b.size < 6:
        return None
    codes_a, quads_a = _quads(top_a)
    codes_b, quads_b = _quads(top_b)
    verify = za[:VERIFY_STARS]
    i, j, distance = close_pairs(codes_a, codes_b, CODE_TOLERANCE)
    best = None
    if i.size:
        order = np.argsort(distance, kind="stable")   # The most alike first
        a, b = _similarity(top_a[quads_a[i[order]]], top_b[quads_b[j[order]]])
        # Same camera, same optics: the scale doesn't change
        for k in np.flatnonzero(np.abs(np.abs(a) - 1) < 0.02):
            found = _matches(a[k] * verify + b[k], zb, MATCH_PX)
            count = int((found >= 0).sum())
            if best is None or count > best[0]:
                best = (count, found)
            if best[0] >= VERIFY_STARS // 2:
                break
    if best is None or best[0] < MIN_MATCHED:
        return None
    count, found = best
    used = found >= 0
    a, _offset = _similarity(verify[used][None, :], zb[found[used]][None, :])
    return math.degrees(float(np.angle(a[0]))), count


def field_turn(path_a, path_b, memo=None):
    """(degrees the field turned from sub a to sub b, matched stars), or
    (None, 0) when their stars don't match. Raises OSError/ValueError when
    either can't be read. memo: see _loaded()."""
    rankings_a, rankings_b = _stars(path_a, memo), _stars(path_b, memo)
    # The brightest stars first; a frame turned a long way shares fewer of them, so then more
    for ranking, brightest in ((0, 15), (1, 15), (0, 25)):
        result = _match(rankings_a[ranking], rankings_b[ranking], brightest)
        if result is not None:
            return result
    return None, 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _file_key(sub):
    size, mtime_ns = sub.size, sub.mtime_ns
    if not size:
        stat = os.stat(sub.path)
        size, mtime_ns = stat.st_size, stat.st_mtime_ns
        sub.size, sub.mtime_ns = size, mtime_ns
    return f"{os.path.normcase(os.path.normpath(sub.path))}|{size}|{mtime_ns}"


class MountCache:
    """Measured turns, frame angles and header hints, kept beside the library
    scan index (in the app's cache folder - rebuildable), so a rescan only
    measures new nights. Safe to share between threads."""

    def __init__(self, path=None):
        if path is None:
            from ResourceManager import ResourceManager
            path = ResourceManager.get_cache_dir("library") / "scan_index.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS mount_turns (key TEXT PRIMARY KEY, turn REAL, stars INTEGER)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS mount_angles (key TEXT PRIMARY KEY, angle REAL, solver TEXT)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS mount_hints (key TEXT PRIMARY KEY, hint TEXT)")
            self._conn.commit()

    def angle(self, key):
        """(position angle, solver) if solved before, else None."""
        with self._lock:
            row = self._conn.execute("SELECT angle, solver FROM mount_angles WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1]) if row else None

    def save_angle(self, key, angle, solver):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO mount_angles (key, angle, solver) VALUES (?, ?, ?)",
                               (key, angle, solver))
            self._conn.commit()

    def turn(self, key):
        """(turn or None, stars) if measured before, else None."""
        with self._lock:
            row = self._conn.execute("SELECT turn, stars FROM mount_turns WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1]) if row else None

    def save_turn(self, key, turn, stars):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO mount_turns (key, turn, stars) VALUES (?, ?, ?)",
                               (key, turn, stars))
            self._conn.commit()

    def hint(self, key):
        with self._lock:
            row = self._conn.execute("SELECT hint FROM mount_hints WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_hint(self, key, hint):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO mount_hints (key, hint) VALUES (?, ?)",
                               (key, json.dumps(hint, default=str)))
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()


def _hint(sub, cache):
    """The mount, site and frame size keywords of a sub's full header ({} if unreadable)."""
    try:
        key = f"{_file_key(sub)}|h{HINT_VERSION}"
    except OSError:
        return {}
    cached = cache.hint(key) if cache else None
    if cached is not None:
        return cached
    try:
        if os.path.splitext(sub.path)[1].lower() in SessionFileScanner.XISF_EXTENSIONS:
            header = SessionFileScanner.extract_xisf_header(sub.path, keywords=None)
        else:
            header = SessionFileScanner.read_fits_header_fast(sub.path)
    except (OSError, ValueError) as e:
        logger.debug(f"Couldn't read the header of {sub.path}: {e}")
        return {}
    hint = {k: header[k] for k in HINT_KEYWORDS if k in header}
    if cache:
        cache.save_hint(key, hint)
    return hint


def _turn(sub_a, sub_b, cache, memo=None):
    """(turn, matched stars) from a to b - (None, 0) when the stars don't
    match. Raises OSError/ValueError when a sub can't be read (not cached: the
    next scan tries again)."""
    key = f"{_file_key(sub_a)}>{_file_key(sub_b)}|v{CACHE_VERSION}"
    cached = cache.turn(key) if cache else None
    if cached is not None:
        return cached
    turn, stars = field_turn(sub_a.path, sub_b.path, memo)
    if cache:
        cache.save_turn(key, turn, stars)
    return turn, stars


# ---------------------------------------------------------------------------
# A fixed camera's frame angle
# ---------------------------------------------------------------------------

def fixed_camera(rig):
    """Whether a rig's frames sit at a known angle to its mount: a Seestar
    S50's camera is fixed to its arm. (Checked on 300+ nights; other smart
    telescopes likely work the same, but haven't been checked.)"""
    return "seestar s50" in rig.split("|")[0]


def _mod180(angle):
    """An angle in -90..90, the same either way round."""
    return (angle + 90.0) % 180.0 - 90.0


def position_angle(wcs):
    """The frame's +y axis on the sky, degrees east of north, from a WCS's CD matrix (or CDELT/CROTA2)."""
    if "CD1_1" in wcs:
        cd12, cd22 = float(wcs.get("CD1_2", 0)), float(wcs.get("CD2_2", 0))
    else:
        rotation = math.radians(float(wcs.get("CROTA2", 0)))
        cd12 = -float(wcs.get("CDELT2", 0)) * math.sin(rotation)
        cd22 = float(wcs.get("CDELT2", 0)) * math.cos(rotation)
    return math.degrees(math.atan2(cd12, cd22))


def _solve_builtin(sub, memo):
    """The built-in solver's solution - only a tight one: its own bar lets
    through the odd wrong match on a poor sub (23 stars, 3 px off), and a wrong
    angle is a wrong verdict."""
    from NativePlateSolver import solve, SolveFailed
    try:
        wcs, info = solve(sub.path, sub.ra, sub.dec, loaded=_loaded(sub.path, memo))
    except SolveFailed as e:
        logger.debug(f"Built-in solve of {sub.path} for its frame angle: {e}")
        return None
    if info["stars"] < SOLVE_MIN_STARS or info["rms_arcsec"] > SOLVE_MAX_RMS_PX * info["scale"]:
        logger.debug(f"Built-in solve of {sub.path} too loose for its frame angle: {info}")
        return None
    return wcs


def _solve_astap(sub):
    """ASTAP's solution of a temporary copy of the sub - ASTAP saves its .wcs
    next to the file it solves, and nothing is written beside the user's files."""
    import shutil
    import tempfile
    from PlateSolver import PlateSolver
    solver = PlateSolver()
    if not solver.is_astap_available():
        return None
    folder = tempfile.mkdtemp(prefix="cosmos_mount_")
    try:
        copy = os.path.join(folder, "sub" + os.path.splitext(sub.path)[1].lower())
        shutil.copyfile(sub.path, copy)
        result = solver.solve_with_astap(copy, {"ra": sub.ra, "dec": sub.dec, "radius": 5}, use_saved=False)
        return result.wcs_header if result.success and result.wcs_header else None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _frame_angle(sub, cache, memo):
    """(the sub's position angle, solver name) by plate solving, or None: the
    built-in solver first (unless left out in Settings), then ASTAP if the
    user's solver order has it. Not astrometry.net - an upload, minutes an image."""
    key = f"{_file_key(sub)}|a{CACHE_VERSION}"
    cached = cache.angle(key) if cache else None
    if cached is not None:
        return cached
    try:
        from PlateSolver import solver_order
        order = solver_order()
    except Exception:
        order = ["builtin", "astap"]
    for name in [n for n in ("builtin", "astap") if n in order]:
        try:
            wcs = _solve_builtin(sub, memo) if name == "builtin" else _solve_astap(sub)
        except Exception as e:  # One solver's surprise leaves the other to try
            logger.debug(f"{name} solve of {sub.path} failed: {e}")
            wcs = None
        if wcs:
            result = (position_angle(wcs), name)
            if cache:
                cache.save_angle(key, *result)
            return result
    return None


# ---------------------------------------------------------------------------
# Judging a night
# ---------------------------------------------------------------------------

def _eqmode(hints):
    for hint in hints:
        value = hint.get("EQMODE")
        if isinstance(value, str):
            value = {"T": 1, "F": 0}.get(value.strip().upper(), _number(value))
        if value in (0, 1):
            return int(value)
    return None


def _site(hints, site):
    for hint in hints:
        lat, lon = site_of(hint)
        if lat is not None:
            return lat, lon
    return site or (None, None)


def _chain(subs, angles, first, last, cache, cancelled, memo=None):
    """The field's total turn from subs[first] to subs[last], measured in steps
    of at most STEP degrees of predicted turn (unwrapped, so it can pass 180),
    and the fewest stars any step matched. (None, 0) if a step's stars don't match."""
    total, fewest = 0.0, None
    k = first
    while k < last:
        if cancelled():
            return None, 0
        within = np.flatnonzero(np.abs(angles[k + 1:last + 1] - angles[k]) <= STEP)
        nxt = k + 1 + int(within[-1]) if within.size else k + 1
        turn, stars = _turn(subs[k], subs[nxt], cache, memo)
        if turn is None and nxt - k > 1:
            # Not enough stars in common: a sub in between, two shorter steps
            middle = (k + nxt) // 2
            first_half, stars_1 = _turn(subs[k], subs[middle], cache, memo)
            second_half, stars_2 = (_turn(subs[middle], subs[nxt], cache, memo) if first_half is not None
                                    else (None, 0))
            if second_half is not None:
                turn, stars = first_half + second_half, min(stars_1, stars_2)
        if turn is None:
            return None, 0
        total += turn
        fewest = stars if fewest is None else min(fewest, stars)
        k = nxt
    return total, fewest or 0


def _judge_rotation(result, subs, lat, lon, cache, cancelled, memo=None):
    """Rotation verdict for subs (time order); lat/lon for those whose own header has no site."""
    pointed = [s for s in subs if s.ra is not None and s.dec is not None]
    if len(pointed) < 2:
        result.evidence = "The subs don't say where they were pointed"
        return
    located = [(s, s.lat, s.lon) if s.lat is not None else (s, lat, lon) for s in pointed]
    located = [item for item in located if item[1] is not None and item[2] is not None]
    if len(located) < 2:
        result.evidence = "No site location to work out the field rotation from"
        return
    best = None
    for field_subs in _fields(located):
        if len(field_subs) < 2:
            continue
        angles = np.degrees(np.unwrap(np.radians(
            [parallactic_angle(s.ra, s.dec, s_lat, s_lon, s.when) for s, s_lat, s_lon in field_subs])))
        spread = float(angles.max() - angles.min())
        if best is None or spread > best[0]:
            best = (spread, [s for s, _lat, _lon in field_subs], angles)
    if best is None:
        result.evidence = "No two subs pointed at the same place"
        return
    _spread, group, angles = best
    lo, hi = int(np.argmin(angles)), int(np.argmax(angles))
    first, last = min(lo, hi), max(lo, hi)
    predicted = float(angles[last] - angles[first])
    hours = (group[last].when - group[first].when).total_seconds() / 3600.0
    apart = separation(predicted)
    if apart < MIN_SEPARATION:
        if abs(_wrap(predicted)) > 90:
            result.evidence = (f"An alt-az mount would have turned the field {abs(predicted):.0f}° over "
                               f"{hours:.1f} h - too near a meridian flip's 180° to tell them apart")
        else:
            result.evidence = (f"An alt-az mount would have turned the field only {abs(predicted):.0f}° "
                               f"over {hours:.1f} h - too little to tell")
        return
    try:
        turn, stars = _chain(group, angles, first, last, cache, cancelled, memo)
    except (OSError, ValueError) as e:
        logger.debug(f"Couldn't measure the field rotation of {group[first].path}: {e}")
        result.evidence = "Its subs couldn't be read to measure the field rotation"
        return
    if turn is None:
        result.evidence = "Too few stars matched between its subs to measure the field rotation"
        return
    to_eq = min(_apart(turn, 0), _apart(turn, 180))
    to_az = min(_apart(turn, predicted), _apart(turn, -predicted))
    alt_az = f"an alt-az mount would have turned it {abs(predicted):.0f}°"
    if to_eq < to_az and to_eq <= TOLERANCE * apart:
        flipped = _apart(turn, 180) < _apart(turn, 0)
        shown = abs(_wrap(turn))
        result.mount, result.source, result.strength = EQUATORIAL, ROTATION, ROTATION_STRENGTH + apart
        result.evidence = (f"The field turned {shown:.1f}° over {hours:.1f} h"
                           f"{' (a meridian flip)' if flipped else ''}; {alt_az}")
    elif to_az <= TOLERANCE * apart:
        result.mount, result.source, result.strength = ALT_AZ, ROTATION, ROTATION_STRENGTH + apart
        # Whole turns can't be seen between two frames: the one nearest the prediction
        shown = min((abs(turn + 360 * k) for k in (-1, 0, 1)), key=lambda t: abs(t - abs(predicted)))
        result.evidence = (f"The field turned {shown:.1f}° over {hours:.1f} h, as an alt-az mount "
                           f"turns it ({abs(predicted):.0f}°)")
    else:
        result.evidence = (f"The field turned {abs(_wrap(turn)):.1f}° over {hours:.1f} h - between an "
                           f"equatorial mount (0°) and an alt-az one ({abs(predicted):.0f}°)")
    logger.debug(f"{result.night} {result.rig}: turn {turn:.2f}, alt-az {predicted:.2f}, {stars} stars "
                 f"-> {result.mount}")


def _judge_angle(result, subs, lat, lon, cache, memo):
    """Frame-angle verdict for a fixed camera's subs, from those whose two
    possible angles were farthest apart (up to ANGLE_TRIES of them, until one
    solves). Returns why it couldn't tell, or ''."""
    candidates = []
    for sub in subs:
        sub_lat, sub_lon = (sub.lat, sub.lon) if sub.lat is not None else (lat, lon)
        if sub.ra is None or sub.dec is None or sub_lat is None or sub_lon is None:
            continue
        q = parallactic_angle(sub.ra, sub.dec, sub_lat, sub_lon, sub.when)
        candidates.append((abs(_mod180(q)), sub, q))
    if not candidates:
        return "No pointing or site to work out its frame angle from"
    candidates.sort(key=lambda c: -c[0])
    if candidates[0][0] < ANGLE_MIN_SEPARATION:
        return "The target was too near the meridian for the frame angle to tell"
    solved = None
    for apart, sub, q in [c for c in candidates if c[0] >= ANGLE_MIN_SEPARATION][:ANGLE_TRIES]:
        hint = _hint(sub, cache)
        width, height = _number(hint.get("NAXIS1")), _number(hint.get("NAXIS2"))
        if not (width and height and width < height):
            return ""  # Not the S50's own portrait frames (cropped or turned by other software)
        try:
            solved = _frame_angle(sub, cache, memo)
        except (OSError, ValueError) as e:
            logger.debug(f"Couldn't read {sub.path} for its frame angle: {e}")
        if solved is not None:
            break
    if solved is None:
        return "Its frames couldn't be plate solved for their angle"
    angle, _solver = solved
    to_eq, to_az = abs(_mod180(angle)), abs(_mod180(angle - q))
    shown, alt_az = round(angle % 180) % 180, round(q % 180) % 180
    if to_eq < to_az and to_eq <= ANGLE_TOLERANCE * apart:
        result.mount, result.source, result.strength = EQUATORIAL, ANGLE, ANGLE_STRENGTH + apart
        result.evidence = (f"Plate solved, the frame sat at {shown}° on the sky, as a Seestar's does in "
                           f"equatorial mode (alt-az would put it at {alt_az}°)")
    elif to_az < to_eq and to_az <= ANGLE_TOLERANCE * apart:
        result.mount, result.source, result.strength = ALT_AZ, ANGLE, ANGLE_STRENGTH + apart
        result.evidence = (f"Plate solved, the frame sat at {shown}° on the sky - the target's parallactic "
                           f"angle ({alt_az}°), as a Seestar's does in alt-az mode (equatorial would put it at 0°)")
    else:
        return (f"Plate solved, the frame sat at {shown}° - neither equatorial mode's 0° nor alt-az's "
                f"{alt_az}°")
    return ""


def detect_night(night, rig, subs, site=None, cache=None, cancelled=None):
    """A NightMount for one night's subs of one rig. site: (lat, lon) when the
    headers don't say where they were taken."""
    cancelled = cancelled or (lambda: False)
    subs = sorted(subs, key=lambda s: s.when)
    result = NightMount(night, rig, subs)
    if not subs:
        return result
    hints = [_hint(subs[0], cache)] + ([_hint(subs[-1], cache)] if len(subs) > 1 else [])
    lat, lon = _site(hints, site)
    memo = {}   # The night's subs read so far (stars and header) - read once for every check
    if len(subs) > 1:
        _judge_rotation(result, subs, lat, lon, cache, cancelled, memo)
    else:
        result.evidence = "One sub - nothing to measure the field rotation against"
    if result.mount is None and fixed_camera(rig) and not cancelled():
        why = _judge_angle(result, subs, lat, lon, cache, memo)
        if why:
            result.evidence = f"{result.evidence}. {why}" if result.evidence else why

    # What the capture app wrote: used when nothing was measured, and noted when a measurement disagrees
    eqmode = _eqmode(hints)
    if eqmode is not None:
        written = EQUATORIAL if eqmode else ALT_AZ
        if result.mount is None:
            result.mount, result.source, result.strength = written, HEADER, HEADER_STRENGTH
            result.evidence = f"The header says it was in {'equatorial' if eqmode else 'alt-az'} mode (EQMODE)"
        elif result.mount != written:
            result.evidence += (f" - though the header says {'equatorial' if eqmode else 'alt-az'} mode "
                                f"(EQMODE, as the capture app wrote it)")
    if result.mount is None:
        sides = {str(h.get("PIERSIDE") or "").strip().title() for h in hints} & {"East", "West"}
        if sides:
            result.mount, result.source, result.strength = EQUATORIAL, PIER_SIDE, PIER_SIDE_STRENGTH
            result.evidence = ("The mount changed pier sides (a meridian flip)" if len(sides) > 1 else
                               f"The mount reported its pier side ({sides.pop()})") + \
                " - only an equatorial mount has one"
    return result


def share_same_night(mounts):
    """Nights that couldn't be told take what the same rig showed on another
    target that night (its strongest verdict)."""
    best = {}
    for mount in mounts:
        if mount.mount and mount.source != SAME_NIGHT:
            key = (mount.night, mount.rig)
            if key not in best or mount.strength > best[key].strength:
                best[key] = mount
    for mount in mounts:
        other = best.get((mount.night, mount.rig)) if not mount.mount else None
        if other is not None:
            mount.mount, mount.source, mount.strength = other.mount, SAME_NIGHT, SAME_NIGHT_STRENGTH
            mount.evidence = _borrowed(other.evidence)
    return mounts


def _borrowed(evidence):
    return f"Same camera and telescope that night, on another target: {evidence[:1].lower()}{evidence[1:]}"


def detect(nights, site=None, cache=None, workers=WORKERS, progress=None, cancel=None):
    """[NightMount] for [(night, rig, [Sub])], in that order, with same-night
    verdicts shared. progress(done, total) after each night; cancel: a
    threading.Event - nights not checked by then are left out."""
    cancelled = cancel.is_set if cancel is not None else (lambda: False)
    results = [None] * len(nights)
    if not nights:
        return []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="mount-detect") as pool:
        futures = {pool.submit(_detect_guarded, night, rig, subs, site, cache, cancelled): index
                   for index, (night, rig, subs) in enumerate(nights)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if progress:
                progress(done, len(nights))
            if cancelled():
                for pending in futures:
                    pending.cancel()
                break
    return share_same_night([r for r in results if r is not None])


def _detect_guarded(night, rig, subs, site, cache, cancelled):
    if cancelled():
        return None
    try:
        result = detect_night(night, rig, subs, site, cache, cancelled)
        return None if cancelled() else result  # Stopped part way: not a verdict
    except Exception as e:  # One night's surprise never stops a scan
        logger.warning(f"Mount detection failed for the night of {night}: {e}", exc_info=True)
        return NightMount(night, rig, list(subs), evidence="Couldn't be checked")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def ensure_schema(conn):
    """The verdict table, the file and session columns pointing at it, and
    verdicts no file points at any more removed. Does not commit."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usersessionmounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            night_date TEXT,        -- the observing night the subs were taken
            rig TEXT,               -- their camera|telescope names, for sharing a night's verdict
            mount_type TEXT,        -- 'Equatorial', 'Alt-Az' or NULL (couldn't tell)
            source TEXT,            -- header, rotation, pier side, same night
            evidence TEXT,
            strength REAL DEFAULT 0
        )
    """)
    for table, column in (("usersessionfiles", "mount_id INTEGER"), ("usersessions", "mount_override TEXT")):
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessionfiles_mount_id ON usersessionfiles(mount_id)")
        _remove_orphans(cursor)
    except sqlite3.OperationalError as e:
        logger.debug(f"No session files table yet: {e}")


def _remove_orphans(cursor):
    cursor.execute("DELETE FROM usersessionmounts WHERE id NOT IN "
                   "(SELECT mount_id FROM usersessionfiles WHERE mount_id IS NOT NULL)")


def save(conn, session_id, mounts, tidy=True):
    """Point the session's light files at their nights' verdicts - matched by
    path, else by sub fingerprint - replacing what they had. With tidy, then
    tidy_saved(): pass False when saving many sessions (after ensure_schema),
    and call tidy_saved() once after. Does not commit."""
    if tidy:
        ensure_schema(conn)
    cursor = conn.cursor()
    by_path, by_key = {}, {}
    for file_id, path, sub_key in cursor.execute(
            "SELECT id, file_path, sub_key FROM usersessionfiles WHERE session_id = ? AND frame_type = 'Light'",
            (session_id,)).fetchall():
        by_path[os.path.normcase(os.path.normpath(path or ""))] = file_id
        if sub_key:
            by_key[sub_key] = file_id
    for mount in mounts:
        ids = {by_path.get(os.path.normcase(os.path.normpath(s.path))) or by_key.get(s.sub_key)
               for s in mount.subs} - {None}
        if not ids:
            continue
        cursor.execute("INSERT INTO usersessionmounts (night_date, rig, mount_type, source, evidence, strength) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (mount.night.isoformat() if mount.night else None, mount.rig, mount.mount, mount.source,
                        mount.evidence, mount.strength))
        mount_id = cursor.lastrowid
        ids = sorted(ids)
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            cursor.execute(f"UPDATE usersessionfiles SET mount_id = ? WHERE id IN ({','.join('?' * len(chunk))})",
                           [mount_id] + chunk)
    if tidy:
        tidy_saved(conn)


def tidy_saved(conn):
    """Remove verdicts no file points at any more, and give undecided nights
    what the same rig showed that night in another session. Does not commit."""
    _remove_orphans(conn.cursor())
    share_same_night_saved(conn)


def share_same_night_saved(conn):
    """share_same_night() across the saved nights of every session. Does not commit."""
    cursor = conn.cursor()
    best = {}
    for night, rig, mount_type, evidence, strength in cursor.execute(
            "SELECT night_date, rig, mount_type, evidence, strength FROM usersessionmounts "
            "WHERE mount_type IS NOT NULL AND source != ?", (SAME_NIGHT,)).fetchall():
        if (night, rig) not in best or (strength or 0) > best[(night, rig)][2]:
            best[(night, rig)] = (mount_type, evidence or "", strength or 0)
    for mount_id, night, rig in cursor.execute(
            "SELECT id, night_date, rig FROM usersessionmounts WHERE mount_type IS NULL").fetchall():
        other = best.get((night, rig))
        if other:
            cursor.execute("UPDATE usersessionmounts SET mount_type = ?, source = ?, evidence = ?, strength = ? "
                           "WHERE id = ?", (other[0], SAME_NIGHT, _borrowed(other[1]), SAME_NIGHT_STRENGTH, mount_id))


def detected_sql(alias="s"):
    """SQL for a session's detected mounts: 'Equatorial', 'Alt-Az' or both
    comma-separated (NULL when none) - pair with summarize()."""
    return (f"(SELECT GROUP_CONCAT(DISTINCT m.mount_type) FROM usersessionfiles f "
            f"JOIN usersessionmounts m ON m.id = f.mount_id "
            f"WHERE f.session_id = {alias}.id AND m.mount_type IS NOT NULL)")


def observation_mounts(conn, session_id):
    """{observation id: (mount or None, source, evidence)} - each night's
    strongest verdict among its files'."""
    best = {}
    try:
        rows = conn.execute("""
            SELECT f.observation_id, m.mount_type, m.source, m.evidence, m.strength
            FROM usersessionfiles f JOIN usersessionmounts m ON m.id = f.mount_id
            WHERE f.session_id = ? AND f.observation_id IS NOT NULL
            GROUP BY f.observation_id, m.id
        """, (session_id,)).fetchall()
    except sqlite3.OperationalError:
        return {}   # A database from before mount detection
    for observation_id, mount_type, source, evidence, strength in rows:
        rank = (mount_type is not None, strength or 0)
        if observation_id not in best or rank > best[observation_id][0]:
            best[observation_id] = (rank, (mount_type, source, evidence or ""))
    return {obs_id: value for obs_id, (_rank, value) in best.items()}


def summarize(detected, override=None, equipment_mount=None):
    """(mount, how) a session shows: the user's own choice, else what its
    nights showed ('Mixed' when they differ), else its telescope's Mount in
    Equipment. detected: detected_sql()'s value or a collection of mounts.
    how is 'set', 'detected', 'equipment' or '' (mount None: unknown)."""
    if override in (EQUATORIAL, ALT_AZ):
        return override, "set"
    if isinstance(detected, str):
        detected = detected.split(",")
    kinds = {m for m in (detected or ()) if m in (EQUATORIAL, ALT_AZ)}
    if len(kinds) == 1:
        return kinds.pop(), "detected"
    if kinds:
        return MIXED, "detected"
    fallback = EQUIPMENT_MOUNTS.get(equipment_mount or "")
    return (fallback, "equipment") if fallback else (None, "")


def describe(mount, how):
    """'Equatorial', 'Alt-Az (from Equipment)', 'Equatorial (set by you)', 'Unknown'."""
    if not mount:
        return "Unknown"
    return mount + {"set": " (set by you)", "equipment": " (from Equipment)"}.get(how, "")


def count_text(mounts):
    """'Equatorial', or 'Equatorial on 3 nights, Alt-Az on 1' - [mount or None] per night."""
    counts = defaultdict(int)
    for mount in mounts:
        counts[mount or "Unknown"] += 1
    if len(counts) == 1:
        return next(iter(counts))
    order = (EQUATORIAL, ALT_AZ, "Unknown")
    parts = [f"{kind} on {counts[kind]} night{'s' if counts[kind] != 1 else ''}" for kind in order if counts[kind]]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Tell each night's mount (equatorial or alt-az) from its lights.")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--lat", type=float, help="Site latitude when the headers don't have it")
    parser.add_argument("--lon", type=float, help="Site longitude when the headers don't have it")
    parser.add_argument("--no-cache", action="store_true", help="Measure again, without the scan index's cache")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = []
    for folder in args.folders:
        files.extend(SessionFileScanner.scan_folder(folder))
    site = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None
    cache = None if args.no_cache else MountCache()
    nights = nights_from_file_dicts(files, args.lon)
    for mount in detect(nights, site, cache, progress=lambda done, total: print(f"\r{done}/{total}", end="")):
        print(f"\r{mount.night}  {len(mount.subs):5} subs  {mount.mount or 'Unknown':10} ({mount.source or '-'}) "
              f"{mount.evidence}")
    if cache:
        cache.close()


if __name__ == "__main__":
    main(sys.argv[1:])
