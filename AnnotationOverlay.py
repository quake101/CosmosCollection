#!/usr/bin/env python3
"""
Annotation Overlay Module for Cosmos Collection
Renders astronomical annotations (DSOs, stars, constellations, grid) on plate-solved images,
and plate solves images in the background (PlateSolveQueue) - shared by every window
that shows annotations.
"""

import os
import re
import json
import math
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, replace

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject, QThread, QRunnable, QThreadPool, QSettings
from PySide6.QtGui import (QPainter, QPen, QColor, QFont, QFontMetrics, QFontMetricsF, QPolygonF, QPainterPath,
                           QTransform)
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


@dataclass
class CelestialObject:
    """Represents a celestial object for annotation"""
    name: str
    ra: float  # degrees
    dec: float  # degrees
    magnitude: float = None
    obj_type: str = None  # 'galaxy', 'nebula', 'cluster', 'star', etc.
    size: float = None  # arcminutes


@dataclass
class ConstellationLine:
    """Represents a constellation line segment"""
    constellation: str
    ra1: float
    dec1: float
    ra2: float
    dec2: float


class WCSTransform:
    """
    Simple WCS transformation for converting between pixel and sky coordinates
    Supports basic TAN (gnomonic) projection
    """

    def __init__(self, wcs_header: Dict[str, Any], image_width: int, image_height: int):
        self.header = wcs_header
        self.width = image_width
        self.height = image_height

        # Reference pixel (FITS 1-based coordinates)
        self.crpix1 = wcs_header.get('CRPIX1', image_width / 2)
        self.crpix2 = wcs_header.get('CRPIX2', image_height / 2)

        # Reference coordinates (degrees)
        self.crval1 = wcs_header.get('CRVAL1', 0)  # RA
        self.crval2 = wcs_header.get('CRVAL2', 0)  # Dec

        # CD matrix (degrees per pixel)
        self.cd1_1 = wcs_header.get('CD1_1', wcs_header.get('CDELT1', -0.001))
        self.cd1_2 = wcs_header.get('CD1_2', 0)
        self.cd2_1 = wcs_header.get('CD2_1', 0)
        self.cd2_2 = wcs_header.get('CD2_2', wcs_header.get('CDELT2', 0.001))

        # Calculate inverse CD matrix
        det = self.cd1_1 * self.cd2_2 - self.cd1_2 * self.cd2_1
        if abs(det) > 1e-10:
            self.cdinv1_1 = self.cd2_2 / det
            self.cdinv1_2 = -self.cd1_2 / det
            self.cdinv2_1 = -self.cd2_1 / det
            self.cdinv2_2 = self.cd1_1 / det
        else:
            self.cdinv1_1 = 1.0
            self.cdinv1_2 = 0.0
            self.cdinv2_1 = 0.0
            self.cdinv2_2 = 1.0

    def sky_to_pixel(self, ra: float, dec: float) -> Tuple[float, float]:
        """Convert RA/Dec (degrees) to display pixel coordinates (0-based)"""
        # Convert to radians
        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)
        ra0_rad = math.radians(self.crval1)
        dec0_rad = math.radians(self.crval2)

        # Gnomonic (TAN) projection
        cos_dec = math.cos(dec_rad)
        sin_dec = math.sin(dec_rad)
        cos_dec0 = math.cos(dec0_rad)
        sin_dec0 = math.sin(dec0_rad)
        cos_dra = math.cos(ra_rad - ra0_rad)
        sin_dra = math.sin(ra_rad - ra0_rad)

        denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra

        if denom <= 0:
            return None, None  # Object behind projection plane

        xi = (cos_dec * sin_dra) / denom
        eta = (sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra) / denom

        # Convert to degrees
        xi_deg = math.degrees(xi)
        eta_deg = math.degrees(eta)

        # Apply inverse CD matrix to get pixel offset
        dx = self.cdinv1_1 * xi_deg + self.cdinv1_2 * eta_deg
        dy = self.cdinv2_1 * xi_deg + self.cdinv2_2 * eta_deg

        # Calculate FITS pixel coordinates (1-based, y increases upward)
        x_fits = self.crpix1 + dx
        y_fits = self.crpix2 + dy

        # Convert to display coordinates:
        # - FITS is 1-based, display is 0-based (subtract 1)
        # - FITS has y increasing upward, display has y increasing downward (flip y)
        x = x_fits - 1
        y = self.height - y_fits

        return x, y

    def pixel_to_sky(self, x: float, y: float) -> Tuple[float, float]:
        """Convert display pixel coordinates (0-based, y down) to RA/Dec (degrees)"""
        # Convert display coordinates to FITS coordinates:
        # - Display is 0-based, FITS is 1-based (add 1)
        # - Display has y increasing downward, FITS has y increasing upward (flip y)
        x_fits = x + 1
        y_fits = self.height - y

        # Pixel offset from reference
        dx = x_fits - self.crpix1
        dy = y_fits - self.crpix2

        # Apply CD matrix to get projection coordinates (degrees)
        xi_deg = self.cd1_1 * dx + self.cd1_2 * dy
        eta_deg = self.cd2_1 * dx + self.cd2_2 * dy

        # Convert to radians
        xi = math.radians(xi_deg)
        eta = math.radians(eta_deg)

        ra0_rad = math.radians(self.crval1)
        dec0_rad = math.radians(self.crval2)

        # Inverse gnomonic projection
        rho = math.sqrt(xi * xi + eta * eta)
        c = math.atan(rho)

        if rho > 0:
            dec_rad = math.asin(math.cos(c) * math.sin(dec0_rad) +
                                eta * math.sin(c) * math.cos(dec0_rad) / rho)
            ra_rad = ra0_rad + math.atan2(xi * math.sin(c),
                                           rho * math.cos(dec0_rad) * math.cos(c) -
                                           eta * math.sin(dec0_rad) * math.sin(c))
        else:
            dec_rad = dec0_rad
            ra_rad = ra0_rad

        return math.degrees(ra_rad), math.degrees(dec_rad)

    def get_field_of_view(self) -> Tuple[float, float, float, float]:
        """Get the field of view bounds (min_ra, max_ra, min_dec, max_dec) in degrees"""
        corners = [
            self.pixel_to_sky(0, 0),
            self.pixel_to_sky(self.width, 0),
            self.pixel_to_sky(0, self.height),
            self.pixel_to_sky(self.width, self.height),
        ]

        ras = [c[0] for c in corners]
        decs = [c[1] for c in corners]

        return min(ras), max(ras), min(decs), max(decs)


def flip_wcs_rows(wcs_header: Dict[str, Any], image_height: int) -> Dict[str, Any]:
    """Return a copy of a WCS header with its y axis reversed.

    ASTAP numbers a FITS file's rows in stored order (first row = y 1) and
    ignores ROWORDER, and FITS images are displayed with the first stored row
    at the top. WCSTransform expects y to count up from the bottom displayed
    row, as in ASTAP's solutions of PNG/JPG images, so FITS solutions need
    their y axis reversed before use.
    """
    flipped = dict(wcs_header)
    if 'CRPIX2' in wcs_header:
        flipped['CRPIX2'] = image_height + 1 - wcs_header['CRPIX2']
    # Reversing y negates the matrix's y column. With a PC matrix CDELTn scales
    # rows, so the PC column is negated rather than CDELT2.
    if 'CD1_1' in wcs_header or 'CD2_2' in wcs_header:
        column = {'CD1_2': 0.0, 'CD2_2': 0.0}
    elif any(key.startswith('PC') for key in wcs_header):
        column = {'PC1_2': 0.0, 'PC2_2': 1.0}  # Missing PC terms default to the identity
    else:
        column = {'CDELT2': 1.0}  # CDELT/CROTA2 form: CDELT2 only appears in the y column
    for key, default in column.items():
        flipped[key] = -float(wcs_header.get(key, default))
    return flipped


# SIMBAD object types containing '*' that are groups of stars, not stars
NON_STAR_OTYPES = {'Cl*', 'As*', 'St*', 'MGr', '**?'}


def query_bright_stars(ra: float, dec: float, radius: float, magnitude_limit: float = 8.0) -> List[CelestialObject]:
    """Named stars brighter than magnitude_limit within radius (degrees) of
    ra, dec from SIMBAD (online) - the brightest 100. None if it can't be reached."""
    stars = []
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        simbad = Simbad()
        simbad.add_votable_fields('flux(V)', 'ids', 'otype')

        logger.info(f"Querying SIMBAD for stars at RA={ra:.2f}, Dec={dec:.2f}, radius={radius:.2f}")
        from OnlineLimits import online_request
        with online_request("simbad"):
            result = simbad.query_region(coord, radius=radius * u.deg)

        if result is not None:
            # astroquery 0.4.8+ returns lowercase columns (main_id, ra, dec in degrees);
            # older versions MAIN_ID, RA/DEC as sexagesimal strings
            columns = {name.lower(): name for name in result.colnames}
            mag_column = columns.get('flux_v') or columns.get('v')
            for row in result:
                try:
                    mag = row[mag_column] if mag_column else None
                    if mag is None or (hasattr(mag, 'mask') and mag.mask) or mag > magnitude_limit:
                        continue

                    # Galaxies and nebulae have V magnitudes too; they're labeled
                    # from the DSO catalog. SIMBAD star types all contain '*'
                    # (except groupings of stars).
                    otype = row[columns['otype']] if 'otype' in columns else '*'
                    if isinstance(otype, bytes):
                        otype = otype.decode('utf-8')
                    if '*' not in str(otype) or str(otype) in NON_STAR_OTYPES:
                        continue

                    ra_obj = row[columns['ra']]
                    dec_obj = row[columns['dec']]
                    if isinstance(ra_obj, (str, bytes)):
                        coord_obj = SkyCoord(ra_obj, dec_obj, unit=(u.hourangle, u.deg))
                    else:
                        coord_obj = SkyCoord(float(ra_obj), float(dec_obj), unit='deg')

                    name = row[columns['main_id']]
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')

                    stars.append(CelestialObject(name=name, ra=coord_obj.ra.deg, dec=coord_obj.dec.deg,
                                                 magnitude=float(mag) if mag else None, obj_type='star'))
                except Exception as row_err:
                    logger.debug(f"Failed to parse star row: {row_err}")
            logger.info(f"SIMBAD query: {len(stars)} stars brighter than magnitude {magnitude_limit}")

    except ImportError:
        logger.warning("astroquery not available for star queries")
        return None
    except Exception as e:
        logger.warning(f"Star query failed: {e}")
        return None

    stars.sort(key=lambda star: star.magnitude if star.magnitude is not None else 99)
    return stars[:100]


def stars_in_view(wcs, width, height, magnitude_limit=8.0):
    """Named bright stars inside an image of width x height pixels (SIMBAD,
    online) - None if SIMBAD can't be reached."""
    min_ra, max_ra, min_dec, max_dec = wcs.get_field_of_view()
    radius = max(abs(max_ra - min_ra), abs(max_dec - min_dec)) / 2
    found = query_bright_stars((min_ra + max_ra) / 2, (min_dec + max_dec) / 2, radius, magnitude_limit)
    if found is None:
        return None
    inside = []
    for star in found:
        x, y = wcs.sky_to_pixel(star.ra, star.dec)
        if x is not None and 0 <= x <= width and 0 <= y <= height:
            inside.append(star)
    return inside


# Constellation line data (abbreviated - major constellations)
CONSTELLATION_LINES = [
    # Orion
    ('Orion', 88.79, 7.41, 81.28, -1.94),
    ('Orion', 81.28, -1.94, 78.63, -8.20),
    ('Orion', 81.28, -1.94, 83.00, -0.30),
    ('Orion', 83.00, -0.30, 83.86, -5.91),
    ('Orion', 83.86, -5.91, 84.05, -1.20),
    ('Orion', 84.05, -1.20, 85.19, -1.94),
    ('Orion', 85.19, -1.94, 88.79, 7.41),
    # Ursa Major (Big Dipper)
    ('UMa', 165.46, 61.75, 178.46, 53.69),
    ('UMa', 178.46, 53.69, 183.86, 57.03),
    ('UMa', 183.86, 57.03, 193.51, 55.96),
    ('UMa', 193.51, 55.96, 200.98, 54.93),
    ('UMa', 200.98, 54.93, 206.89, 49.31),
    ('UMa', 206.89, 49.31, 210.75, 56.38),
    # Add more constellations as needed...
]


# --------------------------------------------------------------------
# Plate solutions - .wcs files next to images, or a FITS file's own header
# --------------------------------------------------------------------

MAX_ANNOTATED_DSOS = 30   # Brightest objects per image, so busy fields stay readable
FITS_EXTENSIONS = (".fits", ".fit", ".fts")


def read_fits_cards(path):
    """A FITS file's primary header as {keyword: value} (numbers as numbers,
    quoted strings unquoted). Header cards are 80-character text, read
    directly - astropy takes seconds to import just for this."""
    cards = {}
    with open(path, "rb") as file:
        for _block in range(64):  # Headers are a few 2880-byte blocks
            block = file.read(2880)
            if len(block) < 80:
                break
            for start in range(0, len(block) - 79, 80):
                card = block[start:start + 80].decode("ascii", "replace")
                if card.startswith("END") and not card[3:].strip():
                    return cards
                if card[8:10] != "= ":
                    continue
                key, value = card[:8].strip(), card[10:].strip()
                if value.startswith("'"):  # A quoted string; "/" starts a comment only outside quotes
                    cards[key] = value[1:].split("'")[0].strip()
                    continue
                value = value.split("/")[0].strip()
                try:
                    cards[key] = int(value) if re.fullmatch(r"[+-]?\d+", value) else float(value)
                except ValueError:
                    cards[key] = value
    return cards


def read_wcs_header(path):
    """An image's plate solution as WCS keywords: its .wcs file (written by
    ASTAP, or by a solve here), else a FITS file's own header - or None."""
    wcs_file = os.path.splitext(path)[0] + ".wcs"
    header = {}
    if os.path.isfile(wcs_file):
        try:
            with open(wcs_file, "r", errors="replace") as file:
                text = file.read()
            # One card per line, or 80-character cards run together
            lines = text.splitlines() if "\n" in text else [text[i:i + 80] for i in range(0, len(text), 80)]
            for line in lines:
                if "=" not in line[:10]:
                    continue
                key, value = line.split("=", 1)
                value = value.strip()
                if value.startswith("'"):
                    header[key.strip()] = value[1:].split("'")[0].strip()
                    continue
                value = value.split("/")[0].strip()
                try:
                    header[key.strip()] = float(value) if re.search(r"[.eE]", value) else int(value)
                except ValueError:
                    header[key.strip()] = value
        except OSError:
            header = {}
    elif os.path.splitext(path)[1].lower() in FITS_EXTENSIONS and os.path.isfile(path):
        header = read_fits_cards(path)
    return header if is_solved_header(header) else None


def is_solved_header(header):
    """Whether WCS keywords are a plate solution."""
    return bool(header) and "CRVAL1" in header and "CRPIX1" in header and ("CD1_1" in header or "CDELT1" in header)


def image_pixel_size(path):
    """(width, height) of an image file, read from its header - not decoded."""
    ext = os.path.splitext(path)[1].lower()
    if ext in FITS_EXTENSIONS:
        cards = read_fits_cards(path)
        return int(cards.get("NAXIS1", 0)), int(cards.get("NAXIS2", 0))
    if ext == ".xisf":
        from SessionFileScanner import read_xisf_xml
        _root, image, _ns = read_xisf_xml(path)
        width, height = (image.get("geometry", "0:0").split(":") + ["0"])[:2] if image is not None else (0, 0)
        return int(width), int(height)
    from PySide6.QtGui import QImageReader
    size = QImageReader(path).size()
    return size.width(), size.height()


def write_wcs_file(path, wcs_header):
    """Save a plate solution next to the image as a .wcs file (like ASTAP
    does), so it's found next time - by every window."""
    wcs_file = os.path.splitext(path)[0] + ".wcs"
    lines = []
    for key, value in wcs_header.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z0-9_-]{1,8}", key):
            continue
        text = f"'{value}'" if isinstance(value, str) else repr(float(value)) if isinstance(value, float) else str(value)
        lines.append(f"{key:<8}= {text}")
    lines.append("END")
    with open(wcs_file, "w") as file:
        file.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------
# What a plate-solved image shows
# --------------------------------------------------------------------

@dataclass
class AnnotationData:
    """What a plate-solved image can show: its solution (pixel <-> sky, for
    the full-size image as displayed) and the objects in it."""
    wcs: object               # WCSTransform
    width: int
    height: int
    dsos: list                # CelestialObject, brightest first
    stars: list = None        # None until looked up (online)

    @property
    def degrees_per_pixel(self):
        w = self.wcs
        return math.sqrt(abs(w.cd1_1 * w.cd2_2 - w.cd1_2 * w.cd2_1)) or 1e-4

    @classmethod
    def from_header(cls, header, width, height, fits_rows=False):
        """AnnotationData for a solution held in memory (e.g. NINA's). fits_rows:
        the solution numbers a FITS file's rows in stored order (see flip_wcs_rows).
        Looks up the local catalog - safe in a worker thread."""
        if fits_rows:
            header = flip_wcs_rows(header, height)
        wcs = WCSTransform(header, width, height)
        return cls(wcs, width, height, _dsos_in_view(wcs, width, height))


# Catalog entries that aren't deep-sky objects: stars the NGC/IC listed, and
# entries for objects that don't exist
NOT_DSO_TYPES = {"1STAR", "2STAR", "3STAR", "4STAR", "NONEX"}
UNKNOWN_MAGNITUDE = 50    # The catalog's stand-ins for "unknown" are 79.9 and 99.9


def _annotation_priority(item):
    """Sort key: Messier, then NGC/IC, then other catalogs (dark nebulae last);
    within those, brightest first, then largest."""
    name, kind = item.name or "", (item.obj_type or "").upper()
    if name.startswith("M "):
        rank = 0
    elif name.startswith(("NGC ", "IC ")):
        rank = 1
    else:
        rank = 3 if kind == "DRKNB" else 2
    magnitude = item.magnitude if item.magnitude is not None and item.magnitude < UNKNOWN_MAGNITUDE else 99
    return rank, magnitude, -(item.size or 0)


def _dsos_in_view(wcs, width, height):
    """Catalog objects inside the image, most notable first (the local catalog)."""
    import sqlite3
    from ResourceManager import ResourceManager, attach_update_catalogs
    min_ra, max_ra, min_dec, max_dec = wcs.get_field_of_view()
    conn = sqlite3.connect(str(ResourceManager.get_database_path()))
    try:
        attach_update_catalogs(conn)
        rows = conn.execute("""
            SELECT d.ra, d.dec, d.magnitude, d.dsotype, COALESCE(d.sizemax, d.sizemin) / 60.0,
                   d.sizemin / 60.0, d.positionangle,
                   (SELECT c.catalogue || ' ' || c.designation FROM cataloguenr c WHERE c.dsodetailid = d.id
                    ORDER BY CASE c.catalogue WHEN 'M' THEN 1 WHEN 'NGC' THEN 2 WHEN 'IC' THEN 3 ELSE 4 END
                    LIMIT 1)
            FROM dsodetail d
            WHERE d.ra BETWEEN ? AND ? AND d.dec BETWEEN ? AND ?
        """, (min_ra, max_ra, min_dec, max_dec)).fetchall()
    finally:
        conn.close()
    found, names = [], set()
    for ra, dec, magnitude, kind, size, minor, angle, name in rows:
        if ra is None or dec is None or not name or name in names or (kind or "").upper() in NOT_DSO_TYPES:
            continue
        x, y = wcs.sky_to_pixel(ra, dec)
        if x is not None and 0 <= x <= width and 0 <= y <= height:
            names.add(name)
            item = CelestialObject(name=name, ra=ra, dec=dec, magnitude=magnitude, obj_type=kind, size=size)
            item.minor, item.angle = minor, angle  # Minor axis (arcmin), position angle (deg, N through E)
            found.append(item)
    found.sort(key=_annotation_priority)
    return found[:MAX_ANNOTATED_DSOS]


def _stars_cache_file(path):
    from ResourceManager import ResourceManager
    wcs_file = os.path.splitext(path)[0] + ".wcs"
    key = f"stars|{os.path.normcase(os.path.abspath(path))}|{os.path.getmtime(wcs_file) if os.path.isfile(wcs_file) else 0}"
    return ResourceManager.get_cache_dir("annotations") / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"


def _bright_stars(wcs, width, height, path):
    """Named bright stars in the image from SIMBAD (online), cached on disk."""
    cache = _stars_cache_file(path)
    if cache.exists():
        try:
            return [CelestialObject(**star) for star in json.loads(cache.read_text())]
        except (ValueError, TypeError):
            pass
    stars = stars_in_view(wcs, width, height)
    if stars is None:
        return []  # Offline - not cached, so it's looked up again next time
    try:
        cache.write_text(json.dumps([asdict(star) for star in stars]))
    except OSError:
        pass
    return stars


def load_annotation_data(path, want_stars=False):
    """AnnotationData for an image, or None if it isn't plate solved. Safe in
    a worker thread (its own database connection)."""
    header = read_wcs_header(path) if path else None
    if header is None:
        return None
    width, height = image_pixel_size(path)
    if width <= 0 or height <= 0:
        return None
    # FITS rows are shown top-down (as in the Image Viewer)
    data = AnnotationData.from_header(header, width, height,
                                      fits_rows=os.path.splitext(path)[1].lower() in FITS_EXTENSIONS)
    if want_stars:
        data.stars = _bright_stars(data.wcs, width, height, path)
    return data


class _AnnotationSignals(QObject):
    loaded = Signal(str, object, str, bool)  # path, AnnotationData (None: not solved), error, with stars


class _AnnotationTask(QRunnable):
    """Loads an image's annotations - the objects, then (if wanted) queues the
    star names, which can take a while online, in star_pool."""

    def __init__(self, path, want_stars, signals, star_pool):
        super().__init__()
        self.path, self.want_stars, self.signals, self.star_pool = path, want_stars, signals, star_pool

    def _emit(self, data, error, with_stars):
        _emit_loaded(self.signals, self.path, data, error, with_stars)

    def run(self):
        try:
            data = load_annotation_data(self.path)
        except Exception as e:
            logger.warning(f"Could not load annotations for {self.path}: {e}")
            self._emit(None, str(e), self.want_stars)
            return
        if data is None or not self.want_stars:
            self._emit(data, "", self.want_stars)
            return
        self._emit(data, "", False)  # The objects show while the stars are looked up
        self.star_pool.start(_StarTask(self.path, data, self.signals))


class _StarTask(QRunnable):
    """Looks up an image's star names (online) for its AnnotationData."""

    def __init__(self, path, data, signals):
        super().__init__()
        self.path, self.data, self.signals = path, data, signals

    def run(self):
        try:
            stars = _bright_stars(self.data.wcs, self.data.width, self.data.height, self.path)
        except Exception as e:
            logger.warning(f"Could not look up stars for {self.path}: {e}")
            stars = []
        _emit_loaded(self.signals, self.path, replace(self.data, stars=stars), "", True)


def _emit_loaded(signals, path, data, error, with_stars):
    try:
        signals.loaded.emit(path, data, error, with_stars)
    except RuntimeError:
        pass


class AnnotationStore(QObject):
    """AnnotationData by image path, loaded in the background on request.
    ready(path) fires when one is known (solved or not)."""

    ready = Signal(str)

    def __init__(self):
        super().__init__()
        self._data = {}       # path -> AnnotationData, or None if not plate solved
        self._stars = set()   # paths whose data includes stars
        self._pending = {}    # path -> wants stars
        self._signals = _AnnotationSignals()
        self._signals.loaded.connect(self._on_loaded)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(2)
        # Star names come from SIMBAD (online), as many at once as it's
        # allowed (OnlineLimits) - in their own queue, so the objects of
        # other images don't wait behind them
        from OnlineLimits import LIMITS
        self._star_pool = QThreadPool()
        self._star_pool.setMaxThreadCount(LIMITS["simbad"])

    def get(self, path, want_stars=False):
        """The image's AnnotationData, None if not plate solved, or None while loading (see known())."""
        if not path:
            return None
        known = path in self._data
        if not known or (want_stars and self._data[path] is not None and path not in self._stars):
            if path not in self._pending or (want_stars and not self._pending[path]):
                self._pending[path] = want_stars
                self._pool.start(_AnnotationTask(path, want_stars, self._signals, self._star_pool))
        return self._data.get(path)

    def known(self, path):
        return path in self._data

    def star_progress(self, paths):
        """(done, total): how many of the plate-solved images among paths have
        their star names - for showing progress while they're looked up."""
        solved = [path for path in set(paths) if self._data.get(path) is not None]
        return sum(1 for path in solved if path in self._stars), len(solved)

    def is_solved(self, path):
        """True / False, or None if not checked yet."""
        return (self._data[path] is not None) if path in self._data else None

    def forget(self, path):
        """Reload next time (e.g. after a new plate solve)."""
        self._data.pop(path, None)
        self._stars.discard(path)

    def _on_loaded(self, path, data, _error, with_stars):
        # Done with the request, unless a newer one for the stars is still running
        if with_stars or not self._pending.get(path):
            self._pending.pop(path, None)
        # A load without stars never replaces one with them
        if with_stars or path not in self._stars:
            self._data[path] = data
            if data is not None and with_stars:
                self._stars.add(path)
        self.ready.emit(path)


_annotation_store = None


def annotation_store():
    global _annotation_store
    if _annotation_store is None:
        _annotation_store = AnnotationStore()
    return _annotation_store


# --------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------

ANNOTATION_LAYERS = (("dsos", "DSO markers and names"), ("dark", "Dark nebulae (LDN, Barnard)"),
                     ("stars", "Star names (online lookup)"), ("grid", "Coordinate grid"),
                     ("lines", "Constellation lines"))
# Dark nebulae: part of the DSOs (only shown with them), on unless switched off -
# they crowd fields like the Heart Nebula
DEFAULT_ANNOTATION_LAYERS = {"dsos": True, "dark": True, "stars": False, "grid": False, "lines": False}
NAME_SCALE = 0.03375      # DSO name height, as a fraction of the drawn image's short side

# Settings keys of the layers chosen in the Image Viewer's Layers menu
# (also used by the NINA live stack)
ANNOTATION_LAYER_SETTINGS = {"dsos": "annotation_show_dsos", "dark": "annotation_show_dark_nebulae",
                             "stars": "annotation_show_stars", "grid": "annotation_show_grid",
                             "lines": "annotation_show_constellations"}


def saved_annotation_layers():
    """The Image Viewer's annotation layers: {"dsos", "stars", "grid", "lines"} -> bool"""
    settings = QSettings("CosmosCollection", "CosmosCollection")
    return {key: settings.value(setting, DEFAULT_ANNOTATION_LAYERS[key], type=bool)
            for key, setting in ANNOTATION_LAYER_SETTINGS.items()}


def screen_name_px(target):
    """DSO name height for an image drawn into target on screen: in step with
    the image, but kept readable when zoomed out and not huge when zoomed in."""
    return max(11.0, min(16.0, min(target.width(), target.height()) * NAME_SCALE))


def paint_annotations(painter, data, layers, image_width, source, target, rotation=0.0, skip_names=(),
                      name_scale=NAME_SCALE, font_px=None, names=True):
    """Annotations on a plate-solved image: white outlined names on
    translucent boxes and subtle white markers.

    image_width: width of the image being drawn (any size of the plate-solved file)
    source: the part of that image drawn into target
    rotation: degrees the painter turns the image - names are kept upright
    skip_names: objects to mark without naming (e.g. already in a caption)
    font_px: DSO name height in target units; None: name_scale x target's short side
        (marker line widths and smallest sizes follow it too)
    names: False to draw markers only (e.g. on small thumbnails)
    """
    if source.width() <= 0 or source.height() <= 0:
        return
    k = image_width / data.width                         # drawn image px per solved px
    sx, sy = target.width() / source.width(), target.height() / source.height()

    def place(px, py):
        return QPointF(target.left() + (px * k - source.left()) * sx, target.top() + (py * k - source.top()) * sy)

    per_px = k * sx                                       # target units per solved px
    short = min(target.width(), target.height())
    if font_px is None:
        font_px = max(5.0, short * name_scale)
    line = max(0.6, font_px * 0.08)
    painter.save()
    painter.setClipRect(target, Qt.IntersectClip)
    painter.setRenderHint(QPainter.Antialiasing)

    def stroke(path_or_draw, width=line, alpha=220):
        """White line with a dark edge, readable on any image."""
        for color, w in ((QColor(0, 0, 0, 150), width * 2.6), (QColor(255, 255, 255, alpha), width)):
            pen = QPen(color, w)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path_or_draw()

    dso_names, star_names = [], []  # Placed after all the markers, DSOs first

    def around(anchor, half_w, half_h, gap):
        """Where a name may go beside a marker of half size half_w x half_h
        centered on anchor, best first: top-left corners as functions of the
        name's (width, height). The first is the upper right, by the marker."""
        x, y = anchor.x(), anchor.y()
        ux, uy = half_w * 0.75 + gap, half_h * 0.75
        return [lambda w, h: QPointF(x + ux, y - uy - h / 2),          # Upper right
                lambda w, h: QPointF(x + half_w + gap, y - h / 2),      # Right
                lambda w, h: QPointF(x + ux, y + uy - h / 2),           # Lower right
                lambda w, h: QPointF(x - ux - w, y - uy - h / 2),       # Upper left
                lambda w, h: QPointF(x - half_w - gap - w, y - h / 2),  # Left
                lambda w, h: QPointF(x - ux - w, y + uy - h / 2),       # Lower left
                lambda w, h: QPointF(x - w / 2, y - half_h - gap - h),  # Above
                lambda w, h: QPointF(x - w / 2, y + half_h + gap)]      # Below

    def place_names():
        """Each name in the first of its spots that's inside the image and clear
        of the names already placed - left out if none is (its marker stays).
        Names stay upright when the image is rotated, turning about their marker."""
        taken = []
        for text, size, anchor, spots in dso_names + star_names:
            font = QFont()
            font.setPixelSize(max(4, round(size)))
            metrics = QFontMetricsF(font)
            pad = size * 0.22
            width_, height_ = metrics.horizontalAdvance(text), metrics.height()
            turn = QTransform()
            if rotation % 360:
                turn.translate(anchor.x(), anchor.y())
                turn.rotate(-rotation)
                turn.translate(-anchor.x(), -anchor.y())
            for spot in spots:
                at = spot(width_, height_)
                box = QRectF(at.x() - pad, at.y() - pad, width_ + 2 * pad, height_ + 2 * pad)
                bounds = turn.mapRect(box)  # Where it lands in the image
                if target.contains(bounds) and not any(bounds.intersects(other) for other in taken):
                    break
            else:
                continue
            taken.append(bounds)
            painter.save()
            painter.setTransform(turn, True)
            painter.fillRect(box, QColor(0, 0, 0, 110))
            path = QPainterPath()
            path.addText(at.x(), at.y() + metrics.ascent(), font, text)
            outline = QPen(QColor(0, 0, 0), max(0.8, size * 0.1))
            outline.setJoinStyle(Qt.RoundJoin)
            painter.strokePath(path, outline)
            painter.fillPath(path, QColor(255, 255, 255))
            painter.restore()

    wcs = data.wcs
    if layers.get("grid"):
        min_ra, max_ra, min_dec, max_dec = wcs.get_field_of_view()
        span = max(abs(max_ra - min_ra), abs(max_dec - min_dec))
        step = next((s for s in (5.0, 1.0, 0.5, 0.25, 0.1, 0.05) if span / s >= 3), 0.05)

        def curve(points):
            mapped = [place(*p) for p in points if p[0] is not None]
            if len(mapped) > 1:
                stroke(lambda: painter.drawPolyline(mapped), line * 0.6, 90)

        value = math.floor(min_ra / step) * step
        while value <= max_ra + step:
            curve([wcs.sky_to_pixel(value, min_dec + (max_dec - min_dec) * i / 40) for i in range(41)])
            value += step
        value = math.floor(min_dec / step) * step
        while value <= max_dec + step:
            curve([wcs.sky_to_pixel(min_ra + (max_ra - min_ra) * i / 40, value) for i in range(41)])
            value += step
    if layers.get("lines"):
        for _name, ra1, dec1, ra2, dec2 in CONSTELLATION_LINES:
            a, b = wcs.sky_to_pixel(ra1, dec1), wcs.sky_to_pixel(ra2, dec2)
            if a[0] is not None and b[0] is not None:
                start, end = place(*a), place(*b)
                stroke(lambda: painter.drawLine(start, end), line, 120)
    if layers.get("stars") and data.stars:
        for star in data.stars:
            x, y = wcs.sky_to_pixel(star.ra, star.dec)
            if x is None:
                continue
            at = place(x, y)
            radius = font_px * 0.35
            stroke(lambda: painter.drawEllipse(at, radius, radius), line * 0.8)
            if names:
                star_names.append((star.name, font_px * 0.75, at, around(at, radius, radius, radius * 0.6)))
    if layers.get("dsos", True):
        skip = set(skip_names)
        dark = layers.get("dark", True)
        for dso in data.dsos:
            if not dark and (dso.obj_type or "").upper() == "DRKNB":
                continue
            x, y = wcs.sky_to_pixel(dso.ra, dso.dec)
            if x is None:
                continue
            at = place(x, y)
            # The object's own size where the catalog knows it
            radius = max((dso.size or 0) / 60 / 2 / data.degrees_per_pixel * per_px, font_px * 0.6)
            kind = (dso.obj_type or "").upper()
            if radius > short * 0.45:
                # Bigger than the image - just its name, by its center: just
                # below or above it first, off the bright core
                if names and dso.name not in skip:
                    gap = font_px * 0.8
                    dso_names.append((dso.name, font_px, at,
                                      [lambda w, h, at=at: QPointF(at.x() - w / 2, at.y() + gap),
                                       lambda w, h, at=at: QPointF(at.x() - w / 2, at.y() - gap - h),
                                       lambda w, h, at=at: QPointF(at.x() - w / 2, at.y() - h / 2)]
                                      + around(at, font_px, font_px, font_px * 0.3)))
                continue
            if "GALXY" in kind or "GX" in kind:
                # Its shape and tilt on the sky: major axis at the position angle
                # (from north through east), as north and east lie in this image
                shape = (getattr(dso, "minor", None) or 0) / dso.size if dso.size else 0.6
                north, east = wcs.sky_to_pixel(dso.ra, dso.dec + 0.01), wcs.sky_to_pixel(
                    dso.ra + 0.01 / max(0.01, math.cos(math.radians(dso.dec))), dso.dec)
                turn = 0.0
                if north[0] is not None and east[0] is not None:
                    n_vec, e_vec = place(*north) - at, place(*east) - at
                    n_len, e_len = math.hypot(n_vec.x(), n_vec.y()) or 1, math.hypot(e_vec.x(), e_vec.y()) or 1
                    pa = math.radians(getattr(dso, "angle", None) or 0)
                    ax = n_vec.x() / n_len * math.cos(pa) + e_vec.x() / e_len * math.sin(pa)
                    ay = n_vec.y() / n_len * math.cos(pa) + e_vec.y() / e_len * math.sin(pa)
                    turn = math.degrees(math.atan2(ay, ax))
                minor_radius = max(radius * min(max(shape, 0.15), 1.0), font_px * 0.35)

                def galaxy():
                    painter.save()
                    painter.translate(at)
                    painter.rotate(turn)
                    painter.drawEllipse(QPointF(0, 0), radius, minor_radius)
                    painter.restore()
                stroke(galaxy)
            elif "CL" in kind and "NB" not in kind:
                def dashed():
                    pen = painter.pen()
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawEllipse(at, radius, radius)
                stroke(dashed)
            elif "NB" in kind or "SNREM" in kind:
                box = QRectF(at.x() - radius, at.y() - radius, radius * 2, radius * 2)
                stroke(lambda: painter.drawRoundedRect(box, radius * 0.25, radius * 0.25))
            else:
                stroke(lambda: painter.drawEllipse(at, radius, radius))
            if names and dso.name not in skip:
                dso_names.append((dso.name, font_px, at, around(at, radius, radius, font_px * 0.3)))
    place_names()
    painter.restore()


# --------------------------------------------------------------------
# Plate solving in the background
# --------------------------------------------------------------------

class PlateSolveQueue(QObject):
    """Plate solves images in the background, one at a time, for every window
    that asks. Solutions are saved next to their images (.wcs) by the solver,
    and the annotation store reloads them.

    An image's state() is "solved", "unsolved", "solving", "queued", "failed"
    or "checking" (not known yet - being looked up)."""

    changed = Signal(str)                   # path - its state changed
    progress = Signal(str, str, str)        # path, name, message
    finished = Signal(str, bool, str, str)  # path, solved, why it failed ("" if solved), name

    def __init__(self):
        super().__init__()
        self._queue = []       # [path, hints, name] waiting - the first is being solved while _worker runs
        self._worker = None    # The running PlateSolverWorker
        self._running = set()  # Workers until their threads end - dropping a running QThread aborts the app
        self._failures = {}    # path -> why its plate solve failed
        annotation_store().ready.connect(self.changed)

    def state(self, path):
        if self._worker is not None and self._queue and self._queue[0][0] == path:
            return "solving"
        if any(entry[0] == path for entry in self._queue):
            return "queued"
        solved = annotation_store().is_solved(path)
        if solved is None:
            annotation_store().get(path)  # Find out (in the background)
            return "checking"
        if solved:
            return "solved"
        return "failed" if path in self._failures else "unsolved"

    def failure(self, path):
        """Why the image's last plate solve failed, or None."""
        return self._failures.get(path)

    def waiting(self, paths=None):
        """How many of paths (None: any) are being or waiting to be solved."""
        wanted = None if paths is None else set(paths)
        return sum(1 for entry in self._queue if wanted is None or entry[0] in wanted)

    def add(self, path, hints=None, name=""):
        """Queue an unsolved (or failed) image; hints: {"ra", "dec", "radius"} in
        degrees. Returns whether it was queued."""
        if not path or self.state(path) not in ("unsolved", "failed"):
            return False
        self._failures.pop(path, None)
        self._queue.append([path, hints, name])
        self.changed.emit(path)
        self._start_next()
        return True

    def stop(self, paths=None):
        """Drop the waiting images among paths (None: all); the one being
        solved finishes its current solver, then stops."""
        wanted = None if paths is None else set(paths)
        running = self._queue[0] if self._worker is not None and self._queue else None
        dropped = [entry[0] for entry in self._queue
                   if entry is not running and (wanted is None or entry[0] in wanted)]
        self._queue = [entry for entry in self._queue if entry is running or entry[0] not in dropped]
        if running is not None and (wanted is None or running[0] in wanted):
            self._worker.cancel()
        for path in dropped:
            self.changed.emit(path)

    def _start_next(self):
        if self._worker is not None or not self._queue:
            return
        from PlateSolver import PlateSolverWorker
        path, hints, name = self._queue[0]
        worker = PlateSolverWorker(path, hints)
        worker.solve_finished.connect(self._on_solved)
        worker.progress.connect(lambda message: self.progress.emit(path, name, message))
        self._running.add(worker)
        worker.finished.connect(lambda: self._running.discard(worker))
        self._worker = worker
        worker.start()
        self.changed.emit(path)

    def _on_solved(self, result):
        path, _hints, name = self._queue.pop(0) if self._queue else (None, None, "")
        self._worker = None
        if path is not None:
            failure = ""
            if result.success and result.wcs_header:
                annotation_store().forget(path)
                annotation_store().get(path)
            else:
                failure = (result.error_message or "No solution").strip().splitlines()[0]
                self._failures[path] = failure
            self.changed.emit(path)
            self.finished.emit(path, not failure, failure, name)
        self._start_next()


_plate_solve_queue = None


def plate_solve_queue():
    global _plate_solve_queue
    if _plate_solve_queue is None:
        _plate_solve_queue = PlateSolveQueue()
    return _plate_solve_queue
