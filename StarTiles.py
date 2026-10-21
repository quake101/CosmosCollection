#!/usr/bin/env python3
"""
Star catalog tiles for Cosmos Collection - the native plate solver's reference stars,
kept on disk so a part of the sky fetched once solves offline afterwards.

The sky is cut into tiles, at several sizes (levels) so that a cone of any
size touches a manageable number of them. A tile holds every catalog star in
it brighter than its magnitude limit, so the brightest stars of any cone can
be put together from the tiles it touches - the answer VizieR would give -
whichever earlier solves fetched them. Tiles that are missing, or don't reach
faint enough, are fetched from VizieR in one request per cone.

    stars = cone_stars("gaia", ra, dec, radius, count)   # (n, 3) ra, dec, mag

Tiles live in one SQLite file, which every thread shares through one
connection. It keeps SQLite's default rollback journal: WAL mode with a
connection per thread could meet WAL's "reset" corruption bug, which SQLite
before 3.51.3 has.
"""

import math
import time
import logging
import sqlite3
import threading

import numpy as np

logger = logging.getLogger(__name__)

VIZIER_URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
CATALOGS = {  # VizieR table, then its ra, dec and magnitude columns
    "tycho2": ("I/259/tyc2", ("RAmdeg", "DEmdeg", "VTmag")),
    "gaia": ("I/355/gaiadr3", ("RA_ICRS", "DE_ICRS", "Gmag")),
}
BANDS = 2880            # Declination bands at level 0 (0.0625 degrees tall), half as many each level up
LEVELS = 7              # Tile heights 0.0625 to 4 degrees
ALL_STARS = 99.0        # Magnitude limit of a tile holding every star the catalog has there
FETCH_MARGIN = 1.3      # Stars asked for beyond what the area fetched should need (stars bunch up)
FETCH_PAD = 1.0         # Tiles fetched reach this many tile heights beyond the cone
MAX_FETCH = 50000       # Most stars asked for in one request
SCHEMA = 1              # Tile table layout (PRAGMA user_version) - another starts the cache afresh
FILE_NAME = "star_tiles.sqlite"


class CatalogUnavailable(Exception):
    """VizieR couldn't be reached for stars the cache doesn't have."""


# ---- Tiles -------------------------------------------------------------------

def _height(level):
    """Tile height (degrees of declination) at level."""
    return 180.0 / (BANDS >> level)


def _slices(level, band):
    """RA slices in a declination band (int or array): each about as wide as
    the band is tall, along the band's widest (equator side) edge."""
    h = _height(level)
    low = -90.0 + np.asarray(band) * h
    nearest = np.where((low < 0) & (low + h > 0), 0.0, np.minimum(np.abs(low), np.abs(low + h)))
    return np.maximum(1, np.ceil(360.0 * np.cos(np.radians(nearest)) / h - 1e-9)).astype(int)


def level_for(radius):
    """The tile level for a cone: the biggest tiles no taller than half its
    radius - small enough that the area fetched isn't much more than the
    cone, big enough that it touches tens of tiles, not thousands."""
    level = math.floor(math.log2(max(radius, 1e-6) / 2 / _height(0)))
    return min(max(level, 0), LEVELS - 1)


def _tiles_for(level, ra, dec, radius):
    """(band, slice) of every tile at level that the cone may touch."""
    h = _height(level)
    first = max(0, math.floor((dec - radius + 90) / h))
    last = min((BANDS >> level) - 1, math.floor((dec + radius + 90) / h))
    if abs(dec) + radius >= 90:
        half = 180.0  # The cone takes in a pole: every RA
    else:
        half = math.degrees(math.asin(min(1.0, math.sin(math.radians(radius)) / math.cos(math.radians(dec)))))
    tiles = []
    for band in range(first, last + 1):
        n = int(_slices(level, band))
        width = 360.0 / n
        lo, hi = math.floor((ra - half) / width), math.floor((ra + half) / width)
        slices = range(n) if hi - lo + 1 >= n else (s % n for s in range(lo, hi + 1))
        tiles += [(band, s) for s in slices]
    return tiles


def _tile_of(level, ras, decs):
    """(bands, slices) arrays: the tile at level each star is in."""
    bands = np.clip(np.floor((decs + 90.0) / _height(level)).astype(int), 0, (BANDS >> level) - 1)
    n = _slices(level, bands)
    slices = np.minimum(np.floor((ras % 360.0) / (360.0 / n)).astype(int), n - 1)
    return bands, slices


def _separation(ra, dec, ras, decs):
    """Angular distances (degrees) from ra, dec to each of ras, decs."""
    ra, dec = math.radians(ra), math.radians(dec)
    ras, decs = np.radians(ras), np.radians(decs)
    a = np.sin((decs - dec) / 2) ** 2 + math.cos(dec) * np.cos(decs) * np.sin((ras - ra) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _reach(level, tiles, ra, dec):
    """How far the farthest point of the tiles is from ra, dec (degrees). It's
    a corner, or where a tile's top or bottom edge crosses the RA opposite
    ra (beyond a pole)."""
    bands = np.array([band for band, _s in tiles])
    slices = np.array([s for _band, s in tiles])
    width = 360.0 / _slices(level, bands)
    ra1, ra2 = slices * width, (slices + 1) * width
    dec1 = -90.0 + bands * _height(level)
    dec2 = dec1 + _height(level)
    opposite = (ra + 180.0) % 360.0
    across = np.where((ra1 <= opposite) & (opposite <= ra2), opposite, ra1)
    ras = np.concatenate([ra1, ra2, ra1, ra2, across, across])
    decs = np.concatenate([dec1, dec1, dec2, dec2, dec1, dec2])
    return float(_separation(ra, dec, ras, decs).max())


# ---- The cache file ----------------------------------------------------------

_lock = threading.Lock()  # One thread at a time uses the connection
_conn = None


def _path():
    from ResourceManager import ResourceManager
    return ResourceManager.get_cache_dir("star_catalog") / FILE_NAME


def _open():
    """The shared connection, opened (and the table made) on first use."""
    global _conn
    if _conn is None:
        path = _path()
        conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
        try:
            if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA:
                conn.execute("DROP TABLE IF EXISTS tiles")
                conn.execute("CREATE TABLE tiles (catalog TEXT NOT NULL, level INTEGER NOT NULL, "
                             "band INTEGER NOT NULL, slice INTEGER NOT NULL, "
                             "mag_limit REAL NOT NULL, "     # Every star brighter than this is here
                             "stars BLOB NOT NULL, "         # float64 ra, dec, mag rows, brightest first
                             "fetched REAL NOT NULL, "       # Unix time
                             "PRIMARY KEY (catalog, level, band, slice)) WITHOUT ROWID")
                conn.execute(f"PRAGMA user_version = {SCHEMA}")
                conn.commit()
                # The cache's earlier form: a JSON file per cone fetched
                for old in path.parent.glob("*.json"):
                    try:
                        old.unlink()
                    except OSError:
                        pass
        except sqlite3.Error:
            conn.close()
            raise
        _conn = conn
    return _conn


def _use(work, default=None):
    """work(connection), one thread at a time - default if the cache can't be
    used just now. A damaged cache file is deleted, to start afresh."""
    global _conn
    with _lock:
        try:
            return work(_open())
        except sqlite3.OperationalError as e:  # Locked by another process, disk full...
            logger.info(f"Star tile cache not available: {e}")
        except sqlite3.DatabaseError as e:
            logger.warning(f"Star tile cache damaged, starting a new one: {e}")
            if _conn is not None:
                _conn.close()
                _conn = None
            try:
                _path().unlink(missing_ok=True)
            except OSError:
                pass
        except OSError as e:
            logger.info(f"Star tile cache not available: {e}")
    return default


def _load(catalog, level, tiles):
    """{(band, slice): (mag_limit, stars)} of the tiles that are on disk."""
    def work(conn):
        found = {}
        for start in range(0, len(tiles), 400):
            chunk = tiles[start:start + 400]
            rows = conn.execute(
                f"SELECT t.band, t.slice, t.mag_limit, t.stars FROM (VALUES {','.join(['(?, ?)'] * len(chunk))}) "
                "AS w JOIN tiles t ON t.catalog = ? AND t.level = ? AND t.band = w.column1 AND t.slice = w.column2",
                [v for tile in chunk for v in tile] + [catalog, level])
            for band, s, limit, blob in rows:
                found[(band, s)] = (limit, np.frombuffer(blob, np.float64).reshape(-1, 3))
        return found
    return _use(work, {})


def _store(catalog, level, fetched):
    """Save fetched tiles - {(band, slice): (mag_limit, stars)} - over any on
    disk that don't reach as faint."""
    now = time.time()
    rows = [(catalog, level, band, s, limit, np.ascontiguousarray(stars, np.float64).tobytes(), now)
            for (band, s), (limit, stars) in fetched.items()]

    def work(conn):
        with conn:
            conn.executemany(
                "INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (catalog, level, band, slice) "
                "DO UPDATE SET mag_limit = excluded.mag_limit, stars = excluded.stars, "
                "fetched = excluded.fetched WHERE excluded.mag_limit > tiles.mag_limit", rows)
    _use(work)


# ---- VizieR ------------------------------------------------------------------

def _vizier(catalog, ra, dec, radius, count, brighter_than=None):
    """The `count` brightest stars of catalog within radius of ra, dec (only
    those brighter than brighter_than, if given), from VizieR: ((n, 3) array
    of ra, dec, magnitude, brightest first; whether VizieR stopped at count).

    When VizieR stops at count it can leave out a few stars brighter than the
    last it sends, even bright ones - stars brighter than a magnitude are all
    there only when it doesn't stop."""
    import requests
    from OnlineLimits import OFFLINE_NOTE, is_offline, online_request
    if is_offline():
        raise CatalogUnavailable(OFFLINE_NOTE)
    source, columns = CATALOGS[catalog]
    params = {"-source": source, "-c": f"{ra:.5f} {dec:+.5f}", "-c.rd": f"{radius:.4f}",
              "-out": ",".join(columns), "-sort": columns[2], "-out.max": str(count)}
    if brighter_than is not None:
        params[columns[2]] = f"<{brighter_than:.3f}"
    for attempt in range(2):
        try:
            with online_request("vizier"):
                response = requests.get(VIZIER_URL, params=params, timeout=(15, 120))
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt:
                raise CatalogUnavailable(str(e))
            logger.info(f"Star catalog query failed, trying again: {e}")
    rows, sent, table = [], 0, False
    for line in response.text.splitlines():
        if not table:
            table = line.startswith("--")  # The dashes under the column names
            continue
        if not line.strip() or line.startswith("#"):
            continue
        sent += 1  # Every row counts toward count, even one without a position or magnitude
        parts = line.split("\t")
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except (ValueError, IndexError):
            continue
    return np.array(rows, dtype=np.float64).reshape(-1, 3), sent >= count


def _fetch(catalog, level, tiles, ra, dec, radius, count):
    """Fetch the tiles from VizieR around ra, dec, with enough stars that the
    brightest `count` within radius should all be brighter than the tiles'
    new limit, and save them. Returns {(band, slice): (mag_limit, stars)} -
    empty if VizieR sent no stars, as an error page would read.

    The brightest stars VizieR sends, up to how many are wanted, show how
    faint the tiles need to reach. When there are more than that, a second
    request fetches every star to that magnitude (see _vizier)."""
    reach = _reach(level, tiles, ra, dec) + 1e-4
    area = (1 - math.cos(math.radians(reach))) / (1 - math.cos(math.radians(max(radius, 1e-3))))
    want = min(MAX_FETCH, math.ceil(count * area * FETCH_MARGIN))
    stars, cut = _vizier(catalog, ra, dec, reach, want)
    if not len(stars):
        return {}
    limit = ALL_STARS
    if cut:
        limit = round(float(stars[:, 2].max()), 3) + 0.001
        stars, cut = _vizier(catalog, ra, dec, reach, 3 * want + 100, brighter_than=limit)
        if not len(stars):
            return {}
        if cut:  # Far more than the first request found: can't be sure of these
            logger.info(f"Too many {catalog} stars to magnitude {limit:.2f} near {ra:.3f} {dec:+.3f}")
            limit = -math.inf
    bands, slices = _tile_of(level, stars[:, 0], stars[:, 1])
    fetched = {(band, s): (limit, stars[(bands == band) & (slices == s)]) for band, s in tiles}
    if limit > -math.inf:
        _store(catalog, level, fetched)
    logger.info(f"Fetched {len(stars)} {catalog} stars within {reach:.2f} deg of {ra:.3f} {dec:+.3f} "
                f"({len(tiles)} level {level} tiles, to magnitude {limit:.2f})")
    return fetched


# ---- Cones -------------------------------------------------------------------

_claims = threading.Condition()
_claimed = set()  # (catalog, level, band, slice) of the tiles being fetched


def _pick(found, tiles, ra, dec, radius, count):
    """(the brightest `count` stars within radius in the tiles found, and
    whether they surely are: every tile is there, and none could have left
    out a brighter star)."""
    have = [found[tile] for tile in tiles if tile in found]
    stars = np.concatenate([stars for _limit, stars in have] + [np.empty((0, 3))])
    stars = stars[_separation(ra, dec, stars[:, 0], stars[:, 1]) <= radius]
    stars = stars[np.argsort(stars[:, 2], kind="stable")]
    if len(have) < len(tiles):
        return stars[:count], False
    limit = min(limit for limit, _stars in have)
    return stars[:count], limit >= ALL_STARS or int((stars[:, 2] < limit).sum()) >= count


def cone_stars(catalog, ra, dec, radius, count):
    """The `count` brightest stars of catalog ("tycho2" or "gaia") within
    radius (degrees) of ra, dec, brightest first: an (n, 3) array of ra, dec,
    magnitude. From the tiles on disk, fetching any missing or not deep
    enough. Offline, the best the tiles on disk have; CatalogUnavailable if
    they have none."""
    ra = ra % 360.0
    level = level_for(radius)
    tiles = _tiles_for(level, ra, dec, radius)
    stars, sure = _pick(_load(catalog, level, tiles), tiles, ra, dec, radius, count)
    if sure:
        return stars

    # Fetch a little beyond the cone, so one a little off (another image of
    # the object) finds its tiles there. One thread fetches a tile at a time -
    # others wait, then use it.
    wider = _tiles_for(level, ra, dec, radius + FETCH_PAD * _height(level))
    keys = {(catalog, level) + tile for tile in wider}
    with _claims:
        while keys & _claimed:
            _claims.wait()
        _claimed.update(keys)
    try:
        found = _load(catalog, level, tiles)
        stars, sure = _pick(found, tiles, ra, dec, radius, count)
        want = count
        for _attempt in range(2):
            if sure:
                break
            try:
                fetched = _fetch(catalog, level, wider, ra, dec, radius, want)
            except CatalogUnavailable:
                if len(stars):
                    logger.info(f"Star catalog offline - using the {len(stars)} {catalog} stars on disk")
                    break
                raise
            if not fetched:
                break
            for tile, (limit, tile_stars) in fetched.items():
                if tile not in found or limit > found[tile][0]:
                    found[tile] = (limit, tile_stars)
            stars, sure = _pick(found, tiles, ra, dec, radius, count)
            want *= 4  # Too few: the stars bunch up away from the cone (a cluster nearby)
        return stars
    finally:
        with _claims:
            _claimed.difference_update(keys)
            _claims.notify_all()
