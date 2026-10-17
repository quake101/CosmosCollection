#!/usr/bin/env python3
"""
Built-in plate solver for Cosmos Collection - numpy only.

Finds where an image is in the sky when roughly where it points is known (the
object it shows). Like twirl (Lionel Garcia, MIT license -
https://github.com/lgrcia/twirl) it matches the shapes of groups of four bright
stars ("quads") between the image and the Gaia catalog. A quad's shape doesn't
change with the image's position, rotation or scale, so the image scale needn't
be known: a few likely fields of view are tried in turn, nearest the FITS
header's (or a hint's) first.

Reference stars come from Gaia DR3, or Tycho-2 for wide fields, through
VizieR (online), cached on disk.

Command line - solve images and compare with their saved (e.g. ASTAP) solutions:
    python NativePlateSolver.py image [image ...] [--ra DEG --dec DEG]
"""

import os
import sys
import json
import math
import time
import hashlib
import logging
import itertools

import numpy as np

logger = logging.getLogger(__name__)

VIZIER_URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
TYCHO_RADIUS = 0.45       # Catalog cones this wide (degrees) try Tycho-2 first
TYCHO_ENOUGH = 100        # ...and use it if it has at least this many stars there
TYCHO = ("I/259/tyc2", ("RAmdeg", "DEmdeg", "VTmag"))
DEEP_DEPTH = 1500         # Catalog stars for the final fit (Tycho-2 fields)
ANALYSIS_SIDE = 2000      # Stars are found in a copy of the image at most this many pixels across
IMAGE_STARS = 15          # Brightest image stars quads are made from
CATALOG_STARS = 25        # Brightest catalog stars in the field quads are made from
VERIFY_STARS = 80         # Image stars checked against the catalog to confirm a match
CATALOG_DEPTH = 400       # Catalog stars fetched per field
CODE_TOLERANCE = 0.02     # How alike two quads' shapes must be
MAX_CANDIDATES = 400      # Closest quad pairs tried per field
MATCH_PX = 3.0            # A star matches the catalog within this many analysis pixels
MIN_MATCHED = 10          # Matched stars needed to accept a solution
FIELD_STEP = 1.45         # Ratio between the fields of view tried
FIELDS = [0.25 * FIELD_STEP ** i for i in range(11)]  # Image diagonals tried, degrees (0.25 to ~10)
FITS_EXTENSIONS = (".fits", ".fit", ".fts")
XISF_EXTENSIONS = (".xisf",)


class SolveFailed(Exception):
    """The image couldn't be solved - the message says why, for the user."""


# ---- Sky <-> tangent plane ---------------------------------------------------

def _project(ra, dec, ra0, dec0):
    """Gnomonic (TAN) projection about ra0, dec0 - degrees in, degrees out
    (xi toward increasing RA, eta toward north)."""
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    cos_c = np.sin(dec) * math.sin(dec0) + np.cos(dec) * math.cos(dec0) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
    eta = (np.sin(dec) * math.cos(dec0) - np.cos(dec) * math.sin(dec0) * np.cos(ra - ra0)) / cos_c
    return np.degrees(xi), np.degrees(eta)


def _deproject(xi, eta, ra0, dec0):
    """Inverse of _project - degrees in, (ra, dec) degrees out."""
    xi, eta = math.radians(xi), math.radians(eta)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    rho = math.hypot(xi, eta)
    if rho == 0:
        return math.degrees(ra0), math.degrees(dec0)
    c = math.atan(rho)
    dec = math.asin(math.cos(c) * math.sin(dec0) + eta * math.sin(c) * math.cos(dec0) / rho)
    ra = ra0 + math.atan2(xi * math.sin(c), rho * math.cos(dec0) * math.cos(c) - eta * math.sin(dec0) * math.sin(c))
    return math.degrees(ra) % 360, math.degrees(dec)


# ---- Reference stars ---------------------------------------------------------

def _cache_dir():
    from ResourceManager import ResourceManager
    return ResourceManager.get_cache_dir("star_catalog")


def _vizier_stars(source, columns, ra, dec, radius, count):
    """The `count` brightest stars of a VizieR catalog within radius (degrees)
    of ra, dec, brightest first: an (n, 3) array of ra, dec, magnitude. columns:
    its ra, dec and magnitude columns. Cached on disk. Raises SolveFailed if
    the catalog can't be reached."""
    key = f"{source}|{ra:.3f}|{dec:.3f}|{radius:.3f}|{count}"
    cache = _cache_dir() / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    if cache.exists():
        try:
            return np.array(json.loads(cache.read_text()), dtype=np.float64).reshape(-1, 3)
        except (ValueError, OSError):
            pass
    import requests
    from OnlineLimits import online_request
    params = {"-source": source, "-c": f"{ra:.5f} {dec:+.5f}", "-c.rd": f"{radius:.4f}",
              "-out": ",".join(columns), "-sort": columns[2], "-out.max": str(count)}
    for attempt in range(2):
        try:
            with online_request("vizier"):
                response = requests.get(VIZIER_URL, params=params, timeout=45)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt:
                raise SolveFailed(f"Couldn't reach the star catalog (online): {e}")
            logger.info(f"Star catalog query failed, trying again: {e}")
    rows = []
    for line in response.text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or line.startswith("#"):
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue  # Header and unit lines, and stars without a position
    stars = np.array(rows, dtype=np.float64).reshape(-1, 3)
    try:
        cache.write_text(json.dumps(stars.round(7).tolist()))
    except OSError:
        pass
    return stars


def gaia_stars(ra, dec, radius, count=CATALOG_DEPTH):
    """The brightest Gaia DR3 stars around ra, dec (see _vizier_stars)."""
    return _vizier_stars("I/355/gaiadr3", ("RA_ICRS", "DE_ICRS", "Gmag"), ra, dec, radius, count)


def reference_stars(ra, dec, radius, count=CATALOG_DEPTH):
    """The brightest stars around ra, dec (see _vizier_stars). A wide field's
    brightest stars are all in Tycho-2, which answers in about a second where
    Gaia can take a minute (it holds millions of stars in a wide cone along
    the Milky Way); narrower fields need Gaia's fainter stars."""
    if radius >= TYCHO_RADIUS:
        stars = _vizier_stars(*TYCHO, ra, dec, radius, count)
        if len(stars) >= TYCHO_ENOUGH:
            return stars
    return gaia_stars(ra, dec, radius, count)


# ---- Image stars -------------------------------------------------------------

def _load_luminance(path):
    """(mono float32 image at most ANALYSIS_SIDE across, original pixels per
    analysis pixel, (original width, height), FITS-style header dict)."""
    ext = os.path.splitext(path)[1].lower()
    header = {}
    if ext in FITS_EXTENSIONS + XISF_EXTENSIONS:
        from FrameQuality import _read_frame, _luminance
        data, header = _read_frame(path)
        height, width = data.shape[:2]
        img, factor, _summed = _luminance(data, header)
        del data
    else:
        from PySide6.QtGui import QImage
        from ImageLoader import read_image_file
        image, _error = read_image_file(path, auto_transform=False)
        if image is None:
            raise SolveFailed("The image couldn't be read")
        image = image.convertToFormat(QImage.Format_Grayscale16)
        width, height = image.width(), image.height()
        row_pixels = image.bytesPerLine() // 2
        img = np.frombuffer(image.constBits(), dtype=np.uint16, count=row_pixels * height)
        img = img.reshape(height, row_pixels)[:, :width].astype(np.float32)
        factor = 1
    k = math.ceil(max(img.shape) / ANALYSIS_SIDE)
    if k > 1:
        h, w = img.shape[0] // k * k, img.shape[1] // k * k
        img = img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3), dtype=np.float32)
        factor *= k
    return img, factor, (width, height), header


def find_stars(img, count=300):
    """Stars in a mono image, brightest first, ranked two ways - [by total
    light, by light above their own surroundings]: (n, 2) arrays of x, y
    centroids in its pixels.

    Neither ranking suits every image. Total light favors big bright stars,
    but bright nebula knots outrank the stars in a nebula-filled field (with
    reduced stars, as in many processed images). Light above a ring around
    each star ignores the nebula, but undercounts stars big enough that the
    ring sits in their own halo."""
    from FrameQuality import TILE, _tile_medians, _upsample, _smooth, _clipped_stats, _detect
    mesh, tile_h, tile_w = _tile_medians(img, TILE)
    sub = img - _upsample(mesh, img.shape, tile_h, tile_w)
    smoothed = _smooth(sub)
    _, noise = _clipped_stats(smoothed[::3, ::3].ravel())
    if noise <= 0:
        # Processed images can have a clipped (perfectly flat) background
        noise = max(float(np.percentile(smoothed, 99.9)) * 0.01, 1e-6)
    ys, xs, _peaks = _detect(sub, smoothed, noise)
    if ys.size == 0:
        return [np.empty((0, 2)), np.empty((0, 2))]

    # The 13x13 around each peak (edges repeated)
    pad = 6
    padded = np.pad(sub, pad, mode="edge")
    offsets = np.arange(-pad, pad + 1)
    boxes = padded[(ys + pad)[:, None, None] + offsets[None, :, None],
                   (xs + pad)[:, None, None] + offsets[None, None, :]].astype(np.float64)
    inner = boxes[:, 3:10, 3:10]                                   # The 7x7 around the peak
    total = np.clip(inner, 0, None).reshape(ys.size, -1).sum(axis=1)
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    ring = np.median(boxes[:, (np.hypot(yy, xx) >= 5) & (np.hypot(yy, xx) <= 6.5)], axis=1)
    above = (inner - ring[:, None, None]).reshape(ys.size, -1).sum(axis=1)

    # Centroids from the 5x5 around each peak, above its surroundings (nebula
    # glow under a star would pull the centroid toward the brighter side)
    core = np.clip(boxes[:, 4:9, 4:9] - ring[:, None, None], 0, None)
    weight = core.sum(axis=(1, 2))
    steps = np.arange(-2, 3)
    cy = ys + np.where(weight > 0, (core.sum(axis=2) * steps).sum(axis=1) / np.maximum(weight, 1e-12), 0)
    cx = xs + np.where(weight > 0, (core.sum(axis=1) * steps).sum(axis=1) / np.maximum(weight, 1e-12), 0)

    rankings = []
    for brightness in (total, above):
        kept = []
        for i in np.argsort(-brightness):
            if brightness[i] <= 0:
                break
            # One detection per star: knots in a big star's halo sit within a few pixels of it
            if all((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2 > 36 for j in kept):
                kept.append(i)
                if len(kept) >= count:
                    break
        rankings.append(np.column_stack([cx[kept], cy[kept]]))
    return rankings


# ---- Quads -------------------------------------------------------------------

_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_OTHERS = np.array([(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)])


def _quads(z):
    """Shape codes of every group of four of the points z (complex): with the
    two farthest apart (A, B) moved to 0 and 1, the other two's positions -
    (n, 4) codes, and the (n, 4) point indices in A, B, C, D order."""
    combos = np.array(list(itertools.combinations(range(z.size), 4)), dtype=np.intp)
    q = z[combos]
    far = np.argmax(np.abs(q[:, _PAIRS[:, 0]] - q[:, _PAIRS[:, 1]]), axis=1)
    rows = np.arange(combos.shape[0])
    order = np.column_stack([_PAIRS[far], _OTHERS[far]])     # A, B, C, D (positions within each combo)
    a, b = q[rows, order[:, 0]], q[rows, order[:, 1]]
    wc = (q[rows, order[:, 2]] - a) / (b - a)
    wd = (q[rows, order[:, 3]] - a) / (b - a)
    # Which end is A: the one the others sit nearer (code unchanged by relabeling)
    flip = (wc.real + wd.real) > 1
    wc, wd = np.where(flip, 1 - wc, wc), np.where(flip, 1 - wd, wd)
    order[flip, 0], order[flip, 1] = order[flip, 1], order[flip, 0].copy()
    # C before D, left to right
    swap = wc.real > wd.real
    wc, wd = np.where(swap, wd, wc), np.where(swap, wc, wd)
    order[swap, 2], order[swap, 3] = order[swap, 3], order[swap, 2].copy()
    codes = np.column_stack([wc.real, wc.imag, wd.real, wd.imag])
    return codes, combos[rows[:, None], order]


def _similarity(src, dst):
    """Least-squares dst = a * src + b for each row of the (k, n) complex arrays."""
    ms, md = src.mean(axis=1, keepdims=True), dst.mean(axis=1, keepdims=True)
    s, d = src - ms, dst - md
    a = (np.conj(s) * d).sum(axis=1) / np.maximum((np.abs(s) ** 2).sum(axis=1), 1e-30)
    return a, md[:, 0] - a * ms[:, 0]


def _matches(points, catalog, tolerance):
    """For each point, the index of the nearest catalog point within tolerance, or -1."""
    best = np.full(points.size, -1)
    for start in range(0, points.size, 64):
        chunk = points[start:start + 64]
        distance = np.abs(chunk[:, None] - catalog[None, :])
        nearest = np.argmin(distance, axis=1)
        close = distance[np.arange(chunk.size), nearest] < tolerance
        best[start:start + chunk.size] = np.where(close, nearest, -1)
    return best


def _affine(xy, sky):
    """Least-squares sky = M @ xy + t - (2x2 M, t) for (n, 2) arrays."""
    design = np.column_stack([xy, np.ones(len(xy))])
    solution, *_ = np.linalg.lstsq(design, sky, rcond=None)
    return solution[:2].T, solution[2]


def _try_field(stars_xy, diag_px, field, ra0, dec0, progress=None):
    """Match the image's stars with the catalog's for an image diag_px pixels
    across covering `field` degrees (diagonal) centered near ra0, dec0.
    Returns (matched image star indices, catalog sky positions as complex
    tangent-plane degrees, the catalog) or None."""
    catalog = reference_stars(ra0, dec0, field * 0.6)
    if len(catalog) < 8:
        return None
    xi, eta = _project(catalog[:, 0], catalog[:, 1], ra0, dec0)
    sky = xi + 1j * eta

    image = stars_xy[:, 0] + 1j * stars_xy[:, 1]
    top_image = image[:IMAGE_STARS]
    inside = np.abs(sky) < field / 2
    top_sky_index = np.flatnonzero(inside)[:CATALOG_STARS]
    if top_image.size < 6 or top_sky_index.size < 6:
        return None
    image_codes, image_quads = _quads(top_image)
    sky_codes, sky_quads = _quads(sky[top_sky_index])
    sky_quads = top_sky_index[sky_quads]

    verify = image[:VERIFY_STARS]
    best = None
    for mirrored in (False, True):
        codes = sky_codes * np.array([1, -1, 1, -1]) if mirrored else sky_codes
        # The closest pairs of shapes
        pairs = []
        for start in range(0, len(image_codes), 256):
            distance = np.sqrt(((image_codes[start:start + 256, None, :] - codes[None, :, :]) ** 2).sum(axis=2))
            i, j = np.nonzero(distance < CODE_TOLERANCE)
            pairs.append(np.column_stack([i + start, j, distance[i, j]]))
        pairs = np.concatenate(pairs) if pairs else np.empty((0, 3))
        if not len(pairs):
            continue
        pairs = pairs[np.argsort(pairs[:, 2])[:MAX_CANDIDATES]]
        src = top_image[image_quads[pairs[:, 0].astype(int)]]
        if mirrored:
            src = np.conj(src)
        dst = sky[sky_quads[pairs[:, 1].astype(int)]]
        a, b = _similarity(src, dst)
        # The field it implies must be near the one being tried
        implied = np.abs(a) * diag_px
        plausible = (implied > field / FIELD_STEP ** 1.5) & (implied < field * FIELD_STEP ** 1.5)
        for k in np.flatnonzero(plausible):
            points = a[k] * (np.conj(verify) if mirrored else verify) + b[k]
            found = _matches(points, sky, MATCH_PX * abs(a[k]))
            count = int((found >= 0).sum())
            if best is None or count > best[0]:
                best = (count, mirrored, a[k], b[k], found)
            if count >= 3 * MIN_MATCHED:
                break
    if best is None or best[0] < MIN_MATCHED:
        return None
    return best, sky, catalog


def _refine(stars_xy, sky, scale_guess, points):
    """An affine fit to every matched star, rematched twice with the better fit.
    points: where the image stars land on the sky (complex tangent plane) by
    the fit so far. Pairs well off the fit (a neighbor taken for the star) are
    left out. Returns the image positions and catalog indices of the pairs used."""
    sky_xy = np.column_stack([sky.real, sky.imag])
    xy, matched = stars_xy[:0], np.empty(0, int)
    for _ in range(3):
        found = _matches(points, sky, MATCH_PX * scale_guess)
        use = found >= 0
        if use.sum() < MIN_MATCHED:
            break
        xy, matched = stars_xy[use], found[use]
        target = sky_xy[matched]
        keep = np.ones(len(xy), bool)
        for _clip in range(4):
            matrix, offset = _affine(xy[keep], target[keep])
            residual = np.hypot(*(xy @ matrix.T + offset - target).T)
            # 3 sigma (from the median), but never tighter than half a pixel
            fit = residual <= max(3 * 1.4826 * np.median(residual[keep]), 0.5 * scale_guess)
            if fit.sum() < MIN_MATCHED or (fit == keep).all():
                break
            keep = fit
        xy, matched = xy[keep], matched[keep]
        matrix, offset = _affine(xy, sky_xy[matched])
        fitted = stars_xy @ matrix.T + offset
        points = fitted[:, 0] + 1j * fitted[:, 1]
    return xy, matched


def solve(path, ra, dec, scale=None, progress=None, cancelled=None):
    """Plate solve an image near ra, dec (degrees) - scale: its arcsec/pixel
    if known (else from the FITS header, or tried over a range).

    Returns (WCS keywords in the same convention as ASTAP's .wcs files, info
    dict) - raises SolveFailed with the reason otherwise."""
    started = time.perf_counter()
    say = progress or (lambda message: None)
    say("Finding stars...")
    img, factor, (width, height), header = _load_luminance(path)
    rankings = [stars for stars in find_stars(img) if len(stars) >= MIN_MATCHED]
    if not rankings:
        raise SolveFailed("Too few stars found in the image")

    # Fields to try, the likeliest first: the given scale, the FITS header's,
    # other images taken with the same equipment, then a sweep out from those
    diag_px = math.hypot(*img.shape)
    scales = [scale] if scale else []
    if header.get("FOCALLEN") and header.get("XPIXSZ"):
        binning = header.get("XBINNING") or 1
        scales.append(206.265 * float(header["XPIXSZ"]) * float(binning) / float(header["FOCALLEN"]))
    scales += equipment_scales(path)
    fields = []
    for guess in [s * math.hypot(width, height) / 3600 for s in scales] + \
            sorted(FIELDS, key=lambda f: abs(math.log(f / (scales[0] * math.hypot(width, height) / 3600
                                                            if scales else 1.5)))):
        if all(abs(math.log(guess / tried)) > math.log(1.2) for tried in fields):
            fields.append(guess)

    catalog_error = None
    for tries, field in enumerate(fields, 1):
        if cancelled and cancelled():
            raise SolveFailed("Cancelled")
        say(f"Matching stars with the catalog for a {field:.2g}° field...")
        attempt = None
        try:
            for stars in rankings:  # The catalog is cached after the first
                attempt = _try_field(stars, diag_px, field, ra, dec, progress)
                if attempt is not None:
                    break
        except SolveFailed as e:
            catalog_error = e
            continue
        if attempt is None:
            continue
        transform, sky, catalog = attempt
        _count, mirrored, a, b, _found = transform
        image = stars[:, 0] + 1j * stars[:, 1]
        xy, matched = _refine(stars, sky, abs(a), a * (np.conj(image) if mirrored else image) + b)
        if len(matched) < MIN_MATCHED:
            continue

        # Once more with fainter stars, where the catalog has them quickly
        # (Tycho-2): the brightest stars of a processed image are often bloated
        # or saturated, and the fainter ones' centers are more exact
        if field * 0.6 >= TYCHO_RADIUS:
            try:
                deep = _vizier_stars(*TYCHO, ra, dec, field * 0.6, DEEP_DEPTH)
            except SolveFailed:
                deep = np.empty((0, 3))
            if len(deep) > len(catalog):
                deep_xi, deep_eta = _project(deep[:, 0], deep[:, 1], ra, dec)
                deep_sky = deep_xi + 1j * deep_eta
                matrix, offset = _affine(xy, np.column_stack([sky.real, sky.imag])[matched])
                fitted = stars @ matrix.T + offset
                deep_xy, deep_matched = _refine(stars, deep_sky, abs(a), fitted[:, 0] + 1j * fitted[:, 1])
                if len(deep_matched) > len(matched):
                    xy, matched, sky, catalog = deep_xy, deep_matched, deep_sky, deep

        # TAN solution about the image center, from the matched stars
        center = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
        sky_xy = np.column_stack([sky.real, sky.imag])
        matrix, offset = _affine(xy, sky_xy[matched])
        ra_c, dec_c = _deproject(*(matrix @ center + offset), ra, dec)
        xi, eta = _project(catalog[matched, 0], catalog[matched, 1], ra_c, dec_c)
        matrix, offset = _affine(xy, np.column_stack([xi, eta]))
        residual = np.hypot(*(xy @ matrix.T + offset - np.column_stack([xi, eta])).T)
        ref = np.linalg.solve(matrix, -offset)                      # Analysis pixel at ra_c, dec_c

        # To original pixels (an analysis pixel covers factor x factor of them)
        cd = matrix / factor
        x0 = (ref[0] + 0.5) * factor - 0.5
        y0 = (ref[1] + 0.5) * factor - 0.5
        if os.path.splitext(path)[1].lower() in FITS_EXTENSIONS:
            # FITS rows numbered in stored order, as ASTAP does (the first row on top)
            crpix2, cd = y0 + 1, cd
        else:
            # Rows numbered up from the bottom row
            crpix2, cd = height - y0, cd @ np.diag([1.0, -1.0])
        solution = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                    "CRPIX1": float(x0 + 1), "CRPIX2": float(crpix2),
                    "CRVAL1": float(ra_c), "CRVAL2": float(dec_c),
                    "CD1_1": float(cd[0, 0]), "CD1_2": float(cd[0, 1]),
                    "CD2_1": float(cd[1, 0]), "CD2_2": float(cd[1, 1]),
                    "PLTSOLVD": "T"}
        pixel_scale = math.sqrt(abs(np.linalg.det(cd))) * 3600
        info = {"stars": len(matched), "rms_arcsec": float(np.sqrt(np.mean(residual ** 2))) * 3600,
                "scale": pixel_scale, "field": field, "tries": tries,
                "seconds": time.perf_counter() - started}
        logger.info(f"Built-in solve of {path}: {info}")
        return solution, info
    if catalog_error is not None:
        raise catalog_error
    raise SolveFailed("No match with the catalog near the object's position")


def equipment_scales(path):
    """Image scales (arcsec/pixel) of the plate-solved images in the library
    taken with the same equipment as path, the most common first - [] if none."""
    try:
        import sqlite3
        from collections import Counter
        from ResourceManager import ResourceManager
        from AnnotationOverlay import read_wcs_header
        import EquipmentMapper
        conn = sqlite3.connect(str(ResourceManager.get_database_path()))
        try:
            rows = [(row[0], EquipmentMapper.image_equipment(row[1:])["equipment_key"]) for row in conn.execute(
                f"SELECT ui.image_path, {EquipmentMapper.select_sql('ui')} FROM userimages ui "
                "WHERE ui.image_path IS NOT NULL AND ui.image_path != ''").fetchall()]
        finally:
            conn.close()
        wanted = os.path.normcase(os.path.normpath(path))
        mine = next((key for p, key in rows if os.path.normcase(os.path.normpath(p)) == wanted), None)
        if mine is None or mine == "|":  # Not in the gallery, or no telescope or camera
            return []
        found = Counter()
        for other, key in rows:
            if key != mine or os.path.normcase(os.path.normpath(other)) == wanted or not os.path.isfile(other):
                continue
            header = read_wcs_header(other)
            if not header:
                continue
            if "CD1_1" in header:
                value = math.sqrt(abs(header["CD1_1"] * header.get("CD2_2", 0)
                                      - header.get("CD1_2", 0) * header.get("CD2_1", 0))) * 3600
            else:
                value = abs(float(header.get("CDELT1", 0))) * 3600
            if value > 0:
                found[round(math.log(value) / math.log(1.05))] += 1  # Within 5% counts as the same
        return [1.05 ** bucket for bucket, _n in found.most_common()]
    except Exception as e:
        logger.debug(f"No equipment scales for {path}: {e}")
        return []


# ---- Command line ------------------------------------------------------------

def _hint_for(path):
    """(ra, dec) of the image's object from the database, or (None, None)."""
    import sqlite3
    from ResourceManager import ResourceManager
    conn = sqlite3.connect(str(ResourceManager.get_database_path()))
    try:
        rows = conn.execute("SELECT ui.image_path, d.ra, d.dec FROM userimages ui "
                            "JOIN dsodetail d ON d.id = ui.dsodetailid").fetchall()
    finally:
        conn.close()
    wanted = os.path.normcase(os.path.normpath(path))  # Stored paths mix / and \
    return next(((ra, dec) for stored, ra, dec in rows
                 if stored and os.path.normcase(os.path.normpath(stored)) == wanted), (None, None))


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Plate solve images with the built-in solver and compare "
                                                 "with their saved (e.g. ASTAP) solutions")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--ra", type=float, help="Approximate center RA, degrees (default: the image's object)")
    parser.add_argument("--dec", type=float, help="Approximate center Dec, degrees")
    args = parser.parse_args(argv)
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication(["NativePlateSolver"])  # For reading images
    from AnnotationOverlay import read_wcs_header, WCSTransform, flip_wcs_rows

    for path in args.images:
        ra, dec = (args.ra, args.dec) if args.ra is not None else _hint_for(path)
        name = os.path.basename(path)
        if ra is None:
            print(f"{name}: no position hint (use --ra/--dec)")
            continue
        try:
            solution, info = solve(path, ra, dec)
        except SolveFailed as e:
            print(f"{name}: FAILED - {e}")
            continue
        line = (f"{name}: {info['stars']} stars, rms {info['rms_arcsec']:.1f}\", "
                f"{info['scale']:.2f}\"/px, {info['seconds']:.1f}s, {info['tries']} field(s) tried")
        saved = read_wcs_header(path)
        if saved:
            from AnnotationOverlay import image_pixel_size
            w, h = image_pixel_size(path)
            fits = os.path.splitext(path)[1].lower() in FITS_EXTENSIONS
            mine = WCSTransform(flip_wcs_rows(solution, h) if fits else solution, w, h)
            theirs = WCSTransform(flip_wcs_rows(saved, h) if fits else saved, w, h)
            # Where the corners and center land, by each solution
            worst = 0.0
            for x, y in ((0, 0), (w, 0), (0, h), (w, h), (w / 2, h / 2)):
                r1, d1 = mine.pixel_to_sky(x, y)
                r2, d2 = theirs.pixel_to_sky(x, y)
                worst = max(worst, math.hypot((r1 - r2) * math.cos(math.radians(d1)), d1 - d2) * 3600)
            their_scale = math.sqrt(abs(theirs.cd1_1 * theirs.cd2_2 - theirs.cd1_2 * theirs.cd2_1)) * 3600
            line += f" | vs saved: worst {worst:.1f}\" ({worst / their_scale:.1f} px)"
        print(line)
    del app


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main(sys.argv[1:])
