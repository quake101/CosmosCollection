#!/usr/bin/env python3
"""
Frame Quality
Measures a light frame's stars (count, FWHM, HFR, eccentricity and how aligned
the elongated stars are), sky background and noise, then grades each frame
against the others in its group (same filter/exposure/binning) so only the best
subs get stacked. Pure numpy/astropy - no Qt - so it runs in worker threads and
from the command line:

    python FrameQuality.py <folder or files...> [--workers N]
"""

import io
import os
import sys
import math
import time
import logging
from dataclasses import dataclass, field, asdict

import numpy as np

import SessionFileScanner

logger = logging.getLogger(__name__)

# Bump whenever measurements change so cached results get re-analyzed.
# 2: added outside_light (guiding jumps, tails and halos)
# 3: eccentricity from adaptive moments; FWHM corrected to the true FWHM
# 4: outside light stored at fixed radii (outside_profile), read at the group's core radius
# 5: satellite/plane trails
ANALYSIS_VERSION = 5

TILE = 64                 # background mesh tile size (analysis pixels)
DETECT_SIGMA = 5.0        # detection threshold on the smoothed image
PEAK_RADIUS = 2           # a star's peak is the maximum of a 5x5 window
HOT_PIXEL_RATIO = 0.15    # 2nd-brightest pixel of a star's 3x3 core vs its brightest
MIN_MEASURE_SNR = 30.0    # stars measured for shape must peak this far above the noise...
MIN_MEASURED = 15         # ...unless that leaves fewer than this many
MAX_MEASURED = 1500       # brightest unsaturated stars measured for shape
# Transparency is the median flux of the stars in a band of brightness ranks
# (saturated stars included in the ranking - they rank first). Seeing, focus
# and moonlight don't change that order, so a band holds the same stars in
# every frame, and cloud dims them. Grading uses, per group, the brightest
# band no frame has saturated stars in: a soft frame's bright stars stop
# saturating, so a band that dips into saturation would measure it brighter.
TRANSPARENCY_BANDS = ((100, 300), (300, 1000), (1000, 3000))
SATURATION_FRACTION = 0.9
MAX_ANALYSIS_PIXELS = 12_000_000  # bigger mono frames are binned 2x2 for speed
# Light outside the star cores: of each bright star's light within OUTSIDE_RADIUS
# sensor pixels, the share beyond a core radius that holds essentially all of a
# clean star. A guiding jump leaves a round core plus a faint tail that the
# shape measurements - which only see the brighter pixels - miss; tails, double
# images and dew/cloud halos all raise this. Each frame stores the share at the
# OUTSIDE_PROFILE_RADII, and grading reads it at OUTSIDE_CORE_FWHM x the larger
# of the frame's own FWHM and its group's median. A core sized only by the
# frame's own FWHM shrinks in good seeing while the faint halo every star has
# stays put, so the sharpest frames read as the worst; one fixed core for the
# whole group lets a soft frame's own wider wings spill past it, counting its
# FWHM twice.
OUTSIDE_RADIUS = 40
OUTSIDE_CORE_FWHM = 3.5   # x the true FWHM
OUTSIDE_PROFILE_RADII = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 28, 33, 40, 48, 57, 68, 80)  # sensor pixels
OUTSIDE_STARS = 200

# Satellite / plane trails, in analysis pixels. A line filter (along a short
# segment vs across it, at TRAIL_ORIENTATIONS angles) on a 2x pooled image picks
# out thin lines but not round stars; a Hough transform finds straight runs of
# them; each candidate is then checked by the median brightness along the line
# against the sky beside it - the median ignores the stars a line crosses, and
# a broad nebula filament is as bright beside the line as on it.
TRAIL_ORIENTATIONS = 12
TRAIL_LINE_SIGMA = 4.0      # line-filter threshold for the Hough points
TRAIL_ANGLE_STEP = 0.25     # degrees
TRAIL_SEGMENT = 40          # pixels per continuity segment along a candidate
TRAIL_LIT_SIGMA = 3.0       # a segment's median excess (noise units) to count as lit
TRAIL_MIN_LENGTH = 300      # lit length of a trail...
TRAIL_SHORT_LENGTH = 120    # ...or this much if it's bright (meteors, flares, corners)
TRAIL_SHORT_EXCESS = 10.0
TRAIL_CANDIDATES = 15       # Hough peaks checked per frame
TRAIL_SKIP_ECCENTRICITY = 0.85  # stars this elongated are trailed themselves - every star is a "trail"

# Moments are taken over pixels above this fraction of each star's peak, so a
# faint star and a bright one are truncated alike (a noise-level cut would make
# stars dimmed by cloud look sharper). For a Gaussian that keeps
# _TRUNCATED_VARIANCE of the true second moment, which is divided back out.
_MOMENT_FRACTION = 0.1
_TRUNCATED_VARIANCE = ((1 - (1 + math.log(1 / _MOMENT_FRACTION)) * _MOMENT_FRACTION)
                       / (1 - _MOMENT_FRACTION))
_PIXEL_VARIANCE = 1 / 12  # a uniform pixel's own contribution to a second moment
_SIGMA_TO_FWHM = 2 * math.sqrt(2 * math.log(2))
_ADAPTIVE_ITERATIONS = 8
_ADAPTIVE_CHUNK = 256     # stars per pass, to bound memory with big boxes

# The moment FWHM reads high: real stars have wider wings than the Gaussian the
# truncation correction assumes, and on a color camera filling in the red and
# blue sites from their green neighbors blurs small stars further. Measured
# FWHM of simulated Moffat (beta 4) stars of known FWHM, measured exactly as
# here, maps a measurement back to the true FWHM. Checked against PixInsight on
# broadband color subs (2.21 vs its 2.27 px); on dual-band subs PixInsight reads
# higher, since its debayered stars include the softer-focused Ha (red) image.
_FWHM_TRUE = (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0)
_FWHM_MEASURED_COLOR = (1.989, 2.099, 2.237, 2.392, 2.542, 2.740, 2.973, 3.275, 3.558, 4.064, 4.623, 5.653, 6.720)
_FWHM_MEASURED_MONO = (1.373, 1.593, 1.779, 2.033, 2.228, 2.459, 2.680, 3.007, 3.331, 3.888, 4.433, 5.527, 6.606)


def _true_fwhm(measured, color):
    """The true FWHM (box pixels) for a moment FWHM measured on a color
    (green-filled) or mono box - past either end of the table, the end's ratio."""
    table = _FWHM_MEASURED_COLOR if color else _FWHM_MEASURED_MONO
    if measured <= table[0]:
        return measured * _FWHM_TRUE[0] / table[0]
    if measured >= table[-1]:
        return measured * _FWHM_TRUE[-1] / table[-1]
    return float(np.interp(measured, table, _FWHM_TRUE))


@dataclass
class FrameMetrics:
    """One frame's measurements. Pixel sizes are in sensor pixels (as captured,
    before this module's own binning). None when it couldn't be measured."""
    path: str
    ok: bool = False
    error: str = ""
    stars: int = 0                    # detected stars
    measured: int = 0                 # stars measured for shape
    fwhm_px: float = None
    fwhm_arcsec: float = None
    hfr_px: float = None
    eccentricity: float = None        # 0 round .. 1 a line (PixInsight's definition)
    alignment: float = None           # 0 random .. 1 every elongated star points the same way
    background: float = None          # median sky, ADU per pixel
    noise: float = None               # background noise, ADU per pixel
    gradient_pct: float = None        # background spread across the frame, % of the median
    star_snr: float = None            # median peak / noise of the measured stars
    star_flux: list = None            # median flux per TRANSPARENCY_BANDS band (None past the last star) - drops under cloud
    saturated: int = 0                # saturated stars
    outside_profile: list = None      # share of star light beyond each OUTSIDE_PROFILE_RADII radius (None past the box) - rises with tails and halos
    trails: list = None               # satellite/plane trails, [x0, y0, x1, y1, excess] in sensor pixels; None if not checked
    width: int = 0
    height: int = 0
    seconds: float = 0.0              # time taken to analyze
    version: int = ANALYSIS_VERSION

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---- Loading ----------------------------------------------------------------

def _read_frame(path):
    """(pixels, header dict) for a FITS or XISF file."""
    from ImageLoader import read_astro_array, _read_fits_array
    ext = os.path.splitext(path)[1].lower()
    if ext in SessionFileScanner.XISF_EXTENSIONS:
        header = SessionFileScanner.extract_xisf_header(path)
        data = read_astro_array(path)
    else:
        # One whole-file read, parsed in memory: about twice as fast from a
        # network share as astropy's many small reads plus a separate header read.
        from astropy.io import fits
        with open(path, "rb") as f:
            raw = f.read()
        # Each open gets its own buffer - astropy closes the one it was given
        with fits.open(io.BytesIO(raw)) as hdul:
            full_header = hdul[0].header
            header = {kw: full_header[kw] for kw in SessionFileScanner.FITS_KEYWORDS if kw in full_header}
        data = _read_fits_array(io.BytesIO(raw))
    if data is None:
        raise ValueError("No image data")
    return data, header


def _full_scale(data):
    """The value a saturated pixel reaches."""
    if np.issubdtype(data.dtype, np.integer):
        return float(np.iinfo(data.dtype).max)
    peak = float(np.nanmax(data))
    return 1.0 if peak <= 1.0 else peak


def _bin2(img):
    """Sum 2x2 blocks (an odd last row/column is dropped)."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    out = img[0:h:2, 0:w:2].astype(np.float32)
    out += img[1:h:2, 0:w:2]
    out += img[0:h:2, 1:w:2]
    out += img[1:h:2, 1:w:2]
    return out


def _luminance(data, header):
    """A float32 mono image to analyze, how many sensor pixels each of its
    pixels covers per side, and how many sensor pixels were summed into one.
    Color-camera (Bayer) subs are binned 2x2 into one superpixel per CFA cell
    so the color pattern can't look like stars or noise."""
    scale = 1
    summed = 1
    if data.ndim == 3:
        img = data.astype(np.float32).mean(axis=2)
    elif header.get("BAYERPAT"):
        img = _bin2(data)
        scale, summed = 2, 4
    else:
        img = data.astype(np.float32)
    if img.size > MAX_ANALYSIS_PIXELS:
        img = _bin2(img)
        scale, summed = scale * 2, summed * 4
    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return img, scale, summed


# ---- Background -------------------------------------------------------------

def _clipped_stats(values, sigma=3.0, iterations=4):
    """Median and robust standard deviation (from the MAD) after sigma clipping."""
    v = values[np.isfinite(values)]
    med, std = 0.0, 0.0
    for _ in range(iterations):
        if v.size == 0:
            break
        med = float(np.median(v))
        std = float(np.median(np.abs(v - med))) * 1.4826
        if std <= 0:
            break
        keep = np.abs(v - med) < sigma * std
        if keep.all():
            break
        v = v[keep]
    return med, std


def _median_last_axis(values):
    """Median along the last axis ignoring NaNs - one vectorized sort instead of
    np.nanmedian's per-row Python loop."""
    ordered = np.sort(values, axis=-1)  # NaNs sort to the end
    count = np.isfinite(ordered).sum(axis=-1, keepdims=True)
    last = ordered.shape[-1] - 1
    lo = np.take_along_axis(ordered, np.clip((count - 1) // 2, 0, last), axis=-1)
    hi = np.take_along_axis(ordered, np.clip(count // 2, 0, last), axis=-1)
    return np.where(count > 0, (lo + hi) / 2, np.nan)


def _tile_medians(img, tile, step=4):
    """Sigma-clipped median of each tile, from every step-th pixel each way."""
    sample = img[::step, ::step]
    t = tile // step
    ny, nx = max(1, sample.shape[0] // t), max(1, sample.shape[1] // t)
    th, tw = sample.shape[0] // ny, sample.shape[1] // nx
    tiles = (sample[:ny * th, :nx * tw]
             .reshape(ny, th, nx, tw).transpose(0, 2, 1, 3).reshape(ny, nx, th * tw))
    for _ in range(3):
        med = _median_last_axis(tiles)
        spread = _median_last_axis(np.abs(tiles - med)) * 1.4826
        tiles = np.where((spread > 0) & (np.abs(tiles - med) > 3 * spread), np.nan, tiles)
    mesh = _median_last_axis(tiles)[..., 0].astype(np.float32)

    # A 3x3 median over the mesh removes tiles dominated by one bright star
    padded = np.pad(mesh, 1, mode="edge")
    shifts = [padded[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3)]
    mesh = np.median(np.stack(shifts), axis=0).astype(np.float32)
    return mesh, th * step, tw * step


def _axis_weights(n_out, n_mesh, step):
    centers = (np.arange(n_out) + 0.5) / step - 0.5
    i0 = np.clip(np.floor(centers).astype(int), 0, max(n_mesh - 2, 0))
    i1 = np.minimum(i0 + 1, n_mesh - 1)
    frac = np.clip(centers - i0, 0, 1).astype(np.float32) if n_mesh > 1 else np.zeros(n_out, np.float32)
    return i0, i1, frac


def _upsample(mesh, shape, tile_h, tile_w):
    """Bilinear interpolation of the tile mesh up to the full image."""
    h, w = shape
    y0, y1, fy = _axis_weights(h, mesh.shape[0], tile_h)
    rows = mesh[y0] * (1 - fy)[:, None] + mesh[y1] * fy[:, None]
    x0, x1, fx = _axis_weights(w, mesh.shape[1], tile_w)
    return rows[:, x0] * (1 - fx) + rows[:, x1] * fx


# ---- Stars ------------------------------------------------------------------

def _smooth(img):
    """Separable [1 2 1] smoothing - suppresses single-pixel noise before detection."""
    p = np.pad(img, 1, mode="edge")
    t = p[:-2] + 2 * p[1:-1] + p[2:]
    return (t[:, :-2] + 2 * t[:, 1:-1] + t[:, 2:]) / 16


def _max_filter(img, radius):
    h, w = img.shape
    p = np.pad(img, radius, mode="constant", constant_values=-np.inf)
    rows = p[0:h].copy()
    for k in range(1, 2 * radius + 1):
        np.maximum(rows, p[k:k + h], out=rows)
    out = rows[:, 0:w].copy()
    for k in range(1, 2 * radius + 1):
        np.maximum(out, rows[:, k:k + w], out=out)
    return out


def _patches(img, ys, xs, radius):
    offsets = np.arange(-radius, radius + 1)
    return img[ys[:, None, None] + offsets[None, :, None], xs[:, None, None] + offsets[None, None, :]]


def _detect(sub, smoothed, smooth_noise):
    """Star peaks: local maxima of the smoothed image above the threshold,
    with hot pixels and cosmic rays (a lone bright pixel) dropped."""
    peaks = (smoothed >= _max_filter(smoothed, PEAK_RADIUS)) & (smoothed > DETECT_SIGMA * smooth_noise)
    margin = PEAK_RADIUS + 1
    peaks[:margin] = peaks[-margin:] = False
    peaks[:, :margin] = peaks[:, -margin:] = False
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return ys, xs, np.empty(0, np.float32)

    # A flat-topped (saturated) core gives several equal maxima - keep one
    _, first = np.unique((ys // 3) * (sub.shape[1] // 3 + 1) + xs // 3, return_index=True)
    ys, xs = ys[first], xs[first]

    core = np.sort(_patches(sub, ys, xs, 1).reshape(ys.size, 9), axis=1)
    real = core[:, -2] > HOT_PIXEL_RATIO * core[:, -1]
    return ys[real], xs[real], core[real, -1]


def _green_odd(header):
    """True when a Bayer sensor's green pixels sit where row + column is odd
    (RGGB/BGGR), False when even (GRBG/GBRG)."""
    pattern = str(header.get("BAYERPAT") or "").strip().upper()
    try:
        shift = int(float(header.get("XBAYROFF") or 0)) + int(float(header.get("YBAYROFF") or 0))
    except (TypeError, ValueError):
        shift = 0
    return (pattern in ("RGGB", "BGGR")) != bool(shift % 2)


def _sensor_boxes(raw, ys, xs, radius, green_odd=None):
    """Boxes of sensor pixels around each star. On a color (Bayer) sensor the
    red and blue pixels are replaced by the mean of their four green
    neighbors, so the star is measured at full resolution without the color
    pattern. green_odd is None for a mono sensor.
    (Shapes from the green pixels alone were tried and checked against
    PixInsight: FWHM barely changed and eccentricity got noisier.)"""
    if green_odd is None:
        return _patches(raw, ys, xs, radius).astype(np.float32)
    p = _patches(raw, ys, xs, radius + 1).astype(np.float32)
    neighbors = (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]) / 4
    offsets = np.arange(-radius, radius + 1)
    parity = (ys[:, None, None] + xs[:, None, None] + offsets[None, :, None] + offsets[None, None, :]) % 2
    return np.where(parity == int(green_odd), p[:, 1:-1, 1:-1], neighbors)


def _adaptive_shape(boxes, inside, yy, xx, fwhm, cy, cx):
    """Per-star eccentricity and angle from adaptive moments: second moments
    weighted by an elliptical Gaussian matched to the star itself, iterated.
    Unlike the peak-fraction cut, the weight falls off smoothly, so the noisy
    pixels at the edge of a star barely count - eccentricity drifts far less
    with noise (moonlight, haze) and follows PixInsight's PSF fits more closely.
    NaN where a star's weighted flux vanished."""
    ecc = np.full(len(boxes), np.nan, np.float32)
    angle = np.zeros(len(boxes), np.float32)
    for start in range(0, len(boxes), _ADAPTIVE_CHUNK):
        part = slice(start, start + _ADAPTIVE_CHUNK)
        b = np.where(inside, boxes[part], 0)
        n = len(b)
        mxx = ((fwhm[part] / _SIGMA_TO_FWHM) ** 2).astype(np.float32)
        myy = mxx.copy()
        mxy = np.zeros(n, np.float32)
        y0, x0 = cy[part].copy(), cx[part].copy()
        ok = np.ones(n, bool)
        for _ in range(_ADAPTIVE_ITERATIONS):
            limit = 0.95 * np.sqrt(mxx * myy)  # keep the weight's ellipse a real ellipse
            mxy = np.clip(mxy, -limit, limit)
            det = mxx * myy - mxy ** 2
            dy = yy - y0[:, None, None]
            dx = xx - x0[:, None, None]
            q = (myy[:, None, None] * dx * dx - 2 * mxy[:, None, None] * dx * dy
                 + mxx[:, None, None] * dy * dy) / det[:, None, None]
            w = np.exp(-0.5 * np.minimum(q, 60)) * b
            flux = w.sum(axis=(1, 2))
            ok &= flux > 0
            flux = np.where(flux > 0, flux, 1.0)
            y0 = np.clip(y0 + (w * dy).sum(axis=(1, 2)) / flux, -2, 2)
            x0 = np.clip(x0 + (w * dx).sum(axis=(1, 2)) / flux, -2, 2)
            dy = yy - y0[:, None, None]
            dx = xx - x0[:, None, None]
            # A matched Gaussian weight halves a Gaussian star's moments - doubled back
            mxx = np.clip(2 * (w * dx * dx).sum(axis=(1, 2)) / flux, 0.3, 400)
            myy = np.clip(2 * (w * dy * dy).sum(axis=(1, 2)) / flux, 0.3, 400)
            mxy = 2 * (w * dx * dy).sum(axis=(1, 2)) / flux
        limit = 0.99 * np.sqrt(mxx * myy)
        mxy = np.clip(mxy, -limit, limit)
        half_sum = (mxx + myy) / 2
        half_diff = np.sqrt(((mxx - myy) / 2) ** 2 + mxy ** 2)
        major = np.maximum(half_sum + half_diff - _PIXEL_VARIANCE, 1e-6)
        minor = np.minimum(np.maximum(half_sum - half_diff - _PIXEL_VARIANCE, 1e-6), major)
        ecc[part] = np.where(ok, np.sqrt(1 - minor / major), np.nan)
        angle[part] = 0.5 * np.arctan2(2 * mxy, mxx - myy)
    return ecc, angle


def _measure(boxes, shape=True):
    """Per-star FWHM/HFR (box pixels) from intensity-weighted second moments,
    eccentricity and angle from adaptive moments (None unless shape), plus each
    star's total flux inside the box's circle (aperture photometry - no
    peak-relative cut, which would let bloated stars keep more of their light)."""
    radius = boxes.shape[1] // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")

    border = np.maximum(np.abs(yy), np.abs(xx)) == radius
    boxes = boxes - np.median(boxes[:, border], axis=1)[:, None, None]
    core = min(2, radius)  # the box is centered on the binned peak, so the true one can be a pixel or two off
    peak = boxes[:, radius - core:radius + core + 1, radius - core:radius + core + 1].reshape(len(boxes), -1).max(axis=1)
    inside = np.hypot(yy, xx) <= radius
    w = np.where((boxes >= _MOMENT_FRACTION * peak[:, None, None]) & inside, boxes, 0)

    flux = w.sum(axis=(1, 2))
    good = flux > 0
    w, flux = w[good], flux[good]
    aperture = np.where(inside, boxes[good], 0).sum(axis=(1, 2))
    cy = (w * yy).sum(axis=(1, 2)) / flux
    cx = (w * xx).sum(axis=(1, 2)) / flux
    dy = yy - cy[:, None, None]
    dx = xx - cx[:, None, None]
    iyy = (w * dy * dy).sum(axis=(1, 2)) / flux
    ixx = (w * dx * dx).sum(axis=(1, 2)) / flux
    ixy = (w * dx * dy).sum(axis=(1, 2)) / flux
    hfr = (w * np.hypot(dy, dx)).sum(axis=(1, 2)) / flux

    half_sum = (ixx + iyy) / 2
    half_diff = np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)
    major = np.maximum((half_sum + half_diff - _PIXEL_VARIANCE) / _TRUNCATED_VARIANCE, 1e-6)
    minor = np.maximum((half_sum - half_diff - _PIXEL_VARIANCE) / _TRUNCATED_VARIANCE, 1e-6)
    minor = np.minimum(minor, major)

    fwhm = _SIGMA_TO_FWHM * np.sqrt((major + minor) / 2)
    eccentricity = angle = None
    if shape:
        eccentricity, angle = _adaptive_shape(boxes[good], inside, yy, xx, fwhm, cy, cx)
    return fwhm, hfr, eccentricity, angle, aperture


def _outside_profile(boxes, wide, radii):
    """Median share of each star's light within `wide` pixels that lies beyond
    each of `radii` pixels (None from `wide` on). The box's corners outside
    `wide` are the local background."""
    half = boxes.shape[1] // 2
    offsets = np.arange(-half, half + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    distance = np.hypot(yy, xx)
    light = boxes - np.median(boxes[:, distance > wide], axis=1)[:, None, None]
    total = light[:, distance <= wide].sum(axis=1)
    good = total > 0
    if not good.any():
        return None
    light, total = light[good], total[good]
    return [float(np.median(1 - light[:, distance <= r].sum(axis=1) / total)) if r < wide else None
            for r in radii]


def outside_at(metrics, group_fwhm):
    """A frame's share of star light beyond its core radius (OUTSIDE_CORE_FWHM x
    the larger of its own FWHM and its group's median), from its
    outside_profile - None when it wasn't measured that far out."""
    profile = metrics.outside_profile
    if not profile or metrics.fwhm_px is None or not group_fwhm:
        return None
    radius = OUTSIDE_CORE_FWHM * max(metrics.fwhm_px, group_fwhm)
    points = [(r, v) for r, v in zip(OUTSIDE_PROFILE_RADII, profile) if v is not None]
    if not points or not points[0][0] <= radius <= points[-1][0]:
        return None
    radii, values = zip(*points)
    return float(np.interp(radius, radii, values))


def _shape_stats(boxes):
    """Frame medians of the per-star shape measurements, dropping blends."""
    fwhm, hfr, ecc, angle, _flux = _measure(boxes)
    if fwhm.size == 0:
        return None
    median_fwhm = np.median(fwhm)
    single = fwhm < 2.5 * median_fwhm   # two touching stars measure as one fat one
    fwhm, hfr = fwhm[single], hfr[single]
    shaped = single.copy()
    shaped[single] = np.isfinite(ecc[single])  # the adaptive fit can fail on a star
    ecc, angle = ecc[shaped], angle[shaped]

    # Mean direction of the elongation (angles double, since 0 and 180 degrees are the same axis)
    weights = ecc.sum()
    alignment = float(abs((ecc * np.exp(2j * angle)).sum()) / weights) if weights > 0 else 0.0
    return {
        "count": int(fwhm.size), "fwhm": float(np.median(fwhm)), "hfr": float(np.median(hfr)),
        "eccentricity": float(np.median(ecc)) if ecc.size else None, "alignment": alignment,
    }


# ---- Trails -----------------------------------------------------------------

def _pool2(img):
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _segment_mean(padded, pad, shape, angle):
    """Per pixel, the dimmest of three 4-pixel sub-segment means along a
    12-pixel segment at angle (noise units): a trail lights all three, a star -
    a few pixels across - only the middle one, so stars don't read as lines."""
    h, w = shape
    dy, dx = math.sin(angle), math.cos(angle)
    dimmest = None
    for group in ((-6, -5, -4, -3), (-2, -1, 0, 1), (2, 3, 4, 5)):
        part = np.zeros(shape, np.float32)
        for k in group:
            oy, ox = int(round(k * dy)), int(round(k * dx))
            part += padded[pad + oy:pad + oy + h, pad + ox:pad + ox + w]
        dimmest = part if dimmest is None else np.minimum(dimmest, part)
    # Each direction against its own noise, so none (the diagonals' pixel steps) is favored
    center, spread = _clipped_stats(dimmest[::3, ::3].ravel())
    return (dimmest - center) / max(spread, 1e-9)


def _line_filter(img):
    """How much brighter each pixel's surroundings are along some direction
    than across it, in noise units - high on thin lines, low on round stars."""
    pad = 7
    padded = np.pad(img, pad, mode="edge")
    half = TRAIL_ORIENTATIONS // 2
    best = None
    for o in range(half):
        along = _segment_mean(padded, pad, img.shape, math.pi * o / TRAIL_ORIENTATIONS)
        across = _segment_mean(padded, pad, img.shape, math.pi * (o + half) / TRAIL_ORIENTATIONS)
        contrast = np.abs(along - across)
        best = contrast if best is None else np.maximum(best, contrast)
    return best


def _hough_votes(ys, xs, thetas, diag):
    """Hough accumulator (angle x 2-pixel distance bins over -diag..diag)."""
    bins = diag + 1
    cos, sin = np.cos(thetas).astype(np.float32), np.sin(thetas).astype(np.float32)
    votes = np.empty((len(thetas), bins), np.int32)
    for i in range(len(thetas)):
        votes[i] = np.bincount(((xs * cos[i] + ys * sin[i] + diag) / 2).astype(np.int32), minlength=bins)[:bins]
    return votes


def _line_profile(sub, noise, theta, rho):
    """Points along a line (x, y) and their excess in noise units: the brightest
    pixel within one of the line minus the median 5-9 pixels to either side."""
    h, w = sub.shape
    c, s = math.cos(theta), math.sin(theta)
    reach = math.hypot(h, w)
    t = np.arange(-reach, reach, 1.0)
    x, y = rho * c - t * s, rho * s + t * c
    inside = (x >= 10) & (x < w - 10) & (y >= 10) & (y < h - 10)
    x, y = x[inside], y[inside]
    if x.size == 0:
        return x, y, x

    def sample(d):
        return sub[np.round(y + d * s).astype(int), np.round(x + d * c).astype(int)]
    on = np.max([sample(d) for d in (-1, 0, 1)], axis=0)
    off = np.median([sample(d) for d in (-9, -7, -5, 5, 7, 9)], axis=0)
    return x, y, (on - off) / noise


def _lit_span(excess):
    """(first, end) samples of the longest run of lit segments along a line -
    one unlit segment at a time is allowed, since plane lights blink - or None."""
    n = len(excess) // TRAIL_SEGMENT
    if n == 0:
        return None
    lit = np.median(excess[:n * TRAIL_SEGMENT].reshape(n, TRAIL_SEGMENT), axis=1) >= TRAIL_LIT_SIGMA
    best = start = last = None
    for i in np.nonzero(lit)[0]:
        if start is None or i - last > 2:
            start = i
        last = i
        if best is None or last - start > best[1] - best[0]:
            best = (start, last)
    return (best[0] * TRAIL_SEGMENT, (best[1] + 1) * TRAIL_SEGMENT) if best else None


def _find_trails(sub, noise):
    """Satellite and plane trails in a background-subtracted analysis image, as
    [(x0, y0, x1, y1, excess)] in its pixels, excess the trail's median
    brightness above the sky in noise units."""
    contrast = _line_filter(_pool2(sub))
    center, spread = _clipped_stats(contrast[::3, ::3].ravel())
    ys, xs = np.nonzero(contrast > center + TRAIL_LINE_SIGMA * spread)
    del contrast
    ys = ys.astype(np.float32) * 2 + 0.5  # pooled pixel centers, in analysis pixels
    xs = xs.astype(np.float32) * 2 + 0.5
    thetas = np.deg2rad(np.arange(0, 180, TRAIL_ANGLE_STEP))
    diag = int(math.ceil(math.hypot(*sub.shape)))
    votes = _hough_votes(ys, xs, thetas, diag)

    trails = []
    for _ in range(TRAIL_CANDIDATES):
        i, j = np.unravel_index(np.argmax(votes), votes.shape)
        if votes[i, j] < TRAIL_SHORT_LENGTH / 4:  # pooled pixels are 2 apart, and allow gaps
            break
        votes[max(0, i - 8):i + 9, max(0, j - 10):j + 11] = 0
        # The Hough cell is coarse - fit the line to the image itself
        best = None
        for dth in np.deg2rad(np.arange(-0.3, 0.31, 0.1)):
            for drho in (-2.0, -1.0, 0.0, 1.0, 2.0):
                theta, rho = thetas[i] + dth, j * 2 - diag + 1.0 + drho
                x, y, excess = _line_profile(sub, noise, theta, rho)
                span = _lit_span(excess)
                if span:
                    key = (span[1] - span[0], float(np.median(excess[span[0]:span[1]])))
                    if best is None or key > best[0]:
                        best = (key, theta, rho, x, y, span)
        if best is None:
            continue
        (length, brightness), theta, rho, x, y, span = best
        if length < TRAIL_MIN_LENGTH and (length < TRAIL_SHORT_LENGTH or brightness < TRAIL_SHORT_EXCESS):
            continue
        degrees = math.degrees(theta) % 90
        if min(degrees, 90 - degrees) < 0.2:
            continue  # along a sensor row or column - a defect, not a trail
        end = min(span[1], len(x)) - 1
        trails.append((float(x[span[0]]), float(y[span[0]]), float(x[end]), float(y[end]), brightness))
        # Take back the votes of the points along it, so it isn't found again
        near = np.abs(xs * math.cos(theta) + ys * math.sin(theta) - rho) <= 8
        if near.any():
            votes -= _hough_votes(ys[near], xs[near], thetas, diag)
    return trails


# ---- Analysis ---------------------------------------------------------------

def analyze_file(path):
    """Measure one light frame. Never raises - failures come back with ok=False."""
    started = time.perf_counter()
    result = FrameMetrics(path=path)
    try:
        data, header = _read_frame(path)
        full_scale = _full_scale(data)
        img, scale, summed = _luminance(data, header)
        result.height, result.width = data.shape[:2]

        mesh, tile_h, tile_w = _tile_medians(img, TILE)
        sub = img - _upsample(mesh, img.shape, tile_h, tile_w)
        smoothed = _smooth(sub)
        _, noise = _clipped_stats(sub[::3, ::3].ravel())
        _, smooth_noise = _clipped_stats(smoothed[::3, ::3].ravel())
        if noise <= 0 or smooth_noise <= 0:
            raise ValueError("Blank frame (no background noise)")

        sky = float(np.median(mesh))
        result.background = sky / summed
        result.noise = noise / math.sqrt(summed)  # summing n pixels grows noise by sqrt(n)
        low, high = np.percentile(mesh, [5, 95])
        result.gradient_pct = float((high - low) / sky * 100) if sky > 0 else None

        ys, xs, peaks = _detect(sub, smoothed, smooth_noise)
        del smoothed
        result.stars = int(ys.size)

        # Shape: the brightest unsaturated stars, far enough from the edges for a full box
        unsaturated = _patches(img, ys, xs, 1).reshape(ys.size, 9).max(axis=1) < SATURATION_FRACTION * full_scale * summed
        order = np.argsort(-peaks)
        order = order[unsaturated[order]]
        bright = order[peaks[order] >= MIN_MEASURE_SNR * noise]
        candidates = bright if bright.size >= MIN_MEASURED else order[:MIN_MEASURED]

        # Measured on the sensor's own pixels (the green ones on a color camera):
        # binned, stars are only a pixel or two across and the pixel grid alone
        # would make round stars look elongated.
        if data.ndim == 2:
            source, pixel = data, 1
            green_odd = _green_odd(header) if header.get("BAYERPAT") else None
            src_y, src_x = ys * scale + scale // 2, xs * scale + scale // 2
        else:
            source, pixel, green_odd = sub, scale, None
            src_y, src_x = ys, xs
        pad = 0 if green_odd is None else 1
        radius = 6 * scale // pixel
        max_radius = 20 * scale // pixel

        def boxable(selection, box_radius=None):
            """The stars in selection far enough from the edges for a full box."""
            edge = (box_radius or radius) + pad
            return selection[(src_y[selection] >= edge) & (src_y[selection] < source.shape[0] - edge)
                             & (src_x[selection] >= edge) & (src_x[selection] < source.shape[1] - edge)]

        stats = None
        for _ in range(2):
            chosen = boxable(candidates)[:MAX_MEASURED]
            if chosen.size == 0:
                break
            measured = _shape_stats(_sensor_boxes(source, src_y[chosen], src_x[chosen], radius, green_odd))
            if measured is None:
                break
            stats = measured
            stats["snr"] = float(np.median(peaks[chosen]) / noise)
            # Big (oversampled or defocused) stars need a bigger box - measure once more
            if stats["fwhm"] * 2 <= radius or radius >= max_radius:
                break
            radius = min(max_radius, int(math.ceil(stats["fwhm"] * 2)))

        if stats:
            result.measured = stats["count"]
            # Box sizes stay on the measured FWHM they were tuned on
            result.fwhm_px = _true_fwhm(stats["fwhm"], green_odd is not None) * pixel
            result.hfr_px = stats["hfr"] * pixel
            result.eccentricity = stats["eccentricity"]
            result.alignment = stats["alignment"]
            result.star_snr = stats["snr"]
            arcsec_per_px = _arcsec_per_pixel(header)
            if arcsec_per_px:
                result.fwhm_arcsec = result.fwhm_px * arcsec_per_px

            by_brightness = np.argsort(-peaks)
            result.saturated = int((~unsaturated).sum())
            result.star_flux = []
            for start, stop in TRANSPARENCY_BANDS:
                band = by_brightness[start:stop]
                band = boxable(band[unsaturated[band]])
                flux = (_measure(_sensor_boxes(source, src_y[band], src_x[band], radius, green_odd), shape=False)[4]
                        if band.size else np.empty(0))
                result.star_flux.append(float(np.median(flux)) if flux.size else None)

            # Wide enough for the tail, and well past the core of bloated stars
            wide = int(math.ceil(max(OUTSIDE_RADIUS / pixel, 4.5 * stats["fwhm"])))
            # Boxes run a little past the circle - that ring is each star's local background
            bright = boxable(candidates, wide + 4)[:OUTSIDE_STARS]
            if bright.size:
                result.outside_profile = _outside_profile(
                    _sensor_boxes(source, src_y[bright], src_x[bright], wide + 4, green_odd), wide,
                    [r / pixel for r in OUTSIDE_PROFILE_RADII])

        # Trails - not when the stars are trailed themselves, as every star would be one
        if result.eccentricity is None or result.eccentricity < TRAIL_SKIP_ECCENTRICITY:
            try:
                offset = (scale - 1) / 2
                result.trails = [[round(x0 * scale + offset, 1), round(y0 * scale + offset, 1),
                                  round(x1 * scale + offset, 1), round(y1 * scale + offset, 1), round(excess, 1)]
                                 for x0, y0, x1, y1, excess in _find_trails(sub, noise)]
            except Exception as e:
                logger.debug(f"Trail detection failed for {path}: {e}")
        result.ok = True
    except Exception as e:
        logger.debug(f"Frame quality analysis failed for {path}: {e}")
        result.error = str(e) or type(e).__name__
    result.seconds = time.perf_counter() - started
    return result


def _auto_stretch(values, background, noise, white, target=0.2):
    """Screen stretch to 0-1 like PixInsight's STF: shadows clipped a little
    below the sky, then a midtones transfer that puts the sky at target."""
    black = background - 2.8 * noise
    span = max(white - black, 1e-6)
    x = np.clip((values - black) / span, 0, 1)
    b = min(max((background - black) / span, 1e-6), 0.5)
    m = b * (target - 1) / (2 * b * target - target - b)
    return ((m - 1) * x / ((2 * m - 1) * x - m)).astype(np.float32)


_RB_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
_G_KERNEL = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4


def _convolve3(image, kernel):
    padded = np.pad(image, 1, mode="reflect")
    height, width = image.shape
    out = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            if kernel[dy, dx]:
                out += kernel[dy, dx] * padded[dy:dy + height, dx:dx + width]
    return out


def _debayer(raw, header):
    """Bilinear demosaic of a Bayer frame into (h, w, 3) float32 RGB."""
    pattern = str(header.get("BAYERPAT") or "RGGB").strip().upper()
    if len(pattern) != 4 or set(pattern) - set("RGB"):
        pattern = "RGGB"
    try:
        x_shift = int(float(header.get("XBAYROFF") or 0)) % 2
        y_shift = int(float(header.get("YBAYROFF") or 0)) % 2
    except (TypeError, ValueError):
        x_shift = y_shift = 0
    # cell[(row, col)] is the color at row % 2, col % 2, after any pattern offset
    cell = {(r, c): pattern[2 * ((r + y_shift) % 2) + (c + x_shift) % 2] for r in range(2) for c in range(2)}
    image = raw.astype(np.float32)
    rgb = np.empty(raw.shape + (3,), np.float32)
    for channel, color in enumerate("RGB"):
        sparse = np.zeros_like(image)
        for (r, c), cell_color in cell.items():
            if cell_color == color:
                sparse[r::2, c::2] = image[r::2, c::2]
        rgb[..., channel] = _convolve3(sparse, _G_KERNEL if color == "G" else _RB_KERNEL)
    return rgb


def screen_stretched(path, target=0.25):
    """A frame at full resolution for close viewing, as uint8 (h, w) or (h, w, 3):
    debayered to color on a color camera, then auto-stretched like PixInsight's
    STF - each channel on its own, which also neutralizes a color cast."""
    data, header = _read_frame(path)
    white = _full_scale(data)
    if data.ndim == 2 and header.get("BAYERPAT"):
        image = _debayer(data, header)
    else:
        image = data.astype(np.float32)
    del data
    out = np.empty(image.shape, np.uint8)
    for channel in range(1 if image.ndim == 2 else image.shape[2]):
        values = image if image.ndim == 2 else image[..., channel]
        background, noise = _clipped_stats(values[::4, ::4].ravel())
        stretched = _auto_stretch(values, background, noise, white, target)
        stretched *= 255
        stretched += 0.5
        if image.ndim == 2:
            out[...] = stretched
        else:
            out[..., channel] = stretched
    return out


def inspection_images(path, crop_radius=32):
    """Display images for reviewing a frame: (overview, overview_scale, crops).
    overview is the whole frame, auto-stretched to 0-1, at 1/overview_scale of
    the sensor's resolution. crops are [(image, (x, y))] - the corners, edges
    and center in reading order, at full sensor resolution (the green pixels on
    a color camera) and stretched alike, with each one's center in overview pixels."""
    data, header = _read_frame(path)
    full_scale = _full_scale(data)
    img, scale, summed = _luminance(data, header)
    background, noise = _clipped_stats(img[::4, ::4].ravel())
    overview = _auto_stretch(img, background, noise, full_scale * summed)

    if data.ndim == 2:
        source, green_odd, white, to_overview = data, None, full_scale, scale
        if header.get("BAYERPAT"):
            green_odd = _green_odd(header)
    else:
        source, green_odd, white, to_overview = img, None, full_scale * summed, 1
    height, width = source.shape[:2]
    radius = max(4, min(crop_radius, (min(height, width) - 4) // 6))
    edge = radius + (0 if green_odd is None else 1)
    rows = (edge, height // 2, height - 1 - edge)
    cols = (edge, width // 2, width - 1 - edge)
    ys = np.array([y for y in rows for _x in cols])
    xs = np.array([x for _y in rows for x in cols])
    boxes = _sensor_boxes(source, ys, xs, radius, green_odd)
    crop_background, crop_noise = _clipped_stats(boxes[:, ::2, ::2].ravel())
    # Gentler than the overview, white at the brightest cores rather than at
    # saturation, so star shapes keep their gradation instead of bloating
    crop_white = min(white, max(float(np.percentile(boxes, 99.9)), crop_background + 50 * crop_noise))
    stretched = _auto_stretch(boxes, crop_background, crop_noise, crop_white, target=0.12)
    crops = [(stretched[i], (xs[i] / to_overview, ys[i] / to_overview)) for i in range(len(ys))]
    return overview, scale, crops


def _arcsec_per_pixel(header):
    try:
        pixel_um, focal_mm = float(header.get("XPIXSZ")), float(header.get("FOCALLEN"))
    except (TypeError, ValueError):
        return None
    if pixel_um <= 0 or focal_mm <= 0:
        return None
    return 206.265 * pixel_um / focal_mm  # XPIXSZ already includes any camera binning


# ---- Grading ----------------------------------------------------------------

GRADE_GOOD, GRADE_MARGINAL, GRADE_REJECT = "Good", "Marginal", "Reject"
GRADE_UNGRADED, GRADE_ERROR = "Not graded", "Error"
MIN_GROUP_SIZE = 3  # fewer frames than this can't be compared with each other


@dataclass
class GradeSettings:
    """How far a frame may fall behind its group's median before it's a Reject.
    Half of each is the Marginal line."""
    flux_drop_pct: float = 40.0        # dimmer stars (clouds, haze, dew)
    star_drop_pct: float = 60.0        # fewer stars (obstruction, heavy cloud) - loose, as moonlight hides faint stars too
    fwhm_rise_pct: float = 25.0        # bigger stars (seeing, focus)
    eccentricity_rise: float = 0.12    # more elongated stars (trailing, wind, guiding) - absolute
    outside_rise_pct: float = 35.0     # more light outside the star cores (guiding jumps, tails, halos)
    background_rise_pct: float = 60.0  # brighter sky (moon, dawn, clouds lit from below) - Marginal at most on its own
    signal_drop_pct: float = 50.0      # less signal (frame_signal) - what a bright sky or haze costs the stack (0 = never)
    trail_count: float = 3.0           # satellite/plane trails in one frame that make it Marginal (0 = never) - a count, not vs the median


@dataclass
class FrameGrade:
    grade: str
    score: float = 0.0                 # 0-100, the group's median frame is about 85
    flags: list = field(default_factory=list)
    badness: dict = field(default_factory=dict)  # per metric, 1.0 = at its Reject line
    relative: dict = field(default_factory=dict)  # per metric, % of the group median
    medians: dict = field(default_factory=dict)   # the group medians graded against (shared by the group)
    outside: float = None              # share of star light beyond the core radius outside_at uses (0-1)


# Total badness at which several smaller problems add up to a Marginal/Reject.
_MARGINAL_TOTAL = 0.6
_REJECT_TOTAL = 1.25
_BEST_CREDIT = -0.5  # how much one better-than-median metric can offset others
_BACKGROUND_CAP = 0.6  # most a bright sky adds toward Marginal - it never counts toward Reject

# The score ranks frames by what they'd add to a stack (grades don't use it):
# each measurement's badness (1.0 = at its Reject line) times its weight.
# Sharpness and signal lead, and only they earn credit for beating the median -
# otherwise a soft frame could top the ranking on side measurements. Signal
# (frame_signal) is the same stars' flux against the noise, which ranks frames
# almost exactly like PixInsight's PSF Signal Weight; it also carries the sky
# (moonlight lowers it), so the background isn't counted again here.
SCORE_WEIGHTS = {"fwhm": 1.5, "signal": 1.5, "flux": 0.5, "eccentricity": 0.75, "outside": 0.75, "stars": 0.25}
# Most credit each one can earn. Signal's is the largest: a dark-sky frame with
# 160% of the median SNR adds far more to a stack than one at 125%, and with
# the same small cap as sharpness both scored alike and moonlit frames with
# slightly sharper stars caught up with them.
SCORE_CREDIT = {"fwhm": _BEST_CREDIT, "signal": -1.5}
SIGNAL_DROP_PCT = 40.0  # signal this far below the group median counts as badness 1.0


def frame_signal(metrics, band):
    """A frame's signal: the flux of the stars in the group's transparency band
    over the background noise. The same stars in every frame - star_snr (the
    measured stars' peaks) is only the fallback when the group has no band:
    in a bright sky fewer stars clear its brightness cut, and the brighter ones
    left made moonlit and dawn frames read about 80% of the median where
    PixInsight's weights put them nearer 45%."""
    if band is None:
        return metrics.star_snr
    flux = band_flux(metrics, band)
    return flux / metrics.noise if flux is not None and metrics.noise else None


def _score(total):
    return float(100 / (1 + math.exp(2.2 * (total - 0.75))))


def _score_total(badness):
    """Weighted badness for the score - credit only where SCORE_CREDIT allows."""
    total = 0.0
    for metric, weight in SCORE_WEIGHTS.items():
        if metric in badness:
            total += weight * min(max(badness[metric], SCORE_CREDIT.get(metric, 0.0)), 2.0)
    return total


def grade_frames(frames, settings=None):
    """Grade frames against their group. frames is an iterable of
    (group_key, FrameMetrics); returns {path: FrameGrade}."""
    settings = settings or GradeSettings()
    groups = {}
    for key, metrics in frames:
        groups.setdefault(key, []).append(metrics)

    grades = {}
    for members in groups.values():
        usable = [m for m in members if m.ok]
        for m in members:
            if not m.ok:
                grades[m.path] = FrameGrade(GRADE_ERROR, 0.0, [f"Could not analyze: {m.error}"])
        if len(usable) < MIN_GROUP_SIZE:
            for m in usable:
                grades[m.path] = FrameGrade(GRADE_UNGRADED, 0.0, ["Too few frames in this group to compare"])
            continue
        band = transparency_band(usable)
        grades.update(_grade_group(usable, settings, band))
    return grades


def _grade_group(frames, settings, band):
    """Grade against the median of the frames that aren't rejected. With bad
    frames in the median it sits lower, so removing them would make it rise and
    reject frames that passed before - each removal peeling off another layer.
    Re-grading against the survivors until nothing changes gives that end
    result at once, so removing the rejects leaves the rest graded the same."""
    reference = frames
    for _ in range(10):
        medians = _group_medians(reference, band)
        grades = {m.path: _grade_one(m, medians, settings, band) for m in frames}
        survivors = [m for m in frames if grades[m.path].grade != GRADE_REJECT]
        # A better reference mostly just adds rejects, so this settles in a few rounds
        if len(survivors) == len(reference) or len(survivors) < MIN_GROUP_SIZE:
            break
        reference = survivors
    return grades


def transparency_band(frames):
    """Index of the TRANSPARENCY_BANDS band to compare these frames' star flux
    in - the brightest one no frame has saturated stars in - or None."""
    most_saturated = max((m.saturated for m in frames), default=0)
    for index, (start, _stop) in enumerate(TRANSPARENCY_BANDS):
        if start >= most_saturated:
            measured = sum(1 for m in frames if band_flux(m, index) is not None)
            # Deeper bands only have fewer stars, so stop at the first clear one
            return index if measured >= len(frames) / 2 else None
    return None


def band_flux(metrics, band):
    """A frame's star flux in the given band, or None."""
    if band is None or not metrics.star_flux or band >= len(metrics.star_flux):
        return None
    return metrics.star_flux[band]


def _median_of(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _group_medians(frames, band):
    medians = {attr: _median_of(getattr(f, attr) for f in frames)
               for attr in ("stars", "fwhm_px", "eccentricity", "background")}
    medians["star_flux"] = _median_of(band_flux(f, band) for f in frames)
    medians["signal"] = _median_of(frame_signal(f, band) for f in frames)
    medians["outside_light"] = _median_of(outside_at(f, medians["fwhm_px"]) for f in frames)
    return medians


def _grade_one(m, med, s, band):
    badness, flags = {}, []

    flux = band_flux(m, band)
    if flux is not None and med["star_flux"]:
        drop = (med["star_flux"] - flux) / med["star_flux"] * 100
        badness["flux"] = drop / s.flux_drop_pct
        if badness["flux"] >= 0.5:
            flags.append(f"Dimmer stars (-{drop:.0f}%) - clouds or haze?")

    if med["stars"]:
        drop = (med["stars"] - m.stars) / med["stars"] * 100
        badness["stars"] = drop / s.star_drop_pct
        if badness["stars"] >= 0.5:
            flags.append(f"Fewer stars ({m.stars} vs {med['stars']:.0f}) - obstruction or cloud?")

    if m.fwhm_px is None:
        badness["fwhm"] = 1.0
        flags.append("No measurable stars")
    elif med["fwhm_px"]:
        rise = (m.fwhm_px - med["fwhm_px"]) / med["fwhm_px"] * 100
        badness["fwhm"] = rise / s.fwhm_rise_pct
        if badness["fwhm"] >= 0.5:
            flags.append(f"Soft stars (FWHM {m.fwhm_px:.2f} vs {med['fwhm_px']:.2f} px) - seeing or focus?")

    if m.eccentricity is not None and med["eccentricity"] is not None:
        badness["eccentricity"] = (m.eccentricity - med["eccentricity"]) / s.eccentricity_rise
        if badness["eccentricity"] >= 0.5:
            cause = "trailing" if (m.alignment or 0) >= 0.6 else "wind, guiding or tilt"
            flags.append(f"Elongated stars (eccentricity {m.eccentricity:.2f} vs "
                         f"{med['eccentricity']:.2f}) - {cause}?")

    outside = outside_at(m, med["fwhm_px"])
    if outside is not None and med["outside_light"]:
        rise = (outside - med["outside_light"]) / med["outside_light"] * 100
        # Never credit: a fainter halo than the median isn't worth offsetting
        # other problems, and the score's credit is for sharpness and signal
        badness["outside"] = max(rise / s.outside_rise_pct, 0.0)
        if badness["outside"] >= 0.5:
            flags.append(f"Light spread outside the star cores (+{rise:.0f}%) - guiding jump, trailing or halos?")

    if med["background"]:
        rise = (m.background - med["background"]) / med["background"] * 100
        if rise / s.background_rise_pct >= 0.5:
            flags.append(f"Bright background (+{rise:.0f}%) - moon, dawn or clouds?")
        # Moonlit subs with good stars are often still worth stacking (the stacker
        # weights them down), so a bright sky alone tops out at Marginal
        badness["background"] = min(rise / s.background_rise_pct, _BACKGROUND_CAP)

    # A bright sky alone stops at Marginal, but the signal it costs can still sink
    # a frame - PixInsight's WBPP drops frames this weak by its own weights.
    # Never credit: a dark-sky frame's extra signal shouldn't excuse soft stars.
    signal = frame_signal(m, band)
    if s.signal_drop_pct and signal is not None and med["signal"]:
        drop = (med["signal"] - signal) / med["signal"] * 100
        badness["signal"] = max(drop / s.signal_drop_pct, 0.0)
        if badness["signal"] >= 0.5:
            flags.append(f"Low signal ({100 - drop:.0f}% of the median) - moonlight, dawn, haze or clouds?")

    # Stacking's pixel rejection removes a trail or two, so they only say so -
    # unless there are enough to risk some surviving where they cross
    if m.trails:
        count = len(m.trails)
        many = bool(s.trail_count) and count >= s.trail_count
        flags.append(f"{count} satellite or plane trail{'s' if count > 1 else ''} - "
                     + ("enough that some may survive stacking" if many else "pixel rejection in stacking removes these"))
        if many:
            badness["trails"] = 0.5

    # The background and trails never count toward a Reject, and star count and
    # signal only on their own - a brighter sky hides faint stars and lowers the
    # signal too, so they aren't separate problems
    worst = max(badness.values(), default=0.0)
    total = sum(min(max(b, _BEST_CREDIT), 2.0) for b in badness.values())
    reject_worst = max((b for metric, b in badness.items() if metric not in ("background", "trails")), default=0.0)
    reject_total = sum(min(max(b, _BEST_CREDIT), 2.0) for metric, b in badness.items()
                       if metric not in ("background", "stars", "signal", "trails"))
    if reject_worst >= 1.0 or reject_total >= _REJECT_TOTAL:
        grade = GRADE_REJECT
    elif worst >= 0.5 or total >= _MARGINAL_TOTAL:
        grade = GRADE_MARGINAL
    else:
        grade = GRADE_GOOD
    # Graded down by the total rather than any one problem - say so, since the
    # flags alone (a bright sky, say) wouldn't explain it
    if (grade == GRADE_REJECT and reject_worst < 1.0) or (grade == GRADE_MARGINAL and worst < 0.5):
        flags.append("Several metrics slightly worse than the rest")

    relative = {}
    for name, value, median in (("flux", flux, med["star_flux"]), ("stars", m.stars, med["stars"]),
                                ("fwhm", m.fwhm_px, med["fwhm_px"]), ("background", m.background, med["background"]),
                                ("outside", outside, med["outside_light"]),
                                ("signal", signal, med["signal"])):
        if value is not None and median:
            relative[name] = value / median * 100

    # The score's own measurement - signal doesn't grade (a moonlit frame with
    # good stars stays Good), it only ranks
    score_badness = dict(badness)
    if "signal" in relative:
        score_badness["signal"] = (100 - relative["signal"]) / SIGNAL_DROP_PCT
    elif med["signal"]:
        score_badness["signal"] = 1.0  # too few stars to measure where the others could
    return FrameGrade(grade, _score(_score_total(score_badness)), flags, badness, relative, med, outside)


# ---- Command line -----------------------------------------------------------

def _collect_paths(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for root, _dirs, files in os.walk(arg):
                paths.extend(os.path.join(root, f) for f in sorted(files)
                             if os.path.splitext(f)[1].lower() in SessionFileScanner.SUPPORTED_EXTENSIONS)
        elif os.path.isfile(arg):
            paths.append(arg)
    return paths


def _group_key(path):
    try:
        ext = os.path.splitext(path)[1].lower()
        header = (SessionFileScanner.extract_xisf_header(path) if ext in SessionFileScanner.XISF_EXTENSIONS
                  else SessionFileScanner.extract_fits_header(path))
    except Exception:
        return ("?", None, None)
    return (str(header.get("FILTER") or "No filter"), header.get("EXPTIME"), header.get("XBINNING") or 1)


def _fmt(value, spec):
    return format(value, spec) if value is not None else "-"


def main(argv):
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    parser = argparse.ArgumentParser(description="Measure and grade light frames.")
    parser.add_argument("paths", nargs="+", help="FITS/XISF files or folders")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", help="JSON file of measurements: reused when it exists (just re-grades), "
                                        "written after analyzing otherwise")
    args = parser.parse_args(argv)

    if args.cache and os.path.isfile(args.cache):
        import json
        with open(args.cache, encoding="utf-8") as f:
            saved = json.load(f)
        keys = [tuple(entry["key"]) for entry in saved]
        results = [FrameMetrics.from_dict(entry["metrics"]) for entry in saved]
        print(f"Re-grading {len(results)} frame(s) from {args.cache}\n")
    else:
        paths = _collect_paths(args.paths)
        if not paths:
            print("No FITS/XISF files found.")
            return 1
        print(f"Analyzing {len(paths)} frame(s) with {args.workers} worker(s)...")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            keys = list(pool.map(_group_key, paths))
            results = []
            for i, metrics in enumerate(pool.map(analyze_file, paths), 1):
                results.append(metrics)
                print(f"\r  {i}/{len(paths)}", end="", flush=True)
        elapsed = time.perf_counter() - started
        print(f"\rDone in {elapsed:.0f}s ({elapsed / len(paths):.2f}s per frame)\n")
        if args.cache:
            import json
            with open(args.cache, "w", encoding="utf-8") as f:
                json.dump([{"key": list(k), "metrics": m.to_dict()} for k, m in zip(keys, results)], f)

    grades = grade_frames(zip(keys, results))
    rows = sorted(zip(keys, results), key=lambda kr: (str(kr[0]), -grades[kr[1].path].score))
    current = None
    for key, m in rows:
        if key != current:
            current = key
            band = transparency_band([r for k, r in zip(keys, results) if k == key and r.ok])
            band_text = f"ranks {TRANSPARENCY_BANDS[band][0]}-{TRANSPARENCY_BANDS[band][1]}" if band is not None else "none"
            print(f"== {key[0]}  {key[1]}s  bin {key[2]}  (flux from star {band_text}) ==")
            print(f"{'Grade':<10} {'Score':>5} {'Stars':>6} {'FWHM':>5} {'arcs':>5} {'HFR':>5} "
                  f"{'Ecc':>5} {'Align':>5} {'Out%':>5} {'Bkgnd':>7} {'Grad%':>6} {'SNR':>6} {'Flux':>8}  File / flags")
        g = grades[m.path]
        print(f"{g.grade:<10} {g.score:>5.0f} {m.stars:>6} {_fmt(m.fwhm_px, '5.2f')} {_fmt(m.fwhm_arcsec, '5.2f')} "
              f"{_fmt(m.hfr_px, '5.2f')} {_fmt(m.eccentricity, '5.2f')} {_fmt(m.alignment, '5.2f')} "
              f"{_fmt(g.outside * 100 if g.outside is not None else None, '5.1f')} "
              f"{_fmt(m.background, '7.0f')} {_fmt(m.gradient_pct, '6.1f')} {_fmt(m.star_snr, '6.0f')} "
              f"{_fmt(band_flux(m, band), '8.0f')}  "
              f"{os.path.basename(m.path)}" + (f"  [{'; '.join(g.flags)}]" if g.flags else ""))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(sys.argv[1:]))
