"""
TelescopeAdvisor.py - Which of the user's telescopes suits an object best.

For an object with no telescope set ("Any"), the suggestion is the enabled
telescope (with one of its cameras, and a reducer/barlow if that frames it
better) whose frame the object fills best - about half of it - with the pixel
scale against typical seeing as a tie-breaker. When the object is too big for
every frame, the widest field comes first, with the mosaic it needs.

Also the shared field-of-view and pixel-scale maths (the FOV Simulator, frame
quality). No Qt widgets - used by the Target List, DSO Details, the FOV
Simulator and the telescope pickers.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# How much of the frame an object should fill (its larger share of either side)
IDEAL_FILL = 0.55
GOOD_FILL = (0.4, 0.7)
# Above this the object (with a margin) doesn't fit one frame: a mosaic
MOSAIC_FILL = 0.9
MOSAIC_MARGIN = 1.1   # 10% sky around the object
MOSAIC_OVERLAP = 0.15
# Pixel scale ("/px) that suits typical seeing
GOOD_SCALE = (0.7, 2.5)


# ---- Field of view and pixel scale -------------------------------------------

def camera_fov(sensor_width, sensor_height, focal_length, factor=1.0):
    """A camera's field of view (width, height in arcminutes) on a telescope of
    focal_length (mm) with a reducer/barlow factor; sensor size in mm."""
    effective = focal_length * (factor or 1.0)
    return (math.degrees(2 * math.atan(sensor_width / (2 * effective))) * 60,
            math.degrees(2 * math.atan(sensor_height / (2 * effective))) * 60)


def pixel_scale(pixel_size, focal_length):
    """Arcseconds per pixel for a pixel size (microns) and focal length (mm);
    None if either is unknown."""
    try:
        pixel_size, focal_length = float(pixel_size), float(focal_length)
    except (TypeError, ValueError):
        return None
    if pixel_size <= 0 or focal_length <= 0:
        return None
    return 206.265 * pixel_size / focal_length


# ---- The user's equipment ----------------------------------------------------

@dataclass(frozen=True)
class Rig:
    """One way to image with a telescope: a camera, maybe a reducer/barlow."""
    telescope_id: int
    telescope_name: str
    aperture: float
    focal_length: float
    camera_id: int
    camera_name: str
    corrector_id: int = None
    corrector_name: str = None
    factor: float = 1.0
    fov_width: float = 0.0   # Arcminutes
    fov_height: float = 0.0
    scale: float = None      # "/px, None if the camera's pixel size is unknown

    @property
    def label(self):
        parts = [self.telescope_name]
        if self.corrector_name:
            parts.append(self.corrector_name)
        parts.append(self.camera_name)
        return " + ".join(parts)


def load_rigs(conn):
    """Every enabled telescope's rigs: each linked camera (all enabled cameras
    if none is linked), bare and with each linked reducer/barlow that has a
    factor. Cameras without a sensor size are left out."""
    try:
        telescopes = conn.execute(
            "SELECT id, name, aperture, focal_length FROM usertelescopes "
            "WHERE is_active = 1 AND focal_length > 0").fetchall()
        equipment = conn.execute(
            "SELECT id, equipment_type, name, sensor_width, sensor_height, pixel_size, factor "
            "FROM userequipment WHERE COALESCE(is_active, 1) = 1").fetchall()
        links = conn.execute("SELECT telescope_id, equipment_id FROM telescope_equipment").fetchall()
    except Exception as e:
        logger.error(f"Error loading equipment for telescope suggestions: {e}")
        return []

    items = {row[0]: row for row in equipment}
    cameras = [row for row in equipment if row[1] == "camera" and (row[3] or 0) > 0 and (row[4] or 0) > 0]
    linked = {}
    for telescope_id, equipment_id in links:
        if equipment_id in items:
            linked.setdefault(telescope_id, []).append(items[equipment_id])

    rigs = []
    for telescope_id, name, aperture, focal_length in telescopes:
        gear = linked.get(telescope_id, [])
        own_cameras = [row for row in gear if row in cameras] or cameras
        correctors = [(None, None, 1.0)] + [(row[0], row[2], row[6]) for row in gear
                                            if row[1] in ("reducer", "barlow") and (row[6] or 0) > 0]
        for camera_id, _type, camera_name, width, height, pixel, _factor in own_cameras:
            for corrector_id, corrector_name, factor in correctors:
                fov_width, fov_height = camera_fov(width, height, focal_length, factor)
                rigs.append(Rig(telescope_id, name, aperture or 0.0, focal_length, camera_id, camera_name,
                                corrector_id, corrector_name, factor, fov_width, fov_height,
                                pixel_scale(pixel, focal_length * factor)))
    return rigs


# ---- An object's size --------------------------------------------------------

def object_size(data):
    """An object's (major, minor) size in arcminutes - a catalog object's
    size_max/size_min, else a Target List entry's size_info text. A missing
    minor axis counts as round. None if the size is unknown."""
    from cosmoscollection.planning.TargetRows import parse_size_info
    if not data:
        return None
    size_min, size_max = data.get("size_min"), data.get("size_max")
    if not (size_min or size_max) and data.get("size_info"):
        size_min, size_max = parse_size_info(data.get("size_info"))
    try:
        sizes = [float(size) for size in (size_min, size_max) if size]
    except (TypeError, ValueError):
        return None
    sizes = [size for size in sizes if size > 0]
    if not sizes:
        return None
    return max(sizes), min(sizes)


# ---- Ranking -----------------------------------------------------------------

@dataclass(frozen=True)
class Fit:
    """How an object fits a rig."""
    rig: Rig
    fill: float          # The object's larger share of the frame's sides (1 = edge to edge)
    mosaic: tuple = None  # (columns, rows) when it needs one
    score: float = 0.0   # Lower is better

    @property
    def panels(self):
        return self.mosaic[0] * self.mosaic[1] if self.mosaic else 1


def _fill(size, rig):
    """The object's larger share of the frame's sides, turned to suit it."""
    major, minor = size
    long_side, short_side = max(rig.fov_width, rig.fov_height), min(rig.fov_width, rig.fov_height)
    return max(major / long_side, minor / short_side)


def _mosaic(size, rig):
    """(columns, rows) to cover the object with a margin, the frame's long side along its major axis."""
    major, minor = size[0] * MOSAIC_MARGIN, size[1] * MOSAIC_MARGIN
    long_side, short_side = max(rig.fov_width, rig.fov_height), min(rig.fov_width, rig.fov_height)
    step = 1 - MOSAIC_OVERLAP

    def count(extent, side):
        return 1 if extent <= side else 1 + math.ceil((extent - side) / (side * step))
    return count(major, long_side), count(minor, short_side)


def _score(fill, rig):
    score = abs(math.log(fill / IDEAL_FILL))
    if GOOD_FILL[0] <= fill <= GOOD_FILL[1]:
        score *= 0.5  # Anything in the good range is close to as good
    if rig.scale is not None:
        low, high = GOOD_SCALE
        if rig.scale < low:
            score += 0.3 * math.log(low / rig.scale)
        elif rig.scale > high:
            score += 0.3 * math.log(rig.scale / high)
    return score


def rank(size, rigs):
    """How the object fits each telescope (its best rig), best first. Objects
    that fit a frame come before mosaics; mosaics with the fewest panels first."""
    if not size or not rigs:
        return []
    best = {}
    for rig in rigs:
        fill = _fill(size, rig)
        mosaic = _mosaic(size, rig) if fill > MOSAIC_FILL else None
        fit = Fit(rig, fill, mosaic, _score(min(fill, MOSAIC_FILL), rig))
        current = best.get(rig.telescope_id)
        if current is None or _order(fit) < _order(current):
            best[rig.telescope_id] = fit
    return sorted(best.values(), key=_order)


def _order(fit):
    return (fit.panels, fit.score, -fit.rig.focal_length * fit.rig.factor if fit.mosaic else 0)


def suggest(size, rigs):
    """The best fit for the object, or None (size unknown, or no usable equipment)."""
    fits = rank(size, rigs)
    return fits[0] if fits else None


def suggest_for(data, rigs):
    """suggest() for a catalog object or Target List entry (see object_size)."""
    return suggest(object_size(data), rigs)


def assigned_telescope(conn, data):
    """The telescope (id, name) set for an object on the Target List - from
    data's telescope_id, else its Target List entry (by name or position); None
    if it has none ("Any") or isn't on the list."""
    from cosmoscollection.planning.TargetRows import find_existing_target
    try:
        telescope_id = data.get("telescope_id")
        if telescope_id is None:
            if "telescope_id" in data and data.get("target_id") is not None:
                return None  # A Target List entry set to "Any"
            entry = find_existing_target(conn, data.get("name", ""), data.get("ra_deg"), data.get("dec_deg"))
            if not entry:
                return None
            row = conn.execute("SELECT telescope_id FROM usertargetlist WHERE id = ?", (entry["id"],)).fetchone()
            telescope_id = row[0] if row else None
        if telescope_id is None:
            return None
        row = conn.execute("SELECT name FROM usertelescopes WHERE id = ?", (telescope_id,)).fetchone()
        return (telescope_id, row[0]) if row else None
    except Exception as e:
        logger.error(f"Error finding the telescope set for {data.get('name')}: {e}")
        return None


# ---- Text --------------------------------------------------------------------

def fit_text(fit, name=True):
    """'ZenithStar 61 + ASI2600MC Pro: fills 55%, 2.15"/px' (or '... 2×1 mosaic')."""
    details = (f"{fit.mosaic[0]}×{fit.mosaic[1]} mosaic" if fit.mosaic
               else f"fills {min(fit.fill, 1.0):.0%}")
    if fit.rig.scale is not None:
        details += f", {fit.rig.scale:.2f}\"/px"
    return f"{fit.rig.label}: {details}" if name else details


def ranking_lines(fits):
    """Each telescope's fit, best first: ['1. Name + Camera: fills 55%, 2.15"/px', ...]."""
    return [f"{number}. {fit_text(fit)}" for number, fit in enumerate(fits, 1)]


def ranking_tooltip(fits):
    """The ranking for a tooltip."""
    if not fits:
        return ""
    return "Suggested telescope (how the object fits each one):\n" + "\n".join(ranking_lines(fits))
