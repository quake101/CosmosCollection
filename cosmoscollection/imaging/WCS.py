"""
Plate solutions (WCS): reading them from .wcs files and FITS headers, writing
.wcs files, and converting between image pixels and sky coordinates.

No Qt (but for one lazy fallback: the size of a PNG/JPG/TIFF) - used by the
annotation overlay, the image viewer, the plate solvers, mount detection, the
FOV Simulator and the command line.
"""

import math
import os
import re
from typing import Any, Dict, Optional, Tuple

from cosmoscollection.sessions.SessionFileScanner import FITS_EXTENSIONS, parse_fits_cards, read_fits_header_fast


# ---- Sky <-> tangent plane ---------------------------------------------------

def project(ra, dec, ra0, dec0):
    """Gnomonic (TAN) projection about ra0, dec0 - degrees in, degrees out (xi
    toward increasing RA, eta toward north). ra and dec may be numpy arrays.
    A single point 90 degrees or more from ra0, dec0 (behind the plane) gives
    (None, None); arrays aren't checked."""
    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
        m = math
    else:
        import numpy as m
    ra, dec = m.radians(ra), m.radians(dec)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    cos_c = m.sin(dec) * math.sin(dec0) + m.cos(dec) * math.cos(dec0) * m.cos(ra - ra0)
    if m is math and cos_c <= 0:
        return None, None
    xi = m.cos(dec) * m.sin(ra - ra0) / cos_c
    eta = (m.sin(dec) * math.cos(dec0) - m.cos(dec) * math.sin(dec0) * m.cos(ra - ra0)) / cos_c
    return m.degrees(xi), m.degrees(eta)


def deproject(xi, eta, ra0, dec0):
    """Inverse of project - degrees in, (ra, dec) degrees out. RA isn't wrapped
    to 0-360: it stays within 180 degrees of ra0, so the corners of a field
    across RA 0 keep a continuous range."""
    xi, eta = math.radians(xi), math.radians(eta)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    rho = math.hypot(xi, eta)
    if rho == 0:
        return math.degrees(ra0), math.degrees(dec0)
    c = math.atan(rho)
    dec = math.asin(math.cos(c) * math.sin(dec0) + eta * math.sin(c) * math.cos(dec0) / rho)
    ra = ra0 + math.atan2(xi * math.sin(c), rho * math.cos(dec0) * math.cos(c) - eta * math.sin(dec0) * math.sin(c))
    return math.degrees(ra), math.degrees(dec)


# ---- Solutions -----------------------------------------------------------------

def _has_matrix(wcs_header: Dict[str, Any], prefix: str) -> bool:
    """Whether a solution has a CD or PC matrix (any of its CDi_j/PCi_j terms)."""
    return any(f'{prefix}{i}_{j}' in wcs_header for i in (1, 2) for j in (1, 2))


def cd_matrix(wcs_header: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """A solution's pixel -> sky matrix (CD1_1, CD1_2, CD2_1, CD2_2), degrees
    per pixel. Solutions come in three forms: a CD matrix (ASTAP's .wcs files),
    CDELT scales with a PC rotation matrix (the FITS headers ASTAP and Siril
    write), or CDELT with a CROTA2 rotation angle (older software)."""
    if _has_matrix(wcs_header, 'CD'):
        return tuple(float(wcs_header.get(key, 0)) for key in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'))
    cdelt1 = float(wcs_header.get('CDELT1', -0.001))
    cdelt2 = float(wcs_header.get('CDELT2', 0.001))
    if _has_matrix(wcs_header, 'PC'):
        # CDELTn scales row n; missing PC terms default to the identity
        return (cdelt1 * float(wcs_header.get('PC1_1', 1)), cdelt1 * float(wcs_header.get('PC1_2', 0)),
                cdelt2 * float(wcs_header.get('PC2_1', 0)), cdelt2 * float(wcs_header.get('PC2_2', 1)))
    rotation = math.radians(float(wcs_header.get('CROTA2', 0)))
    return (cdelt1 * math.cos(rotation), -cdelt2 * math.sin(rotation),
            cdelt1 * math.sin(rotation), cdelt2 * math.cos(rotation))


def pixel_scale_arcsec(wcs_header: Dict[str, Any]) -> Optional[float]:
    """Image scale in arcsec/pixel from a solution's matrix, in any of its
    forms (see cd_matrix) - or None without one. Uses the determinant, which
    holds at any rotation: CD1_1 alone is near zero at 90 degrees."""
    if not wcs_header or not (_has_matrix(wcs_header, 'CD') or 'CDELT1' in wcs_header or 'CDELT2' in wcs_header):
        return None
    try:
        cd11, cd12, cd21, cd22 = cd_matrix(wcs_header)
    except (TypeError, ValueError):
        return None
    return math.sqrt(abs(cd11 * cd22 - cd12 * cd21)) * 3600


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
        self.crpix1 = float(wcs_header.get('CRPIX1', image_width / 2))
        self.crpix2 = float(wcs_header.get('CRPIX2', image_height / 2))

        # Reference coordinates (degrees)
        self.crval1 = float(wcs_header.get('CRVAL1', 0))  # RA
        self.crval2 = float(wcs_header.get('CRVAL2', 0))  # Dec

        # CD matrix (degrees per pixel)
        self.cd1_1, self.cd1_2, self.cd2_1, self.cd2_2 = cd_matrix(wcs_header)

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
        xi_deg, eta_deg = project(float(ra), float(dec), self.crval1, self.crval2)
        if xi_deg is None:
            return None, None  # Object behind projection plane

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
        return deproject(xi_deg, eta_deg, self.crval1, self.crval2)

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
    if _has_matrix(wcs_header, 'CD'):
        column = {'CD1_2': 0.0, 'CD2_2': 0.0}
    elif _has_matrix(wcs_header, 'PC'):
        column = {'PC1_2': 0.0, 'PC2_2': 1.0}  # Missing PC terms default to the identity
    else:
        column = {'CDELT2': 1.0}  # CDELT/CROTA2 form: CDELT2 only appears in the y column
    for key, default in column.items():
        flipped[key] = -float(wcs_header.get(key, default))
    return flipped


def _fits_cards(path):
    """A FITS file's primary header as {keyword: value} - {} if it isn't a FITS
    file. Header cards are 80-character text, read directly: astropy takes
    seconds to import just for this."""
    try:
        return read_fits_header_fast(path)
    except ValueError:
        return {}


def read_wcs_file(wcs_file):
    """The keywords of a .wcs file (written by ASTAP, or by a solve here) - {}
    if it can't be read."""
    try:
        with open(wcs_file, "r", errors="replace") as file:
            text = file.read()
    except OSError:
        return {}
    # One card per line, or 80-character cards run together
    lines = text.splitlines() if "\n" in text else [text[i:i + 80] for i in range(0, len(text), 80)]
    return parse_fits_cards(lines)


def read_wcs_header(path):
    """An image's plate solution as WCS keywords: its .wcs file (written by
    ASTAP, or by a solve here), else a FITS file's own header - or None."""
    wcs_file = os.path.splitext(path)[0] + ".wcs"
    header = {}
    if os.path.isfile(wcs_file):
        header = read_wcs_file(wcs_file)
    elif os.path.splitext(path)[1].lower() in FITS_EXTENSIONS and os.path.isfile(path):
        header = _fits_cards(path)
    return header if is_solved_header(header) else None


def is_solved_header(header):
    """Whether WCS keywords are a plate solution."""
    return (bool(header) and "CRVAL1" in header and "CRVAL2" in header and "CRPIX1" in header
            and ("CD1_1" in header or "CDELT1" in header))


def image_pixel_size(path):
    """(width, height) of an image file, read from its header - not decoded."""
    ext = os.path.splitext(path)[1].lower()
    if ext in FITS_EXTENSIONS:
        cards = _fits_cards(path)
        return int(cards.get("NAXIS1", 0)), int(cards.get("NAXIS2", 0))
    if ext == ".xisf":
        from cosmoscollection.sessions.SessionFileScanner import read_xisf_xml
        _root, image, _ns = read_xisf_xml(path)
        width, height = (image.get("geometry", "0:0").split(":") + ["0"])[:2] if image is not None else (0, 0)
        return int(width), int(height)
    from PySide6.QtGui import QImageReader
    from cosmoscollection.imaging.ImageLoader import QT_IMAGE_IO_LOCK
    # Runs in worker threads: unlocked, it deadlocked the app against another
    # thread's read_image_file() (one holding the GIL inside Qt, the other
    # needing it back from Qt) when the Collage Builder opened
    with QT_IMAGE_IO_LOCK:
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
