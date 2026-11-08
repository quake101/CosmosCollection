"""
Plate solutions (WCS): reading them from .wcs files and FITS headers, writing
.wcs files, and converting between image pixels and sky coordinates.

No Qt (but for one lazy fallback: the size of a PNG/JPG/TIFF) - used by the
annotation overlay (AnnotationOverlay.py re-exports these names), the plate
solvers, mount detection and the command line.
"""

import math
import os
import re
from typing import Any, Dict, Tuple

FITS_EXTENSIONS = (".fits", ".fit", ".fts")


def _has_matrix(wcs_header: Dict[str, Any], prefix: str) -> bool:
    """Whether a solution has a CD or PC matrix (any of its CDi_j/PCi_j terms)."""
    return any(f'{prefix}{i}_{j}' in wcs_header for i in (1, 2) for j in (1, 2))


def _cd_matrix(wcs_header: Dict[str, Any]) -> Tuple[float, float, float, float]:
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
        self.cd1_1, self.cd1_2, self.cd2_1, self.cd2_2 = _cd_matrix(wcs_header)

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
    if _has_matrix(wcs_header, 'CD'):
        column = {'CD1_2': 0.0, 'CD2_2': 0.0}
    elif _has_matrix(wcs_header, 'PC'):
        column = {'PC1_2': 0.0, 'PC2_2': 1.0}  # Missing PC terms default to the identity
    else:
        column = {'CDELT2': 1.0}  # CDELT/CROTA2 form: CDELT2 only appears in the y column
    for key, default in column.items():
        flipped[key] = -float(wcs_header.get(key, default))
    return flipped


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
