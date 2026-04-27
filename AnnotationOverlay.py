#!/usr/bin/env python3
"""
Annotation Overlay Module for Cosmos Collection
Renders astronomical annotations (DSOs, stars, constellations, grid) on plate-solved images
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject, QThread
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics, QPolygonF, QPainterPath
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


class CatalogQueryWorker(QThread):
    """Background worker for querying star/DSO catalogs"""

    finished = Signal(list, list)  # stars, dsos
    progress = Signal(str)

    def __init__(self, wcs: WCSTransform, magnitude_limit: float = 8.0):
        super().__init__()
        self.wcs = wcs
        self.magnitude_limit = magnitude_limit

    def run(self):
        """Query catalogs for objects in field of view"""
        stars = []
        dsos = []

        try:
            min_ra, max_ra, min_dec, max_dec = self.wcs.get_field_of_view()
            center_ra = (min_ra + max_ra) / 2
            center_dec = (min_dec + max_dec) / 2
            radius = max(abs(max_ra - min_ra), abs(max_dec - min_dec)) / 2

            self.progress.emit("Querying star catalogs...")
            stars = self._query_stars(center_ra, center_dec, radius)

            self.progress.emit("Querying DSO catalogs...")
            dsos = self._query_dsos(center_ra, center_dec, radius)

        except Exception as e:
            logger.exception("Catalog query failed")
            self.progress.emit(f"Query failed: {str(e)}")

        self.finished.emit(stars, dsos)

    def _query_stars(self, ra: float, dec: float, radius: float) -> List[CelestialObject]:
        """Query bright stars from Simbad"""
        stars = []
        try:
            from astroquery.simbad import Simbad
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            # Query bright stars
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            simbad = Simbad()
            simbad.add_votable_fields('flux(V)', 'ids')

            logger.info(f"Querying SIMBAD for stars at RA={ra:.2f}, Dec={dec:.2f}, radius={radius:.2f}")
            result = simbad.query_region(coord, radius=radius * u.deg)

            if result is not None:
                logger.info(f"SIMBAD returned {len(result)} objects, columns: {result.colnames}")
                processed = 0
                skipped_mag = 0
                for row in result:
                    try:
                        # Check for magnitude in different column names (SIMBAD varies)
                        mag = None
                        if 'FLUX_V' in result.colnames:
                            mag = row['FLUX_V']
                        elif 'V' in result.colnames:
                            mag = row['V']

                        # Skip if no magnitude or too faint
                        if mag is None or (hasattr(mag, 'mask') and mag.mask) or mag > self.magnitude_limit:
                            skipped_mag += 1
                            continue

                        ra_obj = row['RA']
                        dec_obj = row['DEC']

                        # Parse coordinates
                        coord_obj = SkyCoord(ra_obj, dec_obj, unit=(u.hourangle, u.deg))

                        name = row['MAIN_ID']
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')

                        stars.append(CelestialObject(
                            name=name,
                            ra=coord_obj.ra.deg,
                            dec=coord_obj.dec.deg,
                            magnitude=float(mag) if mag else None,
                            obj_type='star'
                        ))
                        processed += 1
                    except Exception as row_err:
                        logger.debug(f"Failed to parse star row: {row_err}")
                        continue

                logger.info(f"SIMBAD query: {processed} stars processed, {skipped_mag} skipped (magnitude > {self.magnitude_limit})")
            else:
                logger.info("SIMBAD query returned None")

        except ImportError:
            logger.warning("astroquery not available for star queries")
        except Exception as e:
            logger.warning(f"Star query failed: {e}")

        return stars[:100]  # Limit to brightest 100

    def _query_dsos(self, ra: float, dec: float, radius: float) -> List[CelestialObject]:
        """Query DSOs - use local database first, then online"""
        dsos = []

        # Try local database - create a new connection for this thread
        try:
            import sqlite3
            from ResourceManager import ResourceManager, attach_update_catalogs
            db_path = ResourceManager.get_database_path()

            # Create a new connection for this thread (SQLite connections can't be shared)
            conn = sqlite3.connect(str(db_path))
            attach_update_catalogs(conn)
            try:
                cursor = conn.cursor()

                # Query DSOs near the coordinates
                # Use subquery to get the best catalogue name (M > NGC > IC > others)
                cursor.execute("""
                    SELECT d.ra, d.dec, d.magnitude, d.dsotype,
                           COALESCE(d.sizemax, d.sizemin) / 60.0 as size,
                           (SELECT c.catalogue || ' ' || c.designation
                            FROM cataloguenr c
                            WHERE c.dsodetailid = d.id
                            ORDER BY CASE c.catalogue
                                WHEN 'M' THEN 1
                                WHEN 'NGC' THEN 2
                                WHEN 'IC' THEN 3
                                ELSE 4
                            END
                            LIMIT 1) as name
                    FROM dsodetail d
                    WHERE d.ra BETWEEN ? AND ?
                      AND d.dec BETWEEN ? AND ?
                """, (ra - radius, ra + radius, dec - radius, dec + radius))

                for row in cursor.fetchall():
                    if row[0] is not None and row[1] is not None:
                        dsos.append(CelestialObject(
                            name=row[5] or 'Unknown',
                            ra=row[0],
                            dec=row[1],
                            magnitude=row[2],
                            obj_type=row[3],
                            size=row[4]
                        ))

                logger.info(f"Local DSO query found {len(dsos)} objects near RA={ra:.2f}, Dec={dec:.2f}")
            finally:
                conn.close()

        except Exception as e:
            logger.warning(f"Local DSO query failed: {e}")

        return dsos


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


class AnnotationRenderer:
    """Renders annotations on an image"""

    def __init__(self):
        # Annotation visibility settings
        self.show_dsos = True
        self.show_stars = True
        self.show_constellation_lines = True
        self.show_grid = True

        # Styling
        self.dso_color = QColor(255, 200, 50, 200)  # Yellow-orange
        self.star_color = QColor(200, 200, 255, 200)  # Light blue
        self.constellation_color = QColor(100, 150, 255, 100)  # Dim blue
        self.grid_color = QColor(100, 255, 100, 80)  # Dim green

        # Data
        self.wcs: Optional[WCSTransform] = None
        self.stars: List[CelestialObject] = []
        self.dsos: List[CelestialObject] = []

    def set_wcs(self, wcs_header: Dict[str, Any], image_width: int, image_height: int):
        """Set WCS transformation from plate solve result"""
        self.wcs = WCSTransform(wcs_header, image_width, image_height)
        logger.info(f"WCS set: CRPIX=({self.wcs.crpix1:.1f}, {self.wcs.crpix2:.1f}), "
                    f"CRVAL=({self.wcs.crval1:.4f}, {self.wcs.crval2:.4f}), "
                    f"CD=[{self.wcs.cd1_1:.6f}, {self.wcs.cd1_2:.6f}; "
                    f"{self.wcs.cd2_1:.6f}, {self.wcs.cd2_2:.6f}], "
                    f"image={image_width}x{image_height}")

    def set_objects(self, stars: List[CelestialObject], dsos: List[CelestialObject]):
        """Set objects to annotate"""
        self.stars = stars
        self.dsos = dsos

    def render(self, painter: QPainter, scale: float = 1.0, offset_x: float = 0, offset_y: float = 0):
        """
        Render all enabled annotations

        Args:
            painter: QPainter to draw on
            scale: Current zoom scale factor
            offset_x, offset_y: Image offset in display coordinates
        """
        if not self.wcs:
            logger.warning("render() called but no WCS transform available")
            return

        logger.debug(f"render() called: scale={scale}, offset=({offset_x}, {offset_y}), "
                     f"show_grid={self.show_grid}, show_stars={self.show_stars}, "
                     f"show_dsos={self.show_dsos}, stars={len(self.stars)}, dsos={len(self.dsos)}")

        painter.save()

        if self.show_grid:
            self._render_grid(painter, scale, offset_x, offset_y)

        if self.show_constellation_lines:
            self._render_constellations(painter, scale, offset_x, offset_y)

        if self.show_stars:
            self._render_stars(painter, scale, offset_x, offset_y)

        if self.show_dsos:
            self._render_dsos(painter, scale, offset_x, offset_y)

        painter.restore()

    def _to_display_coords(self, px: float, py: float, scale: float, offset_x: float, offset_y: float) -> Tuple[float, float]:
        """Convert pixel coords to display coords"""
        return px * scale + offset_x, py * scale + offset_y

    def _render_grid(self, painter: QPainter, scale: float, offset_x: float, offset_y: float):
        """Render coordinate grid"""
        pen = QPen(self.grid_color)
        pen.setWidth(1)
        painter.setPen(pen)

        font = QFont("Arial", 8)
        painter.setFont(font)

        # Get field bounds
        min_ra, max_ra, min_dec, max_dec = self.wcs.get_field_of_view()

        # Determine grid spacing based on field size
        field_size = max(abs(max_ra - min_ra), abs(max_dec - min_dec))

        if field_size > 20:
            grid_step = 5.0
        elif field_size > 5:
            grid_step = 1.0
        elif field_size > 1:
            grid_step = 0.5
        else:
            grid_step = 0.1

        # Draw RA lines
        ra_start = math.floor(min_ra / grid_step) * grid_step
        for ra in self._frange(ra_start, max_ra + grid_step, grid_step):
            points = []
            for dec in self._frange(min_dec, max_dec, (max_dec - min_dec) / 50):
                px, py = self.wcs.sky_to_pixel(ra, dec)
                if px is not None:
                    dx, dy = self._to_display_coords(px, py, scale, offset_x, offset_y)
                    points.append(QPointF(dx, dy))

            if len(points) > 1:
                painter.drawPolyline(points)

        # Draw Dec lines
        dec_start = math.floor(min_dec / grid_step) * grid_step
        for dec in self._frange(dec_start, max_dec + grid_step, grid_step):
            points = []
            for ra in self._frange(min_ra, max_ra, (max_ra - min_ra) / 50):
                px, py = self.wcs.sky_to_pixel(ra, dec)
                if px is not None:
                    dx, dy = self._to_display_coords(px, py, scale, offset_x, offset_y)
                    points.append(QPointF(dx, dy))

            if len(points) > 1:
                painter.drawPolyline(points)

    def _render_constellations(self, painter: QPainter, scale: float, offset_x: float, offset_y: float):
        """Render constellation lines"""
        pen = QPen(self.constellation_color)
        pen.setWidth(2)
        painter.setPen(pen)

        for const, ra1, dec1, ra2, dec2 in CONSTELLATION_LINES:
            px1, py1 = self.wcs.sky_to_pixel(ra1, dec1)
            px2, py2 = self.wcs.sky_to_pixel(ra2, dec2)

            if px1 is not None and px2 is not None:
                dx1, dy1 = self._to_display_coords(px1, py1, scale, offset_x, offset_y)
                dx2, dy2 = self._to_display_coords(px2, py2, scale, offset_x, offset_y)

                # Only draw if at least partially in view
                if (0 <= dx1 <= painter.device().width() or 0 <= dx2 <= painter.device().width()) and \
                   (0 <= dy1 <= painter.device().height() or 0 <= dy2 <= painter.device().height()):
                    painter.drawLine(QPointF(dx1, dy1), QPointF(dx2, dy2))

    def _render_stars(self, painter: QPainter, scale: float, offset_x: float, offset_y: float):
        """Render star labels"""
        pen = QPen(self.star_color)
        painter.setPen(pen)

        font = QFont("Arial", 9)
        painter.setFont(font)

        for star in self.stars:
            px, py = self.wcs.sky_to_pixel(star.ra, star.dec)
            if px is None:
                continue

            dx, dy = self._to_display_coords(px, py, scale, offset_x, offset_y)

            # Check if in view
            if 0 <= dx <= painter.device().width() and 0 <= dy <= painter.device().height():
                # Draw small circle for star position
                radius = max(2, min(6, 8 - (star.magnitude or 5))) * scale
                painter.drawEllipse(QPointF(dx, dy), radius, radius)

                # Draw label
                painter.drawText(int(dx + radius + 2), int(dy + 4), star.name)

    def _render_dsos(self, painter: QPainter, scale: float, offset_x: float, offset_y: float):
        """Render DSO labels and markers"""
        pen = QPen(self.dso_color)
        pen.setWidth(2)
        painter.setPen(pen)

        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)

        # Log first few DSO positions for debugging
        for i, dso in enumerate(self.dsos[:3]):
            px, py = self.wcs.sky_to_pixel(dso.ra, dso.dec)
            logger.debug(f"DSO '{dso.name}' RA={dso.ra:.4f} Dec={dso.dec:.4f} -> pixel ({px:.1f}, {py:.1f})")

        for dso in self.dsos:
            px, py = self.wcs.sky_to_pixel(dso.ra, dso.dec)
            if px is None:
                continue

            dx, dy = self._to_display_coords(px, py, scale, offset_x, offset_y)

            # Check if in view
            if 0 <= dx <= painter.device().width() and 0 <= dy <= painter.device().height():
                # Draw marker based on object type
                marker_size = 15 * scale

                if dso.obj_type and 'GALXY' in dso.obj_type:
                    # Ellipse for galaxies
                    painter.drawEllipse(QPointF(dx, dy), marker_size, marker_size * 0.6)
                elif dso.obj_type and ('NB' in dso.obj_type or 'NEBULA' in dso.obj_type.upper()):
                    # Square for nebulae
                    painter.drawRect(int(dx - marker_size/2), int(dy - marker_size/2),
                                    int(marker_size), int(marker_size))
                elif dso.obj_type and 'CL' in dso.obj_type:
                    # Dashed circle for clusters
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawEllipse(QPointF(dx, dy), marker_size, marker_size)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                else:
                    # Circle for others
                    painter.drawEllipse(QPointF(dx, dy), marker_size, marker_size)

                # Draw label
                painter.drawText(int(dx + marker_size + 3), int(dy + 5), dso.name)

    def _frange(self, start: float, stop: float, step: float):
        """Float range generator"""
        current = start
        while current < stop:
            yield current
            current += step
