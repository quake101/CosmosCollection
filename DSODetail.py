#!/usr/bin/env python3
"""
DSO Detail Window for Cosmos Collection
Provides detailed object information, image management, and visibility data
"""

import logging
import os
import sys
from typing import Optional, Dict

from PySide6.QtCore import Qt, Signal, QObject, QTimer, QEvent, QThread, QUrl
from PySide6.QtGui import QPixmap, QPainter, QAction
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QLineEdit,
    QComboBox, QTextEdit, QGroupBox, QPushButton, QGridLayout,
    QToolBar, QMessageBox, QMenu, QFileDialog
)

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionManager
from Theme import COLORS
from NINAIntegration import NINAIntegration
from UrlOpener import open_url

# Set up logging
logger = logging.getLogger(__name__)

# Check for optional DSO Visibility Calculator availability
try:
    import DSOVisibilityCalculator
    VISIBILITY_AVAILABLE = True
except ImportError:
    VISIBILITY_AVAILABLE = False
    logger.warning("DSOVisibilityCalculator.py not found. Visibility calculator will be disabled.")


class SimbadQueryThread(QThread):
    """Background thread for querying SIMBAD without blocking the UI"""
    query_complete = Signal(object, str, float, float)
    query_failed = Signal(str, str)

    def __init__(self, object_name, ra_deg, dec_deg, dso_type):
        super().__init__()
        self.object_name = object_name
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.dso_type = dso_type

    def run(self):
        """Query SIMBAD in background thread"""
        try:
            from astroquery.simbad import Simbad
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            if getattr(sys, 'frozen', False):
                try:
                    from astroquery.query import BaseQuery
                    BaseQuery.TIMEOUT = 30
                    import requests
                    original_request = requests.Session.request
                    def patched_request(self, *args, **kwargs):
                        kwargs['verify'] = False
                        return original_request(self, *args, **kwargs)
                    requests.Session.request = patched_request
                    logger.info("Disabled SSL verification for astroquery (PyInstaller workaround)")
                except Exception as ssl_config_error:
                    logger.warning(f"Could not configure SSL for astroquery: {ssl_config_error}")

            logger.debug(f"Querying SIMBAD for emission data: {self.object_name} at RA={self.ra_deg}, Dec={self.dec_deg}")

            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('U', 'B', 'V', 'R', 'I', 'sp', 'otype')

            result = None
            if self.ra_deg is not None and self.dec_deg is not None:
                coords = SkyCoord(ra=self.ra_deg*u.degree, dec=self.dec_deg*u.degree, frame='icrs')
                result = custom_simbad.query_region(coords, radius=3*u.arcmin)
                if result is not None and len(result) > 0:
                    logger.debug(f"SIMBAD data found for {self.object_name} at coordinates")
                else:
                    logger.debug(f"No SIMBAD data found for {self.object_name} at coordinates")
            else:
                logger.debug(f"No coordinates available for {self.object_name}, skipping SIMBAD query")

            self.query_complete.emit(result, self.object_name, self.ra_deg, self.dec_deg)

        except Exception as e:
            logger.error(f"SIMBAD query failed for {self.object_name}: {type(e).__name__}: {str(e)}", exc_info=True)
            self.query_failed.emit(self.object_name, str(e))


class ImageLoaderThread(QThread):
    """Background thread for loading large images without blocking the UI"""
    image_loaded = Signal(object, str)
    load_failed = Signal(str, str)

    def __init__(self, image_path, is_fits=False, fits_colormap='gray'):
        super().__init__()
        self.image_path = image_path
        self.is_fits = is_fits
        self.fits_colormap = fits_colormap

    def run(self):
        """Load image in background thread with progressive loading"""
        try:
            from PySide6.QtGui import QImage, QImageReader

            if not os.path.exists(self.image_path):
                self.load_failed.emit(self.image_path, "File not found")
                return

            file_ext = os.path.splitext(self.image_path)[1].lower()
            file_size_mb = os.path.getsize(self.image_path) / (1024 * 1024)

            logger.info(f"Image loading: {os.path.basename(self.image_path)} - {file_size_mb:.1f}MB - Format: {file_ext}")

            if file_ext == '.png' and file_size_mb > 2:
                logger.info(f"Using Pillow loader for PNG file ({file_size_mb:.1f}MB)")
                try:
                    success = self._load_with_pillow()
                    if success:
                        logger.info("Pillow load completed successfully")
                        return
                    else:
                        logger.warning("Pillow load failed, falling back to Qt")
                except Exception as e:
                    logger.warning(f"Pillow not available or failed: {e}, using Qt")

            if self.is_fits:
                qimage = self._load_fits_image(self.image_path, self.fits_colormap)
                if qimage and not qimage.isNull():
                    self.image_loaded.emit(qimage, self.image_path)
                else:
                    self.load_failed.emit(self.image_path, "Failed to load FITS file")
            else:
                QImageReader.setAllocationLimit(1024)

                reader = QImageReader(self.image_path)
                reader.setAutoTransform(True)
                reader.setQuality(100)

                if not reader.canRead():
                    self.load_failed.emit(self.image_path, f"Cannot read image: {reader.errorString()}")
                    return

                original_size = reader.size()
                logger.debug(f"Loading full resolution with Qt: {original_size.width()}x{original_size.height()}")

                qimage = reader.read()

                if qimage.isNull():
                    error_msg = reader.errorString() or "Unknown error"
                    self.load_failed.emit(self.image_path, f"Failed to load image: {error_msg}")
                else:
                    self.image_loaded.emit(qimage, self.image_path)

        except Exception as e:
            logger.error(f"Error loading image in thread: {str(e)}")
            self.load_failed.emit(self.image_path, str(e))

    def _load_with_pillow(self):
        """Load image using Pillow (much faster for PNG)"""
        try:
            from PIL import Image
            from PySide6.QtGui import QImage

            logger.debug(f"Loading PNG with Pillow: {self.image_path}")

            with Image.open(self.image_path) as pil_image:
                original_width, original_height = pil_image.size
                logger.info(f"Pillow opened image: {original_width}x{original_height}, mode: {pil_image.mode}")

                full_image = pil_image
                logger.info(f"Converting full resolution to QImage: {original_width}x{original_height}")

                if full_image.mode != 'RGB':
                    logger.info(f"Converting {full_image.mode} to RGB for consistent color handling")
                    full_image = full_image.convert('RGB')

                img_data = full_image.tobytes()
                qimage = QImage(
                    img_data,
                    full_image.size[0],
                    full_image.size[1],
                    full_image.size[0] * 3,
                    QImage.Format_RGB888
                ).copy()

                if not qimage.isNull():
                    logger.info(f"Emitting full resolution image: {full_image.size[0]}x{full_image.size[1]}")
                    self.image_loaded.emit(qimage, self.image_path)
                    logger.info("Pillow load completed successfully")
                    return True
                else:
                    return False

        except ImportError:
            logger.debug("Pillow not installed")
            return False
        except Exception as e:
            logger.error(f"Error loading with Pillow: {str(e)}")
            return False

    def _load_fits_image(self, fits_path, colormap='viridis'):
        """Load a FITS image file and convert to QImage"""
        try:
            from astropy.io import fits
            from astropy.visualization import simple_norm
            import numpy as np
            from PySide6.QtGui import QImage

            with fits.open(fits_path) as hdul:
                data = hdul[0].data

                if data is None:
                    return None

                if len(data.shape) == 3:
                    if data.shape[0] == 3:
                        data = np.transpose(data, (1, 2, 0))
                    elif data.shape[0] <= 10:
                        data = data[0]

                is_rgb = len(data.shape) == 3 and data.shape[2] == 3

                if is_rgb:
                    norm = simple_norm(data, 'linear', percent=99.5)
                    normalized_data = norm(data)
                    rgb_data = (normalized_data * 255).astype(np.uint8)
                else:
                    try:
                        import matplotlib.pyplot as plt
                        norm = simple_norm(data, 'linear', percent=99.5)
                        normalized_data = norm(data)
                        cmap = plt.get_cmap(colormap)
                        rgba_data = cmap(normalized_data)
                        rgb_data = (rgba_data[:, :, :3] * 255).astype(np.uint8)
                    except ImportError:
                        norm = simple_norm(data, 'linear', percent=99.5)
                        normalized_data = norm(data)
                        gray_data = (normalized_data * 255).astype(np.uint8)
                        rgb_data = np.stack([gray_data] * 3, axis=-1)

                rgb_data = np.flipud(rgb_data)
                rgb_data = np.ascontiguousarray(rgb_data)

                height, width = rgb_data.shape[:2]
                bytes_per_line = 3 * width
                qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(qimage)

                if not pixmap.isNull():
                    return pixmap.toImage()
                else:
                    return None

        except Exception as e:
            logger.error(f"Error loading FITS file {fits_path}: {str(e)}")
            return None


class ImageCache:
    _instance = None
    _cache: Dict[str, QPixmap] = {}
    _max_size = 10

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCache, cls).__new__(cls)
        return cls._instance

    def get(self, path: str) -> Optional[QPixmap]:
        if path in self._cache:
            return self._cache[path]
        return None

    def put(self, path: str, pixmap: QPixmap):
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = pixmap

    def clear(self):
        self._cache.clear()


class VisibilityCalculationWorker(QObject):
    """Worker for performing heavy visibility calculations in a background thread."""
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, lat: float, lon: float, ra_deg: float, dec_deg: float, object_name: str):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.object_name = object_name

        try:
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            self.calculator = DSOVisibilityCalculator(lat, lon)
        except ImportError:
            self.calculator = None

    def calculate_visibility(self):
        """Calculate visibility seasons in background thread"""
        try:
            if self.calculator is None:
                self.error.emit("Visibility calculator not available.")
                return

            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from datetime import datetime, timedelta
            import numpy as np

            dso_coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)

            current_year = datetime.now().year
            min_altitude = 30

            sample_dates = []
            visibility_results = []

            for day_offset in range(0, 365, 15):
                try:
                    test_date = datetime(current_year, 1, 1) + timedelta(days=day_offset)
                    date_str = test_date.strftime('%Y-%m-%d')

                    time_range, dso_altaz, sun_altaz = self.calculator.calculate_altaz_over_time(
                        dso_coord, date_str, 12)

                    optimal_times = self.calculator.find_optimal_viewing_times(
                        dso_altaz, sun_altaz, min_altitude)

                    results = {"optimal_times": optimal_times}

                    is_visible = False
                    if "error" not in results and np.any(results.get("optimal_times", [])):
                        is_visible = True

                    sample_dates.append(test_date)
                    visibility_results.append(is_visible)

                except Exception:
                    continue

            if any(visibility_results):
                season_strs = []
                in_season = False
                season_start = None

                for i, (date, visible) in enumerate(zip(sample_dates, visibility_results)):
                    if visible and not in_season:
                        season_start = date
                        in_season = True
                    elif not visible and in_season:
                        if season_start:
                            season_strs.append(f"{season_start.strftime('%B %d')} - {sample_dates[i-1].strftime('%B %d')}")
                        in_season = False
                    elif i == len(sample_dates) - 1 and in_season:
                        if season_start:
                            season_strs.append(f"{season_start.strftime('%B %d')} - {date.strftime('%B %d')}")

                if season_strs:
                    visibility_text = f"Best viewing seasons (>30° altitude in dark sky):<br>" + "<br>".join(season_strs)
                    visibility_text += "<br><br><small>Use Visibility Calculator for detailed nightly times.</small>"
                else:
                    visibility_text = "Object not optimally visible from your location this year."
            else:
                visibility_text = "Object not optimally visible from your location this year."

            self.finished.emit(visibility_text)
        except Exception as e:
            logger.error(f"Error calculating visibility: {str(e)}", exc_info=True)
            self.error.emit(f"Error calculating viewing season information:<br>{str(e)}")


class DSODetailWindow(QDialog):
    """
    Detail window for DSO objects with image support including FITS files.
    """
    image_added = Signal()
    FITS_COLORMAP = 'gray'

    def __init__(self, data: dict, parent=None):
        super().__init__(None)
        logger.debug(f"Creating DSODetailWindow for {data['name']}")
        self.setWindowTitle(f"{data['name']} - DSO Detail - Cosmos Collection")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.NonModal)
        self.resize(1000, 800)
        WindowPositionManager.restore_window_position(self, "ObjectDetail")

        self.data = data
        self.current_image_path = None
        self.zoom_factor = 1.0
        self.initial_zoom_factor = 1.0
        self.original_pixmap = None
        self.image_position = [0, 0]
        self.drag_start_position = None
        self.drag_start_image_position = None
        self.image_cache = ImageCache()
        self.db_manager = DatabaseManager()

        self.image_loader_thread = None
        self.simbad_query_thread = None

        self.user_images = []
        self.current_image_index = 0

        self.ra_str, self.dec_str = self._format_coordinates(data["ra_deg"], data["dec_deg"])

        logger.debug(f"About to set up UI for {data['name']}")
        self._setup_ui()

    def showEvent(self, event):
        """Handle window show event"""
        super().showEvent(event)
        QTimer.singleShot(200, self._defer_heavy_calculations)

    def _defer_heavy_calculations(self):
        """Perform heavy calculations after window is shown"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
            if row and row[0] is not None and row[1] is not None:
                QTimer.singleShot(500, self._set_season_label_from_location)
            else:
                self.season_label.setText("Set your location in Settings to see viewing season.")
        except Exception as e:
            logger.error(f"Error checking location in deferred calculations: {str(e)}")
            self.season_label.setText("Set your location in Settings to see viewing season.")

        QTimer.singleShot(300, self._load_user_images)
        QTimer.singleShot(400, self._query_emission_data)

    def _get_update_sources_html(self):
        """Return HTML listing update-database sources that contributed NAME designations to this DSO.

        Returns an empty string when no update databases are loaded or this DSO has no NAME entries.
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT d.id FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ? AND c.designation = ?
                """, (self.data['catalogue'], self.data['id']))
                row = cursor.fetchone()
                if not row:
                    return ""
                dsodetailid = str(row[0])

                cursor.execute("""
                    SELECT DISTINCT source FROM name_provenance_all
                    WHERE dsodetailid = ?
                """, (dsodetailid,))
                source_strings = [r[0] for r in cursor.fetchall()]
                if not source_strings:
                    return ""

                # Source strings can be combined, e.g. "OpenNGC+Wikidata"
                labels = set()
                for s in source_strings:
                    for part in s.split("+"):
                        labels.add(part.strip())

                placeholders = ",".join("?" * len(labels))
                cursor.execute(f"""
                    SELECT source, key, value FROM attribution_all
                    WHERE source IN ({placeholders}) AND key IN ('name', 'url', 'license')
                """, list(labels))
                attr_rows = cursor.fetchall()
        except Exception:
            return ""

        sources = {}
        for source, key, value in attr_rows:
            sources.setdefault(source, {})[key] = value

        if not sources:
            return ""

        parts = []
        for info in sorted(sources.values(), key=lambda x: x.get('name', '')):
            name = info.get('name', '')
            url = info.get('url', '')
            license_ = info.get('license', '')
            entry = f'<a href="{url}">{name}</a>' if url else name
            if license_:
                entry += f" ({license_})"
            parts.append(entry)

        return "<br><b>Name Sources:</b> " + " &middot; ".join(parts)

    def _format_coordinates(self, ra_deg, dec_deg):
        """Format RA and Dec coordinates efficiently"""
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_remaining = (ra_hours - ra_h) * 60
        ra_m = int(ra_remaining)
        ra_s = (ra_remaining - ra_m) * 60
        ra_str = f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s"

        dec_sign = '-' if dec_deg < 0 else '+'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_remaining = (dec_abs - dec_d) * 60
        dec_m = int(dec_remaining)
        dec_s = (dec_remaining - dec_m) * 60
        dec_str = f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""

        return ra_str, dec_str

    def _query_emission_data(self):
        """Query SIMBAD for emission line and spectral information"""
        try:
            dso_type = self.data.get('dso_type', '')
            if dso_type not in ['BRTNB', 'CL+NB', 'PLNNB', 'SNREM']:
                self.emission_label.setText("Not applicable (not an emission nebula)")
                self.emission_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                return

            object_name = self.data.get('name', '')
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')

            self.emission_label.setText("Querying SIMBAD...")
            self.emission_label.setStyleSheet(f"color: {COLORS['text_disabled']};")

            if self.simbad_query_thread and self.simbad_query_thread.isRunning():
                self.simbad_query_thread.quit()
                self.simbad_query_thread.wait()

            self.simbad_query_thread = SimbadQueryThread(object_name, ra_deg, dec_deg, dso_type)
            self.simbad_query_thread.query_complete.connect(self._on_simbad_query_complete)
            self.simbad_query_thread.query_failed.connect(self._on_simbad_query_failed)
            self.simbad_query_thread.start()

            logger.debug(f"Started background SIMBAD query for {object_name}")

        except Exception as e:
            logger.error(f"Error initiating SIMBAD query: {str(e)}", exc_info=True)
            self.emission_label.setText(f"Error querying SIMBAD (check internet connection)")
            self.emission_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_simbad_query_complete(self, result, object_name, ra_deg, dec_deg):
        """Handle SIMBAD query completion"""
        try:
            dso_type = self.data.get('dso_type', '')
            emission_info = self._parse_emission_info(dso_type, result)

            if emission_info:
                self.emission_label.setText(emission_info)
                self.emission_label.setStyleSheet("color: white;")
            else:
                self.emission_label.setText("No specific emission line data available")
                self.emission_label.setStyleSheet(f"color: {COLORS['text_disabled']};")

        except Exception as e:
            logger.error(f"Error processing SIMBAD result: {str(e)}", exc_info=True)
            self.emission_label.setText("Error processing SIMBAD data")
            self.emission_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_simbad_query_failed(self, object_name, error_message):
        """Handle SIMBAD query failure"""
        logger.warning(f"SIMBAD query failed for {object_name}: {error_message}")
        self.emission_label.setText(f"SIMBAD query failed (check internet connection)")
        self.emission_label.setStyleSheet(f"color: {COLORS['error']};")

    def _parse_emission_info(self, dso_type, simbad_result):
        """Parse emission line information based on DSO type and SIMBAD data"""
        emission_lines = {
            'BRTNB': {
                'primary': ['Hα (656.3 nm)', 'Hβ (486.1 nm)'],
                'secondary': ['OIII (495.9, 500.7 nm)', 'SII (671.6, 673.1 nm)', 'NII (658.3 nm)'],
                'description': 'Emission nebula - typically rich in hydrogen and oxygen'
            },
            'CL+NB': {
                'primary': ['Hα (656.3 nm)'],
                'secondary': ['OIII (495.9, 500.7 nm)', 'Hβ (486.1 nm)'],
                'description': 'Star cluster with associated emission nebulosity'
            },
            'PLNNB': {
                'primary': ['OIII (495.9, 500.7 nm)', 'Hα (656.3 nm)'],
                'secondary': ['Hβ (486.1 nm)', 'NII (658.3 nm)'],
                'description': 'Planetary nebula - strong in oxygen and hydrogen'
            },
            'SNREM': {
                'primary': ['OIII (495.9, 500.7 nm)', 'SII (671.6, 673.1 nm)'],
                'secondary': ['Hα (656.3 nm)', 'Hβ (486.1 nm)'],
                'description': 'Supernova remnant - often oxygen and sulfur rich'
            }
        }

        if dso_type not in emission_lines:
            return None

        info = emission_lines[dso_type]
        text = ""

        if simbad_result is not None and len(simbad_result) > 0:
            row = simbad_result[0]
            text += "<b>SIMBAD Data:</b>"

            if 'OTYPE' in row.colnames and row['OTYPE']:
                otype = str(row['OTYPE']).strip()
                text += f"<b>Object Type:</b> {otype}<br>"

            if 'SP_TYPE' in row.colnames and row['SP_TYPE']:
                sp_type = str(row['SP_TYPE']).strip()
                if sp_type and sp_type != '--':
                    text += f"<b>Spectral Type:</b> {sp_type}<br>"

            flux_data = []
            flux_bands = [
                (['U', 'FLUX_U', 'flux_U'], 'U'),
                (['B', 'FLUX_B', 'flux_B'], 'B'),
                (['V', 'FLUX_V', 'flux_V'], 'V'),
                (['R', 'FLUX_R', 'flux_R'], 'R'),
                (['I', 'FLUX_I', 'flux_I'], 'I')
            ]

            for flux_cols, band in flux_bands:
                flux_val = None
                for flux_col in flux_cols:
                    if flux_col in row.colnames and row[flux_col] is not None:
                        try:
                            flux_val = float(row[flux_col])
                            if not (flux_val == 0 or str(flux_val) == 'nan'):
                                break
                        except (ValueError, TypeError):
                            pass
                if flux_val is not None and not (flux_val == 0 or str(flux_val) == 'nan'):
                    flux_data.append(f"{band}={flux_val:.2f}")

            if flux_data:
                text += f"<b>Photometry:</b> {', '.join(flux_data)} mag<br>"

            text += "<br>"

        text += f"<b>{info['description']}</b><br><br>"
        text += "<b>Primary Emission Lines:</b><br>"
        for line in info['primary']:
            if 'Hα' in line or 'Hβ' in line:
                color = '#ff6b6b'
            elif 'OIII' in line:
                color = '#6bcdff'
            elif 'SII' in line:
                color = '#ff9966'
            elif 'NII' in line:
                color = '#ff8888'
            else:
                color = 'white'
            text += f"<span style='color: {color};'>• {line}</span><br>"

        text += "<br><b>Secondary Lines:</b><br>"
        for line in info['secondary']:
            if 'Hα' in line or 'Hβ' in line:
                color = '#ff6b6b'
            elif 'OIII' in line:
                color = '#6bcdff'
            elif 'SII' in line:
                color = '#ff9966'
            elif 'NII' in line:
                color = '#ff8888'
            else:
                color = 'white'
            text += f"<span style='color: {color};'>• {line}</span><br>"

        text += "<br><i>Useful for Hα, OIII, and SII narrowband imaging</i>"

        return text

    def _setup_ui(self):
        """Set up the UI components - called after window is shown"""
        from PySide6.QtWidgets import QToolButton, QMenu
        from PySide6.QtGui import QAction

        logger.debug(f"_setup_ui called for {self.data['name']}")
        try:
            # Create menu bar using QToolBar with proper menu buttons
            menubar = QToolBar()
            menubar.setMovable(False)
            menubar.setStyleSheet("QToolBar { border: 0px; spacing: 3px; }")

            # Add Visibility Calculator button (if available)
            if VISIBILITY_AVAILABLE:
                visibility_button = QToolButton()
                visibility_button.setText("Visibility Calculator")
                visibility_button.setToolTip("Open the Visibility Calculator to see when this object is visible")
                visibility_button.clicked.connect(self._open_visibility_calculator)
                menubar.addWidget(visibility_button)

            # Add Aladin Lite button
            aladin_button = QToolButton()
            aladin_button.setText("FOV Simulator")
            aladin_button.setToolTip("Open interactive sky atlas with telescope field of view simulator")
            aladin_button.clicked.connect(lambda: self._open_aladin_lite(self.data))
            menubar.addWidget(aladin_button)

            # Add Wikipedia button
            wikipedia_button = QToolButton()
            wikipedia_button.setText("Wikipedia")
            wikipedia_button.setToolTip("Open the Wikipedia page for this object in your browser")
            wikipedia_button.clicked.connect(self._open_wikipedia)
            menubar.addWidget(wikipedia_button)

            # Target List menu
            target_menu = QMenu("Target List", self)

            # Add to Target List action (will be hidden/shown based on state)
            self.add_target_action = QAction("Add to Target List", self)
            self.add_target_action.setToolTip("Add this object to your observing target list")
            self.add_target_action.triggered.connect(self._add_to_target_list)
            target_menu.addAction(self.add_target_action)

            # Remove from Target List action (initially hidden)
            self.remove_target_action = QAction("Remove from Target List", self)
            self.remove_target_action.setToolTip("Remove this object from your target list")
            self.remove_target_action.triggered.connect(self._remove_from_target_list)
            self.remove_target_action.setVisible(False)
            target_menu.addAction(self.remove_target_action)

            # Open from Target List action (initially hidden)
            self.open_target_action = QAction("Open from Target List", self)
            self.open_target_action.setToolTip("Open this DSO from your target list")
            self.open_target_action.triggered.connect(self._open_from_target_list)
            self.open_target_action.setVisible(False)
            target_menu.addAction(self.open_target_action)

            # Create Target List button with menu
            target_button = QToolButton()
            target_button.setText("Target List")
            target_button.setToolTip("Manage your observing target list")
            target_button.setMenu(target_menu)
            target_button.setPopupMode(QToolButton.InstantPopup)
            menubar.addWidget(target_button)

            # Add NINA button if integration is enabled
            if NINAIntegration.is_enabled():
                nina_menu = QMenu(self)

                framing_action = QAction("Send to Framing Assistant", self)
                framing_action.triggered.connect(self._send_to_nina)
                nina_menu.addAction(framing_action)

                slew_action = QAction("Slew to Target", self)
                slew_action.triggered.connect(self._slew_to_nina_target)
                nina_menu.addAction(slew_action)

                nina_button = QToolButton()
                nina_button.setText("NINA")
                nina_button.setToolTip("Send to NINA")
                nina_button.setMenu(nina_menu)
                nina_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
                menubar.addWidget(nina_button)

            # Create main vertical layout to hold menubar and content
            window_layout = QVBoxLayout()
            window_layout.setContentsMargins(0, 0, 0, 0)
            window_layout.setSpacing(0)
            window_layout.addWidget(menubar)

            # Create main horizontal layout for content
            main_layout = QHBoxLayout()

            # Left side - Image placeholder and information
            left_layout = QVBoxLayout()

            # Image area (larger)
            image_layout = QVBoxLayout()

            # Create a container for the image and zoom controls
            image_container = QWidget()
            image_container_layout = QVBoxLayout()

            # Add zoom controls
            zoom_layout = QHBoxLayout()
            zoom_out_button = QPushButton("-")
            zoom_out_button.setStyleSheet("QPushButton { font-size: 12pt; }")
            zoom_out_button.clicked.connect(self._zoom_out)
            zoom_layout.addWidget(zoom_out_button)

            zoom_in_button = QPushButton("+")
            zoom_in_button.setStyleSheet("QPushButton { font-size: 12pt; }")
            zoom_in_button.clicked.connect(self._zoom_in)
            zoom_layout.addWidget(zoom_in_button)

            reset_button = QPushButton("Reset")
            reset_button.setFixedSize(60, 30)
            reset_button.clicked.connect(self._reset_zoom)
            zoom_layout.addWidget(reset_button)

            # Add image navigation controls
            nav_separator = QLabel("|")
            nav_separator.setStyleSheet(f"font-size: 12pt; color: {COLORS['border_light']}; padding: 0 5px;")
            zoom_layout.addWidget(nav_separator)

            self.prev_image_button = QPushButton("<-")
            self.prev_image_button.setStyleSheet("QPushButton { font-size: 12pt; }")
            self.prev_image_button.clicked.connect(self._previous_image)
            self.prev_image_button.setToolTip("Previous image")
            zoom_layout.addWidget(self.prev_image_button)

            self.image_counter_label = QLabel("1/1")
            self.image_counter_label.setStyleSheet(f"font-size: 10pt; color: {COLORS['border_light']}; padding: 0 5px;")
            self.image_counter_label.setMinimumWidth(40)
            self.image_counter_label.setAlignment(Qt.AlignCenter)
            zoom_layout.addWidget(self.image_counter_label)

            self.next_image_button = QPushButton("->")
            self.next_image_button.setStyleSheet("QPushButton { font-size: 12pt; }")
            self.next_image_button.clicked.connect(self._next_image)
            self.next_image_button.setToolTip("Next image")
            zoom_layout.addWidget(self.next_image_button)

            # Add image button
            add_separator = QLabel("|")
            add_separator.setStyleSheet(f"font-size: 12pt; color: {COLORS['border_light']}; padding: 0 5px;")
            zoom_layout.addWidget(add_separator)

            self.add_image_button = QPushButton("+")
            self.add_image_button.clicked.connect(self._add_user_image)
            self.add_image_button.setToolTip("Add new image")
            self.add_image_button.setStyleSheet(f"QPushButton {{ color: {COLORS['success']}; font-size: 12pt; }}")
            zoom_layout.addWidget(self.add_image_button)

            # Delete image button
            self.delete_image_button = QPushButton("X")
            self.delete_image_button.clicked.connect(self._delete_current_image)
            self.delete_image_button.setToolTip("Delete current image")
            self.delete_image_button.setStyleSheet(f"QPushButton {{ color: {COLORS['error']}; font-size: 12pt; }}")
            zoom_layout.addWidget(self.delete_image_button)

            # Favorite image button
            self.favorite_button = QPushButton("*")
            self.favorite_button.clicked.connect(self._toggle_favorite)
            self.favorite_button.setToolTip("Mark as favorite")
            self.favorite_button.setStyleSheet("QPushButton { font-size: 12pt; }")
            zoom_layout.addWidget(self.favorite_button)

            zoom_layout.addStretch()
            image_container_layout.addLayout(zoom_layout)

            # Image label
            self.image_label = QLabel("Loading...")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")
            self.image_label.setMinimumSize(600, 400)  # Increased minimum size
            self.image_label.installEventFilter(self)  # Install event filter for mouse events
            self.image_label.setMouseTracking(True)  # Enable mouse tracking
            image_container_layout.addWidget(self.image_label)

            image_container.setLayout(image_container_layout)
            image_layout.addWidget(image_container)
            left_layout.addLayout(image_layout, stretch=2)  # Give image area more stretch

            # Create container for image information form
            self.info_form_container = QGroupBox("Image Information")
            self.info_form_container.setStyleSheet(
                "QGroupBox:title { subcontrol-position: top center; font-size: 16pt; font-weight: bold; }")
            info_form_layout = QGridLayout()
            info_form_layout.setSpacing(5)  # Reduce spacing between elements
            info_form_layout.setVerticalSpacing(5)
            info_form_layout.setHorizontalSpacing(10)

            # Row 0: Integration Time and Date (side by side)
            integration_label = QLabel("Integration:")
            self.integration_edit = QLineEdit()
            self.integration_edit.setPlaceholderText("e.g., 2h 30m")
            info_form_layout.addWidget(integration_label, 0, 0)
            info_form_layout.addWidget(self.integration_edit, 0, 1)

            date_label = QLabel("Date:")
            self.date_edit = QLineEdit()
            self.date_edit.setPlaceholderText("e.g., 2024-03-15")
            info_form_layout.addWidget(date_label, 0, 2)
            info_form_layout.addWidget(self.date_edit, 0, 3)

            # Row 1: Telescope (full width)
            telescope_label = QLabel("Telescope:")
            self.telescope_combo = QComboBox()
            self.telescope_combo.setEditable(True)
            self.telescope_combo.setPlaceholderText("Select telescope or enter custom equipment")
            info_form_layout.addWidget(telescope_label, 1, 0)
            info_form_layout.addWidget(self.telescope_combo, 1, 1, 1, 3)  # Span 3 columns

            # Row 2: Notes (label and text edit on same row)
            notes_label = QLabel("Notes:")
            notes_label.setAlignment(Qt.AlignTop)  # Align label to top
            self.notes_edit = QTextEdit()
            self.notes_edit.setPlaceholderText("Additional notes about the image")
            self.notes_edit.setMaximumHeight(60)  # Limit notes height
            info_form_layout.addWidget(notes_label, 2, 0)
            info_form_layout.addWidget(self.notes_edit, 2, 1, 1, 3)  # Span 3 columns

            # Row 3: Save button
            self.save_button = QPushButton("Save Image Information")
            self.save_button.clicked.connect(self._save_image_info)
            info_form_layout.addWidget(self.save_button, 3, 0, 1, 4)  # Span all columns

            self.info_form_container.setLayout(info_form_layout)
            left_layout.addWidget(self.info_form_container, stretch=1)  # Give info area less stretch

            # Relocate image button (initially hidden)
            self.relocate_button = QPushButton("Relocate Image")
            self.relocate_button.clicked.connect(self._relocate_image)
            self.relocate_button.setVisible(False)
            left_layout.addWidget(self.relocate_button)

            # Load telescopes into the dropdown after UI is set up
            self._load_telescopes()

            # Add left layout to a container widget
            left_container = QWidget()
            left_container.setLayout(left_layout)
            main_layout.addWidget(left_container)

            # Right side - Object Information
            right_layout = QVBoxLayout()

            # Create Object information groupbox
            object_info_groupbox = QGroupBox(self.data["name"])
            object_info_groupbox.setStyleSheet(
                "QGroupBox:title { subcontrol-position: top center; font-size: 28pt; font-weight: bold; }")
            object_info_layout = QVBoxLayout()

            # Add Object information with proper null handling
            magnitude_str = f"{self.data['magnitude']:.2f}" if self.data['magnitude'] is not None else "Unknown"
            surface_brightness_str = f"{self.data['surface_brightness']:.2f} mag/arcmin2" if self.data['surface_brightness'] is not None else "Unknown"

            # Handle size information
            size_min = self.data['size_min'] if self.data['size_min'] is not None else 0
            size_max = self.data['size_max'] if self.data['size_max'] is not None else 0
            if size_min > 0 or size_max > 0:
                size_str = f"{size_min:.1f}' - {size_max:.1f}'"
            else:
                size_str = "Unknown"

            object_info_text = (
                f"<b>Right Ascension:</b> {self.ra_str}<br>"
                f"<b>Declination:</b> {self.dec_str}<br>"
                f"<b>Constellation:</b> {self.data['constellation'] or 'Unknown'}<br><br>"
                f"<b>Magnitude:</b> {magnitude_str}<br>"
                f"<b>Surface Brightness:</b> {surface_brightness_str}<br>"
                f"<b>Size:</b> {size_str}<br>"
                f"<b>Type:</b> {self.data['dso_type'] or 'Unknown'}<br>"
                f"<b>Class:</b> {self.data['dso_class'] or 'Unknown'}<br><br>"
                f"<b>Other Designations:</b><br>"
            )

            # Add object designations with proper formatting
            logger.debug(f"Designations data: {self.data.get('designations')}")
            designations_str = self.data.get('designations')
            if designations_str and designations_str.strip():
                designations = [d.strip() for d in designations_str.split(',') if d.strip()]
                logger.debug(f"Split designations: {designations}")

                # Get the current primary designation (as shown in the title)
                primary_name = self.data.get('name', '').strip()

                # Show all designations except the current primary one
                other_designations = []
                seen = set()
                for designation in designations:
                    if designation and designation != primary_name and designation not in seen:
                        other_designations.append(designation)
                        seen.add(designation)

                # Display other designations
                if other_designations:
                    for designation in other_designations:
                        object_info_text += f"{designation}<br>"
                else:
                    object_info_text += "None<br>"
            else:
                object_info_text += "None<br>"

            object_info_text += "<br><b>Source:</b> N.I.N.A. Database"
            update_sources = self._get_update_sources_html()
            if update_sources:
                object_info_text += update_sources

            object_info_label = QLabel(object_info_text)
            object_info_label.setAlignment(Qt.AlignLeft)
            object_info_label.setWordWrap(True)
            object_info_label.setOpenExternalLinks(True)
            object_info_layout.addWidget(object_info_label)

            object_info_groupbox.setLayout(object_info_layout)
            right_layout.addWidget(object_info_groupbox)

            # --- Emission Lines / Gases GroupBox ---
            self.emission_groupbox = QGroupBox("Emission Lines / Gases")
            self.emission_groupbox.setStyleSheet(
                "QGroupBox:title { subcontrol-position: top center; font-size: 16pt; font-weight: bold; }")
            emission_layout = QVBoxLayout()

            self.emission_label = QLabel("Loading emission data from SIMBAD...")
            self.emission_label.setAlignment(Qt.AlignLeft)
            self.emission_label.setWordWrap(True)
            self.emission_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-style: italic;")
            emission_layout.addWidget(self.emission_label)

            # Add SIMBAD link using coordinates
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')
            if ra_deg is not None and dec_deg is not None:
                # Format coordinates for SIMBAD URL (decimal degrees)
                simbad_url = f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra_deg}+{dec_deg}&Radius=3&Radius.unit=arcmin"
                simbad_link_label = QLabel(f"<br><a href='{simbad_url}'>View more details in SIMBAD</a>")
                simbad_link_label.setAlignment(Qt.AlignLeft)
                simbad_link_label.setOpenExternalLinks(True)  # Enable clickable links
                emission_layout.addWidget(simbad_link_label)

            self.emission_groupbox.setLayout(emission_layout)
            right_layout.addWidget(self.emission_groupbox)

            # --- Season / Dates GroupBox ---
            season_groupbox = QGroupBox("Viewing Season / Dates")
            season_groupbox.setStyleSheet(
                "QGroupBox:title { subcontrol-position: top center; font-size: 16pt; font-weight: bold; }")
            season_layout = QVBoxLayout()
            self.season_label = QLabel("")
            self.season_label.setAlignment(Qt.AlignLeft)
            self.season_label.setWordWrap(True)
            season_layout.addWidget(self.season_label)
            season_groupbox.setLayout(season_layout)
            right_layout.addWidget(season_groupbox)
            # User location loaded below; will update groupbox visibility and season

            # Add some spacing
            right_layout.addStretch()

            # Check if DSO is already in target list and update menu action visibility
            self._update_target_list_buttons()

            # Add right layout to a container widget
            right_container = QWidget()
            right_container.setLayout(right_layout)
            main_layout.addWidget(right_container)

            # Add the main content layout to the window layout
            window_layout.addLayout(main_layout)

            # Set the window layout
            self.setLayout(window_layout)

            # Defer image loading to improve initial window performance
            if self.data.get('image_path'):
                # Show image information form but defer actual image loading
                self.info_form_container.setVisible(True)
            else:
                # Hide image information form
                self.info_form_container.setVisible(False)

                self.season_label.setText("Loading location information...")

                # Ensure the window is properly sized before showing
                self.adjustSize()
                logger.debug("DSODetailWindow setup complete")
        except Exception as e:
            logger.error(f"Error in _setup_ui: {str(e)}", exc_info=True)

    def _open_visibility_calculator(self):
        """Open the DSO Visibility Calculator with the current object pre-loaded"""
        if not VISIBILITY_AVAILABLE:
            QMessageBox.warning(self, "Feature Unavailable",
                                "DSO Visibility Calculator is not available. "
                                "Please ensure DSOVisibilityCalculator.py is in the same directory.")
            return

        try:
            # Import here to avoid circular imports
            from main import CustomDSOVisibilityWindow

            # Create a formatted object name for the visibility calculator
            object_name = self.data['name']

            # Get coordinates from the data
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')

            # Create new visibility window with coordinates
            self.visibility_window = CustomDSOVisibilityWindow(object_name, self, ra_deg, dec_deg)
            self.visibility_window.show()

            logger.debug(f"Opened visibility calculator for {object_name} with coordinates RA={ra_deg}, Dec={dec_deg}")

        except Exception as e:
            logger.error(f"Error opening visibility calculator: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open visibility calculator: {str(e)}")

    def _fit_to_window(self):
        """Fit the image to the window size"""
        if self.original_pixmap is None:
            return

        # Get available size
        available_size = self.image_label.size()

        # Calculate zoom factor to fit
        width_ratio = available_size.width() / self.original_pixmap.width()
        height_ratio = available_size.height() / self.original_pixmap.height()
        self.zoom_factor = min(width_ratio, height_ratio)

        # Reset position
        self.image_position = [0, 0]

        # Update display
        self._update_zoom()

    def resizeEvent(self, event):
        """Handle window resize event"""
        super().resizeEvent(event)

        # If no image is loaded, just return
        if self.original_pixmap is None:
            return

        # Store the current relative position of the image center before resize
        old_size = event.oldSize() if event.oldSize().isValid() else self.size()
        new_size = event.size()

        # Only auto-fit if we're at the initial zoom level (user hasn't manually zoomed)
        if abs(self.zoom_factor - self.initial_zoom_factor) < 0.001:
            # User is at initial zoom - maintain fit to window behavior
            self._fit_to_window()
        else:
            # User has manually zoomed - preserve zoom level and try to maintain image position
            if old_size.isValid() and old_size.width() > 0 and old_size.height() > 0:
                # Calculate the ratio of size change
                width_ratio = new_size.width() / old_size.width()
                height_ratio = new_size.height() / old_size.height()

                # Adjust image position proportionally to maintain relative position
                self.image_position[0] = int(self.image_position[0] * width_ratio)
                self.image_position[1] = int(self.image_position[1] * height_ratio)

            # Update the display with preserved zoom and adjusted position
            self._update_zoom()

    def _load_telescopes(self):
        """Load user telescopes into the telescope dropdown"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM usertelescopes ORDER BY name")
                telescopes = cursor.fetchall()

                # Clear existing items
                self.telescope_combo.clear()

                # Add telescopes to the dropdown
                for telescope in telescopes:
                    self.telescope_combo.addItem(telescope[0])

                logger.debug(f"Loaded {len(telescopes)} telescopes into dropdown")
        except Exception as e:
            logger.error(f"Error loading telescopes: {e}")

    def _load_image_info(self):
        """Load image information into the form fields"""
        self.integration_edit.setText(self.data.get('integration_time', ''))
        # Set telescope combo box text (works for both existing items and custom text)
        self.telescope_combo.setCurrentText(self.data.get('equipment', ''))
        self.date_edit.setText(self.data.get('date_taken', ''))
        self.notes_edit.setText(self.data.get('notes', ''))

    def _set_season_label_from_location(self):
        """
        Set the season_label text with the visibility season/dates string based on user location.
        Reads location directly from the database. Uses background thread for heavy calculations.
        """
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
            if not row or row[0] is None or row[1] is None:
                self.season_label.setText("Set your location in Settings to see viewing season.")
                return

            lat = float(row[0])
            lon = float(row[1])
            ra_deg = self.data.get("ra_deg")
            dec_deg = self.data.get("dec_deg")
            object_name = self.data.get("name")

            # Show loading message
            self.season_label.setText("Calculating viewing seasons...")

            # Create worker thread for visibility calculation
            self.visibility_thread = QThread()
            self.visibility_worker = VisibilityCalculationWorker(lat, lon, ra_deg, dec_deg, object_name)

            # Move worker to thread
            self.visibility_worker.moveToThread(self.visibility_thread)

            # Connect signals
            self.visibility_thread.started.connect(self.visibility_worker.calculate_visibility)
            self.visibility_worker.finished.connect(self._on_visibility_calculated)
            self.visibility_worker.error.connect(self._on_visibility_error)
            self.visibility_worker.finished.connect(self.visibility_thread.quit)
            self.visibility_worker.error.connect(self.visibility_thread.quit)
            self.visibility_thread.finished.connect(self.visibility_thread.deleteLater)

            # Start the thread
            self.visibility_thread.start()

        except Exception as e:
            logger.error(f"Error setting up visibility calculation: {str(e)}")
            self.season_label.setText(f"Error setting up viewing season calculation:<br>{str(e)}")

    def _on_visibility_calculated(self, visibility_text):
        """Handle completion of visibility calculation"""
        self.season_label.setText(visibility_text)
        logger.debug("Visibility calculation completed in background thread")

    def _on_visibility_error(self, error_message):
        """Handle error from visibility calculation"""
        self.season_label.setText(error_message)
        logger.error("Visibility calculation failed in background thread")

    def _save_image_info(self):
        """Save the current image information to the database"""
        try:
            # Only save if we have images and a valid current image
            if not self.user_images or self.current_image_index >= len(self.user_images):
                QMessageBox.warning(self, "No Image", "No image selected to save information for.")
                return

            current_image = self.user_images[self.current_image_index]
            image_id = current_image['id']

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Update the specific image information
                cursor.execute("""
                    UPDATE userimages
                    SET integration_time = ?,
                        equipment = ?,
                        date_taken = ?,
                        notes = ?
                    WHERE id = ?
                """, (
                    self.integration_edit.text().strip(),
                    self.telescope_combo.currentText().strip(),
                    self.date_edit.text().strip(),
                    self.notes_edit.toPlainText().strip(),
                    image_id
                ))
                conn.commit()

                # Update the current image data in our local list
                current_image.update({
                    'integration_time': self.integration_edit.text().strip(),
                    'equipment': self.telescope_combo.currentText().strip(),
                    'date_taken': self.date_edit.text().strip(),
                    'notes': self.notes_edit.toPlainText().strip()
                })

                logger.debug(f"Successfully updated image information for image {image_id}")

                QMessageBox.information(self, "Success", "Image information saved successfully!")

        except Exception as e:
            logger.error(f"Error saving image information: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save image information: {str(e)}")

    def _add_user_image(self):
        """Add a user image for the object and all its designations"""
        # Get the image file
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Image for {self.data['name']}",
            os.path.expanduser("~"),  # Start in user's home directory
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fts);;Regular Images (*.png *.jpg *.jpeg *.tif *.tiff);;FITS Files (*.fits *.fit *.fts);;All Files (*)"
        )

        if file_name:
            try:
                # Store in database using DatabaseManager
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()

                    # Get the dsodetailid for this object
                    cursor.execute("""
                        SELECT d.id
                        FROM dsodetail d
                        JOIN cataloguenr c ON d.id = c.dsodetailid
                        WHERE c.catalogue = ? AND c.designation = ?
                    """, (self.data['catalogue'], self.data['id']))
                    result = cursor.fetchone()

                    if result:
                        dsodetailid = result[0]

                        # Get all designations for this object
                        cursor.execute("""
                            SELECT c.catalogue, c.designation
                            FROM cataloguenr c
                            WHERE c.dsodetailid = ?
                        """, (dsodetailid,))
                        all_designations = cursor.fetchall()

                        logger.debug(f"Found {len(all_designations)} designations for this object")

                        # Insert new image record with blank metadata (user will fill in after)
                        cursor.execute("""
                            INSERT INTO userimages (
                                dsodetailid, image_path, integration_time,
                                equipment, date_taken, notes, created_date
                            ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                        """, (
                            dsodetailid, file_name,
                            '',  # Blank integration time
                            '',  # Blank equipment
                            '',  # Blank date
                            ''   # Blank notes
                        ))

                        # Get the ID of the newly inserted image
                        new_image_id = cursor.lastrowid

                        # Log all designations that will share this image
                        for catalogue, designation in all_designations:
                            logger.debug(f"Image will be available for {catalogue} {designation}")

                        conn.commit()

                        # Reload all user images to get the updated list
                        self._load_user_images()

                        # Navigate to the newly added image (it will be the last one)
                        if self.user_images:
                            # Find the index of the newly added image
                            for i, img in enumerate(self.user_images):
                                if img['id'] == new_image_id:
                                    # Show loading state
                                    self.image_label.setText("Loading image...")
                                    self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")

                                    self.current_image_index = i
                                    self._load_user_image(img['image_path'])
                                    self._load_current_image_info()
                                    self._update_image_navigation()
                                    break

                        # Show the image information form
                        self.info_form_container.setVisible(True)

                        logger.debug(f"Successfully added user image for {len(all_designations)} designations")

                        # Emit signal to notify that an image was added
                        self.image_added.emit()
                    else:
                        logger.error(f"Could not find dsodetailid for {self.data['name']}")

            except Exception as e:
                logger.error(f"Error adding user image: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to add image: {str(e)}")

    def _load_fits_image(self, fits_path, colormap='viridis'):
        """
        Load a FITS image file and convert to QPixmap with color mapping.

        Args:
            fits_path (str): Path to the FITS file
            colormap (str): Matplotlib colormap name ('viridis', 'hot', 'cool', 'plasma', 'inferno', 'gray')
        """
        try:
            from PySide6.QtGui import QImage

            logger.debug(f"Loading FITS file: {fits_path}")

            # Import required libraries
            from astropy.io import fits
            from astropy.visualization import simple_norm
            import numpy as np

            # Open FITS file
            with fits.open(fits_path) as hdul:
                # Get the primary image data (usually the first HDU with data)
                image_data = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        image_data = hdu.data
                        break

                if image_data is None:
                    logger.error(f"No image data found in FITS file: {fits_path}")
                    return None

                # Handle different dimensionalities
                is_rgb = False
                if len(image_data.shape) > 2:
                    # Check if this is an RGB image (3 color planes)
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        # This is an RGB FITS image
                        is_rgb = True
                        logger.debug("Detected RGB FITS image")
                    elif len(image_data.shape) == 3 and image_data.shape[0] == 3:
                        # RGB planes are in first dimension, transpose
                        image_data = np.transpose(image_data, (1, 2, 0))
                        is_rgb = True
                        logger.debug("Detected RGB FITS image (transposed)")
                    elif len(image_data.shape) == 3:
                        # Take the first 2D slice if it's a cube
                        image_data = image_data[0]
                        logger.debug("Using first slice of 3D FITS data")
                    elif len(image_data.shape) == 4:
                        image_data = image_data[0, 0]
                        logger.debug("Using first slice of 4D FITS data")
                    else:
                        logger.error(f"Unsupported FITS image dimensions: {image_data.shape}")
                        return None

                # Normalize the data for display (handle NaN values)
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

                if is_rgb:
                    # Handle RGB FITS data - normalize each channel separately
                    logger.debug("Processing RGB FITS data")
                    normalized_data = np.zeros_like(image_data)

                    for channel in range(3):
                        channel_data = image_data[:, :, channel]
                        # Apply normalization to each color channel
                        try:
                            norm = simple_norm(channel_data, stretch='linear', percent=99.5)
                            normalized_data[:, :, channel] = norm(channel_data)
                        except Exception as e:
                            logger.warning(f"Astropy normalization failed for channel {channel}, using simple scaling: {e}")
                            # Fallback to simple min-max normalization per channel
                            data_min, data_max = np.percentile(channel_data, [0.5, 99.5])
                            if data_max > data_min:
                                normalized_data[:, :, channel] = (channel_data - data_min) / (data_max - data_min)
                            else:
                                normalized_data[:, :, channel] = channel_data

                    # Clip to valid range
                    normalized_data = np.clip(normalized_data, 0, 1)

                    # Convert directly to 8-bit RGB (no false color mapping needed)
                    rgb_data = (normalized_data * 255).astype(np.uint8)

                    # Ensure the array is C-contiguous for QImage
                    if not rgb_data.flags['C_CONTIGUOUS']:
                        rgb_data = np.ascontiguousarray(rgb_data)

                    # Create QImage from RGB array
                    height, width, channels = rgb_data.shape
                    bytes_per_line = width * channels

                    qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    logger.debug("Created RGB QImage from FITS data")

                else:
                    # Handle grayscale FITS data
                    logger.debug("Processing grayscale FITS data")

                    # Apply simple normalization (linear stretch between percentiles)
                    try:
                        norm = simple_norm(image_data, stretch='linear', percent=99.5)
                        normalized_data = norm(image_data)
                    except Exception as e:
                        logger.warning(f"Astropy normalization failed, using simple scaling: {e}")
                        # Fallback to simple min-max normalization
                        data_min, data_max = np.percentile(image_data, [0.5, 99.5])
                        if data_max > data_min:
                            normalized_data = (image_data - data_min) / (data_max - data_min)
                        else:
                            normalized_data = image_data
                        normalized_data = np.clip(normalized_data, 0, 1)

                    # For grayscale data, apply color mapping if specified
                    if colormap == 'gray' or colormap == 'grey':
                        # Display as grayscale
                        image_8bit = (normalized_data * 255).astype(np.uint8)

                        # Ensure the array is C-contiguous for QImage
                        if not image_8bit.flags['C_CONTIGUOUS']:
                            image_8bit = np.ascontiguousarray(image_8bit)

                        height, width = image_8bit.shape
                        bytes_per_line = width
                        qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                        logger.debug("Created grayscale QImage from FITS data")
                    else:
                        # Apply color mapping for better visualization
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib.cm as cm

                            # Apply a color map for better astronomical visualization
                            try:
                                cmap = cm.get_cmap(colormap)
                            except ValueError:
                                logger.warning(f"Unknown colormap '{colormap}', falling back to 'viridis'")
                                cmap = cm.get_cmap('viridis')

                            colored_data = cmap(normalized_data)

                            # Convert to 8-bit RGB
                            rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)

                            # Ensure the array is C-contiguous for QImage
                            if not rgb_data.flags['C_CONTIGUOUS']:
                                rgb_data = np.ascontiguousarray(rgb_data)

                            # Create QImage from RGB array
                            height, width, channels = rgb_data.shape
                            bytes_per_line = width * channels

                            qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            logger.debug(f"Created color-mapped QImage using {colormap}")

                        except ImportError:
                            logger.warning("Matplotlib not available, displaying FITS as grayscale")
                            # Fallback to grayscale
                            image_8bit = (normalized_data * 255).astype(np.uint8)

                            # Ensure the array is C-contiguous for QImage
                            if not image_8bit.flags['C_CONTIGUOUS']:
                                image_8bit = np.ascontiguousarray(image_8bit)

                            height, width = image_8bit.shape
                            bytes_per_line = width
                            qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                        except Exception as e:
                            logger.warning(f"Color mapping failed, using grayscale: {e}")
                            # Fallback to grayscale
                            image_8bit = (normalized_data * 255).astype(np.uint8)

                            # Ensure the array is C-contiguous for QImage
                            if not image_8bit.flags['C_CONTIGUOUS']:
                                image_8bit = np.ascontiguousarray(image_8bit)

                            height, width = image_8bit.shape
                            bytes_per_line = width
                            qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                # Convert to QPixmap
                pixmap = QPixmap.fromImage(qimage)

                if not pixmap.isNull():
                    logger.debug(f"Successfully loaded FITS image: {width}x{height}")
                    return pixmap
                else:
                    logger.error("Failed to convert FITS data to QPixmap")
                    return None

        except ImportError as e:
            logger.error(f"Missing required libraries for FITS support (astropy): {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading FITS file {fits_path}: {str(e)}")
            return None

    def _load_user_image(self, image_path):
        """Load and display a user image with caching (supports FITS files) - async version"""
        try:
            # Store the current image path
            self.current_image_path = image_path

            logger.debug(f"Attempting to load image: {image_path}")

            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                self.image_label.setText(f"Image file not found:\n{os.path.basename(image_path)}\n\nClick 'Relocate Image' button to find the new location")
                self.image_label.setToolTip(f"File path: {image_path}")
                self._show_relocate_button()
                return

            # Check file size
            try:
                file_size = os.path.getsize(image_path)
                logger.debug(f"Image file size: {file_size} bytes")
                if file_size == 0:
                    logger.error(f"Image file is empty: {image_path}")
                    self.image_label.setText(f"Image file is empty:\n{os.path.basename(image_path)}")
                    self.image_label.setToolTip(f"File path: {image_path}")
                    return
            except OSError as e:
                logger.error(f"Error checking file size: {e}")
                self.image_label.setText(f"Cannot access image file:\n{os.path.basename(image_path)}")
                self.image_label.setToolTip(f"File path: {image_path}\nError: {str(e)}")
                return

            # Check cache first
            cached_pixmap = self.image_cache.get(image_path)
            if cached_pixmap:
                logger.debug("Using cached image")
                self.original_pixmap = cached_pixmap
                self._display_loaded_image()
                return

            # Stop any existing loading thread
            if self.image_loader_thread and self.image_loader_thread.isRunning():
                self.image_loader_thread.quit()
                self.image_loader_thread.wait()

            # Determine if FITS file
            file_ext = os.path.splitext(image_path)[1].lower()
            is_fits = file_ext in ['.fits', '.fit', '.fts']

            # FITS files load synchronously on main thread to avoid
            # color corruption from thread-based QPixmap/QImage conversions
            if is_fits:
                logger.debug("Loading FITS file synchronously on main thread")
                pixmap = self._load_fits_image(image_path, self.FITS_COLORMAP)
                if pixmap and not pixmap.isNull():
                    self.original_pixmap = pixmap
                    self.image_cache.put(image_path, pixmap)
                    self._display_loaded_image()
                    logger.info("FITS image loaded successfully")
                else:
                    self._on_image_load_failed(image_path, "Failed to load FITS file")
            else:
                # Use background thread for non-FITS images (PNG, JPG, etc.)
                self.image_loader_thread = ImageLoaderThread(image_path, is_fits, self.FITS_COLORMAP)
                self.image_loader_thread.image_loaded.connect(self._on_image_loaded)
                self.image_loader_thread.load_failed.connect(self._on_image_load_failed)
                self.image_loader_thread.start()
                logger.debug(f"Started background loading for: {image_path}")

        except Exception as e:
            logger.error(f"Error initiating image load: {str(e)}", exc_info=True)
            self.image_label.setText(f"Error loading image:\n{str(e)}")

    def _on_image_loaded(self, qimage, image_path):
        """Handle image loaded signal from background thread"""
        try:
            logger.info(f"Full image received: {qimage.width()}x{qimage.height()}")

            # Convert QImage to QPixmap (must be done on main thread)
            self.original_pixmap = QPixmap.fromImage(qimage)

            if self.original_pixmap.isNull():
                logger.error(f"Failed to convert QImage to QPixmap for: {image_path}")
                self._on_image_load_failed(image_path, "Failed to convert image")
                return

            logger.info(f"Full resolution image converted to pixmap: {self.original_pixmap.width()}x{self.original_pixmap.height()}")

            # Cache the loaded image
            self.image_cache.put(image_path, self.original_pixmap)

            # Display the image
            self._display_loaded_image()
            logger.info("Full resolution image displayed")

        except Exception as e:
            logger.error(f"Error handling loaded image: {str(e)}", exc_info=True)
            self._on_image_load_failed(image_path, str(e))

    def _on_image_load_failed(self, image_path, error_message):
        """Handle image load failure from background thread"""
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext in ['.fits', '.fit', '.fts']:
            self.image_label.setText(f"Failed to load FITS file:\n{os.path.basename(image_path)}\n\nRequires astropy library for FITS support\nError: {error_message}")
        else:
            self.image_label.setText(f"Failed to load image:\n{os.path.basename(image_path)}\n\nError: {error_message}")

        self.image_label.setToolTip(f"File path: {image_path}")
        logger.error(f"Image load failed: {image_path} - {error_message}")

    def _display_loaded_image(self):
        """Display the loaded image with appropriate scaling"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        # Calculate the appropriate size for the image
        label_size = self.image_label.size()
        scaled_pixmap = self.original_pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation  # Use fast transformation for quicker display
        )

        # Calculate initial zoom factor based on the scaled image
        self.initial_zoom_factor = min(
            label_size.width() / self.original_pixmap.width(),
            label_size.height() / self.original_pixmap.height()
        )
        self.zoom_factor = self.initial_zoom_factor
        self.image_position = [0, 0]

        # Update the display
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Add tooltip for image viewer
        self.image_label.setToolTip("Double click to open in the image viewer")
        logger.debug("Image loaded and displayed successfully")

    def _open_aladin_lite(self, data):
        """Open Aladin Lite in a new window"""
        try:
            # Import here to avoid circular imports
            from main import AladinLiteWindow

            if not hasattr(self, 'aladin_window') or not self.aladin_window.isVisible():
                self.aladin_window = AladinLiteWindow(data, self)
                self.aladin_window.show()
                logger.debug(f"Opened Aladin Lite window for {data['name']}")
            else:
                self.aladin_window.raise_()
                self.aladin_window.activateWindow()
                logger.debug(f"Aladin Lite window already open, bringing to front")
        except Exception as e:
            logger.error(f"Error opening Aladin Lite: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

    def _open_wikipedia(self):
        """Open Wikipedia page for the current DSO in the default browser"""
        try:
            import re

            dso_name = self.data.get('name', '').strip()

            if not dso_name:
                QMessageBox.warning(self, "No Name", "Cannot open Wikipedia: DSO name not available.")
                return

            dso_name = re.sub(r'\s*\([^)]*\)', '', dso_name).strip()

            if re.match(r'^M\s*\d+$', dso_name):
                number = re.search(r'\d+', dso_name).group()
                wiki_name = f'Messier_{number}'
            elif re.match(r'^Sh2[\s\-]*(\d+)$', dso_name, re.IGNORECASE):
                match = re.match(r'^Sh2[\s\-]*(\d+)$', dso_name, re.IGNORECASE)
                number = match.group(1)
                wiki_name = f'Sh_2-{number}'
            else:
                wiki_name = dso_name.replace(' ', '_')

            wiki_url = f"https://en.wikipedia.org/wiki/{wiki_name}"

            logger.debug(f"Opening Wikipedia page: {wiki_url}")
            open_url(wiki_url)

        except Exception as e:
            logger.error(f"Error opening Wikipedia: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Wikipedia: {str(e)}")

    def eventFilter(self, obj, event):
        """Handle mouse events for zooming and panning"""
        if obj == self.image_label and self.original_pixmap is not None:
            if event.type() == QEvent.Wheel:
                if event.angleDelta().y() > 0:
                    self._zoom_in()
                else:
                    self._zoom_out()
                return True
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.drag_start_position = event.position()
                self.drag_start_image_position = self.image_position.copy()
                return True
            elif event.type() == QEvent.MouseMove and event.buttons() & Qt.LeftButton:
                if self.drag_start_position is not None:
                    dx = event.position().x() - self.drag_start_position.x()
                    dy = event.position().y() - self.drag_start_position.y()
                    self.image_position[0] = self.drag_start_image_position[0] + dx
                    self.image_position[1] = self.drag_start_image_position[1] + dy
                    self._update_zoom()
                    return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.drag_start_position = None
                self.drag_start_image_position = None
                return True
            elif event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self._open_image_viewer()
                return True
        return super().eventFilter(obj, event)

    def _update_zoom(self):
        """Update the image display with current zoom level and position"""
        if self.original_pixmap is not None:
            new_width = int(self.original_pixmap.width() * self.zoom_factor)
            new_height = int(self.original_pixmap.height() * self.zoom_factor)

            scaled_pixmap = self.original_pixmap.scaled(
                new_width, new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            label_size = self.image_label.size()
            final_pixmap = QPixmap(label_size)
            final_pixmap.fill(Qt.transparent)

            painter = QPainter(final_pixmap)
            x = (label_size.width() - scaled_pixmap.width()) // 2 + self.image_position[0]
            y = (label_size.height() - scaled_pixmap.height()) // 2 + self.image_position[1]
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()

            self.image_label.setPixmap(final_pixmap)

    def _zoom_in(self):
        """Zoom in on the image"""
        if self.original_pixmap is not None:
            self.zoom_factor = min(self.zoom_factor * 1.2, 4.0 * self.initial_zoom_factor)
            self._update_zoom()

    def _zoom_out(self):
        """Zoom out on the image"""
        if self.original_pixmap is not None:
            self.zoom_factor = max(self.zoom_factor / 1.2, self.initial_zoom_factor)
            self._update_zoom()

    def _reset_zoom(self):
        """Reset zoom and position to original"""
        if self.original_pixmap is not None:
            self.zoom_factor = self.initial_zoom_factor
            self.image_position = [0, 0]
            self._update_zoom()

    def _open_image_viewer(self):
        """Open the current image in a new window"""
        if self.original_pixmap is not None:
            try:
                from ImageViewer import ImageViewerWindow

                ra_deg = self.data.get("ra_deg")
                dec_deg = self.data.get("dec_deg")
                viewer = ImageViewerWindow(
                    self.original_pixmap, self.data["name"], self.current_image_path, self,
                    dso_ra=ra_deg, dso_dec=dec_deg
                )
                viewer.setModal(False)
                viewer.show()
                logger.debug(f"Opened image viewer for {self.data['name']} (RA={ra_deg}, Dec={dec_deg})")
            except Exception as e:
                logger.error(f"Error opening image viewer: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to open image viewer: {str(e)}")

    def _load_user_images(self):
        """Load all user images for this DSO from the database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT d.id
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ? AND c.designation = ?
                """, (self.data['catalogue'], self.data['id']))
                result = cursor.fetchone()

                if result:
                    dsodetailid = result[0]

                    cursor.execute("""
                        SELECT id, image_path, integration_time, equipment, date_taken, notes, is_favorite
                        FROM userimages
                        WHERE dsodetailid = ?
                        ORDER BY is_favorite DESC, id ASC
                    """, (dsodetailid,))

                    images = cursor.fetchall()

                    self.user_images = []
                    for img_id, image_path, integration_time, equipment, date_taken, notes, is_favorite in images:
                        self.user_images.append({
                            'id': img_id,
                            'dsodetailid': dsodetailid,
                            'image_path': image_path,
                            'integration_time': integration_time,
                            'equipment': equipment,
                            'date_taken': date_taken,
                            'notes': notes,
                            'is_favorite': is_favorite
                        })

                    logger.debug(f"Loaded {len(self.user_images)} images for {self.data['name']}")

                    self._update_image_navigation()

                    if self.user_images:
                        self.image_label.setText("Loading image...")
                        self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")

                        self.current_image_index = 0
                        current_image = self.user_images[self.current_image_index]
                        self._load_user_image(current_image['image_path'])
                        self._load_current_image_info()
                        self.info_form_container.setVisible(True)
                    else:
                        self.image_label.setText("No image attached to this DSO.")
                        self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")
                        self.info_form_container.setVisible(False)
                else:
                    logger.error(f"Could not find dsodetailid for {self.data['name']}")
                    self.image_label.setText("No image attached to this DSO.")
                    self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")
                    self.info_form_container.setVisible(False)

        except Exception as e:
            logger.error(f"Error loading user images: {str(e)}", exc_info=True)
            self.image_label.setText("Error loading images.")
            self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['error']};")
            self.info_form_container.setVisible(False)

    def _update_image_navigation(self):
        """Update the image navigation controls based on current state"""
        image_count = len(self.user_images)

        if image_count == 0:
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.delete_image_button.setEnabled(False)
            self.image_counter_label.setText("0/0")
        elif image_count == 1:
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.delete_image_button.setEnabled(True)
            self.image_counter_label.setText("1/1")
        else:
            self.prev_image_button.setEnabled(self.current_image_index > 0)
            self.next_image_button.setEnabled(self.current_image_index < image_count - 1)
            self.delete_image_button.setEnabled(True)
            self.image_counter_label.setText(f"{self.current_image_index + 1}/{image_count}")

    def _previous_image(self):
        """Navigate to the previous image"""
        if self.user_images and self.current_image_index > 0:
            self.image_label.setText("Loading image...")
            self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")

            self.current_image_index -= 1
            current_image = self.user_images[self.current_image_index]
            self._load_user_image(current_image['image_path'])
            self._load_current_image_info()
            self._update_image_navigation()
            logger.debug(f"Navigated to previous image: {self.current_image_index + 1}/{len(self.user_images)}")

    def _next_image(self):
        """Navigate to the next image"""
        if self.user_images and self.current_image_index < len(self.user_images) - 1:
            self.image_label.setText("Loading image...")
            self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")

            self.current_image_index += 1
            current_image = self.user_images[self.current_image_index]
            self._load_user_image(current_image['image_path'])
            self._load_current_image_info()
            self._update_image_navigation()
            logger.debug(f"Navigated to next image: {self.current_image_index + 1}/{len(self.user_images)}")

    def _load_current_image_info(self):
        """Load the current image's information into the form fields"""
        if self.user_images and 0 <= self.current_image_index < len(self.user_images):
            current_image = self.user_images[self.current_image_index]
            self.integration_edit.setText(current_image.get('integration_time', ''))
            self.telescope_combo.setCurrentText(current_image.get('equipment', ''))
            self.date_edit.setText(current_image.get('date_taken', ''))
            self.notes_edit.setText(current_image.get('notes', ''))

            is_favorite = current_image.get('is_favorite', 0)
            if is_favorite:
                self.favorite_button.setText("*")
                self.favorite_button.setStyleSheet(f"QPushButton {{ color: {COLORS['favorite']}; font-size: 12pt; }}")
                self.favorite_button.setToolTip("Unmark as favorite")
            else:
                self.favorite_button.setText("*")
                self.favorite_button.setStyleSheet(f"QPushButton {{ color: {COLORS['text_disabled']}; font-size: 12pt; }}")
                self.favorite_button.setToolTip("Mark as favorite")

    def _toggle_favorite(self):
        """Toggle the favorite status of the current image"""
        if not self.user_images or self.current_image_index < 0 or self.current_image_index >= len(self.user_images):
            return

        current_image = self.user_images[self.current_image_index]
        image_id = current_image.get('id')

        if not image_id:
            logger.error("Cannot toggle favorite: image has no ID")
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                current_favorite = current_image.get('is_favorite', 0)
                new_favorite = 1 if not current_favorite else 0

                if new_favorite == 1:
                    dsodetailid = current_image.get('dsodetailid')
                    if not dsodetailid:
                        cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (image_id,))
                        result = cursor.fetchone()
                        if result:
                            dsodetailid = result[0]

                    if dsodetailid:
                        cursor.execute("""
                            UPDATE userimages SET is_favorite = 0
                            WHERE dsodetailid = ? AND id != ?
                        """, (dsodetailid, image_id))

                cursor.execute("UPDATE userimages SET is_favorite = ? WHERE id = ?", (new_favorite, image_id))
                conn.commit()

                current_image['is_favorite'] = new_favorite

                if new_favorite == 1:
                    for img in self.user_images:
                        if img['id'] != image_id:
                            img['is_favorite'] = 0

                self._load_current_image_info()
                logger.debug(f"Toggled favorite status for image {image_id} to {new_favorite}")

        except Exception as e:
            logger.error(f"Error toggling favorite: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to toggle favorite: {str(e)}")

    def _show_relocate_button(self):
        """Show the relocate button when an image cannot be found"""
        self.relocate_button.setVisible(True)
        logger.debug("Relocate button shown due to missing image file")

    def _relocate_image(self):
        """Allow user to select new location for missing image file"""
        try:
            if not hasattr(self, 'current_image_path') or not self.current_image_path:
                QMessageBox.warning(self, "Error", "No image selected for relocation.")
                return

            if not self.user_images or self.current_image_index < 0 or self.current_image_index >= len(self.user_images):
                QMessageBox.warning(self, "Error", "No current image to relocate.")
                return

            current_image = self.user_images[self.current_image_index]
            image_id = current_image.get('id')

            if not image_id:
                QMessageBox.warning(self, "Error", "Cannot identify current image for relocation.")
                return

            original_filename = os.path.basename(self.current_image_path)

            new_image_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select new location for {original_filename}",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.fits *.fit *.fts);;All Files (*)"
            )

            if new_image_path:
                try:
                    self.db_manager.execute_update(
                        "UPDATE userimages SET image_path = ? WHERE id = ?",
                        (new_image_path, image_id)
                    )

                    self.current_image_path = new_image_path
                    current_image['image_path'] = new_image_path

                    self.relocate_button.setVisible(False)
                    self.image_label.setText("Loading image...")
                    self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")
                    self._load_user_image(new_image_path)

                    QMessageBox.information(self, "Success", f"Image location updated successfully!")
                    logger.info(f"Image relocated successfully to {new_image_path}")

                except Exception as db_error:
                    logger.error(f"Database error during image relocation: {str(db_error)}")
                    QMessageBox.critical(self, "Database Error", f"Failed to update image location: {str(db_error)}")

        except Exception as e:
            logger.error(f"Error relocating image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to relocate image: {str(e)}")

    def _delete_current_image(self):
        """Delete the current image from the database and update the display"""
        try:
            if not self.user_images:
                QMessageBox.warning(self, "No Images", "No images available to delete.")
                return

            if self.current_image_index < 0 or self.current_image_index >= len(self.user_images):
                QMessageBox.warning(self, "Error", "No current image selected for deletion.")
                return

            current_image = self.user_images[self.current_image_index]
            image_id = current_image.get('id')
            image_path = current_image.get('image_path', 'Unknown')

            if not image_id:
                QMessageBox.warning(self, "Error", "Cannot identify current image for deletion.")
                return

            filename = os.path.basename(image_path) if image_path != 'Unknown' else 'this image'

            reply = QMessageBox.question(
                self, "Delete Image",
                f"Are you sure you want to delete '{filename}'?\n\nThis will remove the image record from the database.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM userimages WHERE id = ?", (image_id,))
                conn.commit()

                del self.user_images[self.current_image_index]

                if self.user_images:
                    if self.current_image_index >= len(self.user_images):
                        self.current_image_index = len(self.user_images) - 1

                    current_image = self.user_images[self.current_image_index]
                    self._load_user_image(current_image['image_path'])
                    self._load_current_image_info()
                else:
                    self.current_image_index = 0
                    self.image_label.setText("No Image Loaded")
                    self.image_label.setStyleSheet(f"font-size: 14pt; color: {COLORS['text_disabled']};")
                    self._clear_image_info()

                self._update_image_navigation()

                QMessageBox.information(self, "Success", f"'{filename}' has been removed from the database.")
                logger.info(f"Deleted image with ID {image_id}: {filename}")

        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to delete image: {str(e)}")

    def _clear_image_info(self):
        """Clear all image information fields"""
        self.integration_edit.setText("")
        self.telescope_combo.setCurrentText("")
        self.date_edit.setText("")
        self.notes_edit.setText("")

    def _update_target_list_buttons(self):
        """Update target list menu action visibility"""
        try:
            is_in_target_list = self._check_if_in_target_list()

            if is_in_target_list:
                self.add_target_action.setVisible(False)
                self.remove_target_action.setVisible(True)
                self.open_target_action.setVisible(True)
            else:
                self.add_target_action.setVisible(True)
                self.remove_target_action.setVisible(False)
                self.open_target_action.setVisible(False)

        except Exception as e:
            logger.error(f"Error updating target list menu actions: {str(e)}")
            self.add_target_action.setVisible(True)
            self.remove_target_action.setVisible(False)
            self.open_target_action.setVisible(False)

    def _check_if_in_target_list(self):
        """Check if current DSO is already in the target list"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')

                if not dso_name:
                    return False

                cursor.execute("""
                    SELECT COUNT(*) FROM usertargetlist
                    WHERE UPPER(TRIM(name)) = ?
                """, (dso_name.upper(),))

                if cursor.fetchone()[0] > 0:
                    return True

                if ra_deg is not None and dec_deg is not None:
                    cursor.execute("""
                        SELECT COUNT(*) FROM usertargetlist
                        WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001
                    """, (ra_deg, dec_deg))

                    if cursor.fetchone()[0] > 0:
                        return True

                return False

        except Exception as e:
            logger.error(f"Error checking target list status: {str(e)}")
            return False

    def _find_target_list_name(self):
        """Find the actual name used in the target list for this DSO"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')

                if not dso_name:
                    return None

                cursor.execute("""
                    SELECT name FROM usertargetlist
                    WHERE UPPER(TRIM(name)) = ? LIMIT 1
                """, (dso_name.upper(),))

                result = cursor.fetchone()
                if result:
                    return result[0]

                if ra_deg is not None and dec_deg is not None:
                    cursor.execute("""
                        SELECT name FROM usertargetlist
                        WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001 LIMIT 1
                    """, (ra_deg, dec_deg))

                    result = cursor.fetchone()
                    if result:
                        return result[0]

                return None

        except Exception as e:
            logger.error(f"Error finding target list name: {str(e)}")
            return None

    def _add_to_target_list(self):
        """Add this DSO to the target list"""
        try:
            from DSOTargetList import AddTargetDialog

            enhanced_data = self.data.copy()
            visibility_text = self.season_label.text()

            if visibility_text and not visibility_text.startswith("Enter your location") and not visibility_text.startswith("Loading"):
                cleaned_months = self._extract_month_ranges_from_visibility(visibility_text)
                if cleaned_months:
                    enhanced_data['best_months'] = cleaned_months
                    logger.debug(f"Using cleaned visibility info for {self.data.get('name', 'DSO')}: {cleaned_months}")

            dialog = AddTargetDialog(dso_data=enhanced_data, parent=self)
            if dialog.exec():
                self._update_target_list_buttons()

        except ImportError as e:
            logger.error(f"Could not import DSOTargetList: {str(e)}")
            QMessageBox.warning(self, "Import Error", f"Could not load Target List feature: {e}")
        except Exception as e:
            logger.error(f"Error adding to target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {str(e)}")

    def _send_to_nina(self):
        """Send DSO coordinates to NINA Framing Assistant"""
        NINAIntegration.send_to_framing_assistant(
            self.data.get('ra_deg'), self.data.get('dec_deg'),
            self.data.get('name', 'Unknown'), self
        )

    def _slew_to_nina_target(self):
        """Slew mount to the current DSO coordinates via NINA"""
        NINAIntegration.slew_to_coordinates(
            self.data.get('ra_deg'), self.data.get('dec_deg'),
            self.data.get('name', 'Unknown'), self
        )

    def _extract_month_ranges_from_visibility(self, visibility_text):
        """Extract only month ranges from visibility text"""
        import re

        clean_text = re.sub(r'<[^>]+>', '', visibility_text)

        if "not optimally visible" in clean_text.lower() or "error" in clean_text.lower():
            return ""

        month_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+\s*-\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+'

        matches = re.findall(month_pattern, clean_text)

        if matches:
            month_abbrev = {
                'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
                'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
                'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
            }

            ranges = []
            for start_month, end_month in matches:
                start_abbrev = month_abbrev.get(start_month, start_month[:3])
                end_abbrev = month_abbrev.get(end_month, end_month[:3])

                if start_abbrev == end_abbrev:
                    ranges.append(start_abbrev)
                else:
                    ranges.append(f"{start_abbrev}-{end_abbrev}")

            return ", ".join(ranges)

        return ""

    def _remove_from_target_list(self):
        """Remove this DSO from the target list"""
        try:
            reply = QMessageBox.question(
                self, "Remove from Target List",
                f"Remove '{self.data.get('name', 'this DSO')}' from the target list?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')

                if not dso_name:
                    QMessageBox.warning(self, "Error", "Cannot remove: DSO name not found")
                    return

                cursor.execute("""
                    DELETE FROM usertargetlist WHERE UPPER(TRIM(name)) = ?
                """, (dso_name.upper(),))

                if ra_deg is not None and dec_deg is not None:
                    cursor.execute("""
                        DELETE FROM usertargetlist
                        WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001
                    """, (ra_deg, dec_deg))

                conn.commit()

                self._update_target_list_buttons()

                QMessageBox.information(self, "Success", f"'{dso_name}' removed from target list")
                logger.debug(f"Removed {dso_name} from target list")

        except Exception as e:
            logger.error(f"Error removing from target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove from target list: {str(e)}")

    def _open_from_target_list(self):
        """Open DSO from target list to view notes"""
        try:
            dso_name = self.data.get('name', '').strip()
            if not dso_name:
                QMessageBox.warning(self, "Error", "DSO name not found")
                return

            target_name = self._find_target_list_name()
            if not target_name:
                QMessageBox.warning(self, "Not Found", f"Could not find '{dso_name}' in your target list.")
                return

            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, 'target_list_window') or not self.target_list_window.isVisible():
                self.target_list_window = DSOTargetListWindow()

            success = self.target_list_window.open_and_select_target(target_name)

            if not success:
                QMessageBox.warning(self, "Not Found", f"Could not find '{target_name}' in your target list.")

        except Exception as e:
            logger.error(f"Error opening from target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open from target list: {str(e)}")

    def closeEvent(self, event):
        """Save window position when closing"""
        WindowPositionManager.save_window_position(self, "ObjectDetail")
        event.accept()
