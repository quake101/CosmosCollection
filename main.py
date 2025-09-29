import logging
import os
import sys
from typing import Optional, Dict

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QUrl, Signal, QObject, QTimer, QEvent, QThread
from PySide6.QtGui import QPixmap, QPainter, QIcon, QColor, QBrush
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView,
    QVBoxLayout, QWidget, QLabel, QDialog,
    QHeaderView, QPushButton, QHBoxLayout, QLineEdit, QComboBox, QTextEdit, QCheckBox, QGroupBox,
    QToolBar, QMessageBox, QMenu, QScrollArea
)
from PySide6.QtGui import QAction
from astroplan import Observer, FixedTarget
from astropy import units as u
from astropy.coordinates import SkyCoord
# Astropy and astroplan imports for visibility calculations
from astropy.time import Time

# Import DatabaseManager and ResourceManager
from DatabaseManager import DatabaseManager
from ResourceManager import ResourceManager
from CollageBuilder import CollageBuilder, CollageBuilderWindow

# Import the DSO Visibility Calculator
try:
    from DSOVisibilityCalculator import DSOVisibilityApp
    VISIBILITY_AVAILABLE = True
except ImportError:
    VISIBILITY_AVAILABLE = False
    logging.warning("DSOVisibilityCalculator.py not found. Visibility calculator will be disabled.")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the application directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))


class ImageCache:
    _instance = None
    _cache: Dict[str, QPixmap] = {}
    _max_size = 10  # Maximum number of images to cache

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
            # Remove the oldest item
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = pixmap

    def clear(self):
        self._cache.clear()


class ObjectDetailWorker(QObject):
    finished = Signal(dict)  # Signal with parsed result
    error = Signal(str)  # New signal for error reporting

    def __init__(self, name: str, ra_str: str, dec_str: str):
        super().__init__()
        self.name = name
        self.ra_str = ra_str
        self.dec_str = dec_str
        logger.debug(f"Initialized worker for {name} with RA: {ra_str}, Dec: {dec_str}")

    def parse_and_emit(self):
        """Parse RA/Dec in background thread and emit result"""
        try:
            logger.debug("Starting coordinate parsing")
            # Parse RA
            ra_deg = self._parse_ra()
            logger.debug(f"Parsed RA: {ra_deg} degrees")

            # Parse Dec
            dec_deg = self._parse_dec()
            logger.debug(f"Parsed Dec: {dec_deg} degrees")

            base_url = "http://cdsweb.u-strasbg.fr/cgi-bin/DSS/dss2/preview?"
            image_url = f"{base_url}ra={ra_deg:.6f}&dec={dec_deg:.6f}&width=600&height=400"
            logger.debug(f"Generated image URL: {image_url}")

            result = {
                "name": self.name,
                "image_url": image_url,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg
            }
            logger.debug("About to emit finished signal")
            self.finished.emit(result)
            logger.debug("Finished signal emitted")
        except Exception as e:
            logger.error(f"Error in parse_and_emit: {str(e)}", exc_info=True)
            self.error.emit(f"Error processing coordinates: {str(e)}")

    def _parse_ra(self) -> float:
        """Parse Right Ascension from hms format to degrees"""
        try:
            # Remove any whitespace and convert to standard format
            ra_clean = self.ra_str.strip().replace(" ", "")
            h, m, s = map(float, ra_clean.replace("h", ":").replace("m", ":").replace("s", "").split(":"))

            # Validate ranges
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                raise ValueError("RA values out of valid range")

            return 15 * (h + m / 60.0 + s / 3600.0)
        except Exception as e:
            raise ValueError(f"Invalid RA format: {self.ra_str} - {str(e)}")

    def _parse_dec(self) -> float:
        """Parse Declination from dms format to degrees"""
        try:
            # Remove any whitespace and convert to standard format
            dec_clean = self.dec_str.strip().replace(" ", "")

            # Extract sign
            sign = -1 if '-' in dec_clean else 1
            dec_clean = dec_clean.replace('+', '').replace('-', '')

            # Split into components
            parts = dec_clean.replace("°", ":").replace("'", ":").replace('"', "").split(":")
            if len(parts) != 3:
                raise ValueError("Invalid Dec format")

            deg, arcmin, arcsec = map(float, parts)

            # Validate ranges
            if not (0 <= deg <= 90 and 0 <= arcmin < 60 and 0 <= arcsec < 60):
                raise ValueError("Dec values out of valid range")

            return sign * (deg + arcmin / 60.0 + arcsec / 3600.0)
        except Exception as e:
            raise ValueError(f"Invalid Dec format: {self.dec_str} - {str(e)}")


class VisibilityCalculationWorker(QObject):
    """
    Worker for performing heavy visibility calculations in a background thread.
    
    Uses coordinate-based calculations to avoid issues with object name resolution
    (e.g., 'sh2 142' vs 'sh2-142' naming variations in astronomical databases).
    """
    finished = Signal(str)  # Signal with visibility text result
    error = Signal(str)  # Signal for error reporting

    def __init__(self, lat: float, lon: float, ra_deg: float, dec_deg: float, object_name: str):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.object_name = object_name
        
        # Import the centralized calculator
        try:
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            self.calculator = DSOVisibilityCalculator(lat, lon)
        except ImportError:
            self.calculator = None

    def calculate_visibility(self):
        """Calculate visibility seasons in background thread"""
        try:
            if self.calculator is None:
                self.error.emit("Visibility calculator not available. Please ensure DSOVisibilityCalculator.py is properly installed.")
                return
            
            # Create DSO coordinate
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            dso_coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
            
            # Use a more thorough seasonal visibility check that matches Best DSO Tonight logic
            # Check multiple nights throughout the year using the same method as Best DSO Tonight
            seasons = []
            from datetime import datetime, timedelta
            import numpy as np
            
            current_year = datetime.now().year
            min_altitude = 30  # Use 30° minimum altitude for seasonal visibility
            
            # Sample dates throughout the year (every 15 days for better coverage)
            sample_dates = []
            visibility_results = []
            
            # Import required libraries once outside the loop
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            # Create coordinate object once from stored RA/Dec
            dso_coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
            
            for day_offset in range(0, 365, 15):
                try:
                    test_date = datetime(current_year, 1, 1) + timedelta(days=day_offset)
                    date_str = test_date.strftime('%Y-%m-%d')
                    
                    # Use coordinate-based calculation
                    time_range, dso_altaz, sun_altaz = self.calculator.calculate_altaz_over_time(
                        dso_coord, date_str, 12)
                    
                    # Find optimal viewing times using same criteria
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
            
            # Group consecutive visible periods into seasons
            if any(visibility_results):
                season_strs = []
                in_season = False
                season_start = None
                
                for i, (date, visible) in enumerate(zip(sample_dates, visibility_results)):
                    if visible and not in_season:
                        # Start of a visible season
                        season_start = date
                        in_season = True
                    elif not visible and in_season:
                        # End of a visible season
                        if season_start:
                            season_strs.append(f"{season_start.strftime('%B %d')} - {sample_dates[i-1].strftime('%B %d')}")
                        in_season = False
                    elif i == len(sample_dates) - 1 and in_season:
                        # Season extends to end of year
                        if season_start:
                            season_strs.append(f"{season_start.strftime('%B %d')} - {date.strftime('%B %d')}")
                
                if season_strs:
                    visibility_text = f"Best viewing seasons (>30° altitude in dark sky):<br>" + "<br>".join(season_strs)
                    visibility_text += "<br><br><small>Times shown are when object is well-positioned in dark sky.<br>Use Visibility Calculator for detailed nightly times.</small>"
                else:
                    visibility_text = "Object not optimally visible from your location this year.<br>Try checking the Visibility Calculator for detailed viewing times."
            else:
                visibility_text = "Object not optimally visible from your location this year.<br>Try checking the Visibility Calculator for detailed viewing times."

            self.finished.emit(visibility_text)
        except Exception as e:
            logger.error(f"Error calculating visibility: {str(e)}", exc_info=True)
            self.error.emit(f"Error calculating viewing season information:<br>{str(e)}")


# --- Model for displaying DSO data in table ---
class DSOTableModel(QAbstractTableModel):
    def __init__(self, dso_data, parent=None):
        super().__init__(parent)
        self.dso_data = dso_data
        self.filtered_data = dso_data.copy()  # For filtering
        self.headers = ["Catalog", "Designation", "RA (hms)", "Dec (dms)", "Images"]
        self.selected_catalog = None
        self.highlight_no_images = False
        self._cached_formatted_data = {}  # Cache for formatted data

    def rowCount(self, index=QModelIndex()):
        return len(self.filtered_data)

    def columnCount(self, index=QModelIndex()):
        return 5

    def data(self, index, role):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        entry = self.filtered_data[row]

        if role == Qt.ItemDataRole.BackgroundRole:
            if self.highlight_no_images and entry["image_count"] == 0:
                return QBrush(QColor(233, 94, 70, 128))
            elif row % 2 == 1:
                return QBrush(QColor(61, 61, 61))
            return QBrush(QColor(45, 45, 45))
        elif role == Qt.ItemDataRole.DisplayRole:
            cache_key = f"{row}_{col}"
            if cache_key in self._cached_formatted_data:
                return self._cached_formatted_data[cache_key]

            result = self._format_cell_data(entry, col)
            self._cached_formatted_data[cache_key] = result
            return result
        return None

    def _format_cell_data(self, entry, col):
        """Format cell data with caching"""
        if col == 0:
            if self.selected_catalog and self.selected_catalog != "All Catalogs":
                return self.selected_catalog
            return entry["catalogue"]
        elif col == 1:
            designations = entry["designations"].split(", ")
            if self.selected_catalog and self.selected_catalog != "All Catalogs":
                for designation in designations:
                    if designation.startswith(self.selected_catalog + " "):
                        return designation.split(" ", 1)[1]
            return entry["id"]
        elif col == 2:
            return self._format_ra(entry["ra_deg"])
        elif col == 3:
            return self._format_dec(entry["dec_deg"])
        elif col == 4:
            return str(entry["image_count"])
        return None

    def headerData(self, index, orientation, role):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self.headers[index]

    def sort(self, column, order):
        """Sort the data by the specified column"""
        self.layoutAboutToBeChanged.emit()

        # Get the sort key function based on the column
        if column == 0:  # Catalog
            key_func = lambda x: x["catalogue"]
        elif column == 1:  # Designation
            key_func = lambda x: x["id"]
        elif column == 2:  # RA
            key_func = lambda x: x["ra_deg"]
        elif column == 3:  # Dec
            key_func = lambda x: x["dec_deg"]
        elif column == 4:  # Images
            key_func = lambda x: x["image_count"]
        else:
            return

        # Sort the data
        self.filtered_data.sort(key=key_func, reverse=(order == Qt.DescendingOrder))

        # Clear the cache when data changes
        self._cached_formatted_data.clear()

        self.layoutChanged.emit()

    def filter_data(self, search_text, selected_catalog=None, show_images_only=False, selected_type=None):
        """Filter the data based on search text, catalog, image presence, and DSO type"""
        self.layoutAboutToBeChanged.emit()

        # Store the selected catalog for use in data() method
        self.selected_catalog = selected_catalog

        if not search_text and not selected_catalog and not show_images_only and not selected_type:
            self.filtered_data = self.dso_data.copy()
        else:
            search_text = search_text.lower() if search_text else ""
            self.filtered_data = [
                item for item in self.dso_data
                if ((not selected_catalog or
                     selected_catalog == "All Catalogs" or
                     any(designation.startswith(selected_catalog + " ")
                         for designation in item["designations"].split(", "))) and
                    (not selected_type or
                     selected_type == "All Types" or
                     item.get("dso_type", "") == selected_type) and
                    (not show_images_only or item["image_count"] > 0) and
                    (not search_text or
                     search_text in item["catalogue"].lower() or
                     search_text in item["id"].lower() or
                     self._format_ra(item["ra_deg"]).lower() in search_text or
                     self._format_dec(item["dec_deg"]).lower() in search_text or
                     search_text in item["designations"].lower()))
            ]

        # Clear the cache when data changes
        self._cached_formatted_data.clear()

        self.layoutChanged.emit()

    def _format_ra(self, ra_deg):
        """Convert RA in degrees to hms format"""
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_remaining = (ra_hours - ra_h) * 60
        ra_m = int(ra_remaining)
        ra_s = (ra_remaining - ra_m) * 60
        return f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s"

    def _format_dec(self, dec_deg):
        """Convert Dec in degrees to dms format"""
        dec_sign = '-' if dec_deg < 0 else '+'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_remaining = (dec_abs - dec_d) * 60
        dec_m = int(dec_remaining)
        dec_s = (dec_remaining - dec_m) * 60
        return f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""

    def setHighlightNoImages(self, highlight):
        """Set whether to highlight objects without images"""
        self.highlight_no_images = highlight
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))


# --- Custom Visibility Window Class ---
class CustomDSOVisibilityWindow(QDialog):
    """Custom wrapper for the DSO Visibility Calculator"""

    def __init__(self, dso_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{dso_name} - DSO Visibility Calculator - Cosmos Collection")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1200, 800)

        # Create layout
        layout = QVBoxLayout()

        # Create the visibility app widget
        self.visibility_app = DSOVisibilityApp()

        # Pre-populate with the DSO name
        self.visibility_app.dso_input.setText(dso_name)

        # Remove the window frame from the visibility app and add its central widget
        central_widget = self.visibility_app.centralWidget()
        layout.addWidget(central_widget)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)

        # Calculate visibility immediately
        QTimer.singleShot(100, self.visibility_app.calculate_visibility)


# --- Aladin Lite Viewer Window ---
class AladinLiteWindow(QMainWindow):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{data['name']} - Aladin Lite - Cosmos Collection")
        self.resize(1200, 800)
        
        self.data = data
        self.telescopes = []
        self.selected_telescope = None
        self.current_fov = None
        self.current_target = None  # Track current target to preserve user changes

        # Calculate default FOV based on object size (in degrees)
        try:
            size_min = data.get('size_min', 30) or 30  # Default to 30 arcminutes if None
            size_max = data.get('size_max', 30) or 30  # Default to 30 arcminutes if None
            size_max = max(size_min, size_max)  # arcminutes
            self.default_fov = max(size_max / 60.0 * 3.0, 0.5)  # Convert to degrees, 3x object size, min 0.5°
            self.current_fov = self.default_fov

            logger.debug(f"Size values for {data['name']}: min={size_min:.1f}', max={size_max:.1f}'")
            logger.debug(f"Calculated default FOV: {self.default_fov:.3f}°")
        except Exception as e:
            logger.warning(f"Error calculating FOV for {data.get('name', 'Unknown')}: {e}")
            self.default_fov = 1.0  # Safe fallback
            self.current_fov = 1.0

        # Create main layout
        layout = QVBoxLayout()
        
        # Create telescope controls layout
        telescope_layout = QHBoxLayout()
        
        # Telescope selection
        telescope_label = QLabel("Telescope:")
        self.telescope_combo = QComboBox()
        self.telescope_combo.addItem("Default View", None)
        self.telescope_combo.currentTextChanged.connect(self._on_telescope_changed)
        
        # Load telescopes
        self._load_telescopes()
        
        # FOV display controls
        self.show_telescope_fov = QCheckBox("Show Telescope FOV")
        self.show_telescope_fov.setChecked(False)
        self.show_telescope_fov.toggled.connect(self._update_aladin_view)
        
        # Camera/Eyepiece selection (for different FOVs)
        camera_label = QLabel("Camera/Eyepiece:")
        self.camera_combo = QComboBox()
        
        # Visual eyepieces with typical apparent FOV values
        self.camera_combo.addItem("--- EYEPIECES ---", None)
        self.camera_combo.addItem("32mm Eyepiece (52° AFOV)", {"type": "eyepiece", "focal_length": 32, "apparent_fov": 52})
        self.camera_combo.addItem("25mm Eyepiece (52° AFOV)", {"type": "eyepiece", "focal_length": 25, "apparent_fov": 52})
        self.camera_combo.addItem("20mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 20, "apparent_fov": 50})
        self.camera_combo.addItem("15mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 15, "apparent_fov": 50})
        self.camera_combo.addItem("10mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 10, "apparent_fov": 50})
        self.camera_combo.addItem("6mm Eyepiece (50° AFOV)", {"type": "eyepiece", "focal_length": 6, "apparent_fov": 50})
        
        # DSLR cameras
        self.camera_combo.addItem("--- DSLR CAMERAS ---", None)
        self.camera_combo.addItem("Canon Full Frame (36x24mm)", {"type": "camera", "sensor_width": 36, "sensor_height": 24})
        self.camera_combo.addItem("Canon APS-C (22.3x14.9mm)", {"type": "camera", "sensor_width": 22.3, "sensor_height": 14.9})
        self.camera_combo.addItem("Canon APS-H (28.7x19mm)", {"type": "camera", "sensor_width": 28.7, "sensor_height": 19.0})
        self.camera_combo.addItem("Nikon Full Frame (35.9x24mm)", {"type": "camera", "sensor_width": 35.9, "sensor_height": 24.0})
        self.camera_combo.addItem("Nikon APS-C (23.5x15.6mm)", {"type": "camera", "sensor_width": 23.5, "sensor_height": 15.6})
        self.camera_combo.addItem("Sony Full Frame (35.8x23.8mm)", {"type": "camera", "sensor_width": 35.8, "sensor_height": 23.8})
        self.camera_combo.addItem("Sony APS-C (23.5x15.6mm)", {"type": "camera", "sensor_width": 23.5, "sensor_height": 15.6})
        
        # ZWO ASI cameras (popular for astrophotography)
        self.camera_combo.addItem("--- ZWO ASI CAMERAS ---", None)
        self.camera_combo.addItem("ASI6200MM Pro (36x24mm)", {"type": "camera", "sensor_width": 36.0, "sensor_height": 24.0})
        self.camera_combo.addItem("ASI2600MM Pro (23.5x15.7mm)", {"type": "camera", "sensor_width": 23.5, "sensor_height": 15.7})
        self.camera_combo.addItem("ASI533MM Pro (11.3x7.1mm)", {"type": "camera", "sensor_width": 11.3, "sensor_height": 7.1})
        self.camera_combo.addItem("ASI294MM Pro (19.1x13.0mm)", {"type": "camera", "sensor_width": 19.1, "sensor_height": 13.0})
        self.camera_combo.addItem("ASI183MM Pro (13.2x8.8mm)", {"type": "camera", "sensor_width": 13.2, "sensor_height": 8.8})
        self.camera_combo.addItem("ASI585MC (8.3x6.2mm)", {"type": "camera", "sensor_width": 8.3, "sensor_height": 6.2})
        self.camera_combo.addItem("ASI662MC (7.4x5.6mm) (Seestar S30)", {"type": "camera", "sensor_width": 7.4, "sensor_height": 5.6})
        self.camera_combo.addItem("ASI385MC (7.7x4.9mm)", {"type": "camera", "sensor_width": 7.7, "sensor_height": 4.9})
        self.camera_combo.addItem("ASI462MC (2.9x2.9mm) (Seestar S50)", {"type": "camera", "sensor_width": 2.9, "sensor_height": 2.9})
        self.camera_combo.addItem("ASI224MC (3.9x2.8mm)", {"type": "camera", "sensor_width": 3.9, "sensor_height": 2.8})
        self.camera_combo.addItem("ASI120MM (3.8x2.8mm)", {"type": "camera", "sensor_width": 3.8, "sensor_height": 2.8})
        
        # QHY cameras
        self.camera_combo.addItem("--- QHY CAMERAS ---", None)
        self.camera_combo.addItem("QHY600M (36x24mm)", {"type": "camera", "sensor_width": 36.0, "sensor_height": 24.0})
        self.camera_combo.addItem("QHY268M (23.5x15.7mm)", {"type": "camera", "sensor_width": 23.5, "sensor_height": 15.7})
        self.camera_combo.addItem("QHY294M (19.1x13.0mm)", {"type": "camera", "sensor_width": 19.1, "sensor_height": 13.0})
        self.camera_combo.addItem("QHY183M (13.2x8.8mm)", {"type": "camera", "sensor_width": 13.2, "sensor_height": 8.8})
        self.camera_combo.addItem("QHY174M (11.3x7.1mm)", {"type": "camera", "sensor_width": 11.3, "sensor_height": 7.1})
        
        # SBIG cameras
        self.camera_combo.addItem("--- SBIG CAMERAS ---", None)
        self.camera_combo.addItem("SBIG STF-8300M (17.96x13.52mm)", {"type": "camera", "sensor_width": 17.96, "sensor_height": 13.52})
        self.camera_combo.addItem("SBIG ST-2000XM (15.2x15.2mm)", {"type": "camera", "sensor_width": 15.2, "sensor_height": 15.2})
        
        # Atik cameras
        self.camera_combo.addItem("--- ATIK CAMERAS ---", None)
        self.camera_combo.addItem("Atik 460EX (36x24mm)", {"type": "camera", "sensor_width": 36.0, "sensor_height": 24.0})
        self.camera_combo.addItem("Atik 383L+ (23.6x15.8mm)", {"type": "camera", "sensor_width": 23.6, "sensor_height": 15.8})
        
        self.camera_combo.currentTextChanged.connect(self._on_camera_changed)

        # Barlow/Reducer selection
        barlow_label = QLabel("Barlow/Reducer:")
        self.barlow_combo = QComboBox()

        # Optical accessories
        self.barlow_combo.addItem("None (1.0x)", {"factor": 1.0, "type": "none"})
        self.barlow_combo.addItem("--- BARLOWS ---", None)
        self.barlow_combo.addItem("1.25x Barlow", {"factor": 1.25, "type": "barlow"})
        self.barlow_combo.addItem("1.5x Barlow", {"factor": 1.5, "type": "barlow"})
        self.barlow_combo.addItem("2x Barlow", {"factor": 2.0, "type": "barlow"})
        self.barlow_combo.addItem("2.5x Barlow", {"factor": 2.5, "type": "barlow"})
        self.barlow_combo.addItem("3x Barlow", {"factor": 3.0, "type": "barlow"})
        self.barlow_combo.addItem("4x Barlow", {"factor": 4.0, "type": "barlow"})
        self.barlow_combo.addItem("5x Barlow", {"factor": 5.0, "type": "barlow"})
        self.barlow_combo.addItem("--- REDUCERS ---", None)
        self.barlow_combo.addItem("0.5x Reducer", {"factor": 0.5, "type": "reducer"})
        self.barlow_combo.addItem("0.6x Reducer", {"factor": 0.6, "type": "reducer"})
        self.barlow_combo.addItem("0.63x Reducer", {"factor": 0.63, "type": "reducer"})
        self.barlow_combo.addItem("0.67x Reducer", {"factor": 0.67, "type": "reducer"})
        self.barlow_combo.addItem("0.7x Reducer", {"factor": 0.7, "type": "reducer"})
        self.barlow_combo.addItem("0.75x Reducer", {"factor": 0.75, "type": "reducer"})
        self.barlow_combo.addItem("0.8x Reducer", {"factor": 0.8, "type": "reducer"})

        self.barlow_combo.currentTextChanged.connect(self._on_barlow_changed)

        # Arrange telescope controls
        telescope_layout.addWidget(telescope_label)
        telescope_layout.addWidget(self.telescope_combo)
        telescope_layout.addWidget(self.show_telescope_fov)
        telescope_layout.addWidget(camera_label)
        telescope_layout.addWidget(self.camera_combo)
        telescope_layout.addWidget(barlow_label)
        telescope_layout.addWidget(self.barlow_combo)
        telescope_layout.addStretch()
        
        layout.addLayout(telescope_layout)

        # Initialize web view as None initially - we'll create it safely later
        self.web_view = None
        self.web_view_error = None

        # Create a placeholder widget for the web view
        self.web_placeholder = QLabel("Loading Aladin Lite...")
        self.web_placeholder.setAlignment(Qt.AlignCenter)
        self.web_placeholder.setStyleSheet("QLabel { background-color: #2b2b2b; color: white; font-size: 14px; }")
        self.web_placeholder.setMinimumSize(400, 300)

        layout.addWidget(self.web_placeholder)

        # Create a horizontal layout for the bottom controls
        bottom_layout = QHBoxLayout()

        # Add FOV information display
        self.fov_info_label = QLabel()
        self.fov_info_label.setStyleSheet("font-size: 10pt;")
        bottom_layout.addWidget(self.fov_info_label)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom_layout.addWidget(close_button)

        # Add the bottom layout to the main layout
        layout.addLayout(bottom_layout)

        # Create central widget and set layout for QMainWindow
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Initialize overlay data storage
        self.pending_fov_overlay = None
        self.target_coordinates = None
        self.fallback_button = None  # Track fallback button to avoid duplicates
        
        # Add loading timeout
        self.loading_timeout = QTimer()
        self.loading_timeout.timeout.connect(self._handle_loading_timeout)
        self.loading_timeout.setSingleShot(True)

        # Defer web view creation to avoid initialization crashes
        QTimer.singleShot(100, self._create_web_view_safely)

        logger.debug(f"Opened Aladin Lite window with default FOV: {self.default_fov:.2f}'")

    def _create_web_view_safely(self):
        """Safely create the web view with error handling"""
        try:
            logger.debug("Creating web view safely...")

            # Try to create the web view
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
                from PySide6.QtCore import QUrl
            except ImportError as ie:
                raise Exception(f"QWebEngineView not available: {ie}")

            self.web_view = QWebEngineView()
            self.web_view.setMinimumSize(400, 300)

            # Add load progress and error handling
            self.web_view.loadStarted.connect(self._on_load_started)
            self.web_view.loadProgress.connect(self._on_load_progress)
            self.web_view.loadFinished.connect(self._on_load_finished)

            # Enable developer tools for debugging (optional)
            try:
                from PySide6.QtWebEngineCore import QWebEngineSettings
                # Try different attribute names depending on PySide6 version
                try:
                    self.web_view.settings().setAttribute(QWebEngineSettings.WebAttribute.DeveloperExtrasEnabled, True)
                except AttributeError:
                    try:
                        self.web_view.settings().setAttribute(QWebEngineSettings.DeveloperExtrasEnabled, True)
                    except AttributeError:
                        # Alternative approach for older versions
                        settings = self.web_view.settings()
                        settings.setAttribute(settings.DeveloperExtrasEnabled, True)

                # Enable context menu for developer tools
                self.web_view.setContextMenuPolicy(Qt.DefaultContextMenu)
                logger.debug("Developer tools enabled for Aladin window")
            except Exception as e:
                logger.debug(f"Could not enable developer tools: {e}")
                # Continue without developer tools

            # Replace the placeholder with the actual web view
            central_widget = self.centralWidget()
            if central_widget and central_widget.layout() and self.web_placeholder:
                layout = central_widget.layout()
                # Find the placeholder in the layout and replace it
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget() == self.web_placeholder:
                        # Remove placeholder
                        layout.removeWidget(self.web_placeholder)
                        self.web_placeholder.hide()
                        self.web_placeholder.deleteLater()
                        self.web_placeholder = None

                        # Add web view in the same position
                        layout.insertWidget(i, self.web_view)
                        self.web_view.show()
                        logger.debug("Replaced placeholder with web view in layout")
                        break

            # Now that web view is created, load Aladin
            self._update_aladin_view(preserve_target=False)
            logger.debug("Web view created successfully")

        except Exception as e:
            logger.error(f"Failed to create web view safely: {e}")
            self.web_view_error = str(e)

            # Update placeholder to show error and offer browser fallback
            if self.web_placeholder:
                self.web_placeholder.setText(f"Failed to load Aladin Lite\nError: {str(e)}\n\nClick below to open in browser instead.")
                self.web_placeholder.setStyleSheet("QLabel { background-color: #2b2b2b; color: #ff6b6b; font-size: 12px; }")

                # Add a button to open in browser as fallback
                self._add_browser_fallback_button()

    def _add_browser_fallback_button(self):
        """Add a button to open Aladin Lite in the default browser"""
        try:
            # Don't add button if it already exists
            if self.fallback_button is not None:
                return

            # Find the central widget and its layout
            central_widget = self.centralWidget()
            if central_widget and central_widget.layout():
                main_layout = central_widget.layout()

                # Create a fallback button
                self.fallback_button = QPushButton("Open Aladin Lite in Browser")
                self.fallback_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; margin: 10px; padding: 8px; }")
                self.fallback_button.clicked.connect(self._open_in_browser)

                # Insert before the bottom controls (last item should be the bottom layout)
                main_layout.insertWidget(main_layout.count() - 1, self.fallback_button)
                logger.debug("Added browser fallback button")
        except Exception as e:
            logger.error(f"Failed to add browser fallback button: {e}")

    def _open_in_browser(self):
        """Open Aladin Lite in the default browser"""
        try:
            import webbrowser

            # Build the same URL we would use in the web view
            ra = self.data.get('ra_deg', 0)
            dec = self.data.get('dec_deg', 0)
            target_id = f"{ra} {dec}" if ra and dec else self.data.get('name', 'M1')

            url_params = [
                f"target={target_id}",
                f"fov={self.default_fov}",
                "survey=P%2FDSS2%2Fcolor",
                "showReticle=true"
            ]

            base_url = "https://aladin.u-strasbg.fr/AladinLite/?"
            browser_url = f"{base_url}{'&'.join(url_params)}"

            logger.debug(f"Opening Aladin Lite in browser: {browser_url}")
            webbrowser.open(browser_url)

            # Show a message to the user
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Opened in Browser",
                                  f"Aladin Lite has been opened in your default browser for {self.data.get('name', 'the selected object')}.")

        except Exception as e:
            logger.error(f"Failed to open Aladin Lite in browser: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to open Aladin Lite in browser: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event with proper cleanup"""
        try:
            logger.debug("Cleaning up Aladin Lite window")
            # Stop any pending JavaScript operations
            if hasattr(self, 'web_view') and self.web_view:
                self.web_view.stop()
                # Clear the web view content
                self.web_view.setHtml("")
            event.accept()
        except Exception as e:
            logger.warning(f"Error during Aladin window cleanup: {e}")
            event.accept()  # Always accept to prevent hanging

    def _load_telescopes(self):
        """Load user telescopes from database"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length, is_active 
                    FROM usertelescopes 
                    WHERE focal_length IS NOT NULL AND focal_length > 0
                    ORDER BY is_active DESC, name ASC
                """)
                
                telescopes = cursor.fetchall()
                self.telescopes = []
                
                for telescope_id, name, aperture, focal_length, is_active in telescopes:
                    telescope_data = {
                        'id': telescope_id,
                        'name': name,
                        'aperture': aperture,
                        'focal_length': focal_length,
                        'is_active': is_active
                    }
                    self.telescopes.append(telescope_data)
                    
                    # Add to combo box
                    display_name = f"{name} ({focal_length}mm f/{focal_length/aperture:.1f})" if aperture else f"{name} ({focal_length}mm)"
                    if is_active:
                        display_name += " *"
                    self.telescope_combo.addItem(display_name, telescope_data)
                
                logger.debug(f"Loaded {len(telescopes)} telescopes with focal length data")
                
        except Exception as e:
            logger.error(f"Error loading telescopes: {str(e)}")
    
    def _on_telescope_changed(self):
        """Handle telescope selection change"""
        current_data = self.telescope_combo.currentData()
        if current_data:
            self.selected_telescope = current_data
            logger.debug(f"Selected telescope: {current_data['name']} ({current_data['focal_length']}mm)")
        else:
            self.selected_telescope = None
            logger.debug("Selected default view")
        
        self._update_aladin_view()
    
    def _on_camera_changed(self):
        """Handle camera/sensor selection change"""
        self._update_aladin_view()

    def _on_barlow_changed(self):
        """Handle barlow/reducer selection change"""
        self._update_aladin_view()

    def _calculate_telescope_fov(self):
        """Calculate telescope FOV based on selected telescope and camera/eyepiece"""
        if not self.selected_telescope:
            return None
        
        telescope_fl = self.selected_telescope['focal_length']  # mm
        telescope_aperture = self.selected_telescope.get('aperture', 100)  # mm

        # Get barlow/reducer factor
        barlow_data = self.barlow_combo.currentData()
        barlow_factor = 1.0  # Default no change
        if barlow_data and 'factor' in barlow_data:
            barlow_factor = barlow_data['factor']

        # Apply barlow/reducer to effective focal length
        effective_fl = telescope_fl * barlow_factor

        camera_data = self.camera_combo.currentData()

        if not camera_data or camera_data is None:
            return None
        
        import math
        
        if camera_data.get('type') == 'eyepiece':
            # Visual observation with eyepiece
            eyepiece_fl = camera_data['focal_length']  # mm
            apparent_fov = camera_data['apparent_fov']  # degrees
            
            # Calculate magnification using effective focal length
            magnification = effective_fl / eyepiece_fl

            # True FOV = Apparent FOV / Magnification
            true_fov_deg = apparent_fov / magnification
            true_fov_arcmin = true_fov_deg * 60

            barlow_text = f" with {barlow_factor}x" if barlow_factor != 1.0 else ""
            logger.debug(f"Eyepiece FOV calculation: {eyepiece_fl}mm eyepiece{barlow_text}, {apparent_fov}° AFOV, {magnification:.1f}x mag, {true_fov_arcmin:.1f}' true FOV")
            
            barlow_details = f" + {barlow_factor}x" if barlow_factor != 1.0 else ""
            return {
                'width_arcmin': true_fov_arcmin,
                'height_arcmin': true_fov_arcmin,
                'type': 'visual',
                'details': f"{eyepiece_fl}mm eyepiece{barlow_details}, {magnification:.0f}x mag"
            }
        
        elif camera_data.get('type') == 'camera':
            # Camera sensor
            sensor_width = camera_data['sensor_width']  # mm
            sensor_height = camera_data['sensor_height']  # mm
            
            # FOV = 2 * arctan(sensor_size / (2 * effective_focal_length)) * (180/π) * 60 (arcmin)
            fov_width_rad = 2 * math.atan(sensor_width / (2 * effective_fl))
            fov_height_rad = 2 * math.atan(sensor_height / (2 * effective_fl))

            fov_width_arcmin = fov_width_rad * (180 / math.pi) * 60
            fov_height_arcmin = fov_height_rad * (180 / math.pi) * 60

            # Calculate pixel scale for additional info using effective focal length
            pixel_scale_arcsec = 206265 * (sensor_width / 1000) / effective_fl  # arcsec/mm (assuming square pixels)

            barlow_text = f" with {barlow_factor}x" if barlow_factor != 1.0 else ""
            logger.debug(f"Camera FOV calculation: {sensor_width}x{sensor_height}mm sensor, {effective_fl}mm effective FL{barlow_text}, FOV={fov_width_arcmin:.1f}'x{fov_height_arcmin:.1f}'")
            
            barlow_details = f" + {barlow_factor}x" if barlow_factor != 1.0 else ""
            return {
                'width_arcmin': fov_width_arcmin,
                'height_arcmin': fov_height_arcmin,
                'type': 'camera',
                'details': f"{sensor_width}×{sensor_height}mm sensor{barlow_details}",
                'pixel_scale_arcsec': pixel_scale_arcsec
            }
        
        return None
    
    def _update_aladin_view(self, preserve_target=True):
        """Update the Aladin Lite view with current settings

        Args:
            preserve_target: If True, preserve current target when updating FOV overlays
        """
        # Check if web view is available
        if not self.web_view:
            logger.debug("Web view not yet created, skipping Aladin update")
            return
        # Determine FOV to use
        telescope_fov_data = None
        display_fov = self.default_fov
        
        if self.selected_telescope and self.show_telescope_fov.isChecked():
            telescope_fov_data = self._calculate_telescope_fov()
            if telescope_fov_data:
                # Use the larger dimension for display FOV, but convert to degrees and add reasonable margin
                telescope_fov_arcmin = max(telescope_fov_data['width_arcmin'], telescope_fov_data['height_arcmin'])
                display_fov = telescope_fov_arcmin / 60.0 * 1.5  # Convert to degrees and add 50% margin
                logger.debug(f"Telescope FOV: {telescope_fov_arcmin:.1f}' -> Display FOV: {display_fov:.3f}°")
        
        self.current_fov = display_fov
        
        # If preserving target and we already have a page loaded, just update the FOV overlay
        if preserve_target and self.current_target and hasattr(self, 'web_view') and self.web_view.url().toString():
            logger.debug("Preserving target - updating FOV overlay only")
            if telescope_fov_data and self.show_telescope_fov.isChecked():
                self.pending_fov_overlay = telescope_fov_data
                self.target_coordinates = self.current_target
                self._inject_fov_overlay(True)
            else:
                # Remove FOV overlay
                self._remove_fov_overlay()
            self._update_fov_info()
            return

        # Build Aladin URL for full page load
        base_url = "https://aladin.u-strasbg.fr/AladinLite/?"

        # Determine target to use
        target_id = None
        if preserve_target and self.current_target:
            # Use current target for URL
            target_id = self.current_target
            logger.debug(f"Using current target for URL: {target_id}")
        else:
            # Use original data to set initial target
            if 'ra_deg' in self.data and 'dec_deg' in self.data and self.data['ra_deg'] is not None and self.data['dec_deg'] is not None:
                ra = self.data['ra_deg']
                dec = self.data['dec_deg']
                # Format coordinates properly for Aladin (space-separated)
                target_id = f"{ra} {dec}"
                logger.debug(f"Using coordinates for Aladin target: RA={ra}, Dec={dec}")
            else:
                # Fallback to object names
                target_id = self.data.get('name', '')
                logger.debug(f"Using object name for Aladin target: {target_id}")

                # If still no target, try dsodetailid
                if not target_id:
                    target_id = self.data.get('dsodetailid', '')
                    logger.debug(f"Using dsodetailid for Aladin target: {target_id}")

            if not target_id:
                logger.error(f"No valid target found for Aladin. Data keys: {list(self.data.keys())}")
                target_id = "M1"  # Default fallback

            # Store the target
            self.current_target = target_id
        
        # URL encode the target if it contains coordinates
        import urllib.parse
        encoded_target = urllib.parse.quote(str(target_id))
        
        # Build URL with parameters
        url_params = [
            f"target={encoded_target}",
            f"fov={display_fov}",
            "survey=P%2FDSS2%2Fcolor",
            "showReticle=true"
        ]
        
        # Always use standard Aladin URL first
        image_url = f"{base_url}{'&'.join(url_params)}"
        logger.debug(f"Final Aladin URL: {image_url}")

        # Safely load the URL with error handling
        try:
            if hasattr(self, 'web_view') and self.web_view:
                logger.debug(f"Loading Aladin URL: {image_url}")
                self.web_view.setUrl(QUrl(image_url))

                # Test connectivity by trying a simple request first
                self._test_connectivity_async()
            else:
                logger.error("Web view not available for URL loading")
                raise Exception("Web view not available")
        except Exception as e:
            logger.error(f"Error loading Aladin URL: {e}")
            # Show error in placeholder
            if self.web_placeholder:
                self.web_placeholder.setText(f"Error loading Aladin Lite\n{str(e)}\n\nClick below to open in browser instead.")
                self.web_placeholder.setStyleSheet("QLabel { background-color: #2b2b2b; color: #ff6b6b; font-size: 12px; }")
            self._add_browser_fallback_button()
        
        # Add telescope FOV overlay using JavaScript injection if enabled
        if telescope_fov_data and self.show_telescope_fov.isChecked():
            logger.debug(f"Will inject FOV overlay. Telescope: {self.selected_telescope['name']}, FOV: {telescope_fov_data['width_arcmin']:.1f}'x{telescope_fov_data['height_arcmin']:.1f}', Type: {telescope_fov_data['type']}")
            # Store the FOV data for injection after page loads
            self.pending_fov_overlay = telescope_fov_data
            self.target_coordinates = target_id
            # The FOV overlay injection will be handled by the main loadFinished handler
        else:
            self.pending_fov_overlay = None
        
        self._update_fov_info()
        
        logger.debug(f"Updated Aladin view with FOV: {display_fov:.3f}° ({display_fov*60:.1f}')")
    
    def _create_aladin_html_with_overlay(self, target, fov, telescope_fov_data):
        """Create custom HTML with Aladin Lite and telescope FOV overlay"""
        
        # Convert FOV to degrees for JavaScript
        telescope_fov_width_deg = telescope_fov_data['width_arcmin'] / 60.0
        telescope_fov_height_deg = telescope_fov_data['height_arcmin'] / 60.0
        
        # Determine overlay shape and color
        if telescope_fov_data['type'] == 'visual':
            # Circular overlay for eyepieces
            overlay_shape = "circle"
            overlay_color = "#00ff00"  # Green for visual
            overlay_radius = telescope_fov_width_deg / 2.0
        else:
            # Rectangular overlay for cameras
            overlay_shape = "rectangle"
            overlay_color = "#ff8800"  # Orange for cameras
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aladin Lite with Telescope FOV</title>
    <link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" />
    <style>
        body {{ margin: 0; padding: 0; background: #1a1a1a; }}
        #aladin-lite-div {{ width: 100%; height: 100vh; }}
        .info-overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div id="aladin-lite-div"></div>
    <div class="info-overlay">
        <strong>Telescope FOV:</strong> {telescope_fov_data['width_arcmin']:.1f}' × {telescope_fov_data['height_arcmin']:.1f}'<br>
        <strong>Type:</strong> {telescope_fov_data['details']}
    </div>
    
    <script src="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js"></script>
    <script>
        // Initialize Aladin Lite
        A.init.then(() => {{
            let aladin = A.aladin('#aladin-lite-div', {{
                survey: "P/DSS2/color",
                fov: {fov},
                target: "{target}",
                showReticle: true,
                showZoomControl: true,
                showFullscreenControl: true,
                showLayersControl: true,
                showGotoControl: true,
                showProjectionControl: true,
                showFrame: true
            }});
            
            // Add FOV overlay after Aladin loads
            setTimeout(() => {{
                try {{
                    let overlay = A.graphicOverlay({{
                        color: '{overlay_color}',
                        lineWidth: 2
                    }});
                    aladin.addOverlay(overlay);
                    
                    // Parse target coordinates
                    let coords = "{target}".split(' ');
                    let ra = parseFloat(coords[0]);
                    let dec = parseFloat(coords[1]);
                    
                    if (!isNaN(ra) && !isNaN(dec)) {{
                        if ("{overlay_shape}" === "circle") {{
                            // Circular FOV for eyepieces
                            overlay.addShape(A.circle(ra, dec, {overlay_radius:.6f}));
                        }} else {{
                            // Rectangular FOV for cameras
                            let hw = {telescope_fov_width_deg / 2.0:.6f};
                            let hh = {telescope_fov_height_deg / 2.0:.6f};
                            let poly = A.polygon([
                                [ra - hw, dec - hh],
                                [ra + hw, dec - hh],
                                [ra + hw, dec + hh],
                                [ra - hw, dec + hh]
                            ]);
                            overlay.addShape(poly);
                        }}
                        console.log("FOV overlay added successfully");
                    }}
                }} catch(e) {{
                    console.error("Error adding FOV overlay:", e);
                }}
            }}, 1500);
        }}).catch(e => {{
            console.error("Aladin initialization failed:", e);
            document.getElementById('aladin-lite-div').innerHTML = '<div style="color:white;padding:20px;text-align:center;">Failed to load Aladin Lite. Please check your internet connection.</div>';
        }});
    </script>
</body>
</html>"""
        
        return html_template
    
    def _inject_fov_overlay(self, success):
        """Inject FOV overlay JavaScript after Aladin page loads"""
        if not success or not self.pending_fov_overlay or not self.web_view:
            return
            
        try:
            # Disconnect the signal to avoid multiple injections
            self.web_view.loadFinished.disconnect(self._inject_fov_overlay)
        except:
            pass  # Signal might not be connected
        
        fov_data = self.pending_fov_overlay
        target_coords = self.target_coordinates
        
        logger.debug(f"Injecting FOV overlay for {fov_data['type']} with {fov_data['width_arcmin']:.1f}' FOV")
        
        # Convert FOV to degrees for JavaScript
        telescope_fov_width_deg = fov_data['width_arcmin'] / 60.0
        telescope_fov_height_deg = fov_data['height_arcmin'] / 60.0
        
        # Determine overlay properties
        if fov_data['type'] == 'visual':
            overlay_shape = "circle"
            overlay_color = "#00ff00"  # Green for visual
            overlay_radius = telescope_fov_width_deg / 2.0
        else:
            overlay_shape = "rectangle"
            overlay_color = "#ff8800"  # Orange for cameras
        
        # Parse coordinates
        coords = target_coords.split(' ')
        if len(coords) >= 2:
            try:
                ra = float(coords[0])
                dec = float(coords[1])
            except ValueError:
                logger.error(f"Invalid coordinates: {target_coords}")
                return
        else:
            logger.error(f"Invalid coordinate format: {target_coords}")
            return
        
        # Enhanced overlay creation code that integrates with the setup
        if overlay_shape == "circle":
            overlay_creation = f"""
                var overlay = A.graphicOverlay({{
                    color: '{overlay_color}',
                    lineWidth: 3
                }});
                aladinInstance.addOverlay(overlay);
                overlay.addShape(A.circle({ra}, {dec}, {overlay_radius:.6f}));
                console.log('FOV Overlay Debug: Circle overlay added at RA={ra}, Dec={dec}, radius={overlay_radius:.6f}°');
            """
        else:
            half_width = telescope_fov_width_deg / 2.0
            half_height = telescope_fov_height_deg / 2.0
            overlay_creation = f"""
                var overlay = A.graphicOverlay({{
                    color: '{overlay_color}',
                    lineWidth: 3
                }});
                aladinInstance.addOverlay(overlay);
                
                var poly = A.polygon([
                    [{ra - half_width:.6f}, {dec - half_height:.6f}],
                    [{ra + half_width:.6f}, {dec - half_height:.6f}],
                    [{ra + half_width:.6f}, {dec + half_height:.6f}],
                    [{ra - half_width:.6f}, {dec + half_height:.6f}]
                ]);
                overlay.addShape(poly);
                console.log('FOV Overlay Debug: Rectangle overlay added at RA={ra}, Dec={dec}, size={half_width*2:.6f}°x{half_height*2:.6f}°');
            """
        
        # Combined JavaScript code that creates overlay when instance is found
        js_code = f"""
            var originalAddOverlayWhenReady = addOverlayWhenReady;
            addOverlayWhenReady = function(attemptCount) {{
                attemptCount = attemptCount || 0;
                console.log('FOV Overlay Debug: Overlay creation attempt', attemptCount);
                
                var aladinInstance = findAladinInstance();
                
                if (aladinInstance && typeof A !== 'undefined') {{
                    try {{
                        console.log('FOV Overlay Debug: Creating overlay...');
                        
                        {overlay_creation}
                        
                        // Add info overlay
                        var infoDiv = document.createElement('div');
                        infoDiv.style.cssText = 'position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.8);color:white;padding:10px;border-radius:5px;font-family:Arial;font-size:12px;z-index:1000;';
                        infoDiv.innerHTML = '<strong>Telescope FOV:</strong> {fov_data["width_arcmin"]:.1f}\\' × {fov_data["height_arcmin"]:.1f}\\'<br><strong>Type:</strong> {fov_data["details"]}';
                        document.body.appendChild(infoDiv);
                        
                        console.log('FOV Overlay Debug: Overlay and info panel added successfully!');
                        return true;
                        
                    }} catch(e) {{
                        console.error('FOV Overlay Debug: Error creating overlay:', e);
                        console.error('FOV Overlay Debug: Stack trace:', e.stack);
                    }}
                }} else {{
                    console.log('FOV Overlay Debug: Prerequisites not met - Aladin:', !!aladinInstance, 'A defined:', typeof A !== 'undefined');
                    if (attemptCount < 30) {{
                        setTimeout(function() {{ addOverlayWhenReady(attemptCount + 1); }}, 500);
                    }} else {{
                        console.log('FOV Overlay Debug: Giving up after 30 attempts');
                    }}
                }}
                return false;
            }};
        """
        
        # Enhanced setup code with more debugging
        setup_code = f"""
            console.log('FOV Overlay Debug: Starting injection');
            console.log('FOV Overlay Debug: Target coordinates = {target_coords}');
            console.log('FOV Overlay Debug: Overlay type = {overlay_shape}');
            console.log('FOV Overlay Debug: FOV = {fov_data["width_arcmin"]:.1f} arcmin');
            
            // Try multiple methods to find Aladin instance
            var findAladinInstance = function() {{
                var instance = null;
                
                // Method 1: Check global aladin variable
                if (typeof aladin !== 'undefined') {{
                    instance = aladin;
                    console.log('FOV Overlay Debug: Found aladin via global variable');
                }}
                
                // Method 2: Check window.aladin
                if (!instance && typeof window.aladin !== 'undefined') {{
                    instance = window.aladin;
                    console.log('FOV Overlay Debug: Found aladin via window.aladin');
                }}
                
                // Method 3: Look in DOM element
                if (!instance) {{
                    var aladinDiv = document.querySelector('#aladin-lite-div');
                    if (aladinDiv) {{
                        if (aladinDiv._aladin) {{
                            instance = aladinDiv._aladin;
                            console.log('FOV Overlay Debug: Found aladin via DOM._aladin');
                        }} else if (aladinDiv.aladin) {{
                            instance = aladinDiv.aladin;
                            console.log('FOV Overlay Debug: Found aladin via DOM.aladin');
                        }}
                    }}
                }}
                
                // Method 4: Check if A is defined and has instances
                if (!instance && typeof A !== 'undefined') {{
                    console.log('FOV Overlay Debug: A is defined, checking for instances');
                    // Try to get the first aladin instance
                    if (A.aladinInstances && A.aladinInstances.length > 0) {{
                        instance = A.aladinInstances[0];
                        console.log('FOV Overlay Debug: Found aladin via A.aladinInstances[0]');
                    }}
                }}
                
                return instance;
            }};
            
            var addOverlayWhenReady = function(attemptCount) {{
                attemptCount = attemptCount || 0;
                console.log('FOV Overlay Debug: Attempt', attemptCount);
                
                var aladinInstance = findAladinInstance();
                
                if (aladinInstance) {{
                    console.log('FOV Overlay Debug: Aladin instance found!', aladinInstance);
                    window.aladinInstance = aladinInstance;
                    return true;
                }} else {{
                    console.log('FOV Overlay Debug: Aladin instance not found');
                    if (attemptCount < 20) {{
                        setTimeout(function() {{ addOverlayWhenReady(attemptCount + 1); }}, 500);
                    }} else {{
                        console.log('FOV Overlay Debug: Giving up after 20 attempts');
                    }}
                    return false;
                }}
            }};
            
            // Start looking for Aladin instance
            setTimeout(function() {{ addOverlayWhenReady(0); }}, 1000);
        """
        
        # First try the complex Aladin API approach
        self.web_view.page().runJavaScript(setup_code)
        self.web_view.page().runJavaScript(js_code)
        
        # Also add a dynamically scaling HTML overlay
        # Get the current display FOV
        current_display_fov = self.current_fov
        
        simple_overlay_js = f"""
            // Global variables for the overlay system
            window.telescopeFovData = {{
                telescopeFovDegrees: {telescope_fov_width_deg:.6f},
                telescopeFovHeightDegrees: {telescope_fov_height_deg:.6f},
                overlayColor: '{overlay_color}',
                overlayShape: '{overlay_shape}',
                fovDetails: '{fov_data["details"]}',
                fovWidthArcmin: {fov_data["width_arcmin"]:.1f},
                fovHeightArcmin: {fov_data["height_arcmin"]:.1f},
                targetRA: {ra:.3f},
                targetDec: {dec:.3f}
            }};
            
            // Function to update overlay scale based on current Aladin FOV
            window.updateTelescopeFovOverlay = function() {{
                var existingIndicator = document.getElementById('telescope-fov-indicator');
                var existingPanel = document.getElementById('telescope-fov-panel');
                
                // Get current Aladin FOV dynamically
                var currentAladinFov = {current_display_fov};  // Fallback value
                
                // Try to get actual current FOV from Aladin instance
                try {{
                    if (window.aladinInstance && window.aladinInstance.getFov) {{
                        currentAladinFov = window.aladinInstance.getFov()[0]; // Get width FOV
                    }} else if (typeof aladin !== 'undefined' && aladin.getFov) {{
                        currentAladinFov = aladin.getFov()[0];
                    }}
                }} catch(e) {{
                    console.log('Could not get dynamic FOV, using fallback:', currentAladinFov);
                }}
                
                var data = window.telescopeFovData;
                
                // Calculate the size of the overlay as a percentage of the view
                var overlayWidthPercent = (data.telescopeFovDegrees / currentAladinFov) * 100;
                var overlayHeightPercent = (data.telescopeFovHeightDegrees / currentAladinFov) * 100;
                
                // Limit the overlay size to reasonable bounds
                overlayWidthPercent = Math.max(2, Math.min(95, overlayWidthPercent));
                overlayHeightPercent = Math.max(2, Math.min(95, overlayHeightPercent));
                
                console.log('FOV Update: Current Aladin FOV=' + currentAladinFov.toFixed(3) + '°, Telescope FOV=' + data.telescopeFovDegrees.toFixed(3) + '°, Overlay size=' + overlayWidthPercent.toFixed(1) + '%');
                
                // Find the Aladin container
                var aladinContainer = document.querySelector('#aladin-lite-div') || document.querySelector('.aladin-reticleContainer') || document.body;
                
                // Remove existing elements
                if (existingIndicator) existingIndicator.remove();
                if (existingPanel) existingPanel.remove();
                
                // Create new telescope FOV indicator
                var fovIndicator = document.createElement('div');
                fovIndicator.id = 'telescope-fov-indicator';
                fovIndicator.style.cssText = `
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    width: ${{overlayWidthPercent}}%;
                    height: ${{overlayHeightPercent}}%;
                    border: 3px solid ${{data.overlayColor}};
                    border-radius: {50 if overlay_shape == 'circle' else 0}%;
                    background: transparent;
                    transform: translate(-50%, -50%);
                    pointer-events: none;
                    z-index: 1000;
                    box-shadow: 0 0 10px rgba(0,0,0,0.8);
                    transition: all 0.3s ease;
                `;
                
                // Position relative to Aladin container
                if (aladinContainer !== document.body) {{
                    aladinContainer.style.position = 'relative';
                    aladinContainer.appendChild(fovIndicator);
                }} else {{
                    fovIndicator.style.position = 'fixed';
                    document.body.appendChild(fovIndicator);
                }}
                
                // Add crosshair at center
                var crosshair = document.createElement('div');
                crosshair.style.cssText = `
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    width: 20px;
                    height: 20px;
                    transform: translate(-50%, -50%);
                    pointer-events: none;
                    z-index: 1001;
                `;
                crosshair.innerHTML = `
                    <div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:${{data.overlayColor}};transform:translateX(-50%);"></div>
                    <div style="position:absolute;top:50%;left:0;height:1px;width:100%;background:${{data.overlayColor}};transform:translateY(-50%);"></div>
                `;
                fovIndicator.appendChild(crosshair);
                
                // Update info panel
                var scaleInfo = data.telescopeFovDegrees < currentAladinFov ? '📏 TO SCALE' : '⚠️ FOV larger than view';
                
                var infoPanel = document.createElement('div');
                infoPanel.id = 'telescope-fov-panel';
                infoPanel.style.cssText = `
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 12px;
                    border-radius: 8px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 12px;
                    z-index: 1001;
                    border: 2px solid ${{data.overlayColor}};
                    min-width: 200px;
                `;
                
                infoPanel.innerHTML = `
                    <div style="text-align:center;margin-bottom:8px;font-weight:bold;color:${{data.overlayColor}};">🔭 TELESCOPE FOV</div>
                    <div><strong>Size:</strong> ${{data.fovWidthArcmin}}' × ${{data.fovHeightArcmin}}'</div>
                    <div><strong>Setup:</strong> ${{data.fovDetails}}</div>
                    <div><strong>View:</strong> ${{currentAladinFov.toFixed(2)}}° (${{(currentAladinFov*60).toFixed(0)}}')</div>
                    <div><strong>Scale:</strong> ${{scaleInfo}}</div>
                    <div style="font-size:10px;color:#ccc;margin-top:5px;">Target: RA ${{data.targetRA}}° Dec ${{data.targetDec}}°</div>
                `;
                document.body.appendChild(infoPanel);
            }};
            
            // Initial overlay creation
            setTimeout(function() {{
                window.updateTelescopeFovOverlay();
                
                // Set up zoom change detection
                var lastFov = null;
                setInterval(function() {{
                    try {{
                        var currentFov = null;
                        if (window.aladinInstance && window.aladinInstance.getFov) {{
                            currentFov = window.aladinInstance.getFov()[0];
                        }} else if (typeof aladin !== 'undefined' && aladin.getFov) {{
                            currentFov = aladin.getFov()[0];
                        }}
                        
                        if (currentFov && Math.abs(currentFov - lastFov) > 0.001) {{
                            lastFov = currentFov;
                            window.updateTelescopeFovOverlay();
                        }}
                    }} catch(e) {{
                        // Silently ignore errors in polling
                    }}
                }}, 500); // Check every 500ms for zoom changes
                
                console.log('Dynamic FOV overlay system initialized');
            }}, 1500);
        """
        
        # Inject both approaches
        self.web_view.page().runJavaScript(simple_overlay_js)
        
        logger.debug("JavaScript FOV overlay injection completed (with simple fallback)")
    
    def _generate_fov_overlay_script(self, telescope_fov_data):
        """Generate JavaScript for FOV overlay (legacy method)"""
        # This method is now replaced by _inject_fov_overlay
        return None
    
    def _remove_fov_overlay(self):
        """Remove the FOV overlay from the current view"""
        try:
            remove_js = """
                // Remove existing FOV overlay elements
                var existingOverlay = document.querySelector('.telescope-fov-overlay');
                if (existingOverlay) {
                    existingOverlay.remove();
                }
                var existingInfo = document.querySelector('.telescope-fov-info');
                if (existingInfo) {
                    existingInfo.remove();
                }
                console.log('FOV overlay removed');
            """
            self.web_view.page().runJavaScript(remove_js)
            logger.debug("Removed FOV overlay")
        except Exception as e:
            logger.debug(f"Could not remove FOV overlay: {e}")

    def _update_fov_info(self):
        """Update the FOV information display"""
        info_parts = [f"View FOV: {self.current_fov:.2f}° ({self.current_fov*60:.1f}')"]
        
        # Object size info
        obj_size_min = self.data.get('size_min', 0)
        obj_size_max = self.data.get('size_max', 0)
        if obj_size_min > 0 and obj_size_max > 0:
            if abs(obj_size_min - obj_size_max) < 0.1:
                info_parts.append(f"Object: {obj_size_min:.1f}'")
            else:
                info_parts.append(f"Object: {obj_size_min:.1f}'–{obj_size_max:.1f}'")
        
        # Telescope FOV info
        if self.selected_telescope:
            telescope_fov_data = self._calculate_telescope_fov()
            if telescope_fov_data:
                telescope_name = self.selected_telescope['name']
                telescope_fl = self.selected_telescope['focal_length']
                
                if telescope_fov_data['type'] == 'visual':
                    fov_str = f"{telescope_fov_data['width_arcmin']:.1f}'"
                    info_parts.append(f"{telescope_name} ({telescope_fl}mm): {fov_str} {telescope_fov_data['details']}")
                else:
                    fov_str = f"{telescope_fov_data['width_arcmin']:.1f}'×{telescope_fov_data['height_arcmin']:.1f}'"
                    info_parts.append(f"{telescope_name} ({telescope_fl}mm): {fov_str} {telescope_fov_data['details']}")
                    
                    # Add pixel scale if available
                    if 'pixel_scale_arcsec' in telescope_fov_data:
                        info_parts.append(f"Pixel scale: {telescope_fov_data['pixel_scale_arcsec']:.1f}\"/px")
        
        self.fov_info_label.setText(" | ".join(info_parts))

    def _on_load_started(self):
        """Handle web page load started"""
        logger.debug("Aladin Lite: Load started")
        if self.web_placeholder:
            self.web_placeholder.setText("Loading Aladin Lite...")
        # Start timeout timer (30 seconds)
        self.loading_timeout.start(30000)

    def _on_load_progress(self, progress):
        """Handle web page load progress"""
        logger.debug(f"Aladin Lite: Load progress {progress}%")
        if self.web_placeholder:
            self.web_placeholder.setText(f"Loading Aladin Lite... {progress}%")

    def _on_load_finished(self, success):
        """Handle web page load finished"""
        logger.debug(f"Aladin Lite: Load finished, success={success}")
        self.loading_timeout.stop()

        if not success:
            logger.error("Failed to load Aladin Lite")
            if self.web_placeholder:
                self.web_placeholder.setText("Failed to load Aladin Lite\nCheck your internet connection\n\nClick below to open in browser instead.")
                self.web_placeholder.setStyleSheet("QLabel { background-color: #2b2b2b; color: #ff6b6b; font-size: 12px; }")
            self._add_browser_fallback_button()
        else:
            logger.debug("Aladin Lite loaded successfully")
            # Ensure the web view is visible and placeholder is hidden
            self._ensure_web_view_visible()

            # Handle FOV overlay injection if needed
            if self.pending_fov_overlay and self.target_coordinates:
                self._inject_fov_overlay(True)

    def _handle_loading_timeout(self):
        """Handle loading timeout"""
        logger.warning("Aladin Lite loading timed out after 30 seconds")
        if self.web_placeholder:
            self.web_placeholder.setText("Loading timed out\nAladin Lite may be slow to respond\n\nClick below to open in browser instead.")
            self.web_placeholder.setStyleSheet("QLabel { background-color: #2b2b2b; color: #ffaa00; font-size: 12px; }")
        self._add_browser_fallback_button()

    def _ensure_web_view_visible(self):
        """Ensure the web view is visible and replace placeholder if needed"""
        try:
            if self.web_view and self.web_placeholder:
                logger.debug("Replacing placeholder with web view after successful load")

                # Find the central widget and its layout
                central_widget = self.centralWidget()
                if central_widget and central_widget.layout():
                    main_layout = central_widget.layout()

                    # Find the placeholder in the layout and replace it
                    for i in range(main_layout.count()):
                        item = main_layout.itemAt(i)
                        if item and item.widget() == self.web_placeholder:
                            # Remove placeholder
                            main_layout.removeWidget(self.web_placeholder)
                            self.web_placeholder.hide()
                            self.web_placeholder.deleteLater()
                            self.web_placeholder = None

                            # Add web view in the same position
                            main_layout.insertWidget(i, self.web_view)
                            self.web_view.show()
                            logger.debug("Successfully replaced placeholder with web view")
                            break

            elif self.web_view:
                # Just make sure web view is visible
                self.web_view.show()
                logger.debug("Made web view visible")

        except Exception as e:
            logger.error(f"Error ensuring web view visibility: {e}")

    def _test_connectivity_async(self):
        """Test connectivity to Aladin Lite server asynchronously"""
        try:
            import urllib.request
            import threading

            def test_connection():
                try:
                    with urllib.request.urlopen("https://aladin.u-strasbg.fr", timeout=10) as response:
                        if response.status == 200:
                            logger.debug("Aladin server connectivity test successful")
                        else:
                            logger.warning(f"Aladin server responded with status {response.status}")
                except Exception as e:
                    logger.warning(f"Aladin server connectivity test failed: {e}")
                    # Don't show error here as it might succeed anyway

            # Run test in background thread
            threading.Thread(target=test_connection, daemon=True).start()
        except Exception as e:
            logger.debug(f"Could not perform connectivity test: {e}")


# --- Image Viewer Window ---
class ImageViewerWindow(QDialog):
    """Window to display an image in full size with enhanced controls"""
    zoom_changed = Signal(float)  # Signal for zoom level changes

    def __init__(self, pixmap: QPixmap, title: str, file_path: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} - Image Viewer - Cosmos Collection")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(800, 600)

        self.setMinimumSize(300, 300)

        # Store the original pixmap and file path
        self.original_pixmap = pixmap
        self.file_path = file_path
        self.zoom_factor = 1.0
        self.initial_zoom_factor = 1.0
        self.image_position = [0, 0]
        self.last_mouse_pos = None
        self.is_panning = False

        # Create main layout
        main_layout = QVBoxLayout()

        # Create toolbar for controls
        toolbar = QHBoxLayout()

        # Add zoom controls
        zoom_out_button = QPushButton("-")
        zoom_out_button.setFixedSize(30, 30)
        zoom_out_button.clicked.connect(self._zoom_out)
        toolbar.addWidget(zoom_out_button)

        zoom_in_button = QPushButton("+")
        zoom_in_button.setFixedSize(30, 30)
        zoom_in_button.clicked.connect(self._zoom_in)
        toolbar.addWidget(zoom_in_button)

        reset_button = QPushButton("Reset")
        reset_button.setFixedSize(60, 30)
        reset_button.clicked.connect(self._reset_zoom)
        toolbar.addWidget(reset_button)

        # Add fit to window button
        fit_button = QPushButton("Fit to Window")
        fit_button.setFixedSize(100, 30)
        fit_button.clicked.connect(self._fit_to_window)
        toolbar.addWidget(fit_button)

        # Add open file location button if file path is available
        if self.file_path:
            open_location_button = QPushButton("Open File Location")
            open_location_button.setFixedSize(120, 30)
            open_location_button.clicked.connect(self._open_file_location)
            toolbar.addWidget(open_location_button)

        toolbar.addStretch()

        # Add file info toggle button (if file path is available)
        if self.file_path:
            self.info_toggle_button = QPushButton("Show File Info")
            self.info_toggle_button.setFixedSize(100, 30)
            self.info_toggle_button.setCheckable(True)
            self.info_toggle_button.setChecked(False)
            self.info_toggle_button.clicked.connect(self._toggle_file_info)
            toolbar.addWidget(self.info_toggle_button)

        main_layout.addLayout(toolbar)

        # Create horizontal layout for image and file info panel
        content_layout = QHBoxLayout()

        # Create a container widget for the image
        self.image_container = QWidget()
        self.image_container.setLayout(QVBoxLayout())
        self.image_container.setStyleSheet("background-color: black;")

        # Create image label
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        # Add image label to container
        self.image_container.layout().addWidget(self.image_label)
        content_layout.addWidget(self.image_container)

        # Create file information panel as a right-side groupbox (hidden by default)
        self.file_info_panel = QGroupBox("File Information")
        self.file_info_panel.setVisible(False)
        self.file_info_panel.setFixedWidth(350)

        # Dark mode styling for the groupbox
        self.file_info_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12pt;
                color: #e0e0e0;
                background-color: #2b2b2b;
                border: 2px solid #555555;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #ffffff;
                background-color: #2b2b2b;
            }
        """)

        # Create file info layout
        file_info_layout = QVBoxLayout()
        file_info_layout.setContentsMargins(15, 10, 15, 15)

        # Create scrollable area for file info content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Dark mode styling for scroll area
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
        """)

        self.file_info_content = QLabel()
        # Dark mode styling for the content label
        self.file_info_content.setStyleSheet("""
            QLabel {
                font-size: 10pt;
                color: #d0d0d0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                padding: 10px;
                background-color: #1e1e1e;
                border: none;
            }
        """)
        self.file_info_content.setWordWrap(True)
        self.file_info_content.setAlignment(Qt.AlignTop)

        scroll_area.setWidget(self.file_info_content)
        file_info_layout.addWidget(scroll_area)

        self.file_info_panel.setLayout(file_info_layout)
        content_layout.addWidget(self.file_info_panel)

        # Set stretch ratios - image container gets most space, info panel gets fixed width
        content_layout.setStretch(0, 1)  # image_container
        content_layout.setStretch(1, 0)  # file_info_panel

        main_layout.addLayout(content_layout)

        # Add status bar
        self.status_bar = QLabel()
        self.status_bar.setStyleSheet("font-size: 10pt;")
        main_layout.addWidget(self.status_bar)

        # Set stretch to ensure content takes most space and status bar stays at bottom
        main_layout.setStretch(0, 0)  # toolbar
        main_layout.setStretch(1, 1)  # content_layout
        main_layout.setStretch(2, 0)  # status_bar

        self.setLayout(main_layout)

        # Flag to track if initial fit has been done
        self.initial_fit_done = False

        # Update status
        self._update_status()

    def showEvent(self, event):
        """Handle window show event - fit image to window on first show"""
        super().showEvent(event)
        if not self.initial_fit_done:
            # Defer the initial fit to ensure the window is properly sized
            QTimer.singleShot(50, self._do_initial_fit)

    def _do_initial_fit(self):
        """Perform initial fit to window and set up initial zoom factor"""
        if not self.initial_fit_done:
            self._fit_to_window()
            self.initial_zoom_factor = self.zoom_factor
            self.initial_fit_done = True

    def eventFilter(self, obj, event):
        """Handle mouse events for zooming and panning"""
        if obj == self.image_label:
            if event.type() == QEvent.Wheel:
                if event.angleDelta().y() > 0:
                    self._zoom_in()
                else:
                    self._zoom_out()
                return True
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.is_panning = True
                self.last_mouse_pos = event.position()
                self.setCursor(Qt.ClosedHandCursor)
                return True
            elif event.type() == QEvent.MouseMove:
                if self.is_panning and self.last_mouse_pos is not None:
                    # Calculate the drag distance
                    dx = event.position().x() - self.last_mouse_pos.x()
                    dy = event.position().y() - self.last_mouse_pos.y()

                    # Update image position
                    self.image_position[0] += dx
                    self.image_position[1] += dy

                    # Update the display
                    self._update_zoom()

                    self.last_mouse_pos = event.position()
                    return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.is_panning = False
                self.last_mouse_pos = None
                self.setCursor(Qt.ArrowCursor)
                return True
        return super().eventFilter(obj, event)

    def _update_zoom(self):
        """Update the image display with current zoom level and position"""
        if self.original_pixmap is None:
            return

        # Calculate new size
        new_width = int(self.original_pixmap.width() * self.zoom_factor)
        new_height = int(self.original_pixmap.height() * self.zoom_factor)

        # Scale the image
        scaled_pixmap = self.original_pixmap.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Create a new pixmap with the same size as the label
        label_size = self.image_label.size()
        final_pixmap = QPixmap(label_size)
        final_pixmap.fill(Qt.transparent)

        # Create a painter to draw the scaled image at the correct position
        painter = QPainter(final_pixmap)

        # Calculate the position to draw the image
        x = (label_size.width() - scaled_pixmap.width()) // 2 + self.image_position[0]
        y = (label_size.height() - scaled_pixmap.height()) // 2 + self.image_position[1]

        # Draw the image
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()

        # Update the display
        self.image_label.setPixmap(final_pixmap)

        # Update status
        self._update_status()
        self.zoom_changed.emit(self.zoom_factor)  # Emit zoom level change signal

    def _fit_to_window(self):
        """Fit the image to the window size"""
        if self.original_pixmap is None:
            return

        # Get available size
        available_size = self.image_container.size()

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

        if not self.initial_fit_done:
            # If initial fit hasn't been done yet, just update the display
            self._update_zoom()
            return

        # Store the current relative position of the image center before resize
        old_size = event.oldSize() if event.oldSize().isValid() else self.size()
        new_size = event.size()

        # Only auto-fit if we're at the initial zoom level (user hasn't manually zoomed)
        if abs(self.zoom_factor - self.initial_zoom_factor) < 0.001:
            # User is at initial zoom - maintain fit to window behavior
            self._fit_to_window()
            self.initial_zoom_factor = self.zoom_factor
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

    def _zoom_in(self):
        """Zoom in on the image"""
        self.zoom_factor = min(self.zoom_factor * 1.2, 8.0)
        self._update_zoom()

    def _zoom_out(self):
        """Zoom out on the image"""
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self._update_zoom()

    def _reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.image_position = [0, 0]
        self._update_zoom()

    def _update_status(self):
        """Update the status bar with current zoom level and image size"""
        zoom_percent = int(self.zoom_factor * 100)
        image_size = f"{self.original_pixmap.width()}—{self.original_pixmap.height()}"
        self.status_bar.setText(f"Zoom: {zoom_percent}% | Image Size: {image_size} pixels")

    def _open_file_location(self):
        """Open the file location in the system's file explorer"""
        if self.file_path:
            success = ResourceManager.open_file_manager(self.file_path)
            if not success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", "Failed to open file location")

    def _toggle_file_info(self):
        """Toggle the visibility of the file information panel"""
        if hasattr(self, 'file_info_panel'):
            is_visible = self.file_info_panel.isVisible()
            self.file_info_panel.setVisible(not is_visible)

            # Update button text
            if hasattr(self, 'info_toggle_button'):
                if not is_visible:
                    self.info_toggle_button.setText("Hide File Info")
                    # Load and display file information when showing
                    self._load_file_information()
                else:
                    self.info_toggle_button.setText("Show File Info")

    def _load_file_information(self):
        """Load and display file information"""
        if not self.file_path or not hasattr(self, 'file_info_content'):
            return

        try:
            import os
            from datetime import datetime
            from pathlib import Path

            # Get file stats
            file_stats = os.stat(self.file_path)
            file_path_obj = Path(self.file_path)

            # Basic file information
            file_name = file_path_obj.name
            file_dir = str(file_path_obj.parent)
            file_size_bytes = file_stats.st_size

            # Format file size
            if file_size_bytes < 1024:
                file_size = f"{file_size_bytes} bytes"
            elif file_size_bytes < 1024 * 1024:
                file_size = f"{file_size_bytes / 1024:.1f} KB"
            elif file_size_bytes < 1024 * 1024 * 1024:
                file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
            else:
                file_size = f"{file_size_bytes / (1024 * 1024 * 1024):.1f} GB"

            # Dates
            created_time = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            # Image dimensions
            if self.original_pixmap:
                width = self.original_pixmap.width()
                height = self.original_pixmap.height()
                dimensions = f"{width} × {height} pixels"

                # Calculate megapixels
                megapixels = (width * height) / 1000000
                if megapixels >= 1:
                    megapixels_str = f"({megapixels:.1f} MP)"
                else:
                    megapixels_str = f"({megapixels * 1000:.0f}K pixels)"
            else:
                dimensions = "Unknown"
                megapixels_str = ""

            # Try to get EXIF data if available
            exif_info = self._get_exif_info()

            # Try to get FITS header information if available
            fits_info = self._get_fits_info()

            # Build information string
            info_lines = [
                f"Filename: {file_name}",
                f"Location: {file_dir}",
                f"File Size: {file_size}",
                f"Dimensions: {dimensions} {megapixels_str}",
                f"Created: {created_time}",
                f"Modified: {modified_time}",
            ]

            # Add FITS information if available
            if fits_info:
                info_lines.append("")
                info_lines.append("FITS Header Information:")
                for key, value in fits_info.items():
                    info_lines.append(f"  {key}: {value}")

            # Add EXIF information if available
            if exif_info:
                info_lines.append("")
                info_lines.append("EXIF Data:")
                for key, value in exif_info.items():
                    info_lines.append(f"  {key}: {value}")

            # Display the information
            self.file_info_content.setText("\n".join(info_lines))

        except Exception as e:
            self.file_info_content.setText(f"Error loading file information:\n{str(e)}")

    def _get_exif_info(self):
        """Extract basic EXIF information from the image file"""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            # Open image with PIL to read EXIF
            with Image.open(self.file_path) as img:
                exif_data = img.getexif()

                if not exif_data:
                    return None

                # Extract useful EXIF information
                exif_info = {}

                # Common EXIF tags we want to show
                useful_tags = {
                    'Make': 'Camera Make',
                    'Model': 'Camera Model',
                    'DateTime': 'Date Taken',
                    'ExposureTime': 'Exposure Time',
                    'FNumber': 'F-Number',
                    'ISO': 'ISO Speed',
                    'FocalLength': 'Focal Length',
                    'Flash': 'Flash',
                    'WhiteBalance': 'White Balance',
                    'ExposureProgram': 'Exposure Program',
                    'MeteringMode': 'Metering Mode'
                }

                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name in useful_tags:
                        # Format specific values
                        if tag_name == 'ExposureTime' and isinstance(value, tuple):
                            value = f"{value[0]}/{value[1]} sec"
                        elif tag_name == 'FNumber' and isinstance(value, tuple):
                            value = f"f/{value[0]/value[1]:.1f}"
                        elif tag_name == 'FocalLength' and isinstance(value, tuple):
                            value = f"{value[0]/value[1]:.1f}mm"

                        exif_info[useful_tags[tag_name]] = str(value)

                return exif_info if exif_info else None

        except ImportError:
            # PIL not available
            return None
        except Exception as e:
            # Any other error reading EXIF
            return None

    def _get_fits_info(self):
        """Extract FITS header information from the image file"""
        try:
            from astropy.io import fits
            from pathlib import Path

            # Check if file is a FITS file
            file_ext = Path(self.file_path).suffix.lower()
            if file_ext not in ['.fits', '.fit', '.fts']:
                return None

            # Open FITS file and read header
            with fits.open(self.file_path) as hdul:
                header = hdul[0].header

                if not header:
                    return None

                # Extract useful FITS header information
                fits_info = {}

                # Common FITS keywords we want to show
                useful_keywords = {
                    'OBJECT': 'Object Name',
                    'TELESCOP': 'Telescope',
                    'INSTRUME': 'Instrument',
                    'OBSERVER': 'Observer',
                    'DATE-OBS': 'Observation Date',
                    'EXPTIME': 'Exposure Time (s)',
                    'FILTER': 'Filter',
                    'FOCALLEN': 'Focal Length (mm)',
                    'APTDIA': 'Aperture Diameter (mm)',
                    'APTAREA': 'Aperture Area (mm²)',
                    'FWHM': 'FWHM (arcsec)',
                    'EQUINOX': 'Equinox',
                    'RA': 'Right Ascension',
                    'DEC': 'Declination',
                    'OBJCTRA': 'Object RA',
                    'OBJCTDEC': 'Object Dec',
                    'AIRMASS': 'Airmass',
                    'GAIN': 'Gain',
                    'OFFSET': 'Offset',
                    'TEMP': 'Temperature (°C)',
                    'CCD-TEMP': 'CCD Temperature (°C)',
                    'SET-TEMP': 'Set Temperature (°C)',
                    'XBINNING': 'X Binning',
                    'YBINNING': 'Y Binning',
                    'IMAGETYP': 'Image Type',
                    'FRAME': 'Frame Type',
                    'SWCREATE': 'Software Created',
                    'SWMODIFY': 'Software Modified'
                }

                for keyword, description in useful_keywords.items():
                    if keyword in header:
                        value = header[keyword]

                        # Format specific values
                        if keyword in ['DATE-OBS'] and isinstance(value, str):
                            # Try to format the date nicely
                            try:
                                from datetime import datetime
                                if 'T' in value:
                                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    value = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                            except:
                                pass
                        elif keyword in ['EXPTIME'] and isinstance(value, (int, float)):
                            if value >= 60:
                                minutes = int(value // 60)
                                seconds = value % 60
                                if seconds == 0:
                                    value = f"{value} s ({minutes}m)"
                                else:
                                    value = f"{value} s ({minutes}m {seconds:.1f}s)"
                            else:
                                value = f"{value} s"
                        elif keyword in ['RA', 'OBJCTRA'] and isinstance(value, (int, float)):
                            # Convert RA from degrees to hours:minutes:seconds
                            ra_hours = value / 15.0
                            hours = int(ra_hours)
                            minutes = int((ra_hours - hours) * 60)
                            seconds = ((ra_hours - hours) * 60 - minutes) * 60
                            value = f"{value}° ({hours:02d}h {minutes:02d}m {seconds:05.2f}s)"
                        elif keyword in ['DEC', 'OBJCTDEC'] and isinstance(value, (int, float)):
                            # Format declination as degrees:arcminutes:arcseconds
                            dec_deg = abs(value)
                            sign = '+' if value >= 0 else '-'
                            degrees = int(dec_deg)
                            arcmin = int((dec_deg - degrees) * 60)
                            arcsec = ((dec_deg - degrees) * 60 - arcmin) * 60
                            value = f"{value}° ({sign}{degrees:02d}° {arcmin:02d}' {arcsec:05.2f}\")"
                        elif keyword in ['TEMP', 'CCD-TEMP', 'SET-TEMP'] and isinstance(value, (int, float)):
                            value = f"{value}°C"

                        fits_info[description] = str(value)

                # Add image dimensions from FITS if available
                if 'NAXIS1' in header and 'NAXIS2' in header:
                    width = header['NAXIS1']
                    height = header['NAXIS2']
                    if 'NAXIS3' in header:
                        depth = header['NAXIS3']
                        fits_info['Image Dimensions'] = f"{width} × {height} × {depth} pixels"
                    else:
                        fits_info['Image Dimensions'] = f"{width} × {height} pixels"

                # Add pixel scale if available
                if 'PIXSCALE' in header:
                    fits_info['Pixel Scale'] = f"{header['PIXSCALE']} arcsec/pixel"
                elif 'CDELT1' in header:
                    fits_info['Pixel Scale'] = f"{abs(header['CDELT1']) * 3600:.2f} arcsec/pixel"

                return fits_info if fits_info else None

        except ImportError:
            # Astropy not available
            return None
        except Exception as e:
            # Any other error reading FITS
            return None


# --- Collage Selection Dialog ---
class CollageSelectionDialog(QDialog):
    """Dialog for selecting whether to create a new collage or add to existing one"""
    
    def __init__(self, dsodetailid, parent=None):
        super().__init__(parent)
        self.dsodetailid = dsodetailid
        self.selected_action = None
        self.selected_collage = None
        
        self.setWindowTitle("Create Collage")
        self.setModal(True)
        self.resize(400, 300)
        
        self._setup_ui()
        self._load_existing_collages()
        
    def _setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Create Collage")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel("Choose how you want to create your collage:")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Radio buttons for selection
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        
        self.button_group = QButtonGroup()
        
        self.new_collage_radio = QRadioButton("Create new collage")
        self.new_collage_radio.setChecked(True)
        self.new_collage_radio.toggled.connect(self._on_new_collage_selected)
        self.button_group.addButton(self.new_collage_radio)
        layout.addWidget(self.new_collage_radio)
        
        self.existing_collage_radio = QRadioButton("Add to existing collage")
        self.existing_collage_radio.toggled.connect(self._on_existing_collage_selected)
        self.button_group.addButton(self.existing_collage_radio)
        layout.addWidget(self.existing_collage_radio)
        
        # Collage selection dropdown (initially hidden)
        self.collage_selection_widget = QWidget()
        collage_layout = QVBoxLayout(self.collage_selection_widget)
        
        collage_label = QLabel("Select existing collage:")
        collage_layout.addWidget(collage_label)
        
        self.collage_combo = QComboBox()
        collage_layout.addWidget(self.collage_combo)
        
        self.collage_selection_widget.setVisible(False)
        layout.addWidget(self.collage_selection_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._on_ok_clicked)
        self.ok_button.setDefault(True)
        button_layout.addWidget(self.ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def _on_new_collage_selected(self, checked):
        """Handle new collage radio button selection"""
        if checked:
            self.collage_selection_widget.setVisible(False)
            
    def _on_existing_collage_selected(self, checked):
        """Handle existing collage radio button selection"""  
        if checked:
            self.collage_selection_widget.setVisible(True)
            
    def _load_existing_collages(self):
        """Load existing collages for this DSO"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                
                # Tables are now created automatically by DatabaseManager
                
                # Debug: Log that we're loading all existing collages
                logger.debug("Loading all existing collages")
                
                cursor.execute("""
                    SELECT id, name, created_date, modified_date 
                    FROM usercollages 
                    ORDER BY modified_date DESC
                """)
                
                collages = cursor.fetchall()
                
                # Debug: Log the results
                logger.debug(f"Found {len(collages)} existing collages")
                for collage_id, name, created, modified in collages:
                    logger.debug(f"  Collage ID: {collage_id}, Name: {name}")
                
                self.collage_combo.clear()
                for collage_id, name, created, modified in collages:
                    from datetime import datetime
                    modified_date = datetime.fromisoformat(modified).strftime("%Y-%m-%d %H:%M")
                    display_text = f"{name} (modified: {modified_date})"
                    self.collage_combo.addItem(display_text, collage_id)
                
                # Disable existing collage option if no collages exist
                if not collages:
                    self.existing_collage_radio.setEnabled(False)
                    self.existing_collage_radio.setText("Add to existing collage (none available)")
                    
        except Exception as e:
            logger.error(f"Error loading existing collages: {str(e)}")
            self.existing_collage_radio.setEnabled(False)
            self.existing_collage_radio.setText("Add to existing collage (error loading)")
            
    def _on_ok_clicked(self):
        """Handle OK button click"""
        if self.new_collage_radio.isChecked():
            self.selected_action = "new"
            self.selected_collage = None
        elif self.existing_collage_radio.isChecked():
            if self.collage_combo.currentIndex() >= 0:
                self.selected_action = "existing"
                collage_id = self.collage_combo.currentData()
                # Load the full collage data
                self.selected_collage = self._load_collage_data(collage_id)
            else:
                QMessageBox.warning(self, "No Selection", "Please select a collage to add images to.")
                return
        else:
            QMessageBox.warning(self, "No Selection", "Please choose an option.")
            return
            
        self.accept()
        
    def _load_collage_data(self, collage_id):
        """Load full collage data for the selected collage"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, grid_width, grid_height, cell_size, 
                           spacing, background_color
                    FROM usercollages 
                    WHERE id = ?
                """, (collage_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'id': result[0],
                        'name': result[1], 
                        'grid_width': result[2],
                        'grid_height': result[3],
                        'cell_size': result[4],
                        'spacing': result[5],
                        'background_color': result[6]
                    }
        except Exception as e:
            logger.error(f"Error loading collage data: {str(e)}")
            return None
            
    def get_selection(self):
        """Get the user's selection"""
        return self.selected_action, self.selected_collage


# --- Object Detail Window ---
class ObjectDetailWindow(QDialog):
    """
    Detail window for DSO objects with image support including FITS files.
    
    Supports the following image formats:
    - Regular formats: PNG, JPG, JPEG, TIFF, TIF, GIF, BMP
    - Astronomical formats: FITS, FIT, FTS (requires astropy and matplotlib)
    
    FITS files support both natural RGB and false color display:
    - RGB FITS files: Displayed in natural colors (no colormap applied)
    - Grayscale FITS files: Default 'gray' for natural B&W appearance
    - False color options: 'viridis', 'hot', 'cool', 'plasma', 'inferno' for grayscale data
    - Change FITS_COLORMAP class variable to customize display
    """
    image_added = Signal()  # Add this signal at the class level
    
    # FITS colormap configuration - users can change this to their preference
    FITS_COLORMAP = 'gray'  # Options: 'gray' (natural B&W), 'viridis', 'hot', 'cool', 'plasma', 'inferno'

    def __init__(self, data: dict, parent=None):
        super().__init__(None)  # Pass None as parent to make it independent
        logger.debug(f"Creating ObjectDetailWindow for {data['name']}")
        self.setWindowTitle(f"{data['name']} - DSO Detail - Cosmos Collection")
        # Make it an independent window with window management buttons
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.NonModal)  # Ensure it's non-modal
        self.resize(1000, 800)
        self.data = data  # Store data for later use
        self.current_image_path = None  # Store current image path
        self.zoom_factor = 1.0  # Current zoom level
        self.initial_zoom_factor = 1.0  # Initial zoom factor
        self.original_pixmap = None  # Store original image
        self.image_position = [0, 0]  # Current image position [x, y]
        self.drag_start_position = None  # Position where drag started
        self.drag_start_image_position = None  # Image position when drag started
        self.image_cache = ImageCache()
        self.db_manager = DatabaseManager()  # Initialize database manager
        
        # Multiple images support
        self.user_images = []  # List of image data dictionaries
        self.current_image_index = 0  # Current image index

        # Pre-compute coordinate strings (cache them for performance)
        self.ra_str, self.dec_str = self._format_coordinates(data["ra_deg"], data["dec_deg"])
        
        logger.debug(f"About to set up UI for {data['name']}")
        # Set up UI immediately
        self._setup_ui()

    def showEvent(self, event):
        """Handle window show event"""
        super().showEvent(event)
        # Defer heavy calculations until window is actually visible
        QTimer.singleShot(200, self._defer_heavy_calculations)

    def _defer_heavy_calculations(self):
        """Perform heavy calculations after window is shown"""
        # Load user location and update visibility calculations
        self._load_user_location()
        lat_val = self.location_lat_edit.text().strip()
        lon_val = self.location_lon_edit.text().strip()
        if lat_val and lon_val:
            self.location_groupbox.setVisible(False)
            # Defer the season calculation to avoid blocking
            QTimer.singleShot(500, self._set_season_label_from_location)
        else:
            self.location_groupbox.setVisible(True)
            self.season_label.setText("Enter your location above and press Save to see viewing season/dates.")
        
        # Load user images after window is shown and other calculations are done
        QTimer.singleShot(300, self._load_user_images)
        if self.data.get('image_path'):
            self._load_image_info()

    def _format_coordinates(self, ra_deg, dec_deg):
        """Format RA and Dec coordinates efficiently"""
        # Convert RA from degrees to hours (1 hour = 15 degrees)
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_remaining = (ra_hours - ra_h) * 60
        ra_m = int(ra_remaining)
        ra_s = (ra_remaining - ra_m) * 60
        ra_str = f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s"

        # Convert Dec from decimal degrees to dms format
        dec_sign = '-' if dec_deg < 0 else '+'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_remaining = (dec_abs - dec_d) * 60
        dec_m = int(dec_remaining)
        dec_s = (dec_remaining - dec_m) * 60
        dec_str = f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""
        
        return ra_str, dec_str

    def _setup_ui(self):
        """Set up the UI components - called after window is shown"""
        logger.debug(f"_setup_ui called for {self.data['name']}")
        try:
            # Create main horizontal layout
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
            zoom_out_button.setFixedSize(30, 30)
            zoom_out_button.clicked.connect(self._zoom_out)
            zoom_layout.addWidget(zoom_out_button)

            zoom_in_button = QPushButton("+")
            zoom_in_button.setFixedSize(30, 30)
            zoom_in_button.clicked.connect(self._zoom_in)
            zoom_layout.addWidget(zoom_in_button)

            reset_button = QPushButton("Reset")
            reset_button.setFixedSize(60, 30)
            reset_button.clicked.connect(self._reset_zoom)
            zoom_layout.addWidget(reset_button)

            # Add image navigation controls
            nav_separator = QLabel("|")
            nav_separator.setStyleSheet("font-size: 14pt; color: #666666; padding: 0 5px;")
            zoom_layout.addWidget(nav_separator)

            self.prev_image_button = QPushButton("◀")
            self.prev_image_button.setFixedSize(30, 30)
            self.prev_image_button.clicked.connect(self._previous_image)
            self.prev_image_button.setToolTip("Previous image")
            zoom_layout.addWidget(self.prev_image_button)

            self.image_counter_label = QLabel("1/1")
            self.image_counter_label.setStyleSheet("font-size: 10pt; color: #666666; padding: 0 5px;")
            self.image_counter_label.setMinimumWidth(40)
            self.image_counter_label.setAlignment(Qt.AlignCenter)
            zoom_layout.addWidget(self.image_counter_label)

            self.next_image_button = QPushButton("▶")
            self.next_image_button.setFixedSize(30, 30)
            self.next_image_button.clicked.connect(self._next_image)
            self.next_image_button.setToolTip("Next image")
            zoom_layout.addWidget(self.next_image_button)

            # Add image button
            add_separator = QLabel("|")
            add_separator.setStyleSheet("font-size: 14pt; color: #666666; padding: 0 5px;")
            zoom_layout.addWidget(add_separator)

            self.add_image_button = QPushButton("+")
            self.add_image_button.setFixedSize(30, 30)
            self.add_image_button.clicked.connect(self._add_user_image)
            self.add_image_button.setToolTip("Add new image")
            self.add_image_button.setStyleSheet("QPushButton { color: #4CAF50; font-size: 16pt; font-weight: bold; }")
            zoom_layout.addWidget(self.add_image_button)

            # Delete image button
            self.delete_image_button = QPushButton("🗑")
            self.delete_image_button.setFixedSize(30, 30)
            self.delete_image_button.clicked.connect(self._delete_current_image)
            self.delete_image_button.setToolTip("Delete current image")
            self.delete_image_button.setStyleSheet("QPushButton { color: #ff6b6b; font-size: 12pt; }")
            zoom_layout.addWidget(self.delete_image_button)

            zoom_layout.addStretch()
            image_container_layout.addLayout(zoom_layout)

            # Image label
            self.image_label = QLabel("No image attached to this DSO.")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("font-size: 14pt; color: gray;")
            self.image_label.setMinimumSize(600, 400)  # Increased minimum size
            self.image_label.installEventFilter(self)  # Install event filter for mouse events
            self.image_label.setMouseTracking(True)  # Enable mouse tracking
            image_container_layout.addWidget(self.image_label)

            image_container.setLayout(image_container_layout)
            image_layout.addWidget(image_container)
            left_layout.addLayout(image_layout, stretch=2)  # Give image area more stretch

            # Create container for image information form
            self.info_form_container = QWidget()
            info_form_layout = QVBoxLayout()
            info_form_layout.setSpacing(5)  # Reduce spacing between elements

            # Image information: Integration Time
            integration_layout = QHBoxLayout()
            integration_label = QLabel("Integration:")
            self.integration_edit = QLineEdit()
            self.integration_edit.setPlaceholderText("e.g., 2h 30m")
            integration_layout.addWidget(integration_label)
            integration_layout.addWidget(self.integration_edit)
            info_form_layout.addLayout(integration_layout)

            # Image information: Telescope
            telescope_layout = QHBoxLayout()
            telescope_label = QLabel("Telescope:")
            self.telescope_combo = QComboBox()
            self.telescope_combo.setEditable(True)
            self.telescope_combo.setPlaceholderText("Select telescope or enter custom equipment")
            telescope_layout.addWidget(telescope_label)
            telescope_layout.addWidget(self.telescope_combo)
            info_form_layout.addLayout(telescope_layout)

            # Image information: Date Taken
            date_layout = QHBoxLayout()
            date_label = QLabel("Date:")
            self.date_edit = QLineEdit()
            self.date_edit.setPlaceholderText("e.g., 2024-03-15")
            date_layout.addWidget(date_label)
            date_layout.addWidget(self.date_edit)
            info_form_layout.addLayout(date_layout)

            # Image information: Notes
            notes_layout = QVBoxLayout()
            notes_label = QLabel("Notes:")
            self.notes_edit = QTextEdit()
            self.notes_edit.setPlaceholderText("Additional notes about the image")
            self.notes_edit.setMaximumHeight(60)  # Limit notes height
            notes_layout.addWidget(notes_label)
            notes_layout.addWidget(self.notes_edit)
            info_form_layout.addLayout(notes_layout)

            # Image information: Save button
            self.save_button = QPushButton("Save Image Information")
            self.save_button.clicked.connect(self._save_image_info)
            info_form_layout.addWidget(self.save_button)

            self.info_form_container.setLayout(info_form_layout)
            left_layout.addWidget(self.info_form_container, stretch=1)  # Give info area less stretch

            
            # Create collage button
            #collage_button = QPushButton("Create Collage")
            #collage_button.clicked.connect(self._create_collage)
            #left_layout.addWidget(collage_button)
            
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
            surface_brightness_str = f"{self.data['surface_brightness']:.2f} mag/arcmin²" if self.data['surface_brightness'] is not None else "Unknown"
            
            # Handle size information
            size_min = self.data['size_min'] if self.data['size_min'] is not None else 0
            size_max = self.data['size_max'] if self.data['size_max'] is not None else 0
            if size_min > 0 or size_max > 0:
                size_str = f"{size_min:.1f}' — {size_max:.1f}'"
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
                for designation in designations:
                    if designation and designation != primary_name:
                        other_designations.append(designation)
                
                # Display other designations
                if other_designations:
                    for designation in other_designations:
                        object_info_text += f"{designation}<br>"
                else:
                    object_info_text += "None<br>"
            else:
                object_info_text += "None<br>"

            object_info_text += "<br><b>Source:</b> N.I.N.A. Database"

            object_info_label = QLabel(object_info_text)
            object_info_label.setAlignment(Qt.AlignLeft)
            object_info_label.setWordWrap(True)
            object_info_layout.addWidget(object_info_label)

            object_info_groupbox.setLayout(object_info_layout)
            right_layout.addWidget(object_info_groupbox)

            # --- Observer Location GroupBox (NEW or Conditional) ---
            self.location_groupbox = QGroupBox("Observer Location")
            self.location_groupbox.setStyleSheet(
                "QGroupBox:title { subcontrol-position: top center; font-size: 16pt; font-weight: bold; }")
            location_layout = QVBoxLayout()

            lat_layout = QHBoxLayout()
            lat_label = QLabel("Latitude (deg):")
            self.location_lat_edit = QLineEdit()
            self.location_lat_edit.setPlaceholderText("e.g., 40.7128")
            lat_layout.addWidget(lat_label)
            lat_layout.addWidget(self.location_lat_edit)
            location_layout.addLayout(lat_layout)

            lon_layout = QHBoxLayout()
            lon_label = QLabel("Longitude (deg):")
            self.location_lon_edit = QLineEdit()
            self.location_lon_edit.setPlaceholderText("e.g., -74.0060")
            lon_layout.addWidget(lon_label)
            lon_layout.addWidget(self.location_lon_edit)
            location_layout.addLayout(lon_layout)

            # Save button for location
            location_save_btn = QPushButton("Save")
            location_save_btn.clicked.connect(self._on_save_location_clicked)
            location_layout.addWidget(location_save_btn)

            self.location_groupbox.setLayout(location_layout)
            right_layout.addWidget(self.location_groupbox)

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

            # Add buttons layout
            buttons_layout = QVBoxLayout()

            # Add Visibility Calculator button (if available)
            if VISIBILITY_AVAILABLE:
                visibility_button = QPushButton("Open Visibility Calculator")
                visibility_button.clicked.connect(self._open_visibility_calculator)
                buttons_layout.addWidget(visibility_button)

            # Add Aladin Lite button
            aladin_button = QPushButton("Open in Aladin Lite")
            aladin_button.clicked.connect(lambda: self._open_aladin_lite(self.data))
            buttons_layout.addWidget(aladin_button)

            # Add to Target List button (conditional based on whether DSO is already in target list)
            self.target_list_button = QPushButton("Add to Target List")
            self.target_list_button.clicked.connect(self._add_to_target_list)
            
            # Remove from Target List button (initially hidden)
            self.remove_target_button = QPushButton("Remove from Target List")
            self.remove_target_button.clicked.connect(self._remove_from_target_list)
            self.remove_target_button.setVisible(False)
            
            buttons_layout.addWidget(self.target_list_button)
            buttons_layout.addWidget(self.remove_target_button)
            
            # Check if DSO is already in target list and update button visibility
            self._update_target_list_buttons()

            # Add close button at the bottom
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close)
            buttons_layout.addWidget(close_button)

            right_layout.addLayout(buttons_layout)

            # Add right layout to a container widget
            right_container = QWidget()
            right_container.setLayout(right_layout)
            main_layout.addWidget(right_container)

            # Set the main layout
            self.setLayout(main_layout)

            # Defer image loading to improve initial window performance
            if self.data.get('image_path'):
                # Show image information form but defer actual image loading
                self.info_form_container.setVisible(True)
            else:
                # Hide image information form
                self.info_form_container.setVisible(False)

                # Set default visibility for location groupbox - will be updated in deferred calculations
                self.location_groupbox.setVisible(True)
                self.season_label.setText("Loading location information...")

                # Ensure the window is properly sized before showing
                self.adjustSize()
                logger.debug("ObjectDetailWindow setup complete")
        except Exception as e:
            logger.error(f"Error in _setup_ui: {str(e)}", exc_info=True)

    def _open_visibility_calculator(self):
        """Open the DSO Visibility Calculator with the current object pre-loaded"""
        if not VISIBILITY_AVAILABLE:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Feature Unavailable",
                                "DSO Visibility Calculator is not available. "
                                "Please ensure DSOVisibilityCalculator.py is in the same directory.")
            return

        try:
            # Create a formatted object name for the visibility calculator
            object_name = self.data['name']

            # Create new visibility window
            self.visibility_window = CustomDSOVisibilityWindow(object_name, self)
            self.visibility_window.show()

            logger.debug(f"Opened visibility calculator for {object_name}")

        except Exception as e:
            logger.error(f"Error opening visibility calculator: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
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

    def _on_save_location_clicked(self):
        """Handler for saving observer location and updating UI."""
        lat_val = self.location_lat_edit.text().strip()
        lon_val = self.location_lon_edit.text().strip()
        try:
            lat = float(lat_val)
            lon = float(lon_val)
        except ValueError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric latitude and longitude.")
            return

        # Save to DB
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO usersettings (location_lat, location_lon) VALUES (?, ?)", (lat, lon))
                conn.commit()
            self.location_groupbox.setVisible(False)
            self._set_season_label_from_location()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Database Error", f"Could not save location: {str(e)}")

    def _load_user_location(self):
        """Load user location from the usersettings table"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    lat, lon = row
                    self.location_lat_edit.setText(str(lat))
                    self.location_lon_edit.setText(str(lon))
                    logger.debug(f"Loaded user location from DB: lat={lat}, lon={lon}")
                else:
                    logger.debug("No user location found in database")
        except Exception as e:
            logger.error(f"Error loading user location: {str(e)}")

    def _save_user_location(self):
        """Save the user location to the usersettings table"""
        try:
            lat = float(self.location_lat_edit.text().strip())
            lon = float(self.location_lon_edit.text().strip())
        except ValueError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric latitude and longitude.")
            return
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO usersettings (location_lat, location_lon) VALUES (?, ?)", (lat, lon))
                conn.commit()
                logger.debug(f"Saved user location to DB: lat={lat}, lon={lon}")
            self._set_season_label_from_location()
        except Exception as e:
            logger.error(f"Error saving user location: {str(e)}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Database Error", f"Failed to save location: {str(e)}")

    def _set_season_label_from_location(self):
        """
        Set the season_label text with the visibility season/dates string based on user location.
        Uses background thread for heavy calculations.
        """
        try:
            lat_text = self.location_lat_edit.text().strip()
            lon_text = self.location_lon_edit.text().strip()
            if not lat_text or not lon_text:
                self.season_label.setText("Enter your location to see viewing season information.")
                return
            
            lat = float(lat_text)
            lon = float(lon_text)
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
                from PySide6.QtWidgets import QMessageBox
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
                
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", "Image information saved successfully!")

        except Exception as e:
            logger.error(f"Error saving image information: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to save image information: {str(e)}")

    def _add_user_image(self):
        """Add a user image for the object and all its designations"""
        from PySide6.QtWidgets import QFileDialog

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

                        # Insert new image record
                        cursor.execute("""
                            INSERT INTO userimages (
                                dsodetailid, image_path, integration_time, 
                                equipment, date_taken, notes
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            dsodetailid, file_name,
                            self.integration_edit.text().strip(),
                            self.telescope_combo.currentText().strip(),
                            self.date_edit.text().strip(),
                            self.notes_edit.toPlainText().strip()
                        ))

                        # Log all designations that will share this image
                        for catalogue, designation in all_designations:
                            logger.debug(f"Image will be available for {catalogue} {designation}")

                        conn.commit()

                        # Reload all user images to get the updated list
                        self._load_user_images()

                        # Clear the form for the next image
                        self.integration_edit.clear()
                        self.telescope_combo.setCurrentText("")
                        self.date_edit.clear()
                        self.notes_edit.clear()

                        # Show the image information form
                        self.info_form_container.setVisible(True)

                        logger.debug(f"Successfully added user image for {len(all_designations)} designations")

                        # Emit signal to notify that an image was added
                        self.image_added.emit()
                    else:
                        logger.error(f"Could not find dsodetailid for {self.data['name']}")

            except Exception as e:
                logger.error(f"Error adding user image: {str(e)}", exc_info=True)
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to add image: {str(e)}")

    def _load_fits_image(self, fits_path, colormap='viridis'):
        """
        Load a FITS image file and convert to QPixmap with color mapping.
        
        Args:
            fits_path (str): Path to the FITS file
            colormap (str): Matplotlib colormap name ('viridis', 'hot', 'cool', 'plasma', 'inferno', 'gray')
        """
        try:
            logger.debug(f"Loading FITS file: {fits_path}")
            
            # Import required libraries
            from astropy.io import fits
            from astropy.visualization import simple_norm
            import numpy as np
            from PySide6.QtGui import QImage
            
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
        """Load and display a user image with caching (supports FITS files)"""
        try:
            from PySide6.QtGui import QImageReader
            import os
            
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

            # Increase maximum allocation limit for large images (in MB)
            QImageReader.setAllocationLimit(512)

            # Check cache first
            cached_pixmap = self.image_cache.get(image_path)
            if cached_pixmap:
                logger.debug("Using cached image")
                self.original_pixmap = cached_pixmap
            else:
                # Check if this is a FITS file
                file_ext = os.path.splitext(image_path)[1].lower()
                if file_ext in ['.fits', '.fit', '.fts']:
                    # Load FITS file
                    logger.debug("Loading FITS image")
                    self.original_pixmap = self._load_fits_image(image_path, self.FITS_COLORMAP)
                else:
                    # Create QPixmap from regular image file
                    logger.debug("Loading regular image from file")
                    self.original_pixmap = QPixmap(image_path)
                
                if self.original_pixmap is None or self.original_pixmap.isNull():
                    logger.error(f"Failed to load image: {image_path}")
                    
                    if file_ext in ['.fits', '.fit', '.fts']:
                        # FITS file failed to load
                        self.image_label.setText(f"Failed to load FITS file:\n{os.path.basename(image_path)}\n\nRequires astropy library for FITS support")
                    else:
                        # Try to get more specific error information for regular images
                        try:
                            reader = QImageReader(image_path)
                            if reader.canRead():
                                logger.debug(f"QImageReader can read the file, format: {reader.format()}")
                            else:
                                logger.error(f"QImageReader cannot read file, error: {reader.errorString()}")
                        except Exception as e:
                            logger.error(f"Error checking image readability: {e}")
                        
                        self.image_label.setText(f"Failed to load image:\n{os.path.basename(image_path)}\n\nCheck if file is a valid image format")
                    
                    self.image_label.setToolTip(f"File path: {image_path}")
                    return
                
                logger.debug(f"Successfully loaded image: {self.original_pixmap.width()}x{self.original_pixmap.height()}")
                # Cache the loaded image
                self.image_cache.put(image_path, self.original_pixmap)

            # Calculate the appropriate size for the image
            label_size = self.image_label.size()
            scaled_pixmap = self.original_pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
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

        except Exception as e:
            logger.error(f"Error loading user image: {str(e)}", exc_info=True)
            self.image_label.setText(f"Error loading image:\n{os.path.basename(image_path) if image_path else 'Unknown'}")
            self.image_label.setToolTip(f"File path: {image_path}\nError: {str(e)}")  # Clear tooltip on error

    def _open_aladin_lite(self, data):
        """Open Aladin Lite in a new window"""
        try:
            # Store reference to prevent garbage collection and manage window lifecycle
            if not hasattr(self, 'aladin_window') or not self.aladin_window.isVisible():
                self.aladin_window = AladinLiteWindow(data, self)
                self.aladin_window.show()
                logger.debug(f"Opened Aladin Lite window for {data['name']}")
            else:
                # If window is already open, bring it to front and update data
                self.aladin_window.raise_()
                self.aladin_window.activateWindow()
                logger.debug(f"Aladin Lite window already open, bringing to front")
        except Exception as e:
            logger.error(f"Error opening Aladin Lite: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

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
                    # Calculate the drag distance
                    dx = event.position().x() - self.drag_start_position.x()
                    dy = event.position().y() - self.drag_start_position.y()

                    # Update image position
                    self.image_position[0] = self.drag_start_image_position[0] + dx
                    self.image_position[1] = self.drag_start_image_position[1] + dy

                    # Update the display
                    self._update_zoom()
                    return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.drag_start_position = None
                self.drag_start_image_position = None
                return True
            elif event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                # Open image in new window
                self._open_image_viewer()
                return True
        return super().eventFilter(obj, event)

    def _update_zoom(self):
        """Update the image display with current zoom level and position"""
        if self.original_pixmap is not None:
            # Calculate new size
            new_width = int(self.original_pixmap.width() * self.zoom_factor)
            new_height = int(self.original_pixmap.height() * self.zoom_factor)

            # Scale the image
            scaled_pixmap = self.original_pixmap.scaled(
                new_width,
                new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Create a new pixmap with the same size as the label
            label_size = self.image_label.size()
            final_pixmap = QPixmap(label_size)
            final_pixmap.fill(Qt.transparent)

            # Create a painter to draw the scaled image at the correct position
            painter = QPainter(final_pixmap)

            # Calculate the position to draw the image
            x = (label_size.width() - scaled_pixmap.width()) // 2 + self.image_position[0]
            y = (label_size.height() - scaled_pixmap.height()) // 2 + self.image_position[1]

            # Draw the image
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()

            # Update the display
            self.image_label.setPixmap(final_pixmap)

    def _zoom_in(self):
        """Zoom in on the image"""
        if self.original_pixmap is not None:
            # Calculate new zoom factor relative to initial zoom
            self.zoom_factor = min(self.zoom_factor * 1.2, 4.0 * self.initial_zoom_factor)
            self._update_zoom()

    def _zoom_out(self):
        """Zoom out on the image"""
        if self.original_pixmap is not None:
            # Calculate new zoom factor relative to initial zoom
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
                viewer = ImageViewerWindow(self.original_pixmap, self.data["name"], self.current_image_path, self)
                viewer.setModal(False)  # Make window non-modal
                viewer.show()
                logger.debug(f"Opened image viewer for {self.data['name']}")
            except Exception as e:
                logger.error(f"Error opening image viewer: {str(e)}", exc_info=True)
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to open image viewer: {str(e)}")

    def _load_user_images(self):
        """Load all user images for this DSO from the database"""
        try:
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

                    # Get all images for this object
                    cursor.execute("""
                        SELECT id, image_path, integration_time, equipment, date_taken, notes
                        FROM userimages 
                        WHERE dsodetailid = ?
                        ORDER BY id ASC
                    """, (dsodetailid,))
                    
                    images = cursor.fetchall()
                    
                    # Store images in our list
                    self.user_images = []
                    for img_id, image_path, integration_time, equipment, date_taken, notes in images:
                        self.user_images.append({
                            'id': img_id,
                            'image_path': image_path,
                            'integration_time': integration_time,
                            'equipment': equipment,
                            'date_taken': date_taken,
                            'notes': notes
                        })

                    logger.debug(f"Loaded {len(self.user_images)} images for {self.data['name']}")

                    # Update navigation controls
                    self._update_image_navigation()

                    # Load the first image if available
                    if self.user_images:
                        self.current_image_index = 0
                        current_image = self.user_images[self.current_image_index]
                        self._load_user_image(current_image['image_path'])
                        self._load_current_image_info()
                        self.info_form_container.setVisible(True)
                    else:
                        # No images available
                        self.info_form_container.setVisible(False)
                else:
                    logger.error(f"Could not find dsodetailid for {self.data['name']}")

        except Exception as e:
            logger.error(f"Error loading user images: {str(e)}", exc_info=True)

    def _update_image_navigation(self):
        """Update the image navigation controls based on current state"""
        image_count = len(self.user_images)
        
        if image_count == 0:
            # No images
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.delete_image_button.setEnabled(False)
            self.image_counter_label.setText("0/0")
        elif image_count == 1:
            # Only one image
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.delete_image_button.setEnabled(True)
            self.image_counter_label.setText("1/1")
        else:
            # Multiple images
            self.prev_image_button.setEnabled(self.current_image_index > 0)
            self.next_image_button.setEnabled(self.current_image_index < image_count - 1)
            self.delete_image_button.setEnabled(True)
            self.image_counter_label.setText(f"{self.current_image_index + 1}/{image_count}")

    def _previous_image(self):
        """Navigate to the previous image"""
        if self.user_images and self.current_image_index > 0:
            self.current_image_index -= 1
            current_image = self.user_images[self.current_image_index]
            self._load_user_image(current_image['image_path'])
            self._load_current_image_info()
            self._update_image_navigation()
            logger.debug(f"Navigated to previous image: {self.current_image_index + 1}/{len(self.user_images)}")

    def _next_image(self):
        """Navigate to the next image"""
        if self.user_images and self.current_image_index < len(self.user_images) - 1:
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

    def _show_relocate_button(self):
        """Show the relocate button when an image cannot be found"""
        self.relocate_button.setVisible(True)
        logger.debug("Relocate button shown due to missing image file")

    def _relocate_image(self):
        """Allow user to select new location for missing image file"""
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            
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
            
            # Get original filename for the dialog
            import os
            original_filename = os.path.basename(self.current_image_path)
            
            # Open file dialog to select new image location
            new_image_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select new location for {original_filename}",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.fits *.fit *.fts);;Regular Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif);;FITS Files (*.fits *.fit *.fts);;All Files (*)"
            )
            
            if new_image_path:
                # Update the database with new path
                try:
                    self.db_manager.execute_update(
                        "UPDATE userimages SET image_path = ? WHERE id = ?",
                        (new_image_path, image_id)
                    )
                    
                    # Update current path and reload image
                    self.current_image_path = new_image_path
                    current_image['image_path'] = new_image_path
                    
                    # Hide relocate button and reload the image
                    self.relocate_button.setVisible(False)
                    self._load_user_image(new_image_path)
                    
                    QMessageBox.information(self, "Success", f"Image location updated successfully!\n\nNew path: {new_image_path}")
                    logger.info(f"Image relocated successfully from {self.current_image_path} to {new_image_path}")
                    
                except Exception as db_error:
                    logger.error(f"Database error during image relocation: {str(db_error)}")
                    QMessageBox.critical(self, "Database Error", f"Failed to update image location in database:\n{str(db_error)}")
                    
        except Exception as e:
            logger.error(f"Error relocating image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to relocate image:\n{str(e)}")

    def _delete_current_image(self):
        """Delete the current image from the database and update the display"""
        try:
            from PySide6.QtWidgets import QMessageBox
            
            # Check if we have images to delete
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
            
            # Get the filename for the confirmation dialog
            import os
            filename = os.path.basename(image_path) if image_path != 'Unknown' else 'this image'
            
            # Confirm deletion
            reply = QMessageBox.question(
                self, 
                "Delete Image", 
                f"Are you sure you want to delete '{filename}'?\n\nThis will remove the image record from the database but will not delete the actual image file.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Delete from database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM userimages WHERE id = ?", (image_id,))
                conn.commit()
                
                # Remove from local list
                del self.user_images[self.current_image_index]
                
                # Update current image index
                if self.user_images:
                    # Adjust index if we deleted the last image
                    if self.current_image_index >= len(self.user_images):
                        self.current_image_index = len(self.user_images) - 1
                    
                    # Load the new current image
                    current_image = self.user_images[self.current_image_index]
                    self._load_user_image(current_image['image_path'])
                    self._load_current_image_info()
                else:
                    # No more images - reset to default state
                    self.current_image_index = 0
                    self.image_label.setText("No Image Loaded")
                    self.image_label.setStyleSheet("font-size: 14pt; color: gray;")
                    self._clear_image_info()
                
                # Update navigation
                self._update_image_navigation()
                
                QMessageBox.information(self, "Success", f"'{filename}' has been removed from the database.")
                logger.info(f"Deleted image with ID {image_id}: {filename}")
                
        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to delete image:\n{str(e)}")
    
    def _clear_image_info(self):
        """Clear all image information fields"""
        self.integration_edit.setText("")
        self.telescope_combo.setCurrentText("")
        self.date_edit.setText("")
        self.notes_edit.setText("")

    def _update_target_list_buttons(self):
        """Update target list button visibility based on whether DSO is already in target list"""
        try:
            is_in_target_list = self._check_if_in_target_list()
            
            if is_in_target_list:
                self.target_list_button.setVisible(False)
                self.remove_target_button.setVisible(True)
            else:
                self.target_list_button.setVisible(True)
                self.remove_target_button.setVisible(False)
                
        except Exception as e:
            logger.error(f"Error updating target list buttons: {str(e)}")
            # Show add button as fallback
            self.target_list_button.setVisible(True)
            self.remove_target_button.setVisible(False)
    
    def _check_if_in_target_list(self):
        """Check if current DSO is already in the target list"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                
                # Check by DSO name and coordinates (to handle different name formats)
                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')
                
                if not dso_name:
                    return False
                
                # First check by exact name match
                cursor.execute("""
                    SELECT COUNT(*) FROM usertargetlist 
                    WHERE UPPER(TRIM(name)) = ?
                """, (dso_name.upper(),))
                
                if cursor.fetchone()[0] > 0:
                    return True
                
                # If coordinates are available, also check by coordinates (within small tolerance)
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
    
    def _add_to_target_list(self):
        """Add this DSO to the target list with pre-calculated visibility information"""
        try:
            # Import DSOTargetList module
            from DSOTargetList import AddTargetDialog
            
            # Use already calculated visibility information from the season label
            enhanced_data = self.data.copy()
            visibility_text = self.season_label.text()
            
            # Extract useful visibility information if available
            if visibility_text and not visibility_text.startswith("Enter your location") and not visibility_text.startswith("Loading"):
                # Clean the visibility text to extract only month ranges
                cleaned_months = self._extract_month_ranges_from_visibility(visibility_text)
                if cleaned_months:
                    enhanced_data['best_months'] = cleaned_months
                    logger.debug(f"Using cleaned visibility info for {self.data.get('name', 'DSO')}: {cleaned_months}")
            
            # Create dialog with enhanced DSO data including visibility
            dialog = AddTargetDialog(dso_data=enhanced_data, parent=self)
            if dialog.exec():
                # Update button visibility after successful addition
                self._update_target_list_buttons()
                
        except ImportError as e:
            logger.error(f"Could not import DSOTargetList: {str(e)}")
            QMessageBox.warning(self, "Import Error", f"Could not load Target List feature: {e}")
        except Exception as e:
            logger.error(f"Error adding to target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {str(e)}")
    
    def _extract_month_ranges_from_visibility(self, visibility_text):
        """Extract only month ranges from visibility text, removing HTML and descriptive text"""
        import re
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', visibility_text)
        
        # Skip if it contains error messages
        if "not optimally visible" in clean_text.lower() or "error" in clean_text.lower():
            return ""
        
        # Extract date ranges that contain month names
        # Pattern matches things like "January 15 - March 20", "October 01 - December 31"
        month_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+\s*-\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+'
        
        matches = re.findall(month_pattern, clean_text)
        
        if matches:
            # Convert full month names to abbreviated form and create ranges
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
            # Confirm removal
            reply = QMessageBox.question(
                self, 
                "Remove from Target List", 
                f"Remove '{self.data.get('name', 'this DSO')}' from the target list?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
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
                
                # Remove by name first
                cursor.execute("""
                    DELETE FROM usertargetlist 
                    WHERE UPPER(TRIM(name)) = ?
                """, (dso_name.upper(),))
                
                # Also remove by coordinates if available (to catch any duplicates)
                if ra_deg is not None and dec_deg is not None:
                    cursor.execute("""
                        DELETE FROM usertargetlist 
                        WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001
                    """, (ra_deg, dec_deg))
                
                conn.commit()
                
                # Update button visibility
                self._update_target_list_buttons()
                
                QMessageBox.information(self, "Success", f"'{dso_name}' removed from target list")
                logger.debug(f"Removed {dso_name} from target list")
                
        except Exception as e:
            logger.error(f"Error removing from target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove from target list: {str(e)}")
    
    def _create_collage(self):
        """Open the CollageBuilder window for this DSO with option to create new or add to existing collage"""
        try:
            # Check if there are any user images
            if not self.user_images:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "No Images", 
                    "No images have been added for this object yet. "
                    "Add some images first using the + add image button.")
                return
            
            # Get the dsodetailid for this object
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                logger.debug(f"Looking up dsodetailid for catalogue: {self.data['catalogue']}, designation: {self.data['id']}")
                cursor.execute("""
                    SELECT d.id 
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ? AND c.designation = ?
                """, (self.data['catalogue'], self.data['id']))
                result = cursor.fetchone()
                
                if result:
                    dsodetailid = result[0]
                    logger.debug(f"Found dsodetailid: {dsodetailid} for {self.data['name']}")
                else:
                    logger.error(f"Could not find dsodetailid for {self.data['name']} (catalogue: {self.data['catalogue']}, designation: {self.data['id']})")
                    QMessageBox.critical(self, "Error", "Could not determine DSO ID for creating collage.")
                    return
            
            # Get all user images from the database (not just for this DSO)
            all_user_images = []
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ui.id, ui.image_path, ui.integration_time, ui.equipment,
                           ui.date_taken, ui.notes, c.designation as dso_name
                    FROM userimages ui
                    LEFT JOIN cataloguenr c ON ui.dsodetailid = c.dsodetailid
                    ORDER BY ui.id DESC
                """)
                rows = cursor.fetchall()
                for row in rows:
                    all_user_images.append({
                        'id': row[0],
                        'image_path': row[1],
                        'integration_time': row[2] or '',
                        'equipment': row[3] or '',
                        'date_taken': row[4] or '',
                        'notes': row[5] or '',
                        'dso_name': row[6] or 'Unknown DSO'
                    })

            # Show collage selection dialog
            dialog = CollageSelectionDialog(dsodetailid, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                action, collage_data = dialog.get_selection()

                if action == "new":
                    # Create new collage with all user images
                    self.collage_builder_window = CollageBuilderWindow(all_user_images, self.data['name'], dsodetailid, self)
                    self.collage_builder_window.setModal(False)
                    self.collage_builder_window.show()
                    logger.debug(f"Opened new CollageBuilder window for {self.data['name']} with {len(all_user_images)} images")

                elif action == "existing" and collage_data:
                    # Load existing collage and add current images
                    self.collage_builder_window = CollageBuilderWindow(all_user_images, self.data['name'], dsodetailid, self)
                    
                    # Load the existing collage data
                    if self.collage_builder_window._load_collage_data(collage_data):
                        self.collage_builder_window.setModal(False)
                        self.collage_builder_window.show()
                        logger.debug(f"Loaded existing collage '{collage_data['name']}' with {len(self.user_images)} images available")
                    else:
                        QMessageBox.critical(self, "Error", "Failed to load existing collage data.")
                
        except Exception as e:
            logger.error(f"Error opening collage builder: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to open collage builder: {str(e)}")


class CustomTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None

    def setModel(self, model):
        super().setModel(model)
        self.model = model
        # Connect the header's sort indicator change to the model's sort method
        self.horizontalHeader().sortIndicatorChanged.connect(self._on_sort_indicator_changed)

    def _on_sort_indicator_changed(self, logical_index, order):
        """Handle sort indicator changes by calling the model's sort method"""
        if self.model:
            self.model.sort(logical_index, order)


# --- Settings Dialog ---
class SettingsDialog(QDialog):
    """Settings dialog for configuring application preferences"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(500, 400)
        
        self.db_manager = DatabaseManager()
        self._setup_ui()
        self._load_current_settings()
        
    def _setup_ui(self):
        """Set up the settings dialog UI"""
        layout = QVBoxLayout()
        
        # Create tab widget for different setting categories
        from PySide6.QtWidgets import QTabWidget
        tab_widget = QTabWidget()
        
        # Location settings tab
        location_tab = QWidget()
        location_layout = QVBoxLayout(location_tab)
        
        # Location group
        location_group = QGroupBox("Observer Location")
        location_group_layout = QVBoxLayout(location_group)
        
        # Latitude input
        lat_layout = QHBoxLayout()
        lat_label = QLabel("Latitude (degrees):")
        lat_label.setMinimumWidth(120)
        self.latitude_input = QLineEdit()
        self.latitude_input.setPlaceholderText("e.g., 40.7128 (positive for North, negative for South)")
        lat_layout.addWidget(lat_label)
        lat_layout.addWidget(self.latitude_input)
        location_group_layout.addLayout(lat_layout)
        
        # Longitude input
        lon_layout = QHBoxLayout()
        lon_label = QLabel("Longitude (degrees):")
        lon_label.setMinimumWidth(120)
        self.longitude_input = QLineEdit()
        self.longitude_input.setPlaceholderText("e.g., -74.0060 (positive for East, negative for West)")
        lon_layout.addWidget(lon_label)
        lon_layout.addWidget(self.longitude_input)
        location_group_layout.addLayout(lon_layout)
        
        # Location name (optional)
        name_layout = QHBoxLayout()
        name_label = QLabel("Location Name:")
        name_label.setMinimumWidth(120)
        self.location_name_input = QLineEdit()
        self.location_name_input.setPlaceholderText("e.g., New York City (optional)")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.location_name_input)
        location_group_layout.addLayout(name_layout)
        
        location_layout.addWidget(location_group)
        
        # Timezone settings group
        timezone_group = QGroupBox("Time Zone Settings")
        timezone_group_layout = QVBoxLayout(timezone_group)
        
        tz_layout = QHBoxLayout()
        tz_label = QLabel("Time Zone:")
        tz_label.setMinimumWidth(120)
        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
        
        # Add common timezones
        common_timezones = [
            "America/New_York",
            "America/Chicago", 
            "America/Denver",
            "America/Los_Angeles",
            "America/Phoenix",
            "America/Anchorage",
            "Pacific/Honolulu",
            "UTC",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Australia/Sydney"
        ]
        self.timezone_combo.addItems(common_timezones)
        tz_layout.addWidget(tz_label)
        tz_layout.addWidget(self.timezone_combo)
        timezone_group_layout.addLayout(tz_layout)
        
        location_layout.addWidget(timezone_group)
        
        # Help text
        help_text = QLabel("""
<b>Tips:</b>
• Latitude: Positive values for Northern Hemisphere, negative for Southern
• Longitude: Positive values for Eastern Hemisphere, negative for Western  
• You can find coordinates using online tools like Google Maps
• Time zone affects visibility calculation displays and times
        """)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { color: #888888; font-size: 9pt; }")
        location_layout.addWidget(help_text)
        
        location_layout.addStretch()
        tab_widget.addTab(location_tab, "Location && Time Zone")
        
        layout.addWidget(tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Test location button
        self.test_button = QPushButton("Test Location")
        self.test_button.clicked.connect(self._test_location)
        button_layout.addWidget(self.test_button)
        
        button_layout.addStretch()
        
        # Standard dialog buttons
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_settings)
        self.save_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def _load_current_settings(self):
        """Load current settings from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    lat, lon = row
                    self.latitude_input.setText(str(lat))
                    self.longitude_input.setText(str(lon))
                
                # Try to get location name if it exists (we'll need to add this column)
                try:
                    cursor.execute("PRAGMA table_info(usersettings)")
                    columns = [column[1] for column in cursor.fetchall()]
                    if 'location_name' in columns:
                        cursor.execute("SELECT location_name FROM usersettings ORDER BY id DESC LIMIT 1")
                        name_row = cursor.fetchone()
                        if name_row and name_row[0]:
                            self.location_name_input.setText(name_row[0])
                    
                    if 'timezone' in columns:
                        cursor.execute("SELECT timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                        tz_row = cursor.fetchone()
                        if tz_row and tz_row[0]:
                            # Set timezone in combo box
                            index = self.timezone_combo.findText(tz_row[0])
                            if index >= 0:
                                self.timezone_combo.setCurrentIndex(index)
                            else:
                                self.timezone_combo.setEditText(tz_row[0])
                except Exception:
                    # Columns don't exist yet, that's ok
                    pass
                    
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            
    def _test_location(self):
        """Test if the entered coordinates are valid"""
        try:
            lat_text = self.latitude_input.text().strip()
            lon_text = self.longitude_input.text().strip()
            
            if not lat_text or not lon_text:
                QMessageBox.warning(self, "Invalid Input", "Please enter both latitude and longitude.")
                return
                
            lat = float(lat_text)
            lon = float(lon_text)
            
            # Validate ranges
            if not (-90 <= lat <= 90):
                QMessageBox.warning(self, "Invalid Latitude", "Latitude must be between -90 and 90 degrees.")
                return
                
            if not (-180 <= lon <= 180):
                QMessageBox.warning(self, "Invalid Longitude", "Longitude must be between -180 and 180 degrees.")
                return
            
            # Format coordinates nicely for display
            lat_str = f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):.4f}°{'W' if lon < 0 else 'E'}"
            
            QMessageBox.information(self, "Location Test", 
                f"Location coordinates are valid!\n\n"
                f"Latitude: {lat_str}\n"
                f"Longitude: {lon_str}\n\n"
                f"These coordinates will be used for visibility calculations.")
                
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                "Please enter valid numeric values for latitude and longitude.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error testing location: {str(e)}")
            
    def _save_settings(self):
        """Save settings to database"""
        try:
            lat_text = self.latitude_input.text().strip()
            lon_text = self.longitude_input.text().strip()
            
            if not lat_text or not lon_text:
                QMessageBox.warning(self, "Missing Information", 
                    "Please enter both latitude and longitude before saving.")
                return
                
            lat = float(lat_text)
            lon = float(lon_text)
            
            # Validate ranges
            if not (-90 <= lat <= 90):
                QMessageBox.warning(self, "Invalid Latitude", 
                    "Latitude must be between -90 and 90 degrees.")
                return
                
            if not (-180 <= lon <= 180):
                QMessageBox.warning(self, "Invalid Longitude", 
                    "Longitude must be between -180 and 180 degrees.")
                return
            
            # Get additional settings
            location_name = self.location_name_input.text().strip() or None
            timezone = self.timezone_combo.currentText().strip() or "America/New_York"
            
            # Validate timezone
            try:
                import pytz
                pytz.timezone(timezone)  # This will raise an exception if invalid
            except Exception:
                QMessageBox.warning(self, "Invalid Timezone", 
                    f"'{timezone}' is not a valid timezone identifier.")
                return
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # First, ensure the new columns exist
                try:
                    cursor.execute("ALTER TABLE usersettings ADD COLUMN location_name TEXT")
                except Exception:
                    pass  # Column already exists
                    
                try:
                    cursor.execute("ALTER TABLE usersettings ADD COLUMN timezone TEXT")
                except Exception:
                    pass  # Column already exists
                
                # Insert new settings
                cursor.execute("""
                    INSERT INTO usersettings (location_lat, location_lon, location_name, timezone) 
                    VALUES (?, ?, ?, ?)
                """, (lat, lon, location_name, timezone))
                conn.commit()
                
            QMessageBox.information(self, "Settings Saved", 
                "Your location and timezone settings have been saved successfully!\n\n"
                "The new settings will be used for all visibility calculations.")
            
            self.accept()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                "Please enter valid numeric values for latitude and longitude.")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")


# --- Telescope Management Dialog ---
class TelescopeDialog(QDialog):
    """Dialog for managing user telescopes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Telescope Management - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(800, 600)
        
        self.db_manager = DatabaseManager()
        self._setup_ui()
        self._load_telescopes()
        
    def _setup_ui(self):
        """Set up the telescope management UI"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Telescope Management")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Create main horizontal layout
        main_layout = QHBoxLayout()
        
        # Left side - telescope list
        list_layout = QVBoxLayout()
        
        list_label = QLabel("Your Telescopes:")
        list_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        list_layout.addWidget(list_label)
        
        # Telescope list
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        self.telescope_list = QListWidget()
        self.telescope_list.itemSelectionChanged.connect(self._on_telescope_selected)
        list_layout.addWidget(self.telescope_list)
        
        # List action buttons
        list_button_layout = QHBoxLayout()
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._delete_telescope)
        self.delete_button.setEnabled(False)
        list_button_layout.addWidget(self.delete_button)
        
        self.set_active_button = QPushButton("Set as Active")
        self.set_active_button.clicked.connect(self._set_active_telescope)
        self.set_active_button.setEnabled(False)
        list_button_layout.addWidget(self.set_active_button)
        
        list_layout.addLayout(list_button_layout)
        main_layout.addLayout(list_layout)
        
        # Right side - telescope form
        form_layout = QVBoxLayout()
        
        form_label = QLabel("Add/Edit Telescope:")
        form_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        form_layout.addWidget(form_label)
        
        # Form group
        form_group = QGroupBox("Telescope Details")
        form_group_layout = QVBoxLayout(form_group)
        
        # Telescope name
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setMinimumWidth(100)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., My Celestron 8SE")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        form_group_layout.addLayout(name_layout)
        
        # Aperture
        aperture_layout = QHBoxLayout()
        aperture_label = QLabel("Aperture (mm):")
        aperture_label.setMinimumWidth(100)
        self.aperture_input = QLineEdit()
        self.aperture_input.setPlaceholderText("e.g., 203.2")
        aperture_layout.addWidget(aperture_label)
        aperture_layout.addWidget(self.aperture_input)
        form_group_layout.addLayout(aperture_layout)
        
        # Focal length
        focal_layout = QHBoxLayout()
        focal_label = QLabel("Focal Length (mm):")
        focal_label.setMinimumWidth(100)
        self.focal_length_input = QLineEdit()
        self.focal_length_input.setPlaceholderText("e.g., 2032")
        focal_layout.addWidget(focal_label)
        focal_layout.addWidget(self.focal_length_input)
        form_group_layout.addLayout(focal_layout)
        
        # F-ratio (calculated)
        fratio_layout = QHBoxLayout()
        fratio_label = QLabel("F-ratio:")
        fratio_label.setMinimumWidth(100)
        self.fratio_display = QLabel("N/A")
        self.fratio_display.setStyleSheet("color: #888888;")
        fratio_layout.addWidget(fratio_label)
        fratio_layout.addWidget(self.fratio_display)
        form_group_layout.addLayout(fratio_layout)
        
        # Connect aperture and focal length inputs to calculate F-ratio
        self.aperture_input.textChanged.connect(self._calculate_fratio)
        self.focal_length_input.textChanged.connect(self._calculate_fratio)
        
        # Mount type
        mount_layout = QHBoxLayout()
        mount_label = QLabel("Mount Type:")
        mount_label.setMinimumWidth(100)
        self.mount_combo = QComboBox()
        self.mount_combo.addItems([
            "Alt-Az",
            "Equatorial (German)",
            "Equatorial (Fork)",
            "Dobsonian",
            "Tripod",
            "Pier",
            "Other"
        ])
        mount_layout.addWidget(mount_label)
        mount_layout.addWidget(self.mount_combo)
        form_group_layout.addLayout(mount_layout)
        
        # Notes
        notes_layout = QVBoxLayout()
        notes_label = QLabel("Notes:")
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Additional notes about your telescope...")
        self.notes_input.setMaximumHeight(80)
        notes_layout.addWidget(notes_label)
        notes_layout.addWidget(self.notes_input)
        form_group_layout.addLayout(notes_layout)
        
        form_layout.addWidget(form_group)
        
        # Form action buttons
        form_button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Form")
        self.clear_button.clicked.connect(self._clear_form)
        form_button_layout.addWidget(self.clear_button)
        
        form_button_layout.addStretch()
        
        self.save_button = QPushButton("Save Telescope")
        self.save_button.clicked.connect(self._save_telescope)
        self.save_button.setDefault(True)
        form_button_layout.addWidget(self.save_button)
        
        form_layout.addLayout(form_button_layout)
        main_layout.addLayout(form_layout)
        
        layout.addLayout(main_layout)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        help_text = QLabel("Tip: Set one telescope as 'Active' to use it as the default for calculations.")
        help_text.setStyleSheet("color: #888888; font-size: 9pt;")
        bottom_layout.addWidget(help_text)
        
        bottom_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_layout.addWidget(close_button)
        
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
        
        # Track current editing telescope
        self.current_telescope_id = None
        
    def _calculate_fratio(self):
        """Calculate and display F-ratio based on aperture and focal length"""
        try:
            aperture_text = self.aperture_input.text().strip()
            focal_text = self.focal_length_input.text().strip()
            
            if aperture_text and focal_text:
                aperture = float(aperture_text)
                focal_length = float(focal_text)
                
                if aperture > 0:
                    fratio = focal_length / aperture
                    self.fratio_display.setText(f"f/{fratio:.1f}")
                    self.fratio_display.setStyleSheet("color: #ffffff; font-weight: bold;")
                else:
                    self.fratio_display.setText("N/A")
                    self.fratio_display.setStyleSheet("color: #888888;")
            else:
                self.fratio_display.setText("N/A")
                self.fratio_display.setStyleSheet("color: #888888;")
        except ValueError:
            self.fratio_display.setText("N/A")
            self.fratio_display.setStyleSheet("color: #888888;")
    
    def _load_telescopes(self):
        """Load telescopes from database into the list"""
        try:
            self.telescope_list.clear()
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length, mount_type, is_active 
                    FROM usertelescopes 
                    ORDER BY is_active DESC, name ASC
                """)
                
                telescopes = cursor.fetchall()
                
                for telescope_id, name, aperture, focal_length, mount_type, is_active in telescopes:
                    # Create list item text
                    fratio = focal_length / aperture if aperture and aperture > 0 else 0
                    status = " (Active)" if is_active else ""
                    
                    item_text = f"{name}{status}"
                    if aperture:
                        item_text += f" - {aperture}mm"
                    if fratio > 0:
                        item_text += f" f/{fratio:.1f}"
                    
                    # Create list item
                    from PySide6.QtWidgets import QListWidgetItem
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, telescope_id)  # Store telescope ID
                    
                    # Highlight active telescope
                    if is_active:
                        item.setBackground(QColor(0, 120, 212, 50))  # Light blue background
                        
                    self.telescope_list.addItem(item)
                    
        except Exception as e:
            logger.error(f"Error loading telescopes: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Failed to load telescopes: {str(e)}")
    
    def _on_telescope_selected(self):
        """Handle telescope selection"""
        selected_items = self.telescope_list.selectedItems()
        if selected_items:
            self.delete_button.setEnabled(True)
            self.set_active_button.setEnabled(True)
            
            # Load telescope data into form
            telescope_id = selected_items[0].data(Qt.UserRole)
            self._load_telescope_into_form(telescope_id)
        else:
            self.delete_button.setEnabled(False)
            self.set_active_button.setEnabled(False)
    
    def _load_telescope_into_form(self, telescope_id):
        """Load telescope data into the form for editing"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name, aperture, focal_length, mount_type, notes 
                    FROM usertelescopes 
                    WHERE id = ?
                """, (telescope_id,))
                
                row = cursor.fetchone()
                if row:
                    name, aperture, focal_length, mount_type, notes = row
                    
                    self.name_input.setText(name or "")
                    self.aperture_input.setText(str(aperture) if aperture else "")
                    self.focal_length_input.setText(str(focal_length) if focal_length else "")
                    
                    # Set mount type
                    mount_index = self.mount_combo.findText(mount_type or "")
                    if mount_index >= 0:
                        self.mount_combo.setCurrentIndex(mount_index)
                    
                    self.notes_input.setPlainText(notes or "")
                    
                    # Set current editing ID
                    self.current_telescope_id = telescope_id
                    
                    # Update save button text
                    self.save_button.setText("Update Telescope")
                    
        except Exception as e:
            logger.error(f"Error loading telescope data: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Failed to load telescope data: {str(e)}")
    
    def _clear_form(self):
        """Clear the form fields"""
        self.name_input.clear()
        self.aperture_input.clear()
        self.focal_length_input.clear()
        self.mount_combo.setCurrentIndex(0)
        self.notes_input.clear()
        self.current_telescope_id = None
        self.save_button.setText("Save Telescope")
        
        # Clear selection
        self.telescope_list.clearSelection()
        
    def _save_telescope(self):
        """Save or update telescope"""
        try:
            name = self.name_input.text().strip()
            aperture_text = self.aperture_input.text().strip()
            focal_length_text = self.focal_length_input.text().strip()
            mount_type = self.mount_combo.currentText()
            notes = self.notes_input.toPlainText().strip()
            
            # Validate required fields
            if not name:
                QMessageBox.warning(self, "Invalid Input", "Please enter a telescope name.")
                return
                
            # Parse numeric fields (optional)
            aperture = None
            focal_length = None
            
            if aperture_text:
                try:
                    aperture = float(aperture_text)
                    if aperture <= 0:
                        QMessageBox.warning(self, "Invalid Input", "Aperture must be a positive number.")
                        return
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter a valid aperture value.")
                    return
                    
            if focal_length_text:
                try:
                    focal_length = float(focal_length_text)
                    if focal_length <= 0:
                        QMessageBox.warning(self, "Invalid Input", "Focal length must be a positive number.")
                        return
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter a valid focal length value.")
                    return
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.current_telescope_id:
                    # Update existing telescope
                    cursor.execute("""
                        UPDATE usertelescopes 
                        SET name = ?, aperture = ?, focal_length = ?, mount_type = ?, notes = ?
                        WHERE id = ?
                    """, (name, aperture, focal_length, mount_type, notes, self.current_telescope_id))
                    
                    QMessageBox.information(self, "Success", f"Telescope '{name}' has been updated successfully!")
                else:
                    # Insert new telescope
                    cursor.execute("""
                        INSERT INTO usertelescopes (name, aperture, focal_length, mount_type, notes) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, aperture, focal_length, mount_type, notes))
                    
                    QMessageBox.information(self, "Success", f"Telescope '{name}' has been added successfully!")
                
                conn.commit()
                
            # Reload telescopes and clear form
            self._load_telescopes()
            self._clear_form()
            
        except Exception as e:
            logger.error(f"Error saving telescope: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save telescope: {str(e)}")
    
    def _delete_telescope(self):
        """Delete selected telescope"""
        selected_items = self.telescope_list.selectedItems()
        if not selected_items:
            return
            
        telescope_id = selected_items[0].data(Qt.UserRole)
        telescope_name = selected_items[0].text().split(" (")[0]  # Remove status text
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Are you sure you want to delete telescope '{telescope_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM usertelescopes WHERE id = ?", (telescope_id,))
                    conn.commit()
                
                QMessageBox.information(self, "Success", f"Telescope '{telescope_name}' has been deleted.")
                
                # Reload telescopes and clear form
                self._load_telescopes()
                self._clear_form()
                
            except Exception as e:
                logger.error(f"Error deleting telescope: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to delete telescope: {str(e)}")
    
    def _set_active_telescope(self):
        """Set selected telescope as active"""
        selected_items = self.telescope_list.selectedItems()
        if not selected_items:
            return
            
        telescope_id = selected_items[0].data(Qt.UserRole)
        telescope_name = selected_items[0].text().split(" (")[0]  # Remove status text
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # First, set all telescopes as inactive
                cursor.execute("UPDATE usertelescopes SET is_active = 0")
                
                # Then set the selected telescope as active
                cursor.execute("UPDATE usertelescopes SET is_active = 1 WHERE id = ?", (telescope_id,))
                
                conn.commit()
            
            QMessageBox.information(self, "Success", f"Telescope '{telescope_name}' is now set as active.")
            
            # Reload telescopes
            self._load_telescopes()
            
        except Exception as e:
            logger.error(f"Error setting active telescope: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to set active telescope: {str(e)}")


# --- About Dialog ---
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)

        # Set a fixed size that accommodates the new version content
        self.setFixedSize(420, 380)

        # Create main layout with proper margins and spacing
        About_layout = QVBoxLayout(self)
        About_layout.setContentsMargins(30, 30, 30, 30)
        About_layout.setSpacing(20)

        # Create icon label
        icon_label = QLabel()
        icon_path = ResourceManager.get_icon_path()
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            # Scale the icon to a reasonable size (64x64)
            scaled_pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        About_layout.addWidget(icon_label)

        # Title
        title_label = QLabel("Cosmos Collection")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #ffffff;")
        About_layout.addWidget(title_label)

        # Version information
        try:
            from version import version_manager
            version_text = version_manager.get_detailed_version_info()
        except ImportError:
            version_text = "Version information not available"

        version_label = QLabel(version_text)
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setWordWrap(True)  # Allow text wrapping if needed
        version_label.setStyleSheet("font-size: 10pt; color: #bbbbbb; margin: 5px 0px; font-family: monospace;")
        About_layout.addWidget(version_label)

        # Description
        desc_label = QLabel("A personal astrophotography catalog and session planning tools for organizing and exploring your celestial images.")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 11pt; color: #cccccc; margin: 10px 0px;")
        About_layout.addWidget(desc_label)

        # GitHub link
        link_label = QLabel('<a href="https://github.com/quake101/CosmosCollection" style="color: #0078d7;">Visit GitHub Repository</a>')
        link_label.setAlignment(Qt.AlignCenter)
        link_label.setOpenExternalLinks(True)
        link_label.setTextFormat(Qt.RichText)
        link_label.setStyleSheet("font-size: 10pt;")
        About_layout.addWidget(link_label)

        # Add fixed stretch to prevent layout jumping
        About_layout.addStretch(1)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Check for updates button
        try:
            from version import check_for_updates
            self.update_button = QPushButton("Check for Updates")
            self.update_button.setFixedSize(130, 30)
            self.update_button.clicked.connect(self._check_updates)
            button_layout.addWidget(self.update_button)
        except ImportError:
            pass

        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedSize(80, 30)
        close_button.clicked.connect(self.close)
        close_button.setDefault(True)
        button_layout.addWidget(close_button)

        button_layout.addStretch()
        About_layout.addLayout(button_layout)

    def _check_updates(self):
        """Check for updates and show result"""
        try:
            from version import version_manager
            from PySide6.QtWidgets import QMessageBox
            import webbrowser

            # Force refresh the GitHub release info
            version_manager._cached_release_info = None
            version_info = version_manager.get_version_info()

            if not version_info['github_available']:
                QMessageBox.information(self, "Update Check",
                    "Unable to check for updates. Please check your internet connection.")
                return

            if version_info['update_available']:
                msg = QMessageBox()
                msg.setWindowTitle("Update Available")
                msg.setText(f"A new version is available!")
                msg.setInformativeText(
                    f"Current version: {version_info['local_version']}\n"
                    f"Latest version: {version_info['github_version']}\n\n"
                    f"Would you like to visit the download page?"
                )
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)

                if msg.exec() == QMessageBox.Yes and version_info['github_url']:
                    webbrowser.open(version_info['github_url'])
            else:
                QMessageBox.information(self, "No Updates",
                    f"You are running the latest version ({version_info['local_version']}).")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Error checking for updates: {str(e)}")


# --- Main App Window ---
class MainWindow(QMainWindow):
    def __init__(self, dso_data, catalogs):
        super().__init__()
        logger.debug("Initializing MainWindow")

        # Set window title with version
        try:
            from version import get_version_display
            window_title = f"Cosmos Collection - {get_version_display()}"
        except ImportError:
            window_title = "Cosmos Collection"

        self.setWindowTitle(window_title)
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowFlags(Qt.Window)
        self.db_manager = DatabaseManager()
        self._cached_dso_data = None
        self._cached_catalogs = None

        # Create toolbar
        self._create_toolbar()

        # Create central widget and main layout
        central_widget = QWidget()
        central_widget.setMouseTracking(True)
        main_layout = QVBoxLayout()

        # Create search and filter controls
        controls_layout = QHBoxLayout()

        # Catalog filter
        catalog_layout = QHBoxLayout()
        catalog_label = QLabel("Catalog:")
        self.catalog_combo = QComboBox()
        self.catalog_combo.addItem("All Catalogs")
        self.catalog_combo.addItems(catalogs)
        self.catalog_combo.currentTextChanged.connect(self._on_catalog_changed)
        catalog_layout.addWidget(catalog_label)
        catalog_layout.addWidget(self.catalog_combo)
        controls_layout.addLayout(catalog_layout)

        # DSO Type filter
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        self.type_combo = QComboBox()
        self.type_combo.addItem("All Types")
        # Add common DSO types with readable names (ordered by frequency/popularity)
        dso_types = [
            ("GALXY", "Galaxy"),
            ("DRKNB", "Dark Nebula"),
            ("OPNCL", "Open Cluster"),
            ("PLNNB", "Planetary Nebula"),
            ("BRTNB", "Bright Nebula"),
            ("SNREM", "Supernova Remnant"),
            ("GALCL", "Galaxy Cluster"),
            ("GLOCL", "Globular Cluster"),
            ("ASTER", "Asterism"),
            ("2STAR", "Double Star"),
            ("CL+NB", "Cluster + Nebula"),
            ("GX+DN", "Galaxy + Dark Nebula"),
            ("3STAR", "Triple Star"),
            ("4STAR", "Quadruple Star"),
            ("1STAR", "Single Star"),
            ("LMCOC", "LMC Open Cluster"),
            ("LMCCN", "LMC Cluster/Nebula"),
            ("LMCGC", "LMC Globular Cluster"),
            ("LMCDN", "LMC Dark Nebula"),
            ("SMCGC", "SMC Globular Cluster"),
            ("SMCCN", "SMC Cluster/Nebula"),
            ("SMCOC", "SMC Open Cluster"),
            ("SMCDN", "SMC Dark Nebula"),
            ("QUASR", "Quasar"),
            ("NONEX", "Non-existent")
        ]
        for code, name in dso_types:
            self.type_combo.addItem(name, code)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        controls_layout.addLayout(type_layout)

        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter designation, RA, or Dec")
        self.search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        controls_layout.addLayout(search_layout)

        # Add show only images checkbox
        self.show_images_only = QCheckBox("Show Only Objects with Images")
        self.show_images_only.stateChanged.connect(self._on_show_images_changed)
        controls_layout.addWidget(self.show_images_only)

        # Add highlight no images checkbox
        self.highlight_no_images = QCheckBox("Highlight Objects without Images")
        self.highlight_no_images.stateChanged.connect(self._on_highlight_no_images_changed)
        controls_layout.addWidget(self.highlight_no_images)

        # Add clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_filters)
        controls_layout.addWidget(clear_button)

        main_layout.addLayout(controls_layout)

        # Add status label
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

        # Setup model and table
        self.model = DSOTableModel(dso_data)
        self.table_view = CustomTableView()
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.doubleClicked.connect(self._on_double_click)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSortingEnabled(True)

        # Enable context menu
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self._show_context_menu)

        # Set up the table view's style
        self.table_view.setStyleSheet("""
            QTableView {
                /* background-color: #2d2d2d; */
                /* alternate-background-color: #3d3d3d; */
                gridline-color: #4d4d4d;
                color: #ffffff;
            }
            QTableView::item:selected {
                background-color: #0078d7;
                color: white;
            }
            QHeaderView::section {
                background-color: #1d1d1d;
                padding: 4px;
                border: 1px solid #4d4d4d;
                color: #ffffff;
            }
        """)

        main_layout.addWidget(self.table_view)

        # Set the layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Update status
        self._update_status()
        logger.debug("MainWindow initialization complete")

    def _create_toolbar(self):
        """Create the main toolbar with Settings, Telescopes, DSO tools, and About actions"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.setToolTip("Open application settings")
        settings_action.triggered.connect(self._show_settings)
        toolbar.addAction(settings_action)
        
        toolbar.addSeparator()
        
        # Telescopes action  
        telescopes_action = QAction("Telescopes", self)
        telescopes_action.setToolTip("Manage telescope configurations")
        telescopes_action.triggered.connect(self._show_telescopes)
        toolbar.addAction(telescopes_action)
        
        toolbar.addSeparator()
        
        # DSO Visibility Calculator action
        visibility_action = QAction("DSO Visibility", self)
        visibility_action.setToolTip("Calculate DSO visibility from your location")
        visibility_action.triggered.connect(self._show_dso_visibility)
        toolbar.addAction(visibility_action)
        
        # Best DSO Tonight action
        best_dso_action = QAction("Best DSO Tonight", self)
        best_dso_action.setToolTip("Find the best DSOs visible tonight")
        best_dso_action.triggered.connect(self._show_best_dso_tonight)
        toolbar.addAction(best_dso_action)
        
        # Target List action
        target_list_action = QAction("Target List", self)
        target_list_action.setToolTip("Manage your DSO target list")
        target_list_action.triggered.connect(self._show_target_list)
        toolbar.addAction(target_list_action)
        
        # Collage Builder action
        collage_builder_action = QAction("Collage Builder", self)
        collage_builder_action.setToolTip("Create image collages from your DSO photos")
        collage_builder_action.triggered.connect(self._show_collage_builder)
        toolbar.addAction(collage_builder_action)

        # Aladin Lite action
        aladin_lite_action = QAction("Aladin Lite", self)
        aladin_lite_action.setToolTip("Open Aladin Lite sky viewer")
        aladin_lite_action.triggered.connect(self._show_aladin_lite_from_toolbar)
        toolbar.addAction(aladin_lite_action)

        toolbar.addSeparator()
        
        # About action
        about_action = QAction("About", self)
        about_action.setToolTip("About Cosmos Collection")
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

    def _show_settings(self):
        """Show the settings dialog"""
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()

    def _show_telescopes(self):
        """Show the telescopes dialog"""
        telescope_dialog = TelescopeDialog(self)
        telescope_dialog.exec()

    def _show_about(self):
        """Show the about dialog"""
        about_dialog = AboutDialog(self)
        about_dialog.exec()
        
    def _show_dso_visibility(self):
        """Show the DSO Visibility Calculator window"""
        try:
            from DSOVisibilityCalculator import DSOVisibilityApp
            if not hasattr(self, 'dso_visibility_window') or not self.dso_visibility_window.isVisible():
                self.dso_visibility_window = DSOVisibilityApp()
            self.dso_visibility_window.show()
            self.dso_visibility_window.raise_()
            self.dso_visibility_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load DSO Visibility Calculator: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open DSO Visibility Calculator: {e}")
            
    def _show_best_dso_tonight(self):
        """Show the Best DSO Tonight window"""
        try:
            from BestDSOTonight import BestDSOTonightWindow
            if not hasattr(self, 'best_dso_window') or not self.best_dso_window.isVisible():
                self.best_dso_window = BestDSOTonightWindow()
            self.best_dso_window.show()
            self.best_dso_window.raise_()
            self.best_dso_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load Best DSO Tonight: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Best DSO Tonight: {e}")
            
    def _show_target_list(self):
        """Show the DSO Target List window"""
        try:
            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, 'target_list_window') or not self.target_list_window.isVisible():
                self.target_list_window = DSOTargetListWindow()
            self.target_list_window.show()
            self.target_list_window.raise_()
            self.target_list_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load Target List: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Target List: {e}")

    def _show_collage_builder(self):
        """Show the Collage Builder window"""
        try:
            # Get all user images from the database
            user_images = []
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ui.id, ui.image_path, ui.integration_time, ui.equipment, 
                           ui.date_taken, ui.notes, c.designation as dso_name
                    FROM userimages ui
                    LEFT JOIN cataloguenr c ON ui.dsodetailid = c.dsodetailid
                    ORDER BY ui.id DESC
                """)
                rows = cursor.fetchall()
                for row in rows:
                    user_images.append({
                        'id': row[0],
                        'image_path': row[1],
                        'integration_time': row[2] or '',
                        'equipment': row[3] or '',
                        'date_taken': row[4] or '',
                        'notes': row[5] or '',
                        'dso_name': row[6] or 'Unknown DSO'
                    })
            
            if not user_images:
                QMessageBox.information(self, "No Images", 
                    "No user images found. Add some images to DSO objects first, then you can create collages with them.")
                return
            
            # Create collage builder window with all user images
            if not hasattr(self, 'collage_builder_window') or not self.collage_builder_window.isVisible():
                # Use a dummy dsodetailid since we're showing all images
                self.collage_builder_window = CollageBuilderWindow(user_images, "All DSO Images", None, self)
            self.collage_builder_window.show()
            self.collage_builder_window.raise_()
            self.collage_builder_window.activateWindow()
            
        except Exception as e:
            logger.error(f"Error opening collage builder: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Could not open Collage Builder: {str(e)}")

    def _show_aladin_lite_from_toolbar(self):
        """Open Aladin Lite from toolbar with general sky view"""
        try:
            # Create a default data dictionary for M33 (Triangulum Galaxy)
            default_data = {
                'name': 'M33',
                'ra': 1.564,  # 1h 33m 50s
                'dec': 30.66,  # +30° 39' 37"
                'ra_deg': 23.46,  # 1.564 hours * 15 degrees/hour
                'dec_deg': 30.66,   # +30.66 degrees
                'dsodetailid': None,
                'size_min': 70.8,  # M33 is about 70.8 x 41.7 arcminutes
                'size_max': 41.7   # Using actual M33 dimensions
            }

            # Store reference to prevent garbage collection and manage window lifecycle
            if not hasattr(self, 'aladin_window') or not self.aladin_window.isVisible():
                self.aladin_window = AladinLiteWindow(default_data, self)
                self.aladin_window.show()
                logger.debug("Opened Aladin Lite window from toolbar")
            else:
                # If window is already open, bring it to front
                self.aladin_window.raise_()
                self.aladin_window.activateWindow()
                logger.debug("Aladin Lite window already open, bringing to front")
        except Exception as e:
            logger.error(f"Error opening Aladin Lite from toolbar: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

    def _on_show_images_changed(self, state):
        """Handle show images only checkbox state change"""
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()

    def _on_highlight_no_images_changed(self, state):
        self.model.setHighlightNoImages(state != 0)

    def _clear_filters(self):
        """Clear all filters"""
        self.search_input.clear()
        self.catalog_combo.setCurrentIndex(0)
        self.show_images_only.setChecked(False)
        self.highlight_no_images.setChecked(False)
        self.type_combo.setCurrentIndex(0)
        self.model.filter_data("", None, False, None)
        self._update_status()

    def _on_search(self, text):
        """Handle search text changes"""
        self.model.filter_data(
            text,
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()

    def _on_catalog_changed(self, catalog):
        """Handle catalog selection changes"""
        self.model.filter_data(
            self.search_input.text(),
            None if catalog == "All Catalogs" else catalog,
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()

    def _on_type_changed(self, type_text):
        """Handle DSO type selection changes"""
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()

    def _get_selected_type(self):
        """Get the currently selected DSO type code"""
        current_data = self.type_combo.currentData()
        if current_data and self.type_combo.currentText() != "All Types":
            return current_data
        return None

    def _update_status(self):
        """Update the status label"""
        total = len(self.model.dso_data)
        filtered = len(self.model.filtered_data)
        if filtered == total:
            self.status_label.setText(f"Showing all {total} objects")
        else:
            self.status_label.setText(f"Showing {filtered} of {total} objects")

    def _on_double_click(self, index):
        try:
            logger.debug(f"Double click on row {index.row()}")
            row = index.row()
            entry = self.model.filtered_data[row]
            logger.debug(f"Selected entry: {entry}")

            # Get fresh data from database using the connection manager
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Query the object with its user image
                cursor.execute("""
                    WITH object_dsodetailid AS (
                        SELECT d.id 
                        FROM dsodetail d
                        JOIN cataloguenr c ON d.id = c.dsodetailid
                        WHERE c.catalogue = ? AND c.designation = ?
                    )
                    SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness, 
                           CAST(d.sizemin/60.0 AS REAL) as sizemin,
                           CAST(d.sizemax/60.0 AS REAL) as sizemax,
                           d.constellation, d.dsotype, d.dsoclass,
                           (
                               SELECT GROUP_CONCAT(c2.catalogue || ' ' || c2.designation, ', ')
                               FROM cataloguenr c2
                               WHERE c2.dsodetailid = d.id
                           ) as designations,
                           ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                           (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                    WHERE d.id = (SELECT id FROM object_dsodetailid)
                    GROUP BY d.id
                """, (entry["catalogue"], entry["id"]))

                result = cursor.fetchone()
                logger.debug(f"Database result: {result}")

                if not result:
                    logger.error(f"Could not find object {entry['name']}")
                    return

                # Process the result and create the detail window
                self._create_detail_window(result, entry)

        except Exception as e:
            logger.error(f"Error in _on_double_click: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def _create_detail_window(self, result, entry):
        """Create and show the detail window with the given data"""
        obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
            constellation, dso_type, dso_class, designations, image_path, integration_time, \
            equipment, date_taken, notes, image_count = result

        # Get the primary designation
        primary_designation = designations.split(',')[0]
        catalogue, designation = primary_designation.split(' ', 1)

        # Handle size values
        size_min_arcmin = float(size_min) if size_min is not None else 0.0
        size_max_arcmin = float(size_max) if size_max is not None else 0.0

        # Convert coordinates for display
        ra_str = self._format_ra(ra)
        dec_str = self._format_dec(dec)

        data = {
            "name": entry["name"],  # Use the name from the table entry instead of reconstructing it
            "ra": ra_str,
            "dec": dec_str,
            "ra_deg": ra,
            "dec_deg": dec,
            "magnitude": magnitude,
            "surface_brightness": surface_brightness,
            "size_min": size_min_arcmin,
            "size_max": size_max_arcmin,
            "constellation": constellation,
            "dso_type": dso_type,
            "dso_class": dso_class,
            "designations": designations,
            "catalogue": catalogue,
            "id": designation,
            "dsodetailid": obj_id,
            "image_path": image_path,
            "integration_time": integration_time,
            "equipment": equipment,
            "date_taken": date_taken,
            "notes": notes,
            "image_count": image_count
        }

        logger.debug(f"Data dictionary: {data}")

        logger.debug("Creating detail window")
        # Create the window in a new thread
        detail_window = ObjectDetailWindow(data)
        detail_window.image_added.connect(self._refresh_data)
        detail_window.show()
        logger.debug("Detail window shown")

    def _refresh_data(self):
        """Refresh the data in the main window"""
        try:
            # Get fresh data from database using the connection manager
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get all available catalogs
                cursor.execute("SELECT DISTINCT catalogue FROM cataloguenr ORDER BY catalogue")
                self._cached_catalogs = [row[0] for row in cursor.fetchall()]

                # Query all objects from the database with additional fields
                cursor.execute("""
                    SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness, 
                           CAST(d.sizemin/60.0 AS REAL) as sizemin,
                           CAST(d.sizemax/60.0 AS REAL) as sizemax,
                           d.constellation, d.dsotype, d.dsoclass,
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY 
                               CASE c.catalogue 
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation) as designations,
                           NULL as image_path, NULL as integration_time, NULL as equipment, 
                           NULL as date_taken, NULL as notes,
                           (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    GROUP BY d.id
                    ORDER BY c.catalogue, CAST(c.designation AS INTEGER)
                """)

                self._cached_dso_data = []
                for row in cursor.fetchall():
                    # Process each row and add to cached data
                    self._cached_dso_data.append(self._process_dso_row(row))

            # Update the model with new data
            self.model.dso_data = self._cached_dso_data

            # Reapply current filters
            self.model.filter_data(
                self.search_input.text(),
                None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
                self.show_images_only.isChecked(),
                self._get_selected_type()
            )

            # Update status
            self._update_status()

            logger.debug("Main window data refreshed")
        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}", exc_info=True)

    def _process_dso_row(self, row):
        """Process a single row from the DSO query into a dictionary"""
        obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
            constellation, dso_type, dso_class, designations, image_path, integration_time, \
            equipment, date_taken, notes, image_count = row

        # Get the primary designation
        primary_designation = designations.split(',')[0]
        catalogue, designation = primary_designation.split(' ', 1)

        # Handle size values
        size_min_arcmin = float(size_min) if size_min is not None else 0.0
        size_max_arcmin = float(size_max) if size_max is not None else 0.0

        return {
            "id": designation,
            "ra_deg": ra,
            "dec_deg": dec,
            "catalogue": catalogue,
            "name": f"{catalogue} {designation}",
            "magnitude": magnitude,
            "surface_brightness": surface_brightness,
            "size_min": size_min_arcmin,
            "size_max": size_max_arcmin,
            "constellation": constellation,
            "dso_type": dso_type,
            "dso_class": dso_class,
            "designations": designations,
            "image_path": image_path,
            "integration_time": integration_time,
            "equipment": equipment,
            "date_taken": date_taken,
            "notes": notes,
            "image_count": image_count
        }

    def _show_context_menu(self, position):
        """Show context menu when right-clicking on the DSO table"""
        # Get the item at the clicked position
        index = self.table_view.indexAt(position)
        if not index.isValid():
            return  # No item at this position

        # Get the row number
        row = index.row()
        if row < 0 or row >= len(self.model.filtered_data):
            return

        # Create context menu
        context_menu = QMenu(self)

        # Apply dark theme styling to the context menu
        context_menu.setStyleSheet("""
            QMenu {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 2px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 8px 16px;
                border: none;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QMenu::item:hover {
                background-color: #0078d4;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #666666;
                margin: 2px 8px;
            }
        """)

        # Add menu actions
        details_action = context_menu.addAction("View DSO Details")
        details_action.triggered.connect(lambda: self._context_view_details(row))

        visibility_action = context_menu.addAction("Open DSO Visibility Calculator")
        visibility_action.triggered.connect(lambda: self._context_open_visibility(row))

        aladin_action = context_menu.addAction("Open in Aladin Lite")
        aladin_action.triggered.connect(lambda: self._context_open_aladin(row))

        context_menu.addSeparator()

        target_action = context_menu.addAction("Add to Target List")
        target_action.triggered.connect(lambda: self._context_add_to_target_list(row))

        # Show the menu at the clicked position
        context_menu.exec(self.table_view.mapToGlobal(position))

    def _context_view_details(self, row):
        """View DSO details from context menu"""
        # Get the index and trigger the existing double-click method
        model_index = self.model.index(row, 0)
        self._on_double_click(model_index)

    def _context_open_visibility(self, row):
        """Open DSO Visibility Calculator from context menu"""
        try:
            entry = self.model.filtered_data[row]
            dso_name = entry.get("name", "")
            ra_deg = entry.get("ra_deg", 0)
            dec_deg = entry.get("dec_deg", 0)

            if not dso_name:
                QMessageBox.warning(self, "Error", "No DSO name available")
                return

            logger.debug(f"Opening DSO Visibility Calculator for: {dso_name} at RA {ra_deg}° Dec {dec_deg}°")

            # Import and open DSO Visibility Calculator
            from DSOVisibilityCalculator import DSOVisibilityApp

            # Store reference to prevent garbage collection
            self.visibility_window = DSOVisibilityApp()

            # Use coordinates instead of name for more reliable calculation
            # Format coordinates as a string that astropy can parse
            coord_string = f"{ra_deg:.6f} {dec_deg:+.6f}"

            if hasattr(self.visibility_window, 'dso_input'):
                self.visibility_window.dso_input.setText(coord_string)
                logger.debug(f"Set coordinates in input field: {coord_string}")
            else:
                logger.warning("DSO input field not found in visibility window")

            # Show the window immediately
            self.visibility_window.show()
            self.visibility_window.raise_()
            self.visibility_window.activateWindow()

            # Automatically trigger calculation after a short delay
            if hasattr(self.visibility_window, 'calculate_visibility'):
                QTimer.singleShot(500, self.visibility_window.calculate_visibility)
                logger.debug("Triggered automatic visibility calculation")
            else:
                logger.warning("Calculate visibility method not found in visibility window")

            logger.debug("DSO Visibility Calculator window opened successfully")

        except Exception as e:
            logger.error(f"Error opening DSO Visibility Calculator: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open DSO Visibility Calculator: {str(e)}")

    def _context_open_aladin(self, row):
        """Open Aladin Lite from context menu"""
        try:
            entry = self.model.filtered_data[row]

            # Create data dictionary similar to what ObjectDetailWindow creates
            detail_data = {
                'name': entry.get('name', ''),
                'ra_deg': entry.get('ra_deg', 0),
                'dec_deg': entry.get('dec_deg', 0),
                'size_min': entry.get('size_min', 30),
                'size_max': entry.get('size_max', 30),
                'dsodetailid': entry.get('id', '')
            }

            # Import and open Aladin Lite window
            # Store reference to prevent garbage collection and manage window lifecycle
            if not hasattr(self, 'aladin_window') or not self.aladin_window.isVisible():
                self.aladin_window = AladinLiteWindow(detail_data, self)
                self.aladin_window.show()
            else:
                # If window is already open, bring it to front
                self.aladin_window.raise_()
                self.aladin_window.activateWindow()

        except Exception as e:
            logger.error(f"Error opening Aladin Lite: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

    def _context_add_to_target_list(self, row):
        """Add DSO to target list from context menu"""
        try:
            entry = self.model.filtered_data[row]

            # Create data dictionary for target list
            dso_data = {
                'name': entry.get('name', ''),
                'ra_deg': entry.get('ra_deg', 0),
                'dec_deg': entry.get('dec_deg', 0),
                'magnitude': entry.get('magnitude', 0),
                'size_min': entry.get('size_min', 0),
                'size_max': entry.get('size_max', 0),
                'constellation': entry.get('constellation', ''),
                'dso_type': entry.get('dso_type', ''),
                'dso_class': entry.get('dso_class', '')
            }

            # Import and open Target List window, then add the DSO
            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, 'target_list_window') or not self.target_list_window.isVisible():
                self.target_list_window = DSOTargetListWindow()

            self.target_list_window.show()
            self.target_list_window.raise_()
            self.target_list_window.activateWindow()

            # Add the DSO to the target list
            self.target_list_window.add_target_from_dso(dso_data)

        except Exception as e:
            logger.error(f"Error adding to target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {str(e)}")

    def _format_ra(self, ra_deg):
        """Convert RA in degrees to hms format"""
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_remaining = (ra_hours - ra_h) * 60
        ra_m = int(ra_remaining)
        ra_s = (ra_remaining - ra_m) * 60
        return f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s"

    def _format_dec(self, dec_deg):
        """Convert Dec in degrees to dms format"""
        dec_sign = '-' if dec_deg < 0 else '+'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_remaining = (dec_abs - dec_d) * 60
        dec_m = int(dec_remaining)
        dec_s = (dec_remaining - dec_m) * 60
        return f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""

    def closeEvent(self, event):
        """Handle window close event"""
        self.db_manager.close()
        super().closeEvent(event)


# --- Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set application icon
    icon_path = os.path.join(APP_DIR, 'images', 'CosmosCollection.png')
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)

    # Initialize database manager and get data
    db_manager = DatabaseManager()
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check if the catalogs directory exists, create if needed
            catalogs_dir = os.path.join(APP_DIR, 'catalogs')
            if not os.path.exists(catalogs_dir):
                os.makedirs(catalogs_dir)

            # Check if the userimages table exists and create if needed
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='userimages'")
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Check if the new columns exist
                cursor.execute("PRAGMA table_info(userimages)")
                columns = [column[1] for column in cursor.fetchall()]
                new_columns = ['integration_time', 'equipment', 'date_taken', 'notes']

                # Add any missing columns
                for column in new_columns:
                    if column not in columns:
                        cursor.execute(f"ALTER TABLE userimages ADD COLUMN {column} TEXT")
                conn.commit()
            else:
                # Create userimages table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS userimages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dsodetailid INTEGER,
                        image_path TEXT,
                        integration_time TEXT,
                        equipment TEXT,
                        date_taken TEXT,
                        notes TEXT,
                        FOREIGN KEY (dsodetailid) REFERENCES dsodetail(id)
                    )
                """)
                conn.commit()

            # Get all available catalogs
            cursor.execute("SELECT DISTINCT catalogue FROM cataloguenr ORDER BY catalogue")
            catalogs = [row[0] for row in cursor.fetchall()]

            # Query all objects from the database with additional fields
            cursor.execute("""
                SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness, 
                       CAST(d.sizemin/60.0 AS REAL) as sizemin,
                       CAST(d.sizemax/60.0 AS REAL) as sizemax,
                       d.constellation, d.dsotype, d.dsoclass,
                       GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ') as designations,
                       ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                       (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                GROUP BY d.id
                ORDER BY c.catalogue, CAST(c.designation AS INTEGER)
            """)

            dso_data = []
            for row in cursor.fetchall():
                # Process each row and add to data
                obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
                    constellation, dso_type, dso_class, designations, image_path, integration_time, \
                    equipment, date_taken, notes, image_count = row

                # Get the primary designation
                primary_designation = designations.split(',')[0]
                catalogue, designation = primary_designation.split(' ', 1)

                # Handle size values
                size_min_arcmin = float(size_min) if size_min is not None else 0.0
                size_max_arcmin = float(size_max) if size_max is not None else 0.0

                dso_data.append({
                    "id": designation,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "catalogue": catalogue,
                    "name": f"{catalogue} {designation}",
                    "magnitude": magnitude,
                    "surface_brightness": surface_brightness,
                    "size_min": size_min_arcmin,
                    "size_max": size_max_arcmin,
                    "constellation": constellation,
                    "dso_type": dso_type,
                    "dso_class": dso_class,
                    "designations": designations,
                    "image_path": image_path,
                    "integration_time": integration_time,
                    "equipment": equipment,
                    "date_taken": date_taken,
                    "notes": notes,
                    "image_count": image_count
                })

        if not dso_data:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(None, "Error", "Failed to load DSO data from database")
            sys.exit(1)

        window = MainWindow(dso_data, catalogs)
        window.show()
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
        sys.exit(1)
    finally:
        db_manager.close()