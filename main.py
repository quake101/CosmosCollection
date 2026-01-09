import logging
import os
import sys
from typing import Optional, Dict

# Configure SSL certificates BEFORE any network imports (CRITICAL for PyInstaller)
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle - configure SSL first
    try:
        import certifi
        cert_path = os.path.join(sys._MEIPASS, 'certifi', 'cacert.pem')
        if os.path.exists(cert_path):
            os.environ['SSL_CERT_FILE'] = cert_path
            os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            os.environ['CURL_CA_BUNDLE'] = cert_path
        else:
            # Fallback to certifi's default path
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['CURL_CA_BUNDLE'] = certifi.where()
    except Exception as e:
        print(f"Warning: Could not configure SSL certificates: {e}")

# Core PySide6 imports (always needed)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QUrl, Signal, QObject, QTimer, QEvent, QThread, QSettings
from PySide6.QtGui import QPixmap, QPainter, QIcon, QColor, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView,
    QVBoxLayout, QWidget, QLabel, QDialog,
    QHeaderView, QPushButton, QHBoxLayout, QLineEdit, QComboBox, QTextEdit, QCheckBox, QGroupBox,
    QToolBar, QMessageBox, QMenu, QScrollArea, QGridLayout
)

# Local imports (always needed)
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionManager, WindowPositionMixin
from ResourceManager import ResourceManager
from CollageBuilder import CollageBuilder, CollageBuilderWindow

# Import astroquery at module level so PyInstaller detects it
try:
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROQUERY_AVAILABLE = True
except ImportError as e:
    ASTROQUERY_AVAILABLE = False
    print(f"Warning: astroquery not available: {e}")

# Heavy imports - lazy loaded when needed:
# - QWebEngineView (only loaded when Aladin window is created)
# - astroplan/astropy (only loaded when visibility calculations are needed)
# - DSOVisibilityApp (only loaded when visibility calculator is used)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the application directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Log SSL configuration status and configure astroquery
if getattr(sys, 'frozen', False):
    logger.info(f"Running in PyInstaller bundle. SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE')}")

    # Configure astroquery cache directory for PyInstaller
    try:
        from astropy.config.paths import set_temp_cache
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), 'astroquery_cache')
        os.makedirs(cache_dir, exist_ok=True)
        # Set astroquery to use this cache directory
        os.environ['XDG_CACHE_HOME'] = cache_dir
        logger.info(f"Astroquery cache directory set to: {cache_dir}")
    except Exception as e:
        logger.warning(f"Could not configure astroquery cache: {e}")
else:
    logger.debug("Running in normal Python environment")

# Check for optional DSO Visibility Calculator availability
try:
    import DSOVisibilityCalculator
    VISIBILITY_AVAILABLE = True
except ImportError:
    VISIBILITY_AVAILABLE = False
    logging.warning("DSOVisibilityCalculator.py not found. Visibility calculator will be disabled.")


class SimbadQueryThread(QThread):
    """Background thread for querying SIMBAD without blocking the UI"""
    query_complete = Signal(object, str, float, float)  # Signal(result, object_name, ra_deg, dec_deg)
    query_failed = Signal(str, str)  # Signal(object_name, error_message)

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
            import sys

            # Configure SSL for PyInstaller bundle
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

            # Configure SIMBAD to return flux measurements and spectral type
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('U', 'B', 'V', 'R', 'I', 'sp', 'otype')

            # Query SIMBAD using coordinates (more reliable than name)
            result = None
            if self.ra_deg is not None and self.dec_deg is not None:
                # Create coordinate object
                coords = SkyCoord(ra=self.ra_deg*u.degree, dec=self.dec_deg*u.degree, frame='icrs')
                # Query region around coordinates (3 arcmin radius)
                result = custom_simbad.query_region(coords, radius=3*u.arcmin)
                if result is not None and len(result) > 0:
                    logger.debug(f"SIMBAD data found for {self.object_name} at coordinates")
                else:
                    logger.debug(f"No SIMBAD data found for {self.object_name} at coordinates")
            else:
                logger.debug(f"No coordinates available for {self.object_name}, skipping SIMBAD query")

            # Emit success signal with result
            self.query_complete.emit(result, self.object_name, self.ra_deg, self.dec_deg)

        except Exception as e:
            logger.error(f"SIMBAD query failed for {self.object_name}: {type(e).__name__}: {str(e)}", exc_info=True)
            self.query_failed.emit(self.object_name, str(e))


class ImageLoaderThread(QThread):
    """Background thread for loading large images without blocking the UI"""
    image_loaded = Signal(object, str)  # Signal(QImage, image_path)
    load_failed = Signal(str, str)  # Signal(image_path, error_message)

    def __init__(self, image_path, is_fits=False, fits_colormap='gray'):
        super().__init__()
        self.image_path = image_path
        self.is_fits = is_fits
        self.fits_colormap = fits_colormap

    def run(self):
        """Load image in background thread with progressive loading"""
        try:
            import os
            from PySide6.QtGui import QImage, QImageReader
            from PySide6.QtCore import QSize

            if not os.path.exists(self.image_path):
                self.load_failed.emit(self.image_path, "File not found")
                return

            # Check if PNG and try Pillow for faster loading
            file_ext = os.path.splitext(self.image_path)[1].lower()
            file_size_mb = os.path.getsize(self.image_path) / (1024 * 1024)

            logger.info(f"Image loading: {os.path.basename(self.image_path)} - {file_size_mb:.1f}MB - Format: {file_ext}")

            # Use Pillow for PNG files > 2MB (much faster than Qt)
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
                # Load FITS file - returns QImage (after QPixmap conversion for data safety)
                qimage = self._load_fits_image(self.image_path, self.fits_colormap)
                if qimage and not qimage.isNull():
                    self.image_loaded.emit(qimage, self.image_path)
                else:
                    self.load_failed.emit(self.image_path, "Failed to load FITS file")
            else:
                # Use QImageReader for optimized loading
                # Increase allocation limit for large images
                QImageReader.setAllocationLimit(1024)  # 1GB limit

                reader = QImageReader(self.image_path)
                reader.setAutoTransform(True)  # Handle EXIF rotation
                reader.setQuality(100)  # Maximum quality

                if not reader.canRead():
                    self.load_failed.emit(self.image_path, f"Cannot read image: {reader.errorString()}")
                    return

                # Get original image size
                original_size = reader.size()
                logger.debug(f"Loading full resolution with Qt: {original_size.width()}x{original_size.height()}")

                # Load the image at full resolution
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
        """Load image using Pillow (much faster for PNG) - returns True if successful"""
        try:
            from PIL import Image
            from PySide6.QtGui import QImage
            from PySide6.QtCore import QSize
            import io

            logger.debug(f"Loading PNG with Pillow: {self.image_path}")

            # Open image with Pillow
            with Image.open(self.image_path) as pil_image:
                original_width, original_height = pil_image.size
                logger.info(f"Pillow opened image: {original_width}x{original_height}, mode: {pil_image.mode}")

                # Load full resolution image (no downsampling for zoom capability)
                full_image = pil_image
                logger.info(f"Converting full resolution to QImage: {original_width}x{original_height}")

                # Convert all images to RGB for consistent color handling (no alpha channel)
                if full_image.mode != 'RGB':
                    logger.info(f"Converting {full_image.mode} to RGB for consistent color handling")
                    full_image = full_image.convert('RGB')

                # Create QImage in RGB888 format (same as FITS images)
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
            import io

            with fits.open(fits_path) as hdul:
                data = hdul[0].data

                if data is None:
                    return None

                # Handle 3D data (take first slice if needed)
                if len(data.shape) == 3:
                    if data.shape[0] == 3:
                        data = np.transpose(data, (1, 2, 0))
                    elif data.shape[0] <= 10:
                        data = data[0]

                # Check if RGB
                is_rgb = len(data.shape) == 3 and data.shape[2] == 3

                if is_rgb:
                    # RGB FITS - no colormap
                    norm = simple_norm(data, 'linear', percent=99.5)
                    normalized_data = norm(data)
                    rgb_data = (normalized_data * 255).astype(np.uint8)
                else:
                    # Grayscale - apply colormap
                    try:
                        import matplotlib.pyplot as plt
                        norm = simple_norm(data, 'linear', percent=99.5)
                        normalized_data = norm(data)
                        cmap = plt.get_cmap(colormap)
                        rgba_data = cmap(normalized_data)
                        rgb_data = (rgba_data[:, :, :3] * 255).astype(np.uint8)
                    except ImportError:
                        # Fallback without matplotlib
                        norm = simple_norm(data, 'linear', percent=99.5)
                        normalized_data = norm(data)
                        gray_data = (normalized_data * 255).astype(np.uint8)
                        rgb_data = np.stack([gray_data] * 3, axis=-1)

                # Flip vertically for correct orientation
                rgb_data = np.flipud(rgb_data)

                # Ensure data is C-contiguous for QImage
                rgb_data = np.ascontiguousarray(rgb_data)

                height, width = rgb_data.shape[:2]
                bytes_per_line = 3 * width
                qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Convert to QPixmap immediately while numpy data is in scope
                # QPixmap.fromImage() does a deep copy, ensuring data persists
                pixmap = QPixmap.fromImage(qimage)

                if not pixmap.isNull():
                    # Convert back to QImage for thread - this ensures data is properly owned
                    # and avoids color corruption from multiple conversions
                    return pixmap.toImage()
                else:
                    return None

        except Exception as e:
            logger.error(f"Error loading FITS file {fits_path}: {str(e)}")
            return None


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
            
            # Import required libraries for visibility calculations
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from datetime import datetime, timedelta
            import numpy as np

            # Create DSO coordinate
            dso_coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)

            # Use a more thorough seasonal visibility check that matches Best DSO Tonight logic
            # Check multiple nights throughout the year using the same method as Best DSO Tonight
            seasons = []
            current_year = datetime.now().year
            min_altitude = 30  # Use 30° minimum altitude for seasonal visibility

            # Sample dates throughout the year (every 15 days for better coverage)
            sample_dates = []
            visibility_results = []
            
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


# --- Lazy Loading Worker Thread ---
class DataLoadWorker(QThread):
    """Worker thread for loading additional DSO data in background"""
    data_loaded = Signal(list)  # Signal with new data batch
    progress_updated = Signal(int, int)  # loaded count, total count

    def __init__(self, offset, limit, catalog_filter=None, type_filter=None, parent=None):
        super().__init__(parent)
        self.offset = offset
        self.limit = limit
        self.catalog_filter = catalog_filter
        self.type_filter = type_filter

    def run(self):
        """Load data batch in background thread"""
        try:
            # Create a direct SQLite connection for this thread (avoiding singleton DatabaseManager)
            import sqlite3
            from ResourceManager import ResourceManager

            # Use the same database path logic as DatabaseManager
            # ResourceManager is a global instance, not a class
            db_path = ResourceManager.get_database_path()

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query with optional catalog and type filters
            if self.catalog_filter:
                # When catalog filter is active, filter DSOs that have that catalog
                # Special handling for Messier catalog - only numeric designations (M 1 - M 110)
                if self.catalog_filter == 'M':
                    query = """
                        SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness,
                               CAST(d.sizemin/60.0 AS REAL) as sizemin,
                               CAST(d.sizemax/60.0 AS REAL) as sizemax,
                               d.constellation, d.dsotype, d.dsoclass,
                               GROUP_CONCAT(c2.catalogue || ' ' || c2.designation, ', ' ORDER BY
                                   CASE c2.catalogue
                                       WHEN 'M' THEN 1
                                       WHEN 'NGC' THEN 2
                                       WHEN 'IC' THEN 3
                                       ELSE 4
                                   END, c2.designation) as designations,
                               ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                               (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                        FROM dsodetail d
                        JOIN cataloguenr c ON d.id = c.dsodetailid
                            AND c.catalogue = ?
                            AND c.designation NOT LIKE '%-%'
                            AND c.designation NOT LIKE '% %'
                            AND LENGTH(TRIM(c.designation)) <= 3
                            AND CAST(c.designation AS INTEGER) > 0
                            AND CAST(c.designation AS INTEGER) <= 110
                        JOIN cataloguenr c2 ON d.id = c2.dsodetailid
                        LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                    """
                else:
                    query = """
                        SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness,
                               CAST(d.sizemin/60.0 AS REAL) as sizemin,
                               CAST(d.sizemax/60.0 AS REAL) as sizemax,
                               d.constellation, d.dsotype, d.dsoclass,
                               GROUP_CONCAT(c2.catalogue || ' ' || c2.designation, ', ' ORDER BY
                                   CASE c2.catalogue
                                       WHEN 'M' THEN 1
                                       WHEN 'NGC' THEN 2
                                       WHEN 'IC' THEN 3
                                       ELSE 4
                                   END, c2.designation) as designations,
                               ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                               (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                        FROM dsodetail d
                        JOIN cataloguenr c ON d.id = c.dsodetailid AND c.catalogue = ?
                        JOIN cataloguenr c2 ON d.id = c2.dsodetailid
                        LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                    """
                params = [self.catalog_filter]

                if self.type_filter:
                    query += " WHERE d.dsotype = ?"
                    params.append(self.type_filter)

            else:
                # No catalog filter - get all objects
                query = """
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
                           ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                           (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                """
                params = []

                if self.type_filter:
                    query += " WHERE d.dsotype = ?"
                    params.append(self.type_filter)

            query += """
                    GROUP BY d.id
                    ORDER BY c.catalogue, CAST(c.designation AS INTEGER)
                    LIMIT ? OFFSET ?
            """

            params.extend([self.limit, self.offset])

            # Debug: log query and params when catalog filter is active
            if self.catalog_filter:
                logger.debug(f"SQL Query with catalog_filter='{self.catalog_filter}'")
                logger.debug(f"Params: {params}")

            cursor.execute(query, params)

            dso_data = []
            for row in cursor.fetchall():
                obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
                    constellation, dso_type, dso_class, designations, image_path, integration_time, \
                    equipment, date_taken, notes, image_count = row

                # Get the primary designation
                primary_designation = designations.split(',')[0].strip()

                # Handle cases where designation might not have a space
                if ' ' in primary_designation:
                    catalogue, designation = primary_designation.split(' ', 1)
                else:
                    # No space in designation, use entire string as catalogue
                    catalogue = primary_designation
                    designation = ""

                # Debug: log first few entries when catalog filter is active
                if self.catalog_filter and len(dso_data) < 5:
                    logger.debug(f"Loaded DSO: {primary_designation} (all: {designations})")

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

            self.data_loaded.emit(dso_data)
            logger.debug(f"Loaded {len(dso_data)} DSOs from offset {self.offset}")

            # Clean up the direct connection
            conn.close()

        except Exception as e:
            logger.error(f"Error loading data batch: {e}")
            # Clean up on error too
            try:
                conn.close()
            except:
                pass


# --- Model for displaying DSO data in table ---
class DSOTableModel(QAbstractTableModel):
    def __init__(self, dso_data, parent=None, db_manager=None, total_count=None):
        super().__init__(parent)
        self.dso_data = dso_data
        self.filtered_data = dso_data.copy()  # For filtering
        self.headers = ["Catalog", "Designation", "RA (hms)", "Dec (dms)", "Images"]
        self.selected_catalog = None
        self.highlight_no_images = False
        self._cached_formatted_data = {}  # Cache for formatted data

        # Lazy loading support
        self.db_manager = db_manager
        self.total_count = total_count or len(dso_data)
        self.load_offset = len(dso_data)
        self.loading = False
        self.load_worker = None
        self.load_batch_size = 2000
        self.startup_mode = True  # Prevent sort-triggered loading during startup

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
        logger.debug(f"Sort requested: column={column}, order={order}, loaded={len(self.dso_data)}, offset={self.load_offset}, total={self.total_count}, startup_mode={getattr(self, 'startup_mode', False)}")

        # During startup, only sort loaded data to maintain lazy loading performance
        if getattr(self, 'startup_mode', False):
            logger.debug("Startup mode: sorting only currently loaded data")
            # Continue with normal sort of loaded data
        # Check if we need to load all data for proper sorting (only after startup)
        elif self.load_offset < self.total_count:
            logger.debug(f"Sorting requested with partial data ({len(self.dso_data)}/{self.total_count}). Loading all data first...")
            self._load_all_data_for_sort(column, order)
            return

        logger.debug(f"All data loaded, proceeding with sort on {len(self.filtered_data)} items")
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
        logger.debug(f"Sorted {len(self.filtered_data)} items by column {column}")

    def _load_all_data_for_sort(self, column, order):
        """Load all remaining data before sorting"""
        if self.loading:
            logger.debug("Already loading data, sort will be applied when complete")
            # Store the sort request to apply after loading
            self._pending_sort = (column, order)
            return

        # Prevent recursive calls by checking if we already have a pending sort
        if hasattr(self, '_pending_sort') and self._pending_sort:
            logger.debug(f"Sort already pending: {self._pending_sort}, ignoring new request")
            return

        logger.debug(f"Loading all remaining data for sort by column {column}")
        self._pending_sort = (column, order)

        # Load remaining data in larger batches for faster completion
        remaining = self.total_count - self.load_offset
        if remaining > 0:
            # Temporarily increase batch size for faster loading
            old_batch_size = self.load_batch_size
            self.load_batch_size = min(remaining, 5000)  # Load up to 5000 at a time
            logger.debug(f"Starting to load {remaining} remaining items for sort")
            self.load_more_data()
            self.load_batch_size = old_batch_size
        else:
            logger.debug("No remaining data to load, applying sort immediately")
            self._apply_pending_sort()

    def _apply_pending_sort(self):
        """Apply any pending sort after data loading completes"""
        if hasattr(self, '_pending_sort') and self._pending_sort:
            column, order = self._pending_sort
            self._pending_sort = None
            logger.debug(f"Applying pending sort by column {column}")
            self.sort(column, order)

    def filter_data(self, search_text, selected_catalog=None, show_images_only=False, selected_type=None):
        """Filter the data based on search text, catalog, image presence, and DSO type"""
        self.layoutAboutToBeChanged.emit()

        # Check if catalog or type filter changed - if so, reset lazy loading
        catalog_changed = self.selected_catalog != selected_catalog
        type_changed = getattr(self, '_current_selected_type', None) != selected_type

        # Store the selected catalog for use in data() method
        self.selected_catalog = selected_catalog
        # Track current search for lazy loading
        self._current_search = search_text or ''
        self._current_show_images_only = show_images_only
        self._current_selected_type = selected_type

        if catalog_changed or type_changed:
            # Reset lazy loading state for new filter
            self._reset_lazy_loading_for_filter(selected_catalog, selected_type)
            # Trigger immediate load of data for the new filter
            self.load_more_data()
            # Data will be empty until load completes, so set filtered_data to empty
            self.filtered_data = []
            self.layoutChanged.emit()
            return

        if not search_text and not selected_catalog and not show_images_only and not selected_type:
            self.filtered_data = self.dso_data.copy()
        else:
            search_text = search_text.lower() if search_text else ""

            # Improved search logic with priority for exact catalog matches
            matches = []
            for item in self.dso_data:
                # Apply catalog and type filters
                if selected_catalog and selected_catalog != "All Catalogs":
                    if not any(designation.startswith(selected_catalog + " ")
                             for designation in item["designations"].split(", ")):
                        continue

                if selected_type and selected_type != "All Types":
                    if item.get("dso_type", "") != selected_type:
                        continue

                if show_images_only and item["image_count"] == 0:
                    continue

                # Apply search text filter
                if search_text:
                    # If we have a catalog filter, prioritize exact catalog+designation matches
                    if selected_catalog and selected_catalog != "All Catalogs":
                        # Check for exact match: catalog filter + search text = designation
                        designations = item["designations"].split(", ")
                        exact_match = any(
                            designation.lower() == f"{selected_catalog.lower()} {search_text}"
                            for designation in designations
                        )

                        # Also check if the item's ID matches the search
                        id_match = search_text in item["id"].lower()

                        if exact_match or id_match:
                            matches.append((item, 0))  # Priority 0 = exact match
                            continue

                    # Otherwise do regular substring matching
                    if (search_text in item["catalogue"].lower() or
                        search_text in item["id"].lower() or
                        self._format_ra(item["ra_deg"]).lower() in search_text or
                        self._format_dec(item["dec_deg"]).lower() in search_text or
                        search_text in item["designations"].lower()):
                        matches.append((item, 1))  # Priority 1 = substring match
                else:
                    matches.append((item, 1))

            # Sort by priority (exact matches first) and extract items
            matches.sort(key=lambda x: x[1])
            self.filtered_data = [item for item, priority in matches]

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

    def _reset_lazy_loading_for_filter(self, catalog_filter, type_filter):
        """Reset lazy loading state when filter changes and query filtered count"""
        if not self.db_manager:
            return

        try:
            import sqlite3
            from ResourceManager import ResourceManager

            # Query the total count for this specific filter
            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Build count query with filters - must match the data loading query logic
            if catalog_filter == 'M':
                # Special handling for Messier catalog - only numeric designations
                query = """
                    SELECT COUNT(DISTINCT d.id)
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                        AND c.catalogue = ?
                        AND c.designation NOT LIKE '%-%'
                        AND c.designation NOT LIKE '% %'
                        AND LENGTH(TRIM(c.designation)) <= 3
                        AND CAST(c.designation AS INTEGER) > 0
                        AND CAST(c.designation AS INTEGER) <= 110
                """
                params = [catalog_filter]

                if type_filter:
                    query += " WHERE d.dsotype = ?"
                    params.append(type_filter)

            elif catalog_filter:
                # Other catalog filters
                query = """
                    SELECT COUNT(DISTINCT d.id)
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ?
                """
                params = [catalog_filter]

                if type_filter:
                    query += " AND d.dsotype = ?"
                    params.append(type_filter)

            else:
                # No catalog filter
                query = """
                    SELECT COUNT(DISTINCT d.id)
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                """
                params = []

                if type_filter:
                    query += " WHERE d.dsotype = ?"
                    params.append(type_filter)

            cursor.execute(query, params)
            filtered_total = cursor.fetchone()[0]
            conn.close()

            # Cancel any pending load worker
            if hasattr(self, 'load_worker') and self.load_worker:
                try:
                    self.load_worker.disconnect()
                    self.load_worker.terminate()
                    self.load_worker.wait(1000)  # Wait up to 1 second
                    self.load_worker.deleteLater()
                except:
                    pass
                self.load_worker = None

            # Clear existing data and reset offset
            self.loading = False

            # Notify view that we're about to clear all data
            self.beginResetModel()
            self.dso_data = []
            self.filtered_data = []
            self.load_offset = 0
            self.total_count = filtered_total
            self._cached_formatted_data.clear()
            self.endResetModel()

            logger.debug(f"Reset lazy loading for filter: catalog={catalog_filter}, type={type_filter}, total={filtered_total}")

        except Exception as e:
            logger.error(f"Error resetting lazy loading for filter: {e}")

    def setHighlightNoImages(self, highlight):
        """Set whether to highlight objects without images"""
        self.highlight_no_images = highlight
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def check_and_load_more_data(self, view_bottom_row):
        """Check if we need to load more data and trigger loading if needed"""
        filtered_len = len(self.filtered_data)
        loaded_len = len(self.dso_data)

        # FILTER-AWARE LOADING: If we have active filters and very few results, keep loading
        has_active_filters = (hasattr(self, '_current_search') and
                            (self._current_search or self.selected_catalog or
                             getattr(self, '_current_show_images_only', False) or
                             getattr(self, '_current_selected_type', None)))

        # If filters are active and we have very few results, keep loading more aggressively
        # For sparse results (like "show images only"), we need to load much more data
        if has_active_filters and loaded_len < self.total_count:
            if filtered_len < 100:  # Very few results - load aggressively
                filter_needs_more_data = True
            elif filtered_len < 500:  # Moderate results - load when nearing end
                # Load more if we're showing most of what we found
                filter_needs_more_data = view_bottom_row > filtered_len * 0.7
            else:
                # Normal threshold for larger result sets
                filter_needs_more_data = False
        else:
            filter_needs_more_data = False

        # MAJOR FIX: If view_bottom_row seems capped (~2000), use the actual visible rows as reference
        max_visible_rows = max(view_bottom_row + 1, 2000)

        # Use much more aggressive triggering when we hit apparent view limits
        if view_bottom_row >= 1900:  # Near the apparent view limit
            trigger_point = max_visible_rows - 100  # Very aggressive
        else:
            # Normal triggering logic
            trigger_point_rows = filtered_len - 200
            trigger_point_percent = int(filtered_len * 0.8)
            trigger_point = min(trigger_point_rows, trigger_point_percent)

        # Multiple trigger conditions
        near_end_of_visible = view_bottom_row > trigger_point
        displayed_most_data = view_bottom_row > len(self.dso_data) * 0.75
        near_view_limit = view_bottom_row >= 1950  # Emergency trigger when hitting view limits

        if (self.db_manager and
            not self.loading and
            self.load_offset < self.total_count and
            (near_end_of_visible or displayed_most_data or near_view_limit or filter_needs_more_data)):

            # Log only when loading is actually triggered
            trigger_reason = []
            if near_end_of_visible: trigger_reason.append("near end of visible")
            if displayed_most_data: trigger_reason.append("75% of loaded data")
            if near_view_limit: trigger_reason.append("emergency trigger")
            if filter_needs_more_data:
                if filtered_len < 100:
                    trigger_reason.append("sparse filter results - loading more")
                else:
                    trigger_reason.append("filter needs more data")
            logger.debug(f"Triggering lazy load: {', '.join(trigger_reason)} (filtered: {filtered_len}, loaded: {loaded_len}, total: {self.total_count})")
            self.load_more_data()
        else:
            # Reduced debug logging for non-trigger cases
            pass

    def load_more_data(self):
        """Load the next batch of data in background"""
        if self.loading or self.load_offset >= self.total_count:
            logger.debug(f"Load blocked: loading={self.loading}, offset={self.load_offset}, total={self.total_count}")
            return

        # Get current filters
        catalog_filter = self.selected_catalog if self.selected_catalog else None
        type_filter = getattr(self, '_current_selected_type', None)

        logger.debug(f"Loading more data from offset {self.load_offset}, batch size {self.load_batch_size}, catalog={catalog_filter}, type={type_filter}")
        self.loading = True

        # Emit signal to update UI loading state
        if hasattr(self.parent(), '_on_loading_started'):
            self.parent()._on_loading_started()

        self.load_worker = DataLoadWorker(self.load_offset, self.load_batch_size, catalog_filter, type_filter)
        self.load_worker.data_loaded.connect(self._on_data_loaded)
        self.load_worker.start()

    def _on_data_loaded(self, new_data):
        """Handle new data batch loaded from background thread"""
        if new_data:
            # Add new data to existing data
            self.beginInsertRows(QModelIndex(), len(self.dso_data), len(self.dso_data) + len(new_data) - 1)
            self.dso_data.extend(new_data)
            self.endInsertRows()

            # Re-apply current filters to include new data
            # Catalog and type filters are already applied at SQL level, so we only need to filter by:
            # - search text
            # - show_images_only
            old_filtered_len = len(self.filtered_data)

            search_text = getattr(self, '_current_search', '').lower() if hasattr(self, '_current_search') else ""
            show_images_only = getattr(self, '_current_show_images_only', False)

            # Notify view that data is about to change
            self.layoutAboutToBeChanged.emit()

            # Rebuild filtered data from all loaded data
            if search_text or show_images_only:
                self.filtered_data = [
                    item for item in self.dso_data
                    if ((not show_images_only or item["image_count"] > 0) and
                        (not search_text or
                         search_text in item["catalogue"].lower() or
                         search_text in item["id"].lower() or
                         search_text in item["designations"].lower()))
                ]
                logger.debug(f"Applied search/image filters: filtered data is now {len(self.filtered_data)} items from {len(self.dso_data)} loaded")
            else:
                # No additional filters, so all loaded data is visible
                self.filtered_data = self.dso_data.copy()
                logger.debug(f"No additional filters: showing all {len(self.filtered_data)} loaded items")

            new_filtered_len = len(self.filtered_data)
            new_matches = new_filtered_len - old_filtered_len
            logger.debug(f"After filtering: filtered data grew from {old_filtered_len} to {new_filtered_len} (+{new_matches} new matches from {len(new_data)} loaded)")

            # Notify view that layout has changed
            self.layoutChanged.emit()

            self.load_offset += len(new_data)
            logger.debug(f"Added {len(new_data)} DSOs, total now: {len(self.dso_data)}")

        self.loading = False

        # Emit signal to update UI loading state
        if hasattr(self.parent(), '_on_loading_finished'):
            self.parent()._on_loading_finished()

        # Clean up worker
        if self.load_worker:
            self.load_worker.deleteLater()
            self.load_worker = None

        # Apply pending sort if all data is now loaded
        if (self.load_offset >= self.total_count and
            hasattr(self, '_pending_sort') and self._pending_sort):
            logger.debug("All data loaded, applying pending sort")
            # Import QTimer locally to avoid circular imports
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, self._apply_pending_sort)
            return  # Don't trigger more loading if we're done

        # Check if parent is doing a specific catalog search and needs more data
        if hasattr(self.parent(), '_check_search_needs_continue'):
            if self.parent()._check_search_needs_continue():
                logger.debug("Continuing load for specific search...")
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, self.load_more_data)
                return

        # Check if we need to load more data immediately (for sparse filter results)
        filtered_len = len(self.filtered_data)
        if (filtered_len < 100 and
            self.load_offset < self.total_count and
            hasattr(self, '_current_show_images_only') and
            getattr(self, '_current_show_images_only', False)):

            logger.debug(f"Auto-triggering next load: only {filtered_len} images found, continuing search...")
            # Use a timer to avoid recursive loading
            if hasattr(self.parent(), '_schedule_next_load'):
                self.parent()._schedule_next_load()
        # Continue loading if we have a pending sort
        elif (hasattr(self, '_pending_sort') and self._pending_sort and
              self.load_offset < self.total_count):
            logger.debug(f"Continuing to load data for pending sort ({self.load_offset}/{self.total_count})")
            from PySide6.QtCore import QTimer
            QTimer.singleShot(50, self.load_more_data)

    def get_load_progress(self):
        """Get current loading progress for status display"""
        return len(self.dso_data), self.total_count

    def exit_startup_mode(self):
        """Exit startup mode to enable full sorting functionality"""
        logger.debug("Exiting startup mode - full sorting now available")
        self.startup_mode = False


# --- SIMBAD Query Worker Thread ---
class SimbadQueryWorker(QThread):
    """Worker thread for querying SIMBAD and adding object to database"""
    object_found = Signal(dict)  # Signal with object data
    object_not_found = Signal()  # Signal when object not found
    error_occurred = Signal(str)  # Signal with error message

    def __init__(self, search_term, parent=None):
        super().__init__(parent)
        self.search_term = search_term

    def run(self):
        """Query SIMBAD in background thread"""
        try:
            # Configure SSL for PyInstaller bundle before imports
            if getattr(sys, 'frozen', False):
                try:
                    # Disable SSL verification for astroquery in PyInstaller (workaround for cert issues)
                    import requests
                    original_request = requests.Session.request
                    def patched_request(self, *args, **kwargs):
                        kwargs['verify'] = False
                        return original_request(self, *args, **kwargs)
                    requests.Session.request = patched_request
                    logger.info("Disabled SSL verification for astroquery in worker thread (PyInstaller workaround)")
                except Exception as ssl_config_error:
                    logger.warning(f"Could not configure SSL for astroquery in worker: {ssl_config_error}")

            from astroquery.simbad import Simbad
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            logger.debug(f"Querying SIMBAD for: {self.search_term}")

            # Configure SIMBAD to return comprehensive data
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields(
                'otype',  # Object type
                'dimensions',  # Dimensions
                'V',  # V magnitude
                'B',  # B magnitude
            )

            # Try to query by identifier
            result = custom_simbad.query_object(self.search_term)

            if result is None or len(result) == 0:
                logger.debug(f"No SIMBAD data found for {self.search_term}")
                self.object_not_found.emit()
                return

            # Parse the first result
            row = result[0]

            # Debug: print column names
            logger.debug(f"SIMBAD columns: {result.colnames}")

            # Extract coordinates - try different column name variations
            ra_str = None
            dec_str = None

            # Try common RA/DEC column names
            for ra_col in ['RA', 'ra', 'RA_d', 'ra_d', 'RA_ICRS', 'ra_icrs']:
                if ra_col in row.colnames:
                    ra_str = row[ra_col]
                    logger.debug(f"Found RA in column: {ra_col} = {ra_str}")
                    break

            for dec_col in ['DEC', 'dec', 'DEC_d', 'dec_d', 'DEC_ICRS', 'dec_icrs']:
                if dec_col in row.colnames:
                    dec_str = row[dec_col]
                    logger.debug(f"Found DEC in column: {dec_col} = {dec_str}")
                    break

            if ra_str is None or dec_str is None:
                logger.error(f"Could not find RA/DEC in SIMBAD result. Available columns: {result.colnames}")
                self.error_occurred.emit(f"Could not extract coordinates from SIMBAD result")
                return

            # Extract coordinates - SIMBAD returns RA/DEC in degrees
            coords = SkyCoord(
                ra=ra_str,
                dec=dec_str,
                unit=(u.deg, u.deg),
                frame='icrs'
            )

            ra_deg = coords.ra.degree
            dec_deg = coords.dec.degree

            # Extract other fields - use lowercase column names
            main_id = str(row['main_id']) if 'main_id' in row.colnames else str(row.get('MAIN_ID', 'Unknown'))
            obj_type = str(row['otype']) if 'otype' in row.colnames else (str(row['OTYPE']) if 'OTYPE' in row.colnames else None)

            # Get magnitude - try V magnitude first, then B
            magnitude = None
            for mag_col in ['V', 'v', 'B', 'b', 'FLUX_V', 'flux_V', 'FLUX_B', 'flux_B']:
                if mag_col in row.colnames and row[mag_col] is not None:
                    try:
                        magnitude = float(row[mag_col])
                        logger.debug(f"Found magnitude in column {mag_col}: {magnitude}")
                        break
                    except (ValueError, TypeError):
                        pass

            # Get dimensions (in arcminutes)
            size_maj = None
            size_min = None
            for maj_col in ['galdim_majaxis', 'GALDIM_MAJAXIS']:
                if maj_col in row.colnames and row[maj_col] is not None:
                    try:
                        size_maj = float(row[maj_col])
                        logger.debug(f"Found major axis in column {maj_col}: {size_maj}")
                        break
                    except (ValueError, TypeError):
                        pass

            for min_col in ['galdim_minaxis', 'GALDIM_MINAXIS']:
                if min_col in row.colnames and row[min_col] is not None:
                    try:
                        size_min = float(row[min_col])
                        logger.debug(f"Found minor axis in column {min_col}: {size_min}")
                        break
                    except (ValueError, TypeError):
                        pass

            # Map SIMBAD object type to DSO type
            dso_type = self._map_simbad_type(obj_type)

            logger.debug(f"SIMBAD found: {main_id} at RA={ra_deg}, Dec={dec_deg}, Type={obj_type}")

            # Create object data dictionary
            object_data = {
                'main_id': main_id,
                'ra_deg': ra_deg,
                'dec_deg': dec_deg,
                'magnitude': magnitude,
                'size_maj': size_maj,
                'size_min': size_min,
                'dso_type': dso_type,
                'obj_type': obj_type
            }

            self.object_found.emit(object_data)

        except Exception as e:
            logger.error(f"Error querying SIMBAD: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Error querying SIMBAD: {str(e)}")

    def _map_simbad_type(self, simbad_type):
        """Map SIMBAD object type to DSO type used in database"""
        if not simbad_type:
            return None

        simbad_type = simbad_type.upper()

        # Galaxy types
        if any(t in simbad_type for t in ['G', 'GAL', 'SEYFERT', 'LINER', 'AGN', 'QSO']):
            return 'GALXY'
        # Nebula types
        elif any(t in simbad_type for t in ['PN', 'PLNNB']):
            return 'PLNNB'
        elif any(t in simbad_type for t in ['SNREM', 'SNR']):
            return 'SNREM'
        elif any(t in simbad_type for t in ['HII', 'BRTNB', 'EMOBJ']):
            return 'BRTNB'
        elif any(t in simbad_type for t in ['RNE', 'REFNB']):
            return 'REFNB'
        # Cluster types
        elif any(t in simbad_type for t in ['OPNCL', 'CL*']):
            return 'OPNCL'
        elif any(t in simbad_type for t in ['GLOBC', 'GLOCL']):
            return 'GLOCL'
        elif any(t in simbad_type for t in ['ASSC', 'ASSOC']):
            return 'ASSC'
        # Combined types
        elif 'NEB' in simbad_type and 'CL' in simbad_type:
            return 'CL+NB'
        else:
            return None


# --- Loading Dialog ---
class SimbadLoadingDialog(QDialog):
    """Simple loading dialog to show while querying SIMBAD"""

    def __init__(self, search_term, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Searching SIMBAD")
        self.setModal(True)
        self.setFixedSize(400, 100)

        # Remove close button
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        layout = QVBoxLayout()

        # Message label
        message = QLabel(f"Searching for '{search_term}' in SIMBAD database...\n\nPlease wait...")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("font-size: 12pt; color: white;")
        layout.addWidget(message)

        self.setLayout(layout)

        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
            }
        """)


# --- Custom Visibility Window Class ---
class CustomDSOVisibilityWindow(QDialog):
    """Custom wrapper for the DSO Visibility Calculator"""

    def __init__(self, dso_name: str, parent=None, ra_deg=None, dec_deg=None):
        super().__init__(parent)
        self.setWindowTitle(f"{dso_name} - DSO Visibility Calculator - Cosmos Collection")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1200, 800)
        WindowPositionManager.restore_window_position(self, "CustomDSOVisibility")

        # Create layout
        layout = QVBoxLayout()

        # Create the visibility app widget (lazy-loaded)
        from DSOVisibilityCalculator import DSOVisibilityApp
        self.visibility_app = DSOVisibilityApp()

        # Pre-populate with the DSO name
        self.visibility_app.dso_input.setText(dso_name)

        # If coordinates are provided, set them for direct calculation
        if ra_deg is not None and dec_deg is not None:
            self.visibility_app.set_dso_coordinates(ra_deg, dec_deg)

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

    def closeEvent(self, event):
        """Save window position when closing"""
        WindowPositionManager.save_window_position(self, "CustomDSOVisibility")
        event.accept()


# --- Aladin Lite Viewer Window ---
class AladinLiteWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "AladinLite"
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{data['name']} - Aladin Lite - Cosmos Collection")
        self.resize(1200, 800)
        self.setup_window_position()

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
        self.show_telescope_fov.toggled.connect(self._on_show_fov_toggled)
        
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
        
        # ZWO ASI cameras
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

        # Load persistent settings before creating web view
        self._load_aladin_settings()

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

            # Enable WebGL and hardware acceleration for Aladin Lite
            try:
                from PySide6.QtWebEngineCore import QWebEngineSettings
                settings = self.web_view.settings()
                # Enable WebGL - critical for Aladin Lite rendering
                settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanvasAccessEnabled, True)
                # Enable JavaScript (required for Aladin Lite)
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, True)
                # Enable local storage and other web features
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
                logger.debug("WebGL, JavaScript, and hardware acceleration enabled for Aladin window")
            except Exception as e:
                logger.warning(f"Could not enable WebGL settings: {e}")

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

    def _load_aladin_settings(self):
        """Load persistent Aladin Lite settings from QSettings"""
        try:
            settings = QSettings("CosmosCollection", "AladinLite")

            # Block signals during loading to prevent save being triggered
            self.telescope_combo.blockSignals(True)
            self.show_telescope_fov.blockSignals(True)
            self.camera_combo.blockSignals(True)
            self.barlow_combo.blockSignals(True)

            # Load telescope selection
            saved_telescope_name = settings.value("telescope_name", None)
            if saved_telescope_name:
                # Find the telescope in the combo box
                for i in range(self.telescope_combo.count()):
                    data = self.telescope_combo.itemData(i)
                    if data and data.get('name') == saved_telescope_name:
                        self.telescope_combo.setCurrentIndex(i)
                        self.selected_telescope = data
                        logger.debug(f"Restored telescope selection: {saved_telescope_name}")
                        break

            # Load show telescope FOV checkbox
            show_fov = settings.value("show_telescope_fov", False, type=bool)
            self.show_telescope_fov.setChecked(show_fov)
            logger.debug(f"Restored show telescope FOV: {show_fov}")

            # Load camera/eyepiece selection
            saved_camera_text = settings.value("camera_eyepiece", None)
            if saved_camera_text:
                index = self.camera_combo.findText(saved_camera_text)
                if index >= 0:
                    self.camera_combo.setCurrentIndex(index)
                    logger.debug(f"Restored camera/eyepiece selection: {saved_camera_text}")

            # Load barlow/reducer selection
            saved_barlow_text = settings.value("barlow_reducer", None)
            if saved_barlow_text:
                index = self.barlow_combo.findText(saved_barlow_text)
                if index >= 0:
                    self.barlow_combo.setCurrentIndex(index)
                    logger.debug(f"Restored barlow/reducer selection: {saved_barlow_text}")

            # Unblock signals
            self.telescope_combo.blockSignals(False)
            self.show_telescope_fov.blockSignals(False)
            self.camera_combo.blockSignals(False)
            self.barlow_combo.blockSignals(False)

            logger.debug("Aladin Lite settings loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Aladin settings: {str(e)}")
            # Make sure to unblock signals even if there's an error
            self.telescope_combo.blockSignals(False)
            self.show_telescope_fov.blockSignals(False)
            self.camera_combo.blockSignals(False)
            self.barlow_combo.blockSignals(False)

    def _save_aladin_settings(self):
        """Save persistent Aladin Lite settings to QSettings"""
        try:
            settings = QSettings("CosmosCollection", "AladinLite")

            # Save telescope selection
            telescope_data = self.telescope_combo.currentData()
            if telescope_data:
                settings.setValue("telescope_name", telescope_data.get('name'))
            else:
                settings.setValue("telescope_name", None)

            # Save show telescope FOV checkbox
            settings.setValue("show_telescope_fov", self.show_telescope_fov.isChecked())

            # Save camera/eyepiece selection
            settings.setValue("camera_eyepiece", self.camera_combo.currentText())

            # Save barlow/reducer selection
            settings.setValue("barlow_reducer", self.barlow_combo.currentText())

            logger.debug("Aladin Lite settings saved successfully")

        except Exception as e:
            logger.error(f"Error saving Aladin settings: {str(e)}")

    def _on_telescope_changed(self):
        """Handle telescope selection change"""
        current_data = self.telescope_combo.currentData()
        if current_data:
            self.selected_telescope = current_data
            logger.debug(f"Selected telescope: {current_data['name']} ({current_data['focal_length']}mm)")
        else:
            self.selected_telescope = None
            logger.debug("Selected default view")

        self._save_aladin_settings()
        self._update_aladin_view()

    def _on_camera_changed(self):
        """Handle camera/sensor selection change"""
        self._save_aladin_settings()
        self._update_aladin_view()

    def _on_barlow_changed(self):
        """Handle barlow/reducer selection change"""
        self._save_aladin_settings()
        self._update_aladin_view()

    def _on_show_fov_toggled(self):
        """Handle show telescope FOV checkbox toggle"""
        self._save_aladin_settings()
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

            # Check WebGL availability for diagnostics
            self._check_webgl_support()

            # Handle FOV overlay injection if needed
            if self.pending_fov_overlay and self.target_coordinates:
                self._inject_fov_overlay(True)

    def _check_webgl_support(self):
        """Check if WebGL is available in the browser context"""
        if not self.web_view:
            return

        # JavaScript to check WebGL support
        js_code = """
        (function() {
            var canvas = document.createElement('canvas');
            var gl = canvas.getContext('webgl') || canvas.getContext('webgl2') || canvas.getContext('experimental-webgl');
            if (gl) {
                var debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                var vendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown';
                var renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown';
                return JSON.stringify({
                    available: true,
                    version: gl instanceof WebGL2RenderingContext ? 'WebGL2' : 'WebGL1',
                    vendor: vendor,
                    renderer: renderer
                });
            } else {
                return JSON.stringify({available: false, error: 'WebGL not available'});
            }
        })();
        """

        def handle_webgl_result(result):
            try:
                import json
                webgl_info = json.loads(result)
                if webgl_info.get('available'):
                    logger.info(f"WebGL is available: {webgl_info.get('version', 'Unknown')} - Vendor: {webgl_info.get('vendor', 'Unknown')}, Renderer: {webgl_info.get('renderer', 'Unknown')}")
                else:
                    logger.error(f"WebGL is NOT available: {webgl_info.get('error', 'Unknown error')}")
                    logger.error("Aladin Lite will not render properly without WebGL support")
            except Exception as e:
                logger.error(f"Failed to parse WebGL info: {e}, result: {result}")

        try:
            self.web_view.page().runJavaScript(js_code, handle_webgl_result)
        except Exception as e:
            logger.error(f"Failed to check WebGL support: {e}")

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
            import ssl

            def test_connection():
                try:
                    # Disable SSL verification for PyInstaller builds
                    context = ssl._create_unverified_context() if getattr(sys, 'frozen', False) else None
                    with urllib.request.urlopen("https://aladin.u-strasbg.fr", timeout=10, context=context) as response:
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
        WindowPositionManager.restore_window_position(self, "ImageViewer")

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

            # Add set as background button
            set_bg_button = QPushButton("Set as Background")
            set_bg_button.setFixedSize(130, 30)
            set_bg_button.clicked.connect(self._set_as_background)
            toolbar.addWidget(set_bg_button)

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

    def _set_as_background(self):
        """Set the current image as the desktop background"""
        if not self.file_path:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Warning", "No file path available")
            return

        try:
            import platform
            import ctypes
            from pathlib import Path

            # Check if file is a FITS file
            file_ext = Path(self.file_path).suffix.lower()
            if file_ext in ['.fits', '.fit', '.fts']:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Unsupported Format",
                                  "FITS files cannot be set as desktop background.\n"
                                  "Please export to a standard image format (PNG, JPG, etc.) first.")
                return

            # Ensure the file path is absolute
            abs_path = str(Path(self.file_path).resolve())

            if platform.system() == "Windows":
                # Windows implementation using ctypes
                SPI_SETDESKWALLPAPER = 20
                SPIF_UPDATEINIFILE = 0x01
                SPIF_SENDCHANGE = 0x02

                # Call SystemParametersInfoW to set the wallpaper
                result = ctypes.windll.user32.SystemParametersInfoW(
                    SPI_SETDESKWALLPAPER,
                    0,
                    abs_path,
                    SPIF_UPDATEINIFILE | SPIF_SENDCHANGE
                )

                if result:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Success", "Desktop background updated successfully!")
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Error", "Failed to set desktop background")

            elif platform.system() == "Darwin":
                # macOS implementation
                import subprocess
                script = f'''
                tell application "Finder"
                    set desktop picture to POSIX file "{abs_path}"
                end tell
                '''
                subprocess.run(["osascript", "-e", script], check=True)
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", "Desktop background updated successfully!")

            elif platform.system() == "Linux":
                # Linux implementation (works with GNOME)
                import subprocess
                try:
                    # Try GNOME
                    subprocess.run([
                        "gsettings", "set", "org.gnome.desktop.background",
                        "picture-uri", f"file://{abs_path}"
                    ], check=True)
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Success", "Desktop background updated successfully!")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Try other desktop environments if needed
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Warning",
                                      "Could not set background. This feature may not be supported on your desktop environment.")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Warning",
                                  f"Setting desktop background is not supported on {platform.system()}")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to set desktop background: {str(e)}")

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

    def closeEvent(self, event):
        """Save window position when closing"""
        WindowPositionManager.save_window_position(self, "ImageViewer")
        event.accept()


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
        WindowPositionManager.restore_window_position(self, "ObjectDetail")

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

        # Background threads
        self.image_loader_thread = None  # Background thread for async image loading
        self.simbad_query_thread = None  # Background thread for SIMBAD queries

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
        # Note: _load_user_images() will call _load_current_image_info() to populate the form
        QTimer.singleShot(300, self._load_user_images)

        # Query SIMBAD for emission line data (deferred to avoid blocking)
        QTimer.singleShot(400, self._query_emission_data)

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

    def _query_emission_data(self):
        """Query SIMBAD for emission line and spectral information in background thread"""
        try:
            # Only query for nebulae (emission regions)
            dso_type = self.data.get('dso_type', '')
            if dso_type not in ['BRTNB', 'CL+NB', 'PLNNB', 'SNREM']:
                self.emission_label.setText("Not applicable (not an emission nebula)")
                self.emission_label.setStyleSheet("color: gray;")
                return

            # Get object coordinates and name
            object_name = self.data.get('name', '')
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')

            # Show querying status
            self.emission_label.setText("Querying SIMBAD...")
            self.emission_label.setStyleSheet("color: gray;")

            # Stop any existing SIMBAD query thread
            if self.simbad_query_thread and self.simbad_query_thread.isRunning():
                self.simbad_query_thread.quit()
                self.simbad_query_thread.wait()

            # Start background SIMBAD query thread
            self.simbad_query_thread = SimbadQueryThread(object_name, ra_deg, dec_deg, dso_type)
            self.simbad_query_thread.query_complete.connect(self._on_simbad_query_complete)
            self.simbad_query_thread.query_failed.connect(self._on_simbad_query_failed)
            self.simbad_query_thread.start()

            logger.debug(f"Started background SIMBAD query for {object_name}")

        except Exception as e:
            logger.error(f"Error initiating SIMBAD query: {str(e)}", exc_info=True)
            self.emission_label.setText(f"Error querying SIMBAD (check internet connection)")
            self.emission_label.setStyleSheet("color: #ff6b6b;")

    def _on_simbad_query_complete(self, result, object_name, ra_deg, dec_deg):
        """Handle SIMBAD query completion"""
        try:
            dso_type = self.data.get('dso_type', '')

            # Parse and display emission line info based on object type
            emission_info = self._parse_emission_info(dso_type, result)

            if emission_info:
                self.emission_label.setText(emission_info)
                self.emission_label.setStyleSheet("color: white;")
            else:
                self.emission_label.setText("No specific emission line data available")
                self.emission_label.setStyleSheet("color: gray;")

        except Exception as e:
            logger.error(f"Error processing SIMBAD result: {str(e)}", exc_info=True)
            self.emission_label.setText("Error processing SIMBAD data")
            self.emission_label.setStyleSheet("color: #ff6b6b;")

    def _on_simbad_query_failed(self, object_name, error_message):
        """Handle SIMBAD query failure"""
        logger.warning(f"SIMBAD query failed for {object_name}: {error_message}")
        self.emission_label.setText(f"SIMBAD query failed (check internet connection)")
        self.emission_label.setStyleSheet("color: #ff6b6b;")

    def _parse_emission_info(self, dso_type, simbad_result):
        """Parse emission line information based on DSO type and SIMBAD data"""
        # Common emission lines for different nebula types
        emission_lines = {
            'BRTNB': {  # Bright nebula (emission/reflection)
                'primary': ['Hα (656.3 nm)', 'Hβ (486.1 nm)'],
                'secondary': ['OIII (495.9, 500.7 nm)', 'SII (671.6, 673.1 nm)', 'NII (658.3 nm)'],
                'description': 'Emission nebula - typically rich in hydrogen and oxygen'
            },
            'CL+NB': {  # Cluster + nebula
                'primary': ['Hα (656.3 nm)'],
                'secondary': ['OIII (495.9, 500.7 nm)', 'Hβ (486.1 nm)'],
                'description': 'Star cluster with associated emission nebulosity'
            },
            'PLNNB': {  # Planetary nebula
                'primary': ['OIII (495.9, 500.7 nm)', 'Hα (656.3 nm)'],
                'secondary': ['Hβ (486.1 nm)', 'NII (658.3 nm)'],
                'description': 'Planetary nebula - strong in oxygen and hydrogen'
            },
            'SNREM': {  # Supernova remnant
                'primary': ['OIII (495.9, 500.7 nm)', 'SII (671.6, 673.1 nm)'],
                'secondary': ['Hα (656.3 nm)', 'Hβ (486.1 nm)'],
                'description': 'Supernova remnant - often oxygen and sulfur rich'
            }
        }

        if dso_type not in emission_lines:
            return None

        info = emission_lines[dso_type]
        text = ""

        # Add SIMBAD data if available
        simbad_data_available = False
        if simbad_result is not None and len(simbad_result) > 0:
            simbad_data_available = True
            row = simbad_result[0]  # Get first result

            text += "<b>SIMBAD Data:</b>"

            # Object type from SIMBAD
            if 'OTYPE' in row.colnames and row['OTYPE']:
                otype = str(row['OTYPE']).strip()
                text += f"<b>Object Type:</b> {otype}<br>"

            # Spectral type
            if 'SP_TYPE' in row.colnames and row['SP_TYPE']:
                sp_type = str(row['SP_TYPE']).strip()
                if sp_type and sp_type != '--':
                    text += f"<b>Spectral Type:</b> {sp_type}<br>"

            # Flux measurements (photometry)
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
        else:
            # Indicate fallback to built-in data
            text += ""

        # Always show built-in emission line data
        text += f"<b>{info['description']}</b><br><br>"
        text += "<b>Primary Emission Lines:</b><br>"
        for line in info['primary']:
            # Color code the lines
            if 'Hα' in line or 'Hβ' in line:
                color = '#ff6b6b'  # Red for hydrogen
            elif 'OIII' in line:
                color = '#6bcdff'  # Blue-green for oxygen
            elif 'SII' in line:
                color = '#ff9966'  # Orange-red for sulfur
            elif 'NII' in line:
                color = '#ff8888'  # Light red for nitrogen
            else:
                color = 'white'
            text += f"<span style='color: {color};'>• {line}</span><br>"

        text += "<br><b>Secondary Lines:</b><br>"
        for line in info['secondary']:
            # Color code the lines
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
        logger.debug(f"_setup_ui called for {self.data['name']}")
        try:
            # Create menu bar using QToolBar with proper menu buttons
            from PySide6.QtWidgets import QToolButton
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
            aladin_button.setText("Aladin Lite\FOV Simulator")
            aladin_button.setToolTip("Open Aladin Lite interactive sky atlas with telescope field of view simulator")
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

            self.prev_image_button = QPushButton("⬅️")
            self.prev_image_button.setFixedSize(30, 30)
            self.prev_image_button.clicked.connect(self._previous_image)
            self.prev_image_button.setToolTip("Previous image")
            zoom_layout.addWidget(self.prev_image_button)

            self.image_counter_label = QLabel("1/1")
            self.image_counter_label.setStyleSheet("font-size: 10pt; color: #666666; padding: 0 5px;")
            self.image_counter_label.setMinimumWidth(40)
            self.image_counter_label.setAlignment(Qt.AlignCenter)
            zoom_layout.addWidget(self.image_counter_label)

            self.next_image_button = QPushButton("➡️")
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

            # Favorite image button
            self.favorite_button = QPushButton("⭐")
            self.favorite_button.setFixedSize(30, 30)
            self.favorite_button.clicked.connect(self._toggle_favorite)
            self.favorite_button.setToolTip("Mark as favorite")
            self.favorite_button.setStyleSheet("QPushButton { font-size: 12pt; }")
            zoom_layout.addWidget(self.favorite_button)

            zoom_layout.addStretch()
            image_container_layout.addLayout(zoom_layout)

            # Image label
            self.image_label = QLabel("Loading...")
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

            object_info_label = QLabel(object_info_text)
            object_info_label.setAlignment(Qt.AlignLeft)
            object_info_label.setWordWrap(True)
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
            self.emission_label.setStyleSheet("color: gray; font-style: italic;")
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

            # Get coordinates from the data
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')

            # Create new visibility window with coordinates
            self.visibility_window = CustomDSOVisibilityWindow(object_name, self, ra_deg, dec_deg)
            self.visibility_window.show()

            logger.debug(f"Opened visibility calculator for {object_name} with coordinates RA={ra_deg}, Dec={dec_deg}")

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

                        # Insert new image record with blank metadata (user will fill in after)
                        cursor.execute("""
                            INSERT INTO userimages (
                                dsodetailid, image_path, integration_time,
                                equipment, date_taken, notes
                            ) VALUES (?, ?, ?, ?, ?, ?)
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
                                    self.image_label.setStyleSheet("font-size: 14pt; color: gray;")

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
        """Load and display a user image with caching (supports FITS files) - async version"""
        try:
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
        import os
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

    def _open_wikipedia(self):
        """Open Wikipedia page for the current DSO in the default browser"""
        try:
            import webbrowser
            import re

            # Get the DSO name
            dso_name = self.data.get('name', '').strip()

            if not dso_name:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Name", "Cannot open Wikipedia: DSO name not available.")
                return

            # Strip out parentheses and their contents (e.g., "IC 410 (Tadpoles Nebula)" -> "IC 410")
            dso_name = re.sub(r'\s*\([^)]*\)', '', dso_name).strip()

            # Convert M catalog names to "Messier" format (e.g., M31 or M 31 -> Messier_31)
            if re.match(r'^M\s*\d+$', dso_name):
                # Extract the number and format as "Messier_XX"
                number = re.search(r'\d+', dso_name).group()
                wiki_name = f'Messier_{number}'
            else:
                # For other catalogs, replace spaces with underscores
                wiki_name = dso_name.replace(' ', '_')

            wiki_url = f"https://en.wikipedia.org/wiki/{wiki_name}"

            logger.debug(f"Opening Wikipedia page: {wiki_url}")
            webbrowser.open(wiki_url)

        except Exception as e:
            logger.error(f"Error opening Wikipedia: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
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

                    # Get all images for this object, ordering by favorite first, then by id
                    cursor.execute("""
                        SELECT id, image_path, integration_time, equipment, date_taken, notes, is_favorite
                        FROM userimages
                        WHERE dsodetailid = ?
                        ORDER BY is_favorite DESC, id ASC
                    """, (dsodetailid,))

                    images = cursor.fetchall()

                    # Store images in our list
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

                    # Update navigation controls
                    self._update_image_navigation()

                    # Load the first image if available (will be favorite if one exists)
                    if self.user_images:
                        # Update text to show loading state
                        self.image_label.setText("Loading image...")
                        self.image_label.setStyleSheet("font-size: 14pt; color: gray;")

                        self.current_image_index = 0
                        current_image = self.user_images[self.current_image_index]
                        self._load_user_image(current_image['image_path'])
                        self._load_current_image_info()
                        self.info_form_container.setVisible(True)
                    else:
                        # No images available - update to show no image message
                        self.image_label.setText("No image attached to this DSO.")
                        self.image_label.setStyleSheet("font-size: 14pt; color: gray;")
                        self.info_form_container.setVisible(False)
                else:
                    logger.error(f"Could not find dsodetailid for {self.data['name']}")
                    self.image_label.setText("No image attached to this DSO.")
                    self.image_label.setStyleSheet("font-size: 14pt; color: gray;")
                    self.info_form_container.setVisible(False)

        except Exception as e:
            logger.error(f"Error loading user images: {str(e)}", exc_info=True)
            self.image_label.setText("Error loading images.")
            self.image_label.setStyleSheet("font-size: 14pt; color: #ff6b6b;")
            self.info_form_container.setVisible(False)

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
            # Show loading state
            self.image_label.setText("Loading image...")
            self.image_label.setStyleSheet("font-size: 14pt; color: gray;")

            self.current_image_index -= 1
            current_image = self.user_images[self.current_image_index]
            self._load_user_image(current_image['image_path'])
            self._load_current_image_info()
            self._update_image_navigation()
            logger.debug(f"Navigated to previous image: {self.current_image_index + 1}/{len(self.user_images)}")

    def _next_image(self):
        """Navigate to the next image"""
        if self.user_images and self.current_image_index < len(self.user_images) - 1:
            # Show loading state
            self.image_label.setText("Loading image...")
            self.image_label.setStyleSheet("font-size: 14pt; color: gray;")

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

            # Update favorite button appearance
            is_favorite = current_image.get('is_favorite', 0)
            if is_favorite:
                self.favorite_button.setText("⭐")
                self.favorite_button.setStyleSheet("QPushButton { color: #FFD700; font-size: 12pt; }")
                self.favorite_button.setToolTip("Unmark as favorite")
            else:
                self.favorite_button.setText("☆")
                self.favorite_button.setStyleSheet("QPushButton { color: #888888; font-size: 12pt; }")
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

                # Get current favorite status
                current_favorite = current_image.get('is_favorite', 0)
                new_favorite = 1 if not current_favorite else 0

                # If setting as favorite, unfavorite all other images for this DSO
                if new_favorite == 1:
                    dsodetailid = current_image.get('dsodetailid')
                    if not dsodetailid:
                        # Get dsodetailid from the database
                        cursor.execute("""
                            SELECT dsodetailid FROM userimages WHERE id = ?
                        """, (image_id,))
                        result = cursor.fetchone()
                        if result:
                            dsodetailid = result[0]

                    if dsodetailid:
                        # Unfavorite all other images for this DSO
                        cursor.execute("""
                            UPDATE userimages
                            SET is_favorite = 0
                            WHERE dsodetailid = ? AND id != ?
                        """, (dsodetailid, image_id))

                # Toggle favorite status for this image
                cursor.execute("""
                    UPDATE userimages
                    SET is_favorite = ?
                    WHERE id = ?
                """, (new_favorite, image_id))

                conn.commit()

                # Update the local cache
                current_image['is_favorite'] = new_favorite

                # Update all images in the list if we unfavorited others
                if new_favorite == 1:
                    for img in self.user_images:
                        if img['id'] != image_id:
                            img['is_favorite'] = 0

                # Update button appearance
                self._load_current_image_info()

                logger.debug(f"Toggled favorite status for image {image_id} to {new_favorite}")

        except Exception as e:
            logger.error(f"Error toggling favorite: {str(e)}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to toggle favorite: {str(e)}")

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
                    # Show loading state
                    self.image_label.setText("Loading image...")
                    self.image_label.setStyleSheet("font-size: 14pt; color: gray;")
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
        """Update target list menu action visibility based on whether DSO is already in target list"""
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
            # Show add action as fallback
            self.add_target_action.setVisible(True)
            self.remove_target_action.setVisible(False)
            self.open_target_action.setVisible(False)
    
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

    def _find_target_list_name(self):
        """Find the actual name used in the target list for this DSO

        Returns:
            str: The name as stored in the target list, or None if not found
        """
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')

                if not dso_name:
                    return None

                # First try exact name match
                cursor.execute("""
                    SELECT name FROM usertargetlist
                    WHERE UPPER(TRIM(name)) = ?
                    LIMIT 1
                """, (dso_name.upper(),))

                result = cursor.fetchone()
                if result:
                    return result[0]

                # If coordinates are available, search by coordinates
                if ra_deg is not None and dec_deg is not None:
                    cursor.execute("""
                        SELECT name FROM usertargetlist
                        WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001
                        LIMIT 1
                    """, (ra_deg, dec_deg))

                    result = cursor.fetchone()
                    if result:
                        return result[0]

                return None

        except Exception as e:
            logger.error(f"Error finding target list name: {str(e)}")
            return None
    
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

    def _open_from_target_list(self):
        """Open DSO from target list to view notes"""
        try:
            dso_name = self.data.get('name', '').strip()
            if not dso_name:
                QMessageBox.warning(self, "Error", "DSO name not found")
                return

            # First, try to find the actual name in the target list
            target_name = self._find_target_list_name()
            if not target_name:
                QMessageBox.warning(self, "Not Found",
                                  f"Could not find '{dso_name}' in your target list.")
                return

            # Import and open Target List window
            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, 'target_list_window') or not self.target_list_window.isVisible():
                self.target_list_window = DSOTargetListWindow()

            # Open and select the target using the actual name from target list
            success = self.target_list_window.open_and_select_target(target_name)

            if not success:
                QMessageBox.warning(self, "Not Found",
                                  f"Could not find '{target_name}' in your target list.")

        except Exception as e:
            logger.error(f"Error opening from target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open from target list:\n{str(e)}")

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

    def closeEvent(self, event):
        """Save window position when closing"""
        WindowPositionManager.save_window_position(self, "ObjectDetail")
        event.accept()


class CustomTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None

    def setModel(self, model):
        super().setModel(model)
        self.model = model
        # Connect the header's sort indicator change to the model's sort method
        self.horizontalHeader().sortIndicatorChanged.connect(self._on_sort_indicator_changed)
        # Connect scroll events for lazy loading
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def _on_sort_indicator_changed(self, logical_index, order):
        """Handle sort indicator changes by calling the model's sort method"""
        if self.model:
            self.model.sort(logical_index, order)

    def _on_scroll(self, value):
        """Handle scroll events to trigger lazy loading"""
        if hasattr(self.model, 'check_and_load_more_data'):
            # Calculate which row is at the bottom of the visible area
            viewport_height = self.viewport().height()
            row_height = self.rowHeight(0) if self.model.rowCount() > 0 else 25
            visible_rows = viewport_height // row_height if row_height > 0 else 0
            current_top_row = self.rowAt(0)
            bottom_visible_row = current_top_row + visible_rows

            # Get the actual last visible row from viewport
            last_visible_index = self.indexAt(self.viewport().rect().bottomLeft())
            actual_last_visible_row = last_visible_index.row() if last_visible_index.isValid() else -1

            # Use the actual visible row for better accuracy
            effective_bottom_row = max(bottom_visible_row, actual_last_visible_row)

            # Trigger lazy loading if needed
            self.model.check_and_load_more_data(effective_bottom_row)


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
class MainWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "MainWindow"
    def __init__(self, dso_data, catalogs, total_count=None):
        super().__init__()
        logger.debug("Initializing MainWindow")

        self.total_dso_count = total_count or len(dso_data)
        self.loaded_count = len(dso_data)
        self.load_offset = self.loaded_count

        # Set window title with version
        try:
            from version import get_version_display
            window_title = f"Cosmos Collection - {get_version_display()}"
        except ImportError:
            window_title = "Cosmos Collection"

        self.setWindowTitle(window_title)
        self.resize(1200, 800)
        self.setWindowFlags(Qt.Window)
        self.setup_window_position()
        self.db_manager = DatabaseManager()
        self._cached_dso_data = None
        self._cached_catalogs = None

        # Store original data for lazy loading
        self.initial_dso_data = dso_data
        self.all_catalogs = catalogs

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
        self.model = DSOTableModel(dso_data, parent=self, db_manager=self.db_manager, total_count=self.total_dso_count)
        self.table_view = CustomTableView()
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.doubleClicked.connect(self._on_double_click)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSortingEnabled(True)

        # Set default sort by catalog (column 0) in ascending order
        self.table_view.sortByColumn(0, Qt.AscendingOrder)

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

        # Exit startup mode after initialization to enable full sorting
        QTimer.singleShot(1000, self._exit_startup_mode)  # Small delay to ensure everything is loaded

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
        visibility_action = QAction("Visibility Calculator", self)
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

        # DSO Image Gallery action
        gallery_action = QAction("Image Gallery", self)
        gallery_action.setToolTip("Browse all DSO images in a gallery view")
        gallery_action.triggered.connect(self._show_dso_gallery)
        toolbar.addAction(gallery_action)

        # Aladin Lite action
        aladin_lite_action = QAction("Aladin Lite\FOV Simulator", self)
        aladin_lite_action.setToolTip("Open Aladin Lite sky viewer")
        aladin_lite_action.triggered.connect(self._show_aladin_lite_from_toolbar)
        toolbar.addAction(aladin_lite_action)

        toolbar.addSeparator()

        # SIMBAD Search action
        #simbad_action = QAction("Search SIMBAD", self)
        #simbad_action.setToolTip("Search for an object in SIMBAD and add to database")
        #simbad_action.triggered.connect(self._show_simbad_search_dialog)
        #toolbar.addAction(simbad_action)

        #toolbar.addSeparator()
        
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

    def _show_dso_gallery(self):
        """Show the DSO Image Gallery window"""
        try:
            from DSOGallery import DSOGalleryWindow
            if not hasattr(self, 'gallery_window') or not self.gallery_window.isVisible():
                self.gallery_window = DSOGalleryWindow()
            self.gallery_window.show()
            self.gallery_window.raise_()
            self.gallery_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load DSO Image Gallery: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open DSO Image Gallery: {e}")

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

        # Check if we need to load more data due to filter reducing visible results
        self._check_filter_needs_more_data()

    def _check_filter_needs_more_data(self):
        """Check if current filter results are too sparse and trigger loading more data"""
        if hasattr(self.model, 'check_and_load_more_data'):
            # For sparse results, keep loading until we have enough or reach the end
            filtered_len = len(self.model.filtered_data)
            loaded_len = len(self.model.dso_data)

            logger.debug(f"Filter check: {filtered_len} filtered, {loaded_len} loaded, {self.model.total_count} total")

            # Check if we're searching for a specific object with catalog filter
            search_text = self.search_input.text()
            catalog = self.catalog_combo.currentText()
            is_specific_search = (search_text and
                                 catalog != "All Catalogs" and
                                 len(self.model.filtered_data) == 0)

            # If searching for a specific object and found nothing, be very aggressive about loading
            if (is_specific_search and
                loaded_len < self.model.total_count and
                not getattr(self.model, 'loading', False)):

                logger.debug(f"Specific search for {catalog} {search_text} - loading all data...")
                # Mark this as an active catalog search
                self._active_catalog_search = (search_text, catalog)
                # Start loading - the model's _on_data_loaded will check _check_search_needs_continue
                self.model.load_more_data()
                return

            # If we have very few results and more data available, trigger loading immediately
            if (filtered_len < 100 and
                loaded_len < self.model.total_count and
                not getattr(self.model, 'loading', False)):

                logger.debug(f"Triggering immediate load due to sparse filter results ({filtered_len} < 100)")
                self.model.load_more_data()
            else:
                # Normal check with any view position to evaluate all trigger conditions
                self.model.check_and_load_more_data(0)

    def _on_highlight_no_images_changed(self, state):
        self.model.setHighlightNoImages(state != 0)

    def _clear_filters(self):
        """Clear all filters"""
        logger.debug("Clear filters button pressed")
        logger.debug(f"Current state: catalog={self.model.selected_catalog}, type={getattr(self.model, '_current_selected_type', None)}")
        logger.debug(f"Current data: dso_data={len(self.model.dso_data)}, filtered_data={len(self.model.filtered_data)}")

        # Block signals to prevent multiple filter triggers
        self.search_input.blockSignals(True)
        self.catalog_combo.blockSignals(True)
        self.show_images_only.blockSignals(True)
        self.highlight_no_images.blockSignals(True)
        self.type_combo.blockSignals(True)

        self.search_input.clear()
        self.catalog_combo.setCurrentIndex(0)
        self.show_images_only.setChecked(False)
        self.highlight_no_images.setChecked(False)
        self.type_combo.setCurrentIndex(0)

        # Unblock signals
        self.search_input.blockSignals(False)
        self.catalog_combo.blockSignals(False)
        self.show_images_only.blockSignals(False)
        self.highlight_no_images.blockSignals(False)
        self.type_combo.blockSignals(False)

        logger.debug("Calling filter_data with all filters cleared")
        # Manually trigger filter update once
        self.model.filter_data("", None, False, None)

        # Re-apply default sort after clearing filters
        self.table_view.sortByColumn(0, Qt.AscendingOrder)

        self._update_status()

        logger.debug(f"After clear: dso_data={len(self.model.dso_data)}, filtered_data={len(self.model.filtered_data)}")

    def _on_search(self, text):
        """Handle search text changes"""
        selected_catalog = None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText()

        self.model.filter_data(
            text,
            selected_catalog,
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()

        # If searching for a specific designation with a catalog filter, keep loading until we find it or run out of data
        if text and selected_catalog and self.model.load_offset < self.model.total_count:
            # Check if we found an exact match
            found_exact_match = False
            search_lower = text.lower()
            for item in self.model.filtered_data:
                designations = item["designations"].split(", ")
                if any(designation.lower() == f"{selected_catalog.lower()} {search_lower}" for designation in designations):
                    found_exact_match = True
                    break

            # If no exact match and more data available, trigger loading
            if not found_exact_match:
                logger.debug(f"No exact match for {selected_catalog} {text} yet, continuing to load data...")
                self._check_filter_needs_more_data()
        else:
            # Check if we need more data for this filter
            self._check_filter_needs_more_data()

        # Check if we should query SIMBAD for this object
        # Only trigger if:
        # 1. Search text is not empty and looks like an object designation
        # 2. No results found in local database
        # 3. All data has been loaded (not still in lazy loading)
        # 4. No other filters are active (catalog/type filters should be "All")
        if (text and
            len(self.model.filtered_data) == 0 and
            self.model.load_offset >= self.model.total_count and
            self.catalog_combo.currentText() == "All Catalogs" and
            self._get_selected_type() is None and
            not self.show_images_only.isChecked() and
            self._looks_like_object_designation(text)):

            # Use a timer to delay the SIMBAD query slightly
            # This prevents triggering during rapid typing
            if hasattr(self, '_simbad_query_timer'):
                self._simbad_query_timer.stop()

            self._simbad_query_timer = QTimer()
            self._simbad_query_timer.setSingleShot(True)
            self._simbad_query_timer.timeout.connect(lambda: self._prompt_simbad_query(text))
            self._simbad_query_timer.start(1000)  # Wait 1 second after user stops typing

    def _looks_like_object_designation(self, text):
        """Check if text looks like an object designation"""
        text = text.strip()

        # Skip if too long (likely a coordinate or description)
        if len(text) > 30:
            return False

        # Skip if it looks like coordinates (contains ° or : or multiple spaces)
        if any(char in text for char in ['°', ':', '+', '-']) and ' ' in text:
            return False

        # Check if it starts with a common catalog prefix
        common_prefixes = ['M', 'NGC', 'IC', 'Abell', 'UGC', 'PGC', 'Barnard', 'Sharpless', 'LDN', 'Arp', 'VdB']
        for prefix in common_prefixes:
            if text.upper().startswith(prefix.upper() + ' ') or text.upper().startswith(prefix.upper()):
                return True

        # Also accept if it's a simple designation pattern (letters followed by numbers)
        import re
        if re.match(r'^[A-Za-z]+\s*\d+', text):
            return True

        return False

    def _prompt_simbad_query(self, search_term):
        """Prompt user to search SIMBAD for the object"""
        reply = QMessageBox.question(
            self,
            "Object Not Found",
            f"'{search_term}' was not found in the local database.\n\n"
            f"Would you like to search for it in the SIMBAD astronomical database?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self._query_simbad_for_object(search_term)

    def _show_simbad_search_dialog(self):
        """Show dialog to manually search SIMBAD for an object"""
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self,
            "Search SIMBAD",
            "Enter object designation to search in SIMBAD\n(e.g., M 31, NGC 7789, Andromeda Galaxy):",
            QLineEdit.Normal,
            self.search_input.text()  # Pre-populate with current search text
        )

        if ok and text.strip():
            self._query_simbad_for_object(text.strip())

    def _on_catalog_changed(self, catalog):
        """Handle catalog selection changes"""
        self.model.filter_data(
            self.search_input.text(),
            None if catalog == "All Catalogs" else catalog,
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()
        # Check if we need more data for this filter
        self._check_filter_needs_more_data()

    def _on_type_changed(self, type_text):
        """Handle DSO type selection changes"""
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.show_images_only.isChecked(),
            self._get_selected_type()
        )
        self._update_status()
        # Check if we need more data for this filter
        self._check_filter_needs_more_data()

    def _get_selected_type(self):
        """Get the currently selected DSO type code"""
        current_data = self.type_combo.currentData()
        if current_data and self.type_combo.currentText() != "All Types":
            return current_data
        return None

    def _update_status(self):
        """Update the status label"""
        loaded, total_available = self.model.get_load_progress()
        filtered = len(self.model.filtered_data)

        if filtered == loaded:
            if loaded < total_available:
                self.status_label.setText(f"Showing all {loaded} loaded objects ({total_available} total available)")
            else:
                self.status_label.setText(f"Showing all {loaded} objects")
        else:
            if loaded < total_available:
                self.status_label.setText(f"Showing {filtered} of {loaded} loaded objects ({total_available} total available)")
            else:
                self.status_label.setText(f"Showing {filtered} of {loaded} objects")

    def _on_loading_started(self):
        """Handle when background data loading starts"""
        # Update status to show loading
        current_text = self.status_label.text()
        self.status_label.setText(f"{current_text} - Loading more data...")

    def _on_loading_finished(self):
        """Handle when background data loading finishes"""
        # Refresh status display
        self._update_status()

    def _schedule_next_load(self):
        """Schedule the next load after a short delay to avoid recursive loading"""
        # Use a timer to schedule the next load check
        QTimer.singleShot(100, self._check_filter_needs_more_data)

    def _continue_loading_for_search(self, search_text, catalog):
        """Continue loading data until we find a specific object or run out of data"""
        # Check if the search parameters are still the same
        if (self.search_input.text() != search_text or
            self.catalog_combo.currentText() != catalog):
            logger.debug("Search changed, stopping continuous load")
            self._active_catalog_search = None
            return

        # Check if we found the object
        found_exact_match = False
        search_lower = search_text.lower()
        for item in self.model.filtered_data:
            designations = item["designations"].split(", ")
            if any(designation.lower() == f"{catalog.lower()} {search_lower}" for designation in designations):
                found_exact_match = True
                logger.debug(f"Found exact match for {catalog} {search_text}!")
                self._active_catalog_search = None
                break

        # If found or no more data, stop
        if found_exact_match or self.model.load_offset >= self.model.total_count:
            if not found_exact_match:
                logger.debug(f"Reached end of data without finding {catalog} {search_text}")
            self._active_catalog_search = None
            return

        # Otherwise, continue loading
        if not getattr(self.model, 'loading', False):
            logger.debug(f"Continuing to load for {catalog} {search_text}...")
            self.model.load_more_data()
            # Schedule next check
            QTimer.singleShot(200, lambda: self._continue_loading_for_search(search_text, catalog))

    def _check_search_needs_continue(self):
        """Check if we need to continue loading for an active catalog search"""
        if not hasattr(self, '_active_catalog_search') or not self._active_catalog_search:
            return False

        search_text, catalog = self._active_catalog_search

        # Check if search still matches
        if (self.search_input.text() != search_text or
            self.catalog_combo.currentText() != catalog):
            self._active_catalog_search = None
            return False

        # Check if we found it
        search_lower = search_text.lower()
        for item in self.model.filtered_data:
            designations = item["designations"].split(", ")
            if any(designation.lower() == f"{catalog.lower()} {search_lower}" for designation in designations):
                logger.debug(f"Found exact match for {catalog} {search_text}!")
                self._active_catalog_search = None
                return False

        # Still need more data
        return self.model.load_offset < self.model.total_count

    def _exit_startup_mode(self):
        """Exit startup mode for the model to enable full sorting"""
        if hasattr(self.model, 'exit_startup_mode'):
            self.model.exit_startup_mode()

    def _save_simbad_object_to_database(self, object_data):
        """Save SIMBAD object data to the database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Parse main_id to extract catalog and designation for the ID
                # SIMBAD main_id can be like "M 42", "NGC  7789", etc.
                main_id = object_data['main_id'].strip()

                # Remove extra spaces
                main_id_clean = ' '.join(main_id.split())

                # Try to split by common catalog prefixes to create the database ID
                catalog_entries = []
                dso_id = None

                # Common catalogs to look for
                common_catalogs = ['M', 'NGC', 'IC', 'Abell', 'UGC', 'PGC', 'Barnard', 'Sharpless', 'LDN', 'Arp', 'VdB']

                for catalog in common_catalogs:
                    # Check if main_id starts with this catalog
                    if main_id_clean.upper().startswith(catalog.upper() + ' '):
                        designation = main_id_clean[len(catalog)+1:].strip()
                        catalog_entries.append((catalog, designation))
                        # Create database ID (e.g., "NGC7789")
                        dso_id = f"{catalog}{designation.replace(' ', '')}"
                        break

                # If no match found, create a generic ID
                if not catalog_entries:
                    # Try to split on first space
                    parts = main_id_clean.split(' ', 1)
                    if len(parts) == 2:
                        catalog_entries.append((parts[0], parts[1]))
                        dso_id = f"{parts[0]}{parts[1].replace(' ', '')}"
                    else:
                        catalog_entries.append(('SIMBAD', main_id_clean))
                        dso_id = f"SIMBAD{main_id_clean.replace(' ', '')}"

                logger.debug(f"Generated database ID: {dso_id} for {main_id}")

                # Check if object already exists
                cursor.execute("SELECT id FROM dsodetail WHERE id = ?", (dso_id,))
                existing = cursor.fetchone()
                if existing:
                    logger.info(f"Object {dso_id} already exists in database")

                    # Extract catalog and designation for searching
                    catalog, designation = catalog_entries[0] if catalog_entries else (None, None)

                    # Show message and offer to search for it
                    reply = QMessageBox.question(
                        self,
                        "Object Already Exists",
                        f"'{main_id_clean}' (ID: {dso_id}) already exists in the database.\n\n"
                        f"Would you like to search for it and display it?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )

                    if reply == QMessageBox.Yes and catalog:
                        # Clear filters and search for the object
                        self.catalog_combo.setCurrentText(catalog)
                        self.search_input.setText(designation)

                    return dso_id

                # Get constellation from coordinates using astropy
                try:
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    from astropy.coordinates import get_constellation

                    coords = SkyCoord(
                        ra=object_data['ra_deg']*u.degree,
                        dec=object_data['dec_deg']*u.degree,
                        frame='icrs'
                    )
                    constellation = get_constellation(coords)
                except Exception as e:
                    logger.warning(f"Could not determine constellation: {e}")
                    constellation = None

                # Convert size from arcminutes to arcseconds for database storage
                size_maj_arcsec = object_data['size_maj'] * 60.0 if object_data['size_maj'] else None
                size_min_arcsec = object_data['size_min'] * 60.0 if object_data['size_min'] else None

                # Insert into dsodetail table with explicit ID
                cursor.execute("""
                    INSERT INTO dsodetail (
                        id, ra, dec, magnitude, surfacebrightness,
                        sizemin, sizemax, constellation, dsotype, dsoclass
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dso_id,
                    object_data['ra_deg'],
                    object_data['dec_deg'],
                    object_data['magnitude'],
                    None,  # surfacebrightness
                    size_min_arcsec,
                    size_maj_arcsec,
                    constellation,
                    object_data['dso_type'],
                    None  # dsoclass
                ))

                logger.debug(f"Inserted DSO with ID {dso_id}")

                # Insert catalog entries
                for catalog, designation in catalog_entries:
                    cursor.execute("""
                        INSERT INTO cataloguenr (dsodetailid, catalogue, designation)
                        VALUES (?, ?, ?)
                    """, (dso_id, catalog, designation))
                    logger.debug(f"Inserted catalog entry: {catalog} {designation}")

                conn.commit()
                logger.info(f"Successfully saved SIMBAD object: {main_id} with ID {dso_id}")
                return dso_id

        except Exception as e:
            logger.error(f"Error saving SIMBAD object to database: {str(e)}", exc_info=True)
            raise

    def _query_simbad_for_object(self, search_term):
        """Query SIMBAD for an object and add it to the database if found"""
        # Show loading dialog
        self.simbad_loading_dialog = SimbadLoadingDialog(search_term, self)
        self.simbad_loading_dialog.show()

        # Create and start worker thread
        self.simbad_worker = SimbadQueryWorker(search_term, self)
        self.simbad_worker.object_found.connect(self._on_simbad_object_found)
        self.simbad_worker.object_not_found.connect(self._on_simbad_object_not_found)
        self.simbad_worker.error_occurred.connect(self._on_simbad_error)
        self.simbad_worker.start()

    def _on_simbad_object_found(self, object_data):
        """Handle when SIMBAD finds an object"""
        try:
            # Close loading dialog
            if hasattr(self, 'simbad_loading_dialog'):
                self.simbad_loading_dialog.close()

            # Save to database
            self._save_simbad_object_to_database(object_data)

            # Show success message
            QMessageBox.information(
                self,
                "Object Added",
                f"Found '{object_data['main_id']}' in SIMBAD and added it to the database.\n\n"
                f"The object will now appear in your search results."
            )

            # Reload data to show the new object
            self._reload_data()

        except Exception as e:
            logger.error(f"Error handling SIMBAD object: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Found object in SIMBAD but failed to add to database:\n{str(e)}"
            )
        finally:
            # Clean up worker
            if hasattr(self, 'simbad_worker'):
                self.simbad_worker.deleteLater()
                self.simbad_worker = None

    def _on_simbad_object_not_found(self):
        """Handle when SIMBAD doesn't find an object"""
        # Close loading dialog
        if hasattr(self, 'simbad_loading_dialog'):
            self.simbad_loading_dialog.close()

        QMessageBox.information(
            self,
            "Object Not Found",
            "The object was not found in the SIMBAD database.\n\n"
            "Please check the designation and try again."
        )

        # Clean up worker
        if hasattr(self, 'simbad_worker'):
            self.simbad_worker.deleteLater()
            self.simbad_worker = None

    def _on_simbad_error(self, error_message):
        """Handle SIMBAD query errors"""
        # Close loading dialog
        if hasattr(self, 'simbad_loading_dialog'):
            self.simbad_loading_dialog.close()

        QMessageBox.warning(
            self,
            "SIMBAD Query Error",
            f"Error querying SIMBAD:\n\n{error_message}\n\n"
            "Please check your internet connection and try again."
        )

        # Clean up worker
        if hasattr(self, 'simbad_worker'):
            self.simbad_worker.deleteLater()
            self.simbad_worker = None

    def _reload_data(self):
        """Reload all data from database"""
        # Reset the model's data
        self.model.dso_data.clear()
        self.model.filtered_data.clear()
        self.model.load_offset = 0

        # Reload first batch
        self.model.load_more_data()


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
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY
                               CASE c.catalogue
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation) as designations,
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

                # Get major catalogs (with at least 50 objects to filter out minor catalogs)
                cursor.execute("""
                    SELECT catalogue, COUNT(DISTINCT dsodetailid) as count
                    FROM cataloguenr
                    GROUP BY catalogue
                    HAVING count >= 50
                    ORDER BY catalogue
                """)
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

        visibility_action = context_menu.addAction("Visibility Calculator")
        visibility_action.triggered.connect(lambda: self._context_open_visibility(row))

        aladin_action = context_menu.addAction("Aladin Lite\FOV Simulator")
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

            # Set DSO name in input field for display and title
            if hasattr(self.visibility_window, 'dso_input'):
                self.visibility_window.dso_input.setText(dso_name)
                logger.debug(f"Set DSO name in input field: {dso_name}")
            else:
                logger.warning("DSO input field not found in visibility window")

            # Use coordinates for accurate calculation
            if hasattr(self.visibility_window, 'set_dso_coordinates'):
                self.visibility_window.set_dso_coordinates(ra_deg, dec_deg)
                logger.debug(f"Set coordinates: RA {ra_deg}° Dec {dec_deg}°")

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
                'dso_class': entry.get('dso_class', ''),
                'designations': entry.get('designations', '')
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
    # Set environment variables for QtWebEngine to enable WebGL
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--ignore-gpu-blocklist --enable-webgl --enable-webgl2 --enable-gpu-rasterization'

    # Enable WebGL and hardware acceleration for Qt WebEngine (required for Aladin Lite)
    # MUST be done BEFORE QApplication is created
    webgl_args = [
        '--ignore-gpu-blocklist',
        '--enable-gpu-rasterization',
        '--enable-webgl',
        '--enable-webgl2',
        '--disable-software-rasterizer',
        '--use-gl=desktop',
    ]
    # Add WebGL arguments if not already present
    for arg in webgl_args:
        if arg not in sys.argv:
            sys.argv.append(arg)

    # Initialize QtWebEngine BEFORE QApplication to ensure WebGL settings are applied
    try:
        from PySide6.QtWebEngineCore import QtWebEngineCore, QWebEngineProfile
        # This ensures QtWebEngine is initialized with the command-line arguments
        logger.debug("QtWebEngine initialized with WebGL support")
    except ImportError:
        logger.warning("QtWebEngineCore not available - WebGL may not work")

    app = QApplication(sys.argv)

    # Set global WebEngine profile settings
    try:
        from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
        profile = QWebEngineProfile.defaultProfile()
        settings = profile.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        logger.debug("Global WebEngine profile configured with WebGL support")
    except Exception as e:
        logger.warning(f"Could not configure global WebEngine profile: {e}")
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

            # Get major catalogs (with at least 50 objects to filter out minor catalogs)
            cursor.execute("""
                SELECT catalogue, COUNT(DISTINCT dsodetailid) as count
                FROM cataloguenr
                GROUP BY catalogue
                HAVING count >= 50
                ORDER BY catalogue
            """)
            catalogs = [row[0] for row in cursor.fetchall()]

            # Get total count for progress indication
            cursor.execute("SELECT COUNT(DISTINCT d.id) FROM dsodetail d JOIN cataloguenr c ON d.id = c.dsodetailid")
            total_count = cursor.fetchone()[0]
            logger.debug(f"Total DSO count: {total_count}")

            # Load initial batch of objects (first 2000 for faster startup)
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
                LIMIT 2000
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

            logger.debug(f"Loaded initial batch: {len(dso_data)} of {total_count} DSOs")

        if not dso_data:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(None, "Error", "Failed to load DSO data from database")
            sys.exit(1)

        window = MainWindow(dso_data, catalogs, total_count)
        window.show()
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
        sys.exit(1)
    finally:
        db_manager.close()