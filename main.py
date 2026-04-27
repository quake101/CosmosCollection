import argparse
import logging
import os
import sys
import urllib.request
import urllib.error
import json
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
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QUrl, Signal, QObject, QTimer, QEvent, QThread, QSettings, Slot
from PySide6.QtGui import QPixmap, QPainter, QIcon, QColor, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView,
    QVBoxLayout, QWidget, QLabel, QDialog,
    QHeaderView, QPushButton, QHBoxLayout, QLineEdit, QComboBox, QTextEdit, QCheckBox, QGroupBox,
    QToolBar, QMessageBox, QMenu, QScrollArea, QGridLayout, QSpinBox, QFileDialog, QSizePolicy,
    QListWidget, QListWidgetItem, QCompleter, QSplitter, QSystemTrayIcon,
    QTableWidget, QTableWidgetItem
)

# Local imports (always needed)
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionManager, WindowPositionMixin
from ResourceManager import ResourceManager
from CollageBuilder import CollageBuilder, CollageBuilderWindow
from Theme import apply_theme, COLORS
from ImageViewer import ImageViewerWindow
from DSODetail import DSODetailWindow
from FOVSimulator import AladinLiteWindow
from NINAIntegration import NINAIntegration
from SystemTrayManager import SystemTrayManager

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


# --- Initial Startup Data Loader Thread ---
class InitialDataLoadWorker(QThread):
    """Worker thread for loading initial DSO data on startup without blocking UI"""
    data_loaded = Signal(list, list, int)  # dso_data, catalogs, total_count
    load_failed = Signal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """Load initial data batch in background thread"""
        try:
            import sqlite3
            from ResourceManager import ResourceManager

            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            from ResourceManager import attach_update_catalogs
            attach_update_catalogs(conn)
            cursor = conn.cursor()

            # Get list of available catalogs
            cursor.execute("""
                SELECT DISTINCT catalogue
                FROM cataloguenr
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

            logger.debug(f"Loaded initial batch: {len(dso_data)} of {total_count} DSOs in background thread")

            conn.close()

            # Emit the loaded data
            self.data_loaded.emit(dso_data, catalogs, total_count)

        except Exception as e:
            logger.error(f"Error loading initial data in background: {e}", exc_info=True)
            self.load_failed.emit(str(e))


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
            from ResourceManager import attach_update_catalogs
            attach_update_catalogs(conn)
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


# --- Parallel Loading Manager ---
class ParallelDataLoadManager(QObject):
    """Manages multiple DataLoadWorker threads for parallel data loading"""
    all_data_loaded = Signal(list)  # Signal with all combined data
    progress_updated = Signal(int, int)  # loaded count, total count

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workers = []
        self.results = {}  # Dictionary to store results by offset
        self.expected_batches = 0
        self.completed_batches = 0
        self.total_records = 0

    def load_batches_parallel(self, start_offset, total_to_load, batch_size, max_threads, catalog_filter=None, type_filter=None):
        """Load multiple batches in parallel using worker threads"""
        # Calculate how many batches we need
        num_batches = (total_to_load + batch_size - 1) // batch_size  # Ceiling division
        num_batches = min(num_batches, max_threads)  # Don't create more threads than needed

        self.expected_batches = num_batches
        self.completed_batches = 0
        self.results = {}
        self.workers = []
        self.total_records = 0

        logger.debug(f"Starting parallel load: {num_batches} batches, {max_threads} max threads, offset={start_offset}, total_to_load={total_to_load}")

        # Create and start worker threads for each batch
        for i in range(num_batches):
            offset = start_offset + (i * batch_size)
            # Last batch might be smaller
            limit = min(batch_size, total_to_load - (i * batch_size))

            if limit <= 0:
                break

            worker = DataLoadWorker(offset, limit, catalog_filter, type_filter)
            worker.data_loaded.connect(lambda data, offset=offset: self._on_batch_loaded(data, offset))
            self.workers.append(worker)
            worker.start()

    def _on_batch_loaded(self, data, offset):
        """Handle a batch being loaded"""
        self.results[offset] = data
        self.completed_batches += 1
        self.total_records += len(data)

        logger.debug(f"Batch loaded: offset={offset}, records={len(data)}, completed={self.completed_batches}/{self.expected_batches}")

        # Emit progress
        self.progress_updated.emit(self.total_records, self.expected_batches)

        # Check if all batches are complete
        if self.completed_batches >= self.expected_batches:
            self._combine_and_emit_results()

    def _combine_and_emit_results(self):
        """Combine all batch results in order and emit"""
        # Sort by offset to maintain correct order
        sorted_offsets = sorted(self.results.keys())
        combined_data = []

        for offset in sorted_offsets:
            combined_data.extend(self.results[offset])

        logger.debug(f"All batches loaded: {len(combined_data)} total records from {self.expected_batches} batches")
        self.all_data_loaded.emit(combined_data)

        # Clean up workers
        for worker in self.workers:
            if worker.isRunning():
                worker.quit()
                worker.wait()
        self.workers = []
        self.results = {}


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

        # Parallel loading support
        self.parallel_loader = ParallelDataLoadManager(self)
        self.parallel_loader.all_data_loaded.connect(self._on_parallel_data_loaded)
        self.max_threads = self._get_max_threads()
        logger.debug(f"DSOTableModel initialized with max_threads={self.max_threads}")

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
        # Check if we have a matched designation from search
        matched_designation = entry.get("matched_designation")

        if col == 0:
            # Show catalog from matched designation if available
            if matched_designation:
                parts = matched_designation.split(" ", 1)
                return parts[0] if parts else entry["catalogue"]
            elif self.selected_catalog and self.selected_catalog != "All Catalogs":
                return self.selected_catalog
            return entry["catalogue"]
        elif col == 1:
            # Show designation from matched designation if available
            if matched_designation:
                parts = matched_designation.split(" ", 1)
                return parts[1] if len(parts) > 1 else matched_designation

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

    def filter_data(self, search_text, selected_catalog=None, show_images_only=False, selected_type=None, show_no_images_only=False):
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
        self._current_show_no_images_only = show_no_images_only
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

        if not search_text and not selected_catalog and not show_images_only and not selected_type and not show_no_images_only:
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

                if show_no_images_only and item["image_count"] > 0:
                    continue

                # Apply search text filter
                if search_text:
                    matched_designation = None

                    # If we have a catalog filter, prioritize exact catalog+designation matches
                    if selected_catalog and selected_catalog != "All Catalogs":
                        # Check for exact match: catalog filter + search text = designation
                        designations = item["designations"].split(", ")
                        for designation in designations:
                            if designation.lower() == f"{selected_catalog.lower()} {search_text}":
                                matched_designation = designation
                                break

                        # Also check if the item's ID matches the search
                        id_match = search_text in item["id"].lower()

                        if matched_designation or id_match:
                            item_copy = item.copy()
                            if matched_designation:
                                item_copy["matched_designation"] = matched_designation
                            matches.append((item_copy, 0))  # Priority 0 = exact match
                            continue

                    # Otherwise do regular substring matching and find which designation matched
                    designations = item["designations"].split(", ")

                    # Check each designation for a match
                    for designation in designations:
                        if search_text in designation.lower():
                            matched_designation = designation
                            break

                    # Check other fields
                    if (search_text in item["catalogue"].lower() or
                        search_text in item["id"].lower() or
                        self._format_ra(item["ra_deg"]).lower() in search_text or
                        self._format_dec(item["dec_deg"]).lower() in search_text or
                        matched_designation):

                        item_copy = item.copy()
                        if matched_designation:
                            item_copy["matched_designation"] = matched_designation
                        matches.append((item_copy, 1))  # Priority 1 = substring match
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
            from ResourceManager import ResourceManager, attach_update_catalogs

            # Query the total count for this specific filter
            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            attach_update_catalogs(conn)
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
                             getattr(self, '_current_show_no_images_only', False) or
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

    def _get_max_threads(self):
        """Get max_threads setting from QSettings"""
        try:
            settings = QSettings("AstroAssist", "CosmosCollection")
            default_threads = max(1, (os.cpu_count() or 4) - 2)
            max_threads = settings.value("max_threads", default_threads, type=int)
            return max(1, min(max_threads, 128))  # Ensure reasonable bounds
        except Exception as e:
            logger.error(f"Error reading max_threads setting: {e}")
            return max(1, (os.cpu_count() or 4) - 2)

    def load_more_data(self):
        """Load the next batches of data in parallel background threads"""
        if self.loading or self.load_offset >= self.total_count:
            logger.debug(f"Load blocked: loading={self.loading}, offset={self.load_offset}, total={self.total_count}")
            return

        # Get current filters
        catalog_filter = self.selected_catalog if self.selected_catalog else None
        type_filter = getattr(self, '_current_selected_type', None)

        # Calculate how much data remains to load
        remaining = self.total_count - self.load_offset

        # Load up to max_threads * batch_size in this batch (parallel loading)
        total_to_load = min(remaining, self.max_threads * self.load_batch_size)

        logger.debug(f"Starting parallel load from offset {self.load_offset}, loading {total_to_load} records using {self.max_threads} threads, catalog={catalog_filter}, type={type_filter}")
        self.loading = True

        # Emit signal to update UI loading state
        if hasattr(self.parent(), '_on_loading_started'):
            self.parent()._on_loading_started()

        # Use parallel loader
        self.parallel_loader.load_batches_parallel(
            self.load_offset,
            total_to_load,
            self.load_batch_size,
            self.max_threads,
            catalog_filter,
            type_filter
        )

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
            show_no_images_only = getattr(self, '_current_show_no_images_only', False)

            # Notify view that data is about to change
            self.layoutAboutToBeChanged.emit()

            # Rebuild filtered data from all loaded data
            if search_text or show_images_only or show_no_images_only:
                filtered_items = []
                for item in self.dso_data:
                    # Apply show_images_only filter
                    if show_images_only and item["image_count"] == 0:
                        continue

                    if show_no_images_only and item["image_count"] > 0:
                        continue

                    # Apply search text filter
                    if search_text:
                        matched_designation = None

                        # Check each designation for a match
                        designations = item["designations"].split(", ")
                        for designation in designations:
                            if search_text in designation.lower():
                                matched_designation = designation
                                break

                        # Check if any field matches
                        if (search_text in item["catalogue"].lower() or
                            search_text in item["id"].lower() or
                            matched_designation):

                            item_copy = item.copy()
                            if matched_designation:
                                item_copy["matched_designation"] = matched_designation
                            filtered_items.append(item_copy)
                    else:
                        filtered_items.append(item)

                self.filtered_data = filtered_items
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

    def _on_parallel_data_loaded(self, new_data):
        """Handle parallel data batches loaded from background threads"""
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
            show_no_images_only = getattr(self, '_current_show_no_images_only', False)

            # Notify view that data is about to change
            self.layoutAboutToBeChanged.emit()

            # Rebuild filtered data from all loaded data
            if search_text or show_images_only or show_no_images_only:
                filtered_items = []
                for item in self.dso_data:
                    # Apply show_images_only filter
                    if show_images_only and item["image_count"] == 0:
                        continue

                    if show_no_images_only and item["image_count"] > 0:
                        continue

                    # Apply search text filter
                    if search_text:
                        matched_designation = None

                        # Check each designation for a match
                        designations = item["designations"].split(", ")
                        for designation in designations:
                            if search_text in designation.lower():
                                matched_designation = designation
                                break

                        # Check if any field matches
                        if (search_text in item["catalogue"].lower() or
                            search_text in item["id"].lower() or
                            matched_designation):

                            item_copy = item.copy()
                            if matched_designation:
                                item_copy["matched_designation"] = matched_designation
                            filtered_items.append(item_copy)
                    else:
                        filtered_items.append(item)

                self.filtered_data = filtered_items
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
            logger.debug(f"Added {len(new_data)} DSOs from parallel loading, total now: {len(self.dso_data)}")

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
            (hasattr(self, '_current_show_images_only') and getattr(self, '_current_show_images_only', False) or
             hasattr(self, '_current_show_no_images_only') and getattr(self, '_current_show_no_images_only', False))):

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
        # Continue loading all data in background until complete
        elif self.load_offset < self.total_count:
            logger.debug(f"Background loading: {self.load_offset}/{self.total_count} objects loaded, continuing...")
            from PySide6.QtCore import QTimer
            # Use longer delay (200ms) for background loading to not impact UI performance
            QTimer.singleShot(200, self.load_more_data)

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
        self.resize(700, 450)
        
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
        location_layout = QHBoxLayout(location_tab)

        # Track which location is being edited
        self.editing_location_id = None

        # --- Left Panel: Saved Locations List ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        saved_label = QLabel("Saved Locations:")
        saved_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(saved_label)

        self.location_list = QListWidget()
        self.location_list.setMinimumWidth(220)
        self.location_list.currentItemChanged.connect(self._on_location_selected)
        left_layout.addWidget(self.location_list)

        left_btn_layout = QHBoxLayout()
        self.delete_location_btn = QPushButton("Delete")
        self.delete_location_btn.setToolTip("Delete the selected location")
        self.delete_location_btn.clicked.connect(self._delete_location)
        self.set_active_btn = QPushButton("Set as Active")
        self.set_active_btn.setToolTip("Set the selected location as the active observer location")
        self.set_active_btn.clicked.connect(self._set_active_location)
        left_btn_layout.addWidget(self.delete_location_btn)
        left_btn_layout.addWidget(self.set_active_btn)
        left_layout.addLayout(left_btn_layout)

        location_layout.addWidget(left_panel)

        # --- Right Panel: Location Details Form ---
        right_panel = QGroupBox("Location Details")
        right_layout = QVBoxLayout(right_panel)

        # Latitude input
        lat_layout = QHBoxLayout()
        lat_label = QLabel("Latitude:")
        lat_label.setMinimumWidth(100)
        self.latitude_input = QLineEdit()
        self.latitude_input.setPlaceholderText("e.g., 40.7128")
        lat_layout.addWidget(lat_label)
        lat_layout.addWidget(self.latitude_input)
        right_layout.addLayout(lat_layout)

        # Longitude input
        lon_layout = QHBoxLayout()
        lon_label = QLabel("Longitude:")
        lon_label.setMinimumWidth(100)
        self.longitude_input = QLineEdit()
        self.longitude_input.setPlaceholderText("e.g., -74.0060")
        lon_layout.addWidget(lon_label)
        lon_layout.addWidget(self.longitude_input)
        right_layout.addLayout(lon_layout)

        # Location name
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setMinimumWidth(100)
        self.location_name_input = QLineEdit()
        self.location_name_input.setPlaceholderText("e.g., Home Observatory (optional)")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.location_name_input)
        right_layout.addLayout(name_layout)

        # Timezone
        tz_layout = QHBoxLayout()
        tz_label = QLabel("Time Zone:")
        tz_label.setMinimumWidth(100)
        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
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
        right_layout.addLayout(tz_layout)

        # Map picker button
        map_button_layout = QHBoxLayout()
        map_button = QPushButton("Select Location from Map")
        map_button.setToolTip("Open an interactive map to visually select your location")
        map_button.clicked.connect(self._open_map_picker)
        map_button_layout.addStretch()
        map_button_layout.addWidget(map_button)
        map_button_layout.addStretch()
        right_layout.addLayout(map_button_layout)

        # Save / Clear buttons
        form_btn_layout = QHBoxLayout()
        self.save_location_btn = QPushButton("Save Location")
        self.save_location_btn.clicked.connect(self._save_location)
        self.clear_form_btn = QPushButton("Clear Form")
        self.clear_form_btn.clicked.connect(self._clear_location_form)
        form_btn_layout.addStretch()
        form_btn_layout.addWidget(self.save_location_btn)
        form_btn_layout.addWidget(self.clear_form_btn)
        right_layout.addLayout(form_btn_layout)

        right_layout.addStretch()
        location_layout.addWidget(right_panel)

        # Application Settings tab
        app_settings_scroll = QScrollArea()
        app_settings_scroll.setWidgetResizable(True)
        app_settings_scroll.setFrameShape(QScrollArea.NoFrame)
        app_settings_tab = QWidget()
        app_settings_layout = QVBoxLayout(app_settings_tab)
        app_settings_scroll.setWidget(app_settings_tab)

        # UI Preferences group
        ui_prefs_group = QGroupBox("User Interface Preferences")
        ui_prefs_layout = QVBoxLayout(ui_prefs_group)

        # Show Observer Location checkbox
        self.show_observer_location_checkbox = QCheckBox("Show Observer Location")
        self.show_observer_location_checkbox.setToolTip(
            "When enabled, displays your observer location information in various windows\n"
            "(Best DSO Tonight, DSO Visibility Calculator, and Main Window)"
        )
        ui_prefs_layout.addWidget(self.show_observer_location_checkbox)

        # Check for updates on startup checkbox
        self.check_updates_checkbox = QCheckBox("Check for updates on startup")
        self.check_updates_checkbox.setToolTip(
            "When enabled, automatically checks for application updates when the program starts"
        )
        ui_prefs_layout.addWidget(self.check_updates_checkbox)

        # Minimize to system tray checkbox
        self.minimize_to_tray_checkbox = QCheckBox("Minimize to system tray")
        self.minimize_to_tray_checkbox.setToolTip(
            "When enabled, closing or minimizing hides the window to the system tray.\n"
            "Double-click the tray icon to restore. Right-click for quick actions."
        )
        ui_prefs_layout.addWidget(self.minimize_to_tray_checkbox)

        # Enable log file checkbox + open folder button
        log_file_layout = QHBoxLayout()
        self.enable_logfile_checkbox = QCheckBox("Enable log file")
        self.enable_logfile_checkbox.setToolTip(
            "When enabled, saves application logs to CosmosCollection.log\n"
            "in the user data directory. Useful for troubleshooting.\n"
            "Requires application restart to take effect."
        )
        log_file_layout.addWidget(self.enable_logfile_checkbox)
        open_log_folder_btn = QPushButton("Open Log Folder")
        open_log_folder_btn.setToolTip("Open the folder where log files are stored")
        open_log_folder_btn.setFixedWidth(120)
        open_log_folder_btn.clicked.connect(self._open_log_folder)
        log_file_layout.addWidget(open_log_folder_btn)
        log_file_layout.addStretch()
        ui_prefs_layout.addLayout(log_file_layout)

        # Time format setting
        time_format_layout = QHBoxLayout()
        time_format_label = QLabel("Time Format:")
        time_format_label.setMinimumWidth(120)
        self.time_format_combo = QComboBox()
        self.time_format_combo.addItems(["12-hour", "24-hour"])
        self.time_format_combo.setToolTip(
            "12-hour: displays times like 2:30 PM\n"
            "24-hour: displays times like 14:30"
        )
        time_format_layout.addWidget(time_format_label)
        time_format_layout.addWidget(self.time_format_combo)
        time_format_layout.addStretch()
        ui_prefs_layout.addLayout(time_format_layout)

        app_settings_layout.addWidget(ui_prefs_group)

        # Weather Units group
        weather_units_group = QGroupBox("Weather Units")
        weather_units_layout = QVBoxLayout(weather_units_group)

        # Temperature unit setting
        temp_unit_layout = QHBoxLayout()
        temp_unit_label = QLabel("Temperature:")
        temp_unit_label.setMinimumWidth(120)
        self.temp_unit_combo = QComboBox()
        self.temp_unit_combo.addItems(["Celsius", "Fahrenheit"])
        self.temp_unit_combo.setToolTip("Temperature display unit for weather forecast")
        temp_unit_layout.addWidget(temp_unit_label)
        temp_unit_layout.addWidget(self.temp_unit_combo)
        temp_unit_layout.addStretch()
        weather_units_layout.addLayout(temp_unit_layout)

        # Wind speed unit setting
        wind_unit_layout = QHBoxLayout()
        wind_unit_label = QLabel("Wind Speed:")
        wind_unit_label.setMinimumWidth(120)
        self.wind_unit_combo = QComboBox()
        self.wind_unit_combo.addItems(["km/h", "mph", "m/s"])
        self.wind_unit_combo.setToolTip("Wind speed display unit for weather forecast")
        wind_unit_layout.addWidget(wind_unit_label)
        wind_unit_layout.addWidget(self.wind_unit_combo)
        wind_unit_layout.addStretch()
        weather_units_layout.addLayout(wind_unit_layout)

        # Precipitation/visibility unit setting
        precip_unit_layout = QHBoxLayout()
        precip_unit_label = QLabel("Precipitation/Visibility:")
        precip_unit_label.setMinimumWidth(120)
        self.precip_unit_combo = QComboBox()
        self.precip_unit_combo.addItems(["Metric (mm, km)", "Imperial (in, mi)"])
        self.precip_unit_combo.setToolTip("Units for precipitation and visibility in weather forecast")
        precip_unit_layout.addWidget(precip_unit_label)
        precip_unit_layout.addWidget(self.precip_unit_combo)
        precip_unit_layout.addStretch()
        weather_units_layout.addLayout(precip_unit_layout)

        app_settings_layout.addWidget(weather_units_group)

        # Performance Settings group
        perf_settings_group = QGroupBox("Performance Settings")
        perf_settings_layout = QVBoxLayout(perf_settings_group)

        # Thread count setting
        thread_layout = QHBoxLayout()
        thread_label = QLabel("Maximum Threads:")
        thread_label.setMinimumWidth(120)
        self.thread_count_spinbox = QSpinBox()
        self.thread_count_spinbox.setMinimum(1)
        self.thread_count_spinbox.setMaximum(128)
        import os
        default_threads = max(1, (os.cpu_count() or 4) - 2)
        self.thread_count_spinbox.setValue(default_threads)
        self.thread_count_spinbox.setToolTip(
            f"Number of threads to use for parallel operations.\n"
            f"Default: {default_threads} (CPU cores - 2)\n"
            f"Higher values may improve performance but increase CPU usage."
        )
        thread_layout.addWidget(thread_label)
        thread_layout.addWidget(self.thread_count_spinbox)
        thread_layout.addStretch()
        perf_settings_layout.addLayout(thread_layout)

        # Thumbnail disk cache setting
        self.cache_thumbnails_checkbox = QCheckBox("Cache thumbnails to disk")
        self.cache_thumbnails_checkbox.setToolTip(
            "When enabled, generated thumbnails are saved alongside original images.\n"
            "Speeds up gallery loading on subsequent launches."
        )
        perf_settings_layout.addWidget(self.cache_thumbnails_checkbox)

        # Thumbnail cache help text
        cache_help = QLabel(
            "Thumbnails will be saved as <i>{filename}_{size}_Thumbnail.jpg</i> in the same folder as the original image.<br>"
            "This speeds up gallery loading on subsequent launches. Cache files can be safely deleted."
        )
        cache_help.setWordWrap(True)
        cache_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        cache_help.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; margin-left: 20px; }}")
        perf_settings_layout.addWidget(cache_help)

        app_settings_layout.addWidget(perf_settings_group)

        # Plate Solving Settings group
        plate_solve_group = QGroupBox("Plate Solving Settings")
        plate_solve_layout = QVBoxLayout(plate_solve_group)

        # ASTAP path setting
        astap_layout = QHBoxLayout()
        astap_label = QLabel("ASTAP Path:")
        astap_label.setMinimumWidth(120)
        self.astap_path_input = QLineEdit()
        self.astap_path_input.setPlaceholderText("Path to astap_cli executable (auto-detected if empty)")
        astap_browse_btn = QPushButton("Browse...")
        astap_browse_btn.setFixedWidth(80)
        astap_browse_btn.clicked.connect(self._browse_astap_path)
        astap_layout.addWidget(astap_label)
        astap_layout.addWidget(self.astap_path_input)
        astap_layout.addWidget(astap_browse_btn)
        plate_solve_layout.addLayout(astap_layout)

        # ASTAP help text
        astap_help = QLabel(
            "ASTAP is a free, fast local plate solver. Download from: "
            "<a href='https://www.hnsky.org/astap.htm' style='color: #0078d7;'>hnsky.org/astap.htm</a><br>"
            "Point to <b>astap_cli.exe</b> (command-line version), not astap.exe (GUI). "
            "Leave empty to auto-detect."
        )
        astap_help.setOpenExternalLinks(True)
        astap_help.setWordWrap(True)
        astap_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        astap_help.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; margin-left: 120px; }}")
        plate_solve_layout.addWidget(astap_help)

        # Astrometry.net API key setting
        api_key_layout = QHBoxLayout()
        api_key_label = QLabel("Astrometry.net API Key:")
        api_key_label.setMinimumWidth(120)
        self.astrometry_api_key_input = QLineEdit()
        self.astrometry_api_key_input.setPlaceholderText("Required for online plate solving")
        self.astrometry_api_key_input.setEchoMode(QLineEdit.Password)
        show_key_btn = QPushButton("Show")
        show_key_btn.setFixedWidth(50)
        show_key_btn.setCheckable(True)
        show_key_btn.clicked.connect(lambda checked: self.astrometry_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.astrometry_api_key_input)
        api_key_layout.addWidget(show_key_btn)
        plate_solve_layout.addLayout(api_key_layout)

        # API key help text
        api_key_help = QLabel(
            "Get a free API key: Register at "
            "<a href='https://nova.astrometry.net/' style='color: #0078d7;'>nova.astrometry.net</a>, "
            "then find your key in My Account > API.<br>"
            "An API key is required for online plate solving (used when ASTAP fails or is unavailable)."
        )
        api_key_help.setOpenExternalLinks(True)
        api_key_help.setWordWrap(True)
        api_key_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        api_key_help.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; margin-left: 120px; }}")
        plate_solve_layout.addWidget(api_key_help)

        app_settings_layout.addWidget(plate_solve_group)

        # NINA Integration group
        nina_group = QGroupBox("NINA Integration")
        nina_layout = QVBoxLayout(nina_group)

        # Enable checkbox
        self.nina_enabled_checkbox = QCheckBox("Enable NINA Integration")
        self.nina_enabled_checkbox.setToolTip(
            "When enabled, 'Send to NINA Framing Assistant' options will appear\n"
            "in context menus throughout the application."
        )
        nina_layout.addWidget(self.nina_enabled_checkbox)

        nina_ip_layout = QHBoxLayout()
        nina_ip_label = QLabel("API Host:")
        nina_ip_label.setMinimumWidth(120)
        self.nina_ip_input = QLineEdit()
        self.nina_ip_input.setText("localhost")
        self.nina_ip_input.setPlaceholderText("localhost or IP address")
        self.nina_ip_input.setToolTip(
            "IP address or hostname of the machine running NINA.\n"
            "Use 'localhost' if NINA is running on this machine."
        )
        nina_ip_layout.addWidget(nina_ip_label)
        nina_ip_layout.addWidget(self.nina_ip_input)
        nina_layout.addLayout(nina_ip_layout)

        nina_port_layout = QHBoxLayout()
        nina_port_label = QLabel("API Port:")
        nina_port_label.setMinimumWidth(120)
        self.nina_port_spinbox = QSpinBox()
        self.nina_port_spinbox.setMinimum(1)
        self.nina_port_spinbox.setMaximum(65535)
        self.nina_port_spinbox.setValue(1888)
        self.nina_port_spinbox.setToolTip(
            "Port number for NINA's Advanced API.\n"
            "Default: 1888"
        )
        nina_port_layout.addWidget(nina_port_label)
        nina_port_layout.addWidget(self.nina_port_spinbox)
        nina_port_layout.addStretch()
        nina_layout.addLayout(nina_port_layout)

        # Test connection button
        nina_test_layout = QHBoxLayout()
        nina_test_spacer = QLabel("")
        nina_test_spacer.setMinimumWidth(120)
        self.nina_test_btn = QPushButton("Test Connection")
        self.nina_test_btn.setToolTip("Test the connection to NINA's Advanced API")
        self.nina_test_btn.clicked.connect(self._test_nina_connection)
        nina_test_layout.addWidget(nina_test_spacer)
        nina_test_layout.addWidget(self.nina_test_btn)
        nina_test_layout.addStretch()
        nina_layout.addLayout(nina_test_layout)

        nina_help = QLabel(
            "Configure the port used by NINA's Advanced API plugin.<br>"
            "Requires the <b>Advanced API</b> plugin to be installed and enabled in NINA."
        )
        nina_help.setWordWrap(True)
        nina_help.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; }}")
        nina_layout.addWidget(nina_help)

        app_settings_layout.addWidget(nina_group)
        app_settings_layout.addStretch()

        # Backup/Restore tab
        backup_restore_tab = QWidget()
        backup_restore_layout = QVBoxLayout(backup_restore_tab)

        # Backup group
        backup_group = QGroupBox("Backup")
        backup_group_layout = QVBoxLayout(backup_group)

        backup_description = QLabel(
            "Create a backup of all your user data including:\n"
            "• User image metadata\n"
            "• Target list\n"
            "• Telescope profiles\n"
            "• Equipment (cameras, eyepieces, barlows)\n"
            "• Collage projects\n"
            "• Application settings (location, timezone)"
        )
        backup_description.setWordWrap(True)
        backup_description.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        backup_group_layout.addWidget(backup_description)

        backup_btn_layout = QHBoxLayout()
        self.backup_button = QPushButton("Backup Now...")
        self.backup_button.setToolTip("Save all user data to a backup file")
        self.backup_button.clicked.connect(self._perform_backup)
        backup_btn_layout.addWidget(self.backup_button)
        backup_btn_layout.addStretch()
        backup_group_layout.addLayout(backup_btn_layout)

        backup_restore_layout.addWidget(backup_group)

        # Restore group
        restore_group = QGroupBox("Restore")
        restore_group_layout = QVBoxLayout(restore_group)

        restore_description = QLabel(
            "Restore your user data from a previously created backup file.\n\n"
            "<b>Warning:</b> Restoring will replace your current data with the backup data. "
            "It is recommended to create a backup of your current data first."
        )
        restore_description.setWordWrap(True)
        restore_description.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        restore_group_layout.addWidget(restore_description)

        restore_btn_layout = QHBoxLayout()
        self.restore_button = QPushButton("Restore from Backup...")
        self.restore_button.setToolTip("Restore user data from a backup file")
        self.restore_button.clicked.connect(self._perform_restore)
        restore_btn_layout.addWidget(self.restore_button)
        restore_btn_layout.addStretch()
        restore_group_layout.addLayout(restore_btn_layout)

        backup_restore_layout.addWidget(restore_group)
        backup_restore_layout.addStretch()

        # Add tabs
        tab_widget.addTab(app_settings_scroll, "Application Settings")
        tab_widget.addTab(location_tab, "Location Manager")
        tab_widget.addTab(backup_restore_tab, "Backup && Restore")

        layout.addWidget(tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Standard dialog buttons
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_settings)
        # Don't set as default to prevent accidental saves when map picker closes
        # self.save_button.setDefault(True)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def _load_current_settings(self):
        """Load current settings from database and QSettings"""
        # Populate location list
        self._refresh_location_list()

        # Load UI preferences from QSettings
        try:
            settings = QSettings("CosmosCollection", "CosmosCollection")
            show_observer_location = settings.value("show_observer_location", True, type=bool)
            self.show_observer_location_checkbox.setChecked(show_observer_location)

            check_updates = settings.value("check_updates_on_startup", True, type=bool)
            self.check_updates_checkbox.setChecked(check_updates)

            minimize_to_tray = settings.value("minimize_to_tray", True, type=bool)
            self.minimize_to_tray_checkbox.setChecked(minimize_to_tray)

            enable_logfile = settings.value("enable_logfile", False, type=bool)
            self.enable_logfile_checkbox.setChecked(enable_logfile)

            # Load time format setting
            time_format = settings.value("time_format", "12-hour", type=str)
            index = self.time_format_combo.findText(time_format)
            if index >= 0:
                self.time_format_combo.setCurrentIndex(index)

            # Load thread count setting
            import os
            default_threads = max(1, (os.cpu_count() or 4) - 2)
            thread_count = settings.value("max_threads", default_threads, type=int)
            self.thread_count_spinbox.setValue(thread_count)

            # Load thumbnail cache setting
            cache_thumbnails = settings.value("cache_thumbnails_to_disk", True, type=bool)
            self.cache_thumbnails_checkbox.setChecked(cache_thumbnails)

            # Load weather unit settings
            temp_unit = settings.value("temperature_unit", "Celsius", type=str)
            index = self.temp_unit_combo.findText(temp_unit)
            if index >= 0:
                self.temp_unit_combo.setCurrentIndex(index)

            wind_unit = settings.value("wind_speed_unit", "km/h", type=str)
            index = self.wind_unit_combo.findText(wind_unit)
            if index >= 0:
                self.wind_unit_combo.setCurrentIndex(index)

            precip_unit = settings.value("precip_visibility_unit", "Metric (mm, km)", type=str)
            index = self.precip_unit_combo.findText(precip_unit)
            if index >= 0:
                self.precip_unit_combo.setCurrentIndex(index)

            # Load plate solving settings
            astap_path = settings.value("astap_path", "", type=str)
            self.astap_path_input.setText(astap_path)

            astrometry_api_key = settings.value("astrometry_api_key", "", type=str)
            self.astrometry_api_key_input.setText(astrometry_api_key)

            # Load NINA settings
            nina_enabled = settings.value("nina_integration_enabled", False, type=bool)
            self.nina_enabled_checkbox.setChecked(nina_enabled)
            nina_ip = settings.value("nina_api_host", "localhost", type=str)
            self.nina_ip_input.setText(nina_ip)
            nina_port = settings.value("nina_api_port", 1888, type=int)
            self.nina_port_spinbox.setValue(nina_port)

        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            
    def _refresh_location_list(self):
        """Refresh the location list widget from database"""
        self.location_list.clear()
        active_item = None
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, location_lat, location_lon, location_name, timezone, is_active FROM usersettings ORDER BY id")
                rows = cursor.fetchall()
                for row in rows:
                    loc_id, lat, lon, name, tz, is_active = row
                    if name:
                        display = name
                    else:
                        display = f"{lat:.4f}, {lon:.4f}"
                    if is_active:
                        display = "(Active) " + display
                    if tz:
                        display += f" [{tz}]"
                    item = QListWidgetItem(display)
                    item.setData(Qt.UserRole, loc_id)
                    if is_active:
                        item.setBackground(QColor(COLORS.get('accent', '#0078d7')).darker(300))
                        active_item = item
                    self.location_list.addItem(item)
            if active_item:
                self.location_list.setCurrentItem(active_item)
        except Exception as e:
            logger.error(f"Error loading location list: {str(e)}")

    def _on_location_selected(self, current, previous):
        """Handle location list selection - populate form for editing"""
        if not current:
            return
        loc_id = current.data(Qt.UserRole)
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon, location_name, timezone, is_active FROM usersettings WHERE id = ?", (loc_id,))
                row = cursor.fetchone()
                if row:
                    lat, lon, name, tz, is_active = row
                    self.set_active_btn.setEnabled(not is_active)
                    self.latitude_input.setText(str(lat) if lat is not None else "")
                    self.longitude_input.setText(str(lon) if lon is not None else "")
                    self.location_name_input.setText(name if name else "")
                    if tz:
                        index = self.timezone_combo.findText(tz)
                        if index >= 0:
                            self.timezone_combo.setCurrentIndex(index)
                        else:
                            self.timezone_combo.setEditText(tz)
                    else:
                        self.timezone_combo.setCurrentIndex(0)
                    self.editing_location_id = loc_id
                    self.save_location_btn.setText("Update Location")
        except Exception as e:
            logger.error(f"Error loading location details: {str(e)}")

    def _clear_location_form(self):
        """Clear the location form and reset to add mode"""
        self.latitude_input.clear()
        self.longitude_input.clear()
        self.location_name_input.clear()
        self.timezone_combo.setCurrentIndex(0)
        self.editing_location_id = None
        self.save_location_btn.setText("Save Location")
        self.location_list.clearSelection()

    def _save_location(self):
        """Save or update a location in the database"""
        try:
            lat_text = self.latitude_input.text().strip()
            lon_text = self.longitude_input.text().strip()

            if not lat_text or not lon_text:
                QMessageBox.warning(self, "Missing Information",
                    "Please enter both latitude and longitude.")
                return

            lat = float(lat_text)
            lon = float(lon_text)

            if not (-90 <= lat <= 90):
                QMessageBox.warning(self, "Invalid Latitude",
                    "Latitude must be between -90 and 90 degrees.")
                return

            if not (-180 <= lon <= 180):
                QMessageBox.warning(self, "Invalid Longitude",
                    "Longitude must be between -180 and 180 degrees.")
                return

            location_name = self.location_name_input.text().strip() or None
            timezone = self.timezone_combo.currentText().strip() or "America/New_York"

            try:
                import pytz
                pytz.timezone(timezone)
            except Exception:
                QMessageBox.warning(self, "Invalid Timezone",
                    f"'{timezone}' is not a valid timezone identifier.")
                return

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if self.editing_location_id:
                    # Update existing location
                    cursor.execute("""
                        UPDATE usersettings SET location_lat = ?, location_lon = ?, location_name = ?, timezone = ?
                        WHERE id = ?
                    """, (lat, lon, location_name, timezone, self.editing_location_id))
                else:
                    # Insert new location
                    cursor.execute("""
                        INSERT INTO usersettings (location_lat, location_lon, location_name, timezone, is_active)
                        VALUES (?, ?, ?, ?, 0)
                    """, (lat, lon, location_name, timezone))

                    # If it's the only location, auto-set as active
                    cursor.execute("SELECT COUNT(*) FROM usersettings")
                    count = cursor.fetchone()[0]
                    if count == 1:
                        cursor.execute("UPDATE usersettings SET is_active = 1 WHERE id = (SELECT id FROM usersettings LIMIT 1)")

                conn.commit()

            self._clear_location_form()
            self._refresh_location_list()
            logger.debug(f"Location saved: lat={lat}, lon={lon}, name={location_name}, tz={timezone}")

        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                "Please enter valid numeric values for latitude and longitude.")
        except Exception as e:
            logger.error(f"Error saving location: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save location: {str(e)}")

    def _delete_location(self):
        """Delete the selected location"""
        current = self.location_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a location to delete.")
            return

        loc_id = current.data(Qt.UserRole)
        reply = QMessageBox.question(self, "Confirm Delete",
            "Are you sure you want to delete this location?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Check if we're deleting the active location
                cursor.execute("SELECT is_active FROM usersettings WHERE id = ?", (loc_id,))
                was_active = cursor.fetchone()[0]

                cursor.execute("DELETE FROM usersettings WHERE id = ?", (loc_id,))

                if was_active:
                    # Set the most recent remaining location as active
                    cursor.execute("""
                        UPDATE usersettings SET is_active = 1
                        WHERE id = (SELECT id FROM usersettings ORDER BY id DESC LIMIT 1)
                    """)

                conn.commit()

            self._clear_location_form()
            self._refresh_location_list()
        except Exception as e:
            logger.error(f"Error deleting location: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to delete location: {str(e)}")

    def _set_active_location(self):
        """Set the selected location as the active one"""
        current = self.location_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a location to set as active.")
            return

        loc_id = current.data(Qt.UserRole)
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE usersettings SET is_active = 0 WHERE is_active = 1")
                cursor.execute("UPDATE usersettings SET is_active = 1 WHERE id = ?", (loc_id,))
                conn.commit()

            self._refresh_location_list()
        except Exception as e:
            logger.error(f"Error setting active location: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to set active location: {str(e)}")

    def _open_log_folder(self):
        """Open the folder where log files are stored"""
        from PySide6.QtGui import QDesktopServices
        log_dir = ResourceManager.get_data_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    def _save_settings(self):
        """Save application settings (QSettings only - locations are saved separately)"""
        try:
            settings = QSettings("CosmosCollection", "CosmosCollection")
            settings.setValue("show_observer_location", self.show_observer_location_checkbox.isChecked())
            settings.setValue("check_updates_on_startup", self.check_updates_checkbox.isChecked())
            settings.setValue("minimize_to_tray", self.minimize_to_tray_checkbox.isChecked())
            settings.setValue("enable_logfile", self.enable_logfile_checkbox.isChecked())
            settings.setValue("time_format", self.time_format_combo.currentText())
            settings.setValue("max_threads", self.thread_count_spinbox.value())
            settings.setValue("cache_thumbnails_to_disk", self.cache_thumbnails_checkbox.isChecked())

            # Save weather unit settings
            settings.setValue("temperature_unit", self.temp_unit_combo.currentText())
            settings.setValue("wind_speed_unit", self.wind_unit_combo.currentText())
            settings.setValue("precip_visibility_unit", self.precip_unit_combo.currentText())

            # Save plate solving settings
            settings.setValue("astap_path", self.astap_path_input.text().strip())
            settings.setValue("astrometry_api_key", self.astrometry_api_key_input.text().strip())

            # Save NINA settings
            settings.setValue("nina_integration_enabled", self.nina_enabled_checkbox.isChecked())
            settings.setValue("nina_api_host", self.nina_ip_input.text().strip() or "localhost")
            settings.setValue("nina_api_port", self.nina_port_spinbox.value())

            self.accept()

        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def _open_map_picker(self):
        """Open map dialog to select location coordinates"""
        try:
            # Get current coordinates as starting point (or default to NYC)
            current_lat = 40.7128
            current_lon = -74.0060

            try:
                lat_text = self.latitude_input.text().strip()
                lon_text = self.longitude_input.text().strip()
                if lat_text and lon_text:
                    current_lat = float(lat_text)
                    current_lon = float(lon_text)
            except ValueError:
                # Use default coordinates if current values are invalid
                pass

            # Open dialog - parent=None to avoid event propagation issues between nested modal dialogs
            dialog = MapLocationPickerDialog(current_lat, current_lon, parent=None)
            result_code = dialog.exec()

            if result_code == QDialog.Accepted:
                result = dialog.get_selected_coordinates()
                if result:
                    lat, lon, location_name = result
                    self.latitude_input.setText(f"{lat:.6f}")
                    self.longitude_input.setText(f"{lon:.6f}")
                    if location_name:
                        self.location_name_input.setText(location_name)

        except Exception as e:
            logger.error(f"Error opening map picker: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Error",
                f"Failed to open map picker: {str(e)}\n\n"
                "Please enter coordinates manually.")

    def _browse_astap_path(self):
        """Browse for ASTAP executable"""
        import sys
        import os
        if sys.platform == 'win32':
            filter_str = "ASTAP CLI (astap_cli.exe);;Executable Files (*.exe);;All Files (*.*)"
            # Start in ASTAP folder if it exists
            if os.path.isdir("C:/Program Files/astap"):
                start_dir = "C:/Program Files/astap"
            elif os.path.isdir("C:/Program Files (x86)/astap"):
                start_dir = "C:/Program Files (x86)/astap"
            else:
                start_dir = "C:/Program Files"
        else:
            filter_str = "All Files (*)"
            start_dir = "/usr/bin"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASTAP CLI Executable (astap_cli.exe)",
            start_dir,
            filter_str
        )
        if file_path:
            self.astap_path_input.setText(file_path)

    def _test_nina_connection(self):
        """Test the connection to NINA's Advanced API"""
        nina_host = self.nina_ip_input.text().strip() or "localhost"
        nina_port = self.nina_port_spinbox.value()

        try:
            self.nina_test_btn.setEnabled(False)
            self.nina_test_btn.setText("Testing...")
            QApplication.processEvents()

            success, message, version = NINAIntegration.test_connection(nina_host, nina_port)

            if success:
                QMessageBox.information(self, "Connection Successful", message)
            else:
                QMessageBox.warning(self, "Connection Failed", message)

        finally:
            self.nina_test_btn.setEnabled(True)
            self.nina_test_btn.setText("Test Connection")

    def _perform_backup(self):
        """Perform a backup of all user data to a JSON file"""
        import json
        from datetime import datetime

        # Get save file path from user
        default_filename = f"CosmosCollection_Backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Backup File",
            default_filename,
            "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Collect all user data from database
            backup_data = {
                'backup_version': 1,
                'backup_date': datetime.now().isoformat(),
                'tables': {}
            }

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Backup usersettings
                cursor.execute("SELECT * FROM usersettings")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                backup_data['tables']['usersettings'] = {
                    'columns': columns,
                    'rows': [list(row) for row in rows]
                }

                # Backup usertelescopes
                cursor.execute("SELECT * FROM usertelescopes")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                backup_data['tables']['usertelescopes'] = {
                    'columns': columns,
                    'rows': [list(row) for row in rows]
                }

                # Backup userequipment (cameras, eyepieces, barlows)
                try:
                    cursor.execute("SELECT * FROM userequipment")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    backup_data['tables']['userequipment'] = {
                        'columns': columns,
                        'rows': [list(row) for row in rows]
                    }
                except Exception:
                    pass  # Table might not exist

                # Backup telescopeequipment (links equipment to telescopes)
                try:
                    cursor.execute("SELECT * FROM telescopeequipment")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    backup_data['tables']['telescopeequipment'] = {
                        'columns': columns,
                        'rows': [list(row) for row in rows]
                    }
                except Exception:
                    pass  # Table might not exist

                # Backup userimages
                cursor.execute("SELECT * FROM userimages")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                backup_data['tables']['userimages'] = {
                    'columns': columns,
                    'rows': [list(row) for row in rows]
                }

                # Backup usertargetlist
                cursor.execute("SELECT * FROM usertargetlist")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                backup_data['tables']['usertargetlist'] = {
                    'columns': columns,
                    'rows': [list(row) for row in rows]
                }

                # Backup usercollages
                try:
                    cursor.execute("SELECT * FROM usercollages")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    backup_data['tables']['usercollages'] = {
                        'columns': columns,
                        'rows': [list(row) for row in rows]
                    }
                except Exception:
                    pass  # Table might not exist

                # Backup usercollageimages
                try:
                    cursor.execute("SELECT * FROM usercollageimages")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    backup_data['tables']['usercollageimages'] = {
                        'columns': columns,
                        'rows': [list(row) for row in rows]
                    }
                except Exception:
                    pass  # Table might not exist

            # Backup QSettings (only JSON-serializable values)
            settings = QSettings("CosmosCollection", "CosmosCollection")
            qsettings_data = {}
            for key in settings.allKeys():
                value = settings.value(key)
                # Only include basic JSON-serializable types
                if isinstance(value, (str, int, float, bool, type(None))):
                    qsettings_data[key] = value
            backup_data['qsettings'] = qsettings_data

            # Write the JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)

            QMessageBox.information(
                self,
                "Backup Complete",
                f"Backup created successfully!\n\nLocation: {file_path}"
            )
            logger.info(f"Backup completed successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Backup Error", f"Failed to create backup: {str(e)}")

    def _perform_restore(self):
        """Restore user data from a backup JSON file"""
        import json

        # Confirm with user
        confirm = QMessageBox.warning(
            self,
            "Confirm Restore",
            "Restoring from a backup will replace your current data.\n\n"
            "It is strongly recommended to create a backup of your current data first.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        # Get backup file from user
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backup File to Restore",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Validate backup file
            if 'backup_version' not in backup_data or 'tables' not in backup_data:
                QMessageBox.critical(self, "Invalid Backup", "This file does not appear to be a valid Cosmos Collection backup.")
                return

            # Verify backup version
            if backup_data.get('backup_version', 0) > 1:
                QMessageBox.warning(
                    self,
                    "Newer Backup Version",
                    "This backup was created with a newer version of Cosmos Collection.\n"
                    "Some data may not be restored correctly."
                )

            # Restore database tables
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Restore usersettings (clear and insert)
                if 'usersettings' in backup_data['tables']:
                    cursor.execute("DELETE FROM usersettings")
                    table_data = backup_data['tables']['usersettings']
                    columns = table_data['columns']
                    for row in table_data['rows']:
                        placeholders = ', '.join(['?' for _ in columns])
                        cursor.execute(f"INSERT INTO usersettings ({', '.join(columns)}) VALUES ({placeholders})", row)

                # Restore usertelescopes
                if 'usertelescopes' in backup_data['tables']:
                    cursor.execute("DELETE FROM usertelescopes")
                    table_data = backup_data['tables']['usertelescopes']
                    columns = table_data['columns']
                    for row in table_data['rows']:
                        placeholders = ', '.join(['?' for _ in columns])
                        cursor.execute(f"INSERT INTO usertelescopes ({', '.join(columns)}) VALUES ({placeholders})", row)

                # Restore userequipment (cameras, eyepieces, barlows)
                if 'userequipment' in backup_data['tables']:
                    try:
                        cursor.execute("DELETE FROM userequipment")
                        table_data = backup_data['tables']['userequipment']
                        columns = table_data['columns']
                        for row in table_data['rows']:
                            placeholders = ', '.join(['?' for _ in columns])
                            cursor.execute(f"INSERT INTO userequipment ({', '.join(columns)}) VALUES ({placeholders})", row)
                    except Exception:
                        pass  # Table might not exist

                # Restore telescopeequipment (links equipment to telescopes)
                if 'telescopeequipment' in backup_data['tables']:
                    try:
                        cursor.execute("DELETE FROM telescopeequipment")
                        table_data = backup_data['tables']['telescopeequipment']
                        columns = table_data['columns']
                        for row in table_data['rows']:
                            placeholders = ', '.join(['?' for _ in columns])
                            cursor.execute(f"INSERT INTO telescopeequipment ({', '.join(columns)}) VALUES ({placeholders})", row)
                    except Exception:
                        pass  # Table might not exist

                # Restore userimages
                if 'userimages' in backup_data['tables']:
                    cursor.execute("DELETE FROM userimages")
                    table_data = backup_data['tables']['userimages']
                    columns = table_data['columns']
                    for row in table_data['rows']:
                        placeholders = ', '.join(['?' for _ in columns])
                        cursor.execute(f"INSERT INTO userimages ({', '.join(columns)}) VALUES ({placeholders})", row)

                # Restore usertargetlist
                if 'usertargetlist' in backup_data['tables']:
                    cursor.execute("DELETE FROM usertargetlist")
                    table_data = backup_data['tables']['usertargetlist']
                    columns = table_data['columns']
                    for row in table_data['rows']:
                        placeholders = ', '.join(['?' for _ in columns])
                        cursor.execute(f"INSERT INTO usertargetlist ({', '.join(columns)}) VALUES ({placeholders})", row)

                # Restore usercollages
                if 'usercollages' in backup_data['tables']:
                    try:
                        cursor.execute("DELETE FROM usercollages")
                        table_data = backup_data['tables']['usercollages']
                        columns = table_data['columns']
                        for row in table_data['rows']:
                            placeholders = ', '.join(['?' for _ in columns])
                            cursor.execute(f"INSERT INTO usercollages ({', '.join(columns)}) VALUES ({placeholders})", row)
                    except Exception:
                        pass  # Table might not exist

                # Restore usercollageimages
                if 'usercollageimages' in backup_data['tables']:
                    try:
                        cursor.execute("DELETE FROM usercollageimages")
                        table_data = backup_data['tables']['usercollageimages']
                        columns = table_data['columns']
                        for row in table_data['rows']:
                            placeholders = ', '.join(['?' for _ in columns])
                            cursor.execute(f"INSERT INTO usercollageimages ({', '.join(columns)}) VALUES ({placeholders})", row)
                    except Exception:
                        pass  # Table might not exist

                conn.commit()

            # Restore QSettings if present
            if 'qsettings' in backup_data:
                settings = QSettings("CosmosCollection", "CosmosCollection")
                for key, value in backup_data['qsettings'].items():
                    settings.setValue(key, value)

            # Reload the settings dialog with restored data
            self._load_current_settings()

            QMessageBox.information(
                self,
                "Restore Complete",
                f"Backup restored successfully from:\n{file_path}\n\n"
                "Please restart the application for all changes to take effect."
            )
            logger.info(f"Backup restored successfully from {file_path}")

        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Restore Error", f"Failed to restore backup: {str(e)}")


# --- Map Location Picker Dialog ---
class MapBridge(QObject):
    """Bridge object for Qt-JavaScript communication in map picker"""

    searchCompleted = Signal(str)  # Signal to send search results back to JavaScript
    geocodeCompleted = Signal(str)  # Signal to send geocode results back to JavaScript

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    @Slot(float, float, str)
    def selectLocation(self, lat, lon, location_name):
        """Called from JavaScript when user selects a location on the map"""
        logger.debug(f"MapBridge.selectLocation called: lat={lat}, lon={lon}, name={location_name}")
        self.dialog._on_location_selected(lat, lon, location_name)

    @Slot(str)
    def searchLocationFromPython(self, query):
        """Search for location using Python (to avoid CORS issues)"""
        logger.debug(f"MapBridge.searchLocationFromPython called: query={query}")
        result = self.dialog._search_location_python(query)
        logger.debug(f"Emitting search result: {result}")
        self.searchCompleted.emit(result)

    @Slot(float, float)
    def reverseGeocodeFromPython(self, lat, lon):
        """Reverse geocode using Python (to avoid CORS issues)"""
        logger.debug(f"MapBridge.reverseGeocodeFromPython called: lat={lat}, lon={lon}")
        result = self.dialog._reverse_geocode_python(lat, lon)
        logger.debug(f"Emitting geocode result: {result}")
        self.geocodeCompleted.emit(result)


class MapLocationPickerDialog(QDialog):
    """Dialog for selecting location coordinates from an interactive map"""

    def __init__(self, initial_lat=40.7128, initial_lon=-74.0060, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Location from Map - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(900, 700)

        # Store initial coordinates
        self.initial_lat = initial_lat
        self.initial_lon = initial_lon

        # Selected coordinates (initialize with initial values so user can accept without clicking)
        self.selected_lat = initial_lat
        self.selected_lon = initial_lon
        self.selected_location_name = ""

        # Web view and bridge
        self.web_view = None
        self.bridge = None
        self.channel = None

        self._setup_ui()

        # Update coordinates display with initial position
        lat_str = f"{abs(initial_lat):.6f}°{'N' if initial_lat >= 0 else 'S'}"
        lon_str = f"{abs(initial_lon):.6f}°{'W' if initial_lon < 0 else 'E'}"
        self.coords_label.setText(f"Current position: {lat_str}, {lon_str}\nClick on the map to change location")

        # Create web view immediately instead of deferring
        self._load_map()

    def _setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()

        # Search section
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter city name, address, or landmark...")
        self.search_input.returnPressed.connect(self._on_search_clicked)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._on_search_clicked)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)

        # Placeholder for map (will be replaced by web view in _load_map)
        self.map_container = QWidget()
        self.map_container.setMinimumSize(800, 500)
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.map_container, 1)

        # Coordinates display
        self.coords_label = QLabel("Click on the map to select a location")
        self.coords_label.setAlignment(Qt.AlignCenter)
        self.coords_label.setStyleSheet("QLabel { font-size: 12pt; padding: 10px; }")
        layout.addWidget(self.coords_label)

        # Help text
        help_text = QLabel("Tip: You can pan, zoom, and search for locations. Click anywhere on the map to select coordinates.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; padding: 5px; }}")
        help_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(help_text)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.select_button = QPushButton("Select Location")
        self.select_button.clicked.connect(self._on_select_clicked)
        self.select_button.setEnabled(True)  # Enabled since we have initial coordinates
        # Don't set as default to prevent event propagation to parent dialog
        # self.select_button.setDefault(True)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_map_html(self, lat, lon, zoom=10):
        """Create HTML with Leaflet map and JavaScript bridge"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Location Picker</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/qwebchannel@6.2.0/qwebchannel.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 100vh; }}
        .info-box {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            max-width: 500px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box" id="infoBox">Click on the map to select a location</div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var qt_bridge = null;
        var map = null;
        var marker = null;
        var searchTimeout = null;
        var bridgeReady = false;
        var currentGeocodeLat = null;
        var currentGeocodeLon = null;

        // Initialize QWebChannel for Qt-JavaScript communication
        new QWebChannel(qt.webChannelTransport, function(channel) {{
            qt_bridge = channel.objects.qt_bridge;
            bridgeReady = true;
            console.log("Qt bridge initialized successfully");

            // Connect to search results signal
            qt_bridge.searchCompleted.connect(function(resultJson) {{
                console.log("Received search results:", resultJson);
                handleSearchResults(resultJson);
            }});

            // Connect to geocode results signal
            qt_bridge.geocodeCompleted.connect(function(resultJson) {{
                console.log("Received geocode results:", resultJson);
                handleGeocodeResults(resultJson);
            }});
        }});

        // Initialize Leaflet map
        try {{
            map = L.map('map').setView([{lat}, {lon}], {zoom});

            // Add OpenStreetMap tiles
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 19
            }}).addTo(map);

            // Add initial marker at starting position
            marker = L.marker([{lat}, {lon}], {{
                draggable: true
            }}).addTo(map);

            // Handle marker drag
            marker.on('dragend', function(e) {{
                var latlng = e.target.getLatLng();
                onLocationSelected(latlng.lat, latlng.lng);
            }});

            // Handle map clicks
            map.on('click', function(e) {{
                var lat = e.latlng.lat;
                var lon = e.latlng.lng;

                // Update or create marker
                if (marker) {{
                    marker.setLatLng(e.latlng);
                }} else {{
                    marker = L.marker(e.latlng, {{
                        draggable: true
                    }}).addTo(map);

                    marker.on('dragend', function(e) {{
                        var latlng = e.target.getLatLng();
                        onLocationSelected(latlng.lat, latlng.lng);
                    }});
                }}

                onLocationSelected(lat, lon);
            }});

            document.getElementById('infoBox').innerHTML = 'Map loaded successfully - Click to select location';
        }} catch (e) {{
            console.error("Failed to initialize map:", e);
            document.getElementById('infoBox').innerHTML = 'Error loading map: ' + e.message;
        }}

        function onLocationSelected(lat, lon) {{
            // Update info box
            document.getElementById('infoBox').innerHTML =
                'Selected: ' + lat.toFixed(6) + '°, ' + lon.toFixed(6) + '° - Getting location name...';

            // Reverse geocode to get location name
            reverseGeocode(lat, lon);

            // Also send coordinates immediately to Qt (in case reverse geocode fails)
            sendToQt(lat, lon, "");
        }}

        function sendToQt(lat, lon, locationName) {{
            if (bridgeReady && qt_bridge) {{
                try {{
                    qt_bridge.selectLocation(lat, lon, locationName);
                    console.log("Sent to Qt: " + lat + ", " + lon);
                }} catch(e) {{
                    console.error("Error calling Qt bridge:", e);
                }}
            }} else {{
                console.warn("Qt bridge not ready yet, retrying in 100ms...");
                setTimeout(function() {{
                    sendToQt(lat, lon, locationName);
                }}, 100);
            }}
        }}

        function reverseGeocode(lat, lon) {{
            // Store current coordinates for when we receive the result
            currentGeocodeLat = lat;
            currentGeocodeLon = lon;

            // Use Python bridge to reverse geocode (avoids CORS issues)
            if (bridgeReady && qt_bridge && qt_bridge.reverseGeocodeFromPython) {{
                qt_bridge.reverseGeocodeFromPython(lat, lon);
            }} else {{
                console.warn('Bridge not ready or reverseGeocodeFromPython not available');
                document.getElementById('infoBox').innerHTML =
                    'Selected: ' + lat.toFixed(6) + '°, ' + lon.toFixed(6) + '°';
                // Send coordinates without name
                sendToQt(lat, lon, "");
            }}
        }}

        function handleGeocodeResults(resultJson) {{
            try {{
                var data = JSON.parse(resultJson);
                var lat = currentGeocodeLat;
                var lon = currentGeocodeLon;
                var locationName = data.display_name || "";

                // Update info box
                document.getElementById('infoBox').innerHTML =
                    'Selected: ' + lat.toFixed(6) + '°, ' + lon.toFixed(6) + '°<br>' +
                    '<small>' + locationName + '</small>';

                // Send to Qt with location name
                sendToQt(lat, lon, locationName);
            }} catch(error) {{
                console.error('Error handling geocode results:', error);
                document.getElementById('infoBox').innerHTML =
                    'Selected: ' + currentGeocodeLat.toFixed(6) + '°, ' + currentGeocodeLon.toFixed(6) + '°';
                sendToQt(currentGeocodeLat, currentGeocodeLon, "");
            }}
        }}

        // Search function (called from Qt)
        function searchLocation(query) {{
            if (!query || query.trim() === '') {{
                return;
            }}

            document.getElementById('infoBox').innerHTML = 'Searching for: ' + query + '...';

            // Use Python bridge to search (avoids CORS issues)
            if (bridgeReady && qt_bridge && qt_bridge.searchLocationFromPython) {{
                qt_bridge.searchLocationFromPython(query);
            }} else {{
                console.error('Bridge not ready or searchLocationFromPython not available');
                document.getElementById('infoBox').innerHTML = 'Search unavailable. Please try again in a moment.';
            }}
        }}

        function handleSearchResults(resultJson) {{
            try {{
                var data = JSON.parse(resultJson);

                if (data && data.length > 0) {{
                    var result = data[0];
                    var lat = parseFloat(result.lat);
                    var lon = parseFloat(result.lon);

                    // Pan to location
                    map.setView([lat, lon], 13);

                    // Update marker
                    if (marker) {{
                        marker.setLatLng([lat, lon]);
                    }} else {{
                        marker = L.marker([lat, lon], {{
                            draggable: true
                        }}).addTo(map);

                        marker.on('dragend', function(e) {{
                            var latlng = e.target.getLatLng();
                            onLocationSelected(latlng.lat, latlng.lng);
                        }});
                    }}

                    // Select this location
                    onLocationSelected(lat, lon);
                }} else {{
                    document.getElementById('infoBox').innerHTML = 'No results found';
                }}
            }} catch(error) {{
                console.error('Error handling search results:', error);
                document.getElementById('infoBox').innerHTML = 'Search failed. Please try again.';
            }}
        }}
    </script>
</body>
</html>"""
        return html_template

    def _load_map(self):
        """Create and load the web view with the map"""
        try:
            # Import QtWebEngine components
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
                from PySide6.QtWebEngineCore import QWebEngineSettings
                from PySide6.QtWebChannel import QWebChannel
            except ImportError as ie:
                QMessageBox.warning(self, "Feature Unavailable",
                    "Map picker requires QtWebEngine which is not available.\n\n"
                    "You can still enter coordinates manually.")
                logger.error(f"QtWebEngine not available: {ie}")
                self.reject()
                return

            # Create web view
            self.web_view = QWebEngineView()
            self.web_view.setMinimumSize(800, 500)
            self.web_view.setContextMenuPolicy(Qt.DefaultContextMenu)

            # Configure web settings
            settings = self.web_view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

            # Enable developer tools for debugging (optional - don't fail if unavailable)
            try:
                settings.setAttribute(QWebEngineSettings.WebAttribute.DeveloperExtrasEnabled, True)
                logger.debug("Developer tools enabled for map picker")
            except AttributeError:
                logger.debug("DeveloperExtrasEnabled not available in this Qt version")
            except Exception as e:
                logger.debug(f"Could not enable developer tools: {e}")

            # Set up QWebChannel for Qt-JavaScript bridge BEFORE loading HTML
            self.bridge = MapBridge(self)
            self.channel = QWebChannel()
            self.channel.registerObject("qt_bridge", self.bridge)
            self.web_view.page().setWebChannel(self.channel)

            # Generate and load HTML
            html = self._create_map_html(self.initial_lat, self.initial_lon)
            self.web_view.setHtml(html)

            # Add load finished handler to check if everything loaded
            self.web_view.loadFinished.connect(self._on_map_loaded)

            # Add web view to map container
            self.map_layout.addWidget(self.web_view)

            logger.debug("Map view created successfully")
            logger.debug("_load_map() completed successfully - dialog should still be open")

        except Exception as e:
            logger.error(f"Failed to create map view: {e}", exc_info=True)
            QMessageBox.critical(self, "Error",
                f"Failed to load map picker: {str(e)}\n\n"
                "Please enter coordinates manually.")
            logger.debug("About to call self.reject() due to exception")
            self.reject()

    def _on_map_loaded(self, success):
        """Called when the map page finishes loading"""
        if success:
            logger.debug("Map page loaded successfully")

            # Test basic JavaScript execution
            self.web_view.page().runJavaScript(
                "1 + 1",
                lambda result: logger.debug(f"JavaScript execution test (1+1): {result}")
            )

            # Check if QWebChannel is available
            self.web_view.page().runJavaScript(
                "typeof QWebChannel !== 'undefined'",
                lambda result: logger.debug(f"QWebChannel available: {result}")
            )

            # Check if qt object is available
            self.web_view.page().runJavaScript(
                "typeof qt !== 'undefined'",
                lambda result: logger.debug(f"qt object available: {result}")
            )

            # Check bridge status after a delay (give it time to initialize)
            QTimer.singleShot(2000, self._check_bridge_status)
        else:
            logger.error("Map page failed to load")

    def _check_bridge_status(self):
        """Check if the JavaScript bridge is ready"""
        def log_bridge_status(result):
            logger.debug(f"Bridge ready status: {result}")
            if not result:
                logger.error("Bridge failed to initialize! Trying to manually trigger...")
                # Try to manually test the bridge by calling selectLocation
                self.web_view.page().runJavaScript(
                    "if (qt_bridge && qt_bridge.selectLocation) { qt_bridge.selectLocation(40.7128, -74.0060, 'Test Location'); } else { console.log('Bridge not available'); }",
                    lambda r: logger.debug(f"Manual bridge test result: {r}")
                )

        self.web_view.page().runJavaScript("bridgeReady", log_bridge_status)

    def _on_location_selected(self, lat, lon, location_name=""):
        """Called when user selects a location (from JavaScript bridge)"""
        logger.debug(f"_on_location_selected called: lat={lat}, lon={lon}, name={location_name}")
        self.selected_lat = lat
        self.selected_lon = lon
        self.selected_location_name = location_name

        # Update display
        lat_str = f"{abs(lat):.6f}°{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.6f}°{'W' if lon < 0 else 'E'}"

        if location_name:
            # Truncate long location names
            display_name = location_name if len(location_name) <= 60 else location_name[:57] + "..."
            self.coords_label.setText(f"Selected: {lat_str}, {lon_str}\n{display_name}")
        else:
            self.coords_label.setText(f"Selected: {lat_str}, {lon_str}")

        # Enable select button
        self.select_button.setEnabled(True)

        logger.debug(f"Location selected: {lat}, {lon} - {location_name}")

    def _search_location_python(self, query):
        """Search for location using Python's urllib to avoid CORS issues"""
        import urllib.request
        import urllib.parse
        import json

        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_query}&limit=1"

            # Create request with User-Agent header (required by Nominatim)
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'CosmosCollection/1.0'}
            )

            # Make the request
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                logger.debug(f"Search results: {data}")
                return json.dumps(data)  # Return as JSON string

        except Exception as e:
            logger.error(f"Error searching location: {e}")
            return json.dumps([])  # Return empty array on error

    def _reverse_geocode_python(self, lat, lon):
        """Reverse geocode using Python's urllib to avoid CORS issues"""
        import urllib.request
        import json

        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"

            # Create request with User-Agent header (required by Nominatim)
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'CosmosCollection/1.0'}
            )

            # Make the request
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                logger.debug(f"Reverse geocode result: {data}")
                return json.dumps(data)  # Return as JSON string

        except Exception as e:
            logger.error(f"Error reverse geocoding: {e}")
            return json.dumps({})  # Return empty object on error

    def _on_search_clicked(self):
        """Handle search button click"""
        query = self.search_input.text().strip()
        if query and self.web_view:
            # Call JavaScript function which will call back to Python for the actual search
            self.web_view.page().runJavaScript(f"searchLocation({repr(query)})")

    def _on_select_clicked(self):
        """Handle Select Location button click"""
        self.accept()

    def _on_cancel_clicked(self):
        """Handle Cancel button click"""
        self.reject()

    def get_selected_coordinates(self):
        """Get the selected coordinates"""
        if self.selected_lat is not None and self.selected_lon is not None:
            return (self.selected_lat, self.selected_lon, self.selected_location_name)
        return None


# --- Telescope Management Dialog ---
class TelescopeDialog(QDialog):
    """Dialog for managing user telescopes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Telescope Management - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(900, 700)

        self.db_manager = DatabaseManager()
        self._setup_ui()
        self._load_telescopes()
        self._restore_splitter_state()
        
    def _setup_ui(self):
        """Set up the telescope management UI"""
        layout = QVBoxLayout()

        # Create main splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left side - telescope list (in a container widget)
        left_widget = QWidget()
        list_layout = QVBoxLayout(left_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("Your Telescopes:")
        list_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        list_layout.addWidget(list_label)

        # Telescope list
        self.telescope_list = QListWidget()
        self.telescope_list.itemSelectionChanged.connect(self._on_telescope_selected)
        list_layout.addWidget(self.telescope_list)

        # List action buttons
        list_button_layout = QHBoxLayout()
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._delete_telescope)
        self.delete_button.setEnabled(False)
        list_button_layout.addWidget(self.delete_button)

        self.set_active_button = QPushButton("Enable")
        self.set_active_button.setToolTip("Enable/disable telescope in FOV Simulator")
        self.set_active_button.clicked.connect(self._set_active_telescope)
        self.set_active_button.setEnabled(False)
        list_button_layout.addWidget(self.set_active_button)

        list_layout.addLayout(list_button_layout)
        self.main_splitter.addWidget(left_widget)
        
        # Right side - telescope form (in a container widget)
        right_widget = QWidget()
        form_layout = QVBoxLayout(right_widget)
        form_layout.setContentsMargins(0, 0, 0, 0)

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
        self.fratio_display.setStyleSheet(f"color: {COLORS['text_disabled']};")
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

        # Equipment section
        self._setup_equipment_section(form_layout)

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
        self.main_splitter.addWidget(right_widget)

        # Set initial splitter sizes (left panel smaller than right)
        self.main_splitter.setSizes([250, 650])
        self.main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.main_splitter.setStretchFactor(1, 1)  # Right panel stretches

        layout.addWidget(self.main_splitter)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        help_text = QLabel("Tip: Enable telescopes to make them available in the FOV Simulator. Multiple telescopes can be enabled.")
        help_text.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 9pt;")
        bottom_layout.addWidget(help_text)
        
        bottom_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_layout.addWidget(close_button)
        
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

        # Track current editing telescope
        self.current_telescope_id = None
        self.current_telescope_is_active = False
        
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
                    self.fratio_display.setStyleSheet(f"color: {COLORS['text_disabled']};")
            else:
                self.fratio_display.setText("N/A")
                self.fratio_display.setStyleSheet(f"color: {COLORS['text_disabled']};")
        except ValueError:
            self.fratio_display.setText("N/A")
            self.fratio_display.setStyleSheet(f"color: {COLORS['text_disabled']};")

    def _save_splitter_state(self):
        """Save the splitter state to QSettings"""
        try:
            settings = QSettings("CosmosCollection", "TelescopeDialog")
            settings.setValue("splitter_state", self.main_splitter.saveState())
        except Exception as e:
            logger.error(f"Error saving splitter state: {str(e)}")

    def _restore_splitter_state(self):
        """Restore the splitter state from QSettings"""
        try:
            settings = QSettings("CosmosCollection", "TelescopeDialog")
            splitter_state = settings.value("splitter_state")
            if splitter_state:
                self.main_splitter.restoreState(splitter_state)
        except Exception as e:
            logger.error(f"Error restoring splitter state: {str(e)}")

    def closeEvent(self, event):
        """Handle dialog close - save splitter state"""
        self._save_splitter_state()
        event.accept()

    def accept(self):
        """Handle dialog accept - save splitter state"""
        self._save_splitter_state()
        super().accept()

    def reject(self):
        """Handle dialog reject - save splitter state"""
        self._save_splitter_state()
        super().reject()

    def _load_telescopes(self):
        """Load telescopes from database into the list"""
        try:
            # Block signals to prevent selection events during loading
            self.telescope_list.blockSignals(True)
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
                    status = " (Enabled)" if is_active else ""

                    item_text = f"{name}{status}"
                    if aperture:
                        item_text += f" - {aperture}mm"
                    if fratio > 0:
                        item_text += f" f/{fratio:.1f}"

                    # Create list item
                    from PySide6.QtWidgets import QListWidgetItem
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, telescope_id)  # Store telescope ID

                    # Highlight enabled telescope
                    if is_active:
                        item.setBackground(QColor(0, 120, 212, 50))  # Light blue background

                    self.telescope_list.addItem(item)

            # Unblock signals before clearing selection
            self.telescope_list.blockSignals(False)

            # Clear selection and form after loading
            self._clear_form()

        except Exception as e:
            # Make sure to unblock signals even if there's an error
            self.telescope_list.blockSignals(False)
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
                    SELECT name, aperture, focal_length, mount_type, notes, is_active
                    FROM usertelescopes
                    WHERE id = ?
                """, (telescope_id,))

                row = cursor.fetchone()
                if row:
                    name, aperture, focal_length, mount_type, notes, is_active = row

                    self.name_input.setText(name or "")
                    self.aperture_input.setText(str(aperture) if aperture else "")
                    self.focal_length_input.setText(str(focal_length) if focal_length else "")

                    # Set mount type
                    mount_index = self.mount_combo.findText(mount_type or "")
                    if mount_index >= 0:
                        self.mount_combo.setCurrentIndex(mount_index)

                    self.notes_input.setPlainText(notes or "")

                    # Set current editing ID and active status
                    self.current_telescope_id = telescope_id
                    self.current_telescope_is_active = bool(is_active)

                    # Update save button text
                    self.save_button.setText("Update Telescope")

                    # Update the enable/disable button text
                    if self.current_telescope_is_active:
                        self.set_active_button.setText("Disable")
                    else:
                        self.set_active_button.setText("Enable")

                    # Load equipment for this telescope
                    self._load_equipment_for_telescope(telescope_id)

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
        self.current_telescope_is_active = False
        self.save_button.setText("Save Telescope")
        self.set_active_button.setText("Enable")

        # Clear equipment selections (uncheck all)
        self.camera_list.blockSignals(True)
        self.eyepiece_list.blockSignals(True)
        self.barlow_list.blockSignals(True)
        for i in range(self.camera_list.count()):
            self.camera_list.item(i).setCheckState(Qt.Unchecked)
        for i in range(self.eyepiece_list.count()):
            self.eyepiece_list.item(i).setCheckState(Qt.Unchecked)
        for i in range(self.barlow_list.count()):
            self.barlow_list.item(i).setCheckState(Qt.Unchecked)
        self.camera_list.blockSignals(False)
        self.eyepiece_list.blockSignals(False)
        self.barlow_list.blockSignals(False)

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
        telescope_name = selected_items[0].text().split(" (")[0]  # Remove status/specs text
        
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
        """Toggle active/inactive status for selected telescope"""
        selected_items = self.telescope_list.selectedItems()
        if not selected_items:
            return

        telescope_id = selected_items[0].data(Qt.UserRole)
        telescope_name = selected_items[0].text().split(" (")[0]  # Remove status/specs text

        # Validate telescope_id
        if telescope_id is None:
            QMessageBox.warning(self, "Error", "Invalid telescope selection. Please try again.")
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # First, verify the telescope exists and get its current active status
                cursor.execute("SELECT is_active FROM usertelescopes WHERE id = ?", (telescope_id,))
                result = cursor.fetchone()
                if not result:
                    QMessageBox.warning(self, "Error", f"Telescope with ID {telescope_id} not found in database.")
                    return

                current_is_active = bool(result[0])

                # Toggle the active status
                new_status = 0 if current_is_active else 1
                cursor.execute("UPDATE usertelescopes SET is_active = ? WHERE id = ?", (new_status, telescope_id))
                rows_updated = cursor.rowcount

                if rows_updated == 0:
                    QMessageBox.warning(self, "Error", f"Failed to update telescope status. No rows were updated.")
                    conn.rollback()
                    return

                conn.commit()

                # Log the change
                status_text = "inactive" if current_is_active else "active"
                logger.debug(f"Set telescope ID {telescope_id} to {status_text}, rows affected: {rows_updated}")

            # Show success message
            if current_is_active:
                success_message = f"Telescope '{telescope_name}' is now inactive and will not appear in the FOV Simulator."
            else:
                success_message = f"Telescope '{telescope_name}' is now active and will appear in the FOV Simulator."

            QMessageBox.information(self, "Success", success_message)

            # Reload telescopes
            self._load_telescopes()

        except Exception as e:
            logger.error(f"Error toggling telescope active status: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to update telescope status: {str(e)}")

    def _setup_equipment_section(self, parent_layout):
        """Create equipment lists with checkboxes and add buttons"""
        equipment_group = QGroupBox("Equipment for this Telescope (check items to associate)")
        equipment_group_layout = QHBoxLayout(equipment_group)

        # Camera column
        camera_col = QVBoxLayout()
        camera_header = QHBoxLayout()
        camera_label = QLabel("Cameras:")
        camera_label.setStyleSheet("font-weight: bold;")
        add_camera_btn = QPushButton("Add New")
        add_camera_btn.setMaximumWidth(70)
        add_camera_btn.clicked.connect(self._show_add_camera_dialog)
        camera_header.addWidget(camera_label)
        camera_header.addStretch()
        camera_header.addWidget(add_camera_btn)
        camera_col.addLayout(camera_header)
        self.camera_list = QListWidget()
        self.camera_list.setMaximumHeight(100)
        self.camera_list.itemChanged.connect(self._on_equipment_item_changed)
        camera_col.addWidget(self.camera_list)
        equipment_group_layout.addLayout(camera_col)

        # Eyepiece column
        eyepiece_col = QVBoxLayout()
        eyepiece_header = QHBoxLayout()
        eyepiece_label = QLabel("Eyepieces:")
        eyepiece_label.setStyleSheet("font-weight: bold;")
        add_eyepiece_btn = QPushButton("Add New")
        add_eyepiece_btn.setMaximumWidth(70)
        add_eyepiece_btn.clicked.connect(self._show_add_eyepiece_dialog)
        eyepiece_header.addWidget(eyepiece_label)
        eyepiece_header.addStretch()
        eyepiece_header.addWidget(add_eyepiece_btn)
        eyepiece_col.addLayout(eyepiece_header)
        self.eyepiece_list = QListWidget()
        self.eyepiece_list.setMaximumHeight(100)
        self.eyepiece_list.itemChanged.connect(self._on_equipment_item_changed)
        eyepiece_col.addWidget(self.eyepiece_list)
        equipment_group_layout.addLayout(eyepiece_col)

        # Barlow/Reducer column
        barlow_col = QVBoxLayout()
        barlow_header = QHBoxLayout()
        barlow_label = QLabel("Barlows/Reducers:")
        barlow_label.setStyleSheet("font-weight: bold;")
        add_barlow_btn = QPushButton("Add New")
        add_barlow_btn.setMaximumWidth(70)
        add_barlow_btn.clicked.connect(self._show_add_barlow_dialog)
        barlow_header.addWidget(barlow_label)
        barlow_header.addStretch()
        barlow_header.addWidget(add_barlow_btn)
        barlow_col.addLayout(barlow_header)
        self.barlow_list = QListWidget()
        self.barlow_list.setMaximumHeight(100)
        self.barlow_list.itemChanged.connect(self._on_equipment_item_changed)
        barlow_col.addWidget(self.barlow_list)
        equipment_group_layout.addLayout(barlow_col)

        parent_layout.addWidget(equipment_group)

        # Initialize equipment lists
        self._populate_equipment_lists()

    def _populate_equipment_lists(self):
        """Populate equipment lists with checkable items from database"""
        # Block signals during population
        self.camera_list.blockSignals(True)
        self.eyepiece_list.blockSignals(True)
        self.barlow_list.blockSignals(True)

        # Clear existing items
        self.camera_list.clear()
        self.eyepiece_list.clear()
        self.barlow_list.clear()

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Load cameras
                cursor.execute("""
                    SELECT id, name, sensor_width, sensor_height
                    FROM userequipment
                    WHERE equipment_type = 'camera'
                    ORDER BY name ASC
                """)
                for row in cursor.fetchall():
                    eq_id, name, sensor_w, sensor_h = row
                    display_text = f"{name} ({sensor_w}x{sensor_h}mm)" if sensor_w and sensor_h else name
                    item = QListWidgetItem(display_text)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    item.setData(Qt.UserRole, {"id": eq_id, "type": "camera"})
                    self.camera_list.addItem(item)

                # Load eyepieces
                cursor.execute("""
                    SELECT id, name, focal_length, apparent_fov
                    FROM userequipment
                    WHERE equipment_type = 'eyepiece'
                    ORDER BY name ASC
                """)
                for row in cursor.fetchall():
                    eq_id, name, focal_length, apparent_fov = row
                    display_text = f"{name} ({focal_length}mm, {apparent_fov}\u00b0)" if focal_length and apparent_fov else name
                    item = QListWidgetItem(display_text)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    item.setData(Qt.UserRole, {"id": eq_id, "type": "eyepiece"})
                    self.eyepiece_list.addItem(item)

                # Load barlows and reducers
                cursor.execute("""
                    SELECT id, name, factor, equipment_type
                    FROM userequipment
                    WHERE equipment_type IN ('barlow', 'reducer')
                    ORDER BY name ASC
                """)
                for row in cursor.fetchall():
                    eq_id, name, factor, eq_type = row
                    display_text = f"{name} ({factor}x)" if factor else name
                    item = QListWidgetItem(display_text)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    item.setData(Qt.UserRole, {"id": eq_id, "type": eq_type})
                    self.barlow_list.addItem(item)

        except Exception as e:
            logger.error(f"Error loading equipment: {str(e)}")

        # Unblock signals
        self.camera_list.blockSignals(False)
        self.eyepiece_list.blockSignals(False)
        self.barlow_list.blockSignals(False)

    def _load_equipment_for_telescope(self, telescope_id):
        """Load equipment assigned to the selected telescope"""
        # Block signals during loading
        self.camera_list.blockSignals(True)
        self.eyepiece_list.blockSignals(True)
        self.barlow_list.blockSignals(True)

        # Uncheck all items first
        for i in range(self.camera_list.count()):
            self.camera_list.item(i).setCheckState(Qt.Unchecked)
        for i in range(self.eyepiece_list.count()):
            self.eyepiece_list.item(i).setCheckState(Qt.Unchecked)
        for i in range(self.barlow_list.count()):
            self.barlow_list.item(i).setCheckState(Qt.Unchecked)

        if not telescope_id:
            self.camera_list.blockSignals(False)
            self.eyepiece_list.blockSignals(False)
            self.barlow_list.blockSignals(False)
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Find all equipment assigned to this telescope via junction table
                cursor.execute("""
                    SELECT equipment_id
                    FROM telescope_equipment
                    WHERE telescope_id = ?
                """, (telescope_id,))

                assigned_ids = {row[0] for row in cursor.fetchall()}

                # Check the assigned equipment in each list
                for i in range(self.camera_list.count()):
                    item = self.camera_list.item(i)
                    data = item.data(Qt.UserRole)
                    if data and data.get('id') in assigned_ids:
                        item.setCheckState(Qt.Checked)

                for i in range(self.eyepiece_list.count()):
                    item = self.eyepiece_list.item(i)
                    data = item.data(Qt.UserRole)
                    if data and data.get('id') in assigned_ids:
                        item.setCheckState(Qt.Checked)

                for i in range(self.barlow_list.count()):
                    item = self.barlow_list.item(i)
                    data = item.data(Qt.UserRole)
                    if data and data.get('id') in assigned_ids:
                        item.setCheckState(Qt.Checked)

        except Exception as e:
            logger.error(f"Error loading equipment for telescope: {str(e)}")

        # Unblock signals
        self.camera_list.blockSignals(False)
        self.eyepiece_list.blockSignals(False)
        self.barlow_list.blockSignals(False)

    def _on_equipment_item_changed(self, item):
        """Handle equipment checkbox state change"""
        if not self.current_telescope_id:
            return

        data = item.data(Qt.UserRole)
        if not data:
            return

        equipment_id = data.get('id')
        is_checked = item.checkState() == Qt.Checked

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if is_checked:
                    # Add equipment-telescope link (ignore if already exists)
                    cursor.execute("""
                        INSERT OR IGNORE INTO telescope_equipment (telescope_id, equipment_id)
                        VALUES (?, ?)
                    """, (self.current_telescope_id, equipment_id))
                else:
                    # Remove equipment-telescope link
                    cursor.execute("""
                        DELETE FROM telescope_equipment
                        WHERE telescope_id = ? AND equipment_id = ?
                    """, (self.current_telescope_id, equipment_id))

                conn.commit()

        except Exception as e:
            logger.error(f"Error saving equipment assignment: {str(e)}")

    def _get_preset_cameras(self):
        """Return preset camera data for autocomplete suggestions"""
        return {
            # DSLR cameras
            "Canon Full Frame": {"sensor_width": 36, "sensor_height": 24},
            "Canon APS-C": {"sensor_width": 22.3, "sensor_height": 14.9},
            "Canon APS-H": {"sensor_width": 28.7, "sensor_height": 19.0},
            "Nikon Full Frame": {"sensor_width": 35.9, "sensor_height": 24.0},
            "Nikon APS-C": {"sensor_width": 23.5, "sensor_height": 15.6},
            "Sony Full Frame": {"sensor_width": 35.8, "sensor_height": 23.8},
            "Sony APS-C": {"sensor_width": 23.5, "sensor_height": 15.6},
            # ZWO ASI cameras
            "ASI6200MM Pro": {"sensor_width": 36.0, "sensor_height": 24.0},
            "ASI2600MM Pro": {"sensor_width": 23.5, "sensor_height": 15.7},
            "ASI533MM Pro": {"sensor_width": 11.3, "sensor_height": 7.1},
            "ASI294MM Pro": {"sensor_width": 19.1, "sensor_height": 13.0},
            "ASI183MM Pro": {"sensor_width": 13.2, "sensor_height": 8.8},
            "ASI585MC": {"sensor_width": 8.3, "sensor_height": 6.2},
            "ASI662MC (Seestar S30)": {"sensor_width": 7.4, "sensor_height": 5.6},
            "ASI385MC": {"sensor_width": 7.7, "sensor_height": 4.9},
            "ASI462MC (Seestar S50)": {"sensor_width": 2.9, "sensor_height": 2.9},
            "ASI224MC": {"sensor_width": 3.9, "sensor_height": 2.8},
            "ASI120MM": {"sensor_width": 3.8, "sensor_height": 2.8},
            # QHY cameras
            "QHY600M": {"sensor_width": 36.0, "sensor_height": 24.0},
            "QHY268M": {"sensor_width": 23.5, "sensor_height": 15.7},
            "QHY294M": {"sensor_width": 19.1, "sensor_height": 13.0},
            "QHY183M": {"sensor_width": 13.2, "sensor_height": 8.8},
            "QHY174M": {"sensor_width": 11.3, "sensor_height": 7.1},
            # SBIG cameras
            "SBIG STF-8300M": {"sensor_width": 17.96, "sensor_height": 13.52},
            "SBIG ST-2000XM": {"sensor_width": 15.2, "sensor_height": 15.2},
            # Atik cameras
            "Atik 460EX": {"sensor_width": 36.0, "sensor_height": 24.0},
            "Atik 383L+": {"sensor_width": 23.6, "sensor_height": 15.8},
        }

    def _show_add_camera_dialog(self):
        """Dialog to add a new camera"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Camera")
        dialog.setModal(True)
        dialog.resize(400, 220)

        layout = QVBoxLayout(dialog)

        # Hint label
        hint_label = QLabel("Start typing to see suggestions from preset cameras")
        hint_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 9pt;")
        layout.addWidget(hint_label)

        # Name field with autocomplete
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setMinimumWidth(120)
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., ASI294MM Pro")

        # Set up autocomplete
        preset_cameras = self._get_preset_cameras()
        completer = QCompleter(list(preset_cameras.keys()))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        name_input.setCompleter(completer)

        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)

        # Sensor width field
        width_layout = QHBoxLayout()
        width_label = QLabel("Sensor Width (mm):")
        width_label.setMinimumWidth(120)
        width_input = QLineEdit()
        width_input.setPlaceholderText("e.g., 19.1")
        width_layout.addWidget(width_label)
        width_layout.addWidget(width_input)
        layout.addLayout(width_layout)

        # Sensor height field
        height_layout = QHBoxLayout()
        height_label = QLabel("Sensor Height (mm):")
        height_label.setMinimumWidth(120)
        height_input = QLineEdit()
        height_input.setPlaceholderText("e.g., 13.0")
        height_layout.addWidget(height_label)
        height_layout.addWidget(height_input)
        layout.addLayout(height_layout)

        # Auto-fill when a preset is selected
        def on_completer_activated(text):
            if text in preset_cameras:
                data = preset_cameras[text]
                width_input.setText(str(data["sensor_width"]))
                height_input.setText(str(data["sensor_height"]))

        completer.activated.connect(on_completer_activated)

        # Also check on text change for exact matches
        def on_name_changed(text):
            if text in preset_cameras:
                data = preset_cameras[text]
                width_input.setText(str(data["sensor_width"]))
                height_input.setText(str(data["sensor_height"]))

        name_input.textChanged.connect(on_name_changed)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        def save_camera():
            name = name_input.text().strip()
            width_text = width_input.text().strip()
            height_text = height_input.text().strip()

            if not name:
                QMessageBox.warning(dialog, "Invalid Input", "Please enter a camera name.")
                return

            try:
                sensor_width = float(width_text) if width_text else None
                sensor_height = float(height_text) if height_text else None

                if sensor_width is not None and sensor_width <= 0:
                    QMessageBox.warning(dialog, "Invalid Input", "Sensor width must be positive.")
                    return
                if sensor_height is not None and sensor_height <= 0:
                    QMessageBox.warning(dialog, "Invalid Input", "Sensor height must be positive.")
                    return

            except ValueError:
                QMessageBox.warning(dialog, "Invalid Input", "Please enter valid numeric values for sensor dimensions.")
                return

            # Save to database
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO userequipment (equipment_type, name, sensor_width, sensor_height)
                        VALUES ('camera', ?, ?, ?)
                    """, (name, sensor_width, sensor_height))
                    conn.commit()

                dialog.accept()
                self._populate_equipment_lists()
                QMessageBox.information(self, "Success", f"Camera '{name}' has been added.")

            except Exception as e:
                logger.error(f"Error saving camera: {str(e)}")
                QMessageBox.critical(dialog, "Error", f"Failed to save camera: {str(e)}")

        save_btn.clicked.connect(save_camera)
        dialog.exec()

    def _get_preset_eyepieces(self):
        """Return preset eyepiece data for autocomplete suggestions"""
        return {
            "32mm Plossl (52 AFOV)": {"focal_length": 32, "apparent_fov": 52},
            "25mm Plossl (52 AFOV)": {"focal_length": 25, "apparent_fov": 52},
            "20mm Plossl (50 AFOV)": {"focal_length": 20, "apparent_fov": 50},
            "15mm Plossl (50 AFOV)": {"focal_length": 15, "apparent_fov": 50},
            "10mm Plossl (50 AFOV)": {"focal_length": 10, "apparent_fov": 50},
            "6mm Plossl (50 AFOV)": {"focal_length": 6, "apparent_fov": 50},
            # Televue Ethos (100 AFOV)
            "Televue 21mm Ethos": {"focal_length": 21, "apparent_fov": 100},
            "Televue 17mm Ethos": {"focal_length": 17, "apparent_fov": 100},
            "Televue 13mm Ethos": {"focal_length": 13, "apparent_fov": 100},
            "Televue 10mm Ethos": {"focal_length": 10, "apparent_fov": 100},
            "Televue 8mm Ethos": {"focal_length": 8, "apparent_fov": 100},
            "Televue 6mm Ethos": {"focal_length": 6, "apparent_fov": 100},
            "Televue 4.7mm Ethos": {"focal_length": 4.7, "apparent_fov": 100},
            "Televue 3.7mm Ethos": {"focal_length": 3.7, "apparent_fov": 100},
            # Televue Nagler (82 AFOV)
            "Televue 31mm Nagler": {"focal_length": 31, "apparent_fov": 82},
            "Televue 22mm Nagler": {"focal_length": 22, "apparent_fov": 82},
            "Televue 17mm Nagler": {"focal_length": 17, "apparent_fov": 82},
            "Televue 16mm Nagler": {"focal_length": 16, "apparent_fov": 82},
            "Televue 13mm Nagler": {"focal_length": 13, "apparent_fov": 82},
            "Televue 12mm Nagler": {"focal_length": 12, "apparent_fov": 82},
            "Televue 11mm Nagler": {"focal_length": 11, "apparent_fov": 82},
            "Televue 9mm Nagler": {"focal_length": 9, "apparent_fov": 82},
            "Televue 7mm Nagler": {"focal_length": 7, "apparent_fov": 82},
            "Televue 5mm Nagler": {"focal_length": 5, "apparent_fov": 82},
            "Televue 3.5mm Nagler": {"focal_length": 3.5, "apparent_fov": 82},
            # Televue Panoptic (68 AFOV)
            "Televue 41mm Panoptic": {"focal_length": 41, "apparent_fov": 68},
            "Televue 35mm Panoptic": {"focal_length": 35, "apparent_fov": 68},
            "Televue 27mm Panoptic": {"focal_length": 27, "apparent_fov": 68},
            "Televue 24mm Panoptic": {"focal_length": 24, "apparent_fov": 68},
            "Televue 19mm Panoptic": {"focal_length": 19, "apparent_fov": 68},
            "Televue 15mm Panoptic": {"focal_length": 15, "apparent_fov": 68},
            # Explore Scientific (82 AFOV)
            "ES 30mm 82": {"focal_length": 30, "apparent_fov": 82},
            "ES 24mm 82": {"focal_length": 24, "apparent_fov": 82},
            "ES 18mm 82": {"focal_length": 18, "apparent_fov": 82},
            "ES 14mm 82": {"focal_length": 14, "apparent_fov": 82},
            "ES 11mm 82": {"focal_length": 11, "apparent_fov": 82},
            "ES 8.8mm 82": {"focal_length": 8.8, "apparent_fov": 82},
            "ES 6.7mm 82": {"focal_length": 6.7, "apparent_fov": 82},
            "ES 4.7mm 82": {"focal_length": 4.7, "apparent_fov": 82},
            # Explore Scientific (68 AFOV)
            "ES 40mm 68": {"focal_length": 40, "apparent_fov": 68},
            "ES 34mm 68": {"focal_length": 34, "apparent_fov": 68},
            "ES 28mm 68": {"focal_length": 28, "apparent_fov": 68},
            "ES 24mm 68": {"focal_length": 24, "apparent_fov": 68},
            "ES 20mm 68": {"focal_length": 20, "apparent_fov": 68},
            "ES 16mm 68": {"focal_length": 16, "apparent_fov": 68},
        }

    def _show_add_eyepiece_dialog(self):
        """Dialog to add a new eyepiece"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Eyepiece")
        dialog.setModal(True)
        dialog.resize(400, 220)

        layout = QVBoxLayout(dialog)

        # Hint label
        hint_label = QLabel("Start typing to see suggestions from preset eyepieces")
        hint_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 9pt;")
        layout.addWidget(hint_label)

        # Name field with autocomplete
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setMinimumWidth(140)
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Televue 13mm Ethos")

        # Set up autocomplete
        preset_eyepieces = self._get_preset_eyepieces()
        completer = QCompleter(list(preset_eyepieces.keys()))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        name_input.setCompleter(completer)

        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)

        # Focal length field
        focal_layout = QHBoxLayout()
        focal_label = QLabel("Focal Length (mm):")
        focal_label.setMinimumWidth(140)
        focal_input = QLineEdit()
        focal_input.setPlaceholderText("e.g., 13")
        focal_layout.addWidget(focal_label)
        focal_layout.addWidget(focal_input)
        layout.addLayout(focal_layout)

        # Apparent FOV field
        afov_layout = QHBoxLayout()
        afov_label = QLabel("Apparent FOV (degrees):")
        afov_label.setMinimumWidth(140)
        afov_input = QLineEdit()
        afov_input.setPlaceholderText("e.g., 100")
        afov_layout.addWidget(afov_label)
        afov_layout.addWidget(afov_input)
        layout.addLayout(afov_layout)

        # Auto-fill when a preset is selected
        def on_completer_activated(text):
            if text in preset_eyepieces:
                data = preset_eyepieces[text]
                focal_input.setText(str(data["focal_length"]))
                afov_input.setText(str(data["apparent_fov"]))

        completer.activated.connect(on_completer_activated)

        # Also check on text change for exact matches
        def on_name_changed(text):
            if text in preset_eyepieces:
                data = preset_eyepieces[text]
                focal_input.setText(str(data["focal_length"]))
                afov_input.setText(str(data["apparent_fov"]))

        name_input.textChanged.connect(on_name_changed)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        def save_eyepiece():
            name = name_input.text().strip()
            focal_text = focal_input.text().strip()
            afov_text = afov_input.text().strip()

            if not name:
                QMessageBox.warning(dialog, "Invalid Input", "Please enter an eyepiece name.")
                return

            try:
                focal_length = float(focal_text) if focal_text else None
                apparent_fov = float(afov_text) if afov_text else None

                if focal_length is not None and focal_length <= 0:
                    QMessageBox.warning(dialog, "Invalid Input", "Focal length must be positive.")
                    return
                if apparent_fov is not None and (apparent_fov <= 0 or apparent_fov > 180):
                    QMessageBox.warning(dialog, "Invalid Input", "Apparent FOV must be between 0 and 180 degrees.")
                    return

            except ValueError:
                QMessageBox.warning(dialog, "Invalid Input", "Please enter valid numeric values.")
                return

            # Save to database
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO userequipment (equipment_type, name, focal_length, apparent_fov)
                        VALUES ('eyepiece', ?, ?, ?)
                    """, (name, focal_length, apparent_fov))
                    conn.commit()

                dialog.accept()
                self._populate_equipment_lists()
                QMessageBox.information(self, "Success", f"Eyepiece '{name}' has been added.")

            except Exception as e:
                logger.error(f"Error saving eyepiece: {str(e)}")
                QMessageBox.critical(dialog, "Error", f"Failed to save eyepiece: {str(e)}")

        save_btn.clicked.connect(save_eyepiece)
        dialog.exec()

    def _get_preset_barlows(self):
        """Return preset barlow/reducer data for autocomplete suggestions"""
        return {
            # Barlows
            "1.25x Barlow": {"factor": 1.25, "type": "barlow"},
            "1.5x Barlow": {"factor": 1.5, "type": "barlow"},
            "2x Barlow": {"factor": 2.0, "type": "barlow"},
            "2.5x Barlow": {"factor": 2.5, "type": "barlow"},
            "3x Barlow": {"factor": 3.0, "type": "barlow"},
            "4x Barlow": {"factor": 4.0, "type": "barlow"},
            "5x Barlow": {"factor": 5.0, "type": "barlow"},
            # Televue Powermates
            "Televue 2x Powermate": {"factor": 2.0, "type": "barlow"},
            "Televue 2.5x Powermate": {"factor": 2.5, "type": "barlow"},
            "Televue 4x Powermate": {"factor": 4.0, "type": "barlow"},
            "Televue 5x Powermate": {"factor": 5.0, "type": "barlow"},
            # Reducers
            "0.5x Reducer": {"factor": 0.5, "type": "reducer"},
            "0.6x Reducer": {"factor": 0.6, "type": "reducer"},
            "0.63x Reducer": {"factor": 0.63, "type": "reducer"},
            "0.67x Reducer": {"factor": 0.67, "type": "reducer"},
            "0.7x Reducer": {"factor": 0.7, "type": "reducer"},
            "0.75x Reducer": {"factor": 0.75, "type": "reducer"},
            "0.8x Reducer": {"factor": 0.8, "type": "reducer"},
            # Starizona reducers
            "Starizona SCT Corrector 0.63x": {"factor": 0.63, "type": "reducer"},
            "Starizona Hyperstar": {"factor": 0.33, "type": "reducer"},
        }

    def _show_add_barlow_dialog(self):
        """Dialog to add a new barlow/reducer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Barlow/Reducer")
        dialog.setModal(True)
        dialog.resize(400, 220)

        layout = QVBoxLayout(dialog)

        # Hint label
        hint_label = QLabel("Start typing to see suggestions from preset barlows/reducers")
        hint_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 9pt;")
        layout.addWidget(hint_label)

        # Name field with autocomplete
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setMinimumWidth(100)
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Televue 2x Powermate")

        # Set up autocomplete
        preset_barlows = self._get_preset_barlows()
        completer = QCompleter(list(preset_barlows.keys()))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        name_input.setCompleter(completer)

        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)

        # Type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_label.setMinimumWidth(100)
        type_combo = QComboBox()
        type_combo.addItems(["Barlow", "Reducer"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Factor field
        factor_layout = QHBoxLayout()
        factor_label = QLabel("Factor:")
        factor_label.setMinimumWidth(100)
        factor_input = QLineEdit()
        factor_input.setPlaceholderText("e.g., 2.0 for barlow, 0.63 for reducer")
        factor_layout.addWidget(factor_label)
        factor_layout.addWidget(factor_input)
        layout.addLayout(factor_layout)

        # Auto-fill when a preset is selected
        def on_completer_activated(text):
            if text in preset_barlows:
                data = preset_barlows[text]
                factor_input.setText(str(data["factor"]))
                type_combo.setCurrentText(data["type"].title())

        completer.activated.connect(on_completer_activated)

        # Also check on text change for exact matches
        def on_name_changed(text):
            if text in preset_barlows:
                data = preset_barlows[text]
                factor_input.setText(str(data["factor"]))
                type_combo.setCurrentText(data["type"].title())

        name_input.textChanged.connect(on_name_changed)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        def save_barlow():
            name = name_input.text().strip()
            factor_text = factor_input.text().strip()
            eq_type = type_combo.currentText().lower()

            if not name:
                QMessageBox.warning(dialog, "Invalid Input", "Please enter a name.")
                return

            try:
                factor = float(factor_text) if factor_text else None

                if factor is not None and factor <= 0:
                    QMessageBox.warning(dialog, "Invalid Input", "Factor must be positive.")
                    return

            except ValueError:
                QMessageBox.warning(dialog, "Invalid Input", "Please enter a valid numeric factor.")
                return

            # Save to database
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO userequipment (equipment_type, name, factor)
                        VALUES (?, ?, ?)
                    """, (eq_type, name, factor))
                    conn.commit()

                dialog.accept()
                self._populate_equipment_lists()
                QMessageBox.information(self, "Success", f"{eq_type.title()} '{name}' has been added.")

            except Exception as e:
                logger.error(f"Error saving barlow/reducer: {str(e)}")
                QMessageBox.critical(dialog, "Error", f"Failed to save: {str(e)}")

        save_btn.clicked.connect(save_barlow)
        dialog.exec()


# --- Bulk Add to Target List Dialog ---
class BulkAddToTargetDialog(QDialog):
    """Dialog for bulk-adding DSOs without images to the target list"""

    def __init__(self, dso_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bulk Add to Target List")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(700, 500)

        self.dso_list = sorted(dso_list, key=lambda e: e.get('name', ''))
        # Build a lookup by name for retrieval during save
        self._dso_by_name = {e['name']: e for e in dso_list}
        self.db_manager = DatabaseManager()

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        count = len(self.dso_list)
        label = QLabel(f"{count} object{'s' if count != 1 else ''} without images are not yet in your target list.")
        layout.addWidget(label)

        # Table
        self.table = QTableWidget(count, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Priority", "Scope"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 110)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(2, 170)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        # Load telescope options once
        telescope_options = [("Any", None)]
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, name FROM usertelescopes WHERE is_active = 1 ORDER BY name"
                )
                for tid, tname in cursor.fetchall():
                    telescope_options.append((tname, tid))
        except Exception:
            pass

        for row, dso in enumerate(self.dso_list):
            # Name (read-only)
            name_item = QTableWidgetItem(dso.get('name', ''))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            # Priority combo
            priority_combo = QComboBox()
            priority_combo.addItems(["Low", "Medium", "High", "Urgent"])
            priority_combo.setCurrentText("Medium")
            self.table.setCellWidget(row, 1, priority_combo)

            # Scope combo
            scope_combo = QComboBox()
            for label, tid in telescope_options:
                scope_combo.addItem(label, tid)
            self.table.setCellWidget(row, 2, scope_combo)

        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        add_btn = QPushButton(f"Add {count} Object{'s' if count != 1 else ''} to Target List")
        add_btn.setDefault(True)
        add_btn.clicked.connect(self._add_all)
        btn_layout.addWidget(add_btn)

        layout.addLayout(btn_layout)

    def _add_all(self):
        """Insert all rows into the target list database"""
        from datetime import datetime
        date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inserted = 0

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                for row in range(self.table.rowCount()):
                    name_item = self.table.item(row, 0)
                    if name_item is None:
                        continue
                    name = name_item.text()
                    dso = self._dso_by_name.get(name, {})

                    priority = self.table.cellWidget(row, 1).currentText()
                    telescope_id = self.table.cellWidget(row, 2).currentData()

                    size_min = dso.get('size_min', 0) or 0
                    size_max = dso.get('size_max', 0) or 0
                    size_info = f"{size_min:.1f} x {size_max:.1f}" if (size_min or size_max) else ""

                    cursor.execute("""
                        INSERT INTO usertargetlist (
                            name, dso_type, constellation, ra_deg, dec_deg,
                            magnitude, size_info, priority, status, date_added, telescope_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Not Observed', ?, ?)
                    """, (
                        name,
                        dso.get('dso_type', ''),
                        dso.get('constellation', ''),
                        dso.get('ra_deg', 0),
                        dso.get('dec_deg', 0),
                        dso.get('magnitude', 0),
                        size_info,
                        priority,
                        date_added,
                        telescope_id,
                    ))
                    inserted += 1
                conn.commit()

            QMessageBox.information(
                self, "Success",
                f"{inserted} object{'s' if inserted != 1 else ''} added to your target list."
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add objects: {str(e)}")


# --- About Dialog ---
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        # Set minimum size to accommodate content on all platforms
        self.setMinimumSize(450, 420)

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
        desc_label.setStyleSheet(f"font-size: 11pt; color: {COLORS['text_secondary']}; margin: 10px 0px;")
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
            from PySide6.QtCore import QUrl
            from UrlOpener import open_url

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
                    open_url(version_info['github_url'])
            else:
                QMessageBox.information(self, "No Updates",
                    f"You are running the latest version ({version_info['local_version']}).")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Error checking for updates: {str(e)}")


class StayOpenMenu(QMenu):
    """A QMenu that stays open when checkable actions are clicked."""
    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action and action.isCheckable():
            action.trigger()
        else:
            super().mouseReleaseEvent(event)


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
        self._showed_dso_data = None
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

        # Add Filters dropdown button with checkable actions
        filters_button = QPushButton("Filters \u25be")
        filters_menu = StayOpenMenu(self)

        self.action_show_images_only = filters_menu.addAction("Show Only Objects with Images")
        self.action_show_images_only.setCheckable(True)
        self.action_show_images_only.toggled.connect(self._on_show_images_changed)

        self.action_highlight_no_images = filters_menu.addAction("Highlight Objects without Images")
        self.action_highlight_no_images.setCheckable(True)
        self.action_highlight_no_images.toggled.connect(self._on_highlight_no_images_changed)

        self.action_show_no_images_only = filters_menu.addAction("Show Only Objects without Images")
        self.action_show_no_images_only.setCheckable(True)
        self.action_show_no_images_only.toggled.connect(self._on_show_no_images_changed)

        filters_button.setMenu(filters_menu)
        controls_layout.addWidget(filters_button)

        # Add Actions dropdown button
        actions_button = QPushButton("Actions \u25be")
        actions_menu = QMenu(self)
        action_bulk_add = actions_menu.addAction("Add missing objects to target list")
        action_bulk_add.triggered.connect(self._on_bulk_add_missing_objects)
        actions_button.setMenu(actions_menu)
        controls_layout.addWidget(actions_button)

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

        # Start background loading of all objects after a short delay
        QTimer.singleShot(1500, self._start_background_loading)

        # Check for updates on startup if enabled
        QTimer.singleShot(2000, self._check_updates_on_startup)

        # Initialize system tray if enabled
        self._tray_manager: Optional[SystemTrayManager] = None
        self._setup_system_tray_if_enabled()

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

        # Telescopes action
        telescopes_action = QAction("Telescopes", self)
        telescopes_action.setToolTip("Manage telescope configurations")
        telescopes_action.triggered.connect(self._show_telescopes)
        toolbar.addAction(telescopes_action)

        toolbar.addSeparator()

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

        # Weather Forecast action
        weather_action = QAction("Weather", self)
        weather_action.setToolTip("View 7-day weather forecast for astrophotography")
        weather_action.triggered.connect(self._show_weather_forecast)
        toolbar.addAction(weather_action)

        # NINA Dashboard action (only visible if NINA integration is enabled)
        self.nina_dashboard_action = QAction("NINA Dashboard", self)
        self.nina_dashboard_action.setToolTip("View real-time NINA status, imaging, and guiding")
        self.nina_dashboard_action.triggered.connect(self._show_nina_dashboard)
        toolbar.addAction(self.nina_dashboard_action)
        self.nina_dashboard_action.setVisible(NINAIntegration.is_enabled())

        toolbar.addSeparator()

        # DSO Image Gallery action
        gallery_action = QAction("Image Gallery", self)
        gallery_action.setToolTip("Browse all DSO images in a gallery view")
        gallery_action.triggered.connect(self._show_dso_gallery)
        toolbar.addAction(gallery_action)

        # Collage Builder action
        collage_builder_action = QAction("Collage Builder", self)
        collage_builder_action.setToolTip("Create image collages from your DSO photos")
        collage_builder_action.triggered.connect(self._show_collage_builder)
        toolbar.addAction(collage_builder_action)

        toolbar.addSeparator()

        # DSO Visibility Calculator action
        visibility_action = QAction("Visibility Calculator", self)
        visibility_action.setToolTip("Calculate DSO visibility from your location")
        visibility_action.triggered.connect(self._show_dso_visibility)
        toolbar.addAction(visibility_action)

        # Aladin Lite action
        aladin_lite_action = QAction("FOV Simulator", self)
        aladin_lite_action.setToolTip("Open interactive sky atlas with telescope field of view simulator")
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
        # Update NINA Dashboard action visibility based on current settings
        self.nina_dashboard_action.setVisible(NINAIntegration.is_enabled())

    def _check_location_on_startup(self):
        """Check if user has configured their location and prompt if not"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()

                if not row or row[0] is None or row[1] is None:
                    # Location not configured, show dialog
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Location Required")
                    msg.setText("Welcome to Cosmos Collection!")
                    msg.setInformativeText(
                        "Some features require your observer location to work properly:\n\n"
                        "• Best DSO Tonight - Find optimal objects for your location\n"
                        "• Visibility Calculator - Calculate when objects are visible\n"
                        "• Altitude/Azimuth calculations\n\n"
                        "Would you like to set your location now?"
                    )
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg.setDefaultButton(QMessageBox.Yes)

                    if msg.exec() == QMessageBox.Yes:
                        self._show_settings()

        except Exception as e:
            logger.error(f"Error checking location on startup: {str(e)}")

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

    def _show_weather_forecast(self):
        """Show the Weather Forecast window"""
        try:
            from WeatherForecast import WeatherForecastWindow
            if not hasattr(self, 'weather_forecast_window') or not self.weather_forecast_window.isVisible():
                self.weather_forecast_window = WeatherForecastWindow()
            self.weather_forecast_window.show()
            self.weather_forecast_window.raise_()
            self.weather_forecast_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load Weather Forecast: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Weather Forecast: {e}")

    def _show_nina_dashboard(self):
        """Show the NINA Dashboard window"""
        from NINAIntegration import NINAIntegration
        if not NINAIntegration.is_enabled():
            QMessageBox.information(
                self, "NINA Integration",
                "NINA integration is not enabled.\n\n"
                "Please enable it in Settings > NINA Integration."
            )
            return

        try:
            from NINADashboard import NINADashboardWindow
            if not hasattr(self, 'nina_dashboard_window') or not self.nina_dashboard_window.isVisible():
                self.nina_dashboard_window = NINADashboardWindow()
            self.nina_dashboard_window.show()
            self.nina_dashboard_window.raise_()
            self.nina_dashboard_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load NINA Dashboard: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open NINA Dashboard: {e}")

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
        checked = self.action_show_images_only.isChecked()
        if checked:
            self.action_show_no_images_only.blockSignals(True)
            self.action_show_no_images_only.setChecked(False)
            self.action_show_no_images_only.blockSignals(False)
        self.action_highlight_no_images.setEnabled(not checked)
        self.action_show_no_images_only.setEnabled(not checked)
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            checked,
            self._get_selected_type(),
            False
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

    def _on_show_no_images_changed(self, state):
        """Handle show no images only checkbox state change"""
        checked = self.action_show_no_images_only.isChecked()
        if checked:
            self.action_show_images_only.blockSignals(True)
            self.action_show_images_only.setChecked(False)
            self.action_show_images_only.blockSignals(False)
            self.action_highlight_no_images.setEnabled(True)
        self.action_show_images_only.setEnabled(not checked)
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            False,
            self._get_selected_type(),
            checked
        )
        self._update_status()
        self._check_filter_needs_more_data()

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
        self.action_show_images_only.blockSignals(True)
        self.action_highlight_no_images.blockSignals(True)
        self.action_show_no_images_only.blockSignals(True)
        self.type_combo.blockSignals(True)

        self.search_input.clear()
        self.catalog_combo.setCurrentIndex(0)
        self.action_show_images_only.setChecked(False)
        self.action_highlight_no_images.setChecked(False)
        self.action_show_no_images_only.setChecked(False)
        self.type_combo.setCurrentIndex(0)

        # Unblock signals
        self.search_input.blockSignals(False)
        self.catalog_combo.blockSignals(False)
        self.action_show_images_only.blockSignals(False)
        self.action_highlight_no_images.blockSignals(False)
        self.action_show_no_images_only.blockSignals(False)
        self.type_combo.blockSignals(False)

        logger.debug("Calling filter_data with all filters cleared")
        # Manually trigger filter update once
        self.model.filter_data("", None, False, None, False)

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
            self.action_show_images_only.isChecked(),
            self._get_selected_type(),
            self.action_show_no_images_only.isChecked()
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
            not self.action_show_images_only.isChecked() and
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
            self.action_show_images_only.isChecked(),
            self._get_selected_type(),
            self.action_show_no_images_only.isChecked()
        )
        self._update_status()
        # Check if we need more data for this filter
        self._check_filter_needs_more_data()

    def _on_type_changed(self, type_text):
        """Handle DSO type selection changes"""
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.action_show_images_only.isChecked(),
            self._get_selected_type(),
            self.action_show_no_images_only.isChecked()
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

    def _start_background_loading(self):
        """Start background loading of all objects in chunks"""
        if hasattr(self.model, 'load_offset') and hasattr(self.model, 'total_count'):
            if self.model.load_offset < self.model.total_count:
                logger.info(f"Starting background loading: {self.model.load_offset}/{self.model.total_count} objects loaded")
                self.model.load_more_data()
            else:
                logger.info(f"All {self.model.total_count} objects already loaded")

    def _check_updates_on_startup(self):
        """Silently check for updates on startup if enabled in settings"""
        try:
            # Check if the setting is enabled
            settings = QSettings("CosmosCollection", "CosmosCollection")
            check_updates = settings.value("check_updates_on_startup", True, type=bool)

            if not check_updates:
                logger.debug("Update check on startup is disabled")
                return

            logger.debug("Checking for updates on startup")

            from version import version_manager
            from PySide6.QtWidgets import QMessageBox
            from UrlOpener import open_url

            # Get version info
            version_info = version_manager.get_version_info()

            # Only show message if update is available
            if version_info.get('github_available') and version_info.get('update_available'):
                logger.info(f"Update available: {version_info.get('github_version')}")
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

                if msg.exec() == QMessageBox.Yes and version_info.get('github_url'):
                    open_url(version_info['github_url'])
            else:
                logger.debug("No updates available or unable to check")

        except Exception as e:
            # Silently fail - don't bother user on startup with error messages
            logger.error(f"Error checking for updates on startup: {str(e)}")

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
                        INSERT INTO main.cataloguenr (dsodetailid, catalogue, designation)
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
        detail_window = DSODetailWindow(data)
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
                self.action_show_images_only.isChecked(),
                self._get_selected_type(),
                self.action_show_no_images_only.isChecked()
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

        # Add menu actions
        details_action = context_menu.addAction("View DSO Details")
        details_action.triggered.connect(lambda: self._context_view_details(row))

        visibility_action = context_menu.addAction("Visibility Calculator")
        visibility_action.triggered.connect(lambda: self._context_open_visibility(row))

        aladin_action = context_menu.addAction("FOV Simulator")
        aladin_action.triggered.connect(lambda: self._context_open_aladin(row))

        if NINAIntegration.is_enabled():
            context_menu.addSeparator()
            nina_menu = context_menu.addMenu("NINA")
            nina_action = nina_menu.addAction("Send to Framing Assistant")
            nina_action.triggered.connect(lambda: self._context_send_to_nina(row))
            slew_action = nina_menu.addAction("Slew to Target")
            slew_action.triggered.connect(lambda: self._context_slew_to_target(row))

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

            # Create data dictionary similar to what DSODetailWindow creates
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

    def _on_bulk_add_missing_objects(self):
        """Open bulk-add dialog for all DSOs without images not yet in the target list"""
        if not hasattr(self, 'model') or not self.model.dso_data:
            QMessageBox.information(self, "No Data", "DSO data not loaded yet.")
            return

        missing = [e for e in self.model.dso_data if e.get('image_count', 0) == 0]

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT LOWER(name) FROM usertargetlist")
                existing = {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error reading target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to read target list: {str(e)}")
            return

        to_add = [e for e in missing if e.get('name', '').lower() not in existing]

        if not to_add:
            QMessageBox.information(
                self, "Nothing to Add",
                "All objects without images are already in your target list."
            )
            return

        dialog = BulkAddToTargetDialog(to_add, parent=self)
        if dialog.exec() == QDialog.Accepted:
            if hasattr(self, 'target_list_window') and self.target_list_window.isVisible():
                self.target_list_window._load_targets()

    def _context_send_to_nina(self, row):
        """Send DSO coordinates to NINA Framing Assistant"""
        entry = self.model.filtered_data[row]
        NINAIntegration.send_to_framing_assistant(
            entry.get("ra_deg"), entry.get("dec_deg"),
            entry.get("name", "Unknown"), self
        )

    def _context_slew_to_target(self, row):
        """Slew mount to DSO coordinates"""
        entry = self.model.filtered_data[row]
        NINAIntegration.slew_to_coordinates(
            entry.get("ra_deg"), entry.get("dec_deg"),
            entry.get("name", "Unknown"), self
        )

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
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # If minimize to tray is enabled, minimize instead of closing
        if settings.value("minimize_to_tray", True, type=bool):
            if self._tray_manager and self._tray_manager.is_available:
                event.ignore()
                self._minimize_to_tray()
                return

        # Clean up tray manager
        if self._tray_manager:
            self._tray_manager.cleanup()

        self.db_manager.close()
        super().closeEvent(event)

    def changeEvent(self, event):
        """Handle window state changes (minimize, etc.)"""
        if event.type() == QEvent.WindowStateChange:
            settings = QSettings("CosmosCollection", "CosmosCollection")
            if settings.value("minimize_to_tray", True, type=bool):
                if self.windowState() & Qt.WindowMinimized:
                    if self._tray_manager and self._tray_manager.is_available:
                        # Use a timer to allow the minimize animation to complete
                        QTimer.singleShot(100, self._minimize_to_tray)
        super().changeEvent(event)

    def _setup_system_tray_if_enabled(self):
        """Initialize system tray if the setting is enabled"""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        if not settings.value("minimize_to_tray", True, type=bool):
            return

        try:
            self._tray_manager = SystemTrayManager(self)

            # Get the application icon
            app_icon = QApplication.instance().windowIcon()
            if app_icon.isNull():
                # Fallback to loading icon directly
                icon_path = os.path.join(APP_DIR, 'images', 'CosmosCollection.png')
                app_icon = QIcon(icon_path)

            if self._tray_manager.setup(app_icon):
                # Connect signals
                self._tray_manager.restore_requested.connect(self._restore_from_tray)
                self._tray_manager.quit_requested.connect(self._quit_application)
                self._tray_manager.action_triggered.connect(self._handle_tray_action)

                # Register for weather updates
                self._register_weather_callback()

                logger.debug("System tray initialized")
            else:
                logger.warning("Failed to setup system tray")
                self._tray_manager = None

        except Exception as e:
            logger.error(f"Error setting up system tray: {e}", exc_info=True)
            self._tray_manager = None

    def _register_weather_callback(self):
        """Register callback to update tray tooltip when weather data changes"""
        try:
            from WeatherForecast import WeatherCache
            cache = WeatherCache()
            cache.add_update_callback(self._on_weather_updated)

            # Also update with existing cached data if available
            existing_data = cache.get_cached_data()
            if existing_data:
                self._on_weather_updated(existing_data)
        except Exception as e:
            logger.debug(f"Could not register weather callback: {e}")

    def _on_weather_updated(self, weather_data):
        """Called when weather data is updated"""
        if self._tray_manager:
            self._tray_manager.update_tooltip(weather_data)

    def _minimize_to_tray(self):
        """Hide window to system tray"""
        if self._tray_manager and self._tray_manager.is_available:
            self.hide()
            self._tray_manager.show()
            # Prevent Qt from quitting when child windows (e.g. Weather) are closed
            QApplication.instance().setQuitOnLastWindowClosed(False)
            # Start background weather refresh if auto-refresh is enabled
            self._start_tray_weather_refresh()
            # Start hourly update check
            self._start_tray_update_check()
            logger.debug("Window minimized to tray")

    def _restore_from_tray(self):
        """Restore window from system tray"""
        # Stop background timers
        self._stop_tray_weather_refresh()
        self._stop_tray_update_check()

        if self._tray_manager:
            self._tray_manager.hide()

        # Restore default quit behavior now that the main window is visible
        QApplication.instance().setQuitOnLastWindowClosed(True)

        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()
        logger.debug("Window restored from tray")

    def _start_tray_weather_refresh(self):
        """Start background weather refresh timer when in tray"""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Check if auto-refresh is enabled in weather settings
        if not settings.value("weather_auto_refresh_enabled", False, type=bool):
            return

        # Get the refresh interval (index into combo: 0=15min, 1=30min, 2=1hr, 3=2hr, 4=4hr)
        interval_index = settings.value("weather_auto_refresh_interval", 2, type=int)
        interval_minutes = [15, 30, 60, 120, 240][min(interval_index, 4)]

        # Create timer if needed
        if not hasattr(self, '_tray_weather_timer') or self._tray_weather_timer is None:
            self._tray_weather_timer = QTimer(self)
            self._tray_weather_timer.timeout.connect(self._fetch_weather_for_tray)

        # Start the timer
        self._tray_weather_timer.start(interval_minutes * 60 * 1000)
        logger.debug(f"Started tray weather refresh timer: {interval_minutes} minutes")

        # Also do an immediate refresh if cache is stale
        self._fetch_weather_for_tray()

    def _stop_tray_weather_refresh(self):
        """Stop background weather refresh timer"""
        if hasattr(self, '_tray_weather_timer') and self._tray_weather_timer is not None:
            self._tray_weather_timer.stop()
            logger.debug("Stopped tray weather refresh timer")

    def _start_tray_update_check(self):
        """Start hourly update check timer when in tray"""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Only check if update checking is enabled
        if not settings.value("check_updates_on_startup", True, type=bool):
            return

        # Create timer if needed
        if not hasattr(self, '_tray_update_timer') or self._tray_update_timer is None:
            self._tray_update_timer = QTimer(self)
            self._tray_update_timer.timeout.connect(self._check_updates_for_tray)

        # Start the timer - check every hour (60 minutes)
        self._tray_update_timer.start(60 * 60 * 1000)
        logger.debug("Started tray update check timer: 60 minutes")

        # Also do an immediate check
        QTimer.singleShot(5000, self._check_updates_for_tray)

    def _stop_tray_update_check(self):
        """Stop background update check timer"""
        if hasattr(self, '_tray_update_timer') and self._tray_update_timer is not None:
            self._tray_update_timer.stop()
            logger.debug("Stopped tray update check timer")

    def _check_updates_for_tray(self):
        """Check for updates and show tray notification if available"""
        try:
            from version import version_manager

            # Get version info
            version_info = version_manager.get_version_info()

            # Show tray notification if update is available
            if version_info.get('github_available') and version_info.get('update_available'):
                logger.info(f"Update available (tray check): {version_info.get('github_version')}")

                if self._tray_manager and self._tray_manager._tray_icon:
                    self._tray_manager._tray_icon.showMessage(
                        "Update Available",
                        f"Cosmos Collection {version_info['github_version']} is available.\n"
                        f"You are running {version_info['local_version']}.",
                        QSystemTrayIcon.Information,
                        10000  # Show for 10 seconds
                    )
            else:
                logger.debug("Tray update check: no updates available")

        except Exception as e:
            logger.debug(f"Error checking for updates in tray: {e}")

    def _fetch_weather_for_tray(self):
        """Fetch weather data in background for tray tooltip"""
        try:
            from WeatherForecast import WeatherCache, WeatherWorker

            # Get user location
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT location_lat, location_lon, timezone
                    FROM usersettings WHERE is_active = 1
                """)
                row = cursor.fetchone()

            if not row or row[0] is None or row[1] is None:
                logger.debug("No location configured for tray weather refresh")
                return

            lat, lon, timezone = row

            # Check if cache is still valid (don't re-fetch if recently updated)
            cache = WeatherCache()
            cached_data = cache.get(lat, lon)
            if cached_data:
                logger.debug("Weather cache still valid, skipping tray refresh")
                return

            # Create worker to fetch in background
            self._tray_weather_worker = WeatherWorker(lat, lon, timezone)
            self._tray_weather_worker.weather_loaded.connect(self._on_tray_weather_loaded)
            self._tray_weather_worker.error_occurred.connect(
                lambda e: logger.debug(f"Tray weather fetch error: {e}")
            )
            self._tray_weather_worker.start()
            logger.debug("Started background weather fetch for tray")

        except Exception as e:
            logger.debug(f"Error fetching weather for tray: {e}")

    def _on_tray_weather_loaded(self, weather_data):
        """Handle weather data loaded for tray"""
        try:
            from WeatherForecast import WeatherCache

            # Get location to store in cache
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT location_lat, location_lon
                    FROM usersettings WHERE is_active = 1
                """)
                row = cursor.fetchone()

            if row and row[0] is not None and row[1] is not None:
                lat, lon = row
                # Store in cache (this will trigger the callback to update tooltip)
                cache = WeatherCache()
                cache.set(lat, lon, weather_data)
                logger.debug("Tray weather data cached and tooltip updated")

        except Exception as e:
            logger.debug(f"Error caching tray weather data: {e}")

    def _handle_tray_action(self, action_name: str):
        """Handle quick actions from tray menu - opens tools without restoring main window"""
        if action_name == "best_dso":
            self._show_best_dso_tonight()
        elif action_name == "target_list":
            self._show_target_list()
        elif action_name == "weather":
            self._show_weather_forecast()
        elif action_name == "gallery":
            self._show_dso_gallery()
        elif action_name == "nina_dashboard":
            self._show_nina_dashboard()

    def _quit_application(self):
        """Quit the application from tray menu"""
        # Stop background timers
        self._stop_tray_weather_refresh()
        self._stop_tray_update_check()

        # Clean up tray manager
        if self._tray_manager:
            self._tray_manager.cleanup()
            self._tray_manager = None

        # Close database
        self.db_manager.close()

        # Quit the application
        QApplication.instance().quit()


# --- Command Line Interface ---
def parse_cli_arguments():
    """Parse command line arguments for CLI operations"""
    # Determine the program name for help text
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        prog_name = os.path.basename(sys.executable)
    else:
        # Running from source
        prog_name = "python main.py"

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description='Cosmos Collection - Deep Sky Object Image Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Add an image to a DSO
  {prog_name} --add-image --dso "M31" --image "/path/to/image.jpg"

  # Add an image with metadata
  {prog_name} --add-image --dso "NGC 7000" --image "/path/to/image.tif" \\
      --equipment "Telescope: AT72ED, Camera: ASI2600" \\
      --integration "3600" --date "2024-01-15" --notes "First light!"

  # List all DSOs in the database
  {prog_name} --list-dsos

  # Search for a DSO
  {prog_name} --search-dso "M31"
"""
    )

    # CLI operation flags
    parser.add_argument('--add-image', action='store_true',
                        help='Add an image to a DSO in the database')
    parser.add_argument('--list-dsos', action='store_true',
                        help='List all DSOs in the database')
    parser.add_argument('--search-dso', type=str, metavar='NAME',
                        help='Search for a DSO by name')

    # Image addition arguments
    parser.add_argument('--dso', type=str, metavar='NAME',
                        help='DSO name (e.g., M31, NGC 7000, IC 1396)')
    parser.add_argument('--image', type=str, metavar='PATH',
                        help='Path to the image file')
    parser.add_argument('--equipment', type=str, default='',
                        help='Equipment used (optional)')
    parser.add_argument('--integration', type=str, default='',
                        help='Integration time in seconds (optional)')
    parser.add_argument('--date', type=str, default='',
                        help='Date taken YYYY-MM-DD (optional)')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes about the image (optional)')
    parser.add_argument('--set-favorite', action='store_true',
                        help='Set this image as the favorite for the DSO')

    # Only parse known args to avoid conflicts with Qt arguments
    args, unknown = parser.parse_known_args()

    return args, unknown


def find_dso_by_name(dso_name: str) -> Optional[Dict]:
    """Find a DSO in the database by name or catalog number"""
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Normalize the search name
            search_name = dso_name.strip().upper()

            # Try matching via cataloguenr table (e.g., M31, NGC 7000)
            import re
            match = re.match(r'^([A-Z]+)\s*(\d+)([A-Z]?)$', search_name)
            if match:
                catalog = match.group(1)
                designation = match.group(2)
                suffix = match.group(3) or ''

                cursor.execute("""
                    SELECT d.id,
                           c.catalogue || ' ' || c.designation as name,
                           d.dsotype,
                           d.constellation
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE UPPER(c.catalogue) = ? AND c.designation = ?
                    LIMIT 1
                """, (catalog, designation + suffix))

                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'commonnames': None,
                        'type': row[2],
                        'constellation': row[3]
                    }

            # Try partial match on catalog entries
            cursor.execute("""
                SELECT d.id,
                       c.catalogue || ' ' || c.designation as name,
                       d.dsotype,
                       d.constellation
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE UPPER(c.catalogue || c.designation) LIKE ?
                   OR UPPER(c.catalogue || ' ' || c.designation) LIKE ?
                LIMIT 1
            """, (f'%{search_name}%', f'%{search_name}%'))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'commonnames': None,
                    'type': row[2],
                    'constellation': row[3]
                }

            return None

    except Exception as e:
        print(f"Error searching for DSO: {e}")
        return None


def add_image_cli(args) -> bool:
    """Add an image to the database via command line"""
    # Validate required arguments
    if not args.dso:
        print("Error: --dso argument is required")
        return False

    if not args.image:
        print("Error: --image argument is required")
        return False

    # Check if image file exists
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False

    # Find the DSO in the database
    print(f"Searching for DSO: {args.dso}")
    dso = find_dso_by_name(args.dso)

    if not dso:
        print(f"Error: DSO '{args.dso}' not found in database")
        print("Use --search-dso to find available DSOs or --list-dsos to see all")
        return False

    print(f"Found DSO: {dso['name']}", end="")
    if dso['commonnames']:
        print(f" ({dso['commonnames']})", end="")
    print(f" - {dso['type']} in {dso['constellation']}")

    # Add the image to the database
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO userimages (
                    dsodetailid, image_path, integration_time,
                    equipment, date_taken, notes, created_date
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                dso['id'],
                image_path,
                args.integration,
                args.equipment,
                args.date,
                args.notes
            ))

            image_id = cursor.lastrowid

            # Set as favorite if requested
            if args.set_favorite:
                cursor.execute("""
                    UPDATE dsodetail SET favourite_image = ? WHERE id = ?
                """, (image_id, dso['id']))
                print(f"Set as favorite image for {dso['name']}")

            conn.commit()

        print(f"Successfully added image to {dso['name']}")
        print(f"  Image path: {image_path}")
        if args.equipment:
            print(f"  Equipment: {args.equipment}")
        if args.integration:
            print(f"  Integration: {args.integration}s")
        if args.date:
            print(f"  Date: {args.date}")

        return True

    except Exception as e:
        print(f"Error adding image to database: {e}")
        return False


def list_dsos_cli() -> bool:
    """List all DSOs in the database"""
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                           ORDER BY CASE c.catalogue
                               WHEN 'M' THEN 1
                               WHEN 'NGC' THEN 2
                               WHEN 'IC' THEN 3
                               ELSE 4
                           END) as name,
                       d.dsotype,
                       d.constellation,
                       (SELECT COUNT(*) FROM userimages u WHERE u.dsodetailid = d.id) as image_count
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                GROUP BY d.id
                ORDER BY
                    CASE
                        WHEN MIN(c.catalogue) = 'M' THEN 1
                        WHEN MIN(c.catalogue) = 'NGC' THEN 2
                        WHEN MIN(c.catalogue) = 'IC' THEN 3
                        ELSE 4
                    END,
                    MIN(CAST(c.designation AS INTEGER))
            """)

            rows = cursor.fetchall()

            print(f"Found {len(rows)} DSOs in database:\n")
            print(f"{'Name':<30} {'Type':<20} {'Constellation':<15} {'Images':<8}")
            print("-" * 80)

            for row in rows:
                name = (row[0] or '')[:29]
                dso_type = (row[1] or '')[:19]
                constellation = (row[2] or '')[:14]
                images = row[3] or 0

                print(f"{name:<30} {dso_type:<20} {constellation:<15} {images:<8}")

            return True

    except Exception as e:
        print(f"Error listing DSOs: {e}")
        return False


def search_dso_cli(search_term: str) -> bool:
    """Search for DSOs matching a term"""
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            search_pattern = f'%{search_term.upper()}%'

            cursor.execute("""
                SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                           ORDER BY CASE c.catalogue
                               WHEN 'M' THEN 1
                               WHEN 'NGC' THEN 2
                               WHEN 'IC' THEN 3
                               ELSE 4
                           END) as name,
                       d.dsotype,
                       d.constellation,
                       (SELECT COUNT(*) FROM userimages u WHERE u.dsodetailid = d.id) as image_count
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE UPPER(c.catalogue || c.designation) LIKE ?
                   OR UPPER(c.catalogue || ' ' || c.designation) LIKE ?
                GROUP BY d.id
                ORDER BY
                    CASE
                        WHEN MIN(c.catalogue) = 'M' THEN 1
                        WHEN MIN(c.catalogue) = 'NGC' THEN 2
                        WHEN MIN(c.catalogue) = 'IC' THEN 3
                        ELSE 4
                    END,
                    MIN(CAST(c.designation AS INTEGER))
                LIMIT 50
            """, (search_pattern, search_pattern))

            rows = cursor.fetchall()

            if not rows:
                print(f"No DSOs found matching '{search_term}'")
                return True

            print(f"Found {len(rows)} DSOs matching '{search_term}':\n")
            print(f"{'Name':<30} {'Type':<20} {'Constellation':<15} {'Images':<8}")
            print("-" * 80)

            for row in rows:
                name = (row[0] or '')[:29]
                dso_type = (row[1] or '')[:19]
                constellation = (row[2] or '')[:14]
                images = row[3] or 0

                print(f"{name:<30} {dso_type:<20} {constellation:<15} {images:<8}")

            return True

    except Exception as e:
        print(f"Error searching DSOs: {e}")
        return False


def run_cli_command(args) -> Optional[bool]:
    """
    Run a CLI command if specified.
    Returns True/False for success/failure, or None if no CLI command was requested.
    """
    if args.add_image:
        return add_image_cli(args)

    if args.list_dsos:
        return list_dsos_cli()

    if args.search_dso:
        return search_dso_cli(args.search_dso)

    # No CLI command requested
    return None


# --- Entry Point ---
if __name__ == "__main__":
    # Parse CLI arguments first
    cli_args, remaining_args = parse_cli_arguments()

    # Check if a CLI command was requested
    cli_result = run_cli_command(cli_args)
    if cli_result is not None:
        # A CLI command was run, exit with appropriate code
        sys.exit(0 if cli_result else 1)

    # No CLI command - continue with GUI startup
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

    # Configure file logging if enabled (must be after QApplication for QSettings to work on Windows)
    try:
        settings = QSettings("CosmosCollection", "CosmosCollection")
        if settings.value("enable_logfile", False, type=bool):
            from logging.handlers import RotatingFileHandler
            log_dir = ResourceManager.get_data_dir()
            log_file_path = os.path.join(log_dir, "CosmosCollection.log")
            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file_path}")
    except Exception as e:
        logger.warning(f"Could not configure file logging: {e}")
        QMessageBox.warning(None, "Logging Error",
            f"Could not create log file:\n{e}\n\n"
            "Logging to file has been disabled for this session.")

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

    # Apply global dark theme
    apply_theme(app)

    # Initialize database manager and get data
    db_manager = DatabaseManager()

    # Global reference to hold the initial data loader
    initial_loader = None
    window = None

    def on_initial_data_loaded(dso_data, catalogs, total_count):
        """Handle initial data loaded from background thread"""
        global window
        try:
            logger.debug(f"Initial data loaded in background: {len(dso_data)} DSOs")

            # Create and show the main window with loaded data
            window = MainWindow(dso_data, catalogs, total_count)
            window.show()

            # Check if location is configured after window is shown
            QTimer.singleShot(500, window._check_location_on_startup)

        except Exception as e:
            logger.error(f"Error creating main window: {e}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
            sys.exit(1)

    def on_initial_load_failed(error_msg):
        """Handle initial data load failure"""
        from PySide6.QtWidgets import QMessageBox
        logger.error(f"Failed to load initial data: {error_msg}")
        QMessageBox.critical(None, "Error", f"Failed to load DSO data from database:\n{error_msg}")
        sys.exit(1)

    try:
        # Check if catalogs directory exists, create if needed
        catalogs_dir = os.path.join(APP_DIR, 'catalogs')
        if not os.path.exists(catalogs_dir):
            os.makedirs(catalogs_dir)

        # Start background thread to load initial data
        logger.debug("Starting background thread to load initial DSO data")
        initial_loader = InitialDataLoadWorker()
        initial_loader.data_loaded.connect(on_initial_data_loaded)
        initial_loader.load_failed.connect(on_initial_load_failed)
        initial_loader.start()

        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
        sys.exit(1)
    finally:
        db_manager.close()