#!/usr/bin/env python3
"""
DSO Image Gallery
Displays all DSO objects with images in a responsive grid gallery format
"""

import sys
import os
import re
import logging
from datetime import datetime
from PySide6.QtCore import Qt, Signal, QTimer, QThreadPool, QRunnable, QObject
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QGroupBox,
                               QMessageBox, QScrollArea, QComboBox, QLineEdit,
                               QFrame, QGridLayout, QMenu, QApplication,
                               QDialog, QFileDialog, QFormLayout, QDialogButtonBox,
                               QCompleter, QSlider, QProgressDialog, QPlainTextEdit,
                               QSizePolicy)
from PySide6.QtCore import QSettings
from PySide6.QtGui import QPixmap

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from ImageLoader import load_astro_pixmap

logger = logging.getLogger(__name__)

# Image extensions supported for DSO images throughout the gallery
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit', '.fts', '.xisf'}


class ThumbnailCache:
    """Cache for storing generated thumbnails to avoid regeneration"""

    def __init__(self, max_size=200):
        self._cache = {}  # image_path -> QPixmap
        self._max_size = max_size
        self._access_order = []  # Track access order for LRU eviction

    def get(self, image_path):
        """Get cached thumbnail for image path"""
        if image_path in self._cache:
            # Move to end (most recently used)
            if image_path in self._access_order:
                self._access_order.remove(image_path)
            self._access_order.append(image_path)
            return self._cache[image_path]
        return None

    def put(self, image_path, pixmap):
        """Store thumbnail in cache"""
        if image_path in self._cache:
            # Update existing entry
            if image_path in self._access_order:
                self._access_order.remove(image_path)
        elif len(self._cache) >= self._max_size:
            # Remove least recently used item
            if self._access_order:
                lru_path = self._access_order.pop(0)
                if lru_path in self._cache:
                    del self._cache[lru_path]

        self._cache[image_path] = pixmap
        self._access_order.append(image_path)

    def clear(self):
        """Clear all cached thumbnails"""
        self._cache.clear()
        self._access_order.clear()


class ThumbnailSignals(QObject):
    """Signals for ThumbnailRunnable (QRunnable doesn't support signals directly)"""
    thumbnail_ready = Signal(object, QPixmap)  # card, pixmap
    thumbnail_error = Signal(object, str)      # card, error_message


class ThumbnailRunnable(QRunnable):
    """Runnable task for generating a single thumbnail in a thread pool"""

    # Map thumbnail sizes to size names for disk cache filenames
    SIZE_NAMES = {
        100: 'Small',
        150: 'Medium',
        300: 'Large',
        500: 'ExtraLarge'
    }

    def __init__(self, card, image_path, cache, signals, cancelled_flag, thumbnail_size=150):
        """
        Initialize thumbnail runnable

        Args:
            card: GalleryCard instance to update
            image_path: Path to image file
            cache: ThumbnailCache instance
            signals: ThumbnailSignals instance for emitting signals
            cancelled_flag: List with single boolean for cancellation check
            thumbnail_size: Size of thumbnail (width and height in pixels)
        """
        super().__init__()
        self.card = card
        self.image_path = image_path
        self.cache = cache
        self.signals = signals
        self.cancelled_flag = cancelled_flag
        self.thumbnail_size = thumbnail_size

    def _get_disk_cache_path(self):
        """Get the path for the disk-cached thumbnail file"""
        directory = os.path.dirname(self.image_path)
        basename = os.path.basename(self.image_path)
        name_without_ext = os.path.splitext(basename)[0]
        size_name = self.SIZE_NAMES.get(self.thumbnail_size, f'{self.thumbnail_size}px')
        cache_filename = f"{name_without_ext}_{size_name}_Thumbnail.jpg"
        return os.path.join(directory, cache_filename)

    def _is_disk_cache_valid(self, cache_path):
        """Check if disk cache file exists and is newer than the original image"""
        if not os.path.exists(cache_path):
            return False
        try:
            cache_mtime = os.path.getmtime(cache_path)
            original_mtime = os.path.getmtime(self.image_path)
            return cache_mtime >= original_mtime
        except OSError:
            return False

    def _load_from_disk_cache(self, cache_path):
        """Load thumbnail from disk cache"""
        try:
            pixmap = QPixmap(cache_path)
            if not pixmap.isNull():
                return pixmap
        except Exception:
            pass
        return None

    def _save_to_disk_cache(self, pixmap, cache_path):
        """Save thumbnail to disk cache as JPEG at 85% quality"""
        try:
            pixmap.save(cache_path, "JPEG", 85)
        except Exception:
            pass  # Silently fail if we can't save cache

    def run(self):
        """Generate thumbnail for single image"""
        # Check if cancelled before starting
        if self.cancelled_flag[0]:
            return

        from PySide6.QtGui import QImageReader

        try:
            # Check memory cache first
            if self.cache:
                cached_pixmap = self.cache.get(self.image_path)
                if cached_pixmap:
                    self.signals.thumbnail_ready.emit(self.card, cached_pixmap)
                    return

            # Check if cancelled
            if self.cancelled_flag[0]:
                return

            # Check if disk caching is enabled
            settings = QSettings("CosmosCollection", "CosmosCollection")
            disk_cache_enabled = settings.value("cache_thumbnails_to_disk", True, type=bool)
            disk_cache_path = self._get_disk_cache_path() if disk_cache_enabled else None

            # Try loading from disk cache if enabled and valid
            if disk_cache_enabled and self._is_disk_cache_valid(disk_cache_path):
                disk_pixmap = self._load_from_disk_cache(disk_cache_path)
                if disk_pixmap and not disk_pixmap.isNull():
                    # Store in memory cache too
                    if self.cache:
                        self.cache.put(self.image_path, disk_pixmap)
                    self.signals.thumbnail_ready.emit(self.card, disk_pixmap)
                    return

            # Check if cancelled
            if self.cancelled_flag[0]:
                return

            if os.path.exists(self.image_path):
                # Check file size
                file_size = os.path.getsize(self.image_path)
                if file_size == 0:
                    self.signals.thumbnail_error.emit(self.card, "Empty File")
                    return

                # Get file extension
                _, ext = os.path.splitext(self.image_path.lower())

                pixmap = None

                # Handle FITS/XISF files
                if ext in ['.fits', '.fit', '.fts', '.xisf']:
                    pixmap = load_astro_pixmap(self.image_path)
                    if pixmap is None:
                        self.signals.thumbnail_error.emit(self.card, "Image Load Error")
                        return
                else:
                    # Load regular image formats
                    QImageReader.setAllocationLimit(512)

                    # Try standard QPixmap loading
                    pixmap = QPixmap(self.image_path)

                    # If failed, try QImageReader
                    if pixmap.isNull():
                        try:
                            reader = QImageReader(self.image_path)
                            if reader.canRead():
                                # Set explicit format
                                if ext in ['.jpg', '.jpeg']:
                                    reader.setFormat(b"JPEG")
                                elif ext == '.png':
                                    reader.setFormat(b"PNG")
                                elif ext in ['.tiff', '.tif']:
                                    reader.setFormat(b"TIFF")

                                image = reader.read()
                                if not image.isNull():
                                    pixmap = QPixmap.fromImage(image)
                        except Exception:
                            pass

                # Check if cancelled before emitting
                if self.cancelled_flag[0]:
                    return

                if pixmap and not pixmap.isNull():
                    # Scale to thumbnail size (gallery card size)
                    scaled_pixmap = pixmap.scaled(self.thumbnail_size, self.thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    # Cache the thumbnail in memory
                    if self.cache:
                        self.cache.put(self.image_path, scaled_pixmap)

                    # Save to disk cache if enabled
                    if disk_cache_enabled and disk_cache_path:
                        self._save_to_disk_cache(scaled_pixmap, disk_cache_path)

                    self.signals.thumbnail_ready.emit(self.card, scaled_pixmap)
                else:
                    error_msg = f"Load Error"
                    self.signals.thumbnail_error.emit(self.card, error_msg)
            else:
                self.signals.thumbnail_error.emit(self.card, "File Not Found")

        except Exception as e:
            self.signals.thumbnail_error.emit(self.card, f"Error: {str(e)[:20]}")


class DataLoaderSignals(QObject):
    """Signals for DataLoaderRunnable"""
    data_loaded = Signal(list)  # Emits list of loaded items
    load_error = Signal(str)    # Emits error message


class DataLoaderRunnable(QRunnable):
    """Runnable task for loading gallery data in background"""

    def __init__(self, signals):
        """
        Initialize data loader runnable

        Args:
            signals: DataLoaderSignals instance for emitting signals
        """
        super().__init__()
        self.signals = signals

    def _get_friendly_type_name(self, dso_type):
        """Convert DSO type code to user-friendly name"""
        type_mapping = {
            "GALXY": "Galaxy",
            "DRKNB": "Dark Nebula",
            "OPNCL": "Open Cluster",
            "PLNNB": "Planetary Nebula",
            "BRTNB": "Bright Nebula",
            "SNREM": "Supernova Remnant",
            "GALCL": "Galaxy Cluster",
            "GLOCL": "Globular Cluster",
            "CL+NB": "Cluster + Nebula",
            "GX+DN": "Galaxy + Dark Nebula",
            "ASTER": "Asterism",
            "2STAR": "Double Star",
            "3STAR": "Triple Star",
            "4STAR": "Quadruple Star",
            "1STAR": "Single Star",
            "QUASR": "Quasar",
            "NONEX": "Non-existent",
            "LMCCN": "LMC Cluster/Nebula",
            "LMCDN": "LMC Dark Nebula",
            "LMCGC": "LMC Globular Cluster",
            "LMCOC": "LMC Open Cluster",
            "SMCCN": "SMC Cluster/Nebula",
            "SMCDN": "SMC Dark Nebula",
            "SMCGC": "SMC Globular Cluster",
            "SMCOC": "SMC Open Cluster"
        }
        return type_mapping.get(dso_type, dso_type)

    def run(self):
        """Load gallery data from database"""
        import sqlite3
        from ResourceManager import ResourceManager

        try:
            # Create new SQLite connection in this thread (DatabaseManager is a singleton)
            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            from ResourceManager import attach_update_catalogs
            attach_update_catalogs(conn)

            # Ensure created_date column exists (migration for older databases)
            # Do this before setting row_factory
            # Note: ALTER TABLE cannot use CURRENT_TIMESTAMP as default, so we use NULL
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(userimages)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'created_date' not in columns:
                cursor.execute("ALTER TABLE userimages ADD COLUMN created_date TEXT")
                conn.commit()

            conn.row_factory = sqlite3.Row

            try:
                cursor = conn.cursor()

                # One row per image (not per DSO) so every attached image gets its own card
                query = """
                SELECT
                    d.id as dsodetailid,
                    ui.id as imageid,
                    ui.image_path,
                    ui.equipment,
                    ui.is_favorite,
                    d.dsotype,
                    d.constellation,
                    GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                        ORDER BY
                            CASE c.catalogue
                                WHEN 'M' THEN 1
                                WHEN 'NGC' THEN 2
                                WHEN 'IC' THEN 3
                                ELSE 4
                            END, c.designation) as name,
                    ui.created_date,
                    d.ra,
                    d.dec
                FROM userimages ui
                INNER JOIN dsodetail d ON d.id = ui.dsodetailid
                INNER JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE ui.image_path IS NOT NULL AND ui.image_path != ''
                GROUP BY ui.id
                ORDER BY name
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                # Convert rows to dictionaries
                items = []
                for row in rows:
                    item = {
                        'dsodetailid': row[0],
                        'imageid': row[1],
                        'image_path': row[2],
                        'equipment': row[3] or '',
                        'is_favorite': row[4],
                        'dsotype': row[5] or '',
                        'constellation': row[6] or '',
                        'name': row[7] or 'Unknown',
                        'friendly_type': self._get_friendly_type_name(row[5] or ''),
                        'created_date': row[8] or '',
                        'ra_deg': row[9],
                        'dec_deg': row[10]
                    }
                    items.append(item)

                # Emit success signal with loaded data
                self.signals.data_loaded.emit(items)

            finally:
                # Close the connection
                conn.close()

        except Exception as e:
            # Emit error signal
            self.signals.load_error.emit(str(e))


def _load_preview_pixmap(image_path, max_dim=200):
    """Load a small preview pixmap for an image path, including FITS/XISF files.

    Returns None if the file can't be read/decoded (e.g. unsupported format).
    """
    if not image_path or not os.path.exists(image_path):
        return None

    _, ext = os.path.splitext(image_path.lower())

    if ext in ('.fits', '.fit', '.fts', '.xisf'):
        return load_astro_pixmap(image_path, max_dim=max_dim)

    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        return None

    return pixmap.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class AddImageDialog(WindowPositionMixin, QDialog):
    """Dialog for adding a new image to a DSO"""

    WINDOW_POSITION_KEY = "AddImageDialog"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Image to DSO")
        self.selected_file = None
        self.dso_data = []  # List of (dsodetailid, name) tuples, index-aligned with dso_combo
        self._dso_auto_selected = False
        self._capture_header = None  # FITS/XISF header dict for the selected file, or None
        self._telescope_auto_filled = False
        self._camera_auto_filled = False
        self._date_auto_filled = False
        self._integration_auto_filled = False
        self._matching_sessions = []  # Session Manager sessions found for the current DSO

        self.setAcceptDrops(True)
        self._init_ui()
        self._load_dso_list()
        self._load_equipment_list()
        self._update_preview()
        self.setup_window_position()

    def _init_ui(self):
        """Create the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Instructions
        instructions = QLabel("Select an image file and choose which DSO to attach it to.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # --- Image preview / drop zone + DSO selection -----------------
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # Grows with the dialog (both wider and taller) so enlarging the
        # window makes the thumbnail bigger instead of the text fields;
        # the loaded source pixmap is re-scaled to fit on every resize.
        self._preview_source_pixmap = None
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(150, 150)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setCursor(Qt.PointingHandCursor)
        self.preview_label.setToolTip("Click to browse, or drag & drop an image onto this dialog")
        self.preview_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['background_light']};
                border: 2px dashed {COLORS['border_light']};
                border-radius: 6px;
                color: {COLORS['text_secondary']};
                font-size: 9pt;
                padding: 6px;
            }}
        """)
        self.preview_label.mousePressEvent = lambda event: self._browse_file()
        top_row.addWidget(self.preview_label, 1)

        # Fields column stays at its natural width (stretch 0) so extra
        # horizontal space is given to the preview above instead.
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected...")
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setMinimumWidth(200)
        self.file_path_edit.setMaximumWidth(260)
        file_layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)
        right_col.addLayout(file_layout)

        dso_label = QLabel("Attach to DSO:")
        right_col.addWidget(dso_label)

        # DSO selection with search
        self.dso_combo = QComboBox()
        self.dso_combo.setEditable(True)
        self.dso_combo.setInsertPolicy(QComboBox.NoInsert)
        self.dso_combo.lineEdit().setPlaceholderText("Search for DSO...")
        # Keep the closed combo box narrow regardless of how long individual
        # DSO entries are (some objects have many catalogue designations);
        # the popup list itself is widened separately so full names stay readable.
        self.dso_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.dso_combo.setMinimumContentsLength(20)
        self.dso_combo.setMinimumWidth(200)
        self.dso_combo.setMaximumWidth(260)
        self.dso_combo.view().setMinimumWidth(380)
        self.dso_combo.activated.connect(self._on_dso_manually_changed)
        right_col.addWidget(self.dso_combo)

        self.detected_label = QLabel("")
        self.detected_label.setStyleSheet(f"color: {COLORS['info']}; font-size: 9pt;")
        self.detected_label.setWordWrap(True)
        self.detected_label.setMaximumWidth(260)
        self.detected_label.hide()
        right_col.addWidget(self.detected_label)

        # Offers to import telescope/camera/date/integration time from an
        # existing Session Manager session for the same DSO, when the file's
        # own metadata didn't supply them (e.g. a PixelMath-composited XISF
        # result, which carries no acquisition metadata at all). Never
        # auto-applied - always an explicit click, matching this app's
        # existing "always confirm, never silently auto-attach" convention
        # for cross-feature data reuse (see SessionManager.DropMatchDialog).
        self.session_import_btn = QPushButton("Import from Session...")
        self.session_import_btn.setMaximumWidth(260)
        self.session_import_btn.clicked.connect(self._show_session_import_menu)
        self.session_import_btn.hide()
        right_col.addWidget(self.session_import_btn)
        right_col.addStretch()

        top_row.addLayout(right_col)
        # Give the image/DSO row the extra vertical space on a taller resize
        # too, so the preview grows in both directions.
        layout.addLayout(top_row, 1)

        # --- Optional capture details -----------------------------------
        # Capped to a fixed max width so resizing the dialog doesn't stretch
        # these fields - all the extra space goes to the preview instead.
        FIELD_MAX_WIDTH = 260

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.telescope_combo = QComboBox()
        self.telescope_combo.setEditable(True)
        self.telescope_combo.setInsertPolicy(QComboBox.NoInsert)
        self.telescope_combo.lineEdit().setPlaceholderText("e.g., 8\" SCT")
        self.telescope_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.telescope_combo.setMinimumContentsLength(18)
        self.telescope_combo.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Telescope:", self.telescope_combo)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.setInsertPolicy(QComboBox.NoInsert)
        self.camera_combo.lineEdit().setPlaceholderText("e.g., ASI294MC Pro")
        self.camera_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(18)
        self.camera_combo.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Camera:", self.camera_combo)

        self.integration_edit = QLineEdit()
        self.integration_edit.setPlaceholderText("e.g., 2h 30m")
        self.integration_edit.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Integration Time:", self.integration_edit)

        date_layout = QHBoxLayout()
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("e.g., 2024-01-15")
        self.date_edit.setMaximumWidth(FIELD_MAX_WIDTH - 70)
        date_layout.addWidget(self.date_edit)
        today_btn = QPushButton("Today")
        today_btn.setToolTip("Fill in today's date")
        today_btn.clicked.connect(self._fill_today_date)
        date_layout.addWidget(today_btn)
        date_layout.addStretch()
        form_layout.addRow("Date Taken:", date_layout)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes about this image")
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Notes:", self.notes_edit)

        layout.addLayout(form_layout)

        # Inline validation feedback (shown instead of popping a dialog
        # for every missing field)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['error']};")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setMinimumWidth(460)
        self.resize(460, self.sizeHint().height())

    def _load_dso_list(self):
        """Load all DSOs from database for the combo box"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT d.id as dsodetailid,
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                               ORDER BY CASE c.catalogue
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation) as name,
                           d.constellation,
                           d.dsotype
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    GROUP BY d.id
                    ORDER BY
                        CASE
                            WHEN name LIKE 'M %' THEN 1
                            WHEN name LIKE 'NGC %' THEN 2
                            WHEN name LIKE 'IC %' THEN 3
                            ELSE 4
                        END,
                        name
                """
                cursor.execute(query)
                rows = cursor.fetchall()

                # Clear and populate combo box
                self.dso_combo.clear()
                self.dso_data = []

                for row in rows:
                    dsodetailid, name, constellation, dsotype = row
                    display_text = f"{name} ({constellation})"
                    self.dso_combo.addItem(display_text, dsodetailid)
                    self.dso_data.append((dsodetailid, name))

                # Setup completer for search functionality
                completer = QCompleter([self.dso_combo.itemText(i) for i in range(self.dso_combo.count())])
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                completer.setFilterMode(Qt.MatchContains)
                self.dso_combo.setCompleter(completer)

                # Clear selection so placeholder text is shown
                self.dso_combo.setCurrentIndex(-1)
                self.dso_combo.lineEdit().clear()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load DSO list: {str(e)}")

    def _load_equipment_list(self):
        """Load user telescopes and cameras into their respective dropdowns"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Load telescopes
                cursor.execute("SELECT name FROM usertelescopes ORDER BY name")
                for row in cursor.fetchall():
                    self.telescope_combo.addItem(row[0])

                # Load cameras
                cursor.execute("SELECT name FROM userequipment WHERE equipment_type = 'camera' ORDER BY name")
                for row in cursor.fetchall():
                    self.camera_combo.addItem(row[0])

            # Clear selection so placeholder text is shown
            self.telescope_combo.setCurrentIndex(-1)
            self.telescope_combo.lineEdit().clear()
            self.camera_combo.setCurrentIndex(-1)
            self.camera_combo.lineEdit().clear()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load equipment list: {str(e)}")

    def _browse_file(self):
        """Open file dialog to select an image"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            os.path.expanduser("~"),
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fts *.xisf);;"
            "PNG Files (*.png);;"
            "JPEG Files (*.jpg *.jpeg);;"
            "TIFF Files (*.tif *.tiff);;"
            "FITS Files (*.fits *.fit *.fts);;"
            "XISF Files (*.xisf);;"
            "All Files (*.*)"
        )
        if file_name:
            self.set_file_path(file_name)

    def set_file_path(self, path):
        """Set the selected file path, refresh the preview and try to auto-detect
        the DSO and capture info (telescope/camera/date) from the file's metadata"""
        self.selected_file = path
        self.file_path_edit.setText(path)
        self._clear_error(self.file_path_edit)
        self._capture_header = self._read_capture_header()
        self._update_preview()
        self._try_auto_detect_dso()
        self._try_auto_detect_capture_info()
        self._check_for_matching_sessions()

    def _read_capture_header(self):
        """Read FITS/XISF header keywords (OBJECT, TELESCOP, INSTRUME, DATE-OBS,
        ...) from the selected file, if it's one of those formats. Returns None
        for other formats, or if the header can't be read."""
        if not self.selected_file:
            return None
        ext = os.path.splitext(self.selected_file)[1].lower()
        try:
            if ext in ('.fits', '.fit', '.fts'):
                from SessionFileScanner import extract_fits_header
                return extract_fits_header(self.selected_file)
            elif ext == '.xisf':
                from SessionFileScanner import extract_xisf_header
                return extract_xisf_header(self.selected_file)
        except Exception as e:
            logger.debug(f"Could not read capture header from {self.selected_file}: {e}")
        return None

    def _update_preview(self):
        """(Re)load the preview source for the selected file and render it at the
        preview box's current size. Loaded once per file at a resolution large
        enough to still look sharp when the dialog is enlarged."""
        self._preview_source_pixmap = (
            _load_preview_pixmap(self.selected_file, max_dim=600) if self.selected_file else None
        )
        self._render_preview()

    def _render_preview(self):
        """Scale the cached preview source to fit the preview box's current size"""
        if self._preview_source_pixmap:
            scaled = self._preview_source_pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)
        elif self.selected_file:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Preview not\navailable")
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Drag && drop\nan image here\nor click to browse")

    def resizeEvent(self, event):
        """Re-scale the preview thumbnail to fill the enlarged/shrunk preview box"""
        super().resizeEvent(event)
        self._render_preview()

    def _match_dso_index(self, text):
        """Look for a DSO catalogue designation (e.g. 'M31', 'NGC 7000') embedded
        in an arbitrary piece of text - a filename, or a FITS/XISF header's
        OBJECT value."""
        # Keep hyphens (but drop other punctuation/whitespace) so distinct
        # designations that only differ by a hyphen - e.g. Messier "M 16"
        # vs. Minkowski "M 1-6" - don't collapse into the same "M16" string
        # and get confused for one another.
        text_norm = re.sub(r'[^A-Z0-9-]', '', text.upper())
        if not text_norm:
            return None

        best_index = None
        best_len = 0
        for index, (dsodetailid, name) in enumerate(self.dso_data):
            for designation in name.split(','):
                d_norm = re.sub(r'[^A-Z0-9-]', '', designation.upper())
                if len(d_norm) >= 2 and len(d_norm) > best_len and d_norm in text_norm:
                    best_index = index
                    best_len = len(d_norm)
        return best_index

    def _guess_dso_index_from_filename(self, file_path):
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return self._match_dso_index(basename)

    def _try_auto_detect_dso(self):
        """Auto-select the DSO combo, preferring the OBJECT name from the file's
        FITS/XISF header over a filename guess, without overriding a selection
        the user made themselves"""
        if not self.selected_file or (self.dso_combo.currentIndex() >= 0 and not self._dso_auto_selected):
            return

        source = None
        index = None
        object_name = self._capture_header.get('OBJECT') if self._capture_header else None
        if object_name:
            index = self._match_dso_index(str(object_name))
            if index is not None:
                source = "image metadata"
        if index is None:
            index = self._guess_dso_index_from_filename(self.selected_file)
            if index is not None:
                source = "filename"

        if index is not None:
            self.dso_combo.setCurrentIndex(index)
            self._dso_auto_selected = True
            self.detected_label.setText(f"Auto-detected from {source} — please verify this is correct.")
            self.detected_label.show()
        elif self._dso_auto_selected:
            self.dso_combo.setCurrentIndex(-1)
            self.dso_combo.lineEdit().clear()
            self._dso_auto_selected = False
            self.detected_label.hide()

    def _try_auto_detect_capture_info(self):
        """Fill telescope/camera/date from the file's FITS/XISF header, each
        only if that field is still empty - mirrors _try_auto_detect_dso's
        auto-vs-manual tracking so switching to a different dropped file
        replaces a previous auto-fill but never clobbers a manual edit."""
        header = self._capture_header or {}

        telescope = header.get('TELESCOP')
        if telescope and (self._telescope_auto_filled or not self.telescope_combo.currentText().strip()):
            self.telescope_combo.setCurrentText(str(telescope).strip())
            self._telescope_auto_filled = True
        elif not telescope and self._telescope_auto_filled:
            self.telescope_combo.setCurrentText("")
            self._telescope_auto_filled = False

        camera = header.get('INSTRUME')
        if camera and (self._camera_auto_filled or not self.camera_combo.currentText().strip()):
            self.camera_combo.setCurrentText(str(camera).strip())
            self._camera_auto_filled = True
        elif not camera and self._camera_auto_filled:
            self.camera_combo.setCurrentText("")
            self._camera_auto_filled = False

        date_str = self._format_date_obs(header.get('DATE-OBS'))
        if date_str and (self._date_auto_filled or not self.date_edit.text().strip()):
            self.date_edit.setText(date_str)
            self._date_auto_filled = True
        elif not date_str and self._date_auto_filled:
            self.date_edit.setText("")
            self._date_auto_filled = False

    @staticmethod
    def _format_date_obs(date_obs):
        """Reformat a FITS/XISF DATE-OBS value (e.g. '2024-01-15T22:30:00.000')
        to just the date portion, matching this dialog's date_edit convention."""
        if not date_obs:
            return None
        return str(date_obs).strip()[:10] or None

    def _find_matching_sessions(self, dso_index):
        """Find Session Manager sessions for the DSO at dso_data[dso_index],
        matching by normalized designation (the same normalization
        _match_dso_index uses) since a session's dso_name is free text a user
        typed when logging it, not a catalog ID. Only returns sessions that
        have at least one importable field (telescope/camera/integration
        time) - a bare 'Planned' session with nothing filled in yet isn't
        worth suggesting."""
        if dso_index is None or dso_index < 0 or dso_index >= len(self.dso_data):
            return []

        _, designations_str = self.dso_data[dso_index]
        designations_norm = {
            re.sub(r'[^A-Z0-9-]', '', d.upper()) for d in designations_str.split(',')
        }

        matches = []
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.id, s.dso_name, s.session_date, tel.name as telescope_name,
                           s.camera, s.integration_seconds, s.status
                    FROM usersessions s
                    LEFT JOIN usertelescopes tel ON s.telescope_id = tel.id
                    ORDER BY s.session_date DESC, s.id DESC
                """)
                for session_id, dso_name, session_date, telescope_name, camera, integration_seconds, status in cursor.fetchall():
                    if not dso_name:
                        continue
                    name_norm = re.sub(r'[^A-Z0-9-]', '', dso_name.upper())
                    if name_norm not in designations_norm:
                        continue
                    if not (telescope_name or camera or integration_seconds):
                        continue
                    matches.append({
                        "id": session_id, "dso_name": dso_name, "session_date": session_date,
                        "telescope_name": telescope_name, "camera": camera,
                        "integration_seconds": integration_seconds, "status": status,
                    })
        except Exception as e:
            logger.debug(f"Error finding matching sessions: {e}")

        return matches

    def _check_for_matching_sessions(self):
        """Show/hide the 'Import from Session...' button depending on whether
        the currently-selected DSO has any matching sessions with data to offer."""
        self._matching_sessions = self._find_matching_sessions(self.dso_combo.currentIndex())
        if self._matching_sessions:
            count = len(self._matching_sessions)
            self.session_import_btn.setText(
                "Import from Session..." if count == 1 else f"Import from Session... ({count} found)")
            self.session_import_btn.show()
        else:
            self.session_import_btn.hide()

    def _show_session_import_menu(self):
        """Let the user pick which matching session to import equipment/date/
        integration time from, when there's more than one - never guesses."""
        if not self._matching_sessions:
            return

        menu = QMenu(self)
        for session in self._matching_sessions:
            hours = (session["integration_seconds"] or 0) / 3600.0
            parts = [session["session_date"] or "Unknown date"]
            if session["telescope_name"]:
                parts.append(session["telescope_name"])
            if session["camera"]:
                parts.append(session["camera"])
            if hours > 0:
                parts.append(f"{hours:.1f}h")
            parts.append(f"({session['status']})")
            action = menu.addAction(" — ".join(parts))
            action.triggered.connect(lambda checked=False, s=session: self._import_session_data(s))

        menu.exec(self.session_import_btn.mapToGlobal(
            self.session_import_btn.rect().bottomLeft()))

    def _import_session_data(self, session):
        """Apply a chosen session's telescope/camera/date/integration time.
        An explicit user click (via the menu above), so unlike the passive
        metadata auto-fill this always overwrites the fields - the user just
        told us which session's data they want."""
        if session["telescope_name"]:
            self.telescope_combo.setCurrentText(session["telescope_name"])
            self._telescope_auto_filled = True
        if session["camera"]:
            self.camera_combo.setCurrentText(session["camera"])
            self._camera_auto_filled = True
        if session["session_date"]:
            self.date_edit.setText(str(session["session_date"])[:10])
            self._date_auto_filled = True
        if session["integration_seconds"]:
            self.integration_edit.setText(self._format_integration_seconds(session["integration_seconds"]))
            self._integration_auto_filled = True

        self._clear_error(self.telescope_combo)
        self._clear_error(self.camera_combo)

    @staticmethod
    def _format_integration_seconds(seconds):
        """Format a session's total integration_seconds as 'Xh Ym', matching
        this dialog's integration_edit placeholder convention (e.g. '2h 30m')."""
        total_minutes = round(seconds / 60.0)
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            return f"{hours}h {minutes}m"
        elif hours:
            return f"{hours}h"
        return f"{minutes}m"

    def _on_dso_manually_changed(self, index):
        """User picked a DSO themselves; stop treating the selection as a guess"""
        self._dso_auto_selected = False
        self.detected_label.hide()
        self._clear_error(self.dso_combo)
        self._check_for_matching_sessions()

    def _fill_today_date(self):
        """Fill the date field with today's date"""
        self.date_edit.setText(datetime.now().strftime('%Y-%m-%d'))

    def _mark_error(self, widget):
        widget.setStyleSheet(f"border: 1px solid {COLORS['error']};")

    def _clear_error(self, widget):
        widget.setStyleSheet("")

    def dragEnterEvent(self, event):
        """Accept drag if it is a single supported image file"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Set the dropped file as the selected image"""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                event.acceptProposedAction()
                self.set_file_path(file_path)

    def _validate_and_accept(self):
        """Validate inputs before accepting, showing inline feedback instead of popups"""
        errors = []
        self._clear_error(self.file_path_edit)
        self._clear_error(self.dso_combo)

        if not self.selected_file:
            errors.append("Select an image file.")
            self._mark_error(self.file_path_edit)
        elif not os.path.exists(self.selected_file):
            errors.append("The selected image file no longer exists.")
            self._mark_error(self.file_path_edit)

        if self.dso_combo.currentIndex() < 0:
            errors.append("Choose which DSO to attach the image to.")
            self._mark_error(self.dso_combo)

        if errors:
            self.status_label.setText("  •  ".join(errors))
            self.status_label.show()
            return

        self.status_label.hide()
        self.accept()

    def get_image_data(self):
        """Return the entered image data"""
        return {
            'dsodetailid': self.dso_combo.currentData(),
            'image_path': self.selected_file,
            'equipment': ', '.join(filter(None, [self.telescope_combo.currentText().strip(),
                                                     self.camera_combo.currentText().strip()])),
            'integration_time': self.integration_edit.text().strip(),
            'date_taken': self.date_edit.text().strip(),
            'notes': self.notes_edit.toPlainText().strip()
        }


class GalleryCard(QFrame):
    """Individual card widget displaying a DSO thumbnail and info"""

    double_clicked = Signal(dict)  # Emits item_data when double-clicked
    context_menu_requested = Signal(dict, object)  # Emits item_data and position

    def __init__(self, item_data, parent=None, thumbnail_size=150):
        """
        Initialize gallery card

        Args:
            item_data (dict): Dictionary describing a single image (one DSO can
                have multiple images, and therefore multiple cards)
                - dsodetailid: DSO ID
                - imageid: userimages.id for this specific image
                - name: DSO name
                - dsotype: DSO type code
                - image_path: Path to this image
                - equipment: Equipment used for this image
            thumbnail_size (int): Size of thumbnail in pixels (default 150)
        """
        super().__init__(parent)
        self.item_data = item_data
        self.thumbnail_size = thumbnail_size
        self._init_ui()

    def _init_ui(self):
        """Create card layout with thumbnail and labels"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Thumbnail label (dynamic size)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['background_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
            }}
        """)

        # Placeholder text
        self.thumbnail_label.setText("Loading...")
        layout.addWidget(self.thumbnail_label)

        # DSO name label
        name_label = QLabel(self.item_data.get('name', 'Unknown'))
        name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

        # DSO type label
        type_label = QLabel(self.item_data.get('friendly_type', 'Unknown'))
        type_label.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']};")
        type_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(type_label)

        # Tooltip with per-image details, since a DSO with multiple images
        # will have multiple cards that otherwise look identical
        equipment = self.item_data.get('equipment', '').strip()
        tooltip_lines = [self.item_data.get('name', 'Unknown')]
        if equipment:
            tooltip_lines.append(f"Equipment: {equipment}")
        if self.item_data.get('is_favorite'):
            tooltip_lines.append("Favorite image")
        self.setToolTip('\n'.join(tooltip_lines))

        # Card styling - card width is thumbnail size + padding (20px for margins and borders)
        card_width = self.thumbnail_size + 20
        self.setFixedWidth(card_width)
        self.setStyleSheet(f"""
            GalleryCard {{
                background-color: {COLORS['background_lighter']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
            }}
            GalleryCard:hover {{
                border: 2px solid {COLORS['accent']};
                background-color: {COLORS['background_hover']};
            }}
        """)
        self.setCursor(Qt.PointingHandCursor)

    def set_thumbnail(self, pixmap):
        """Update thumbnail with actual image"""
        if pixmap and not pixmap.isNull():
            # Scale to fit thumbnail size while maintaining aspect ratio
            scaled = pixmap.scaled(self.thumbnail_size, self.thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumbnail_label.setPixmap(scaled)
            self.thumbnail_label.setText("")  # Clear placeholder text

    def set_error(self, error_message):
        """Display error on card"""
        self.thumbnail_label.setText(f"Error:\n{error_message[:30]}")
        self.thumbnail_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['error_bg']};
                border: 1px solid {COLORS['error']};
                border-radius: 3px;
                color: {COLORS['error']};
            }}
        """)

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click - emit signal"""
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.item_data)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """Handle right-click context menu"""
        self.context_menu_requested.emit(self.item_data, event.globalPos())
        event.accept()


class DSOGalleryWindow(WindowPositionMixin, QMainWindow):
    """Main window for DSO Image Gallery"""

    WINDOW_POSITION_KEY = "DSOGallery"
    _SUPPORTED_DROP_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS

    def __init__(self):
        """Initialize DSO Image Gallery window"""
        super().__init__()
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("DSO Image Gallery - Cosmos Collection")
        self.resize(1200, 800)

        # Initialize data structures
        self.db_manager = DatabaseManager()
        self.all_items = []
        self.filtered_items = []
        self.data_loaded = False  # Track whether initial data load is complete
        self.current_columns = 1
        self.current_filters = {
            'search': '',
            'catalog': 'All',
            'type': 'All',
            'equipment': 'All',
            'sort': 'Name (A-Z)'
        }

        # Restore previously used search/filter selections
        filter_settings = QSettings("CosmosCollection", "CosmosCollection")
        self.current_filters['search'] = filter_settings.value("gallery_filter_search", '', type=str)
        self.current_filters['catalog'] = filter_settings.value("gallery_filter_catalog", 'All', type=str)
        self.current_filters['type'] = filter_settings.value("gallery_filter_type", 'All', type=str)
        self.current_filters['equipment'] = filter_settings.value("gallery_filter_equipment", 'All', type=str)
        self.current_filters['sort'] = filter_settings.value("gallery_filter_sort", 'Name (A-Z)', type=str)

        # Thumbnail size options: Small (100), Medium (150), Large (200), Extra Large (250)
        self.thumbnail_size_options = {
            'Small': 100,
            'Medium': 150,
            'Large': 300,
            'Extra Large': 500
        }
        # Load thumbnail size from settings (default to Medium)
        settings = QSettings("CosmosCollection", "CosmosCollection")
        saved_size_name = settings.value("gallery_thumbnail_size", "Medium")
        self.thumbnail_size = self.thumbnail_size_options.get(saved_size_name, 150)

        # Thumbnail cache and thread pool
        self.thumbnail_cache = ThumbnailCache(max_size=200)
        self.thread_pool = QThreadPool.globalInstance()
        # Get thread count from user settings
        cpu_count = os.cpu_count() or 4
        default_threads = max(1, cpu_count - 2)
        thread_count = settings.value("max_threads", default_threads, type=int)
        self.thread_pool.setMaxThreadCount(thread_count)
        self.thumbnail_signals = ThumbnailSignals()
        self._thumbnail_signals_connected = False
        self.cancelled_flag = [False]  # Mutable flag for cancellation

        # Thumbnail loading progress tracking
        self.thumbnail_progress_dialog = None
        self.thumbnails_to_load = 0
        self.thumbnails_loaded = 0
        self.is_loading_thumbnails = False

        # Data loader signals
        self.data_loader_signals = DataLoaderSignals()
        self.data_loader_signals.data_loaded.connect(self._on_data_loaded)
        self.data_loader_signals.load_error.connect(self._on_data_load_error)

        # Search debounce timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._apply_filters)

        # Resize debounce timer (longer delay to reduce rebuilds)
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._handle_resize)

        # Flag to track if resize is in progress
        self.resize_in_progress = False

        # Count of outstanding QApplication.setOverrideCursor(WaitCursor) pushes
        # that still need a matching restoreOverrideCursor(). A plain boolean
        # can't survive overlapping triggers (e.g. typing quickly in the search
        # box while a previous grid populate is still batching in), so we use
        # a counter instead to keep every push paired with exactly one pop.
        self._pending_wait_cursors = 0

        # Lazy loading tracking
        self.thumbnail_loaded_indices = set()  # Track which card indices have been queued
        self.scroll_debounce_timer = QTimer()
        self.scroll_debounce_timer.setSingleShot(True)
        self.scroll_debounce_timer.timeout.connect(self._on_scroll_debounced)
        self.visible_buffer_rows = 2  # Load this many extra rows above/below visible area

        # Enable drag-and-drop of image files from file manager
        self.setAcceptDrops(True)

        # Initialize UI
        self._init_ui()

        # Setup window position persistence
        self.setup_window_position()

        # Load data in background (defer grid population until data is loaded)
        self._initial_load_pending = True
        self._start_background_data_load()

    def _init_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header_label = QLabel("DSO Image Gallery")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(header_label)

        # Filters group
        filters_group = QGroupBox("Search & Filter")
        filters_layout = QHBoxLayout(filters_group)

        # Search input
        filters_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search DSO name...")
        self.search_input.setText(self.current_filters['search'])
        self.search_input.textChanged.connect(lambda: self.search_timer.start(300))
        filters_layout.addWidget(self.search_input)

        # Catalog filter
        filters_layout.addWidget(QLabel("Catalog:"))
        self.catalog_combo = QComboBox()
        self.catalog_combo.addItems(["All", "M", "NGC", "IC", "Sh2", "B", "Cr", "Mel"])
        if self.current_filters['catalog'] in [self.catalog_combo.itemText(i) for i in range(self.catalog_combo.count())]:
            self.catalog_combo.setCurrentText(self.current_filters['catalog'])
        self.catalog_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.catalog_combo)

        # Type filter
        filters_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("All")
        self.type_combo.setMinimumWidth(150)
        self.type_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.type_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.type_combo)

        # Equipment filter
        filters_layout.addWidget(QLabel("Equipment:"))
        self.equipment_combo = QComboBox()
        self.equipment_combo.addItem("All")
        self.equipment_combo.setMinimumWidth(200)
        self.equipment_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.equipment_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.equipment_combo)

        # Sort dropdown
        filters_layout.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name (A-Z)", "Name (Z-A)", "Date Added (Newest)", "Date Added (Oldest)", "Type", "Constellation"])
        self.sort_combo.setMinimumWidth(130)
        if self.current_filters['sort'] in [self.sort_combo.itemText(i) for i in range(self.sort_combo.count())]:
            self.sort_combo.setCurrentText(self.current_filters['sort'])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        filters_layout.addWidget(self.sort_combo)

        # Thumbnail size selector
        filters_layout.addWidget(QLabel("Thumbnail Size:"))
        self.size_combo = QComboBox()
        for size_name in self.thumbnail_size_options.keys():
            self.size_combo.addItem(size_name)
        # Set current selection based on loaded setting
        current_size_name = [k for k, v in self.thumbnail_size_options.items() if v == self.thumbnail_size]
        if current_size_name:
            self.size_combo.setCurrentText(current_size_name[0])
        self.size_combo.currentTextChanged.connect(self._on_thumbnail_size_changed)
        filters_layout.addWidget(self.size_combo)

        # Clear filters button
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_filters)
        filters_layout.addWidget(clear_btn)

        # Add Image button
        add_image_btn = QPushButton("Add Image")
        add_image_btn.clicked.connect(self._show_add_image_dialog)
        filters_layout.addWidget(add_image_btn)

        filters_layout.addStretch()
        main_layout.addWidget(filters_group)

        # Scroll area for grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for grid
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)

        self.scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll_area)

        # Connect scroll bar to lazy loading
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)

        # Status label
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("padding: 5px;")
        main_layout.addWidget(self.status_label)

    def _start_background_data_load(self):
        """Start loading gallery data in background thread"""
        loader = DataLoaderRunnable(self.data_loader_signals)
        self.thread_pool.start(loader)

    def _on_data_loaded(self, items):
        """Handle data loaded from background thread"""
        self.all_items = items
        self.data_loaded = True  # Mark data as loaded

        # Populate filter dropdowns (restores saved type/equipment selection if still valid)
        self._populate_type_filter()
        self._populate_equipment_filter()

        # Apply restored search/filter/sort state now that data and options are available
        self._filter_and_sort_items()

        # Update status
        count = len(self.all_items)
        self.status_label.setText(f"Loaded {count} image{'s' if count != 1 else ''}")

        # Populate grid if window is already shown
        if not self._initial_load_pending:
            self._populate_grid()

    def _on_data_load_error(self, error_message):
        """Handle error loading data from background thread"""
        self.status_label.setText(f"Error loading gallery: {error_message}")
        QMessageBox.critical(self, "Error", f"Failed to load DSO gallery:\n{error_message}")

    def _populate_type_filter(self):
        """Populate type filter dropdown with unique types from loaded data"""
        # Get unique friendly types
        types = set()
        for item in self.all_items:
            if item['friendly_type']:
                types.add(item['friendly_type'])

        # Sort and add to combo box
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        self.type_combo.addItem("All")
        for dso_type in sorted(types):
            self.type_combo.addItem(dso_type)

        # Restore saved filter selection if it still exists among the options
        saved_type = self.current_filters.get('type', 'All')
        if saved_type in [self.type_combo.itemText(i) for i in range(self.type_combo.count())]:
            self.type_combo.setCurrentText(saved_type)
        else:
            self.current_filters['type'] = 'All'
        self.type_combo.blockSignals(False)

        # Ensure dropdown view is wide enough to show full text
        self.type_combo.view().setMinimumWidth(self.type_combo.minimumSizeHint().width())

    def _populate_equipment_filter(self):
        """Populate equipment filter dropdown with unique equipment from loaded data"""
        # Get unique equipment (non-empty)
        equipment_set = set()
        for item in self.all_items:
            if item['equipment'].strip():
                equipment_set.add(item['equipment'].strip())

        # Sort and add to combo box
        self.equipment_combo.blockSignals(True)
        self.equipment_combo.clear()
        self.equipment_combo.addItem("All")
        for equipment in sorted(equipment_set):
            self.equipment_combo.addItem(equipment)

        # Restore saved filter selection if it still exists among the options
        saved_equipment = self.current_filters.get('equipment', 'All')
        if saved_equipment in [self.equipment_combo.itemText(i) for i in range(self.equipment_combo.count())]:
            self.equipment_combo.setCurrentText(saved_equipment)
        else:
            self.current_filters['equipment'] = 'All'
        self.equipment_combo.blockSignals(False)

        # Ensure dropdown view is wide enough to show full text
        self.equipment_combo.view().setMinimumWidth(self.equipment_combo.minimumSizeHint().width())

    def _calculate_grid_columns(self):
        """Calculate number of columns based on available width"""
        # Card width is thumbnail_size + 20px padding (from GalleryCard._init_ui)
        card_width = self.thumbnail_size + 20
        grid_spacing = 10  # Grid layout spacing
        grid_margins = 20  # Grid layout margins (10 left + 10 right)

        # Use viewport width for accurate calculation
        viewport_width = self.scroll_area.viewport().width()
        available_width = viewport_width - grid_margins

        # Each card takes up card_width + spacing
        card_width_with_spacing = card_width + grid_spacing

        # Calculate how many cards fit (no arbitrary cap)
        columns = max(1, available_width // card_width_with_spacing)

        return columns

    def _push_wait_cursor(self):
        """Push a WaitCursor and track it so it's always paired with a restore."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._pending_wait_cursors += 1

    def _pop_wait_cursor(self):
        """Restore a previously pushed WaitCursor, if one is still outstanding."""
        if self._pending_wait_cursors > 0:
            QApplication.restoreOverrideCursor()
            self._pending_wait_cursors -= 1

    def _populate_grid(self):
        """Populate grid with gallery cards"""
        # Close any existing progress dialog
        if self.thumbnail_progress_dialog:
            self.thumbnail_progress_dialog.close()
            self.thumbnail_progress_dialog = None
        self.is_loading_thumbnails = False

        # Disable updates during grid rebuild to prevent excessive repainting
        self.grid_container.setUpdatesEnabled(False)

        # Clear existing cards
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Check if there are items to display
        if not self.filtered_items:
            # Show different message if still loading data vs no results
            if not self.data_loaded:
                # Still loading initial data
                loading_label = QLabel("Loading DSO images from database...")
                loading_label.setStyleSheet("font-size: 14px; color: #cccccc; padding: 50px;")
            elif len(self.all_items) == 0:
                # Data loaded but database has no images
                loading_label = QLabel("No images in your database.\n\nAdd images to DSO objects to see them here.")
                loading_label.setStyleSheet("font-size: 14px; color: #888888; padding: 50px;")
            else:
                # Data loaded but no matches for current filters
                loading_label = QLabel("No DSO images found matching your filters")
                loading_label.setStyleSheet("font-size: 14px; color: #888888; padding: 50px;")
            loading_label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(loading_label, 0, 0)

            if self.all_items:
                self.status_label.setText(f"Showing 0 of {len(self.all_items)} images")

            # Re-enable updates
            self.grid_container.setUpdatesEnabled(True)
            return

        # Calculate columns based on window width
        cols = self._calculate_grid_columns()
        self.current_columns = cols

        # Store cards for thumbnail loading
        self.cards = []

        # Show wait cursor during loading if many items
        showing = len(self.filtered_items)
        total = len(self.all_items)
        cursor_pushed = showing > 50
        if cursor_pushed:
            self._push_wait_cursor()

        # Set up progress tracking for thumbnail loading
        self.thumbnails_to_load = showing
        self.thumbnails_loaded = 0
        self.is_loading_thumbnails = True

        # Create and show progress dialog for loading thumbnails
        if showing > 0:
            self.thumbnail_progress_dialog = QProgressDialog(
                f"Loading thumbnails... (0/{showing})",
                None,  # No cancel button
                0,
                showing,
                self
            )
            self.thumbnail_progress_dialog.setWindowTitle("Loading Gallery")
            self.thumbnail_progress_dialog.setWindowModality(Qt.WindowModal)
            self.thumbnail_progress_dialog.setMinimumDuration(500)  # Only show if takes > 500ms
            self.thumbnail_progress_dialog.setMinimumWidth(400)
            self.thumbnail_progress_dialog.setMinimumHeight(120)
            self.thumbnail_progress_dialog.setStyleSheet("""
                QProgressDialog {
                    font-size: 12pt;
                }
                QProgressBar {
                    min-height: 25px;
                    font-size: 11pt;
                }
                QLabel {
                    font-size: 12pt;
                }
            """)
            self.thumbnail_progress_dialog.setValue(0)
            QApplication.processEvents()  # Ensure dialog can be displayed

        # Update status immediately
        if showing == total:
            self.status_label.setText(f"Loading {total} image{'s' if total != 1 else ''}...")
        else:
            self.status_label.setText(f"Loading {showing} of {total} images...")

        # Create cards in batches to keep UI responsive
        self._create_cards_batch(0, cols, cursor_pushed=cursor_pushed)

    def _create_cards_batch(self, start_idx, cols, batch_size=15, cursor_pushed=False):
        """Create a batch of gallery cards to keep UI responsive"""
        end_idx = min(start_idx + batch_size, len(self.filtered_items))

        # Create cards for this batch
        for idx in range(start_idx, end_idx):
            item = self.filtered_items[idx]
            row = idx // cols
            col = idx % cols

            # Create card with current thumbnail size
            card = GalleryCard(item, thumbnail_size=self.thumbnail_size)
            card.double_clicked.connect(self._on_card_double_clicked)
            card.context_menu_requested.connect(self._show_card_context_menu)

            # Add to grid
            self.grid_layout.addWidget(card, row, col)

            # Store reference
            self.cards.append(card)

        # Update progress in status bar
        total = len(self.filtered_items)
        self.status_label.setText(f"Loading gallery... {end_idx}/{total}")

        # If there are more cards to create, schedule next batch
        # Use QTimer.singleShot(0, ...) to allow UI events to process between batches
        if end_idx < len(self.filtered_items):
            QTimer.singleShot(0, lambda: self._create_cards_batch(end_idx, cols, batch_size, cursor_pushed))
        else:
            # All cards created - finalize grid layout
            self.grid_layout.setRowStretch(len(self.filtered_items) // cols + 1, 1)
            self.grid_layout.setColumnStretch(cols, 1)

            # Re-enable updates now that grid is built
            self.grid_container.setUpdatesEnabled(True)

            # Restore cursor if this batch chain pushed one
            if cursor_pushed:
                self._pop_wait_cursor()

            # Update status
            showing = len(self.filtered_items)
            total = len(self.all_items)
            if showing == total:
                self.status_label.setText(f"Showing all {total} image{'s' if total != 1 else ''}")
            else:
                self.status_label.setText(f"Showing {showing} of {total} images")

            # Load thumbnails in background
            self._load_thumbnails()

    def _load_thumbnails(self):
        """Load all thumbnails, prioritizing visible cards first"""
        # Cancel any pending tasks (they will check the flag and exit early)
        self.cancelled_flag[0] = True
        # Create new cancellation flag for new batch of tasks
        self.cancelled_flag = [False]

        # Reset tracking for new grid
        self.thumbnail_loaded_indices.clear()

        # Connect signals (disconnect first to avoid duplicates). Only attempt
        # disconnect if we know we're connected - PySide6 emits a
        # RuntimeWarning (not a catchable exception) when disconnecting a
        # signal with no matching connection, so a try/except can't suppress it.
        if self._thumbnail_signals_connected:
            self.thumbnail_signals.thumbnail_ready.disconnect(self._on_thumbnail_ready)
            self.thumbnail_signals.thumbnail_error.disconnect(self._on_thumbnail_error)

        self.thumbnail_signals.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumbnail_signals.thumbnail_error.connect(self._on_thumbnail_error)
        self._thumbnail_signals_connected = True

        # Load visible thumbnails first (priority)
        self._load_visible_thumbnails()

        # Schedule loading of remaining thumbnails after visible ones are queued
        QTimer.singleShot(200, self._load_remaining_thumbnails)

    def _get_visible_card_indices(self):
        """Calculate which card indices are currently visible in the viewport"""
        if not hasattr(self, 'cards') or not self.cards or self.current_columns == 0:
            return set()

        # Get scroll area viewport geometry
        viewport = self.scroll_area.viewport()
        viewport_height = viewport.height()
        scroll_pos = self.scroll_area.verticalScrollBar().value()

        # Estimate card height (thumbnail + name label ~20 + type label ~16 + margins ~30 + spacing)
        card_height = self.thumbnail_size + 70  # Approximate height of each card including spacing

        # Calculate visible row range
        first_visible_row = max(0, scroll_pos // card_height - self.visible_buffer_rows)
        last_visible_row = (scroll_pos + viewport_height) // card_height + self.visible_buffer_rows

        # Convert rows to card indices
        visible_indices = set()
        total_cards = len(self.cards)

        for row in range(first_visible_row, last_visible_row + 1):
            for col in range(self.current_columns):
                idx = row * self.current_columns + col
                if 0 <= idx < total_cards:
                    visible_indices.add(idx)

        return visible_indices

    def _load_visible_thumbnails(self):
        """Load thumbnails for currently visible cards that haven't been loaded yet"""
        visible_indices = self._get_visible_card_indices()

        # Find indices that need loading (visible but not yet queued)
        indices_to_load = visible_indices - self.thumbnail_loaded_indices

        if not indices_to_load:
            return

        # Mark these indices as queued
        self.thumbnail_loaded_indices.update(indices_to_load)

        # Queue thumbnail loading for new visible cards
        for idx in indices_to_load:
            if idx < len(self.cards):
                card = self.cards[idx]
                image_path = card.item_data['image_path']
                runnable = ThumbnailRunnable(
                    card,
                    image_path,
                    self.thumbnail_cache,
                    self.thumbnail_signals,
                    self.cancelled_flag,
                    self.thumbnail_size
                )
                self.thread_pool.start(runnable)

    def _load_remaining_thumbnails(self, batch_start=0, batch_size=20):
        """Load remaining thumbnails that weren't in the initial visible set"""
        if not hasattr(self, 'cards') or not self.cards:
            return

        # Check if cancelled (grid was rebuilt)
        if self.cancelled_flag[0]:
            return

        total_cards = len(self.cards)
        batch_end = min(batch_start + batch_size, total_cards)
        loaded_count = 0

        # Queue thumbnails for this batch (skip already queued ones)
        for idx in range(batch_start, batch_end):
            if idx not in self.thumbnail_loaded_indices:
                self.thumbnail_loaded_indices.add(idx)
                card = self.cards[idx]
                image_path = card.item_data['image_path']
                runnable = ThumbnailRunnable(
                    card,
                    image_path,
                    self.thumbnail_cache,
                    self.thumbnail_signals,
                    self.cancelled_flag,
                    self.thumbnail_size
                )
                self.thread_pool.start(runnable)
                loaded_count += 1

        # Schedule next batch if there are more cards
        if batch_end < total_cards:
            # Small delay between batches to keep UI responsive
            QTimer.singleShot(50, lambda: self._load_remaining_thumbnails(batch_end, batch_size))

    def _on_scroll(self, value):
        """Handle scroll events - debounce and trigger lazy loading"""
        # Use debounce to avoid excessive loading during fast scrolling
        self.scroll_debounce_timer.start(100)  # 100ms debounce for smoother scrolling

    def _on_scroll_debounced(self):
        """Handle debounced scroll - load newly visible thumbnails"""
        self._load_visible_thumbnails()

    def _on_thumbnail_ready(self, card, pixmap):
        """Handle thumbnail loaded successfully"""
        try:
            # Check if card still exists (hasn't been deleted by grid refresh)
            if card and not card.isHidden() and card.parent():
                card.set_thumbnail(pixmap)
        except RuntimeError:
            # Card was deleted, ignore
            pass

        # Update resize progress if active
        self._update_thumbnail_progress()

    def _on_thumbnail_error(self, card, error_message):
        """Handle thumbnail load error"""
        try:
            # Check if card still exists (hasn't been deleted by grid refresh)
            if card and not card.isHidden() and card.parent():
                card.set_error(error_message)
        except RuntimeError:
            # Card was deleted, ignore
            pass

        # Update resize progress if active
        self._update_thumbnail_progress()

    def _update_thumbnail_progress(self):
        """Update the thumbnail loading progress dialog"""
        if not self.is_loading_thumbnails:
            return

        self.thumbnails_loaded += 1

        # Update dialog if it exists and is visible
        if self.thumbnail_progress_dialog is not None:
            try:
                self.thumbnail_progress_dialog.setValue(self.thumbnails_loaded)
                self.thumbnail_progress_dialog.setLabelText(
                    f"Loading thumbnails... ({self.thumbnails_loaded}/{self.thumbnails_to_load})"
                )
            except (RuntimeError, AttributeError):
                # Dialog was deleted or not fully initialized
                pass

        # Check if all thumbnails are complete
        if self.thumbnails_loaded >= self.thumbnails_to_load:
            self._finish_thumbnail_loading()

    def _finish_thumbnail_loading(self):
        """Close the thumbnail progress dialog and clean up"""
        self.is_loading_thumbnails = False
        self.thumbnails_to_load = 0
        self.thumbnails_loaded = 0

        if self.thumbnail_progress_dialog:
            self.thumbnail_progress_dialog.close()
            self.thumbnail_progress_dialog = None

    def _on_card_double_clicked(self, item_data):
        """Handle card double-click - open image viewer"""
        try:
            # Import ImageViewerWindow from main
            from main import ImageViewerWindow

            # Check if image file exists
            image_path = item_data['image_path']
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Image Not Found",
                                  f"Image file not found:\n{image_path}\n\n"
                                  f"The file may have been moved or deleted.")
                return

            # Open the viewer straight away with no pixmap - it loads the image in
            # a background thread, so large files don't freeze the UI
            # (pixmap, title, file_path, parent, dso_ra, dso_dec)
            ra_deg = item_data.get('ra_deg')
            dec_deg = item_data.get('dec_deg')
            self.image_viewer = ImageViewerWindow(
                None, item_data['name'], image_path, self,
                dso_ra=ra_deg, dso_dec=dec_deg
            )
            self.image_viewer.show()
            self.image_viewer.raise_()
            self.image_viewer.activateWindow()

        except ImportError as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to import ImageViewerWindow:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to open image viewer:\n{str(e)}")

    def _open_dso_details(self, item_data):
        """Open DSO detail window for the selected object"""
        try:
            from main import DSODetailWindow

            dsodetailid = item_data['dsodetailid']

            # Query database for full DSO details
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

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
                    LEFT JOIN userimages ui ON d.id = ui.dsodetailid AND ui.is_favorite = 1
                    WHERE d.id = ?
                    GROUP BY d.id
                """

                cursor.execute(query, (dsodetailid,))
                result = cursor.fetchone()

                if not result:
                    QMessageBox.warning(self, "Error", "Could not load DSO details")
                    return

                # Unpack result
                obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
                    constellation, dso_type, dso_class, designations, image_path, integration_time, \
                    equipment, date_taken, notes, image_count = result

                # Get primary designation for catalogue and id
                primary_designation = designations.split(',')[0]
                catalogue, designation = primary_designation.strip().split(' ', 1)

                # Format RA/Dec for display
                ra_str = self._format_ra(ra)
                dec_str = self._format_dec(dec)

                # Build data dictionary
                data = {
                    "name": item_data['name'],
                    "ra": ra_str,
                    "dec": dec_str,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "magnitude": magnitude,
                    "surface_brightness": surface_brightness,
                    "size_min": size_min if size_min else 0.0,
                    "size_max": size_max if size_max else 0.0,
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

                # Create and show detail window
                detail_window = DSODetailWindow(data)
                detail_window.show()
                detail_window.raise_()
                detail_window.activateWindow()

        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to open DSO details:\n{str(e)}")

    def _format_ra(self, ra_deg):
        """Format RA in degrees to HH:MM:SS.SS format"""
        ra_hours = ra_deg / 15.0
        hours = int(ra_hours)
        minutes = int((ra_hours - hours) * 60)
        seconds = ((ra_hours - hours) * 60 - minutes) * 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    def _format_dec(self, dec_deg):
        """Format Dec in degrees to DD:MM:SS.S format"""
        sign = '+' if dec_deg >= 0 else '-'
        dec_abs = abs(dec_deg)
        degrees = int(dec_abs)
        minutes = int((dec_abs - degrees) * 60)
        seconds = ((dec_abs - degrees) * 60 - minutes) * 60
        return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:04.1f}"

    def _show_card_context_menu(self, item_data, position):
        """Show context menu when right-clicking on a card"""
        context_menu = QMenu(self)

        # Add menu actions
        view_action = context_menu.addAction("View Full Image")
        view_action.triggered.connect(lambda: self._on_card_double_clicked(item_data))

        details_action = context_menu.addAction("View DSO Details")
        details_action.triggered.connect(lambda: self._open_dso_details(item_data))

        # Show menu at cursor position
        context_menu.exec(position)

    def _on_filter_changed(self):
        """Handle filter combo box changes"""
        self._apply_filters()

    def _on_thumbnail_size_changed(self, size_name):
        """Handle thumbnail size selector change"""
        new_size = self.thumbnail_size_options.get(size_name, 150)
        if new_size != self.thumbnail_size:
            self.thumbnail_size = new_size

            # Save setting
            settings = QSettings("CosmosCollection", "CosmosCollection")
            settings.setValue("gallery_thumbnail_size", size_name)

            # Clear thumbnail cache since cached images are at the old size
            self.thumbnail_cache.clear()
            self.thumbnail_loaded_indices.clear()

            # Cancel pending thumbnail tasks
            self.cancelled_flag[0] = True

            # Repopulate grid with new size (progress dialog shown by _populate_grid)
            self._populate_grid()

    def _apply_filters(self):
        """Apply all filters and refresh grid"""
        # Update current filter state
        self.current_filters['search'] = self.search_input.text().strip()
        self.current_filters['catalog'] = self.catalog_combo.currentText()
        self.current_filters['type'] = self.type_combo.currentText()
        self.current_filters['equipment'] = self.equipment_combo.currentText()
        self.current_filters['sort'] = self.sort_combo.currentText()

        # Persist filter/search state so it's remembered next time the gallery opens
        self._save_filter_settings()

        self._filter_and_sort_items()

        # Refresh grid
        self._populate_grid()

    def _save_filter_settings(self):
        """Persist current search/filter/sort state to settings"""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("gallery_filter_search", self.current_filters['search'])
        settings.setValue("gallery_filter_catalog", self.current_filters['catalog'])
        settings.setValue("gallery_filter_type", self.current_filters['type'])
        settings.setValue("gallery_filter_equipment", self.current_filters['equipment'])
        settings.setValue("gallery_filter_sort", self.current_filters['sort'])

    def _filter_and_sort_items(self):
        """Filter and sort all_items into filtered_items based on current_filters"""
        self.filtered_items = [item for item in self.all_items if self._matches_filters(item)]
        self._sort_items()

    def _on_sort_changed(self):
        """Handle sort dropdown change"""
        self._apply_filters()

    def _sort_items(self):
        """Sort filtered items based on current sort selection"""
        sort_option = self.current_filters['sort']

        if sort_option == "Name (A-Z)":
            self.filtered_items.sort(key=lambda x: x['name'].lower())
        elif sort_option == "Name (Z-A)":
            self.filtered_items.sort(key=lambda x: x['name'].lower(), reverse=True)
        elif sort_option == "Date Added (Newest)":
            self.filtered_items.sort(key=lambda x: x.get('created_date', '') or '', reverse=True)
        elif sort_option == "Date Added (Oldest)":
            self.filtered_items.sort(key=lambda x: x.get('created_date', '') or '')
        elif sort_option == "Type":
            self.filtered_items.sort(key=lambda x: (x['friendly_type'].lower(), x['name'].lower()))
        elif sort_option == "Constellation":
            self.filtered_items.sort(key=lambda x: (x['constellation'].lower(), x['name'].lower()))

    def _matches_filters(self, item):
        """Check if item matches all current filters"""
        # Text search (case-insensitive substring match in name)
        search_text = self.current_filters['search']
        if search_text and search_text.lower() not in item['name'].lower():
            return False

        # Catalog filter (name starts with catalog prefix)
        catalog = self.current_filters['catalog']
        if catalog != 'All':
            # Check if name starts with the catalog (e.g., "M ", "NGC ")
            if not item['name'].startswith(catalog + ' '):
                return False

        # Type filter (friendly type name match)
        type_filter = self.current_filters['type']
        if type_filter != 'All':
            if item['friendly_type'] != type_filter:
                return False

        # Equipment filter (substring match)
        equipment = self.current_filters['equipment']
        if equipment != 'All':
            if equipment not in item.get('equipment', ''):
                return False

        return True

    def _clear_filters(self):
        """Clear all filters"""
        self.search_input.clear()
        self.catalog_combo.setCurrentText("All")
        self.type_combo.setCurrentText("All")
        self.equipment_combo.setCurrentText("All")
        self._apply_filters()

    def _show_add_image_dialog(self, file_path=None):
        """Show dialog to add a new image to a DSO"""
        dialog = AddImageDialog(self)
        if file_path:
            dialog.set_file_path(file_path)
        if dialog.exec() == QDialog.Accepted:
            image_data = dialog.get_image_data()
            self._add_image_to_database(image_data)

    def _add_image_to_database(self, image_data):
        """Add an image to the database and refresh the gallery"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO userimages (
                        dsodetailid, image_path, integration_time,
                        equipment, date_taken, notes, created_date
                    ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    image_data['dsodetailid'],
                    image_data['image_path'],
                    image_data['integration_time'],
                    image_data['equipment'],
                    image_data['date_taken'],
                    image_data['notes']
                ))
                conn.commit()

            # Show success message
            QMessageBox.information(self, "Image Added",
                                   f"Image successfully added to database.")

            # Refresh the gallery data
            self._refresh_gallery()

        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to add image to database:\n{str(e)}")

    def _refresh_gallery(self):
        """Reload gallery data from database"""
        # Clear current data
        self.all_items = []
        self.filtered_items = []
        self.data_loaded = False
        self.thumbnail_loaded_indices.clear()

        # Cancel pending thumbnail tasks
        self.cancelled_flag[0] = True

        # Clear thumbnail cache for this item (in case image changed)
        self.thumbnail_cache.clear()

        # Update status
        self.status_label.setText("Refreshing gallery...")

        # Reload data from database
        self._start_background_data_load()

    def showEvent(self, event):
        """Handle window show - populate grid on first show"""
        super().showEvent(event)
        if self._initial_load_pending:
            self._initial_load_pending = False
            # Use QTimer to ensure window is fully laid out
            QTimer.singleShot(0, self._populate_grid)
        else:
            # Trigger a resize check to ensure grid fits current window
            QTimer.singleShot(0, self._handle_resize)

    def _handle_resize(self):
        """Handle deferred resize - recalculate grid if needed"""
        # Only process if we have data loaded
        if not self.filtered_items or self.resize_in_progress:
            return

        new_cols = self._calculate_grid_columns()
        # Always update if columns changed, even if grid exists
        if new_cols != self.current_columns and new_cols > 0:
            # Set flag to prevent multiple simultaneous resizes
            self.resize_in_progress = True

            # Show visual feedback - change cursor and status
            self._push_wait_cursor()
            old_status = self.status_label.text()
            self.status_label.setText("Reorganizing gallery layout...")
            self.status_label.setStyleSheet("padding: 5px; color: #ffcc00;")

            # Schedule grid rebuild after UI update
            QTimer.singleShot(10, lambda: self._rebuild_grid_for_resize(new_cols, old_status))

    def _rebuild_grid_for_resize(self, new_cols, old_status):
        """Rebuild grid with new column count and restore status"""
        self.current_columns = new_cols
        self._populate_grid()

        # Restore original status after a brief delay
        QTimer.singleShot(100, lambda: self._restore_status_after_resize(old_status))

    def _restore_status_after_resize(self, old_status):
        """Restore status label and cursor after resize completes"""
        self.status_label.setText(old_status)
        self.status_label.setStyleSheet("padding: 5px;")
        self.resize_in_progress = False
        # Restore normal cursor
        self._pop_wait_cursor()

    def resizeEvent(self, event):
        """Handle window resize - defer grid recalculation"""
        super().resizeEvent(event)

        # Show real-time column count during resize
        if self.filtered_items and not self.resize_in_progress:
            current_cols = self._calculate_grid_columns()

            # Calculate pixels needed for next column
            card_width = self.thumbnail_size + 20  # Card width is thumbnail_size + padding
            grid_spacing = 10
            grid_margins = 20
            viewport_width = self.scroll_area.viewport().width()

            # Width needed for next column
            card_width_with_spacing = card_width + grid_spacing
            next_col_viewport_width = (current_cols + 1) * card_width_with_spacing + grid_margins
            pixels_needed = next_col_viewport_width - viewport_width

            if pixels_needed > 0:
                self.status_label.setText(f"Columns: {current_cols} | +{pixels_needed}px wider for next column")
            else:
                self.status_label.setText(f"Columns: {current_cols}")
            self.status_label.setStyleSheet("padding: 5px; color: #88ccff;")

        # Restart timer to debounce resize events (300ms delay reduces rebuild frequency)
        self.resize_timer.start(300)

    def dragEnterEvent(self, event):
        """Accept drag if it is a single supported image file"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in self._SUPPORTED_DROP_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Open AddImageDialog with the dropped file path pre-filled"""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in self._SUPPORTED_DROP_EXTENSIONS:
                event.acceptProposedAction()
                self._show_add_image_dialog(file_path)

    def closeEvent(self, event):
        """Handle window close - cleanup thread pool and cursor"""
        # Restore any WaitCursor pushes that never got matched with a restore
        # (e.g. window closed while a grid populate or resize was still in flight)
        while self._pending_wait_cursors > 0:
            self._pop_wait_cursor()

        # Cancel all pending thumbnail tasks
        self.cancelled_flag[0] = True
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds for tasks to finish
        super().closeEvent(event)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = DSOGalleryWindow()
    window.show()
    sys.exit(app.exec())
