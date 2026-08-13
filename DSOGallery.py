#!/usr/bin/env python3
"""
DSO Image Gallery
Displays all DSO objects with images in a responsive grid gallery format
"""

import sys
import os
from PySide6.QtCore import Qt, Signal, QTimer, QThreadPool, QRunnable, QObject
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QGroupBox,
                               QMessageBox, QScrollArea, QComboBox, QLineEdit,
                               QFrame, QGridLayout, QMenu, QApplication,
                               QDialog, QFileDialog, QFormLayout, QDialogButtonBox,
                               QCompleter, QSlider, QProgressDialog)
from PySide6.QtCore import QSettings
from PySide6.QtGui import QPixmap, QImage

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
import numpy as np


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

    def _load_fits_thumbnail(self, fits_path):
        """Load a FITS file and convert to QPixmap thumbnail"""
        try:
            from astropy.io import fits
            from astropy.visualization import simple_norm

            # Open FITS file
            with fits.open(fits_path) as hdul:
                # Get the primary image data
                image_data = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        image_data = hdu.data
                        break

                if image_data is None:
                    return None

                # Handle different dimensionalities
                is_rgb = False
                if len(image_data.shape) > 2:
                    # Check if this is an RGB image (3 color planes)
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        is_rgb = True
                    elif len(image_data.shape) == 3 and image_data.shape[0] == 3:
                        # RGB planes in first dimension, transpose
                        image_data = np.transpose(image_data, (1, 2, 0))
                        is_rgb = True
                    elif len(image_data.shape) == 3:
                        # Take first 2D slice
                        image_data = image_data[0]
                    elif len(image_data.shape) == 4:
                        image_data = image_data[0, 0]
                    else:
                        return None

                # Normalize the data
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

                if is_rgb:
                    # Handle RGB FITS - normalize each channel separately
                    normalized_data = np.zeros_like(image_data)
                    for channel in range(3):
                        channel_data = image_data[:, :, channel]
                        try:
                            norm = simple_norm(channel_data, stretch='linear', percent=99.5)
                            normalized_data[:, :, channel] = norm(channel_data)
                        except Exception:
                            data_min, data_max = np.percentile(channel_data, [0.5, 99.5])
                            if data_max > data_min:
                                normalized_data[:, :, channel] = (channel_data - data_min) / (data_max - data_min)
                            else:
                                normalized_data[:, :, channel] = channel_data

                    # Clip and convert to 8-bit RGB
                    normalized_data = np.clip(normalized_data, 0, 1)
                    rgb_data = (normalized_data * 255).astype(np.uint8)

                    if not rgb_data.flags['C_CONTIGUOUS']:
                        rgb_data = np.ascontiguousarray(rgb_data)

                    height, width, channels = rgb_data.shape
                    bytes_per_line = width * channels
                    qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    # Handle grayscale FITS
                    try:
                        norm = simple_norm(image_data, stretch='linear', percent=99.5)
                        normalized_data = norm(image_data)
                    except Exception:
                        data_min, data_max = np.percentile(image_data, [0.5, 99.5])
                        if data_max > data_min:
                            normalized_data = (image_data - data_min) / (data_max - data_min)
                        else:
                            normalized_data = image_data
                        normalized_data = np.clip(normalized_data, 0, 1)

                    # Convert to 8-bit grayscale
                    image_8bit = (normalized_data * 255).astype(np.uint8)

                    if not image_8bit.flags['C_CONTIGUOUS']:
                        image_8bit = np.ascontiguousarray(image_8bit)

                    height, width = image_8bit.shape
                    bytes_per_line = width
                    qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                # Convert to QPixmap
                return QPixmap.fromImage(qimage)

        except Exception as e:
            return None

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

                # Handle FITS files
                if ext in ['.fits', '.fit', '.fts']:
                    pixmap = self._load_fits_thumbnail(self.image_path)
                    if pixmap is None:
                        self.signals.thumbnail_error.emit(self.card, "FITS Load Error")
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


class AddImageDialog(WindowPositionMixin, QDialog):
    """Dialog for adding a new image to a DSO"""

    WINDOW_POSITION_KEY = "AddImageDialog"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Image to DSO")
        self.setMinimumWidth(250)
        self.selected_file = None
        self.dso_data = []  # List of (dsodetailid, name) tuples

        self._init_ui()
        self._load_dso_list()
        self._load_equipment_list()
        self.setup_window_position()

    def _init_ui(self):
        """Create the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Instructions
        instructions = QLabel("Select an image file and choose which DSO to attach it to.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Form layout for inputs
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # Image file selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected...")
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)
        form_layout.addRow("Image File:", file_layout)

        # DSO selection with search
        self.dso_combo = QComboBox()
        self.dso_combo.setEditable(True)
        self.dso_combo.setInsertPolicy(QComboBox.NoInsert)
        self.dso_combo.lineEdit().setPlaceholderText("Search for DSO...")
        self.dso_combo.setMinimumWidth(300)
        form_layout.addRow("Attach to DSO:", self.dso_combo)

        # Optional metadata fields
        self.telescope_combo = QComboBox()
        self.telescope_combo.setEditable(True)
        self.telescope_combo.setInsertPolicy(QComboBox.NoInsert)
        self.telescope_combo.lineEdit().setPlaceholderText("e.g., 8\" SCT")
        form_layout.addRow("Telescope:", self.telescope_combo)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.setInsertPolicy(QComboBox.NoInsert)
        self.camera_combo.lineEdit().setPlaceholderText("e.g., ASI294MC Pro")
        form_layout.addRow("Camera:", self.camera_combo)

        self.integration_edit = QLineEdit()
        self.integration_edit.setPlaceholderText("e.g., 2h 30m")
        form_layout.addRow("Integration Time:", self.integration_edit)

        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("e.g., 2024-01-15")
        form_layout.addRow("Date Taken:", self.date_edit)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional notes about this image")
        form_layout.addRow("Notes:", self.notes_edit)

        layout.addLayout(form_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

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
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fts);;"
            "PNG Files (*.png);;"
            "JPEG Files (*.jpg *.jpeg);;"
            "TIFF Files (*.tif *.tiff);;"
            "FITS Files (*.fits *.fit *.fts);;"
            "All Files (*.*)"
        )
        if file_name:
            self.selected_file = file_name
            self.file_path_edit.setText(file_name)

    def set_file_path(self, path):
        """Pre-populate the file path (e.g. from drag-and-drop)"""
        self.selected_file = path
        self.file_path_edit.setText(path)

    def _validate_and_accept(self):
        """Validate inputs before accepting"""
        if not self.selected_file:
            QMessageBox.warning(self, "Missing Image", "Please select an image file.")
            return

        if not os.path.exists(self.selected_file):
            QMessageBox.warning(self, "File Not Found", "The selected image file does not exist.")
            return

        if self.dso_combo.currentIndex() < 0:
            QMessageBox.warning(self, "No DSO Selected", "Please select a DSO to attach the image to.")
            return

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
            'notes': self.notes_edit.text().strip()
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
    _SUPPORTED_DROP_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit', '.fts'}

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
        self.search_input.textChanged.connect(lambda: self.search_timer.start(300))
        filters_layout.addWidget(self.search_input)

        # Catalog filter
        filters_layout.addWidget(QLabel("Catalog:"))
        self.catalog_combo = QComboBox()
        self.catalog_combo.addItems(["All", "M", "NGC", "IC", "Sh2", "B", "Cr", "Mel"])
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
        self.filtered_items = self.all_items.copy()
        self.data_loaded = True  # Mark data as loaded

        # Populate filter dropdowns
        self._populate_type_filter()
        self._populate_equipment_filter()

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
        self.type_combo.clear()
        self.type_combo.addItem("All")
        for dso_type in sorted(types):
            self.type_combo.addItem(dso_type)

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
        self.equipment_combo.clear()
        self.equipment_combo.addItem("All")
        for equipment in sorted(equipment_set):
            self.equipment_combo.addItem(equipment)

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

            # Load the image into a QPixmap
            pixmap = QPixmap(image_path)

            # Check if image loaded successfully
            if pixmap.isNull():
                # Try FITS format if standard loading failed
                _, ext = os.path.splitext(image_path.lower())
                if ext in ['.fits', '.fit', '.fts']:
                    # Use FITS loader
                    from astropy.io import fits
                    from astropy.visualization import simple_norm

                    try:
                        with fits.open(image_path) as hdul:
                            image_data = None
                            for hdu in hdul:
                                if hdu.data is not None and len(hdu.data.shape) >= 2:
                                    image_data = hdu.data
                                    break

                            if image_data is None:
                                raise ValueError("No valid image data in FITS file")

                            # Handle different dimensionalities
                            is_rgb = False
                            if len(image_data.shape) > 2:
                                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                                    is_rgb = True
                                elif len(image_data.shape) == 3 and image_data.shape[0] == 3:
                                    image_data = np.transpose(image_data, (1, 2, 0))
                                    is_rgb = True
                                elif len(image_data.shape) == 3:
                                    image_data = image_data[0]
                                elif len(image_data.shape) == 4:
                                    image_data = image_data[0, 0]

                            # Normalize
                            image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

                            if is_rgb:
                                # RGB FITS - normalize each channel
                                normalized_data = np.zeros_like(image_data)
                                for channel in range(3):
                                    channel_data = image_data[:, :, channel]
                                    try:
                                        norm = simple_norm(channel_data, stretch='linear', percent=99.5)
                                        normalized_data[:, :, channel] = norm(channel_data)
                                    except Exception:
                                        data_min, data_max = np.percentile(channel_data, [0.5, 99.5])
                                        if data_max > data_min:
                                            normalized_data[:, :, channel] = (channel_data - data_min) / (data_max - data_min)
                                        else:
                                            normalized_data[:, :, channel] = channel_data

                                normalized_data = np.clip(normalized_data, 0, 1)
                                rgb_data = (normalized_data * 255).astype(np.uint8)
                                if not rgb_data.flags['C_CONTIGUOUS']:
                                    rgb_data = np.ascontiguousarray(rgb_data)

                                height, width, channels = rgb_data.shape
                                bytes_per_line = width * channels
                                qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            else:
                                # Grayscale FITS
                                try:
                                    norm = simple_norm(image_data, stretch='linear', percent=99.5)
                                    normalized_data = norm(image_data)
                                except Exception:
                                    data_min, data_max = np.percentile(image_data, [0.5, 99.5])
                                    if data_max > data_min:
                                        normalized_data = (image_data - data_min) / (data_max - data_min)
                                    else:
                                        normalized_data = image_data
                                    normalized_data = np.clip(normalized_data, 0, 1)

                                image_8bit = (normalized_data * 255).astype(np.uint8)
                                if not image_8bit.flags['C_CONTIGUOUS']:
                                    image_8bit = np.ascontiguousarray(image_8bit)

                                height, width = image_8bit.shape
                                qimage = QImage(image_8bit.data, width, height, width, QImage.Format_Grayscale8)

                            pixmap = QPixmap.fromImage(qimage)
                    except Exception as fits_error:
                        QMessageBox.critical(self, "Error",
                                           f"Failed to load FITS image:\n{str(fits_error)}")
                        return
                else:
                    QMessageBox.critical(self, "Error",
                                       f"Failed to load image:\n{image_path}\n\n"
                                       f"The file may be corrupted or in an unsupported format.")
                    return

            # Create and show image viewer window (pixmap, title, file_path, parent, dso_ra, dso_dec)
            ra_deg = item_data.get('ra_deg')
            dec_deg = item_data.get('dec_deg')
            self.image_viewer = ImageViewerWindow(
                pixmap, item_data['name'], image_path, self,
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

        # Filter items
        self.filtered_items = [item for item in self.all_items if self._matches_filters(item)]

        # Apply sorting
        self._sort_items()

        # Refresh grid
        self._populate_grid()

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
