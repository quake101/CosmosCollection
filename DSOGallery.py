#!/usr/bin/env python3
"""
DSO Image Gallery
Displays all DSO objects with images in a responsive grid gallery format
"""

import sys
import os
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QGroupBox,
                               QMessageBox, QScrollArea, QComboBox, QLineEdit,
                               QFrame, QGridLayout, QMenu)
from PySide6.QtGui import QPixmap, QImage

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
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


class ThumbnailWorker(QThread):
    """Worker thread for generating thumbnails in background"""

    # Signals
    thumbnail_ready = Signal(object, QPixmap)  # card, pixmap
    thumbnail_error = Signal(object, str)      # card, error_message

    def __init__(self, image_requests, cache=None):
        """
        Initialize thumbnail worker

        Args:
            image_requests: List of (card, image_path) tuples
            cache: ThumbnailCache instance
        """
        super().__init__()
        self.image_requests = image_requests
        self.cache = cache
        self.cancelled = False

    def cancel(self):
        """Cancel the thumbnail generation"""
        self.cancelled = True

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
        """Generate thumbnails in background"""
        from PySide6.QtGui import QImageReader

        for card, image_path in self.image_requests:
            if self.cancelled:
                break

            try:
                # Check cache first
                if self.cache:
                    cached_pixmap = self.cache.get(image_path)
                    if cached_pixmap:
                        self.thumbnail_ready.emit(card, cached_pixmap)
                        continue

                if os.path.exists(image_path):
                    # Check file size
                    file_size = os.path.getsize(image_path)
                    if file_size == 0:
                        self.thumbnail_error.emit(card, "Empty File")
                        continue

                    # Get file extension
                    _, ext = os.path.splitext(image_path.lower())

                    pixmap = None

                    # Handle FITS files
                    if ext in ['.fits', '.fit', '.fts']:
                        pixmap = self._load_fits_thumbnail(image_path)
                        if pixmap is None:
                            self.thumbnail_error.emit(card, "FITS Load Error")
                            continue
                    else:
                        # Load regular image formats
                        QImageReader.setAllocationLimit(512)

                        # Try standard QPixmap loading
                        pixmap = QPixmap(image_path)

                        # If failed, try QImageReader
                        if pixmap.isNull():
                            try:
                                reader = QImageReader(image_path)
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

                    if pixmap and not pixmap.isNull():
                        # Scale to 150x150 (gallery card size)
                        scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                        # Cache the thumbnail
                        if self.cache:
                            self.cache.put(image_path, scaled_pixmap)

                        self.thumbnail_ready.emit(card, scaled_pixmap)
                    else:
                        error_msg = f"Load Error"
                        self.thumbnail_error.emit(card, error_msg)
                else:
                    self.thumbnail_error.emit(card, "File Not Found")

            except Exception as e:
                self.thumbnail_error.emit(card, f"Error: {str(e)[:20]}")


class GalleryCard(QFrame):
    """Individual card widget displaying a DSO thumbnail and info"""

    double_clicked = Signal(dict)  # Emits item_data when double-clicked
    context_menu_requested = Signal(dict, object)  # Emits item_data and position

    def __init__(self, item_data, parent=None):
        """
        Initialize gallery card

        Args:
            item_data (dict): Dictionary containing DSO data
                - dsodetailid: DSO ID
                - name: DSO name
                - dsotype: DSO type code
                - image_path: Path to favorite image
                - equipment: Equipment used
        """
        super().__init__(parent)
        self.item_data = item_data
        self._init_ui()

    def _init_ui(self):
        """Create card layout with thumbnail and labels"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Thumbnail label (150x150)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(150, 150)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 3px;
            }
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
        type_label.setStyleSheet("font-size: 10px; color: #cccccc;")
        type_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(type_label)

        # Card styling
        self.setFixedWidth(170)
        self.setStyleSheet("""
            GalleryCard {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 5px;
            }
            GalleryCard:hover {
                border: 2px solid #0078d4;
                background-color: #4a4a4a;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)

    def set_thumbnail(self, pixmap):
        """Update thumbnail with actual image"""
        if pixmap and not pixmap.isNull():
            # Scale to fit 150x150 while maintaining aspect ratio
            scaled = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumbnail_label.setPixmap(scaled)
            self.thumbnail_label.setText("")  # Clear placeholder text

    def set_error(self, error_message):
        """Display error on card"""
        self.thumbnail_label.setText(f"Error:\n{error_message[:30]}")
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #4a2020;
                border: 1px solid #aa4444;
                border-radius: 3px;
                color: #ff8888;
            }
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

    def __init__(self):
        """Initialize DSO Image Gallery window"""
        super().__init__()
        self.setWindowTitle("DSO Image Gallery - Cosmos Collection")
        self.resize(1200, 800)

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QLineEdit, QComboBox {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #404040;
                color: #ffffff;
                selection-background-color: #0078d4;
                border: 1px solid #666666;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        # Initialize data structures
        self.db_manager = DatabaseManager()
        self.all_items = []
        self.filtered_items = []
        self.current_columns = 1
        self.current_filters = {
            'search': '',
            'catalog': 'All',
            'type': 'All',
            'equipment': 'All'
        }

        # Thumbnail cache and worker
        self.thumbnail_cache = ThumbnailCache(max_size=200)
        self.thumbnail_worker = None

        # Search debounce timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._apply_filters)

        # Initialize UI
        self._init_ui()

        # Setup window position persistence
        self.setup_window_position()

        # Load data (defer grid population until window is shown)
        self._initial_load_pending = True
        self._load_gallery_items()

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
        self.type_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.type_combo)

        # Equipment filter
        filters_layout.addWidget(QLabel("Equipment:"))
        self.equipment_combo = QComboBox()
        self.equipment_combo.addItem("All")
        self.equipment_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.equipment_combo)

        # Clear filters button
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_filters)
        filters_layout.addWidget(clear_btn)

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

        # Status label
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("padding: 5px;")
        main_layout.addWidget(self.status_label)

    def _load_gallery_items(self):
        """Load DSOs with favorite images from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Query DSOs with images (prefer favorite, but show any image if no favorite)
                query = """
                SELECT DISTINCT
                    d.id as dsodetailid,
                    (SELECT image_path FROM userimages WHERE dsodetailid = d.id ORDER BY is_favorite DESC, id ASC LIMIT 1) as image_path,
                    (SELECT equipment FROM userimages WHERE dsodetailid = d.id ORDER BY is_favorite DESC, id ASC LIMIT 1) as equipment,
                    (SELECT is_favorite FROM userimages WHERE dsodetailid = d.id ORDER BY is_favorite DESC, id ASC LIMIT 1) as is_favorite,
                    d.dsotype,
                    d.constellation,
                    GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                        ORDER BY
                            CASE c.catalogue
                                WHEN 'M' THEN 1
                                WHEN 'NGC' THEN 2
                                WHEN 'IC' THEN 3
                                ELSE 4
                            END, c.designation) as name
                FROM dsodetail d
                INNER JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE EXISTS (SELECT 1 FROM userimages WHERE dsodetailid = d.id)
                GROUP BY d.id
                ORDER BY name
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                # Convert rows to dictionaries
                self.all_items = []
                for row in rows:
                    item = {
                        'dsodetailid': row[0],
                        'image_path': row[1],
                        'equipment': row[2] or '',
                        'is_favorite': row[3],
                        'dsotype': row[4] or '',
                        'constellation': row[5] or '',
                        'name': row[6] or 'Unknown',
                        'friendly_type': self._get_friendly_type_name(row[4] or '')
                    }
                    self.all_items.append(item)

                # Initially, filtered items = all items
                self.filtered_items = self.all_items.copy()

                # Populate filter dropdowns
                self._populate_type_filter()
                self._populate_equipment_filter()

                # Update status
                count = len(self.all_items)
                self.status_label.setText(f"Loaded {count} DSO{'s' if count != 1 else ''} with images")

                # Populate grid (unless initial load - will be done in showEvent)
                if not self._initial_load_pending:
                    self._populate_grid()

        except Exception as e:
            self.status_label.setText(f"Error loading gallery: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load DSO gallery:\n{str(e)}")

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
            "SMCGC": "SMC Globular Cluster",
            "SMCCN": "SMC Cluster/Nebula",
            "SMCOC": "SMC Open Cluster"
        }
        return type_mapping.get(dso_type, dso_type)

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

    def _calculate_grid_columns(self):
        """Calculate number of columns based on available width"""
        card_width = 170  # Card width + spacing
        available_width = self.scroll_area.width() - 30  # Subtract margins and scrollbar
        columns = max(1, available_width // card_width)
        return min(columns, 8)  # Cap at 8 columns

    def _populate_grid(self):
        """Populate grid with gallery cards"""
        # Clear existing cards
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Check if there are items to display
        if not self.filtered_items:
            # Show message when no items
            no_items_label = QLabel("No DSO images found matching your filters")
            no_items_label.setStyleSheet("font-size: 14px; color: #888888; padding: 50px;")
            no_items_label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(no_items_label, 0, 0)
            self.status_label.setText(f"Showing 0 of {len(self.all_items)} DSOs")
            return

        # Calculate columns based on window width
        cols = self._calculate_grid_columns()
        self.current_columns = cols

        # Store cards for thumbnail loading
        self.cards = []

        # Create and add cards to grid
        for idx, item in enumerate(self.filtered_items):
            row = idx // cols
            col = idx % cols

            # Create card
            card = GalleryCard(item)
            card.double_clicked.connect(self._on_card_double_clicked)
            card.context_menu_requested.connect(self._show_card_context_menu)

            # Add to grid
            self.grid_layout.addWidget(card, row, col)

            # Store reference
            self.cards.append(card)

        # Add stretch to push cards to top-left
        self.grid_layout.setRowStretch(len(self.filtered_items) // cols + 1, 1)
        self.grid_layout.setColumnStretch(cols, 1)

        # Update status
        showing = len(self.filtered_items)
        total = len(self.all_items)
        if showing == total:
            self.status_label.setText(f"Showing all {total} DSO{'s' if total != 1 else ''}")
        else:
            self.status_label.setText(f"Showing {showing} of {total} DSOs")

        # Load thumbnails in background
        self._load_thumbnails()

    def _load_thumbnails(self):
        """Load thumbnails in background thread"""
        # Cancel existing worker if running
        if self.thumbnail_worker and self.thumbnail_worker.isRunning():
            self.thumbnail_worker.cancel()
            self.thumbnail_worker.wait()
            # Disconnect old signals to prevent updates to deleted cards
            try:
                self.thumbnail_worker.thumbnail_ready.disconnect()
                self.thumbnail_worker.thumbnail_error.disconnect()
            except:
                pass

        # Create list of thumbnail requests (card, image_path)
        requests = []
        for card in self.cards:
            image_path = card.item_data['image_path']
            requests.append((card, image_path))

        # Start thumbnail worker
        if requests:
            self.thumbnail_worker = ThumbnailWorker(requests, self.thumbnail_cache)
            self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
            self.thumbnail_worker.thumbnail_error.connect(self._on_thumbnail_error)
            self.thumbnail_worker.start()

    def _on_thumbnail_ready(self, card, pixmap):
        """Handle thumbnail loaded successfully"""
        try:
            # Check if card still exists (hasn't been deleted by grid refresh)
            if card and not card.isHidden() and card.parent():
                card.set_thumbnail(pixmap)
        except RuntimeError:
            # Card was deleted, ignore
            pass

    def _on_thumbnail_error(self, card, error_message):
        """Handle thumbnail load error"""
        try:
            # Check if card still exists (hasn't been deleted by grid refresh)
            if card and not card.isHidden() and card.parent():
                card.set_error(error_message)
        except RuntimeError:
            # Card was deleted, ignore
            pass

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

            # Create and show image viewer window (pixmap, title, file_path, parent)
            self.image_viewer = ImageViewerWindow(pixmap, item_data['name'], image_path, self)
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
            from main import ObjectDetailWindow

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
                detail_window = ObjectDetailWindow(data)
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

        # Apply dark theme styling
        context_menu.setStyleSheet("""
            QMenu {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
            QMenu::separator {
                height: 1px;
                background-color: #666666;
                margin: 5px 0;
            }
        """)

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

    def _apply_filters(self):
        """Apply all filters and refresh grid"""
        # Update current filter state
        self.current_filters['search'] = self.search_input.text().strip()
        self.current_filters['catalog'] = self.catalog_combo.currentText()
        self.current_filters['type'] = self.type_combo.currentText()
        self.current_filters['equipment'] = self.equipment_combo.currentText()

        # Filter items
        self.filtered_items = [item for item in self.all_items if self._matches_filters(item)]

        # Refresh grid
        self._populate_grid()

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

    def showEvent(self, event):
        """Handle window show - populate grid on first show"""
        super().showEvent(event)
        if self._initial_load_pending:
            self._initial_load_pending = False
            # Use QTimer to ensure window is fully laid out
            QTimer.singleShot(0, self._populate_grid)

    def resizeEvent(self, event):
        """Handle window resize - recalculate grid if needed"""
        super().resizeEvent(event)
        new_cols = self._calculate_grid_columns()
        if new_cols != self.current_columns:
            self.current_columns = new_cols
            self._populate_grid()

    def closeEvent(self, event):
        """Handle window close - cleanup thumbnail worker"""
        # Cancel thumbnail worker if running
        if self.thumbnail_worker and self.thumbnail_worker.isRunning():
            self.thumbnail_worker.cancel()
            self.thumbnail_worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = DSOGalleryWindow()
    window.show()
    sys.exit(app.exec())
