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
    QToolBar, QMessageBox
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
    """Worker for performing heavy visibility calculations in a background thread"""
    finished = Signal(str)  # Signal with visibility text result
    error = Signal(str)  # Signal for error reporting

    def __init__(self, lat: float, lon: float, ra_deg: float, dec_deg: float, object_name: str):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.object_name = object_name

    def calculate_visibility(self):
        """Calculate visibility seasons in background thread"""
        try:
            from datetime import date, timedelta
            
            observer = Observer(latitude=self.lat * u.deg, longitude=self.lon * u.deg, name="User Location")
            target = FixedTarget(coord=SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg), 
                               name=self.object_name)

            # Start from today and check the next 365 days
            today = date.today()
            observable_dates = []

            for day_offset in range(365):
                current_date = today + timedelta(days=day_offset)
                
                # Create midnight time for this date (local solar midnight)
                midnight_time = Time(f"{current_date.year}-{current_date.month:02d}-{current_date.day:02d} 00:00:00", 
                                   location=observer.location)
                
                # Check conditions at local solar midnight (best viewing time)
                try:
                    # Get sun position at midnight
                    sun_altaz = observer.sun_altaz(midnight_time)
                    
                    # Only consider if sun is down (astronomical twilight or darker)
                    if sun_altaz.alt.degree < -12:
                        # Get target position at midnight
                        target_altaz = observer.altaz(midnight_time, target)
                        
                        # Object is well-visible if altitude > 30 degrees
                        if target_altaz.alt.degree > 30:
                            observable_dates.append(current_date)
                except:
                    # If calculation fails for this date, skip it
                    continue

            if not observable_dates:
                visibility_text = "Object not optimally visible from your location this year.<br>Try checking the Visibility Calculator for detailed viewing times."
            else:
                # Group contiguous dates as visibility seasons
                seasons = []
                if observable_dates:
                    start_date = observable_dates[0]
                    end_date = start_date
                    
                    for i in range(1, len(observable_dates)):
                        # Allow small gaps (1-2 days) to be part of same season
                        if (observable_dates[i] - end_date).days <= 3:
                            end_date = observable_dates[i]
                        else:
                            # Only add seasons that are at least 7 days long
                            if (end_date - start_date).days >= 6:
                                seasons.append((start_date, end_date))
                            start_date = observable_dates[i]
                            end_date = start_date
                    
                    # Don't forget the last season
                    if (end_date - start_date).days >= 6:
                        seasons.append((start_date, end_date))

                if not seasons:
                    visibility_text = "Object has brief visibility periods.<br>Check the Visibility Calculator for specific viewing times."
                else:
                    # Format season strings
                    season_strs = []
                    for start, end in seasons:
                        if start == end:
                            season_strs.append(f"{start.strftime('%B %d')}")
                        else:
                            season_strs.append(f"{start.strftime('%B %d')} - {end.strftime('%B %d')}")
                    
                    visibility_text = f"Best viewing seasons (>30° altitude at midnight):<br>" + "<br>".join(season_strs)
                    visibility_text += "<br><br><small>Times shown are when object is highest in dark sky.<br>Use Visibility Calculator for detailed times.</small>"

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

    def filter_data(self, search_text, selected_catalog=None, show_images_only=False):
        """Filter the data based on search text, catalog, and image presence"""
        self.layoutAboutToBeChanged.emit()

        # Store the selected catalog for use in data() method
        self.selected_catalog = selected_catalog

        if not search_text and not selected_catalog and not show_images_only:
            self.filtered_data = self.dso_data.copy()
        else:
            search_text = search_text.lower() if search_text else ""
            self.filtered_data = [
                item for item in self.dso_data
                if ((not selected_catalog or
                     selected_catalog == "All Catalogs" or
                     any(designation.startswith(selected_catalog + " ")
                         for designation in item["designations"].split(", "))) and
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
        self.setWindowTitle(f"DSO Visibility Calculator - {dso_name}")
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
class AladinLiteWindow(QDialog):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Aladin Lite - {data['name']}")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.resize(1200, 800)

        # Calculate FOV based on object size
        # Use the larger dimension and scale it down for better viewing
        size_max = max(data['size_min'], data['size_max'])
        fov = size_max * 0.05  # Scale down to 5% of the object's size

        logger.debug(f"Size values for {data['name']}: min={data['size_min']:.1f}', max={data['size_max']:.1f}'")
        logger.debug(f"Calculated FOV: {fov:.2f}'")

        # Create main layout
        layout = QVBoxLayout()

        # Add Aladin Lite viewer
        self.web_view = QWebEngineView()
        base_url = "https://aladin.u-strasbg.fr/AladinLite/?target="
        
        # Use multiple fallback methods to get target
        target_id = data.get('dsodetailid', '')
        if not target_id:
            # Try using object name directly
            target_id = data.get('name', '')
        if not target_id and 'ra_deg' in data and 'dec_deg' in data:
            # Use coordinates if available
            ra = data['ra_deg']
            dec = data['dec_deg']
            target_id = f"{ra}+{dec}"
        
        image_url = f"{base_url}{target_id}&fov={fov}&survey=P%2FDSS2%2Fcolor"
        self.web_view.setUrl(QUrl(image_url))
        layout.addWidget(self.web_view)

        # Create a horizontal layout for the bottom controls
        bottom_layout = QHBoxLayout()

        # Add compact size information
        size_info = QLabel(f"FOV: {fov:.2f}' | Size: {data['size_min']:.1f}'—{data['size_max']:.1f}'")
        size_info.setStyleSheet("font-size: 10pt;")
        bottom_layout.addWidget(size_info)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom_layout.addWidget(close_button)

        # Add the bottom layout to the main layout
        layout.addLayout(bottom_layout)

        self.setLayout(layout)
        logger.debug(f"Opened Aladin Lite window with FOV: {fov:.2f}'")


# --- Image Viewer Window ---
class ImageViewerWindow(QDialog):
    """Window to display an image in full size with enhanced controls"""
    zoom_changed = Signal(float)  # Signal for zoom level changes

    def __init__(self, pixmap: QPixmap, title: str, file_path: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Image Viewer - {title}")
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
        layout = QVBoxLayout()

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

        layout.addLayout(toolbar)

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
        layout.addWidget(self.image_container)

        # Add status bar
        self.status_bar = QLabel()
        self.status_bar.setStyleSheet("font-size: 10pt;")
        layout.addWidget(self.status_bar)

        # Set stretch to ensure image container takes most space and status bar stays at bottom
        layout.setStretch(0, 0)  # toolbar
        layout.setStretch(1, 1)  # image_container
        layout.setStretch(2, 0)  # status_bar

        self.setLayout(layout)

        # Calculate initial zoom to fit window
        self._fit_to_window()

        # Update status
        self._update_status()

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
        # Only update the zoom if we're at the initial zoom level
        if self.zoom_factor == self.initial_zoom_factor:
            self._fit_to_window()
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


# --- Object Detail Window ---
class ObjectDetailWindow(QDialog):
    image_added = Signal()  # Add this signal at the class level

    def __init__(self, data: dict, parent=None):
        super().__init__(None)  # Pass None as parent to make it independent
        logger.debug(f"Creating ObjectDetailWindow for {data['name']}")
        self.setWindowTitle(data["name"])
        # Make it an independent window with window management buttons
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.NonModal)  # Ensure it's non-modal
        self.resize(1200, 800)
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

            zoom_layout.addStretch()
            image_container_layout.addLayout(zoom_layout)

            # Add image label
            self.image_label = QLabel("No Image Loaded")
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

            # Add image button
            user_image_button = QPushButton("Add User Image")
            user_image_button.clicked.connect(self._add_user_image)
            left_layout.addWidget(user_image_button)
            
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

            # Add Object information
            object_info_text = (
                f"<b>Right Ascension:</b> {self.ra_str}<br>"
                f"<b>Declination:</b> {self.dec_str}<br>"
                f"<b>Constellation:</b> {self.data['constellation']}<br><br>"
                f"<b>Magnitude:</b> {self.data['magnitude']:.2f}<br>"
                f"<b>Surface Brightness:</b> {self.data['surface_brightness']:.2f} mag/arcmin²<br>"
                f"<b>Size:</b> {self.data['size_min']:.1f}' — {self.data['size_max']:.1f}'<br>"
                f"<b>Type:</b> {self.data['dso_type']}<br>"
                f"<b>Class:</b> {self.data['dso_class']}<br><br>"
                f"<b>Other Designations:</b><br>"
            )

            # Add object designations with proper formatting
            logger.debug(f"Designations data: {self.data.get('designations')}")
            if self.data.get('designations'):
                designations = self.data['designations'].split(', ')
                logger.debug(f"Split designations: {designations}")
                # Skip the first designation as it's the primary one
                for designation in designations[1:]:
                    object_info_text += f"{designation}<br>"

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
        # Only update the zoom if we're at the initial zoom level
        if self.zoom_factor == self.initial_zoom_factor:
            self._fit_to_window()
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
            self._update_visibility_info()
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

    def _update_visibility_info(self):
        """Calculate and display visibility season/dates using astroplan"""
        try:
            lat_text = self.location_lat_edit.text().strip()
            lon_text = self.location_lon_edit.text().strip()
            if not lat_text or not lon_text:
                self.visibility_label.setText("Enter your location to see visibility information.")
                return
            lat = float(lat_text)
            lon = float(lon_text)
            observer = Observer(latitude=lat * u.deg, longitude=lon * u.deg, name="User Location")
            ra = self.data.get("ra_deg")
            dec = self.data.get("dec_deg")
            target = FixedTarget(coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), name=self.data.get("name"))

            # Use current year and calculate next visibility window
            time_now = Time.now()
            midnight = time_now.midnight()
            delta_days = range(0, 365)
            observable_dates = []

            for day in delta_days:
                obs_time = midnight + day * u.day
                altaz = observer.altaz(obs_time, target)
                if altaz.alt.degree > 20:  # Visible if altitude > 20 degrees
                    observable_dates.append(obs_time.datetime.date())

            if not observable_dates:
                visibility_text = "Object not visible from your location this year."
            else:
                # Group contiguous dates as visibility seasons
                seasons = []
                start_date = observable_dates[0]
                end_date = start_date
                for i in range(1, len(observable_dates)):
                    if (observable_dates[i] - end_date).days == 1:
                        end_date = observable_dates[i]
                    else:
                        seasons.append((start_date, end_date))
                        start_date = observable_dates[i]
                        end_date = start_date
                seasons.append((start_date, end_date))

                # Format season strings
                season_strs = [f"{s[0].strftime('%b %d')} - {s[1].strftime('%b %d')}" for s in seasons]
                visibility_text = "Visibility season(s):<br>" + "<br>".join(season_strs)

            self.visibility_label.setText(visibility_text)
        except Exception as e:
            logger.error(f"Error calculating visibility: {str(e)}")
            self.visibility_label.setText("Error calculating visibility information.")

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
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
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

    def _load_user_image(self, image_path):
        """Load and display a user image with caching"""
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
                # Create QPixmap from the image file
                logger.debug("Loading image from file")
                self.original_pixmap = QPixmap(image_path)
                if self.original_pixmap.isNull():
                    logger.error(f"QPixmap failed to load image: {image_path}")
                    # Try to get more specific error information
                    reader = QImageReader(image_path)
                    if reader.canRead():
                        logger.debug(f"QImageReader can read the file, format: {reader.format()}")
                    else:
                        logger.error(f"QImageReader cannot read file, error: {reader.errorString()}")
                    
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
            aladin_window = AladinLiteWindow(data, self)
            aladin_window.setModal(False)  # Make window non-modal
            aladin_window.show()
            logger.debug(f"Opened Aladin Lite window for {data['name']}")
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
            self.image_counter_label.setText("0/0")
        elif image_count == 1:
            # Only one image
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.image_counter_label.setText("1/1")
        else:
            # Multiple images
            self.prev_image_button.setEnabled(self.current_image_index > 0)
            self.next_image_button.setEnabled(self.current_image_index < image_count - 1)
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
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif);;All Files (*)"
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
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

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

        # Description
        desc_label = QLabel("Your personal astrophotography catalog and image viewer application.")
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

        # Add stretch to push everything up
        About_layout.addStretch()

        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedSize(80, 30)
        close_button.clicked.connect(self.close)
        close_button.setDefault(True)
        
        # Center the close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        
        About_layout.addLayout(button_layout)


# --- Main App Window ---
class MainWindow(QMainWindow):
    def __init__(self, dso_data, catalogs):
        super().__init__()
        logger.debug("Initializing MainWindow")
        self.setWindowTitle("Cosmos Collection")
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

    def _on_show_images_changed(self, state):
        """Handle show images only checkbox state change"""
        self.model.filter_data(
            self.search_input.text(),
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.show_images_only.isChecked()
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
        self.model.filter_data("", None, False)
        self._update_status()

    def _on_search(self, text):
        """Handle search text changes"""
        self.model.filter_data(
            text,
            None if self.catalog_combo.currentText() == "All Catalogs" else self.catalog_combo.currentText(),
            self.show_images_only.isChecked()
        )
        self._update_status()

    def _on_catalog_changed(self, catalog):
        """Handle catalog selection changes"""
        self.model.filter_data(
            self.search_input.text(),
            None if catalog == "All Catalogs" else catalog,
            self.show_images_only.isChecked()
        )
        self._update_status()

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
            "name": f"{catalogue} {designation}",
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
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ') as designations,
                           ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                           (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    LEFT JOIN userimages ui ON d.id = ui.dsodetailid
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
                self.show_images_only.isChecked()
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