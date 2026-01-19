#!/usr/bin/env python3
"""
ImageViewer module for Cosmos Collection
Provides the ImageViewerWindow class for displaying images with zoom, pan, and annotation support.
"""

import logging
import platform
import ctypes
import subprocess
import os
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QSettings
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QGroupBox, QScrollArea, QFileDialog, QMessageBox, QCheckBox, QProgressBar
)

from Theme import COLORS
from WindowPositionManager import WindowPositionManager
from ResourceManager import ResourceManager

# Set up logging
logger = logging.getLogger(__name__)


class ImageViewerWindow(QDialog):
    """Window to display an image in full size with enhanced controls"""
    zoom_changed = Signal(float)  # Signal for zoom level changes

    def __init__(self, pixmap: QPixmap, title: str, file_path: str = None, parent=None,
                 dso_ra: float = None, dso_dec: float = None):
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

        # Store DSO coordinates for plate solving hints
        self.dso_ra = dso_ra  # RA in degrees
        self.dso_dec = dso_dec  # Dec in degrees

        # Create main layout
        main_layout = QVBoxLayout()

        # Create toolbar for controls
        toolbar = QHBoxLayout()

        # Add zoom controls
        zoom_out_button = QPushButton("-")
        zoom_out_button.setStyleSheet("QPushButton { font-size: 12pt; }")
        zoom_out_button.setToolTip("Zoom out")
        zoom_out_button.clicked.connect(self._zoom_out)
        toolbar.addWidget(zoom_out_button)

        zoom_in_button = QPushButton("+")
        zoom_in_button.setStyleSheet("QPushButton { font-size: 12pt; }")
        zoom_in_button.setToolTip("Zoom in")
        zoom_in_button.clicked.connect(self._zoom_in)
        toolbar.addWidget(zoom_in_button)

        reset_button = QPushButton("Reset")
        reset_button.setFixedHeight(30)
        reset_button.setToolTip("Reset zoom to 100%")
        reset_button.clicked.connect(self._reset_zoom)
        toolbar.addWidget(reset_button)

        # Add file-specific buttons if file path is available
        if self.file_path:
            open_location_button = QPushButton("Open File Location")
            open_location_button.setFixedHeight(30)
            open_location_button.setToolTip("Open the folder containing this image")
            open_location_button.clicked.connect(self._open_file_location)
            toolbar.addWidget(open_location_button)

            set_bg_button = QPushButton("Set as Background")
            set_bg_button.setFixedHeight(30)
            set_bg_button.setToolTip("Set this image as your desktop background")
            set_bg_button.clicked.connect(self._set_as_background)
            toolbar.addWidget(set_bg_button)

            self.annotations_button = QPushButton("Show Annotations")
            self.annotations_button.setFixedHeight(30)
            self.annotations_button.setToolTip("Configure and display star/DSO annotations")
            self.annotations_button.clicked.connect(self._show_annotations_dialog)
            toolbar.addWidget(self.annotations_button)

            self.save_annotated_button = QPushButton("Save with Annotations")
            self.save_annotated_button.setFixedHeight(30)
            self.save_annotated_button.setToolTip("Save image with annotations overlaid")
            self.save_annotated_button.clicked.connect(self._save_with_annotations)
            toolbar.addWidget(self.save_annotated_button)

        toolbar.addStretch()

        # Add file info toggle button on the right side
        if self.file_path:
            self.info_toggle_button = QPushButton("Show File Info")
            self.info_toggle_button.setFixedHeight(30)
            self.info_toggle_button.setToolTip("Show/hide EXIF and file information")
            self.info_toggle_button.setCheckable(True)
            self.info_toggle_button.setChecked(False)
            self.info_toggle_button.clicked.connect(self._toggle_file_info)
            toolbar.addWidget(self.info_toggle_button)

        # Initialize annotation system
        self.annotation_renderer = None
        self._annotation_status_text = None  # For status bar display
        # Load annotation enabled state from settings
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.annotations_enabled = settings.value("annotation_enabled", False, type=bool)
        self.plate_solve_result = None
        self.plate_solve_worker = None

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
        self.file_info_panel.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 12pt;
                color: #e0e0e0;
                background-color: {COLORS['background']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: {COLORS['text']};
                background-color: {COLORS['background']};
            }}
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
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                background-color: {COLORS['background']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['border']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['border_light']};
            }}
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

            # Auto-load annotations if enabled and cached WCS exists
            if self.annotations_enabled and self.file_path:
                self._auto_load_annotations()

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

        # Draw annotations if enabled
        if self.annotations_enabled and self.annotation_renderer:
            logger.debug(f"Drawing annotations: zoom={self.zoom_factor}, pos=({x}, {y})")
            self.annotation_renderer.render(painter, self.zoom_factor, x, y)

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

    def _update_status(self, annotation_status=None):
        """Update the status bar with current zoom level and image size"""
        zoom_percent = int(self.zoom_factor * 100)
        image_size = f"{self.original_pixmap.width()}x{self.original_pixmap.height()}"
        base_status = f"Zoom: {zoom_percent}% | Image Size: {image_size} pixels"

        if annotation_status:
            self.status_bar.setText(f"{base_status} | {annotation_status}")
        elif hasattr(self, '_annotation_status_text') and self._annotation_status_text:
            self.status_bar.setText(f"{base_status} | {self._annotation_status_text}")
        else:
            self.status_bar.setText(base_status)

    def _open_file_location(self):
        """Open the file location in the system's file explorer"""
        if self.file_path:
            success = ResourceManager.open_file_manager(self.file_path)
            if not success:
                QMessageBox.critical(self, "Error", "Failed to open file location")

    def _set_as_background(self):
        """Set the current image as the desktop background"""
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "No file path available")
            return

        try:
            # Check if file is a FITS file
            file_ext = Path(self.file_path).suffix.lower()
            if file_ext in ['.fits', '.fit', '.fts']:
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
                    QMessageBox.information(self, "Success", "Desktop background updated successfully!")
                else:
                    QMessageBox.critical(self, "Error", "Failed to set desktop background")

            elif platform.system() == "Darwin":
                # macOS implementation
                script = f'''
                tell application "Finder"
                    set desktop picture to POSIX file "{abs_path}"
                end tell
                '''
                subprocess.run(["osascript", "-e", script], check=True)
                QMessageBox.information(self, "Success", "Desktop background updated successfully!")

            elif platform.system() == "Linux":
                # Linux implementation (works with GNOME)
                try:
                    # Try GNOME
                    subprocess.run([
                        "gsettings", "set", "org.gnome.desktop.background",
                        "picture-uri", f"file://{abs_path}"
                    ], check=True)
                    QMessageBox.information(self, "Success", "Desktop background updated successfully!")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Try other desktop environments if needed
                    QMessageBox.warning(self, "Warning",
                                      "Could not set background. This feature may not be supported on your desktop environment.")
            else:
                QMessageBox.warning(self, "Warning",
                                  f"Setting desktop background is not supported on {platform.system()}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set desktop background: {str(e)}")

    def _save_with_annotations(self):
        """Save the current image with annotations overlaid"""
        if not self.file_path or not self.original_pixmap:
            QMessageBox.warning(self, "Warning", "No image available to save")
            return

        if not self.annotation_renderer:
            QMessageBox.warning(self, "No Annotations",
                              "No annotations available. Please plate solve the image first\n"
                              "and enable annotations using 'Show Annotations'.")
            return

        try:
            # Create a copy of the original pixmap to draw on
            annotated_pixmap = self.original_pixmap.copy()

            # Create a painter to draw annotations
            painter = QPainter(annotated_pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)

            # Render annotations at 100% zoom (1.0) with no offset
            self.annotation_renderer.render(painter, 1.0, 0, 0)
            painter.end()

            # Generate default filename
            original_path = Path(self.file_path)
            default_name = f"{original_path.stem}_annotated{original_path.suffix}"
            default_path = str(original_path.parent / default_name)

            # Open save dialog
            save_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save Annotated Image",
                default_path,
                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*.*)"
            )

            if save_path:
                # Determine format from extension
                save_ext = Path(save_path).suffix.lower()
                if save_ext in ['.jpg', '.jpeg']:
                    quality = 95
                    annotated_pixmap.save(save_path, "JPEG", quality)
                else:
                    annotated_pixmap.save(save_path, "PNG")

                QMessageBox.information(self, "Success",
                                      f"Annotated image saved successfully!\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotated image: {str(e)}")

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
                dimensions = f"{width} x {height} pixels"

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
                    'APTAREA': 'Aperture Area (mm^2)',
                    'FWHM': 'FWHM (arcsec)',
                    'EQUINOX': 'Equinox',
                    'RA': 'Right Ascension',
                    'DEC': 'Declination',
                    'OBJCTRA': 'Object RA',
                    'OBJCTDEC': 'Object Dec',
                    'AIRMASS': 'Airmass',
                    'GAIN': 'Gain',
                    'OFFSET': 'Offset',
                    'TEMP': 'Temperature (C)',
                    'CCD-TEMP': 'CCD Temperature (C)',
                    'SET-TEMP': 'Set Temperature (C)',
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
                            value = f"{value} deg ({hours:02d}h {minutes:02d}m {seconds:05.2f}s)"
                        elif keyword in ['DEC', 'OBJCTDEC'] and isinstance(value, (int, float)):
                            # Format declination as degrees:arcminutes:arcseconds
                            dec_deg = abs(value)
                            sign = '+' if value >= 0 else '-'
                            degrees = int(dec_deg)
                            arcmin = int((dec_deg - degrees) * 60)
                            arcsec = ((dec_deg - degrees) * 60 - arcmin) * 60
                            value = f"{value} deg ({sign}{degrees:02d} deg {arcmin:02d}' {arcsec:05.2f}\")"
                        elif keyword in ['TEMP', 'CCD-TEMP', 'SET-TEMP'] and isinstance(value, (int, float)):
                            value = f"{value} C"

                        fits_info[description] = str(value)

                # Add image dimensions from FITS if available
                if 'NAXIS1' in header and 'NAXIS2' in header:
                    width = header['NAXIS1']
                    height = header['NAXIS2']
                    if 'NAXIS3' in header:
                        depth = header['NAXIS3']
                        fits_info['Image Dimensions'] = f"{width} x {height} x {depth} pixels"
                    else:
                        fits_info['Image Dimensions'] = f"{width} x {height} pixels"

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

    def _show_annotations_dialog(self):
        """Show the annotations dialog for plate solving and annotation settings"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Image Annotations")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Check if WCS file exists for this image
        wcs_exists = False
        if self.file_path:
            wcs_file = Path(self.file_path).with_suffix('.wcs')
            wcs_exists = wcs_file.exists()

        # Auto-load cached WCS if available and not already loaded
        if wcs_exists and not (self.plate_solve_result and self.plate_solve_result.success):
            self._load_cached_wcs()

        # Status label
        self.annotation_status = QLabel("Ready to plate solve")
        self.annotation_status.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 5px;")
        layout.addWidget(self.annotation_status)

        # Progress bar
        self.annotation_progress = QProgressBar()
        self.annotation_progress.setRange(0, 0)  # Indeterminate
        self.annotation_progress.setVisible(False)
        layout.addWidget(self.annotation_progress)

        # Plate solve button - show "Re-Plate Solve" if WCS exists or already solved
        solve_layout = QHBoxLayout()
        if self.plate_solve_result and self.plate_solve_result.success:
            self.solve_button = QPushButton("Re-Plate Solve Image")
            self.annotation_status.setText(
                f"Solved: RA {self.plate_solve_result.ra_center:.4f}, "
                f"Dec {self.plate_solve_result.dec_center:.4f}, "
                f"Scale {self.plate_solve_result.pixel_scale:.2f}\"/px"
            )
        elif wcs_exists:
            self.solve_button = QPushButton("Re-Plate Solve Image")
            self.annotation_status.setText("Cached plate solve available - click to load")
        else:
            self.solve_button = QPushButton("Plate Solve Image")

        self.solve_button.clicked.connect(lambda: self._start_plate_solve(dialog))
        solve_layout.addWidget(self.solve_button)

        solve_layout.addStretch()
        layout.addLayout(solve_layout)

        # Load saved annotation settings
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Annotation toggles (only enabled after plate solve)
        toggles_group = QGroupBox("Annotation Layers")
        toggles_layout = QVBoxLayout(toggles_group)

        self.show_dsos_check = QCheckBox("Show DSO Labels")
        self.show_dsos_check.setChecked(settings.value("annotation_show_dsos", True, type=bool))
        self.show_dsos_check.stateChanged.connect(self._update_annotation_settings)
        toggles_layout.addWidget(self.show_dsos_check)

        self.show_stars_check = QCheckBox("Show Star Labels")
        self.show_stars_check.setChecked(settings.value("annotation_show_stars", True, type=bool))
        self.show_stars_check.stateChanged.connect(self._update_annotation_settings)
        toggles_layout.addWidget(self.show_stars_check)

        self.show_constellations_check = QCheckBox("Show Constellation Lines")
        self.show_constellations_check.setChecked(settings.value("annotation_show_constellations", True, type=bool))
        self.show_constellations_check.stateChanged.connect(self._update_annotation_settings)
        toggles_layout.addWidget(self.show_constellations_check)

        self.show_grid_check = QCheckBox("Show Coordinate Grid")
        self.show_grid_check.setChecked(settings.value("annotation_show_grid", True, type=bool))
        self.show_grid_check.stateChanged.connect(self._update_annotation_settings)
        toggles_layout.addWidget(self.show_grid_check)

        # Enable/disable based on solve state
        has_solution = bool(self.plate_solve_result and self.plate_solve_result.success)
        toggles_group.setEnabled(has_solution)
        logger.debug(f"Annotations dialog: has_solution={has_solution}, annotations_enabled={self.annotations_enabled}, "
                     f"annotation_renderer={self.annotation_renderer is not None}")

        layout.addWidget(toggles_group)

        # Enable/disable annotations toggle
        self.enable_annotations_check = QCheckBox("Enable Annotations Overlay")
        logger.info(f"Creating enable_annotations_check: will setChecked({self.annotations_enabled}), setEnabled({has_solution})")
        self.enable_annotations_check.setChecked(self.annotations_enabled)
        self.enable_annotations_check.setEnabled(has_solution)
        self.enable_annotations_check.stateChanged.connect(self._toggle_annotations)
        logger.info(f"enable_annotations_check created: isChecked={self.enable_annotations_check.isChecked()}, isEnabled={self.enable_annotations_check.isEnabled()}")
        layout.addWidget(self.enable_annotations_check)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _start_plate_solve(self, dialog):
        """Start the plate solving process"""
        if not self.file_path:
            QMessageBox.warning(self, "Error", "No image file path available")
            return

        try:
            from PlateSolver import PlateSolverWorker
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Plate solver not available: {e}")
            return

        # Update UI
        self.solve_button.setEnabled(False)
        self.annotation_progress.setVisible(True)
        self.annotation_status.setText("Starting plate solve...")

        # Build hints from DSO coordinates if available
        hints = {}
        if self.dso_ra is not None and self.dso_dec is not None:
            hints['ra'] = self.dso_ra
            hints['dec'] = self.dso_dec
            hints['radius'] = 15  # Search within 15 degrees of the DSO
            self.annotation_status.setText(f"Plate solving near RA={self.dso_ra:.2f}, Dec={self.dso_dec:.2f}...")

        # Create worker
        self.plate_solve_worker = PlateSolverWorker(self.file_path, hints if hints else None)
        self.plate_solve_worker.progress.connect(
            lambda msg: self.annotation_status.setText(msg)
        )
        self.plate_solve_worker.solve_finished.connect(
            lambda result: self._on_plate_solve_finished(result, dialog)
        )
        self.plate_solve_worker.start()

    def _on_plate_solve_finished(self, result, dialog):
        """Handle plate solve completion"""
        self.annotation_progress.setVisible(False)
        self.solve_button.setEnabled(True)
        self.plate_solve_result = result

        if result.success:
            self.annotation_status.setText(
                f"Solved with {result.solver_used}: RA {result.ra_center:.4f}, "
                f"Dec {result.dec_center:.4f}, Scale {result.pixel_scale:.2f}\"/px"
            )
            self.solve_button.setText("Re-Solve Image")

            # Enable annotation controls
            if hasattr(self, 'enable_annotations_check'):
                self.enable_annotations_check.setEnabled(True)
            if hasattr(self, 'show_dsos_check'):
                self.show_dsos_check.parent().setEnabled(True)

            # Initialize annotation renderer
            self._init_annotation_renderer()

            QMessageBox.information(self, "Plate Solve Complete",
                f"Image successfully plate solved!\n\n"
                f"Solver: {result.solver_used}\n"
                f"Center: RA {result.ra_center:.4f}, Dec {result.dec_center:.4f}\n"
                f"Pixel Scale: {result.pixel_scale:.2f} arcsec/pixel"
            )
        else:
            self.annotation_status.setText(f"Solve failed!")
            QMessageBox.warning(self, "Plate Solve Failed",
                f"Could not plate solve the image.\n\n{result.error_message}" or "Unknown error",
            )

    def _auto_load_annotations(self):
        """Auto-load annotations if cached WCS exists and annotations are enabled"""
        if not self.file_path:
            return

        # Check if WCS file exists
        wcs_file = Path(self.file_path).with_suffix('.wcs')
        if not wcs_file.exists():
            logger.debug(f"No cached WCS file for auto-load: {wcs_file}")
            return

        # Load cached WCS if not already loaded
        if not (self.plate_solve_result and self.plate_solve_result.success):
            logger.info("Auto-loading cached WCS for annotations")
            self._annotation_status_text = "Loading plate solve data..."
            self._update_status()
            self._load_cached_wcs()

    def _load_cached_wcs(self):
        """Load cached WCS file if available"""
        if not self.file_path:
            return

        try:
            from PlateSolver import PlateSolver, PlateSolveResult

            wcs_file = Path(self.file_path).with_suffix('.wcs')
            if not wcs_file.exists():
                return

            logger.info(f"Loading cached WCS from {wcs_file}")

            # Use PlateSolver to parse the WCS file
            solver = PlateSolver()
            wcs_header = solver._parse_wcs_file(wcs_file)

            if wcs_header and wcs_header.get('CRVAL1') is not None:
                # Create a PlateSolveResult from cached data
                result = PlateSolveResult()
                result.success = True
                result.solver_used = 'ASTAP (cached)'
                result.wcs_header = wcs_header
                result.ra_center = wcs_header.get('CRVAL1')
                result.dec_center = wcs_header.get('CRVAL2')

                # Calculate pixel scale
                if 'CD1_1' in wcs_header:
                    result.pixel_scale = abs(wcs_header['CD1_1']) * 3600
                elif 'CDELT1' in wcs_header:
                    result.pixel_scale = abs(wcs_header['CDELT1']) * 3600

                self.plate_solve_result = result
                logger.info(f"Loaded cached WCS: RA={result.ra_center}, Dec={result.dec_center}, scale={result.pixel_scale}")

                # Initialize annotation renderer
                self._init_annotation_renderer()

        except Exception as e:
            logger.warning(f"Failed to load cached WCS: {e}")

    def _init_annotation_renderer(self):
        """Initialize the annotation renderer with WCS data"""
        if not self.plate_solve_result or not self.plate_solve_result.success:
            logger.warning("Cannot init annotation renderer - no successful plate solve result")
            return

        try:
            from AnnotationOverlay import AnnotationRenderer, CatalogQueryWorker

            self.annotation_renderer = AnnotationRenderer()
            self.annotation_renderer.set_wcs(
                self.plate_solve_result.wcs_header,
                self.original_pixmap.width(),
                self.original_pixmap.height()
            )

            # Apply saved layer settings to the renderer
            settings = QSettings("CosmosCollection", "CosmosCollection")
            self.annotation_renderer.show_dsos = settings.value("annotation_show_dsos", True, type=bool)
            self.annotation_renderer.show_stars = settings.value("annotation_show_stars", True, type=bool)
            self.annotation_renderer.show_constellation_lines = settings.value("annotation_show_constellations", True, type=bool)
            self.annotation_renderer.show_grid = settings.value("annotation_show_grid", True, type=bool)

            logger.info(f"Annotation renderer initialized with WCS. Image size: {self.original_pixmap.width()}x{self.original_pixmap.height()}")

            # Update status to show catalog query is starting
            self._annotation_status_text = "Querying star/DSO catalogs..."
            self._update_status()

            # Query catalogs for objects
            self.catalog_worker = CatalogQueryWorker(
                self.annotation_renderer.wcs,
                magnitude_limit=8.0
            )
            self.catalog_worker.progress.connect(self._on_catalog_query_progress)
            self.catalog_worker.finished.connect(self._on_catalog_query_finished)
            self.catalog_worker.start()

        except Exception as e:
            logger.exception("Failed to initialize annotation renderer")
            QMessageBox.warning(self, "Error", f"Failed to initialize annotations: {e}")

    def _on_catalog_query_progress(self, message):
        """Handle catalog query progress updates"""
        self._annotation_status_text = message
        self._update_status()

    def _on_catalog_query_finished(self, stars, dsos):
        """Handle catalog query completion"""
        logger.info(f"Catalog query finished: {len(stars)} stars, {len(dsos)} DSOs")

        # Update status with results
        if self.annotations_enabled:
            self._annotation_status_text = f"Annotations: {len(stars)} stars, {len(dsos)} DSOs"
        else:
            self._annotation_status_text = f"Annotations ready: {len(stars)} stars, {len(dsos)} DSOs (disabled)"
        self._update_status()

        if self.annotation_renderer:
            self.annotation_renderer.set_objects(stars, dsos)
            if self.annotations_enabled:
                self._update_zoom()

    def _update_annotation_settings(self):
        """Update annotation visibility settings"""
        if self.annotation_renderer:
            self.annotation_renderer.show_dsos = self.show_dsos_check.isChecked()
            self.annotation_renderer.show_stars = self.show_stars_check.isChecked()
            self.annotation_renderer.show_constellation_lines = self.show_constellations_check.isChecked()
            self.annotation_renderer.show_grid = self.show_grid_check.isChecked()

            if self.annotations_enabled:
                self._update_zoom()

        # Save settings
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("annotation_show_dsos", self.show_dsos_check.isChecked())
        settings.setValue("annotation_show_stars", self.show_stars_check.isChecked())
        settings.setValue("annotation_show_constellations", self.show_constellations_check.isChecked())
        settings.setValue("annotation_show_grid", self.show_grid_check.isChecked())

    def _toggle_annotations(self, state):
        """Toggle annotations overlay on/off"""
        # state can be Qt.CheckState enum or int - just check if it's truthy/checked
        if hasattr(state, 'value'):
            # It's an enum, get its integer value
            state_int = state.value
        else:
            state_int = int(state)

        self.annotations_enabled = (state_int == 2)  # 2 = Qt.CheckState.Checked
        logger.info(f"Annotations toggled: state={state} (int={state_int}), enabled={self.annotations_enabled}, "
                    f"renderer exists={self.annotation_renderer is not None}")

        # Update status bar
        if self.annotation_renderer:
            stars = len(self.annotation_renderer.stars)
            dsos = len(self.annotation_renderer.dsos)
            if self.annotations_enabled:
                self._annotation_status_text = f"Annotations: {stars} stars, {dsos} DSOs"
            else:
                self._annotation_status_text = None
        self._update_status()

        # Save setting
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("annotation_enabled", self.annotations_enabled)

        self._update_zoom()

    def closeEvent(self, event):
        """Save window position when closing"""
        # Cancel any running plate solve
        if self.plate_solve_worker and self.plate_solve_worker.isRunning():
            self.plate_solve_worker.cancel()
            self.plate_solve_worker.wait(1000)

        WindowPositionManager.save_window_position(self, "ImageViewer")
        event.accept()
