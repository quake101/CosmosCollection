#!/usr/bin/env python3
"""
NINA Dashboard Window for Cosmos Collection
Displays real-time NINA status, current imaging, live stack images, and guiding graphs.
"""

import hashlib
import logging
import sys
from collections import deque
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use('Qt5Agg')

# Suppress matplotlib font_manager debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Set dark theme for matplotlib
plt.style.use('dark_background')

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings, QByteArray, QPointF, QRectF
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QGroupBox, QProgressBar, QComboBox, QFrame,
    QGridLayout, QSizePolicy, QDockWidget, QCheckBox, QSpinBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLineEdit,
    QTabWidget
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent

from NINAIntegration import NINAIntegration
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from TimeFormatHelper import format_time

# Set up logging
logger = logging.getLogger(__name__)


class ZoomableImageWidget(QWidget):
    """Widget that displays an image with zoom (mouse wheel) and pan (drag) support."""

    def __init__(self, parent=None, placeholder_text="No image available"):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._pan_offset = QPointF(0, 0)
        self._last_mouse_pos = None
        self._placeholder_text = placeholder_text

        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

    def setPixmap(self, pixmap):
        """Set the image to display."""
        self._pixmap = pixmap
        self._reset_view()
        self.update()

    def setPlaceholderText(self, text):
        """Set the placeholder text shown when no image is loaded."""
        self._placeholder_text = text
        self.update()

    def _reset_view(self):
        """Reset zoom and pan to fit the image in the widget."""
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)

    def _get_fit_zoom(self):
        """Calculate the zoom level that fits the image in the widget."""
        if not self._pixmap or self._pixmap.isNull():
            return 1.0
        widget_size = self.size()
        pixmap_size = self._pixmap.size()
        scale_x = (widget_size.width() - 10) / pixmap_size.width()
        scale_y = (widget_size.height() - 10) / pixmap_size.height()
        return min(scale_x, scale_y, 1.0)  # Don't upscale beyond 100%

    def paintEvent(self, event):
        """Paint the image with current zoom and pan."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fill background
        painter.fillRect(self.rect(), Qt.black)

        if not self._pixmap or self._pixmap.isNull():
            # Draw placeholder text
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder_text)
            return

        # Calculate the effective zoom (fit zoom * user zoom)
        fit_zoom = self._get_fit_zoom()
        effective_zoom = fit_zoom * self._zoom

        # Calculate scaled image size
        scaled_width = self._pixmap.width() * effective_zoom
        scaled_height = self._pixmap.height() * effective_zoom

        # Center the image, then apply pan offset
        x = (self.width() - scaled_width) / 2 + self._pan_offset.x()
        y = (self.height() - scaled_height) / 2 + self._pan_offset.y()

        # Draw the image
        target_rect = QRectF(x, y, scaled_width, scaled_height)
        source_rect = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(target_rect, self._pixmap, source_rect)

        # Draw zoom indicator if zoomed
        if self._zoom != 1.0:
            zoom_text = f"{self._zoom * 100:.0f}%"
            painter.setPen(Qt.white)
            painter.drawText(10, 20, zoom_text)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if not self._pixmap or self._pixmap.isNull():
            return

        # Get zoom delta from wheel
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1 / 1.15

        # Calculate new zoom, clamped to limits
        new_zoom = self._zoom * zoom_factor
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))

        if new_zoom != self._zoom:
            # Zoom toward mouse position
            mouse_pos = event.position()
            old_zoom = self._zoom
            self._zoom = new_zoom

            # Adjust pan to zoom toward mouse position
            zoom_change = new_zoom / old_zoom
            center = QPointF(self.width() / 2, self.height() / 2)
            mouse_offset = mouse_pos - center - self._pan_offset
            self._pan_offset = self._pan_offset - mouse_offset * (zoom_change - 1)

            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on mouse press."""
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan the image on mouse drag."""
        if self._last_mouse_pos is not None and self._pixmap and not self._pixmap.isNull():
            delta = event.position() - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End panning on mouse release."""
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Reset view on double-click."""
        self._reset_view()
        self.update()

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        self.update()


class NINAStatusWorker(QThread):
    """Background thread for polling NINA API endpoints."""

    status_updated = Signal(dict)  # Emits combined status data
    image_updated = Signal(bytes, dict)  # Emits image data and metadata
    livestack_updated = Signal(bytes, dict)  # Emits livestack image and status
    guiding_updated = Signal(list)  # Emits guiding graph data points
    error_occurred = Signal(str)  # Emits error message
    connection_changed = Signal(bool, str)  # Emits connected state and version

    # Adaptive polling rates
    POLL_RATE_ACTIVE = 0.5  # when exposing/guiding
    POLL_RATE_IDLE = 2    # when idle

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self._running = False
        self._poll_interval = self.POLL_RATE_IDLE  # Start with idle rate
        self._fetch_images = True
        self._was_exposing = False  # Track exposure state to detect when exposure completes
        self._exposure_end_time = None  # Expected end time of current exposure
        self._waiting_for_new_image = False  # Keep checking until new image is saved
        self._initial_image_check_done = False  # Have we done the initial image check?
        self._last_image_index = -1  # Track the last known image index (-1 = no images yet)
        self._last_livestack_hash = None  # Track livestack image hash
        self._last_livestack_running = False  # Track if livestack was running
        self._consecutive_failures = 0  # Track consecutive API failures to detect disconnect
        self._version = ""  # Store NINA version for reconnection

    def run(self):
        """Main polling loop."""
        self._running = True

        # Test connection first
        success, message, version = NINAIntegration.test_connection(self.host, self.port)
        if success:
            self._version = version or "Unknown"
            self.connection_changed.emit(True, self._version)
        else:
            self.connection_changed.emit(False, "")
            self.error_occurred.emit(message)
            return

        while self._running:
            try:
                # Fetch equipment status
                status_data = {}

                camera_info = NINAIntegration.get_camera_info(self.host, self.port)
                if camera_info and isinstance(camera_info, dict):
                    status_data['camera'] = camera_info

                mount_info = NINAIntegration.get_mount_info(self.host, self.port)
                if mount_info and isinstance(mount_info, dict):
                    status_data['mount'] = mount_info

                guider_info = NINAIntegration.get_guider_info(self.host, self.port)
                if guider_info and isinstance(guider_info, dict):
                    status_data['guider'] = guider_info

                filterwheel_info = NINAIntegration.get_filterwheel_info(self.host, self.port)
                if filterwheel_info and isinstance(filterwheel_info, dict):
                    status_data['filterwheel'] = filterwheel_info

                focuser_info = NINAIntegration.get_focuser_info(self.host, self.port)
                if focuser_info and isinstance(focuser_info, dict):
                    status_data['focuser'] = focuser_info

                # Always emit status update so UI can show current state
                self.status_updated.emit(status_data)

                # Adaptive polling: faster when active, slower when idle
                camera = status_data.get('camera', {})
                guider = status_data.get('guider', {})
                is_exposing = camera.get('IsExposing', False) if isinstance(camera, dict) else False
                guider_state = guider.get('State', '') if isinstance(guider, dict) else ''
                is_guiding = guider.get('Connected', False) and guider_state == 'Guiding' if isinstance(guider, dict) else False

                if is_exposing or is_guiding:
                    self._poll_interval = self.POLL_RATE_ACTIVE
                else:
                    self._poll_interval = self.POLL_RATE_IDLE

                # Check if we got any data - if not, NINA might be disconnected
                if status_data:
                    if self._consecutive_failures >= 2:
                        # Reconnected after being disconnected
                        logger.debug("NINA connection restored")
                        self.connection_changed.emit(True, self._version)
                    self._consecutive_failures = 0  # Reset on success
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        logger.debug(f"NINA connection lost ({self._consecutive_failures} consecutive failures)")
                        self.connection_changed.emit(False, "")
                        # Keep counting but don't reset - we want to stay disconnected

                # Fetch image thumbnail based on exposure state
                if self._fetch_images:
                    camera = status_data.get('camera', {})
                    is_exposing = camera.get('IsExposing', False) if isinstance(camera, dict) else False

                    # Track exposure end time while exposing
                    if is_exposing and isinstance(camera, dict):
                        exposure_end_str = camera.get('ExposureEndTime')
                        if exposure_end_str:
                            try:
                                from datetime import timezone
                                self._exposure_end_time = datetime.fromisoformat(
                                    exposure_end_str.replace('Z', '+00:00')
                                )
                            except (ValueError, TypeError):
                                pass

                    # Detect when exposure ends
                    if self._was_exposing and not is_exposing:
                        # Check if exposure was cancelled (ended before expected time)
                        was_cancelled = False
                        if self._exposure_end_time:
                            try:
                                from datetime import timezone
                                now = datetime.now(timezone.utc) if self._exposure_end_time.tzinfo else datetime.now()
                                # If we're more than 2 seconds before expected end, it was cancelled
                                time_remaining = (self._exposure_end_time - now).total_seconds()
                                if time_remaining > 2:
                                    was_cancelled = True
                                    logger.debug(f"Exposure cancelled ({time_remaining:.1f}s remaining)")
                            except Exception:
                                pass

                        if was_cancelled:
                            logger.debug("Exposure was cancelled, not waiting for new image")
                        else:
                            logger.debug("Exposure completed, waiting for new image to be saved...")
                            self._waiting_for_new_image = True

                        self._exposure_end_time = None  # Reset for next exposure

                    self._was_exposing = is_exposing

                    # Initial fetch on startup - find and display the latest image (only once)
                    if not self._initial_image_check_done:
                        self._initial_image_check_done = True
                        image_count = NINAIntegration.get_image_count(self.host, self.port)
                        if image_count > 0:
                            self._last_image_index = image_count - 1
                            logger.debug(f"Initial image fetch (index {self._last_image_index})")
                            image_data, image_meta = NINAIntegration.get_image_thumbnail(
                                self.host, self.port, self._last_image_index, 400
                            )
                            if image_data:
                                self.image_updated.emit(image_data, image_meta or {})
                        else:
                            logger.debug("No images available yet, waiting for first exposure")

                    # Keep checking for new image after exposure completes
                    elif self._waiting_for_new_image:
                        # If no images exist yet, check for index 0, otherwise check next index
                        next_index = 0 if self._last_image_index == -1 else self._last_image_index + 1
                        if NINAIntegration._image_exists(self.host, self.port, next_index):
                            logger.debug(f"New image available at index {next_index}")
                            self._last_image_index = next_index
                            self._waiting_for_new_image = False
                            image_data, image_meta = NINAIntegration.get_image_thumbnail(
                                self.host, self.port, next_index, 400
                            )
                            if image_data:
                                self.image_updated.emit(image_data, image_meta or {})

                # Fetch livestack status and image
                livestack_status = NINAIntegration.get_livestack_status(self.host, self.port)
                is_livestacking = (livestack_status and
                                   isinstance(livestack_status, dict) and
                                   livestack_status.get('running', False))
                if is_livestacking:
                    livestack_image = NINAIntegration.get_livestack_image(self.host, self.port)
                    if livestack_image:
                        # Emit if livestack image changed
                        current_hash = hashlib.md5(livestack_image).hexdigest()
                        if current_hash != self._last_livestack_hash:
                            self._last_livestack_hash = current_hash
                            self._last_livestack_running = True
                            self.livestack_updated.emit(livestack_image, livestack_status)
                    elif not self._last_livestack_running:
                        # Livestacking is running but no image yet - emit status to update tab
                        self._last_livestack_running = True
                        self.livestack_updated.emit(b'', livestack_status)
                else:
                    # Emit empty to reset tab (only if state changed)
                    if self._last_livestack_running or self._last_livestack_hash is not None:
                        self._last_livestack_hash = None
                        self._last_livestack_running = False
                        self.livestack_updated.emit(b'', {'running': False})

                # Fetch guiding graph data only if guider is connected and guiding
                guider = status_data.get('guider', {})
                guider_state = guider.get('State', '') if isinstance(guider, dict) else ''
                is_guiding = guider.get('Connected', False) and guider_state == 'Guiding'
                if is_guiding:
                    guiding_data = NINAIntegration.get_guiding_graph_data(self.host, self.port)
                    if guiding_data and isinstance(guiding_data, list):
                        self.guiding_updated.emit(guiding_data)

            except Exception as e:
                logger.error(f"Error in NINA status worker: {e}")
                self.error_occurred.emit(str(e))

            # Sleep for polling interval
            if self._running:
                # Sleep in small increments (50ms) so we can stop quickly
                sleep_iterations = int(self._poll_interval * 20)  # 50ms per iteration
                for _ in range(max(1, sleep_iterations)):
                    if not self._running:
                        break
                    self.msleep(50)

    def stop(self):
        """Stop the polling loop."""
        self._running = False

    def set_fetch_images(self, enabled):
        """Enable or disable image fetching."""
        self._fetch_images = enabled


class GuidingGraph(FigureCanvas):
    """Matplotlib canvas for RA/Dec guiding deviation plot."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 2.5), facecolor='#2b2b2b')
        super().__init__(self.figure)
        self.setParent(parent)

        # Circular buffer for last 5 minutes of data (at ~1 point/sec = 300 points)
        self.max_points = 300
        self.ra_data = deque(maxlen=self.max_points)
        self.dec_data = deque(maxlen=self.max_points)
        self.time_data = deque(maxlen=self.max_points)

        self.ax = None
        self._create_empty_chart()

    def _create_empty_chart(self):
        """Create an empty chart placeholder."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(-3, 3)
        self.ax.set_ylabel('Deviation (arcsec)', color=COLORS['text'], fontsize=9)
        self.ax.set_xlabel('Time', color=COLORS['text'], fontsize=9)
        self.ax.set_title('Guiding Performance', color=COLORS['text'], fontsize=10, fontweight='bold')

        # Add threshold lines
        self.ax.axhline(y=1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1, label='+1"')
        self.ax.axhline(y=-1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1, label='-1"')
        self.ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.3, linewidth=1)

        # Style
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.spines['top'].set_color(COLORS['border'])
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['right'].set_color(COLORS['border'])
        self.ax.yaxis.grid(True, linestyle=':', alpha=0.3, color=COLORS['border'])

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#4488ff', linewidth=2, label='RA'),
            Line2D([0], [0], color='#ff8844', linewidth=2, label='Dec'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                       facecolor='#353535', edgecolor=COLORS['border'], labelcolor=COLORS['text'])

        self.figure.tight_layout()
        self.draw()

    def update_data(self, guiding_data):
        """Update the graph with new guiding data points."""
        if not guiding_data:
            return

        # Process incoming data - NINA API returns list of guide points
        for point in guiding_data:
            ra_error = point.get('RADistanceRawDisplay', 0) or point.get('RADistanceDisplay', 0) or 0
            dec_error = point.get('DECDistanceRawDisplay', 0) or point.get('DECDistanceDisplay', 0) or 0

            self.ra_data.append(ra_error)
            self.dec_data.append(dec_error)
            self.time_data.append(len(self.time_data))

        self._redraw_chart()

    def _redraw_chart(self):
        """Redraw the chart with current data."""
        if not self.ra_data:
            return

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        x_data = list(range(len(self.ra_data)))

        # Plot RA and Dec
        self.ax.plot(x_data, list(self.ra_data), color='#4488ff', linewidth=1.5, label='RA')
        self.ax.plot(x_data, list(self.dec_data), color='#ff8844', linewidth=1.5, label='Dec')

        # Add threshold lines
        self.ax.axhline(y=1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
        self.ax.axhline(y=-1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
        self.ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.3, linewidth=1)

        # Calculate RMS for display
        if self.ra_data:
            import math
            ra_rms = math.sqrt(sum(x**2 for x in self.ra_data) / len(self.ra_data))
            dec_rms = math.sqrt(sum(x**2 for x in self.dec_data) / len(self.dec_data))
            total_rms = math.sqrt(ra_rms**2 + dec_rms**2)
            self.ax.set_title(f'Guiding Performance  |  RMS: {total_rms:.2f}" (RA: {ra_rms:.2f}", Dec: {dec_rms:.2f}")',
                              color=COLORS['text'], fontsize=10, fontweight='bold')
        else:
            self.ax.set_title('Guiding Performance', color=COLORS['text'], fontsize=10, fontweight='bold')

        # Axis limits
        self.ax.set_xlim(0, max(len(self.ra_data), 60))
        y_max = max(3, max(abs(min(self.ra_data)), abs(max(self.ra_data)),
                          abs(min(self.dec_data)), abs(max(self.dec_data))) * 1.2)
        self.ax.set_ylim(-y_max, y_max)

        self.ax.set_ylabel('Deviation (arcsec)', color=COLORS['text'], fontsize=9)
        self.ax.set_xlabel('Samples', color=COLORS['text'], fontsize=9)

        # Style
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.spines['top'].set_color(COLORS['border'])
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['right'].set_color(COLORS['border'])
        self.ax.yaxis.grid(True, linestyle=':', alpha=0.3, color=COLORS['border'])

        # Legend
        self.ax.legend(loc='upper right', fontsize=8,
                       facecolor='#353535', edgecolor=COLORS['border'], labelcolor=COLORS['text'])

        self.figure.tight_layout()
        self.draw()

    def clear_data(self):
        """Clear all guiding data."""
        self.ra_data.clear()
        self.dec_data.clear()
        self.time_data.clear()
        self._create_empty_chart()


class CaptureSettingsDialog(QDialog):
    """Dialog for configuring capture settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Form layout for settings
        form_layout = QFormLayout()

        # Duration
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(0.001, 3600)  # 1ms to 1 hour
        self.duration_spinbox.setDecimals(3)
        self.duration_spinbox.setSuffix(" s")
        form_layout.addRow("Duration:", self.duration_spinbox)

        # Gain
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setRange(-1, 1000)  # -1 means use camera default
        self.gain_spinbox.setSpecialValueText("Default")
        form_layout.addRow("Gain:", self.gain_spinbox)

        # Image type
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["SNAPSHOT", "LIGHT", "DARK", "BIAS", "FLAT"])
        form_layout.addRow("Image Type:", self.image_type_combo)

        # Save to disk
        self.save_checkbox = QCheckBox()
        form_layout.addRow("Save to Disk:", self.save_checkbox)

        layout.addLayout(form_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load saved settings
        self._load_settings()

    def _load_settings(self):
        """Load saved capture settings."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.duration_spinbox.setValue(settings.value("capture_duration", 1.0, type=float))
        self.gain_spinbox.setValue(settings.value("capture_gain", -1, type=int))
        self.image_type_combo.setCurrentText(settings.value("capture_image_type", "SNAPSHOT", type=str))
        self.save_checkbox.setChecked(settings.value("capture_save", True, type=bool))

    def _save_settings(self):
        """Save capture settings."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("capture_duration", self.duration_spinbox.value())
        settings.setValue("capture_gain", self.gain_spinbox.value())
        settings.setValue("capture_image_type", self.image_type_combo.currentText())
        settings.setValue("capture_save", self.save_checkbox.isChecked())

    def _on_accepted(self):
        """Handle dialog accepted - save settings and close."""
        self._save_settings()
        self.accept()

    def get_settings(self):
        """Return the capture settings as a dict."""
        gain = self.gain_spinbox.value()
        return {
            'duration': self.duration_spinbox.value(),
            'gain': gain if gain >= 0 else None,  # None means use camera default
            'image_type': self.image_type_combo.currentText(),
            'save': self.save_checkbox.isChecked()
        }


class SlewDialog(QDialog):
    """Dialog for entering slew coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slew to Coordinates")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Search group
        search_group = QGroupBox("Search Object")
        search_layout = QHBoxLayout(search_group)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("e.g., M31, NGC 7000, Vega...")
        self.search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._on_search)
        search_layout.addWidget(self.search_btn)

        layout.addWidget(search_group)

        self.search_status_label = QLabel("")
        self.search_status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.search_status_label)

        # Form layout for coordinates
        form_layout = QFormLayout()

        # RA input (hours)
        ra_widget = QWidget()
        ra_layout = QHBoxLayout(ra_widget)
        ra_layout.setContentsMargins(0, 0, 0, 0)
        ra_layout.setSpacing(2)

        self.ra_h_spinbox = QSpinBox()
        self.ra_h_spinbox.setRange(0, 23)
        self.ra_h_spinbox.setSuffix("h")
        ra_layout.addWidget(self.ra_h_spinbox)

        self.ra_m_spinbox = QSpinBox()
        self.ra_m_spinbox.setRange(0, 59)
        self.ra_m_spinbox.setSuffix("m")
        ra_layout.addWidget(self.ra_m_spinbox)

        self.ra_s_spinbox = QDoubleSpinBox()
        self.ra_s_spinbox.setRange(0, 59.99)
        self.ra_s_spinbox.setDecimals(2)
        self.ra_s_spinbox.setSuffix("s")
        ra_layout.addWidget(self.ra_s_spinbox)

        form_layout.addRow("RA:", ra_widget)

        # Dec input (degrees)
        dec_widget = QWidget()
        dec_layout = QHBoxLayout(dec_widget)
        dec_layout.setContentsMargins(0, 0, 0, 0)
        dec_layout.setSpacing(2)

        self.dec_sign_combo = QComboBox()
        self.dec_sign_combo.addItems(["+", "-"])
        dec_layout.addWidget(self.dec_sign_combo)

        self.dec_d_spinbox = QSpinBox()
        self.dec_d_spinbox.setRange(0, 90)
        self.dec_d_spinbox.setSuffix("°")
        dec_layout.addWidget(self.dec_d_spinbox)

        self.dec_m_spinbox = QSpinBox()
        self.dec_m_spinbox.setRange(0, 59)
        self.dec_m_spinbox.setSuffix("'")
        dec_layout.addWidget(self.dec_m_spinbox)

        self.dec_s_spinbox = QDoubleSpinBox()
        self.dec_s_spinbox.setRange(0, 59.99)
        self.dec_s_spinbox.setDecimals(2)
        self.dec_s_spinbox.setSuffix('"')
        dec_layout.addWidget(self.dec_s_spinbox)

        form_layout.addRow("Dec:", dec_widget)

        layout.addLayout(form_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load saved coordinates
        self._load_settings()

    def _load_settings(self):
        """Load saved slew coordinates."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.ra_h_spinbox.setValue(settings.value("slew_ra_h", 0, type=int))
        self.ra_m_spinbox.setValue(settings.value("slew_ra_m", 0, type=int))
        self.ra_s_spinbox.setValue(settings.value("slew_ra_s", 0.0, type=float))
        self.dec_sign_combo.setCurrentText(settings.value("slew_dec_sign", "+", type=str))
        self.dec_d_spinbox.setValue(settings.value("slew_dec_d", 0, type=int))
        self.dec_m_spinbox.setValue(settings.value("slew_dec_m", 0, type=int))
        self.dec_s_spinbox.setValue(settings.value("slew_dec_s", 0.0, type=float))

    def _save_settings(self):
        """Save slew coordinates."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("slew_ra_h", self.ra_h_spinbox.value())
        settings.setValue("slew_ra_m", self.ra_m_spinbox.value())
        settings.setValue("slew_ra_s", self.ra_s_spinbox.value())
        settings.setValue("slew_dec_sign", self.dec_sign_combo.currentText())
        settings.setValue("slew_dec_d", self.dec_d_spinbox.value())
        settings.setValue("slew_dec_m", self.dec_m_spinbox.value())
        settings.setValue("slew_dec_s", self.dec_s_spinbox.value())

    def _on_accepted(self):
        """Handle dialog accepted - save settings and close."""
        self._save_settings()
        self.accept()

    def get_coordinates_degrees(self):
        """Return RA and Dec in degrees."""
        # Convert RA from hours to degrees (1h = 15°)
        ra_hours = self.ra_h_spinbox.value() + self.ra_m_spinbox.value() / 60 + self.ra_s_spinbox.value() / 3600
        ra_deg = ra_hours * 15

        # Convert Dec from DMS to degrees
        dec_deg = self.dec_d_spinbox.value() + self.dec_m_spinbox.value() / 60 + self.dec_s_spinbox.value() / 3600
        if self.dec_sign_combo.currentText() == "-":
            dec_deg = -dec_deg

        return ra_deg, dec_deg

    def _on_search(self):
        """Search for an object by name in the local database."""
        import re
        from DatabaseManager import DatabaseManager

        object_name = self.search_input.text().strip()
        if not object_name:
            self.search_status_label.setText("Enter an object name to search")
            self.search_status_label.setStyleSheet(f"color: {COLORS['warning']};")
            return

        self.search_status_label.setText(f"Searching for '{object_name}'...")
        self.search_status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.search_btn.setEnabled(False)

        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            # Normalize search term (e.g., "M31" -> "M 31", "NGC7000" -> "NGC 7000")
            search_upper = object_name.upper().strip()
            match = re.match(r'^([A-Z]+)\s*(\d+)([A-Z]?)$', search_upper)
            if match:
                catalog = match.group(1)
                number = match.group(2)
                suffix = match.group(3)
                search_catalog = catalog
                search_designation = f"{number}{suffix}"
            else:
                search_catalog = None
                search_designation = search_upper

            db = DatabaseManager()

            # Search by catalogue and designation
            if search_catalog:
                # Try exact match first
                rows = db.execute_query("""
                    SELECT d.ra, d.dec, c.catalogue, c.designation
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE UPPER(c.catalogue) = ? AND UPPER(c.designation) = ?
                    LIMIT 1
                """, (search_catalog, search_designation))
            else:
                rows = []

            if not rows:
                # Try partial match on designation
                rows = db.execute_query("""
                    SELECT d.ra, d.dec, c.catalogue, c.designation
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE UPPER(c.catalogue || ' ' || c.designation) LIKE ?
                       OR UPPER(c.catalogue || c.designation) LIKE ?
                    LIMIT 1
                """, (f"%{search_upper}%", f"%{search_upper.replace(' ', '')}%"))

            if not rows:
                self.search_status_label.setText(f"'{object_name}' not found in database")
                self.search_status_label.setStyleSheet(f"color: {COLORS['error']};")
                return

            row = rows[0]
            ra_deg = float(row[0])
            dec_deg = float(row[1])
            found_name = f"{row[2]} {row[3]}"

            # Convert RA from degrees to hours
            ra_hours = ra_deg / 15.0
            ra_h = int(ra_hours)
            ra_m = int((ra_hours - ra_h) * 60)
            ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60

            # Convert Dec to DMS
            dec_sign = "+" if dec_deg >= 0 else "-"
            dec_abs = abs(dec_deg)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60

            # Update the coordinate fields
            self.ra_h_spinbox.setValue(ra_h)
            self.ra_m_spinbox.setValue(ra_m)
            self.ra_s_spinbox.setValue(round(ra_s, 2))
            self.dec_sign_combo.setCurrentText(dec_sign)
            self.dec_d_spinbox.setValue(dec_d)
            self.dec_m_spinbox.setValue(dec_m)
            self.dec_s_spinbox.setValue(round(dec_s, 2))

            self.search_status_label.setText(f"Found: {found_name}")
            self.search_status_label.setStyleSheet(f"color: {COLORS['success']};")

        except Exception as e:
            self.search_status_label.setText(f"Search failed: {str(e)}")
            self.search_status_label.setStyleSheet(f"color: {COLORS['error']};")
        finally:
            self.search_btn.setEnabled(True)


class NINADashboardWindow(WindowPositionMixin, QMainWindow):
    """Main NINA Dashboard window."""
    WINDOW_POSITION_KEY = "NINADashboard"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NINA Dashboard - Cosmos Collection")
        self.resize(900, 700)
        self.setup_window_position()

        self.worker = None

        self._connected = False
        self._version = ""
        self._last_update = None
        self._exposure_start_time = None
        self._exposure_end_time = None
        self._current_image_pixmap = None  # Store original pixmap for rescaling
        self._current_livestack_pixmap = None  # Store original livestack pixmap

        self._setup_ui()
        self._auto_connect()
        # Defer settings restore until after window is shown (dock widgets need visible geometry)
        QTimer.singleShot(100, self._restore_settings)

    def _setup_ui(self):
        """Set up the main window UI with dockable panels."""
        # Central widget - contains header, image panel, and status bar
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Header bar
        header_layout = QHBoxLayout()

        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.setToolTip("Reconnect to NINA")
        self.reconnect_btn.clicked.connect(self._reconnect)
        header_layout.addWidget(self.reconnect_btn)

        self.connection_label = QLabel("Connection: Disconnected")
        self.connection_label.setStyleSheet(f"color: {COLORS['warning']};")
        header_layout.addWidget(self.connection_label)

        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Image panel in central widget (main content area)
        self._create_image_panel(main_layout)

        # Status bar at bottom of central widget
        status_layout_h = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        status_layout_h.addWidget(self.status_label)
        status_layout_h.addStretch()
        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        status_layout_h.addWidget(self.countdown_label)
        main_layout.addLayout(status_layout_h)

        # Create dock widgets (added to window in _restore_settings)
        self._create_equipment_docks()
        self._create_actions_docks()
        self._create_guiding_dock()

        # Set up View menu
        self._setup_view_menu()

    def _create_equipment_docks(self):
        """Create individual dock widgets for each equipment type."""
        dock_features = (
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock_areas = (
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )

        # Camera dock
        self.camera_dock = QDockWidget("Camera", self)
        self.camera_dock.setObjectName("CameraDock")
        self.camera_dock.setAllowedAreas(dock_areas)
        self.camera_dock.setFeatures(dock_features)

        camera_widget = QWidget()
        camera_layout = QGridLayout(camera_widget)
        camera_layout.setContentsMargins(8, 8, 8, 8)

        camera_layout.addWidget(QLabel("Name:"), 0, 0)
        self.camera_name_label = QLabel("--")
        camera_layout.addWidget(self.camera_name_label, 0, 1)
        camera_layout.addWidget(QLabel("Status:"), 1, 0)
        self.camera_status_label = QLabel("--")
        camera_layout.addWidget(self.camera_status_label, 1, 1)
        camera_layout.addWidget(QLabel("Exposure:"), 2, 0)
        self.camera_exposure_label = QLabel("--")
        camera_layout.addWidget(self.camera_exposure_label, 2, 1)
        camera_layout.addWidget(QLabel("Progress:"), 3, 0)
        self.camera_progress = QProgressBar()
        self.camera_progress.setMaximum(100)
        self.camera_progress.setValue(0)
        camera_layout.addWidget(self.camera_progress, 3, 1)
        camera_layout.addWidget(QLabel("Temp:"), 4, 0)
        self.camera_temp_label = QLabel("--")
        camera_layout.addWidget(self.camera_temp_label, 4, 1)

        # Cooling controls
        camera_layout.addWidget(QLabel("Cooling:"), 5, 0)
        cooling_widget = QWidget()
        cooling_layout = QHBoxLayout(cooling_widget)
        cooling_layout.setContentsMargins(0, 0, 0, 0)
        cooling_layout.setSpacing(5)
        self.camera_cooling_checkbox = QCheckBox("On")
        self.camera_cooling_checkbox.setChecked(False)  # Explicitly start unchecked
        self.camera_cooling_checkbox.setToolTip("Enable/disable camera cooling")
        self.camera_cooling_checkbox.stateChanged.connect(self._on_cooling_changed)
        logger.debug(f"[Cooling] Checkbox created, initial checked state: {self.camera_cooling_checkbox.isChecked()}")
        cooling_layout.addWidget(self.camera_cooling_checkbox)
        self.camera_target_temp_spinbox = QSpinBox()
        self.camera_target_temp_spinbox.setRange(-40, 20)
        self.camera_target_temp_spinbox.setValue(-10)
        self.camera_target_temp_spinbox.setSuffix("°C")
        self.camera_target_temp_spinbox.setToolTip("Target cooling temperature")
        self.camera_target_temp_spinbox.valueChanged.connect(self._on_target_temp_changed)
        cooling_layout.addWidget(self.camera_target_temp_spinbox)
        cooling_layout.addStretch()
        camera_layout.addWidget(cooling_widget, 5, 1)

        # Dew heater control
        camera_layout.addWidget(QLabel("Dew Heater:"), 6, 0)
        self.camera_dewheater_checkbox = QCheckBox("On")
        self.camera_dewheater_checkbox.setToolTip("Enable/disable dew heater")
        self.camera_dewheater_checkbox.stateChanged.connect(self._on_dewheater_changed)
        camera_layout.addWidget(self.camera_dewheater_checkbox, 6, 1)

        camera_layout.setRowStretch(7, 1)

        # Track if we're updating from API to avoid triggering callbacks
        self._updating_camera_controls = False
        # Track if user recently changed settings (prevents sync from overriding)
        self._user_changing_cooling = False
        self._user_changing_dewheater = False
        # Track the last cooling state we intentionally set (to avoid duplicate API calls)
        self._last_cooling_enabled = None
        self._last_cooling_temp = None
        # Timers to clear the user-changing flags
        self._cooling_change_timer = QTimer(self)
        self._cooling_change_timer.setSingleShot(True)
        self._cooling_change_timer.timeout.connect(self._clear_cooling_change_flag)
        self._dewheater_change_timer = QTimer(self)
        self._dewheater_change_timer.setSingleShot(True)
        self._dewheater_change_timer.timeout.connect(self._clear_dewheater_change_flag)

        self.camera_dock.setWidget(camera_widget)

        # Mount dock
        self.mount_dock = QDockWidget("Mount", self)
        self.mount_dock.setObjectName("MountDock")
        self.mount_dock.setAllowedAreas(dock_areas)
        self.mount_dock.setFeatures(dock_features)

        mount_widget = QWidget()
        mount_layout = QGridLayout(mount_widget)
        mount_layout.setContentsMargins(8, 8, 8, 8)

        mount_layout.addWidget(QLabel("Name:"), 0, 0)
        self.mount_name_label = QLabel("--")
        mount_layout.addWidget(self.mount_name_label, 0, 1)
        mount_layout.addWidget(QLabel("Status:"), 1, 0)
        self.mount_status_label = QLabel("--")
        mount_layout.addWidget(self.mount_status_label, 1, 1)
        mount_layout.addWidget(QLabel("RA/Dec:"), 2, 0)
        self.mount_coords_label = QLabel("--")
        mount_layout.addWidget(self.mount_coords_label, 2, 1)
        mount_layout.setRowStretch(3, 1)

        self.mount_dock.setWidget(mount_widget)

        # Guider dock
        self.guider_dock = QDockWidget("Guider", self)
        self.guider_dock.setObjectName("GuiderDock")
        self.guider_dock.setAllowedAreas(dock_areas)
        self.guider_dock.setFeatures(dock_features)

        guider_widget = QWidget()
        guider_layout = QGridLayout(guider_widget)
        guider_layout.setContentsMargins(8, 8, 8, 8)

        guider_layout.addWidget(QLabel("Name:"), 0, 0)
        self.guider_name_label = QLabel("--")
        guider_layout.addWidget(self.guider_name_label, 0, 1)
        guider_layout.addWidget(QLabel("Status:"), 1, 0)
        self.guider_status_label = QLabel("--")
        guider_layout.addWidget(self.guider_status_label, 1, 1)
        guider_layout.addWidget(QLabel("RMS:"), 2, 0)
        self.guider_rms_label = QLabel("--")
        guider_layout.addWidget(self.guider_rms_label, 2, 1)
        guider_layout.setRowStretch(3, 1)

        self.guider_dock.setWidget(guider_widget)

        # Filter Wheel dock
        self.filterwheel_dock = QDockWidget("Filter Wheel", self)
        self.filterwheel_dock.setObjectName("FilterWheelDock")
        self.filterwheel_dock.setAllowedAreas(dock_areas)
        self.filterwheel_dock.setFeatures(dock_features)

        filterwheel_widget = QWidget()
        filterwheel_layout = QGridLayout(filterwheel_widget)
        filterwheel_layout.setContentsMargins(8, 8, 8, 8)

        filterwheel_layout.addWidget(QLabel("Name:"), 0, 0)
        self.filterwheel_name_label = QLabel("--")
        filterwheel_layout.addWidget(self.filterwheel_name_label, 0, 1)
        filterwheel_layout.addWidget(QLabel("Status:"), 1, 0)
        self.filterwheel_status_label = QLabel("--")
        filterwheel_layout.addWidget(self.filterwheel_status_label, 1, 1)
        filterwheel_layout.addWidget(QLabel("Filter:"), 2, 0)
        self.filterwheel_combo = QComboBox()
        self.filterwheel_combo.setToolTip("Select filter to change to")
        self.filterwheel_combo.setEnabled(False)
        self.filterwheel_combo.currentIndexChanged.connect(self._on_filter_changed)
        filterwheel_layout.addWidget(self.filterwheel_combo, 2, 1)
        filterwheel_layout.setRowStretch(3, 1)

        # Track filter wheel state to prevent feedback loops
        self._updating_filterwheel = False
        self._user_changing_filter = False
        self._last_filter_id = None
        self._available_filters = []  # List of {'Id': int, 'Name': str}

        self.filterwheel_dock.setWidget(filterwheel_widget)

        # Focuser dock
        self.focuser_dock = QDockWidget("Focuser", self)
        self.focuser_dock.setObjectName("FocuserDock")
        self.focuser_dock.setAllowedAreas(dock_areas)
        self.focuser_dock.setFeatures(dock_features)

        focuser_widget = QWidget()
        focuser_layout = QGridLayout(focuser_widget)
        focuser_layout.setContentsMargins(8, 8, 8, 8)

        focuser_layout.addWidget(QLabel("Name:"), 0, 0)
        self.focuser_name_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_name_label, 0, 1)
        focuser_layout.addWidget(QLabel("Status:"), 1, 0)
        self.focuser_status_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_status_label, 1, 1)
        focuser_layout.addWidget(QLabel("Position:"), 2, 0)
        self.focuser_position_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_position_label, 2, 1)
        focuser_layout.addWidget(QLabel("Temp:"), 3, 0)
        self.focuser_temp_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_temp_label, 3, 1)
        focuser_layout.setRowStretch(4, 1)

        self.focuser_dock.setWidget(focuser_widget)

    def _create_actions_docks(self):
        """Create action dock widgets (Imaging, etc.)."""
        dock_features = (
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock_areas = (
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )

        # Imaging dock
        self.imaging_dock = QDockWidget("Imaging", self)
        self.imaging_dock.setObjectName("ImagingDock")
        self.imaging_dock.setAllowedAreas(dock_areas)
        self.imaging_dock.setFeatures(dock_features)

        imaging_widget = QWidget()
        imaging_layout = QHBoxLayout(imaging_widget)
        imaging_layout.setContentsMargins(8, 8, 8, 8)

        # Capture group
        capture_group = QGroupBox("Capture")
        capture_layout = QHBoxLayout(capture_group)
        capture_layout.setContentsMargins(8, 4, 8, 4)

        self.imaging_start_btn = QPushButton("Start")
        self.imaging_start_btn.clicked.connect(self._on_imaging_start)
        self.imaging_start_btn.setEnabled(False)  # Disabled until connected
        self.imaging_stop_btn = QPushButton("Stop")
        self.imaging_stop_btn.clicked.connect(self._on_imaging_stop)
        self.imaging_stop_btn.setEnabled(False)  # Disabled until exposing

        capture_layout.addWidget(self.imaging_start_btn)
        capture_layout.addWidget(self.imaging_stop_btn)

        # AutoFocus group
        autofocus_group = QGroupBox("AutoFocus")
        autofocus_layout = QHBoxLayout(autofocus_group)
        autofocus_layout.setContentsMargins(8, 4, 8, 4)

        self.autofocus_start_btn = QPushButton("Start")
        self.autofocus_start_btn.clicked.connect(self._on_autofocus_start)
        self.autofocus_start_btn.setEnabled(False)  # Disabled until connected
        self.autofocus_cancel_btn = QPushButton("Cancel")
        self.autofocus_cancel_btn.clicked.connect(self._on_autofocus_cancel)
        self.autofocus_cancel_btn.setEnabled(False)  # Disabled until autofocus running

        autofocus_layout.addWidget(self.autofocus_start_btn)
        autofocus_layout.addWidget(self.autofocus_cancel_btn)

        imaging_layout.addWidget(capture_group)
        imaging_layout.addWidget(autofocus_group)

        imaging_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.imaging_dock.setWidget(imaging_widget)

        # Guider dock
        self.guider_action_dock = QDockWidget("Guider", self)
        self.guider_action_dock.setObjectName("GuiderActionDock")
        self.guider_action_dock.setAllowedAreas(dock_areas)
        self.guider_action_dock.setFeatures(dock_features)

        guider_action_widget = QWidget()
        guider_action_layout = QHBoxLayout(guider_action_widget)
        guider_action_layout.setContentsMargins(8, 8, 8, 8)

        # Guiding group
        guiding_group = QGroupBox("Guiding")
        guiding_layout = QHBoxLayout(guiding_group)
        guiding_layout.setContentsMargins(8, 4, 8, 4)

        self.guiding_start_btn = QPushButton("Start")
        self.guiding_start_btn.clicked.connect(self._on_guiding_start)
        self.guiding_start_btn.setEnabled(False)  # Disabled until connected
        self.guiding_stop_btn = QPushButton("Stop")
        self.guiding_stop_btn.clicked.connect(self._on_guiding_stop)
        self.guiding_stop_btn.setEnabled(False)  # Disabled until guiding

        guiding_layout.addWidget(self.guiding_start_btn)
        guiding_layout.addWidget(self.guiding_stop_btn)

        guider_action_layout.addWidget(guiding_group)

        guider_action_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.guider_action_dock.setWidget(guider_action_widget)

        # Mount dock
        self.mount_action_dock = QDockWidget("Mount", self)
        self.mount_action_dock.setObjectName("MountActionDock")
        self.mount_action_dock.setAllowedAreas(dock_areas)
        self.mount_action_dock.setFeatures(dock_features)

        mount_action_widget = QWidget()
        mount_action_layout = QHBoxLayout(mount_action_widget)
        mount_action_layout.setContentsMargins(8, 8, 8, 8)

        # Parking group
        parking_group = QGroupBox("Parking")
        parking_layout = QHBoxLayout(parking_group)
        parking_layout.setContentsMargins(8, 4, 8, 4)

        self.mount_home_btn = QPushButton("Home")
        self.mount_home_btn.clicked.connect(self._on_mount_home)
        self.mount_home_btn.setEnabled(False)  # Disabled until connected
        self.mount_park_btn = QPushButton("Park")
        self.mount_park_btn.clicked.connect(self._on_mount_park)
        self.mount_park_btn.setEnabled(False)  # Disabled until connected
        self.mount_unpark_btn = QPushButton("Unpark")
        self.mount_unpark_btn.clicked.connect(self._on_mount_unpark)
        self.mount_unpark_btn.setEnabled(False)  # Disabled until connected

        parking_layout.addWidget(self.mount_home_btn)
        parking_layout.addWidget(self.mount_park_btn)
        parking_layout.addWidget(self.mount_unpark_btn)

        # Slew group
        slew_group = QGroupBox("Slew")
        slew_layout = QHBoxLayout(slew_group)
        slew_layout.setContentsMargins(8, 4, 8, 4)

        self.mount_slew_btn = QPushButton("Slew...")
        self.mount_slew_btn.clicked.connect(self._on_mount_slew)
        self.mount_slew_btn.setEnabled(False)  # Disabled until connected

        slew_layout.addWidget(self.mount_slew_btn)

        mount_action_layout.addWidget(parking_group)
        mount_action_layout.addWidget(slew_group)

        mount_action_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mount_action_dock.setWidget(mount_action_widget)

        # Spacer dock - absorbs extra space so action docks stay compact
        self.actions_spacer_dock = QDockWidget("", self)
        self.actions_spacer_dock.setObjectName("ActionsSpacerDock")
        self.actions_spacer_dock.setAllowedAreas(dock_areas)
        self.actions_spacer_dock.setFeatures(QDockWidget.DockWidgetMovable)  # No close button
        self.actions_spacer_dock.setTitleBarWidget(QWidget())  # Hide title bar

        spacer_widget = QWidget()
        spacer_widget.setMinimumWidth(0)
        spacer_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.actions_spacer_dock.setWidget(spacer_widget)

    def _create_image_panel(self, parent_layout):
        """Create the image display panel with tabs for Latest Image and Live Stack."""
        # Create tab widget instead of group box
        self.image_tabs = QTabWidget()
        self.image_tabs.setDocumentMode(True)

        # Tab 1: Latest Image
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)
        image_layout.setSpacing(2)

        self.image_label = ZoomableImageWidget(placeholder_text="No image available")
        image_layout.addWidget(self.image_label, 1)

        self.image_info_label = QLabel("Target: -- | Exp: --")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setFixedHeight(20)
        self.image_info_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        image_layout.addWidget(self.image_info_label, 0)

        self.image_tabs.addTab(image_widget, "Latest Image")

        # Tab 2: Live Stack
        livestack_widget = QWidget()
        livestack_layout = QVBoxLayout(livestack_widget)
        livestack_layout.setContentsMargins(5, 5, 5, 5)
        livestack_layout.setSpacing(2)

        self.livestack_label = ZoomableImageWidget(placeholder_text="Live stack not active")
        livestack_layout.addWidget(self.livestack_label, 1)

        self.livestack_info_label = QLabel("")
        self.livestack_info_label.setAlignment(Qt.AlignCenter)
        self.livestack_info_label.setFixedHeight(20)
        self.livestack_info_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        livestack_layout.addWidget(self.livestack_info_label, 0)

        self.image_tabs.addTab(livestack_widget, "Live Stack")

        parent_layout.addWidget(self.image_tabs, 1)  # stretch factor 1 to expand

    def _create_guiding_dock(self):
        """Create the Guiding Graph dock widget."""
        self.guiding_dock = QDockWidget("Guiding Graph", self)
        self.guiding_dock.setObjectName("GuidingGraphDock")
        self.guiding_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.guiding_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        # Guiding graph content
        guiding_widget = QWidget()
        guiding_layout = QVBoxLayout(guiding_widget)
        guiding_layout.setContentsMargins(5, 5, 5, 5)

        self.guiding_graph = GuidingGraph(self)
        self.guiding_graph.setMinimumHeight(120)
        self.guiding_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        guiding_layout.addWidget(self.guiding_graph)

        self.guiding_dock.setWidget(guiding_widget)

    def _setup_view_menu(self):
        """Set up the View menu for panel visibility and layout reset."""
        view_menu = self.menuBar().addMenu("View")

        # Actions docks
        actions_menu = view_menu.addMenu("Actions")
        actions_menu.addAction(self.imaging_dock.toggleViewAction())
        actions_menu.addAction(self.guider_action_dock.toggleViewAction())
        actions_menu.addAction(self.mount_action_dock.toggleViewAction())

        # Equipment docks
        equipment_menu = view_menu.addMenu("Equipment")
        equipment_menu.addAction(self.camera_dock.toggleViewAction())
        equipment_menu.addAction(self.mount_dock.toggleViewAction())
        equipment_menu.addAction(self.guider_dock.toggleViewAction())
        equipment_menu.addAction(self.filterwheel_dock.toggleViewAction())
        equipment_menu.addAction(self.focuser_dock.toggleViewAction())

        view_menu.addAction(self.guiding_dock.toggleViewAction())
        view_menu.addSeparator()
        reset_action = view_menu.addAction("Reset Layout")
        reset_action.triggered.connect(self._reset_layout)

    def _set_default_layout(self):
        """Set default dock positions (matches original layout)."""
        # Configure corners so top dock spans full width and bottom dock spans full width
        self.setCorner(Qt.TopLeftCorner, Qt.TopDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.TopDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)

        # Action docks at top, side by side with spacer absorbing extra space
        self.addDockWidget(Qt.TopDockWidgetArea, self.imaging_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.guider_action_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.mount_action_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.actions_spacer_dock)
        # Arrange horizontally: Imaging | Guider | Mount | Spacer
        self.splitDockWidget(self.imaging_dock, self.guider_action_dock, Qt.Horizontal)
        self.splitDockWidget(self.guider_action_dock, self.mount_action_dock, Qt.Horizontal)
        self.splitDockWidget(self.mount_action_dock, self.actions_spacer_dock, Qt.Horizontal)

        # Add equipment docks to left area, tabified together
        self.addDockWidget(Qt.LeftDockWidgetArea, self.camera_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.mount_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.guider_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filterwheel_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.focuser_dock)

        # Stack them vertically in the left area
        self.splitDockWidget(self.camera_dock, self.mount_dock, Qt.Vertical)
        self.splitDockWidget(self.mount_dock, self.guider_dock, Qt.Vertical)
        self.splitDockWidget(self.guider_dock, self.filterwheel_dock, Qt.Vertical)
        self.splitDockWidget(self.filterwheel_dock, self.focuser_dock, Qt.Vertical)

        # Add guiding graph at bottom
        self.addDockWidget(Qt.BottomDockWidgetArea, self.guiding_dock)

        # Set initial sizes
        self.resizeDocks([self.imaging_dock], [60], Qt.Vertical)  # Compact height for imaging
        self.resizeDocks([self.camera_dock], [250], Qt.Horizontal)
        self.resizeDocks([self.guiding_dock], [200], Qt.Vertical)

    def _reset_layout(self):
        """Reset dock layout to defaults."""
        # Remove all docks first
        self.removeDockWidget(self.imaging_dock)
        self.removeDockWidget(self.guider_action_dock)
        self.removeDockWidget(self.mount_action_dock)
        self.removeDockWidget(self.actions_spacer_dock)
        self.removeDockWidget(self.camera_dock)
        self.removeDockWidget(self.mount_dock)
        self.removeDockWidget(self.guider_dock)
        self.removeDockWidget(self.filterwheel_dock)
        self.removeDockWidget(self.focuser_dock)
        self.removeDockWidget(self.guiding_dock)

        # Re-add in default positions
        self._set_default_layout()

        # Show all docks
        self.imaging_dock.show()
        self.guider_action_dock.show()
        self.mount_action_dock.show()
        self.actions_spacer_dock.show()
        self.camera_dock.show()
        self.mount_dock.show()
        self.guider_dock.show()
        self.filterwheel_dock.show()
        self.focuser_dock.show()
        self.guiding_dock.show()

    def _restore_settings(self):
        """Restore saved dock layout and refresh rate."""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Check for saved dock state
        dock_state = settings.value("nina_dashboard_dock_state")

        if dock_state is not None:
            # Handle different types that QSettings might return
            if isinstance(dock_state, QByteArray):
                state_bytes = dock_state
            elif isinstance(dock_state, bytes):
                state_bytes = QByteArray(dock_state)
            else:
                logger.debug(f"Unexpected dock_state type: {type(dock_state)}")
                self._set_default_layout()
                return

            if not state_bytes.isEmpty():
                # Add docks first (required for restoreState), then restore positions
                self.addDockWidget(Qt.TopDockWidgetArea, self.imaging_dock)
                self.addDockWidget(Qt.TopDockWidgetArea, self.guider_action_dock)
                self.addDockWidget(Qt.TopDockWidgetArea, self.mount_action_dock)
                self.addDockWidget(Qt.TopDockWidgetArea, self.actions_spacer_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.camera_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.mount_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.guider_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.filterwheel_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.focuser_dock)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.guiding_dock)
                self.restoreState(state_bytes)
                logger.debug(f"Restored dock state, size={state_bytes.size()}")
            else:
                self._set_default_layout()
        else:
            self._set_default_layout()

    def _save_settings(self):
        """Save dock layout and refresh rate."""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Only save dock state if docks are properly attached to the window
        camera_area = self.dockWidgetArea(self.camera_dock)
        if camera_area == Qt.NoDockWidgetArea:
            logger.debug("Skipping dock state save - docks not attached")
            settings.sync()
            return

        # Save dock state
        dock_state = self.saveState()
        state_bytes = bytes(dock_state.data())
        settings.setValue("nina_dashboard_dock_state", state_bytes)
        logger.debug(f"Saved dock state, size={dock_state.size()}")

        settings.sync()

    def _auto_connect(self):
        """Automatically attempt to connect when the window opens."""
        if NINAIntegration.is_enabled():
            self._reconnect()
        else:
            self.connection_label.setText("Connection: NINA integration disabled")
            self.connection_label.setStyleSheet(f"color: {COLORS['warning']};")
            self.status_label.setText("Enable NINA integration in Settings to use this dashboard")

    def _reconnect(self):
        """Reconnect to NINA."""
        # Stop existing worker
        self._stop_worker()

        self.connection_label.setText("Connection: Connecting...")
        self.connection_label.setStyleSheet(f"color: {COLORS['info']};")

        # Get settings
        host, port = NINAIntegration.get_settings()

        # Create and start worker (uses adaptive polling)
        self.worker = NINAStatusWorker(host, port)
        self.worker.connection_changed.connect(self._on_connection_changed)
        self.worker.status_updated.connect(self._on_status_updated)
        self.worker.image_updated.connect(self._on_image_updated)
        self.worker.livestack_updated.connect(self._on_livestack_updated)
        self.worker.guiding_updated.connect(self._on_guiding_updated)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _stop_worker(self):
        """Stop the worker thread."""
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

    def _on_connection_changed(self, connected, version):
        """Handle connection state change."""
        self._connected = connected
        self._version = version

        if connected:
            self.connection_label.setText(f"Connection: Connected to NINA {version}")
            self.connection_label.setStyleSheet(f"color: {COLORS['success']};")
            self.status_label.setText("Connected - adaptive polling active")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            # Enable start buttons (stop/cancel remain disabled until active)
            self.imaging_start_btn.setEnabled(True)
            self.autofocus_start_btn.setEnabled(True)
            self.guiding_start_btn.setEnabled(True)
            self.mount_home_btn.setEnabled(True)
            self.mount_park_btn.setEnabled(True)
            self.mount_unpark_btn.setEnabled(True)
            self.mount_slew_btn.setEnabled(True)
        else:
            self.connection_label.setText("Connection: Disconnected")
            self.connection_label.setStyleSheet(f"color: {COLORS['error']};")
            self.status_label.setText("NINA disconnected - click Reconnect to retry")
            self.status_label.setStyleSheet(f"color: {COLORS['warning']};")
            # Disable all action buttons when disconnected
            self.imaging_start_btn.setEnabled(False)
            self.imaging_stop_btn.setEnabled(False)
            self.autofocus_start_btn.setEnabled(False)
            self.autofocus_cancel_btn.setEnabled(False)
            self.guiding_start_btn.setEnabled(False)
            self.guiding_stop_btn.setEnabled(False)
            self.mount_home_btn.setEnabled(False)
            self.mount_park_btn.setEnabled(False)
            self.mount_unpark_btn.setEnabled(False)
            self.mount_slew_btn.setEnabled(False)

    def _on_status_updated(self, status_data):
        """Handle status update from worker."""
        self._last_update = datetime.now()

        # Update last update display
        update_str = format_time(self._last_update)
        self.countdown_label.setText(f"Last update: {update_str}")

        # Update camera info
        camera = status_data.get('camera', {})
        if camera:
            is_exposing = camera.get('IsExposing', False)
            camera_connected = camera.get('Connected', False)

            # Update imaging button states based on connection and exposure status
            self.imaging_start_btn.setEnabled(self._connected and camera_connected and not is_exposing)
            self.imaging_stop_btn.setEnabled(self._connected and camera_connected and is_exposing)

            name = camera.get('Name') or camera.get('DeviceName', '--')
            self.camera_name_label.setText(name)

            connected = camera.get('Connected', False)
            if not connected:
                self.camera_status_label.setText("Disconnected")
                self.camera_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.camera_exposure_label.setText("--")
                self.camera_progress.setValue(0)
                self.camera_temp_label.setText("--")
                self._exposure_start_time = None
                self._exposure_end_time = None
                # Disable controls when disconnected (block signals to prevent callbacks)
                self.camera_cooling_checkbox.blockSignals(True)
                self.camera_cooling_checkbox.setChecked(False)
                self.camera_cooling_checkbox.setEnabled(False)
                self.camera_cooling_checkbox.blockSignals(False)

                self.camera_target_temp_spinbox.blockSignals(True)
                self.camera_target_temp_spinbox.setEnabled(False)
                self.camera_target_temp_spinbox.blockSignals(False)

                self.camera_dewheater_checkbox.blockSignals(True)
                self.camera_dewheater_checkbox.setChecked(False)
                self.camera_dewheater_checkbox.setEnabled(False)
                self.camera_dewheater_checkbox.blockSignals(False)
            else:
                # Determine camera state
                if is_exposing:
                    self.camera_status_label.setText("Exposing")
                    self.camera_status_label.setStyleSheet(f"color: {COLORS['success']};")
                else:
                    self.camera_status_label.setText("Idle")
                    self.camera_status_label.setStyleSheet(f"color: {COLORS['text']};")

                # Exposure progress - calculate from ExposureEndTime
                exposure_end_str = camera.get('ExposureEndTime')
                if is_exposing and exposure_end_str:
                    try:
                        # Parse ISO format timestamp
                        from datetime import timezone
                        exposure_end = datetime.fromisoformat(exposure_end_str.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc) if exposure_end.tzinfo else datetime.now()

                        # If exposure end time changed, this is a new exposure
                        if self._exposure_end_time != exposure_end:
                            self._exposure_end_time = exposure_end
                            self._exposure_start_time = now

                        # Calculate remaining and total time
                        remaining = (exposure_end - now).total_seconds()
                        if remaining < 0:
                            remaining = 0

                        if self._exposure_start_time:
                            total_time = (exposure_end - self._exposure_start_time).total_seconds()
                            if total_time > 0:
                                elapsed = total_time - remaining
                                progress_pct = (elapsed / total_time) * 100
                                self.camera_progress.setValue(int(min(100, max(0, progress_pct))))
                                self.camera_exposure_label.setText(f"{remaining:.0f}s / {total_time:.0f}s")
                            else:
                                self.camera_exposure_label.setText(f"{remaining:.0f}s remaining")
                                self.camera_progress.setValue(0)
                        else:
                            self.camera_exposure_label.setText(f"{remaining:.0f}s remaining")
                            self.camera_progress.setValue(0)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing ExposureEndTime: {e}")
                        self.camera_exposure_label.setText("Exposing...")
                        self.camera_progress.setValue(0)
                elif is_exposing:
                    # Exposing but no end time info
                    self.camera_exposure_label.setText("Exposing...")
                    self.camera_progress.setValue(0)
                else:
                    self.camera_exposure_label.setText("--")
                    self.camera_progress.setValue(0)
                    self._exposure_start_time = None
                    self._exposure_end_time = None

                # Camera temperature
                temp = camera.get('Temperature')
                if temp is not None:
                    self.camera_temp_label.setText(f"{temp:.1f}°C")
                else:
                    self.camera_temp_label.setText("--")

                # Sync cooling controls with camera state (skip if user recently changed)
                self.camera_cooling_checkbox.setEnabled(True)
                self.camera_target_temp_spinbox.setEnabled(True)
                self.camera_dewheater_checkbox.setEnabled(True)

                # Only sync cooling on/off state on initial load (when we haven't set it yet)
                # After user sets it, we respect their choice and don't override
                # Note: We don't sync target temp - that's user-controlled only
                cooling_on_raw = camera.get('CoolerOn', False)
                cooler_power = camera.get('CoolerPower', 0)
                target_temp = camera.get('TargetTemp') or camera.get('TemperatureSetPoint')
                at_target = camera.get('AtTargetTemp', False)

                # Handle potential string values from API
                if isinstance(cooling_on_raw, str):
                    cooling_on = cooling_on_raw.lower() in ('true', '1', 'yes')
                else:
                    cooling_on = bool(cooling_on_raw)
                    logger.debug(f"[Cooling Sync] API: CoolerOn={cooling_on_raw!r}, CoolerPower={cooler_power}, TargetTemp={target_temp}, AtTarget={at_target}")

                if self._last_cooling_enabled is None and not self._user_changing_cooling:
                    logger.debug(f"[Cooling Sync] Initial sync - setting checkbox to {cooling_on}")
                    # Block signals to prevent triggering callbacks
                    self.camera_cooling_checkbox.blockSignals(True)
                    self.camera_cooling_checkbox.setChecked(cooling_on)
                    self.camera_cooling_checkbox.blockSignals(False)
                    logger.debug(f"[Cooling Sync] After setChecked({cooling_on}), checkbox is now: {self.camera_cooling_checkbox.isChecked()}")
                    # Mark as synced so we don't repeat
                    self._last_cooling_enabled = cooling_on

                # Only sync dew heater state if user isn't actively changing it
                dewheater_on = camera.get('DewHeaterOn', False)
                logger.debug(f"[DewHeater Sync] API DewHeaterOn={dewheater_on}, _user_changing_dewheater={self._user_changing_dewheater}")
                if not self._user_changing_dewheater:
                    self.camera_dewheater_checkbox.blockSignals(True)
                    self.camera_dewheater_checkbox.setChecked(dewheater_on)
                    self.camera_dewheater_checkbox.blockSignals(False)
        else:
            # No camera data from API - show as disconnected
            self.camera_name_label.setText("--")
            self.camera_status_label.setText("Disconnected")
            self.camera_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
            self.camera_exposure_label.setText("--")
            self.camera_progress.setValue(0)
            self.camera_temp_label.setText("--")
            self.camera_cooling_checkbox.blockSignals(True)
            self.camera_cooling_checkbox.setChecked(False)
            self.camera_cooling_checkbox.setEnabled(False)
            self.camera_cooling_checkbox.blockSignals(False)
            self.camera_target_temp_spinbox.setEnabled(False)
            self.camera_dewheater_checkbox.blockSignals(True)
            self.camera_dewheater_checkbox.setChecked(False)
            self.camera_dewheater_checkbox.setEnabled(False)
            self.camera_dewheater_checkbox.blockSignals(False)
            self._last_cooling_enabled = None

        # Update mount info
        mount = status_data.get('mount', {})
        if mount:
            name = mount.get('Name') or mount.get('DeviceName', '--')
            self.mount_name_label.setText(name)

            mount_connected = mount.get('Connected', False)
            if not mount_connected:
                self.mount_status_label.setText("Disconnected")
                self.mount_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.mount_coords_label.setText("--")
                # Disable mount buttons when mount disconnected
                self.mount_home_btn.setEnabled(False)
                self.mount_park_btn.setEnabled(False)
                self.mount_unpark_btn.setEnabled(False)
                self.mount_slew_btn.setEnabled(False)
            else:
                tracking = mount.get('TrackingEnabled', False) or mount.get('Tracking', False)
                slewing = mount.get('Slewing', False)
                at_park = mount.get('AtPark', False)

                if slewing:
                    self.mount_status_label.setText("Slewing")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['info']};")
                elif at_park:
                    self.mount_status_label.setText("Parked")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['text']};")
                elif tracking:
                    self.mount_status_label.setText("Tracking")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['success']};")
                else:
                    self.mount_status_label.setText("Idle")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['text']};")

                # Update mount button states based on park status
                self.mount_home_btn.setEnabled(self._connected and not at_park and not slewing)
                self.mount_park_btn.setEnabled(self._connected and not at_park and not slewing)
                self.mount_unpark_btn.setEnabled(self._connected and at_park)
                self.mount_slew_btn.setEnabled(self._connected and not at_park and not slewing)

                # Coordinates
                ra = mount.get('RightAscension', 0) or mount.get('RA', 0)
                dec = mount.get('Declination', 0) or mount.get('Dec', 0)
                if ra or dec:
                    # Convert RA from hours to HH:MM:SS
                    ra_h = int(ra)
                    ra_m = int((ra - ra_h) * 60)
                    ra_s = ((ra - ra_h) * 60 - ra_m) * 60
                    # Dec in degrees
                    dec_sign = '+' if dec >= 0 else '-'
                    dec_abs = abs(dec)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    self.mount_coords_label.setText(f"{ra_h:02d}h{ra_m:02d}m / {dec_sign}{dec_d}d{dec_m:02d}m")
                else:
                    self.mount_coords_label.setText("--")
        else:
            # No mount data from API - show as disconnected
            self.mount_name_label.setText("--")
            self.mount_status_label.setText("Disconnected")
            self.mount_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
            self.mount_coords_label.setText("--")
            # Disable mount buttons when no mount data
            self.mount_home_btn.setEnabled(False)
            self.mount_park_btn.setEnabled(False)
            self.mount_unpark_btn.setEnabled(False)
            self.mount_slew_btn.setEnabled(False)

        # Update guider info
        guider = status_data.get('guider', {})
        if guider:
            name = guider.get('Name') or guider.get('DeviceName', '--')
            self.guider_name_label.setText(name)

            guider_connected = guider.get('Connected', False)
            if not guider_connected:
                self.guider_status_label.setText("Disconnected")
                self.guider_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.guider_rms_label.setText("--")
                # Disable guiding buttons when guider disconnected
                self.guiding_start_btn.setEnabled(False)
                self.guiding_stop_btn.setEnabled(False)
            else:
                # Check State field for guiding status
                guider_state = guider.get('State', 'Stopped')
                is_guiding = guider_state == 'Guiding'
                if is_guiding:
                    self.guider_status_label.setText("Guiding")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['success']};")
                elif guider_state == 'Calibrating':
                    self.guider_status_label.setText("Calibrating")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['warning']};")
                elif guider_state == 'Looping':
                    self.guider_status_label.setText("Looping")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['text']};")
                elif guider_state == 'LostLock':
                    self.guider_status_label.setText("Lost Lock")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['error']};")
                else:
                    self.guider_status_label.setText("Idle")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['text']};")

                # Update guiding button states
                self.guiding_start_btn.setEnabled(self._connected and not is_guiding)
                self.guiding_stop_btn.setEnabled(self._connected and is_guiding)

                # RMS - extract from nested RMSError object
                rms_error = guider.get('RMSError', {})
                if isinstance(rms_error, dict):
                    # RMSError contains RA, Dec, Total objects with Pixel and Arcseconds values
                    rms_total = rms_error.get('Total', {})
                    if isinstance(rms_total, dict):
                        total_arcsec = rms_total.get('Arcseconds', 0) or 0
                        if total_arcsec:
                            self.guider_rms_label.setText(f'{total_arcsec:.2f}"')
                        else:
                            self.guider_rms_label.setText("--")
                    else:
                        self.guider_rms_label.setText("--")
                else:
                    self.guider_rms_label.setText("--")
        else:
            # No guider data from API - show as disconnected
            self.guider_name_label.setText("--")
            self.guider_status_label.setText("Disconnected")
            self.guider_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
            self.guider_rms_label.setText("--")
            # Disable guiding buttons when no guider data
            self.guiding_start_btn.setEnabled(False)
            self.guiding_stop_btn.setEnabled(False)

        # Update filter wheel info
        filterwheel = status_data.get('filterwheel', {})
        if filterwheel:
            name = filterwheel.get('Name') or filterwheel.get('DeviceName', '--')
            self.filterwheel_name_label.setText(name)

            connected = filterwheel.get('Connected', False)
            is_moving = filterwheel.get('IsMoving', False)

            if not connected:
                self.filterwheel_status_label.setText("Disconnected")
                self.filterwheel_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.filterwheel_combo.setEnabled(False)
                self._updating_filterwheel = True
                self.filterwheel_combo.clear()
                self._updating_filterwheel = False
                self._available_filters = []
                self._last_filter_id = None
            else:
                if is_moving:
                    self.filterwheel_status_label.setText("Moving...")
                    self.filterwheel_status_label.setStyleSheet(f"color: {COLORS['warning']};")
                    self.filterwheel_combo.setEnabled(False)
                else:
                    self.filterwheel_status_label.setText("Connected")
                    self.filterwheel_status_label.setStyleSheet(f"color: {COLORS['success']};")
                    # Only enable if user isn't actively changing
                    if not self._user_changing_filter:
                        self.filterwheel_combo.setEnabled(True)

                # Update available filters list if changed
                available = filterwheel.get('AvailableFilters', [])
                if available != self._available_filters:
                    logger.debug(f"[FilterWheel] Available filters changed: {available}")
                    self._available_filters = available
                    self._updating_filterwheel = True
                    self.filterwheel_combo.clear()
                    for f in available:
                        filter_name = f.get('Name', f'Filter {f.get("Id", "?")}')
                        filter_id = f.get('Id', -1)
                        self.filterwheel_combo.addItem(filter_name, filter_id)
                    self._updating_filterwheel = False

                # Sync current filter selection (only if user isn't actively changing)
                if not self._user_changing_filter:
                    selected = filterwheel.get('SelectedFilter') or filterwheel.get('Filter')
                    if selected and isinstance(selected, dict):
                        current_id = selected.get('Id')
                        if current_id is not None and current_id != self._last_filter_id:
                            logger.debug(f"[FilterWheel Sync] Current filter changed: ID={current_id}, last={self._last_filter_id}")
                            self._last_filter_id = current_id
                            # Find and select the matching filter in combo
                            self._updating_filterwheel = True
                            for i in range(self.filterwheel_combo.count()):
                                if self.filterwheel_combo.itemData(i) == current_id:
                                    self.filterwheel_combo.setCurrentIndex(i)
                                    break
                            self._updating_filterwheel = False
        else:
            # No filter wheel data from API - show as disconnected
            self.filterwheel_name_label.setText("--")
            self.filterwheel_status_label.setText("Disconnected")
            self.filterwheel_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
            self.filterwheel_combo.setEnabled(False)
            self._updating_filterwheel = True
            self.filterwheel_combo.clear()
            self._updating_filterwheel = False
            self._available_filters = []
            self._last_filter_id = None

        # Update focuser info
        focuser = status_data.get('focuser', {})
        if focuser:
            name = focuser.get('Name') or focuser.get('DeviceName', '--')
            self.focuser_name_label.setText(name)

            focuser_connected = focuser.get('Connected', False)
            if not focuser_connected:
                self.focuser_status_label.setText("Disconnected")
                self.focuser_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.focuser_position_label.setText("--")
                self.focuser_temp_label.setText("--")
                # Disable autofocus buttons when focuser disconnected
                self.autofocus_start_btn.setEnabled(False)
                self.autofocus_cancel_btn.setEnabled(False)
            else:
                is_moving = focuser.get('IsMoving', False)
                if is_moving:
                    self.focuser_status_label.setText("Moving")
                    self.focuser_status_label.setStyleSheet(f"color: {COLORS['info']};")
                else:
                    self.focuser_status_label.setText("Connected")
                    self.focuser_status_label.setStyleSheet(f"color: {COLORS['success']};")

                # Update autofocus button states based on focuser movement
                self.autofocus_start_btn.setEnabled(self._connected and not is_moving)
                self.autofocus_cancel_btn.setEnabled(self._connected and is_moving)

                # Position
                position = focuser.get('Position')
                if position is not None:
                    self.focuser_position_label.setText(str(position))
                else:
                    self.focuser_position_label.setText("--")

                # Temperature
                temp = focuser.get('Temperature')
                if temp is not None:
                    self.focuser_temp_label.setText(f"{temp:.1f}°C")
                else:
                    self.focuser_temp_label.setText("--")
        else:
            # No focuser data from API - show as disconnected
            self.focuser_name_label.setText("--")
            self.focuser_status_label.setText("Disconnected")
            self.focuser_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
            self.focuser_position_label.setText("--")
            self.focuser_temp_label.setText("--")
            # Disable autofocus buttons when no focuser data
            self.autofocus_start_btn.setEnabled(False)
            self.autofocus_cancel_btn.setEnabled(False)

    def _on_image_updated(self, image_data, image_meta):
        """Handle image update from worker."""
        if image_data:
            try:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                if not pixmap.isNull():
                    self._current_image_pixmap = pixmap
                    self.image_label.setPixmap(pixmap)
            except Exception as e:
                logger.error(f"Error loading image: {e}")

        # Update image info from metadata
        if image_meta:
            # Try various field names that might be in the history
            target = (
                image_meta.get('TargetName') or
                image_meta.get('Target') or
                image_meta.get('targetName') or
                '--'
            )
            exp = (
                image_meta.get('ExposureTime') or
                image_meta.get('Duration') or
                image_meta.get('Exposure') or
                image_meta.get('exposureTime') or
                '--'
            )
            if isinstance(exp, (int, float)):
                exp = f"{exp:.0f}s"

            # Try to get filter info
            filter_name = (
                image_meta.get('FilterName') or
                image_meta.get('Filter') or
                image_meta.get('filter') or
                ''
            )

            info_text = f"Target: {target} | Exp: {exp}"
            if filter_name:
                info_text += f" | {filter_name}"

            self.image_info_label.setText(info_text)
        else:
            self.image_info_label.setText("Target: -- | Exp: --")

    def _on_livestack_updated(self, image_data, status):
        """Handle livestack update from worker."""
        is_running = status.get('running', False)
        if is_running:
            # Update livestack tab with indicator
            self.image_tabs.setTabText(1, "Live Stack *")
            if image_data:
                try:
                    pixmap = QPixmap()
                    pixmap.loadFromData(image_data)
                    if not pixmap.isNull():
                        self._current_livestack_pixmap = pixmap
                        self.livestack_label.setPixmap(pixmap)
                        self.livestack_info_label.setText("Live stack active")
                except Exception as e:
                    logger.error(f"Error loading livestack image: {e}")
            else:
                # Running but no image yet
                self.livestack_label.setPlaceholderText("Waiting for first image...")
                self.livestack_label.setPixmap(None)
                self.livestack_info_label.setText("Live stack active")
        else:
            # Reset tab text and show inactive message
            self.image_tabs.setTabText(1, "Live Stack")
            self.livestack_label.setPlaceholderText("Live stack not active")
            self.livestack_label.setPixmap(None)
            self._current_livestack_pixmap = None
            self.livestack_info_label.setText("")

    def _on_guiding_updated(self, guiding_data):
        """Handle guiding data update from worker."""
        if guiding_data:
            self.guiding_graph.update_data(guiding_data)

    def _on_error(self, error_message):
        """Handle error from worker."""
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_cooling_changed(self, state):
        """Handle cooling checkbox change."""
        if self._updating_camera_controls:
            logger.debug(f"[Cooling] Ignoring change - _updating_camera_controls is True")
            return

        enabled = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else state == 2
        temp = self.camera_target_temp_spinbox.value() if enabled else None

        logger.debug(f"[Cooling] User changed: enabled={enabled}, temp={temp}, last_enabled={self._last_cooling_enabled}, last_temp={self._last_cooling_temp}")

        # Skip if this is the same state we already sent
        if enabled == self._last_cooling_enabled and (not enabled or temp == self._last_cooling_temp):
            logger.debug(f"[Cooling] Skipping - same state already sent")
            return

        # Set flag and start timer to clear it after 15 seconds
        self._user_changing_cooling = True
        self._cooling_change_timer.start(15000)
        logger.debug(f"[Cooling] Set _user_changing_cooling=True, started 15s timer")

        if enabled:
            self.status_label.setText(f"Setting cooling to {temp}°C...")
        else:
            self.status_label.setText("Disabling cooling...")

        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.set_camera_cooling(host, port, enabled, temp)

        if success:
            self._last_cooling_enabled = enabled
            self._last_cooling_temp = temp
            logger.debug(f"[Cooling] Success - updated last_enabled={enabled}, last_temp={temp}")
            self.status_label.setText("Cooling " + ("enabled" if enabled else "disabled"))
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        else:
            logger.debug(f"[Cooling] Failed - reverting checkbox")
            self.status_label.setText("Failed to change cooling - check console")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")
            # Revert checkbox state
            self.camera_cooling_checkbox.blockSignals(True)
            self.camera_cooling_checkbox.setChecked(not enabled)
            self.camera_cooling_checkbox.blockSignals(False)
            # Clear the flag since we reverted
            self._user_changing_cooling = False
            self._cooling_change_timer.stop()

    def _clear_cooling_change_flag(self):
        """Clear the cooling change flag after timeout."""
        logger.debug(f"[Cooling] Timer expired - clearing _user_changing_cooling flag")
        self._user_changing_cooling = False

    def _on_target_temp_changed(self):
        """Handle target temperature change."""
        if self._updating_camera_controls:
            logger.debug(f"[Cooling] Target temp change ignored - _updating_camera_controls is True")
            return

        temp = self.camera_target_temp_spinbox.value()

        # Skip if cooling is off - just remember the temp for when it's turned on
        if not self.camera_cooling_checkbox.isChecked():
            logger.debug(f"[Cooling] Target temp change ignored - cooling is off")
            return

        # Skip if this is the same temp we already sent
        if temp == self._last_cooling_temp:
            logger.debug(f"[Cooling] Target temp change ignored - same as last ({temp})")
            return

        logger.debug(f"[Cooling] User changed target temp: {temp}°C (last was {self._last_cooling_temp})")

        # Set flag to prevent sync from overriding
        self._user_changing_cooling = True
        self._cooling_change_timer.start(15000)

        self.status_label.setText(f"Setting target temp to {temp}°C...")

        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.set_camera_cooling(host, port, True, temp)

        if success:
            self._last_cooling_temp = temp
            logger.debug(f"[Cooling] Target temp success - last_temp={temp}")
            self.status_label.setText(f"Target temp set to {temp}°C")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        else:
            logger.debug(f"[Cooling] Target temp failed")
            self.status_label.setText("Failed to set target temperature")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_dewheater_changed(self, state):
        """Handle dew heater checkbox change."""
        if self._updating_camera_controls:
            logger.debug(f"[DewHeater] Ignoring change - _updating_camera_controls is True")
            return

        # Set flag and start timer to clear it after 15 seconds
        self._user_changing_dewheater = True
        self._dewheater_change_timer.start(15000)

        enabled = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else state == 2
        logger.debug(f"[DewHeater] User changed: enabled={enabled}")
        self.status_label.setText("Turning dew heater " + ("on" if enabled else "off") + "...")

        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.set_camera_dew_heater(host, port, enabled)

        if success:
            logger.debug(f"[DewHeater] Success")
            self.status_label.setText("Dew heater " + ("enabled" if enabled else "disabled"))
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        else:
            logger.debug(f"[DewHeater] Failed - reverting checkbox")
            self.status_label.setText("Failed to change dew heater")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")
            # Revert checkbox state
            self._updating_camera_controls = True
            self.camera_dewheater_checkbox.setChecked(not enabled)
            self._updating_camera_controls = False
            # Clear the flag since we reverted
            self._user_changing_dewheater = False
            self._dewheater_change_timer.stop()

    def _clear_dewheater_change_flag(self):
        """Clear the dew heater change flag after timeout."""
        logger.debug(f"[DewHeater] Timer expired - clearing _user_changing_dewheater flag")
        self._user_changing_dewheater = False

    def _on_imaging_start(self):
        """Start a camera capture."""
        dialog = CaptureSettingsDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        settings = dialog.get_settings()
        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.capture_image(
            host, port,
            duration=settings['duration'],
            gain=settings['gain'],
            save=settings['save'],
            image_type=settings['image_type']
        )
        if success:
            self.status_label.setText(f"Capture started ({settings['duration']}s)")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.imaging_start_btn.setEnabled(False)
            self.imaging_stop_btn.setEnabled(True)
        else:
            self.status_label.setText("Failed to start capture")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_imaging_stop(self):
        """Abort the current exposure."""
        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.abort_exposure(host, port)
        if success:
            self.status_label.setText("Exposure aborted")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.imaging_start_btn.setEnabled(True)
            self.imaging_stop_btn.setEnabled(False)
        else:
            self.status_label.setText("Failed to abort exposure")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_autofocus_start(self):
        """Start an autofocus run."""
        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.start_autofocus(host, port)
        if success:
            self.status_label.setText("AutoFocus started")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.autofocus_start_btn.setEnabled(False)
            self.autofocus_cancel_btn.setEnabled(True)
        else:
            self.status_label.setText("Failed to start AutoFocus")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_autofocus_cancel(self):
        """Cancel the running autofocus."""
        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.cancel_autofocus(host, port)
        if success:
            self.status_label.setText("AutoFocus cancelled")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.autofocus_start_btn.setEnabled(True)
            self.autofocus_cancel_btn.setEnabled(False)
        else:
            self.status_label.setText("Failed to cancel AutoFocus")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_guiding_start(self):
        """Start guiding."""
        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.start_guiding(host, port)
        if success:
            self.status_label.setText("Guiding started")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.guiding_start_btn.setEnabled(False)
            self.guiding_stop_btn.setEnabled(True)
        else:
            self.status_label.setText("Failed to start guiding")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_guiding_stop(self):
        """Stop guiding."""
        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.stop_guiding(host, port)
        if success:
            self.status_label.setText("Guiding stopped")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.guiding_start_btn.setEnabled(True)
            self.guiding_stop_btn.setEnabled(False)
        else:
            self.status_label.setText("Failed to stop guiding")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_mount_home(self):
        """Home the mount."""
        host, port = NINAIntegration.get_settings()
        self.mount_home_btn.setEnabled(False)
        self.status_label.setText("Homing mount...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        def do_home():
            return NINAIntegration.home_mount(host, port)

        def on_home_complete(success):
            if success:
                self.status_label.setText("Mount homing...")
            else:
                self.status_label.setText("Failed to home mount")
                self.status_label.setStyleSheet(f"color: {COLORS['error']};")
            # Button state will be updated by status polling

        self._run_in_background(do_home, on_home_complete)

    def _on_mount_park(self):
        """Park the mount."""
        host, port = NINAIntegration.get_settings()
        self.mount_park_btn.setEnabled(False)
        self.status_label.setText("Parking mount...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        def do_park():
            return NINAIntegration.park_mount(host, port)

        def on_park_complete(success):
            if success:
                self.status_label.setText("Mount parking...")
                self.mount_unpark_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to park mount")
                self.status_label.setStyleSheet(f"color: {COLORS['error']};")
                self.mount_park_btn.setEnabled(True)

        self._run_in_background(do_park, on_park_complete)

    def _on_mount_unpark(self):
        """Unpark the mount."""
        host, port = NINAIntegration.get_settings()
        self.mount_unpark_btn.setEnabled(False)
        self.status_label.setText("Unparking mount...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        def do_unpark():
            return NINAIntegration.unpark_mount(host, port)

        def on_unpark_complete(success):
            if success:
                self.status_label.setText("Mount unparking...")
                self.mount_park_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to unpark mount")
                self.status_label.setStyleSheet(f"color: {COLORS['error']};")
                self.mount_unpark_btn.setEnabled(True)

        self._run_in_background(do_unpark, on_unpark_complete)

    def _on_mount_slew(self):
        """Slew the mount to coordinates."""
        dialog = SlewDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        ra_deg, dec_deg = dialog.get_coordinates_degrees()
        host, port = NINAIntegration.get_settings()

        self.status_label.setText(f"Slewing to RA={ra_deg:.4f}° Dec={dec_deg:.4f}°...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.mount_slew_btn.setEnabled(False)

        # Run slew in background thread to avoid blocking UI
        def do_slew():
            return NINAIntegration.slew_mount(host, port, ra_deg, dec_deg)

        def on_slew_complete(success):
            if success:
                self.status_label.setText("Slew started")
                self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            else:
                self.status_label.setText("Failed to start slew")
                self.status_label.setStyleSheet(f"color: {COLORS['error']};")
            # Button will be re-enabled by status updates when slew completes

        self._run_in_background(do_slew, on_slew_complete)

    def _on_filter_changed(self, index):
        """Handle filter selection change."""
        if self._updating_filterwheel:
            logger.debug(f"[FilterWheel] Ignoring change - _updating_filterwheel is True")
            return

        if index < 0:
            return

        filter_id = self.filterwheel_combo.itemData(index)
        filter_name = self.filterwheel_combo.itemText(index)

        # Skip if this is the same filter we already have
        if filter_id == self._last_filter_id:
            logger.debug(f"[FilterWheel] Skipping - same filter already selected (ID={filter_id})")
            return

        logger.debug(f"[FilterWheel] User selected: {filter_name} (ID={filter_id}), last={self._last_filter_id}")

        # Set flag to prevent sync from overriding during change
        self._user_changing_filter = True
        self.filterwheel_combo.setEnabled(False)
        self.status_label.setText(f"Changing to {filter_name}...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        host, port = NINAIntegration.get_settings()
        success = NINAIntegration.change_filter(host, port, filter_id)

        if success:
            self._last_filter_id = filter_id
            logger.debug(f"[FilterWheel] Success - filter changed to {filter_name}")
            self.status_label.setText(f"Filter changed to {filter_name}")
            self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        else:
            logger.debug(f"[FilterWheel] Failed - reverting selection")
            self.status_label.setText("Failed to change filter")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")
            # Revert combo selection to last known good filter
            if self._last_filter_id is not None:
                self._updating_filterwheel = True
                for i in range(self.filterwheel_combo.count()):
                    if self.filterwheel_combo.itemData(i) == self._last_filter_id:
                        self.filterwheel_combo.setCurrentIndex(i)
                        break
                self._updating_filterwheel = False

        # Clear the flag and re-enable combo
        self._user_changing_filter = False
        self.filterwheel_combo.setEnabled(True)

    def _run_in_background(self, func, callback):
        """Run a function in a background thread and call callback with result on completion."""
        class BackgroundWorker(QThread):
            finished_with_result = Signal(object)

            def __init__(self, func):
                super().__init__()
                self._func = func

            def run(self):
                result = self._func()
                self.finished_with_result.emit(result)

        worker = BackgroundWorker(func)
        worker.finished_with_result.connect(callback)
        worker.finished.connect(worker.deleteLater)
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_background_workers'):
            self._background_workers = []
        self._background_workers.append(worker)
        worker.finished.connect(lambda: self._background_workers.remove(worker) if worker in self._background_workers else None)
        worker.start()

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        # ZoomableImageWidget handles its own resizing

    def closeEvent(self, event):
        """Clean up when window is closed."""
        self._save_settings()
        self._stop_worker()
        super().closeEvent(event)


def main():
    """Main entry point for standalone testing."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = NINADashboardWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
