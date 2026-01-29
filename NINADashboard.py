#!/usr/bin/env python3
"""
NINA Dashboard Window for Cosmos Collection
Displays real-time NINA status, current imaging, live stack images, and guiding graphs.
"""

import hashlib
import logging
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

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QGroupBox, QProgressBar, QComboBox, QSplitter, QFrame,
    QGridLayout, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage

from NINAIntegration import NINAIntegration
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from TimeFormatHelper import format_time

# Set up logging
logger = logging.getLogger(__name__)


class NINAStatusWorker(QThread):
    """Background thread for polling NINA API endpoints."""

    status_updated = Signal(dict)  # Emits combined status data
    image_updated = Signal(bytes, dict)  # Emits image data and metadata
    livestack_updated = Signal(bytes, dict)  # Emits livestack image and status
    guiding_updated = Signal(list)  # Emits guiding graph data points
    error_occurred = Signal(str)  # Emits error message
    connection_changed = Signal(bool, str)  # Emits connected state and version

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self._running = False
        self._fetch_images = True
        self._was_exposing = False  # Track exposure state to detect when exposure completes
        self._exposure_end_time = None  # Expected end time of current exposure
        self._waiting_for_new_image = False  # Keep checking until new image is saved
        self._initial_image_check_done = False  # Have we done the initial image check?
        self._last_image_index = -1  # Track the last known image index (-1 = no images yet)
        self._last_livestack_hash = None  # Track livestack image hash
        self._consecutive_failures = 0  # Track consecutive API failures to detect disconnect

    def run(self):
        """Main polling loop."""
        self._running = True

        # Test connection first
        success, message, version = NINAIntegration.test_connection(self.host, self.port)
        if success:
            self.connection_changed.emit(True, version or "Unknown")
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

                # Check if we got any data - if not, NINA might be disconnected
                if status_data:
                    self._consecutive_failures = 0  # Reset on success
                    self.status_updated.emit(status_data)
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 3:
                        logger.debug("NINA connection lost (3 consecutive failures)")
                        self.connection_changed.emit(False, "")
                        self._consecutive_failures = 0  # Reset to avoid repeated signals

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
                if livestack_status and isinstance(livestack_status, dict) and livestack_status.get('Enabled'):
                    livestack_image = NINAIntegration.get_livestack_image(self.host, self.port)
                    if livestack_image:
                        # Only emit if livestack image changed
                        current_hash = hashlib.md5(livestack_image).hexdigest()
                        if current_hash != self._last_livestack_hash:
                            self._last_livestack_hash = current_hash
                            self.livestack_updated.emit(livestack_image, livestack_status)
                else:
                    # Emit empty to hide livestack panel (only if it was visible)
                    if self._last_livestack_hash is not None:
                        self._last_livestack_hash = None
                        self.livestack_updated.emit(b'', {})

                # Fetch guiding graph data only if guider is connected and guiding
                guider = status_data.get('guider', {})
                is_guiding = guider.get('Connected', False) and (
                    guider.get('Guiding', False) or guider.get('IsGuiding', False)
                )
                if is_guiding:
                    guiding_data = NINAIntegration.get_guiding_graph_data(self.host, self.port)
                    if guiding_data and isinstance(guiding_data, list):
                        self.guiding_updated.emit(guiding_data)

            except Exception as e:
                logger.error(f"Error in NINA status worker: {e}")
                self.error_occurred.emit(str(e))

            # Sleep for polling interval (handled by caller via timer)
            if self._running:
                self.msleep(100)  # Small sleep to allow thread control

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


class NINADashboardWindow(WindowPositionMixin, QMainWindow):
    """Main NINA Dashboard window."""
    WINDOW_POSITION_KEY = "NINADashboard"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NINA Dashboard - Cosmos Collection")
        self.resize(900, 700)
        self.setup_window_position()

        self.worker = None
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_nina)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._update_countdown)

        self._connected = False
        self._version = ""
        self._last_update = None
        self._next_refresh_seconds = 0
        self._exposure_start_time = None
        self._exposure_end_time = None
        self._user_adjusted_splitter = False  # Track if user manually adjusted splitter
        self._current_image_pixmap = None  # Store original pixmap for rescaling
        self._current_livestack_pixmap = None  # Store original livestack pixmap

        self._setup_ui()
        self._auto_connect()
        # Defer settings restore until after window is shown (splitters need visible geometry)
        QTimer.singleShot(100, self._restore_settings)

    def _setup_ui(self):
        """Set up the main window UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

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

        header_layout.addWidget(QLabel("Refresh:"))
        self.refresh_combo = QComboBox()
        self.refresh_combo.addItem("2s", 2)
        self.refresh_combo.addItem("5s", 5)
        self.refresh_combo.addItem("10s", 10)
        self.refresh_combo.addItem("15s", 15)
        self.refresh_combo.addItem("30s", 30)
        self.refresh_combo.setCurrentIndex(1)  # Default 5s
        self.refresh_combo.currentIndexChanged.connect(self._on_refresh_changed)
        header_layout.addWidget(self.refresh_combo)

        main_layout.addLayout(header_layout)

        # Main content area with splitter
        self.horizontal_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Equipment Status
        status_group = QGroupBox("Equipment Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)

        # Camera status
        camera_frame = QFrame()
        camera_frame.setFrameShape(QFrame.StyledPanel)
        camera_layout = QGridLayout(camera_frame)
        camera_layout.setContentsMargins(8, 8, 8, 8)

        camera_layout.addWidget(QLabel("<b>Camera</b>"), 0, 0, 1, 2)
        camera_layout.addWidget(QLabel("Name:"), 1, 0)
        self.camera_name_label = QLabel("--")
        camera_layout.addWidget(self.camera_name_label, 1, 1)
        camera_layout.addWidget(QLabel("Status:"), 2, 0)
        self.camera_status_label = QLabel("--")
        camera_layout.addWidget(self.camera_status_label, 2, 1)
        camera_layout.addWidget(QLabel("Exposure:"), 3, 0)
        self.camera_exposure_label = QLabel("--")
        camera_layout.addWidget(self.camera_exposure_label, 3, 1)
        camera_layout.addWidget(QLabel("Progress:"), 4, 0)
        self.camera_progress = QProgressBar()
        self.camera_progress.setMaximum(100)
        self.camera_progress.setValue(0)
        camera_layout.addWidget(self.camera_progress, 4, 1)

        status_layout.addWidget(camera_frame)

        # Mount status
        mount_frame = QFrame()
        mount_frame.setFrameShape(QFrame.StyledPanel)
        mount_layout = QGridLayout(mount_frame)
        mount_layout.setContentsMargins(8, 8, 8, 8)

        mount_layout.addWidget(QLabel("<b>Mount</b>"), 0, 0, 1, 2)
        mount_layout.addWidget(QLabel("Name:"), 1, 0)
        self.mount_name_label = QLabel("--")
        mount_layout.addWidget(self.mount_name_label, 1, 1)
        mount_layout.addWidget(QLabel("Status:"), 2, 0)
        self.mount_status_label = QLabel("--")
        mount_layout.addWidget(self.mount_status_label, 2, 1)
        mount_layout.addWidget(QLabel("RA/Dec:"), 3, 0)
        self.mount_coords_label = QLabel("--")
        mount_layout.addWidget(self.mount_coords_label, 3, 1)

        status_layout.addWidget(mount_frame)

        # Guider status
        guider_frame = QFrame()
        guider_frame.setFrameShape(QFrame.StyledPanel)
        guider_layout = QGridLayout(guider_frame)
        guider_layout.setContentsMargins(8, 8, 8, 8)

        guider_layout.addWidget(QLabel("<b>Guider</b>"), 0, 0, 1, 2)
        guider_layout.addWidget(QLabel("Name:"), 1, 0)
        self.guider_name_label = QLabel("--")
        guider_layout.addWidget(self.guider_name_label, 1, 1)
        guider_layout.addWidget(QLabel("Status:"), 2, 0)
        self.guider_status_label = QLabel("--")
        guider_layout.addWidget(self.guider_status_label, 2, 1)
        guider_layout.addWidget(QLabel("RMS:"), 3, 0)
        self.guider_rms_label = QLabel("--")
        guider_layout.addWidget(self.guider_rms_label, 3, 1)

        status_layout.addWidget(guider_frame)
        status_layout.addStretch()

        self.horizontal_splitter.addWidget(status_group)

        # Right panel - Images
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Current Image
        image_group = QGroupBox("Current Image")
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(5, 5, 5, 5)
        image_layout.setSpacing(2)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 150)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(f"background-color: {COLORS['background_light']}; border: 1px solid {COLORS['border']};")
        self.image_label.setText("No image available")
        image_layout.addWidget(self.image_label, 1)  # Stretch factor 1 - takes all available space

        self.image_info_label = QLabel("Target: -- | Exp: --")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setFixedHeight(20)
        self.image_info_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        image_layout.addWidget(self.image_info_label, 0)  # Stretch factor 0 - fixed size

        right_layout.addWidget(image_group, 1)  # Image group takes most space

        # Live Stack (hidden by default)
        self.livestack_group = QGroupBox("Live Stack")
        livestack_layout = QVBoxLayout(self.livestack_group)
        livestack_layout.setContentsMargins(5, 5, 5, 5)
        livestack_layout.setSpacing(2)

        self.livestack_label = QLabel()
        self.livestack_label.setAlignment(Qt.AlignCenter)
        self.livestack_label.setMinimumSize(200, 100)
        self.livestack_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.livestack_label.setStyleSheet(f"background-color: {COLORS['background_light']}; border: 1px solid {COLORS['border']};")
        self.livestack_label.setText("Live stack not active")
        livestack_layout.addWidget(self.livestack_label, 1)  # Stretch factor 1

        self.livestack_info_label = QLabel("Stack: -- frames | -- total")
        self.livestack_info_label.setAlignment(Qt.AlignCenter)
        self.livestack_info_label.setFixedHeight(20)
        self.livestack_info_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        livestack_layout.addWidget(self.livestack_info_label, 0)  # Stretch factor 0

        right_layout.addWidget(self.livestack_group, 0)  # Livestack group is smaller
        self.livestack_group.setVisible(False)

        self.horizontal_splitter.addWidget(right_panel)

        # Set initial horizontal splitter sizes and stretch factors
        # Stretch 0 for left panel (fixed), 1 for right panel (stretches)
        self.horizontal_splitter.setStretchFactor(0, 0)
        self.horizontal_splitter.setStretchFactor(1, 1)
        self.horizontal_splitter.setSizes([250, 550])

        # Guiding graph
        guiding_group = QGroupBox("Guiding Graph")
        guiding_layout = QVBoxLayout(guiding_group)
        guiding_layout.setContentsMargins(5, 5, 5, 5)

        self.guiding_graph = GuidingGraph(self)
        self.guiding_graph.setMinimumHeight(120)
        self.guiding_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        guiding_layout.addWidget(self.guiding_graph)

        # Vertical splitter to make guiding graph resizable
        self.vertical_splitter = QSplitter(Qt.Vertical)
        self.vertical_splitter.addWidget(self.horizontal_splitter)
        self.vertical_splitter.addWidget(guiding_group)
        self.vertical_splitter.setSizes([450, 200])  # Default sizes
        self.vertical_splitter.setChildrenCollapsible(False)  # Prevent fully collapsing panels

        main_layout.addWidget(self.vertical_splitter, 1)

        # Status bar
        status_layout_h = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        status_layout_h.addWidget(self.status_label)
        status_layout_h.addStretch()
        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        status_layout_h.addWidget(self.countdown_label)
        main_layout.addLayout(status_layout_h)

        # Track when user manually adjusts splitter (to distinguish from automatic layout changes)
        self.horizontal_splitter.splitterMoved.connect(self._on_splitter_moved)
        self.vertical_splitter.splitterMoved.connect(self._on_splitter_moved)

    def _on_splitter_moved(self):
        """Track when user manually moves a splitter."""
        self._user_adjusted_splitter = True

    def _restore_settings(self):
        """Restore saved panel sizes and refresh rate."""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Restore refresh rate
        refresh_index = settings.value("nina_dashboard_refresh_index", 1, type=int)
        if 0 <= refresh_index < self.refresh_combo.count():
            self.refresh_combo.setCurrentIndex(refresh_index)

        # Restore horizontal splitter sizes (stored as comma-separated string)
        h_sizes_str = settings.value("nina_dashboard_h_splitter", "", type=str)
        if h_sizes_str:
            try:
                sizes = [int(s.strip()) for s in h_sizes_str.split(",")]
                if len(sizes) == 2:
                    self.horizontal_splitter.setSizes(sizes)
                    logger.debug(f"Restored h_splitter: {sizes}")
            except (ValueError, TypeError):
                pass

        # Restore vertical splitter sizes (stored as comma-separated string)
        v_sizes_str = settings.value("nina_dashboard_v_splitter", "", type=str)
        if v_sizes_str:
            try:
                sizes = [int(s.strip()) for s in v_sizes_str.split(",")]
                if len(sizes) == 2:
                    self.vertical_splitter.setSizes(sizes)
                    logger.debug(f"Restored v_splitter: {sizes}")
            except (ValueError, TypeError):
                pass

    def _save_settings(self):
        """Save panel sizes and refresh rate."""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Save refresh rate
        settings.setValue("nina_dashboard_refresh_index", self.refresh_combo.currentIndex())

        # Only save splitter sizes if user manually adjusted them
        # This prevents automatic layout changes from overwriting saved values
        if self._user_adjusted_splitter:
            h_sizes = self.horizontal_splitter.sizes()
            settings.setValue("nina_dashboard_h_splitter", f"{h_sizes[0]},{h_sizes[1]}")

            v_sizes = self.vertical_splitter.sizes()
            settings.setValue("nina_dashboard_v_splitter", f"{v_sizes[0]},{v_sizes[1]}")

            logger.debug(f"Saved splitter sizes: h={h_sizes}, v={v_sizes}")
        else:
            logger.debug("Skipped saving splitter sizes (no user adjustment)")

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

        # Create and start worker
        self.worker = NINAStatusWorker(host, port)
        self.worker.connection_changed.connect(self._on_connection_changed)
        self.worker.status_updated.connect(self._on_status_updated)
        self.worker.image_updated.connect(self._on_image_updated)
        self.worker.livestack_updated.connect(self._on_livestack_updated)
        self.worker.guiding_updated.connect(self._on_guiding_updated)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _stop_worker(self):
        """Stop the worker thread and timers."""
        self.poll_timer.stop()
        self.countdown_timer.stop()

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
            self.status_label.setText("Connected - polling for data...")

            # Start polling timer
            interval = self.refresh_combo.currentData() * 1000
            self.poll_timer.start(interval)
            self._next_refresh_seconds = self.refresh_combo.currentData()
            self.countdown_timer.start(1000)
        else:
            self.connection_label.setText("Connection: Disconnected")
            self.connection_label.setStyleSheet(f"color: {COLORS['error']};")
            self.status_label.setText("NINA disconnected - click Reconnect to retry")
            self.status_label.setStyleSheet(f"color: {COLORS['warning']};")

    def _on_refresh_changed(self, index):
        """Handle refresh interval change."""
        self._save_settings()
        if self._connected:
            interval = self.refresh_combo.currentData() * 1000
            self.poll_timer.start(interval)
            self._next_refresh_seconds = self.refresh_combo.currentData()

    def _poll_nina(self):
        """Trigger a poll cycle."""
        if self.worker and self.worker.isRunning():
            # Worker will poll on next iteration
            self._next_refresh_seconds = self.refresh_combo.currentData()

    def _update_countdown(self):
        """Update the countdown display."""
        if self._next_refresh_seconds > 0:
            self._next_refresh_seconds -= 1

        if self._last_update:
            update_str = format_time(self._last_update)
            self.countdown_label.setText(f"Last update: {update_str} | Next refresh in {self._next_refresh_seconds}s")
        else:
            self.countdown_label.setText(f"Next refresh in {self._next_refresh_seconds}s")

    def _on_status_updated(self, status_data):
        """Handle status update from worker."""
        self._last_update = datetime.now()

        # Update camera info
        camera = status_data.get('camera', {})
        if camera:
            is_exposing = camera.get('IsExposing', False)

            name = camera.get('Name') or camera.get('DeviceName', '--')
            self.camera_name_label.setText(name)

            connected = camera.get('Connected', False)
            if not connected:
                self.camera_status_label.setText("Disconnected")
                self.camera_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.camera_exposure_label.setText("--")
                self.camera_progress.setValue(0)
                self._exposure_start_time = None
                self._exposure_end_time = None
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

        # Update mount info
        mount = status_data.get('mount', {})
        if mount:
            name = mount.get('Name') or mount.get('DeviceName', '--')
            self.mount_name_label.setText(name)

            connected = mount.get('Connected', False)
            if not connected:
                self.mount_status_label.setText("Disconnected")
                self.mount_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.mount_coords_label.setText("--")
            else:
                tracking = mount.get('TrackingEnabled', False) or mount.get('Tracking', False)
                slewing = mount.get('Slewing', False)
                if slewing:
                    self.mount_status_label.setText("Slewing")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['info']};")
                elif tracking:
                    self.mount_status_label.setText("Tracking")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['success']};")
                else:
                    self.mount_status_label.setText("Parked/Idle")
                    self.mount_status_label.setStyleSheet(f"color: {COLORS['text']};")

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

        # Update guider info
        guider = status_data.get('guider', {})
        if guider:
            name = guider.get('Name') or guider.get('DeviceName', '--')
            self.guider_name_label.setText(name)

            connected = guider.get('Connected', False)
            if not connected:
                self.guider_status_label.setText("Disconnected")
                self.guider_status_label.setStyleSheet(f"color: {COLORS['text_disabled']};")
                self.guider_rms_label.setText("--")
            else:
                guiding = guider.get('Guiding', False) or guider.get('IsGuiding', False)
                if guiding:
                    self.guider_status_label.setText("Guiding")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['success']};")
                else:
                    self.guider_status_label.setText("Idle")
                    self.guider_status_label.setStyleSheet(f"color: {COLORS['text']};")

                # RMS
                rms_ra = guider.get('RMSErrorRA', 0) or 0
                rms_dec = guider.get('RMSErrorDec', 0) or 0
                if rms_ra or rms_dec:
                    import math
                    total_rms = math.sqrt(rms_ra**2 + rms_dec**2)
                    self.guider_rms_label.setText(f'{total_rms:.2f}"')
                else:
                    self.guider_rms_label.setText("--")

    def _on_image_updated(self, image_data, image_meta):
        """Handle image update from worker."""
        if image_data:
            try:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                if not pixmap.isNull():
                    self._current_image_pixmap = pixmap
                    self._scale_image_to_label(self.image_label, pixmap)
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
        if image_data and status.get('Enabled'):
            self.livestack_group.setVisible(True)
            try:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                if not pixmap.isNull():
                    self._current_livestack_pixmap = pixmap
                    self._scale_image_to_label(self.livestack_label, pixmap)
            except Exception as e:
                logger.error(f"Error loading livestack image: {e}")

            # Update info
            frames = status.get('StackedImages', 0) or status.get('FrameCount', 0)
            total_exp = status.get('TotalExposure', 0)
            if total_exp:
                minutes = total_exp / 60
                self.livestack_info_label.setText(f"Stack: {frames} frames | {minutes:.1f} min total")
            else:
                self.livestack_info_label.setText(f"Stack: {frames} frames")
        else:
            self.livestack_group.setVisible(False)

    def _on_guiding_updated(self, guiding_data):
        """Handle guiding data update from worker."""
        if guiding_data:
            self.guiding_graph.update_data(guiding_data)

    def _on_error(self, error_message):
        """Handle error from worker."""
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _scale_image_to_label(self, label, pixmap):
        """Scale a pixmap to fit within a label while maintaining aspect ratio."""
        if pixmap is None or pixmap.isNull():
            return

        # Get the label's current size
        label_size = label.size()
        max_width = max(100, label_size.width() - 10)
        max_height = max(100, label_size.height() - 10)

        scaled = pixmap.scaled(
            max_width, max_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def resizeEvent(self, event):
        """Handle window resize - rescale images to fit."""
        super().resizeEvent(event)
        # Rescale images to fit new size
        if self._current_image_pixmap:
            self._scale_image_to_label(self.image_label, self._current_image_pixmap)
        if self._current_livestack_pixmap:
            self._scale_image_to_label(self.livestack_label, self._current_livestack_pixmap)

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
