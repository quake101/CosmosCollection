import logging
import math
import sys
import ssl
import threading
import urllib.parse
import urllib.request
from PySide6.QtCore import Qt, QUrl, QTimer, QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout,
    QComboBox, QCheckBox, QPushButton, QMessageBox,
    QToolButton, QMenu
)
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from UrlOpener import open_url
from NINAIntegration import NINAIntegration

logger = logging.getLogger(__name__)


class AladinLiteWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "AladinLite"

    SMART_TELESCOPES = [
        {'name': 'ZWO Seestar S30',         'aperture': 30,  'focal_length': 150, 'camera_text': 'ASI662MC (7.4x5.6mm) (Seestar S30)',            'smart_telescope': True},
        {'name': 'ZWO Seestar S30 Pro',     'aperture': 30,  'focal_length': 160, 'camera_text': 'IMX585 (11.1x6.3mm) (Seestar S30 Pro)',         'smart_telescope': True},
        {'name': 'ZWO Seestar S50',         'aperture': 50,  'focal_length': 250, 'camera_text': 'ASI462MC (2.9x2.9mm) (Seestar S50)',            'smart_telescope': True},
        {'name': 'Vaonis Vespera II',        'aperture': 50,  'focal_length': 250, 'camera_text': 'IMX585 (11.1x6.3mm) (Vaonis Vespera II)',       'smart_telescope': True},
        {'name': 'Celestron Origin',         'aperture': 152, 'focal_length': 335, 'camera_text': 'IMX178 (7.4x4.9mm) (Celestron Origin)',         'smart_telescope': True},
        {'name': 'Celestron Origin Mark II', 'aperture': 152, 'focal_length': 335, 'camera_text': 'IMX678 (7.7x4.3mm) (Celestron Origin Mark II)', 'smart_telescope': True},
        {'name': 'Dwarf 3',                  'aperture': 35,  'focal_length': 150, 'camera_text': 'IMX678 (7.7x4.3mm) (Dwarf 3)',                  'smart_telescope': True},
        {'name': 'Dwarf Mini',               'aperture': 30,  'focal_length': 150, 'camera_text': 'IMX662 (5.6x3.2mm) (Dwarf Mini)',               'smart_telescope': True},
    ]

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{data['name']} - FOV Simulator - Cosmos Collection")
        self.resize(1200, 800)
        self.setup_window_position()

        self.data = data
        self.telescopes = []
        self.selected_telescope = None
        self.current_fov = None
        self.current_target = None  # Track current target to preserve user changes
        self.observer_lat = None
        self.observer_lon = None

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
        self.telescope_combo.setMinimumWidth(280)
        self.telescope_combo.addItem("Default View", None)
        self.telescope_combo.currentTextChanged.connect(self._on_telescope_changed)

        self.show_smart_telescopes = QCheckBox("Show Smart Telescopes")
        self.show_smart_telescopes.setChecked(False)
        self.show_smart_telescopes.toggled.connect(self._on_show_smart_telescopes_toggled)

        # Load telescopes
        self._load_telescopes()

        # FOV display controls
        self.show_telescope_fov = QCheckBox("Show Telescope FOV")
        self.show_telescope_fov.setChecked(False)
        self.show_telescope_fov.toggled.connect(self._on_show_fov_toggled)

        self.show_horizon = QCheckBox("Virtual Horizon")
        self.show_horizon.setChecked(False)
        self.show_horizon.toggled.connect(self._on_show_horizon_toggled)

        # Camera/Eyepiece selection (for different FOVs)
        camera_label = QLabel("Camera/Eyepiece:")
        self.camera_combo = QComboBox()

        # Load user equipment first
        self._load_user_cameras_and_eyepieces()

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

        # ZWO cameras
        self.camera_combo.addItem("--- ZWO ASI CAMERAS ---", None)
        self.camera_combo.addItem("ASI6200MM Pro (36x24mm)", {"type": "camera", "sensor_width": 36.0, "sensor_height": 24.0})
        self.camera_combo.addItem("ASI2600MM Pro (23.5x15.7mm)", {"type": "camera", "sensor_width": 23.5, "sensor_height": 15.7})
        self.camera_combo.addItem("ASI533MM Pro (11.3x7.1mm)", {"type": "camera", "sensor_width": 11.3, "sensor_height": 7.1})
        self.camera_combo.addItem("ASI294MM Pro (19.1x13.0mm)", {"type": "camera", "sensor_width": 19.1, "sensor_height": 13.0})
        self.camera_combo.addItem("ASI183MM Pro (13.2x8.8mm)", {"type": "camera", "sensor_width": 13.2, "sensor_height": 8.8})
        self.camera_combo.addItem("ASI585MC (8.3x6.2mm)", {"type": "camera", "sensor_width": 8.3, "sensor_height": 6.2})
        self.camera_combo.addItem("ASI662MC (7.4x5.6mm) (Seestar S30)", {"type": "camera", "sensor_width": 7.4, "sensor_height": 5.6})
        self.camera_combo.addItem("IMX585 (11.1x6.3mm) (Seestar S30 Pro)", {"type": "camera", "sensor_width": 11.1, "sensor_height": 6.3})
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

        # Vaonis cameras
        self.camera_combo.addItem("--- VAONIS CAMERAS ---", None)
        self.camera_combo.addItem("IMX585 (11.1x6.3mm) (Vaonis Vespera II)", {"type": "camera", "sensor_width": 11.1, "sensor_height": 6.3})

        # Celestron cameras
        self.camera_combo.addItem("--- CELESTRON CAMERAS ---", None)
        self.camera_combo.addItem("IMX178 (7.4x4.9mm) (Celestron Origin)", {"type": "camera", "sensor_width": 7.4, "sensor_height": 4.9})
        self.camera_combo.addItem("IMX678 (7.7x4.3mm) (Celestron Origin Mark II)", {"type": "camera", "sensor_width": 7.7, "sensor_height": 4.3})

        self.camera_combo.currentTextChanged.connect(self._on_camera_changed)

        # Barlow/Reducer selection
        barlow_label = QLabel("Barlow/Reducer:")
        self.barlow_combo = QComboBox()

        # Optical accessories
        self.barlow_combo.addItem("None (1.0x)", {"factor": 1.0, "type": "none"})

        # Load user barlows/reducers
        self._load_user_barlows()

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
        telescope_layout.addWidget(self.show_smart_telescopes)
        telescope_layout.addWidget(self.show_telescope_fov)
        telescope_layout.addWidget(self.show_horizon)
        telescope_layout.addWidget(camera_label)
        telescope_layout.addWidget(self.camera_combo)
        telescope_layout.addWidget(barlow_label)
        telescope_layout.addWidget(self.barlow_combo)
        telescope_layout.addStretch()

        # Target List button
        target_menu = QMenu(self)

        self.add_target_action = QAction("Add to Target List", self)
        self.add_target_action.triggered.connect(self._add_to_target_list)
        target_menu.addAction(self.add_target_action)

        self.remove_target_action = QAction("Remove from Target List", self)
        self.remove_target_action.triggered.connect(self._remove_from_target_list)
        self.remove_target_action.setVisible(False)
        target_menu.addAction(self.remove_target_action)

        self.open_target_action = QAction("Open in Target List", self)
        self.open_target_action.triggered.connect(self._open_from_target_list)
        self.open_target_action.setVisible(False)
        target_menu.addAction(self.open_target_action)

        target_button = QToolButton()
        target_button.setText("Target List")
        target_button.setMenu(target_menu)
        target_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        telescope_layout.addWidget(target_button)

        # Add NINA button if integration is enabled
        if NINAIntegration.is_enabled():
            nina_menu = QMenu(self)

            framing_action = QAction("Send to Framing Assistant", self)
            framing_action.triggered.connect(self._send_to_nina)
            nina_menu.addAction(framing_action)

            slew_action = QAction("Slew to Target", self)
            slew_action.triggered.connect(self._slew_to_nina_target)
            nina_menu.addAction(slew_action)

            nina_button = QToolButton()
            nina_button.setText("NINA")
            nina_button.setToolTip("Send to NINA")
            nina_button.setMenu(nina_menu)
            nina_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            telescope_layout.addWidget(nina_button)

        layout.addLayout(telescope_layout)

        # Initialize web view as None initially - we'll create it safely later
        self.web_view = None
        self.web_view_error = None

        # Create a placeholder widget for the web view
        self.web_placeholder = QLabel("Loading Aladin Lite...")
        self.web_placeholder.setAlignment(Qt.AlignCenter)
        self.web_placeholder.setStyleSheet(f"QLabel {{ background-color: {COLORS['background']}; color: white; font-size: 14px; }}")
        self.web_placeholder.setMinimumSize(400, 300)

        layout.addWidget(self.web_placeholder)

        # Create a horizontal layout for the bottom controls
        bottom_layout = QHBoxLayout()

        # Add FOV information display
        self.fov_info_label = QLabel()
        self.fov_info_label.setStyleSheet("font-size: 10pt;")
        bottom_layout.addWidget(self.fov_info_label)
        bottom_layout.addStretch()

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

        # Update target list button state
        self._update_target_list_button()

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
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
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
            self.web_view.page().renderProcessTerminated.connect(self._on_render_process_terminated)

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
                self.web_placeholder.setStyleSheet(f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: 12px; }}")

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
                self.fallback_button.setStyleSheet(f"QPushButton {{ background-color: {COLORS['success']}; color: white; font-weight: bold; margin: 10px; padding: 8px; }}")
                self.fallback_button.clicked.connect(self._open_in_browser)

                # Insert before the bottom controls (last item should be the bottom layout)
                main_layout.insertWidget(main_layout.count() - 1, self.fallback_button)
                logger.debug("Added browser fallback button")
        except Exception as e:
            logger.error(f"Failed to add browser fallback button: {e}")

    def _open_in_browser(self):
        """Open Aladin Lite in the default browser"""
        try:
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
            open_url(browser_url)

            # Show a message to the user
            QMessageBox.information(self, "Opened in Browser",
                                  f"Aladin Lite has been opened in your default browser for {self.data.get('name', 'the selected object')}.")

        except Exception as e:
            logger.error(f"Failed to open Aladin Lite in browser: {e}")
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
        """Load active user telescopes from database"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length, is_active
                    FROM usertelescopes
                    WHERE focal_length IS NOT NULL AND focal_length > 0 AND is_active = 1
                    ORDER BY name ASC
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
                    self.telescope_combo.addItem(display_name, telescope_data)

                logger.debug(f"Loaded {len(telescopes)} active telescopes with focal length data")

        except Exception as e:
            logger.error(f"Error loading telescopes: {str(e)}")

    def _load_user_cameras_and_eyepieces(self):
        """Load user cameras and eyepieces from database to populate camera combo"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Load user cameras
                cursor.execute("""
                    SELECT id, name, sensor_width, sensor_height
                    FROM userequipment
                    WHERE equipment_type = 'camera'
                    ORDER BY name ASC
                """)
                user_cameras = cursor.fetchall()

                # Load user eyepieces
                cursor.execute("""
                    SELECT id, name, focal_length, apparent_fov
                    FROM userequipment
                    WHERE equipment_type = 'eyepiece'
                    ORDER BY name ASC
                """)
                user_eyepieces = cursor.fetchall()

                # Add user cameras section if there are any
                if user_cameras:
                    self.camera_combo.addItem("--- YOUR CAMERAS ---", None)
                    for eq_id, name, sensor_width, sensor_height in user_cameras:
                        if sensor_width and sensor_height:
                            display_text = f"{name} ({sensor_width}x{sensor_height}mm)"
                            self.camera_combo.addItem(display_text, {
                                "type": "camera",
                                "sensor_width": sensor_width,
                                "sensor_height": sensor_height,
                                "source": "user",
                                "id": eq_id
                            })

                # Add user eyepieces section if there are any
                if user_eyepieces:
                    self.camera_combo.addItem("--- YOUR EYEPIECES ---", None)
                    for eq_id, name, focal_length, apparent_fov in user_eyepieces:
                        if focal_length and apparent_fov:
                            display_text = f"{name} ({focal_length}mm, {apparent_fov}\u00b0 AFOV)"
                            self.camera_combo.addItem(display_text, {
                                "type": "eyepiece",
                                "focal_length": focal_length,
                                "apparent_fov": apparent_fov,
                                "source": "user",
                                "id": eq_id
                            })

                logger.debug(f"Loaded {len(user_cameras)} user cameras and {len(user_eyepieces)} user eyepieces")

        except Exception as e:
            logger.error(f"Error loading user cameras/eyepieces: {str(e)}")

    def _load_user_barlows(self):
        """Load user barlows and reducers from database to populate barlow combo"""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Load user barlows and reducers
                cursor.execute("""
                    SELECT id, name, factor, equipment_type
                    FROM userequipment
                    WHERE equipment_type IN ('barlow', 'reducer')
                    ORDER BY equipment_type, factor DESC
                """)
                user_barlows = cursor.fetchall()

                # Add user barlows/reducers section if there are any
                if user_barlows:
                    self.barlow_combo.addItem("--- YOUR EQUIPMENT ---", None)
                    for eq_id, name, factor, eq_type in user_barlows:
                        if factor:
                            display_text = f"{name} ({factor}x)"
                            self.barlow_combo.addItem(display_text, {
                                "type": eq_type,
                                "factor": factor,
                                "source": "user",
                                "id": eq_id
                            })

                logger.debug(f"Loaded {len(user_barlows)} user barlows/reducers")

        except Exception as e:
            logger.error(f"Error loading user barlows/reducers: {str(e)}")

    def _load_aladin_settings(self):
        """Load persistent Aladin Lite settings from QSettings"""
        try:
            settings = QSettings("CosmosCollection", "AladinLite")

            # Block signals during loading to prevent save being triggered
            self.telescope_combo.blockSignals(True)
            self.show_telescope_fov.blockSignals(True)
            self.show_smart_telescopes.blockSignals(True)
            self.camera_combo.blockSignals(True)
            self.barlow_combo.blockSignals(True)
            self.show_horizon.blockSignals(True)

            # Load show smart telescopes checkbox and populate combo if needed
            # (must happen before telescope name restore so smart entries are in the combo)
            show_smart = settings.value("show_smart_telescopes", False, type=bool)
            self.show_smart_telescopes.setChecked(show_smart)
            if show_smart:
                self._add_smart_telescopes()
            logger.debug(f"Restored show smart telescopes: {show_smart}")

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

            # Load virtual horizon checkbox
            show_horizon = settings.value("show_horizon", False, type=bool)
            self.show_horizon.setChecked(show_horizon)
            logger.debug(f"Restored show virtual horizon: {show_horizon}")

            # Unblock signals
            self.telescope_combo.blockSignals(False)
            self.show_telescope_fov.blockSignals(False)
            self.show_smart_telescopes.blockSignals(False)
            self.camera_combo.blockSignals(False)
            self.barlow_combo.blockSignals(False)
            self.show_horizon.blockSignals(False)

            logger.debug("Aladin Lite settings loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Aladin settings: {str(e)}")
            # Make sure to unblock signals even if there's an error
            self.telescope_combo.blockSignals(False)
            self.show_telescope_fov.blockSignals(False)
            self.show_smart_telescopes.blockSignals(False)
            self.camera_combo.blockSignals(False)
            self.barlow_combo.blockSignals(False)
            self.show_horizon.blockSignals(False)

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

            # Save show smart telescopes checkbox
            settings.setValue("show_smart_telescopes", self.show_smart_telescopes.isChecked())

            # Save camera/eyepiece selection
            settings.setValue("camera_eyepiece", self.camera_combo.currentText())

            # Save barlow/reducer selection
            settings.setValue("barlow_reducer", self.barlow_combo.currentText())

            # Save virtual horizon checkbox
            settings.setValue("show_horizon", self.show_horizon.isChecked())

            logger.debug("Aladin Lite settings saved successfully")

        except Exception as e:
            logger.error(f"Error saving Aladin settings: {str(e)}")

    def _add_smart_telescopes(self):
        """Append smart telescope entries to the telescope combo box."""
        self.telescope_combo.addItem("--- SMART TELESCOPES ---", None)
        for st in self.SMART_TELESCOPES:
            display = f"{st['name']} ({st['focal_length']}mm f/{st['focal_length']/st['aperture']:.1f})"
            self.telescope_combo.addItem(display, dict(st))

    def _remove_smart_telescopes(self):
        """Remove all smart telescope entries (and separator) from the telescope combo box."""
        i = 0
        while i < self.telescope_combo.count():
            text = self.telescope_combo.itemText(i)
            data = self.telescope_combo.itemData(i)
            if text == "--- SMART TELESCOPES ---" or (data and data.get('smart_telescope')):
                self.telescope_combo.removeItem(i)
            else:
                i += 1

    def _on_show_smart_telescopes_toggled(self, checked):
        """Handle Show Smart Telescopes checkbox toggle."""
        if checked:
            self._add_smart_telescopes()
        else:
            # If a smart telescope is currently selected, reset to Default View first
            current_data = self.telescope_combo.currentData()
            if current_data and current_data.get('smart_telescope'):
                self.telescope_combo.setCurrentIndex(0)
            self._remove_smart_telescopes()
        self._save_aladin_settings()

    def _on_telescope_changed(self):
        """Handle telescope selection change"""
        current_data = self.telescope_combo.currentData()
        if current_data:
            self.selected_telescope = current_data
            logger.debug(f"Selected telescope: {current_data['name']} ({current_data['focal_length']}mm)")

            if current_data.get('smart_telescope'):
                camera_text = current_data.get('camera_text')
                if camera_text:
                    idx = self.camera_combo.findText(camera_text)
                    if idx >= 0:
                        self.camera_combo.blockSignals(True)
                        self.camera_combo.setCurrentIndex(idx)
                        self.camera_combo.blockSignals(False)
                self.barlow_combo.blockSignals(True)
                self.barlow_combo.setCurrentIndex(0)  # "None (1.0x)"
                self.barlow_combo.blockSignals(False)
            else:
                # Auto-select equipment associated with this telescope
                self._select_telescope_equipment(current_data.get('id'))
        else:
            self.selected_telescope = None
            logger.debug("Selected default view")

        self._save_aladin_settings()
        self._update_aladin_view()

    def _select_telescope_equipment(self, telescope_id):
        """Auto-select equipment associated with the given telescope"""
        if not telescope_id:
            return

        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Get all equipment IDs associated with this telescope
                cursor.execute("""
                    SELECT e.id, e.equipment_type
                    FROM userequipment e
                    JOIN telescope_equipment te ON e.id = te.equipment_id
                    WHERE te.telescope_id = ?
                    ORDER BY e.name ASC
                """, (telescope_id,))

                equipment = cursor.fetchall()

                # Find first camera or eyepiece and select it
                camera_selected = False
                barlow_selected = False

                for eq_id, eq_type in equipment:
                    # Select first camera/eyepiece in the camera combo
                    if not camera_selected and eq_type in ('camera', 'eyepiece'):
                        for i in range(self.camera_combo.count()):
                            data = self.camera_combo.itemData(i)
                            if data and data.get('source') == 'user' and data.get('id') == eq_id:
                                self.camera_combo.blockSignals(True)
                                self.camera_combo.setCurrentIndex(i)
                                self.camera_combo.blockSignals(False)
                                camera_selected = True
                                logger.debug(f"Auto-selected camera/eyepiece ID {eq_id} for telescope")
                                break

                    # Select first barlow/reducer in the barlow combo
                    if not barlow_selected and eq_type in ('barlow', 'reducer'):
                        for i in range(self.barlow_combo.count()):
                            data = self.barlow_combo.itemData(i)
                            if data and data.get('source') == 'user' and data.get('id') == eq_id:
                                self.barlow_combo.blockSignals(True)
                                self.barlow_combo.setCurrentIndex(i)
                                self.barlow_combo.blockSignals(False)
                                barlow_selected = True
                                logger.debug(f"Auto-selected barlow/reducer ID {eq_id} for telescope")
                                break

        except Exception as e:
            logger.error(f"Error selecting telescope equipment: {str(e)}")

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
                self.web_placeholder.setStyleSheet(f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: 12px; }}")
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
                var currentAladinFov = {current_display_fov};  // Fallback (width FOV in degrees)
                var currentAladinFovH = currentAladinFov;      // Fallback height FOV

                // Try to get actual current FOV from Aladin instance
                // getFov() returns [widthDeg, heightDeg]
                try {{
                    var fovArr = null;
                    if (window.aladinInstance && window.aladinInstance.getFov) {{
                        fovArr = window.aladinInstance.getFov();
                    }} else if (typeof aladin !== 'undefined' && aladin.getFov) {{
                        fovArr = aladin.getFov();
                    }}
                    if (fovArr) {{
                        currentAladinFov  = fovArr[0];
                        currentAladinFovH = fovArr[1] || fovArr[0];
                    }}
                }} catch(e) {{
                    console.log('Could not get dynamic FOV, using fallback:', currentAladinFov);
                }}

                var data = window.telescopeFovData;

                // Width %  = telescope_fov_width  / aladin_width_fov  → % of container width
                // Height % = telescope_fov_height / aladin_height_fov → % of container height
                var overlayWidthPercent  = (data.telescopeFovDegrees       / currentAladinFov)  * 100;
                var overlayHeightPercent = (data.telescopeFovHeightDegrees / currentAladinFovH) * 100;

                // Limit the overlay size to reasonable bounds
                overlayWidthPercent = Math.max(2, Math.min(95, overlayWidthPercent));
                overlayHeightPercent = Math.max(2, Math.min(95, overlayHeightPercent));

                console.log('FOV Update: Aladin FOV=' + currentAladinFov.toFixed(3) + '°x' + currentAladinFovH.toFixed(3) + '°, Telescope FOV=' + data.telescopeFovDegrees.toFixed(3) + '°x' + data.telescopeFovHeightDegrees.toFixed(3) + '°, Overlay=' + overlayWidthPercent.toFixed(1) + '%x' + overlayHeightPercent.toFixed(1) + '%');

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
                    <div><strong>View:</strong> ${{currentAladinFov.toFixed(2)}}° × ${{currentAladinFovH.toFixed(2)}}° (${{(currentAladinFov*60).toFixed(0)}}')</div>
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

    def _load_observer_location(self):
        """Load observer lat/lon from DB. Returns True on success."""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings LIMIT 1")
                row = cursor.fetchone()
                if row and row[0] is not None and row[1] is not None:
                    self.observer_lat = float(row[0])
                    self.observer_lon = float(row[1])
                    return True
        except Exception as e:
            logger.error(f"Error loading observer location: {e}")
        return False

    def _calculate_horizon_data(self):
        """Convert horizon from alt-az to RA/Dec using astropy. Returns dict or None."""
        logger.debug(f"_calculate_horizon_data: lat={self.observer_lat}, lon={self.observer_lon}")
        try:
            location = EarthLocation(lat=self.observer_lat * u.deg, lon=self.observer_lon * u.deg, height=250 * u.m)
            now = Time.now()
            altaz_frame = AltAz(obstime=now, location=location)
            logger.debug(f"_calculate_horizon_data: altaz_frame created, time={now}")

            # Sample horizon: alt=0.5° every 5° in azimuth → 72 points
            horizon_points = []
            for az_deg in range(0, 360, 5):
                altaz_coord = SkyCoord(alt=0.5 * u.deg, az=az_deg * u.deg, frame=altaz_frame)
                radec = altaz_coord.icrs
                horizon_points.append([float(radec.ra.deg), float(radec.dec.deg)])

            # Cardinal and intercardinal directions at alt=3° for label visibility
            cardinal_azimuths = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
            cardinals = {}
            for label, az_deg in cardinal_azimuths.items():
                altaz_coord = SkyCoord(alt=3.0 * u.deg, az=az_deg * u.deg, frame=altaz_frame)
                radec = altaz_coord.icrs
                cardinals[label] = [float(radec.ra.deg), float(radec.dec.deg)]

            # Target alt/az
            ra = self.data.get('ra_deg', 0)
            dec = self.data.get('dec_deg', 0)
            target_altaz = None
            target_alt = None
            target_az = None
            try:
                target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
                target_altaz = target_coord.transform_to(altaz_frame)
                target_alt = float(target_altaz.alt.deg)
                target_az = float(target_altaz.az.deg)
            except Exception:
                pass

            # Zenith RA/Dec for globe view centering
            zenith_coord = SkyCoord(alt=90.0 * u.deg, az=0.0 * u.deg, frame=altaz_frame)
            zenith_radec = zenith_coord.icrs
            zenith_ra = float(zenith_radec.ra.deg)
            zenith_dec = float(zenith_radec.dec.deg)

            logger.debug(f"_calculate_horizon_data: {len(horizon_points)} horizon points, target alt={target_alt}, az={target_az}")
            return {
                'horizon_points': horizon_points,
                'cardinals': cardinals,
                'target_alt': target_alt,
                'target_az': target_az,
                'zenith_ra': zenith_ra,
                'zenith_dec': zenith_dec,
            }
        except Exception as e:
            logger.error(f"Error calculating horizon data: {e}", exc_info=True)
            return None

    def _inject_horizon_overlay(self):
        """Inject a virtual horizon overlay into the Aladin Lite view."""
        logger.debug("_inject_horizon_overlay called")
        if not self.web_view:
            logger.warning("_inject_horizon_overlay: web_view is None, skipping")
            return

        # Inject a placeholder panel immediately so we know JS runs
        self.web_view.page().runJavaScript("""
            (function() {
                var e = document.getElementById('horizon-info-panel');
                if (e) e.remove();
                var p = document.createElement('div');
                p.id = 'horizon-info-panel';
                p.style.cssText = 'position:fixed;bottom:10px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.85);color:white;padding:10px;border-radius:8px;font-size:12px;z-index:9999;border:2px solid #ffaa00;min-width:160px;';
                p.innerHTML = '<div style="color:#ffaa00;font-weight:bold;text-align:center;">VIRTUAL HORIZON</div><div>Calculating...</div>';
                document.body.appendChild(p);
                console.log('Horizon: placeholder panel injected');
            })();
        """)

        horizon_data = self._calculate_horizon_data()
        if horizon_data is None:
            logger.error("Failed to calculate horizon data")
            self.web_view.page().runJavaScript("""
                var p = document.getElementById('horizon-info-panel');
                if (p) p.innerHTML = '<div style="color:#ffaa00;font-weight:bold;text-align:center;">VIRTUAL HORIZON</div><div style="color:red;">Calculation failed</div>';
            """)
            return

        import json
        points_json = json.dumps(horizon_data['horizon_points'])
        cardinals_json = json.dumps(horizon_data['cardinals'])
        target_alt = horizon_data['target_alt']
        target_az = horizon_data['target_az']
        zenith_ra = horizon_data['zenith_ra']
        zenith_dec = horizon_data['zenith_dec']

        if target_alt is not None and target_az is not None:
            alt_str = f"{target_alt:.1f}"
            az_str = f"{target_az:.1f}"
        else:
            alt_str = "N/A"
            az_str = "N/A"

        lat_str = f"{self.observer_lat:.4f}"
        lon_str = f"{self.observer_lon:.4f}"

        target_alt_js = target_alt if target_alt is not None else 'null'
        target_az_js = target_az if target_az is not None else 'null'
        alt_text = f"{target_alt:.1f}" if target_alt is not None else "N/A"
        az_text = f"{target_az:.1f}" if target_az is not None else "N/A"

        # Step 1: always inject the info panel immediately — no Aladin dependency
        panel_js = f"""
        (function() {{
            var existing = document.getElementById('horizon-info-panel');
            if (existing) existing.remove();
            var panel = document.createElement('div');
            panel.id = 'horizon-info-panel';
            panel.style.cssText = 'position:fixed;bottom:10px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.85);' +
                'color:white;padding:10px;border-radius:8px;font-family:"Segoe UI",Arial,sans-serif;' +
                'font-size:12px;z-index:9999;border:2px solid #ffaa00;min-width:160px;';
            panel.innerHTML = '<div style="text-align:center;margin-bottom:6px;font-weight:bold;color:#ffaa00;">VIRTUAL HORIZON</div>' +
                '<div><strong>Alt:</strong> {alt_text}\u00b0</div>' +
                '<div><strong>Az:</strong> {az_text}\u00b0</div>' +
                '<div style="font-size:10px;color:#ccc;margin-top:4px;">Lat: {self.observer_lat:.4f}\u00b0  Lon: {self.observer_lon:.4f}\u00b0</div>';
            document.body.appendChild(panel);
        }})();
        """
        self.web_view.page().runJavaScript(panel_js)

        # Inject edge indicator as a separate call — independent of js_code so it always shows
        arrow_char = '\u25bc' if (target_alt is None or target_alt >= 0) else '\u25b2'
        ind_pos = 'bottom:38px' if (target_alt is None or target_alt >= 0) else 'top:10px'
        indicator_js = f"""
        (function() {{
            var ind = document.getElementById('horizon-edge-indicator');
            if (ind) ind.remove();
            ind = document.createElement('div');
            ind.id = 'horizon-edge-indicator';
            ind.style.cssText = 'position:fixed;{ind_pos};left:0;right:0;text-align:center;' +
                'color:#ffaa00;font-weight:bold;font-size:13px;z-index:9999;pointer-events:none;' +
                'font-family:"Segoe UI",Arial,sans-serif;text-shadow:0 0 4px #000,0 0 4px #000;';
            ind.textContent = '{arrow_char} Horizon (zoom out to see)';
            document.body.appendChild(ind);
        }})();
        """
        self.web_view.page().runJavaScript(indicator_js)

        # Diagnostic: check page context and iframe structure
        def _on_horizon_diag(result):
            logger.debug(f"Horizon JS diagnostic: {result}")

        self.web_view.page().runJavaScript("""
            (function() {
                var iframes = document.querySelectorAll('iframe');
                var iframeInfo = [];
                for (var i = 0; i < iframes.length; i++) {
                    var f = iframes[i];
                    var info = {src: f.src, id: f.id};
                    try {
                        var iw = f.contentWindow;
                        info.aladinInFrame = iw && typeof iw.aladin !== 'undefined' && !!iw.aladin;
                        info.AInFrame = iw && typeof iw.A !== 'undefined';
                        info.divInFrame = iw && !!iw.document.querySelector('#aladin-lite-div');
                    } catch(e) { info.crossOriginErr = e.message; }
                    iframeInfo.push(info);
                }
                return JSON.stringify({
                    href: window.location.href,
                    aladinDefined: typeof aladin !== 'undefined' && !!aladin,
                    ADefined: typeof A !== 'undefined',
                    aladinDivExists: !!document.querySelector('#aladin-lite-div'),
                    iframeCount: iframes.length,
                    iframes: iframeInfo
                });
            })();
        """, _on_horizon_diag)

        # Step 2: canvas-based horizon drawing — uses world2pix, redraws on pan/zoom events
        js_code = f"""
        (function() {{
        try {{
            // Clean up any previous horizon canvas and timers
            var existingCanvas = document.getElementById('horizon-overlay-canvas');
            if (existingCanvas) existingCanvas.remove();
            document.querySelectorAll('.horizon-cardinal-label').forEach(function(el) {{ el.remove(); }});
            if (window.horizonRefreshInterval) {{
                clearInterval(window.horizonRefreshInterval);
                window.horizonRefreshInterval = null;
            }}

            var horizonPoints = {points_json};
            var zenithRa  = {zenith_ra};
            var zenithDec = {zenith_dec};
            var cardinals = {cardinals_json};
            var targetAlt = {target_alt_js};

            var findAladinInstance = function() {{
                // Direct (HTML-content mode or URL page with global aladin)
                if (typeof aladin !== 'undefined' && aladin) return aladin;
                if (typeof window.aladin !== 'undefined' && window.aladin) return window.aladin;
                if (window.aladinInstance) return window.aladinInstance;
                var d = document.querySelector('#aladin-lite-div');
                if (d) {{
                    if (d._aladin) return d._aladin;
                    if (d.aladin)  return d.aladin;
                }}
                if (typeof A !== 'undefined' && A.aladinInstances && A.aladinInstances.length > 0)
                    return A.aladinInstances[0];
                // Search same-origin iframes (Aladin URL page embeds viewer in an iframe)
                var iframes = document.querySelectorAll('iframe');
                for (var i = 0; i < iframes.length; i++) {{
                    try {{
                        var iw = iframes[i].contentWindow;
                        if (!iw) continue;
                        if (iw.aladin) return iw.aladin;
                        var id = iw.document.querySelector('#aladin-lite-div');
                        if (id && id._aladin) return id._aladin;
                        if (iw.A && iw.A.aladinInstances && iw.A.aladinInstances.length > 0)
                            return iw.A.aladinInstances[0];
                    }} catch(e) {{ /* cross-origin, skip */ }}
                }}
                return null;
            }};

            // Returns the bounding rect to use for canvas placement.
            // When Aladin lives in an iframe, use the iframe element's rect.
            var getAladinRect = function() {{
                var d = document.querySelector('#aladin-lite-div');
                if (d) return d.getBoundingClientRect();
                var iframes = document.querySelectorAll('iframe');
                for (var i = 0; i < iframes.length; i++) {{
                    try {{
                        var iw = iframes[i].contentWindow;
                        if (iw && (iw.aladin || (iw.A && iw.A.aladinInstances && iw.A.aladinInstances.length))) {{
                            return iframes[i].getBoundingClientRect();
                        }}
                    }} catch(e) {{}}
                }}
                return document.body.getBoundingClientRect();
            }};

            var drawHorizon = function(al) {{
                var rect = getAladinRect();

                // Create or reuse overlay canvas — always, regardless of world2pix
                var cv = document.getElementById('horizon-overlay-canvas');
                if (!cv) {{
                    cv = document.createElement('canvas');
                    cv.id = 'horizon-overlay-canvas';
                    cv.style.cssText = 'position:fixed;pointer-events:none;z-index:9998;';
                    document.body.appendChild(cv);
                }}
                cv.width  = Math.round(rect.width)  || 800;
                cv.height = Math.round(rect.height) || 600;
                cv.style.left = rect.left + 'px';
                cv.style.top  = rect.top  + 'px';

                var ctx = cv.getContext('2d');
                ctx.clearRect(0, 0, cv.width, cv.height);
                ctx.strokeStyle = '#ffaa00';
                ctx.lineWidth = 2;
                ctx.lineJoin  = 'round';

                var inViewCount = 0;

                // Draw horizon line + cardinal labels only when world2pix is available
                if (al && typeof al.world2pix === 'function') {{
                    // Convert all horizon points to canvas pixels
                    var pxpts = horizonPoints.map(function(pt) {{
                        try {{ return al.world2pix(pt[0], pt[1]); }} catch(e) {{ return null; }}
                    }});

                    // Count in-view points
                    var validPts = 0;
                    var avgY = 0;
                    for (var vi = 0; vi < pxpts.length; vi++) {{
                        var vpx = pxpts[vi];
                        if (!vpx || isNaN(vpx[0]) || isNaN(vpx[1])) continue;
                        avgY += vpx[1];
                        validPts++;
                        if (vpx[0] >= 0 && vpx[0] <= cv.width && vpx[1] >= 0 && vpx[1] <= cv.height)
                            inViewCount++;
                    }}

                    if (inViewCount > 0) {{
                        // Draw connected segments, breaking at large screen-space jumps
                        ctx.beginPath();
                        var penDown = false;
                        var prevPx = null;
                        for (var i = 0; i <= horizonPoints.length; i++) {{
                            var px = pxpts[i % horizonPoints.length];
                            if (!px || isNaN(px[0]) || isNaN(px[1])) {{ penDown = false; prevPx = null; continue; }}
                            if (penDown && prevPx &&
                                (Math.abs(px[0] - prevPx[0]) > cv.width / 2 ||
                                 Math.abs(px[1] - prevPx[1]) > cv.height / 2)) {{
                                penDown = false;
                            }}
                            if (!penDown) {{ ctx.moveTo(px[0], px[1]); penDown = true; }}
                            else          {{ ctx.lineTo(px[0], px[1]); }}
                            prevPx = px;
                        }}
                        ctx.stroke();
                    }}

                    // Cardinal labels — remove stale ones then place updated ones
                    document.querySelectorAll('.horizon-cardinal-label').forEach(function(el) {{ el.remove(); }});
                    Object.keys(cardinals).forEach(function(label) {{
                        try {{
                            var px = al.world2pix(cardinals[label][0], cardinals[label][1]);
                            if (px && !isNaN(px[0]) && !isNaN(px[1]) &&
                                px[0] >= 0 && px[0] <= cv.width && px[1] >= 0 && px[1] <= cv.height) {{
                                var div = document.createElement('div');
                                div.className = 'horizon-cardinal-label';
                                div.textContent = label;
                                div.style.cssText = 'position:fixed;color:#ffaa00;font-weight:bold;font-size:13px;' +
                                    'font-family:"Segoe UI",Arial,sans-serif;text-shadow:0 0 4px #000,0 0 4px #000;' +
                                    'pointer-events:none;z-index:9999;' +
                                    'left:' + (rect.left + px[0] - 8) + 'px;' +
                                    'top:'  + (rect.top  + px[1] - 10) + 'px;';
                                document.body.appendChild(div);
                            }}
                        }} catch(e) {{}}
                    }});
                }}

                // Show/hide edge indicator div based on whether horizon is in view
                var ind = document.getElementById('horizon-edge-indicator');
                if (inViewCount === 0) {{
                    if (!ind) {{
                        ind = document.createElement('div');
                        ind.id = 'horizon-edge-indicator';
                        document.body.appendChild(ind);
                    }}
                    var arrow = (targetAlt === null || targetAlt >= 0) ? '\u25bc' : '\u25b2';
                    var pos   = (targetAlt === null || targetAlt >= 0) ? 'bottom:38px' : 'top:10px';
                    ind.style.cssText = 'position:fixed;' + pos + ';left:0;right:0;text-align:center;' +
                        'color:#ffaa00;font-weight:bold;font-size:13px;z-index:9999;pointer-events:none;' +
                        'font-family:"Segoe UI",Arial,sans-serif;text-shadow:0 0 4px #000,0 0 4px #000;';
                    ind.textContent = arrow + ' Horizon (zoom out to see)';
                }} else if (ind) {{
                    ind.remove();
                }}
            }};

            var startHorizonWhenReady = function(attemptCount) {{
                attemptCount = attemptCount || 0;
                var al = findAladinInstance();
                if (!al) {{
                    if (attemptCount < 30) {{
                        setTimeout(function() {{ startHorizonWhenReady(attemptCount + 1); }}, 500);
                    }} else {{
                        var p = document.getElementById('horizon-info-panel');
                        if (p) p.innerHTML += '<div style="color:orange;font-size:10px;margin-top:4px;">Aladin instance not found</div>';
                    }}
                    return;
                }}
                window.aladinInstance = al;

                // Switch to SIN (orthographic/globe) projection, keep zoom, re-center on target
                var _rd = null;
                try {{ _rd = al.getRaDec ? al.getRaDec() : null; }} catch(e) {{}}
                window.__horizonSavedRa  = _rd ? _rd[0] : null;
                window.__horizonSavedDec = _rd ? _rd[1] : null;
                try {{ al.setProjection('SIN'); }} catch(e) {{}}
                try {{ al.gotoRaDec({self.data.get('ra_deg', 0)}, {self.data.get('dec_deg', 0)}); }} catch(e) {{}}

                drawHorizon(al);

                // Refresh on pan/zoom events; backup timer every 2 s
                try {{
                    al.on('positionChanged', function() {{ drawHorizon(al); }});
                    al.on('zoomChanged',     function() {{ drawHorizon(al); }});
                }} catch(e) {{ /* event API not available, timer-only fallback */ }}
                window.horizonRefreshInterval = setInterval(function() {{ drawHorizon(al); }}, 2000);
            }};

            setTimeout(function() {{ startHorizonWhenReady(0); }}, 1000);
        }} catch(e) {{
            window.__horizonJSError = (e.stack || e.message || String(e));
            var p = document.getElementById('horizon-info-panel');
            if (p) p.innerHTML += '<div style="color:red;font-size:10px;margin-top:4px;word-break:break-all;">JS Error: ' + (e.message || e) + '</div>';
        }}
        }})();
        """
        self.web_view.page().runJavaScript(js_code)
        logger.debug("Horizon overlay injection initiated")

        def _read_js_error(result):
            if result:
                logger.error(f"Horizon JS runtime error: {result}")
        QTimer.singleShot(2000, lambda: self.web_view.page().runJavaScript(
            "window.__horizonJSError || null", _read_js_error))

    def _remove_horizon_overlay(self):
        """Remove the virtual horizon overlay from the Aladin Lite view."""
        if not self.web_view:
            return
        js_code = """
        (function() {
            var panel = document.getElementById('horizon-info-panel');
            if (panel) panel.remove();
            var canvas = document.getElementById('horizon-overlay-canvas');
            if (canvas) canvas.remove();
            var ind = document.getElementById('horizon-edge-indicator');
            if (ind) ind.remove();
            document.querySelectorAll('.horizon-cardinal-label').forEach(function(el) { el.remove(); });
            if (window.horizonRefreshInterval) {
                clearInterval(window.horizonRefreshInterval);
                window.horizonRefreshInterval = null;
            }
            // Restore TAN projection and re-center on original target
            var al = window.aladinInstance;
            if (al) {
                try { al.setProjection('TAN'); } catch(e) {}
                if (window.__horizonSavedRa !== null && window.__horizonSavedDec !== null) {
                    try { al.gotoRaDec(window.__horizonSavedRa, window.__horizonSavedDec); } catch(e) {}
                }
                setTimeout(function() {
                    if (window.updateTelescopeFovOverlay) window.updateTelescopeFovOverlay();
                }, 300);
            }
            window.__horizonSavedRa  = null;
            window.__horizonSavedDec = null;
        })();
        """
        self.web_view.page().runJavaScript(js_code)
        logger.debug("Horizon overlay removed")

    def _on_show_horizon_toggled(self):
        """Handle Virtual Horizon checkbox toggle."""
        self._save_aladin_settings()
        if self.show_horizon.isChecked():
            if not self._load_observer_location():
                QMessageBox.warning(self, "No Location Set",
                    "Please set your observing location in Settings to use the Virtual Horizon.")
                self.show_horizon.blockSignals(True)
                self.show_horizon.setChecked(False)
                self.show_horizon.blockSignals(False)
                return
            self._inject_horizon_overlay()
        else:
            self._remove_horizon_overlay()

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

    def _on_render_process_terminated(self, termination_status, exit_code):
        """Handle GPU/renderer process crash without crashing the whole app"""
        from PySide6.QtWebEngineCore import QWebEnginePage
        status_names = {
            QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus: "Normal",
            QWebEnginePage.RenderProcessTerminationStatus.AbnormalTerminationStatus: "Abnormal",
            QWebEnginePage.RenderProcessTerminationStatus.CrashedTerminationStatus: "Crashed",
            QWebEnginePage.RenderProcessTerminationStatus.KilledTerminationStatus: "Killed",
        }
        status_name = status_names.get(termination_status, f"Unknown({termination_status})")
        logger.error(f"Aladin Lite renderer process terminated: {status_name}, exit code: {exit_code}")
        self.loading_timeout.stop()

        if self.web_placeholder is None:
            # Placeholder was already removed; re-create it so we can show the error
            central_widget = self.centralWidget()
            if central_widget and central_widget.layout():
                self.web_placeholder = QLabel("Renderer crashed")
                central_widget.layout().insertWidget(1, self.web_placeholder)

        if self.web_placeholder:
            self.web_placeholder.setText(
                "Aladin Lite renderer crashed (GPU issue)\n\n"
                "This is usually caused by a GPU driver incompatibility.\n"
                "Click below to open in your browser instead."
            )
            self.web_placeholder.setStyleSheet(
                f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: 12px; }}"
            )
            self.web_placeholder.show()
        self._add_browser_fallback_button()

    def _on_load_finished(self, success):
        """Handle web page load finished"""
        logger.debug(f"Aladin Lite: Load finished, success={success}")
        self.loading_timeout.stop()

        if not success:
            logger.error("Failed to load Aladin Lite")
            if self.web_placeholder:
                self.web_placeholder.setText("Failed to load Aladin Lite\nCheck your internet connection\n\nClick below to open in browser instead.")
                self.web_placeholder.setStyleSheet(f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['error']}; font-size: 12px; }}")
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

            # Re-inject horizon overlay if enabled — delay to let Aladin's A.init.then() resolve
            if self.show_horizon.isChecked() and (self.observer_lat is not None or self._load_observer_location()):
                QTimer.singleShot(3000, self._inject_horizon_overlay)

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
            self.web_placeholder.setStyleSheet(f"QLabel {{ background-color: {COLORS['background']}; color: {COLORS['warning']}; font-size: 12px; }}")
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

    # ------------------------------------------------------------------
    # Target List helpers
    # ------------------------------------------------------------------

    def _check_if_in_target_list(self):
        """Check if the current DSO is already in the target list."""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')
                if not dso_name:
                    return False
                cursor.execute(
                    "SELECT COUNT(*) FROM usertargetlist WHERE UPPER(TRIM(name)) = ?",
                    (dso_name.upper(),)
                )
                if cursor.fetchone()[0] > 0:
                    return True
                if ra_deg is not None and dec_deg is not None:
                    cursor.execute(
                        "SELECT COUNT(*) FROM usertargetlist WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001",
                        (ra_deg, dec_deg)
                    )
                    if cursor.fetchone()[0] > 0:
                        return True
                return False
        except Exception as e:
            logger.error(f"Error checking target list status: {e}")
            return False

    def _find_target_list_name(self):
        """Return the stored target list name for the current DSO, or None."""
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                dso_name = self.data.get('name', '').strip()
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')
                if not dso_name:
                    return None
                cursor.execute(
                    "SELECT name FROM usertargetlist WHERE UPPER(TRIM(name)) = ? LIMIT 1",
                    (dso_name.upper(),)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                if ra_deg is not None and dec_deg is not None:
                    cursor.execute(
                        "SELECT name FROM usertargetlist WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001 LIMIT 1",
                        (ra_deg, dec_deg)
                    )
                    result = cursor.fetchone()
                    if result:
                        return result[0]
                return None
        except Exception as e:
            logger.error(f"Error finding target list name: {e}")
            return None

    def _update_target_list_button(self):
        """Show/hide target list actions based on current list membership."""
        try:
            in_list = self._check_if_in_target_list()
            self.add_target_action.setVisible(not in_list)
            self.remove_target_action.setVisible(in_list)
            self.open_target_action.setVisible(in_list)
        except Exception as e:
            logger.error(f"Error updating target list button: {e}")
            self.add_target_action.setVisible(True)
            self.remove_target_action.setVisible(False)
            self.open_target_action.setVisible(False)

    def _add_to_target_list(self):
        """Open the Add Target dialog for the current DSO."""
        try:
            from DSOTargetList import AddTargetDialog
            dialog = AddTargetDialog(dso_data=self.data.copy(), parent=self)
            if dialog.exec():
                self._update_target_list_button()
        except ImportError as e:
            logger.error(f"Could not import DSOTargetList: {e}")
            QMessageBox.warning(self, "Import Error", f"Could not load Target List feature: {e}")
        except Exception as e:
            logger.error(f"Error adding to target list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {e}")

    def _remove_from_target_list(self):
        """Remove the current DSO from the target list after confirmation."""
        try:
            dso_name = self.data.get('name', 'this DSO')
            reply = QMessageBox.question(
                self, "Remove from Target List",
                f"Remove '{dso_name}' from the target list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM usertargetlist WHERE UPPER(TRIM(name)) = ?",
                    (dso_name.upper(),)
                )
                ra_deg = self.data.get('ra_deg')
                dec_deg = self.data.get('dec_deg')
                if ra_deg is not None and dec_deg is not None:
                    cursor.execute(
                        "DELETE FROM usertargetlist WHERE ABS(ra_deg - ?) < 0.001 AND ABS(dec_deg - ?) < 0.001",
                        (ra_deg, dec_deg)
                    )
                conn.commit()
            self._update_target_list_button()
            QMessageBox.information(self, "Success", f"'{dso_name}' removed from target list.")
        except Exception as e:
            logger.error(f"Error removing from target list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove from target list: {e}")

    def _open_from_target_list(self):
        """Open the Target List window and highlight the current DSO."""
        try:
            target_name = self._find_target_list_name()
            if not target_name:
                QMessageBox.warning(self, "Not Found",
                    f"Could not find '{self.data.get('name', '')}' in your target list.")
                return
            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, '_target_list_window') or not self._target_list_window.isVisible():
                self._target_list_window = DSOTargetListWindow()
            self._target_list_window.show()
            self._target_list_window.raise_()
            success = self._target_list_window.open_and_select_target(target_name)
            if not success:
                QMessageBox.warning(self, "Not Found",
                    f"Could not navigate to '{target_name}' in the target list.")
        except Exception as e:
            logger.error(f"Error opening from target list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open target list: {e}")

    def _send_to_nina(self):
        """Send DSO coordinates to NINA Framing Assistant"""
        NINAIntegration.send_to_framing_assistant(
            self.data.get('ra_deg'), self.data.get('dec_deg'),
            self.data.get('name', 'Unknown'), self
        )

    def _slew_to_nina_target(self):
        """Slew mount to the current DSO coordinates via NINA"""
        NINAIntegration.slew_to_coordinates(
            self.data.get('ra_deg'), self.data.get('dec_deg'),
            self.data.get('name', 'Unknown'), self
        )
