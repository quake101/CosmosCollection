"""The main window: DSO table, search, toolbar and tool launchers."""

import logging
import os
from typing import Optional

from PySide6.QtCore import QEvent, QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSystemTrayIcon,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from cosmoscollection.app.SystemTrayManager import SystemTrayManager
from cosmoscollection.catalog.DSODetail import DSODetailWindow
from cosmoscollection.catalog.SimbadWorker import SimbadLoadingDialog, SimbadQueryWorker
from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.core.ResourceManager import ResourceManager
from cosmoscollection.imaging.CollageBuilder import open_collage_builder
from cosmoscollection.nina.NINAIntegration import NINAIntegration
from cosmoscollection.planning.FOVSimulator import AladinLiteWindow
from cosmoscollection.ui.WindowLifetime import keep_when_closed, window_is_open
from cosmoscollection.ui.WindowPositionManager import WindowPositionMixin

from cosmoscollection.app.AboutDialog import AboutDialog
from cosmoscollection.app.BulkAddToTargetDialog import BulkAddToTargetDialog
from cosmoscollection.app.DSOData import DSO_QUERY, dso_row_to_dict, format_dec, format_ra, load_all_dso_data
from cosmoscollection.app.DSOTableModel import DSOTableModel
from cosmoscollection.app.SettingsDialog import SettingsDialog
from cosmoscollection.app.UpdateDialogs import offer_update_and_apply, run_update_check

logger = logging.getLogger(__name__)


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
    def __init__(self, dso_data, catalogs):
        super().__init__()
        logger.debug("Initializing MainWindow")

        # Set window title with version - the local version now, plus
        # "(Latest)" / "(Update available: ...)" once GitHub answers
        try:
            from cosmoscollection.version import version_manager
            self.setWindowTitle(f"Cosmos Collection - v{version_manager.get_local_version()}")
            run_update_check(self, self._show_version_in_title)
        except ImportError:
            self.setWindowTitle("Cosmos Collection")
        self.resize(1200, 800)
        self.setWindowFlags(Qt.Window)
        self.setup_window_position()
        self.setAcceptDrops(True)
        self.db_manager = DatabaseManager()
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
        # Debounce so the full filter pass runs once typing pauses, not per keystroke
        self._search_debounce_timer = QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.setInterval(200)
        self._search_debounce_timer.timeout.connect(lambda: self._on_search(self.search_input.text()))
        self.search_input.textChanged.connect(lambda _text: self._search_debounce_timer.start())
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
        self.model = DSOTableModel(dso_data, parent=self)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.doubleClicked.connect(self._on_double_click)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)

        # Set default sort by catalog (column 0) in ascending order
        self.table_view.sortByColumn(0, Qt.AscendingOrder)

        # Enable context menu
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self._show_context_menu)

        # The table's colors come from the global theme (Theme.py), so they
        # follow theme changes without a widget-level style sheet here.
        main_layout.addWidget(self.table_view)

        # Set the layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Update status
        self._update_status()

        # Check for updates on startup if enabled
        QTimer.singleShot(2000, self._check_updates_on_startup)

        # Clean up leftover staged-update directories from past failed or
        # interrupted updates, off the UI thread
        QTimer.singleShot(3000, self._cleanup_stale_updates)

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
        telescopes_action = QAction("Equipment", self)
        telescopes_action.setToolTip("Your telescopes, cameras, eyepieces and barlows/reducers")
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

        # Session Manager action
        session_manager_action = QAction("Session Manager", self)
        session_manager_action.setToolTip("Plan future observing sessions and log past ones")
        session_manager_action.triggered.connect(self._show_session_manager)
        toolbar.addAction(session_manager_action)

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

        # Import Wizard action
        import_action = QAction("Import Wizard", self)
        import_action.setToolTip("Find the sessions and finished images in your folders and add them to your collection")
        import_action.triggered.connect(self._show_import_wizard)
        toolbar.addAction(import_action)

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

        toolbar.addSeparator()

        # Quit action - closing the window only minimizes to the tray when that's enabled
        quit_action = QAction("Quit", self)
        quit_action.setToolTip("Quit Cosmos Collection (Ctrl+Q)")
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))  # QKeySequence.Quit has no binding on Windows
        quit_action.triggered.connect(self._quit_application)
        toolbar.addAction(quit_action)

    def _show_settings(self):
        """Show the settings dialog"""
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()
        # Update NINA Dashboard action visibility based on current settings
        self.nina_dashboard_action.setVisible(NINAIntegration.is_enabled())
        self._update_status()  # Shows whether offline mode is on

    def _reopen_after_restart(self):
        """After Restart App on a view broken by a graphics reset: open that tool again
        (see core/GpuReset.py)."""
        from cosmoscollection.core.GpuReset import take_reopen
        note = take_reopen()
        if not note:
            return
        try:
            if note.get("tool") == "fov" and isinstance(note.get("data"), dict):
                logger.debug(f"Reopening the FOV Simulator after a restart: {note['data'].get('name')}")
                self.aladin_window = AladinLiteWindow(note["data"], self, session_id=note.get("session_id"))
                self.aladin_window.show()
        except Exception as e:
            logger.error(f"Could not reopen {note.get('tool')} after the restart: {e}", exc_info=True)

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
        """Show the Equipment window"""
        from cosmoscollection.equipment.EquipmentManager import open_equipment_window
        open_equipment_window(self)

    def _show_about(self):
        """Show the about dialog"""
        about_dialog = AboutDialog(self)
        about_dialog.exec()
        
    def _show_dso_visibility(self):
        """Show the DSO Visibility Calculator window"""
        try:
            from cosmoscollection.planning.DSOVisibilityCalculator import DSOVisibilityApp
            if not window_is_open(getattr(self, 'dso_visibility_window', None)):
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
            from cosmoscollection.planning.WeatherForecast import WeatherForecastWindow
            if not window_is_open(getattr(self, 'weather_forecast_window', None)):
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
        from cosmoscollection.nina.NINAIntegration import NINAIntegration
        if not NINAIntegration.is_enabled():
            QMessageBox.information(
                self, "NINA Integration",
                "NINA integration is not enabled.\n\n"
                "Please enable it in Settings > NINA Integration."
            )
            return

        try:
            from cosmoscollection.nina.NINADashboard import NINADashboardWindow
            if not window_is_open(getattr(self, 'nina_dashboard_window', None)):
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
            from cosmoscollection.planning.BestDSOTonight import BestDSOTonightWindow
            # Reuse the existing window (even if it was closed/hidden) rather than
            # rebuilding it from scratch each time - reconstruction re-runs the
            # catalog load and twilight calculation, which is expensive.
            needs_new = not hasattr(self, 'best_dso_window') or self.best_dso_window is None
            if not needs_new:
                try:
                    self.best_dso_window.isVisible()  # touch to detect a deleted C++ object
                except RuntimeError:
                    needs_new = True
            if needs_new:
                self.best_dso_window = BestDSOTonightWindow()
                # This one window is shown again each time, so it isn't freed on close
                keep_when_closed(self.best_dso_window)
            else:
                # Location/timezone may have changed in settings since the window
                # was last shown - refresh the displayed label (cheap single query).
                # Calculation results are always re-read from the DB fresh regardless.
                self.best_dso_window.load_location_info()
                # Refresh the twilight-based time defaults if they're from a
                # previous day (also re-reads location/timezone from the DB).
                self.best_dso_window.refresh_twilight_if_stale()
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
            from cosmoscollection.planning.DSOTargetList import DSOTargetListWindow
            if not window_is_open(getattr(self, 'target_list_window', None)):
                self.target_list_window = DSOTargetListWindow()
            self.target_list_window.show()
            self.target_list_window.raise_()
            self.target_list_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load Target List: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Target List: {e}")

    def _show_session_manager(self):
        """Show the Session Manager window"""
        try:
            from cosmoscollection.sessions.SessionManager import SessionManagerWindow
            if not window_is_open(getattr(self, 'session_manager_window', None)):
                self.session_manager_window = SessionManagerWindow()
            self.session_manager_window.show()
            self.session_manager_window.raise_()
            self.session_manager_window.activateWindow()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load Session Manager: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Session Manager: {e}")

    def _show_import_wizard(self):
        """Show the Import Wizard window"""
        from cosmoscollection.sessions.ImportWizard import open_import_wizard
        self.import_wizard_window = open_import_wizard(self)

    def _show_collage_builder(self):
        """Show the Collage Builder window"""
        try:
            with DatabaseManager().get_connection() as conn:
                image_count = conn.execute("SELECT COUNT(*) FROM userimages").fetchone()[0]
            if not image_count:
                QMessageBox.information(self, "No Images",
                    "No user images found. Add some images to DSO objects first, then you can create collages with them.")
                return

            # The one Collage Builder window, shared with the Gallery and DSO Detail
            self.collage_builder_window = open_collage_builder(self)

        except Exception as e:
            logger.error(f"Error opening collage builder: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Could not open Collage Builder: {str(e)}")

    def _show_dso_gallery(self):
        """Show the DSO Image Gallery window"""
        try:
            from cosmoscollection.imaging.DSOGallery import DSOGalleryWindow
            if not window_is_open(getattr(self, 'gallery_window', None)):
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
            if not window_is_open(getattr(self, 'aladin_window', None)):
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

    def _on_highlight_no_images_changed(self, state):
        self.model.setHighlightNoImages(state != 0)

    def _clear_filters(self):
        """Clear all filters"""
        logger.debug("Clear filters button pressed")
        logger.debug(f"Current state: catalog={self.model.selected_catalog}, type={getattr(self.model, '_current_selected_type', None)}")
        logger.debug(f"Current data: dso_data={len(self.model.dso_data)}, filtered_data={len(self.model.filtered_data)}")

        # Block signals to prevent multiple filter triggers
        self._search_debounce_timer.stop()
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

        # Check if we should query SIMBAD for this object
        # Only trigger if:
        # 1. Search text is not empty and looks like an object designation
        # 2. No results found in local database
        # 3. No other filters are active (catalog/type filters should be "All")
        # 4. SIMBAD lookup on search failure is enabled in settings (and not offline)
        from cosmoscollection.core.OnlineLimits import is_offline
        if (text and
            QSettings("CosmosCollection", "CosmosCollection").value(
                "simbad_lookup_on_search_failure", True, type=bool) and
            not is_offline() and
            len(self.model.filtered_data) == 0 and
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

    def _get_selected_type(self):
        """Get the currently selected DSO type code"""
        current_data = self.type_combo.currentData()
        if current_data and self.type_combo.currentText() != "All Types":
            return current_data
        return None

    def _update_status(self):
        """Update the status label"""
        total = len(self.model.dso_data)
        filtered = len(self.model.filtered_data)

        if filtered == total:
            self.status_label.setText(f"Showing all {total} objects")
        else:
            self.status_label.setText(f"Showing {filtered} of {total} objects")
        from cosmoscollection.core.OnlineLimits import is_offline
        if is_offline():
            self.status_label.setText(self.status_label.text() + " · Offline mode")

    def _cleanup_stale_updates(self):
        """Kick off background cleanup of leftover updates\\staged-* dirs
        from past failed/interrupted updates. Runs on its own thread since
        it does filesystem work, and is unconditional (unlike the update
        check itself) since it's just housekeeping."""
        try:
            from cosmoscollection.core.AppUpdater import update_manager
            update_manager.cleanup_stale_staged_dirs_async()
        except Exception as e:
            logger.debug(f"Could not start staged-update cleanup: {e}")

    def _check_updates_on_startup(self):
        """Silently check for updates on startup if enabled in settings"""
        try:
            # Surface a prior failed/incomplete update once, regardless of
            # whether startup checking is enabled - this reports on an
            # action the user already took, not a new check.
            from cosmoscollection.core.AppUpdater import update_manager
            failure_message = update_manager.check_previous_update_failure()
            if failure_message:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Update Did Not Complete", failure_message)

            # Check if the setting is enabled
            settings = QSettings("CosmosCollection", "CosmosCollection")
            check_updates = settings.value("check_updates_on_startup", True, type=bool)

            if not check_updates:
                logger.debug("Update check on startup is disabled")
                return
            from cosmoscollection.core.OnlineLimits import is_offline
            if is_offline():
                logger.debug("Update check on startup skipped - offline mode")
                return

            logger.debug("Checking for updates on startup")
            self._run_update_check(self._on_startup_update_check)

        except Exception as e:
            # Silently fail - don't bother user on startup with error messages
            logger.error(f"Error checking for updates on startup: {str(e)}")

    def _run_update_check(self, on_result):
        """Check GitHub for a newer release in the background; on_result(version_info)."""
        run_update_check(self, on_result)

    def _show_version_in_title(self, version_info):
        from cosmoscollection.version import version_manager
        self.setWindowTitle(f"Cosmos Collection - {version_manager.get_version_display_string(version_info)}")

    def _automatic_update_due(self, version_info):
        """Whether an automatic check should mention this update - not while
        the user has snoozed it or skipped this version, and at most once a day."""
        if not (version_info.get('github_available') and version_info.get('update_available')):
            return False
        from cosmoscollection.core.AppUpdater import update_notice_due, record_update_notice
        version = version_info.get('github_version') or ""
        if not update_notice_due(version):
            logger.debug(f"Update {version} available, but reminders are snoozed or it was skipped")
            return False
        record_update_notice(version)
        return True

    def _on_startup_update_check(self, version_info):
        if self._automatic_update_due(version_info):
            logger.info(f"Update available: {version_info.get('github_version')}")
            offer_update_and_apply(self, version_info)
        else:
            logger.debug("No update to mention on startup")

    def _save_simbad_object_to_database(self, object_data):
        """Save SIMBAD object data to the database.

        Returns (dsodetailid, created) - created is False when the object was already there."""
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

                # Check if object already exists. Catalog objects are keyed by their
                # own ID scheme (M 31 is "NGC224"), so also look the designation up
                # in cataloguenr rather than relying on the generated ID alone.
                cursor.execute("SELECT id FROM dsodetail WHERE id = ?", (dso_id,))
                existing = cursor.fetchone()
                if not existing:
                    catalog, designation = catalog_entries[0]
                    cursor.execute("""
                        SELECT dsodetailid FROM cataloguenr
                        WHERE UPPER(catalogue) = UPPER(?) AND designation = ?
                        LIMIT 1
                    """, (catalog, designation))
                    existing = cursor.fetchone()

                if existing:
                    logger.info(f"Object {main_id_clean} already exists in database as {existing[0]}")

                    # Extract catalog and designation for searching
                    catalog, designation = catalog_entries[0]

                    # Show message and offer to search for it
                    reply = QMessageBox.question(
                        self,
                        "Object Already Exists",
                        f"'{main_id_clean}' already exists in the database.\n\n"
                        f"Would you like to search for it and display it?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )

                    if reply == QMessageBox.Yes:
                        # Search all catalogs: the SIMBAD catalog name may not be one
                        # of the catalog filter entries
                        self.catalog_combo.setCurrentIndex(0)
                        self.search_input.setText(f"{catalog} {designation}")

                    return existing[0], False

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
                return dso_id, True

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

            # Save to database (an existing object was already offered to the user there)
            _, created = self._save_simbad_object_to_database(object_data)

            if created:
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

        QMessageBox.warning(self, "SIMBAD Query Error", error_message)

        # Clean up worker
        if hasattr(self, 'simbad_worker'):
            self.simbad_worker.deleteLater()
            self.simbad_worker = None

    def _reload_data(self):
        """Reload all data from database (e.g. after an object was added)"""
        with self.db_manager.get_connection() as conn:
            dso_data, catalogs = load_all_dso_data(conn)

        # A new object can bring a new catalog; rebuild the filter list, keeping the selection
        if catalogs != self.all_catalogs:
            self.all_catalogs = catalogs
            current = self.catalog_combo.currentText()
            self.catalog_combo.blockSignals(True)
            self.catalog_combo.clear()
            self.catalog_combo.addItem("All Catalogs")
            self.catalog_combo.addItems(catalogs)
            self.catalog_combo.setCurrentText(current)
            self.catalog_combo.blockSignals(False)

        self.model.set_data(dso_data)
        self._update_status()

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
                cursor.execute(DSO_QUERY + " AND d.id = ?", (entry["dsodetailid"],))

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
        data = dso_row_to_dict(result)
        # Use the name from the table entry instead of reconstructing it
        data["name"] = entry["name"]
        # Coordinates formatted for display
        data["ra"] = format_ra(data["ra_deg"])
        data["dec"] = format_dec(data["dec_deg"])

        logger.debug(f"Data dictionary: {data}")

        logger.debug("Creating detail window")
        # Create the window in a new thread
        detail_window = DSODetailWindow(data)
        detail_window.show()
        logger.debug("Detail window shown")

    def _refresh_data(self):
        """Pick up image count changes after images are added"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT dsodetailid, COUNT(*) FROM userimages GROUP BY dsodetailid")
                counts = {str(dsodetailid): count for dsodetailid, count in cursor.fetchall()}

            self.model.refresh_image_counts(counts)
            self._update_status()

            logger.debug("Main window data refreshed")
        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}", exc_info=True)

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

        # Is it already on the Target List / in the Session Manager?
        target_entry, sessions = self._context_list_status(row)

        if sessions:
            sessions_menu = context_menu.addMenu(f"Sessions ({len(sessions)})")
            for session_id, session_name, date, status, _ in sessions:
                session_action = sessions_menu.addAction(f"{date or 'No date'} – {session_name} ({status})")
                session_action.triggered.connect(lambda checked=False, sid=session_id: self._context_open_session(sid))
            sessions_menu.addSeparator()
            plan_action = sessions_menu.addAction("Plan a New Session")
        else:
            plan_action = context_menu.addAction("Plan a Session")
        plan_action.triggered.connect(lambda: self._context_plan_session(row, target_entry))

        if target_entry:
            target_action = context_menu.addAction("Open in Target List")
            target_action.triggered.connect(lambda: self._context_open_in_target_list(target_entry))
        else:
            target_action = context_menu.addAction("Add to Target List")
            target_action.triggered.connect(lambda: self._context_add_to_target_list(row))

        # Show the menu at the clicked position
        context_menu.exec(self.table_view.mapToGlobal(position))

    def _context_list_status(self, row):
        """The row's Target List entry (or None) and its Session Manager sessions -
        matched by name, other designations or position, like everywhere else."""
        entry = self.model.filtered_data[row]
        name = entry.get('name', '')
        ra_deg, dec_deg = entry.get('ra_deg'), entry.get('dec_deg')
        try:
            from cosmoscollection.planning.DSOTargetList import find_existing_target
            from cosmoscollection.planning.FOVSimulator import sessions_for_object
            with self.db_manager.get_connection() as conn:
                return (find_existing_target(conn, name, ra_deg, dec_deg),
                        sessions_for_object(conn, name, ra_deg, dec_deg, entry.get('dsodetailid')))
        except Exception as e:
            logger.error(f"Error checking Target List / sessions for {name}: {e}", exc_info=True)
            return None, []

    def _context_open_in_target_list(self, target_entry):
        """Open the Target List with this object's entry selected"""
        try:
            from cosmoscollection.planning.DSOTargetList import DSOTargetListWindow
            if not window_is_open(getattr(self, 'target_list_window', None)):
                self.target_list_window = DSOTargetListWindow()
            if not self.target_list_window.open_and_select_target(target_entry['name']):
                QMessageBox.warning(self, "Not Found", f"Could not find '{target_entry['name']}' in your target list.")
        except Exception as e:
            logger.error(f"Error opening Target List: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Target List: {str(e)}")

    def _context_open_session(self, session_id):
        """Open the Session Manager with this session selected"""
        try:
            from cosmoscollection.sessions.SessionManager import SessionManagerWindow
            if not window_is_open(getattr(self, 'session_manager_window', None)):
                self.session_manager_window = SessionManagerWindow()
            self.session_manager_window.open_and_select_session(session_id)
        except Exception as e:
            logger.error(f"Error opening Session Manager: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Session Manager: {str(e)}")

    def _context_plan_session(self, row, target_entry=None):
        """Open Session Manager pre-filled with this DSO (linked to its Target List entry)"""
        try:
            entry = self.model.filtered_data[row]
            dso_data = {
                'id': target_entry['id'] if target_entry else None,
                'name': entry.get('name', ''),
                'dsodetailid': entry.get('dsodetailid'),
                'ra_deg': entry.get('ra_deg'),
                'dec_deg': entry.get('dec_deg'),
                'designations': entry.get('designations', '')
            }
            from cosmoscollection.sessions.SessionManager import SessionManagerWindow
            if not window_is_open(getattr(self, 'session_manager_window', None)):
                self.session_manager_window = SessionManagerWindow()
            self.session_manager_window.create_session_from_dso(dso_data)
        except Exception as e:
            logger.error(f"Error opening Session Manager: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Session Manager: {str(e)}")

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
            from cosmoscollection.planning.DSOVisibilityCalculator import DSOVisibilityApp

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
                'dsodetailid': entry.get('dsodetailid', '')
            }

            # Import and open Aladin Lite window
            # Store reference to prevent garbage collection and manage window lifecycle
            if not window_is_open(getattr(self, 'aladin_window', None)):
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
            from cosmoscollection.planning.DSOTargetList import DSOTargetListWindow
            if not window_is_open(getattr(self, 'target_list_window', None)):
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
            from cosmoscollection.planning.DSOTargetList import TargetListIndex
            with self.db_manager.get_connection() as conn:
                # Same rules as everywhere else: an object listed under another
                # spelling or designation ("M17" / "M 17", "IC 4725" / "M 25") counts
                on_list = TargetListIndex(conn)
                to_add = [e for e in missing if on_list.match(e.get('name', '')) is None]
        except Exception as e:
            logger.error(f"Error reading target list: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to read target list: {str(e)}")
            return

        if not to_add:
            QMessageBox.information(
                self, "Nothing to Add",
                "All objects without images are already in your target list."
            )
            return

        dialog = BulkAddToTargetDialog(to_add, parent=self)
        if dialog.exec() == QDialog.Accepted:
            if window_is_open(getattr(self, 'target_list_window', None)):
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

    def dragEnterEvent(self, event):
        """Accept drag if it is a single supported image file"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                from cosmoscollection.imaging.DSOGallery import SUPPORTED_IMAGE_EXTENSIONS
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Open the Add Image dialog with the dropped file pre-filled"""
        from cosmoscollection.imaging.DSOGallery import SUPPORTED_IMAGE_EXTENSIONS
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                event.acceptProposedAction()
                from cosmoscollection.imaging.DSOGallery import import_image
                # Deferred: a dialog shown inside dropEvent keeps the source app's drag (e.g. Explorer) blocked
                QTimer.singleShot(0, lambda: import_image(self, file_path))

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
                icon_path = str(ResourceManager.get_icon_path())
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
            from cosmoscollection.planning.WeatherForecast import WeatherCache
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
            from cosmoscollection.core.Diagnostics import log_memory
            log_memory("minimized to tray")

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
        from cosmoscollection.core.Diagnostics import log_memory
        log_memory("restored from tray")

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
        """Check for updates in the background; a tray notification follows if one is available"""
        from cosmoscollection.core.OnlineLimits import is_offline
        if is_offline():
            logger.debug("Tray update check skipped - offline mode")
            return
        try:
            self._run_update_check(self._on_tray_update_check)
        except Exception as e:
            logger.debug(f"Error checking for updates in tray: {e}")

    def _on_tray_update_check(self, version_info):
        """Show a tray notification for an available update, unless the user
        snoozed it, skipped this version, or was already told today"""
        try:
            if not (self._tray_manager and self._tray_manager._tray_icon):
                return
            if self._automatic_update_due(version_info):
                logger.info(f"Update available (tray check): {version_info.get('github_version')}")
                self._tray_manager._tray_icon.showMessage(
                    "Update Available",
                    f"Cosmos Collection {version_info['github_version']} is available.\n"
                    f"Click here to install it or choose when to be reminded.",
                    QSystemTrayIcon.Information,
                    10000  # Show for 10 seconds
                )
                # Re-wire the click handler on every check so it always
                # offers to update using this check's version info.
                try:
                    self._tray_manager._tray_icon.messageClicked.disconnect()
                except (TypeError, RuntimeError):
                    pass  # nothing was connected yet
                self._tray_manager._tray_icon.messageClicked.connect(
                    lambda vi=version_info: offer_update_and_apply(self, vi)
                )
            else:
                logger.debug("Tray update check: nothing to mention")

        except Exception as e:
            logger.debug(f"Error checking for updates in tray: {e}")

    def _fetch_weather_for_tray(self):
        """Fetch weather data in background for tray tooltip"""
        from cosmoscollection.core.OnlineLimits import is_offline
        if is_offline():
            logger.debug("Tray weather refresh skipped - offline mode")
            return
        try:
            from cosmoscollection.planning.WeatherForecast import WeatherCache, WeatherWorker

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
            from cosmoscollection.planning.WeatherForecast import WeatherCache

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
        elif action_name == "session_manager":
            self._show_session_manager()
        elif action_name == "weather":
            self._show_weather_forecast()
        elif action_name == "gallery":
            self._show_dso_gallery()
        elif action_name == "nina_dashboard":
            self._show_nina_dashboard()

    def _quit_application(self):
        """Quit the application (tray menu or toolbar Quit)"""
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
