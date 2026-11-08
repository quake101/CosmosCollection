#!/usr/bin/env python3
"""
Best DSO Tonight Calculator
Calculates and displays the best Deep Sky Objects visible from user's location tonight
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex
from PySide6.QtCore import QTime
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QGroupBox, QMessageBox,
                               QHeaderView, QProgressBar, QSpinBox, QComboBox, QMenu, QCheckBox, QApplication,
                               QTimeEdit)

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import pytz
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.planning.Observer import active_site, coordinates_text
from cosmoscollection.catalog.DSONames import type_name
from cosmoscollection.core.Settings import app_settings
from cosmoscollection.ui.WindowPositionManager import WindowPositionMixin
from cosmoscollection.ui.WindowLifetime import release_on_close, window_is_open
from cosmoscollection.ui.Theme import COLORS, adapt_color, font_px, theme_manager, themed_style, tint
from cosmoscollection.ui.TimeFormatHelper import format_time, format_datetime
from cosmoscollection.nina.NINAIntegration import NINAIntegration


from cosmoscollection.planning.BestTonightEngine import (  # noqa: E402,F401
    TIME_DEFAULT, TIME_SETTING, BestTonightCalculation, TonightError, _get_available_catalogs,
    describe_time_window, load_time_choice, resolve_time_window)


class NumericTableWidgetItem(QTableWidgetItem):
    """Custom QTableWidgetItem that sorts by numeric value stored in UserRole"""
    def __lt__(self, other):
        """Override less-than operator for proper numeric sorting"""
        self_value = self.data(Qt.UserRole)
        other_value = other.data(Qt.UserRole)

        # Handle None values
        if self_value is None:
            return True
        if other_value is None:
            return False

        # Compare numeric values
        try:
            return float(self_value) < float(other_value)
        except (ValueError, TypeError):
            # Fallback to string comparison if conversion fails
            return super().__lt__(other)


class DSOCalculationThread(QThread):
    """Runs a BestTonightCalculation (see BestTonightEngine) off the UI
    thread. Its arguments are the calculation's; .calculation has the
    resolved time window and blocked_by_horizon once it has run."""
    progress = Signal(int)
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calculation = BestTonightCalculation(*args, **kwargs)

    def run(self):
        try:
            self.result_ready.emit(self.calculation.calculate(progress=self.progress.emit))
        except TonightError as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(f"Calculation error: {str(e)}")


class TimeWindowThread(QThread):
    """Resolves the chosen time (e.g. tonight's dusk and dawn) off the GUI
    thread, for the note beside the Time dropdown - this astropy calculation can
    also trigger an IERS data fetch."""
    result_ready = Signal(str)  # note text, '' if it couldn't be resolved

    def __init__(self, time_choice):
        super().__init__()
        self.time_choice = time_choice

    def run(self):
        try:
            from cosmoscollection.core.DatabaseManager import open_connection
            # Own connection: the DatabaseManager singleton's is bound to the GUI thread
            conn = open_connection()
            try:
                from cosmoscollection.planning.Observer import load_observer
                observer = load_observer(conn)
            finally:
                conn.close()
            if not observer:
                self.result_ready.emit("")
                return
            lat, lon, tz_name = observer
            location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
            start, end, note, _times = resolve_time_window(self.time_choice, location, tz_name)
            text = describe_time_window(self.time_choice[0], start, end)
            self.result_ready.emit(f"({text}{' - ' + note if note else ''})")
        except Exception as e:
            logger.error(f"Error resolving the time window: {e}")
            self.result_ready.emit("")


class BestDSOTonightWindow(WindowPositionMixin, QMainWindow):
    """Main window for Best DSO Tonight calculator"""
    WINDOW_POSITION_KEY = "BestDSOTonight"

    def __init__(self, use_target_list=False):
        super().__init__()
        release_on_close(self)  # Freed once closed (see WindowLifetime)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("Best DSO Tonight - Cosmos Collection")
        self.resize(930, 920)
        self.auto_use_target_list = use_target_list
        self.setup_window_position()

        self.calc_thread = None
        self.twilight_thread = None
        self._twilight_calc_date = None  # date (YYYY-MM-DD) the twilight window was last computed
        self.available_catalogs = []
        self.visible_dsos_data = []  # Store DSO data for detail window
        self.init_ui()
        # The moon column's colors are item data, so recolor them on theme changes
        # (widget styles follow on their own)
        theme_manager().theme_changed.connect(self._recolor_moon_column)
        self.load_location_info()

        # Set time frame defaults based on twilight calculation
        # Use QTimer to defer this until after UI is shown
        QTimer.singleShot(0, self._complete_initialization)

    def _complete_initialization(self):
        """Complete initialization after UI is shown - kicks off the twilight
        calculation in a background thread so the window stays responsive
        (this astropy calculation can also trigger an IERS data fetch)."""
        self.status_label.setText("Loading astronomical ephemeris data...")
        themed_style(self.status_label, lambda: f"color: {COLORS['info']};")

        self._resolve_time_note()
        self._initialized = False

    def _resolve_time_note(self):
        """Show the chosen time's actual window (e.g. tonight's dusk -> dawn) beside the dropdown."""
        if self.twilight_thread and self.twilight_thread.isRunning():
            self._time_note_stale = True  # Resolve again when this one finishes
            return
        self._time_note_stale = False
        self.twilight_thread = TimeWindowThread(self._time_choice())
        self.twilight_thread.result_ready.connect(self._on_twilight_calculated)
        self.twilight_thread.start()

    def _on_twilight_calculated(self, note):
        """Show the resolved time window and, the first time, finish initialization."""
        self.time_note_label.setText(note)
        self._twilight_calc_date = datetime.now().strftime("%Y-%m-%d")
        if getattr(self, "_time_note_stale", False):
            QTimer.singleShot(0, self._resolve_time_note)
        if getattr(self, "_initialized", True):
            return
        self._initialized = True

        # Update status to ready
        self.status_label.setText("Ready to calculate best DSOs for tonight")
        self.status_label.setStyleSheet("")  # Reset to default color

        # Auto-select target list and calculate if requested
        if self.auto_use_target_list:
            self.use_target_list_checkbox.setChecked(True)
            # Use QTimer to trigger calculation after UI is fully initialized
            QTimer.singleShot(100, self.calculate_best_dsos)
        elif self._should_auto_calculate():
            # Auto Calculate is enabled, and either no results are showing yet
            # (always true for a freshly opened window) or the last
            # calculation wasn't done today
            QTimer.singleShot(100, self.calculate_best_dsos)

    def refresh_twilight_if_stale(self):
        """Recompute the twilight-based start hour/duration defaults if they
        were last computed on a previous calendar day. Called when a
        previously-built window instance is reopened, since twilight times
        (and possibly the observer's location/timezone) can change day to day.
        No-op if already refreshed today or a refresh is already in flight.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._twilight_calc_date == today_str:
            return
        self._resolve_time_note()

    def _needs_auto_calculate(self):
        """Return True if Auto Calculate should run because the last calculation
        wasn't performed today (or has never been performed)."""
        settings = app_settings()
        last_calculated_date = settings.value("BestDSOTonight/last_calculated_date", "", type=str)
        today_str = datetime.now().strftime("%Y-%m-%d")
        return last_calculated_date != today_str

    def _should_auto_calculate(self):
        """Return True if the Auto Calculate checkbox is checked and a
        calculation is warranted: either no results are currently displayed
        or the last calculation wasn't done today."""
        if not self.auto_calculate_checkbox.isChecked():
            return False
        if self.results_table.rowCount() == 0:
            return True
        return self._needs_auto_calculate()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Calculation Settings Group
        settings_group = QGroupBox(self.SETTINGS_TITLE)
        self.settings_group = settings_group  # Its title also shows the observer location
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(4)

        # First row - Basic settings
        settings_row1 = QHBoxLayout()

        # Minimum altitude
        settings_row1.addWidget(QLabel("Min Altitude:"))
        self.min_altitude_spin = QSpinBox()
        self.min_altitude_spin.setRange(10, 90)
        self.min_altitude_spin.setValue(30)
        self.min_altitude_spin.setSuffix("°")
        self.min_altitude_spin.setToolTip("Minimum altitude for DSO visibility (used with all modes)")
        settings_row1.addWidget(self.min_altitude_spin)

        # Maximum magnitude
        settings_row1.addWidget(QLabel("Max Magnitude:"))
        self.max_magnitude_combo = QComboBox()
        self.max_magnitude_combo.addItems(["8.0", "10.0", "12.0", "14.0", "16.0"])
        self.max_magnitude_combo.setCurrentText("12.0")
        self.max_magnitude_combo.setToolTip("Maximum magnitude for DSO selection (ignored when using Target List)")
        settings_row1.addWidget(self.max_magnitude_combo)

        # Catalog dropdown selection
        settings_row1.addWidget(QLabel("Catalog:"))
        self.catalog_combo = QComboBox()
        self.load_catalog_options()
        self.catalog_combo.addItem("All Catalogs")
        for catalog in sorted(self.available_catalogs):
            self.catalog_combo.addItem(catalog)
        self.catalog_combo.setCurrentText("All Catalogs")
        self.catalog_combo.setToolTip("Filter by catalog (ignored when using target list)")
        settings_row1.addWidget(self.catalog_combo)

        # DSO Type filter
        settings_row1.addWidget(QLabel("Type:"))
        self.dso_type_combo = QComboBox()
        self.dso_type_combo.setMinimumWidth(150)
        self.dso_type_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.load_dso_type_options()
        self.dso_type_combo.setToolTip("Filter by DSO type (ignored when using target list)")
        settings_row1.addWidget(self.dso_type_combo)

        settings_row1.addStretch()
        settings_layout.addLayout(settings_row1)

        # Second row - Time frame and DSO limit
        settings_row2 = QHBoxLayout()

        # DSO Limit
        settings_row2.addWidget(QLabel("DSO Limit:"))
        self.dso_limit_spin = QSpinBox()
        self.dso_limit_spin.setRange(50, 1000)
        self.dso_limit_spin.setValue(200)
        self.dso_limit_spin.setSingleStep(50)
        self.dso_limit_spin.setToolTip("Maximum number of DSOs to process (ignored when using target list)")
        settings_row2.addWidget(self.dso_limit_spin)

        # Time: when the objects should be up (the night, a moment, or a time frame)
        from cosmoscollection.planning.NightTimes import DIRECTION_MODES
        mode, at, start, end = load_time_choice()
        settings_row2.addWidget(QLabel("Time:"))
        self.time_mode_combo = QComboBox()
        for key, label in DIRECTION_MODES:
            self.time_mode_combo.addItem(label, key)
        self.time_mode_combo.setToolTip(
            "When the objects should be up:\n"
            "\u2022 Tonight: up (above Min Altitude, in dark sky) at some point between dusk and dawn\n"
            "\u2022 Now / Dusk / Midnight / Dawn / Custom time: up at that moment\n"
            "\u2022 Custom time frame: up for the whole frame, e.g. 01:00 to 02:00\n"
            "Dusk and dawn are astronomical twilight. Times are for the coming night\n"
            "(or, before dawn, the night you're in). Highest in the sky during the time ranks first.")
        settings_row2.addWidget(self.time_mode_combo)

        def time_edit(hhmm, tooltip):
            edit = QTimeEdit(QTime.fromString(hhmm, "HH:mm"))
            edit.setDisplayFormat("HH:mm")
            edit.setToolTip(tooltip)
            edit.timeChanged.connect(lambda _t: self._on_time_choice_changed())
            return edit

        self.time_at_edit = time_edit(at, "Time tonight")
        settings_row2.addWidget(self.time_at_edit)
        self.time_from_edit = time_edit(start, "Start of the time frame")
        settings_row2.addWidget(self.time_from_edit)
        self.time_to_label = QLabel("to")
        settings_row2.addWidget(self.time_to_label)
        self.time_to_edit = time_edit(end, "End of the time frame (after midnight is fine)")
        settings_row2.addWidget(self.time_to_edit)
        self.time_note_label = QLabel("")
        themed_style(self.time_note_label, lambda: f"color: {COLORS['text_secondary']};")
        settings_row2.addWidget(self.time_note_label)

        index = self.time_mode_combo.findData(mode)
        self.time_mode_combo.setCurrentIndex(max(index, 0))
        self.time_mode_combo.currentIndexChanged.connect(lambda _i: self._on_time_choice_changed())
        self._show_time_edits()

        settings_row2.addStretch()
        settings_layout.addLayout(settings_row2)

        # Third row - options, then the Calculate button at the right
        action_layout = QHBoxLayout()

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Best DSOs Tonight")
        self.calculate_btn.clicked.connect(self.calculate_best_dsos)

        # Use Target List checkbox
        self.use_target_list_checkbox = QCheckBox("Use My Target List")
        self.use_target_list_checkbox.setToolTip(
            "Calculate visibility only for DSOs in your target list.\n\n"
            "When checked:\n"
            "• Min Altitude setting is still used\n"
            "• Start Hour and Duration settings are still used\n"
            "• Catalog, Type, Max Magnitude, and DSO Limit settings are ignored"
        )
        self.use_target_list_checkbox.toggled.connect(self._on_target_list_checkbox_changed)

        # Don't Show Completed checkbox - only applies to the target list, so
        # it's enabled by _on_target_list_checkbox_changed() alongside it
        settings = app_settings()
        self.hide_completed_checkbox = QCheckBox("Don't Show Completed")
        self.hide_completed_checkbox.setToolTip(
            "Skip target list entries whose status is \"Completed\".\n\n"
            "Only applies when Use My Target List is checked."
        )
        self.hide_completed_checkbox.setChecked(
            settings.value("BestDSOTonight/hide_completed", False, type=bool))
        self.hide_completed_checkbox.setEnabled(False)

        # Restore saved checkbox state
        saved_use_target_list = settings.value("BestDSOTonight/use_target_list", False, type=bool)
        if saved_use_target_list:
            self.use_target_list_checkbox.setChecked(True)
        action_layout.addWidget(self.use_target_list_checkbox)
        action_layout.addWidget(self.hide_completed_checkbox)

        # Use Custom Horizon checkbox - enabled by _refresh_horizon_state()
        # only when the active location has a custom horizon
        self.active_horizon = None
        self.use_horizon_checkbox = QCheckBox("Use Custom Horizon")
        self.use_horizon_checkbox.setChecked(
            settings.value("BestDSOTonight/use_custom_horizon", True, type=bool))
        action_layout.addWidget(self.use_horizon_checkbox)

        # Auto Calculate checkbox
        self.auto_calculate_checkbox = QCheckBox("Auto Calculate")
        self.auto_calculate_checkbox.setToolTip(
            "Automatically calculate best DSOs using the settings above.\n\n"
            "Runs automatically when:\n"
            "• This window is opened and no results are shown yet\n"
            "• This window is opened and the last calculation wasn't done today\n"
            "• You check this box while no results are currently shown\n\n"
            "Uses whichever settings are currently set (including Use My\n"
            "Target List, if enabled)."
        )
        self.auto_calculate_checkbox.toggled.connect(self._on_auto_calculate_checkbox_changed)
        saved_auto_calculate = settings.value("BestDSOTonight/auto_calculate", False, type=bool)
        if saved_auto_calculate:
            self.auto_calculate_checkbox.setChecked(True)
        action_layout.addWidget(self.auto_calculate_checkbox)

        action_layout.addStretch()
        action_layout.addWidget(self.calculate_btn)
        settings_layout.addLayout(action_layout)

        main_layout.addWidget(settings_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Results table
        results_group = QGroupBox("Tonight's Best DSOs")
        self.results_group = results_group
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "DSO", "Type", "Constellation", "Magnitude", 
            "Max Alt.", "Best Time", "Direction", "Visible Hours"
        ])
        
        # Enable sorting and disable editing
        self.results_table.setSortingEnabled(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Set column widths - allow user resizing
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        header.setSectionResizeMode(6, QHeaderView.Interactive)
        header.setSectionResizeMode(7, QHeaderView.Interactive)
        self.results_table.setColumnWidth(3, 80)
        self.results_table.setColumnWidth(4, 70)
        self.results_table.setColumnWidth(5, 80)
        self.results_table.setColumnWidth(6, 150)  # "NE (below) → W"
        self.results_table.setColumnWidth(7, 110)
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Connect double-click handler
        self.results_table.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Enable context menu
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_context_menu)
        
        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_group)
        
        # Status label
        self.status_label = QLabel("Initializing astronomical calculations...")
        themed_style(self.status_label, lambda: f"color: {COLORS['info']};")
        main_layout.addWidget(self.status_label)

    def _time_choice(self):
        """(mode, at, from, to) currently chosen."""
        return (self.time_mode_combo.currentData(),
                self.time_at_edit.time().toString("HH:mm"),
                self.time_from_edit.time().toString("HH:mm"),
                self.time_to_edit.time().toString("HH:mm"))

    def _show_time_edits(self):
        mode = self.time_mode_combo.currentData()
        self.time_at_edit.setVisible(mode == "time")
        for widget in (self.time_from_edit, self.time_to_label, self.time_to_edit):
            widget.setVisible(mode == "frame")

    def _on_time_choice_changed(self):
        self._show_time_edits()
        settings = app_settings()
        settings.setValue(TIME_SETTING, "|".join(self._time_choice()))
        self._resolve_time_note()

    def _refresh_horizon_state(self):
        """Load the active location's custom horizon and enable the Use Custom
        Horizon checkbox only when there is one"""
        from cosmoscollection.planning.HorizonProfile import load_active_horizon
        self.active_horizon = load_active_horizon()
        if self.active_horizon is not None:
            self.use_horizon_checkbox.setEnabled(True)
            self.use_horizon_checkbox.setToolTip(
                "Treat objects as blocked while they're below the active location's\n"
                f"custom horizon ({self.active_horizon.summary()}).\n\n"
                "Min Altitude still applies: an object must clear both."
            )
        else:
            self.use_horizon_checkbox.setEnabled(False)
            self.use_horizon_checkbox.setToolTip(
                "The active location has no custom horizon.\n\n"
                "Import a NINA or Stellarium horizon file in\n"
                "Settings > Location Manager."
            )

    SETTINGS_TITLE = "Calculation Settings"

    def _show_location(self, text):
        """Observer location (or a location problem) in the settings box title."""
        self.settings_group.setTitle(f"{self.SETTINGS_TITLE}  —  {text}" if text else self.SETTINGS_TITLE)

    def load_location_info(self):
        """Load and display location information"""
        self._refresh_horizon_state()

        # Check if observer location should be shown
        settings = app_settings()
        show_location = settings.value("show_observer_location", True, type=bool)

        # The location is only shown (in the settings title) when the setting is on
        self._show_location(None)
        if not show_location:
            return

        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                site = active_site(conn)
                if site is not None:
                    location_text = f"{site.name} - {coordinates_text(site.lat, site.lon)}"
                    if site.timezone:
                        try:
                            location_text += f" ({datetime.now(pytz.timezone(site.timezone)).strftime('%Z')})"
                        except Exception:
                            pass
                    if self.active_horizon is not None:
                        location_text += f" · Horizon: {self.active_horizon.name or 'custom'}"
                    self._show_location(location_text)
                    self.calculate_btn.setEnabled(True)
                    return
                
                # No location configured
                self._show_location("Location not configured - please set it in the main application")
                self.calculate_btn.setEnabled(False)
                
        except Exception:
            self._show_location("Error loading location")
            self.calculate_btn.setEnabled(False)

    def load_catalog_options(self):
        """Load available catalogs from database"""
        try:
            # Query catalogs directly - avoids building a full
            # DSOCalculationThread/DSOVisibilityCalculator (and the location/
            # timezone DB round trips that come with it) just for this list.
            self.available_catalogs = _get_available_catalogs()
        except Exception as e:
            print(f"Error loading catalogs: {e}")
            self.available_catalogs = ["M", "NGC", "IC"]  # Fallback default catalogs
    
    def load_dso_type_options(self):
        """Load curated list of common DSO types for astrophotography"""
        # Curated list of common types for astrophotography planning
        common_types = [(code, type_name(code)) for code in
                        ("GALXY", "BRTNB", "DRKNB", "PLNNB", "SNREM", "CL+NB", "OPNCL", "GLOCL", "GALCL")]

        # Add "All Types" option first
        self.dso_type_combo.addItem("All Types")

        # Add curated types sorted alphabetically by friendly name
        for dso_type, friendly_name in sorted(common_types, key=lambda x: x[1]):
            self.dso_type_combo.addItem(friendly_name, dso_type)

        self.dso_type_combo.setCurrentText("All Types")

        # Ensure dropdown view is wide enough to show full text
        self.dso_type_combo.view().setMinimumWidth(self.dso_type_combo.minimumSizeHint().width())
    

    def closeEvent(self, event):
        """Save settings and window position when closing"""
        settings = app_settings()
        settings.setValue("BestDSOTonight/use_target_list", self.use_target_list_checkbox.isChecked())
        settings.setValue("BestDSOTonight/hide_completed", self.hide_completed_checkbox.isChecked())
        settings.setValue("BestDSOTonight/auto_calculate", self.auto_calculate_checkbox.isChecked())
        settings.setValue("BestDSOTonight/use_custom_horizon", self.use_horizon_checkbox.isChecked())
        super().closeEvent(event)

    def _on_target_list_checkbox_changed(self, checked):
        """Handle target list checkbox state change"""
        # checked is a boolean from the toggled signal

        # Disable/enable settings based on whether target list is used
        # When using target list, catalog, type, and magnitude filters don't apply
        self.catalog_combo.setEnabled(not checked)
        self.dso_type_combo.setEnabled(not checked)
        self.max_magnitude_combo.setEnabled(not checked)

        # DSO limit doesn't apply when using target list (uses all targets)
        self.dso_limit_spin.setEnabled(not checked)

        # Don't Show Completed only applies to the target list
        self.hide_completed_checkbox.setEnabled(checked)

        # The global stylesheet's :disabled state will automatically
        # grey out the disabled widgets

    def _on_auto_calculate_checkbox_changed(self, checked):
        """Handle Auto Calculate checkbox state change.

        If Auto Calculate is being turned on and there are no results
        currently displayed, trigger a calculation right away.
        """
        if not checked:
            return

        # Guard against firing before the UI (results_table) is fully built,
        # e.g. when the checkbox state is restored from settings during init.
        if not hasattr(self, 'results_table'):
            return

        if self._should_auto_calculate():
            self.calculate_best_dsos()

    def calculate_best_dsos(self):
        """Start the calculation of best DSOs for tonight"""
        if self.calc_thread and self.calc_thread.isRunning():
            return

        # Get settings
        min_altitude = self.min_altitude_spin.value()
        max_magnitude = float(self.max_magnitude_combo.currentText())
        dso_limit = self.dso_limit_spin.value()
        use_target_list = self.use_target_list_checkbox.isChecked()
        hide_completed = use_target_list and self.hide_completed_checkbox.isChecked()

        # Store whether we're using target list for display_results
        self._using_target_list = use_target_list

        # Get selected catalog (ignored if using target list)
        selected_catalog = self.catalog_combo.currentText()
        selected_catalogs = [] if selected_catalog == "All Catalogs" else [selected_catalog]

        # Get selected DSO type (ignored if using target list)
        selected_dso_type_text = self.dso_type_combo.currentText()
        if selected_dso_type_text == "All Types":
            selected_dso_types = []
        else:
            # Get the database code from the combo box data
            selected_dso_type = self.dso_type_combo.currentData()
            selected_dso_types = [selected_dso_type] if selected_dso_type else []

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Calculating...")

        # Create status text
        if use_target_list:
            self.status_label.setText(f"Calculating visibility for DSOs in your target list... (may download ephemeris data on first run)")
        else:
            catalog_text = "all catalogs" if not selected_catalogs else f"{selected_catalog} catalog"
            type_text = "all types" if not selected_dso_types else f"{selected_dso_type} objects"
            self.status_label.setText(f"Calculating visibility for {type_text} from {catalog_text}... (may download ephemeris data on first run)")

        themed_style(self.status_label, lambda: f"color: {COLORS['info']};")

        # When the objects should be up
        time_choice = self._time_choice()

        # Re-check the active location's horizon - it may have been imported
        # or cleared in Settings while this window was open
        self._refresh_horizon_state()
        use_horizon = self.use_horizon_checkbox.isEnabled() and self.use_horizon_checkbox.isChecked()

        # Start calculation thread
        self.calc_thread = DSOCalculationThread(min_altitude, max_magnitude, selected_catalogs, dso_limit, selected_dso_types, use_target_list, time_choice, use_horizon, hide_completed)
        self.calc_thread.progress.connect(self.progress_bar.setValue)
        self.calc_thread.result_ready.connect(self.display_results)
        self.calc_thread.error_occurred.connect(self.handle_error)
        self.calc_thread.start()

    @staticmethod
    def _color_moon_item(moon_item, moon_sep):
        """Color a moon separation cell green-to-red by how far the object is from the moon"""
        from PySide6.QtGui import QColor
        if moon_sep is None:
            moon_item.setBackground(QColor(COLORS['background']))
            moon_item.setForeground(QColor(COLORS['text_disabled']))
            return
        # rank keeps the green-to-red order in Night Vision
        if moon_sep > 45:
            moon_color, rank = "#66cc66", 1.0
        elif moon_sep > 20:
            moon_color, rank = "#cccc44", 0.66
        elif moon_sep > 8:
            moon_color, rank = "#cc8844", 0.33
        else:
            moon_color, rank = "#cc4444", 0.0
        moon_bg = tint(moon_color, 0.25, rank=rank)
        moon_item.setBackground(QColor(moon_bg))
        moon_item.setForeground(QColor(adapt_color(moon_color, background=moon_bg, rank=rank)))

    def _recolor_moon_column(self):
        """Apply the current theme's colors to the moon separation cells"""
        moon_col = getattr(self, '_moon_col', None)
        if moon_col is None:
            return
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, moon_col)
            if item is not None:
                moon_sep = item.data(Qt.UserRole)
                self._color_moon_item(item, None if moon_sep is None or moon_sep < 0 else moon_sep)

    def display_results(self, visible_dsos):
        """Display the calculation results in the table"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Best DSOs Tonight")

        now = datetime.now()
        last_calculated = format_datetime(now)

        # Persist the date of this calculation so Auto Calculate knows not to
        # re-run again today
        settings = app_settings()
        settings.setValue("BestDSOTonight/last_calculated_date", now.strftime("%Y-%m-%d"))

        thread = self.calc_thread.calculation if self.calc_thread else None
        blocked = thread.blocked_by_horizon if thread else 0
        blocked_text = f" ({blocked} blocked by custom horizon)" if blocked else ""
        when = "tonight"
        if thread is not None and thread.window_start is not None:
            when = describe_time_window(thread.time_mode, thread.window_start, thread.window_end)
            if thread.whole_window:
                when = f"for all of {when}"
        self.results_group.setTitle(f"Best DSOs {when}")

        if not visible_dsos:
            self.status_label.setText(f"No DSOs meet the visibility criteria {when}{blocked_text}. Last calculated: {last_calculated}")
            self.status_label.setStyleSheet("")  # Reset to default color
            self.results_table.setRowCount(0)
            return

        self.status_label.setText(f"Found {len(visible_dsos)} visible DSOs {when}{blocked_text}. Last calculated: {last_calculated}")
        self.status_label.setStyleSheet("")  # Reset to default color
        
        # Disable sorting temporarily while populating
        self.results_table.setSortingEnabled(False)

        # Configure columns based on whether we're using target list
        moon_illumination = visible_dsos[0].get("moon_illumination", 0.5) if visible_dsos else 0.5
        moon_col_header = f"Moon ({moon_illumination:.0%} lit)"

        if getattr(self, '_using_target_list', False):
            self.results_table.setColumnCount(10)
            self.results_table.setHorizontalHeaderLabels([
                "DSO", "Type", "Constellation", "Magnitude",
                "Max Alt.", "Best Time", "Direction", "Visible Hours",
                "Telescope", moon_col_header
            ])
            header = self.results_table.horizontalHeader()
            header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(9, QHeaderView.Interactive)
            self.results_table.setColumnWidth(9, 90)
            moon_col = 9
        else:
            self.results_table.setColumnCount(9)
            self.results_table.setHorizontalHeaderLabels([
                "DSO", "Type", "Constellation", "Magnitude",
                "Max Alt.", "Best Time", "Direction", "Visible Hours",
                moon_col_header
            ])
            header = self.results_table.horizontalHeader()
            header.setSectionResizeMode(8, QHeaderView.Interactive)
            self.results_table.setColumnWidth(8, 90)
            moon_col = 8

        self._moon_col = moon_col  # for recoloring on theme changes

        # Populate table
        self.results_table.setRowCount(len(visible_dsos))
        
        # Store the DSO data for later use in double-click handler
        self.visible_dsos_data = visible_dsos
        
        for row, dso_data in enumerate(visible_dsos):
            dso_info = dso_data["dso_info"]
            
            # DSO name - store DSO data in item for sorting
            name_item = QTableWidgetItem(dso_info["name"])
            name_item.setData(Qt.UserRole, dso_data)  # Store full DSO data
            self.results_table.setItem(row, 0, name_item)
            
            # Type - use friendly name
            friendly_type = type_name(dso_info["type"])
            self.results_table.setItem(row, 1, QTableWidgetItem(friendly_type))
            
            # Constellation
            self.results_table.setItem(row, 2, QTableWidgetItem(dso_info["constellation"]))
            
            # Magnitude - use numeric sorting, show "N/A" for unknown magnitude
            mag_value = dso_info['magnitude']
            mag_item = NumericTableWidgetItem()
            if mag_value >= 99.0:
                mag_item.setData(Qt.DisplayRole, "N/A")
            else:
                mag_item.setData(Qt.DisplayRole, f"{mag_value:.1f}")
            mag_item.setData(Qt.UserRole, mag_value)  # Store numeric value for sorting
            mag_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 3, mag_item)

            # Maximum altitude - use numeric sorting
            alt_item = NumericTableWidgetItem()
            alt_item.setData(Qt.DisplayRole, f"{dso_data['max_altitude']:.0f}°")
            alt_item.setData(Qt.UserRole, dso_data['max_altitude'])  # Store numeric value for sorting
            alt_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 4, alt_item)

            # Best time - store time as sortable value
            time_str = format_time(dso_data["optimal_time"])
            time_item = NumericTableWidgetItem()
            time_item.setData(Qt.DisplayRole, time_str)
            time_item.setData(Qt.UserRole, dso_data["optimal_time"].timestamp())  # 01:00 sorts after 23:00
            time_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 5, time_item)

            # Direction
            dir_item = QTableWidgetItem(dso_data["direction"])
            dir_item.setTextAlignment(Qt.AlignCenter)
            dir_item.setToolTip(dso_data.get("direction_tooltip", ""))
            self.results_table.setItem(row, 6, dir_item)

            # Visible hours - use numeric sorting
            hours_item = NumericTableWidgetItem()
            visible_hours = dso_data['visible_hours']
            hours_item.setData(Qt.DisplayRole, f"{visible_hours:.1f}h" if visible_hours is not None else "\u2014")
            hours_item.setData(Qt.UserRole, visible_hours or 0)  # Store numeric value for sorting
            hours_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 7, hours_item)

            # Telescope column (only when using target list)
            if getattr(self, '_using_target_list', False):
                telescope_name = dso_info.get("telescope_name", "Any")
                telescope_item = QTableWidgetItem(telescope_name)
                telescope_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, 8, telescope_item)

            # Moon separation cell
            moon_sep = dso_data.get("moon_separation")
            moon_item = NumericTableWidgetItem()
            if moon_sep is not None:
                moon_item.setData(Qt.DisplayRole, f"{moon_sep:.0f}°")
                moon_item.setData(Qt.UserRole, moon_sep)
            else:
                moon_item.setData(Qt.DisplayRole, "N/A")
                moon_item.setData(Qt.UserRole, -1.0)  # sorts to bottom
            moon_item.setTextAlignment(Qt.AlignCenter)
            self._color_moon_item(moon_item, moon_sep)
            if moon_sep is None:
                moon_item.setToolTip("Moon separation unavailable.")
            else:
                if moon_sep > 45:
                    quality = "Good — minimal moon interference expected."
                elif moon_sep > 20:
                    quality = "Moderate — some moon glow may affect faint objects."
                elif moon_sep > 8:
                    quality = "Poor — significant moon interference likely."
                else:
                    quality = "Very poor — object is very close to the moon."
                moon_item.setToolTip(
                    f"Moon Separation: {moon_sep:.1f}°\n"
                    f"Angular distance between this object and the moon at its best viewing time.\n\n"
                    f"{quality}\n\n"
                    f"Color guide:  >45° green  |  20–45° yellow  |  8–20° orange  |  <8° red"
                )
            self.results_table.setItem(row, moon_col, moon_item)

        # Re-enable sorting
        self.results_table.setSortingEnabled(True)

        # Sort by visible hours (column 7) in descending order by default
        self.results_table.sortByColumn(7, Qt.DescendingOrder)

    def on_item_double_clicked(self, item):
        """Handle double-click on table item to show object details"""
        try:
            # Get DSO data from the name item (column 0) of the clicked row
            name_item = self.results_table.item(item.row(), 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    self.show_object_detail(dso_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open object details: {e}")
    
    def show_object_detail(self, dso_data):
        """Show detailed information for the selected DSO"""
        try:
            from cosmoscollection.catalog.DSODetail import DSODetailWindow
            
            # Get DSO coordinates from astropy result
            coordinates = dso_data.get("coordinates")
            if coordinates:
                ra_deg = coordinates.ra.degree
                dec_deg = coordinates.dec.degree
            else:
                # Fallback - try to get coordinates again
                from astropy.coordinates import SkyCoord
                try:
                    coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                    ra_deg = coord.ra.degree
                    dec_deg = coord.dec.degree
                except:
                    ra_deg = 0.0
                    dec_deg = 0.0
            
            # Debug: log designations
            logger.debug(f"BestDSO opening detail for: {dso_data['dso_info']['name']}")
            logger.debug(f"Designations from dso_info: {dso_data['dso_info']['designations']}")

            # Create the data dictionary expected by DSODetailWindow
            detail_data = {
                "name": dso_data["dso_info"]["name"],
                "type": dso_data["dso_info"]["type"],
                "constellation": dso_data["dso_info"]["constellation"],
                "magnitude": dso_data["dso_info"]["magnitude"],
                "surface_brightness": dso_data["dso_info"]["surface_brightness"],
                "size_min": dso_data["dso_info"]["size_min"],
                "size_max": dso_data["dso_info"]["size_max"],
                "dso_type": dso_data["dso_info"]["type"],
                "dso_class": dso_data["dso_info"]["dso_class"],
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "catalogue": dso_data["dso_info"]["name"].split()[0],  # Extract catalog prefix
                "id": dso_data["dso_info"]["name"].split()[1] if len(dso_data["dso_info"]["name"].split()) > 1 else "",
                "designations": dso_data["dso_info"]["designations"],
                # Add visibility info as additional data
                "visibility_info": {
                    "max_altitude": dso_data["max_altitude"],
                    "optimal_time": dso_data["optimal_time"],
                    "direction": dso_data["direction"],
                    "visible_hours": dso_data["visible_hours"]
                }
            }
            
            # Create and show the detail window
            detail_window = DSODetailWindow(detail_data, self)
            detail_window.show()
            
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load DSODetailWindow: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open object details: {e}")

    def handle_error(self, error_msg):
        """Handle calculation errors"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Best DSOs Tonight")
        self.status_label.setText(f"Error: {error_msg}")
        themed_style(self.status_label, lambda: f"color: {COLORS['error']};")  # Red for errors
        QMessageBox.warning(self, "Calculation Error", error_msg)

    def _show_context_menu(self, position):
        """Show context menu when right-clicking on the DSO table"""
        # Get the item at the clicked position
        index = self.results_table.indexAt(position)
        if not index.isValid():
            return  # No item at this position

        # Get the row number
        row = index.row()
        if row < 0 or row >= self.results_table.rowCount():
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

        plan_session_action = context_menu.addAction("Plan a Session")
        plan_session_action.triggered.connect(lambda: self._context_plan_session(row))

        context_menu.addSeparator()

        # Check if DSO is already in target list
        name_item = self.results_table.item(row, 0)
        if name_item:
            dso_data = name_item.data(Qt.UserRole)
            if dso_data:
                dso_name = dso_data["dso_info"]["name"]

                # Check if in target list
                is_in_target_list = self._is_dso_in_target_list(dso_name)

                if is_in_target_list:
                    # Show "Open from Target List" option
                    open_target_action = context_menu.addAction("Open from Target List")
                    open_target_action.triggered.connect(lambda: self._context_open_from_target_list(row))
                else:
                    # Show "Add to Target List" option
                    target_action = context_menu.addAction("Add to Target List")
                    target_action.triggered.connect(lambda: self._context_add_to_target_list(row))

        # Show the menu at the clicked position
        context_menu.exec(self.results_table.mapToGlobal(position))

    def _context_view_details(self, row):
        """View DSO details from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    self.show_object_detail(dso_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open DSO details: {str(e)}")

    def _context_open_visibility(self, row):
        """Open DSO Visibility Calculator from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    dso_name = dso_data["dso_info"]["name"]

                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_name)
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Import and open DSO Visibility Calculator
                    from cosmoscollection.planning.DSOVisibilityCalculator import DSOVisibilityApp

                    # Store reference to prevent garbage collection - use list to allow multiple windows
                    if not hasattr(self, 'visibility_windows'):
                        self.visibility_windows = []

                    visibility_window = DSOVisibilityApp()

                    # Set DSO name in input field for display and title
                    if hasattr(visibility_window, 'dso_input'):
                        visibility_window.dso_input.setText(dso_name)

                    # Use coordinates for accurate calculation
                    if hasattr(visibility_window, 'set_dso_coordinates'):
                        visibility_window.set_dso_coordinates(ra_deg, dec_deg)

                    # Show the window immediately
                    visibility_window.show()
                    visibility_window.raise_()
                    visibility_window.activateWindow()

                    # Store reference after showing to prevent garbage collection
                    # (closed ones dropped - they're freed once closed)
                    self.visibility_windows = [w for w in self.visibility_windows if window_is_open(w)]
                    self.visibility_windows.append(visibility_window)

                    # Automatically trigger calculation after a short delay
                    if hasattr(visibility_window, 'calculate_visibility'):
                        QTimer.singleShot(500, visibility_window.calculate_visibility)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open DSO Visibility Calculator: {str(e)}")

    def _context_open_aladin(self, row):
        """Open Aladin Lite from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Create data dictionary for Aladin Lite
                    detail_data = {
                        'name': dso_data["dso_info"]["name"],
                        'ra_deg': ra_deg,
                        'dec_deg': dec_deg,
                        'size_min': dso_data["dso_info"]["size_min"],
                        'size_max': dso_data["dso_info"]["size_max"],
                        'dsodetailid': dso_data["dso_info"]["name"]
                    }

                    # Import and open Aladin Lite window
                    from cosmoscollection.planning.FOVSimulator import AladinLiteWindow
                    aladin_window = AladinLiteWindow(detail_data, self)
                    aladin_window.show()
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

    def _context_send_to_nina(self, row):
        """Send DSO coordinates to NINA Framing Assistant"""
        name_item = self.results_table.item(row, 0)
        if not name_item:
            return

        dso_data = name_item.data(Qt.UserRole)
        if not dso_data:
            QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
            return

        # Get coordinates from SkyCoord object or fallback to dso_info
        coordinates = dso_data.get("coordinates")
        if coordinates:
            ra_deg = coordinates.ra.degree
            dec_deg = coordinates.dec.degree
        else:
            dso_info = dso_data.get("dso_info", {})
            ra_deg = dso_info.get("ra_deg")
            dec_deg = dso_info.get("dec_deg")

        target_name = dso_data.get("dso_info", {}).get("name", "Unknown")
        NINAIntegration.send_to_framing_assistant(ra_deg, dec_deg, target_name, self)

    def _context_slew_to_target(self, row):
        """Slew mount to DSO coordinates"""
        name_item = self.results_table.item(row, 0)
        if not name_item:
            return

        dso_data = name_item.data(Qt.UserRole)
        if not dso_data:
            QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
            return

        # Get coordinates from SkyCoord object or fallback to dso_info
        coordinates = dso_data.get("coordinates")
        if coordinates:
            ra_deg = coordinates.ra.degree
            dec_deg = coordinates.dec.degree
        else:
            dso_info = dso_data.get("dso_info", {})
            ra_deg = dso_info.get("ra_deg")
            dec_deg = dso_info.get("dec_deg")

        target_name = dso_data.get("dso_info", {}).get("name", "Unknown")
        NINAIntegration.slew_to_coordinates(ra_deg, dec_deg, target_name, self)

    def _context_plan_session(self, row):
        """Open Session Manager pre-filled with this DSO, from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Create data dictionary for Session Manager
                    target_data = {
                        'name': dso_data["dso_info"]["name"],
                        'ra_deg': ra_deg,
                        'dec_deg': dec_deg,
                        'magnitude': dso_data["dso_info"]["magnitude"],
                        'size_min': dso_data["dso_info"]["size_min"],
                        'size_max': dso_data["dso_info"]["size_max"],
                        'constellation': dso_data["dso_info"]["constellation"],
                        'dso_type': dso_data["dso_info"]["type"],
                        'dso_class': dso_data["dso_info"]["dso_class"],
                        'designations': dso_data["dso_info"].get("designations", "")
                    }

                    # Import and open Session Manager, then create a session for the DSO
                    from cosmoscollection.sessions.SessionManager import SessionManagerWindow
                    if not window_is_open(getattr(self, 'session_manager_window', None)):
                        self.session_manager_window = SessionManagerWindow()

                    self.session_manager_window.create_session_from_dso(target_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Session Manager: {str(e)}")

    def _context_add_to_target_list(self, row):
        """Add DSO to target list from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Create data dictionary for target list
                    target_data = {
                        'name': dso_data["dso_info"]["name"],
                        'ra_deg': ra_deg,
                        'dec_deg': dec_deg,
                        'magnitude': dso_data["dso_info"]["magnitude"],
                        'size_min': dso_data["dso_info"]["size_min"],
                        'size_max': dso_data["dso_info"]["size_max"],
                        'constellation': dso_data["dso_info"]["constellation"],
                        'dso_type': dso_data["dso_info"]["type"],
                        'dso_class': dso_data["dso_info"]["dso_class"],
                        'designations': dso_data["dso_info"].get("designations", "")
                    }

                    # Import and open Target List window, then add the DSO
                    from cosmoscollection.planning.DSOTargetList import DSOTargetListWindow
                    if not window_is_open(getattr(self, 'target_list_window', None)):
                        self.target_list_window = DSOTargetListWindow()

                    self.target_list_window.show()
                    self.target_list_window.raise_()
                    self.target_list_window.activateWindow()

                    # Add the DSO to the target list
                    self.target_list_window.add_target_from_dso(target_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {str(e)}")

    def _is_dso_in_target_list(self, dso_name):
        """Check if a DSO is in the target list

        Args:
            dso_name: Name of the DSO to check

        Returns:
            bool: True if DSO is in target list, False otherwise
        """
        try:
            from cosmoscollection.planning.DSOTargetList import find_existing_target
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                return find_existing_target(conn, dso_name) is not None
        except Exception as e:
            logger.error(f"Error checking if DSO is in target list: {str(e)}")
            return False

    def _context_open_from_target_list(self, row):
        """Open DSO from target list to view notes"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    dso_name = dso_data["dso_info"]["name"]

                    # Import and open Target List window
                    from cosmoscollection.planning.DSOTargetList import DSOTargetListWindow
                    if not window_is_open(getattr(self, 'target_list_window', None)):
                        self.target_list_window = DSOTargetListWindow()

                    # Open and select the target
                    success = self.target_list_window.open_and_select_target(dso_name)

                    if not success:
                        QMessageBox.warning(self, "Not Found",
                                          f"Could not find '{dso_name}' in your target list.")
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open from target list: {str(e)}")


def main():
    """Main entry point for the application"""
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = BestDSOTonightWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()