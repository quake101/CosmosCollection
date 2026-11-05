"""The Settings dialog, plus backup/restore and the restore image fixup it runs."""

import logging
import os
import sqlite3
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.core.ResourceManager import ResourceManager
from cosmoscollection.nina.NINAIntegration import NINAIntegration
from cosmoscollection.ui.MapLocationPicker import MapLocationPickerDialog
from cosmoscollection.ui.Theme import (
    COLORS,
    apply_theme,
    font_size,
    load_settings as load_theme_settings,
    save_settings as save_theme_settings,
    theme_manager,
    themed_style,
    tint,
)
from cosmoscollection.ui.ThemeTab import ThemeSettingsTab

logger = logging.getLogger(__name__)


# User tables included in backups, in restore order (parents before children)
BACKUP_TABLES = [
    "usersettings", "usertelescopes", "userequipment", "telescope_equipment",
    "userimages", "usertargetlist", "usercollages", "usercollageimages",
    "usersessions", "usersessionobservations", "usersessionobservationfilters", "usersessionfiles",
]


class HorizonPreview(QWidget):
    """Small altitude-vs-azimuth plot of a custom horizon for the Location Manager.

    Painted with QPainter rather than matplotlib so opening Settings stays cheap.
    Hovering shows the horizon altitude at that azimuth."""

    MAX_ALT = 90.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile = None
        self._altitudes = []  # horizon altitude at each whole degree of azimuth, 0-360
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        theme_manager().theme_changed.connect(self.update)

    def set_profile(self, profile):
        self.profile = profile
        self._altitudes = [float(alt) for alt in profile.altitude_at(list(range(361)))] if profile else []
        self.update()

    def sizeHint(self):
        return QSize(300, 110)

    def _plot_rect(self):
        """Plot area inside the axis labels"""
        metrics = self.fontMetrics()
        left = metrics.horizontalAdvance("90°") + 6
        top = metrics.ascent() / 2 + 2  # room for the 90° label centered on the top gridline
        bottom = metrics.height() + 4
        return QRectF(left, top, max(1, self.width() - left - 6), max(1, self.height() - bottom - top))

    def _to_point(self, rect, az, alt):
        return QPointF(rect.left() + rect.width() * az / 360.0,
                       rect.bottom() - rect.height() * max(0.0, min(alt, self.MAX_ALT)) / self.MAX_ALT)

    def paintEvent(self, event):
        from PySide6.QtGui import QPen, QPolygonF
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self._plot_rect()
        painter.fillRect(rect, QColor(COLORS['background_lighter']))

        # Altitude gridlines and labels
        text_color = QColor(COLORS['text_secondary'])
        grid_pen = QPen(QColor(COLORS['border_light']))
        grid_pen.setStyle(Qt.DotLine)
        metrics = self.fontMetrics()
        for alt in (0, 30, 60, 90):
            y = self._to_point(rect, 0, alt).y()
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(text_color)
            label = f"{alt}°"
            painter.drawText(QPointF(rect.left() - metrics.horizontalAdvance(label) - 4,
                                     y + metrics.ascent() / 2 - 1), label)

        # Cardinal direction gridlines and labels
        for az, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W"), (360, "N")):
            x = self._to_point(rect, az, 0).x()
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setPen(text_color)
            label_x = min(max(x - metrics.horizontalAdvance(label) / 2, rect.left()),
                          rect.right() - metrics.horizontalAdvance(label))
            painter.drawText(QPointF(label_x, rect.bottom() + metrics.ascent() + 2), label)

        # Horizon profile, filled down to 0°
        if self._altitudes:
            accent = QColor(COLORS['accent'])
            outline = [self._to_point(rect, az, alt) for az, alt in enumerate(self._altitudes)]
            fill = QColor(accent)
            fill.setAlpha(110)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawPolygon(QPolygonF([self._to_point(rect, 0, 0)] + outline + [self._to_point(rect, 360, 0)]))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(accent, 1.5))
            painter.drawPolyline(QPolygonF(outline))
        painter.end()

    def mouseMoveEvent(self, event):
        rect = self._plot_rect()
        pos = event.position()
        if not self._altitudes or not rect.contains(pos):
            QToolTip.hideText()
            return
        az = round((pos.x() - rect.left()) / rect.width() * 360.0) % 360
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        direction = directions[int((az + 11.25) / 22.5) % 16]
        QToolTip.showText(event.globalPosition().toPoint(),
                          f"Azimuth {az}° ({direction}): horizon {self._altitudes[az]:.1f}°", self)


# --- Settings Dialog ---
class SettingsDialog(QDialog):
    """Settings dialog for configuring application preferences"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(800, 560)

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
        location_layout = QHBoxLayout(location_tab)

        # Track which location is being edited
        self.editing_location_id = None

        # --- Left Panel: Saved Locations List ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        saved_label = QLabel("Saved Locations:")
        saved_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(saved_label)

        self.location_list = QListWidget()
        self.location_list.setMinimumWidth(220)
        self.location_list.currentItemChanged.connect(self._on_location_selected)
        left_layout.addWidget(self.location_list)

        left_btn_layout = QHBoxLayout()
        self.delete_location_btn = QPushButton("Delete")
        self.delete_location_btn.setToolTip("Delete the selected location")
        self.delete_location_btn.clicked.connect(self._delete_location)
        self.set_active_btn = QPushButton("Set as Active")
        self.set_active_btn.setToolTip("Set the selected location as the active observer location")
        self.set_active_btn.clicked.connect(self._set_active_location)
        left_btn_layout.addWidget(self.delete_location_btn)
        left_btn_layout.addWidget(self.set_active_btn)
        left_layout.addLayout(left_btn_layout)

        location_layout.addWidget(left_panel)

        # --- Right Panel: Location Details Form ---
        right_panel = QGroupBox("Location Details")
        right_layout = QVBoxLayout(right_panel)

        # Labels in column 0; fields span columns 1-2, and the button rows
        # split columns 1-2 evenly so every field and button shares the same
        # left and right edges
        form_grid = QGridLayout()
        form_grid.setColumnStretch(1, 1)
        form_grid.setColumnStretch(2, 1)
        right_layout.addLayout(form_grid)

        # Latitude input
        self.latitude_input = QLineEdit()
        self.latitude_input.setPlaceholderText("e.g., 40.7128")
        form_grid.addWidget(QLabel("Latitude:"), 0, 0)
        form_grid.addWidget(self.latitude_input, 0, 1, 1, 2)

        # Longitude input
        self.longitude_input = QLineEdit()
        self.longitude_input.setPlaceholderText("e.g., -74.0060")
        form_grid.addWidget(QLabel("Longitude:"), 1, 0)
        form_grid.addWidget(self.longitude_input, 1, 1, 1, 2)

        # Location name
        self.location_name_input = QLineEdit()
        self.location_name_input.setPlaceholderText("e.g., Home Observatory (optional)")
        form_grid.addWidget(QLabel("Name:"), 2, 0)
        form_grid.addWidget(self.location_name_input, 2, 1, 1, 2)

        # Timezone
        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
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
        form_grid.addWidget(QLabel("Time Zone:"), 3, 0)
        form_grid.addWidget(self.timezone_combo, 3, 1, 1, 2)

        # Custom horizon - saved immediately for an existing location, or held
        # until Save Location for a new one
        self.pending_horizon = None
        self.horizon_status_label = QLabel()
        # Take whatever width is left rather than widening the dialog for a
        # long file name - the full text is also in the tooltip
        self.horizon_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.import_horizon_btn = QPushButton("Import Horizon...")
        self.import_horizon_btn.setToolTip(
            "Import a custom horizon for this location from a NINA horizon file (.hrz)\n"
            "or a Stellarium landscape (landscape.ini or .zip).\n\n"
            "Visibility calculations treat objects below this horizon as blocked\n"
            "(e.g. by trees or buildings) while this is the active location."
        )
        self.import_horizon_btn.clicked.connect(self._import_horizon)
        self.clear_horizon_btn = QPushButton("Clear Horizon")
        self.clear_horizon_btn.setToolTip("Remove the custom horizon from this location")
        self.clear_horizon_btn.clicked.connect(self._clear_horizon)
        # Preview plot, shown only when the location has a horizon
        self.horizon_preview = HorizonPreview()
        form_grid.addWidget(QLabel("Horizon:"), 4, 0)
        form_grid.addWidget(self.horizon_status_label, 4, 1, 1, 2)
        form_grid.addWidget(self.horizon_preview, 5, 1, 1, 2)
        form_grid.addWidget(self.import_horizon_btn, 6, 1)
        form_grid.addWidget(self.clear_horizon_btn, 6, 2)
        self._show_horizon_status(None)

        # Map picker button
        map_button = QPushButton("Select Location from Map")
        map_button.setToolTip("Open an interactive map to visually select your location")
        map_button.clicked.connect(self._open_map_picker)
        form_grid.addWidget(map_button, 7, 1, 1, 2)

        # Save / Clear buttons, set apart from the field buttons above
        form_grid.setRowMinimumHeight(8, 8)
        self.save_location_btn = QPushButton("Save Location")
        self.save_location_btn.clicked.connect(self._save_location)
        self.clear_form_btn = QPushButton("Clear Form")
        self.clear_form_btn.clicked.connect(self._clear_location_form)
        form_grid.addWidget(self.save_location_btn, 9, 1)
        form_grid.addWidget(self.clear_form_btn, 9, 2)

        right_layout.addStretch()
        location_layout.addWidget(right_panel)

        # Application Settings tab
        app_settings_scroll = QScrollArea()
        app_settings_scroll.setWidgetResizable(True)
        app_settings_scroll.setFrameShape(QScrollArea.NoFrame)
        app_settings_tab = QWidget()
        app_settings_layout = QVBoxLayout(app_settings_tab)
        app_settings_scroll.setWidget(app_settings_tab)

        # Network group - offline mode (OnlineLimits.py)
        network_group = QGroupBox("Network")
        network_layout = QVBoxLayout(network_group)

        self.offline_mode_checkbox = QCheckBox("Offline mode")
        self.offline_mode_checkbox.setToolTip(
            "When enabled, nothing in the app goes online: no update checks, weather,\n"
            "SIMBAD lookups, star catalog downloads, astrometry.net, FOV Simulator sky view\n"
            "or map picker. Links that open your web browser and NINA still work."
        )
        self.offline_mode_checkbox.toggled.connect(self._on_offline_mode_toggled)
        network_layout.addWidget(self.offline_mode_checkbox)

        offline_help = QLabel(
            "Turns off update checks, the weather forecast, SIMBAD lookups, astrometry.net, the FOV Simulator's "
            "sky view and the map picker. The built-in plate solver uses only the star catalog areas already "
            "downloaded. ASTAP, NINA and links that open your web browser still work."
        )
        offline_help.setWordWrap(True)
        offline_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(offline_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 20px; }}")
        network_layout.addWidget(offline_help)

        app_settings_layout.addWidget(network_group)

        # UI Preferences group
        ui_prefs_group = QGroupBox("User Interface Preferences")
        ui_prefs_layout = QVBoxLayout(ui_prefs_group)

        # Show Observer Location checkbox
        self.show_observer_location_checkbox = QCheckBox("Show Observer Location")
        self.show_observer_location_checkbox.setToolTip(
            "When enabled, displays your observer location information in various windows\n"
            "(Best DSO Tonight, DSO Visibility Calculator, and Main Window)"
        )
        ui_prefs_layout.addWidget(self.show_observer_location_checkbox)

        # Check for updates on startup checkbox
        self.check_updates_checkbox = QCheckBox("Check for updates on startup")
        self.check_updates_checkbox.setToolTip(
            "When enabled, automatically checks for application updates when the program starts"
        )
        ui_prefs_layout.addWidget(self.check_updates_checkbox)

        # A snoozed or skipped update (chosen in the update dialog), with a way to undo it
        self.update_reminder_row = QWidget()
        update_reminder_layout = QHBoxLayout(self.update_reminder_row)
        update_reminder_layout.setContentsMargins(20, 0, 0, 0)
        self.update_reminder_label = QLabel()
        self.update_reminder_label.setWordWrap(True)
        themed_style(self.update_reminder_label,
                     lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; }}")
        update_reminder_layout.addWidget(self.update_reminder_label, 1)
        update_reminder_reset = QPushButton("Remind Me Again")
        update_reminder_reset.setAutoDefault(False)
        update_reminder_reset.setToolTip("Forget the snooze / skipped version - the next check tells you about updates")
        update_reminder_reset.clicked.connect(self._reset_update_reminders)
        update_reminder_layout.addWidget(update_reminder_reset)
        ui_prefs_layout.addWidget(self.update_reminder_row)
        self._update_reminder_status()

        # Minimize to system tray checkbox
        self.minimize_to_tray_checkbox = QCheckBox("Minimize to system tray")
        self.minimize_to_tray_checkbox.setToolTip(
            "When enabled, closing or minimizing hides the window to the system tray.\n"
            "Double-click the tray icon to restore. Right-click for quick actions."
        )
        ui_prefs_layout.addWidget(self.minimize_to_tray_checkbox)

        # SIMBAD lookup on search failure checkbox
        self.simbad_lookup_checkbox = QCheckBox("Offer SIMBAD lookup when search finds no results")
        self.simbad_lookup_checkbox.setToolTip(
            "When enabled, searching for an object that isn't in the local database\n"
            "prompts you to look it up in the SIMBAD astronomical database"
        )
        ui_prefs_layout.addWidget(self.simbad_lookup_checkbox)

        # Enable log file checkbox + open folder button
        log_file_layout = QHBoxLayout()
        self.enable_logfile_checkbox = QCheckBox("Enable log file")
        self.enable_logfile_checkbox.setToolTip(
            "When enabled, saves application logs to CosmosCollection.log\n"
            "in the user data directory. Useful for troubleshooting.\n"
            "Requires application restart to take effect."
        )
        log_file_layout.addWidget(self.enable_logfile_checkbox)
        open_log_folder_btn = QPushButton("Open Log Folder")
        open_log_folder_btn.setToolTip("Open the folder where log files are stored")
        open_log_folder_btn.setFixedWidth(120)
        open_log_folder_btn.clicked.connect(self._open_log_folder)
        log_file_layout.addWidget(open_log_folder_btn)
        log_file_layout.addStretch()
        ui_prefs_layout.addLayout(log_file_layout)

        # Time format setting
        time_format_layout = QHBoxLayout()
        time_format_label = QLabel("Time Format:")
        time_format_label.setMinimumWidth(120)
        self.time_format_combo = QComboBox()
        self.time_format_combo.addItems(["12-hour", "24-hour"])
        self.time_format_combo.setToolTip(
            "12-hour: displays times like 2:30 PM\n"
            "24-hour: displays times like 14:30"
        )
        time_format_layout.addWidget(time_format_label)
        time_format_layout.addWidget(self.time_format_combo)
        time_format_layout.addStretch()
        ui_prefs_layout.addLayout(time_format_layout)

        app_settings_layout.addWidget(ui_prefs_group)

        # Weather Units group
        weather_units_group = QGroupBox("Weather Units")
        weather_units_layout = QVBoxLayout(weather_units_group)

        # Temperature unit setting
        temp_unit_layout = QHBoxLayout()
        temp_unit_label = QLabel("Temperature:")
        temp_unit_label.setMinimumWidth(120)
        self.temp_unit_combo = QComboBox()
        self.temp_unit_combo.addItems(["Celsius", "Fahrenheit"])
        self.temp_unit_combo.setToolTip("Temperature display unit for weather forecast")
        temp_unit_layout.addWidget(temp_unit_label)
        temp_unit_layout.addWidget(self.temp_unit_combo)
        temp_unit_layout.addStretch()
        weather_units_layout.addLayout(temp_unit_layout)

        # Wind speed unit setting
        wind_unit_layout = QHBoxLayout()
        wind_unit_label = QLabel("Wind Speed:")
        wind_unit_label.setMinimumWidth(120)
        self.wind_unit_combo = QComboBox()
        self.wind_unit_combo.addItems(["km/h", "mph", "m/s"])
        self.wind_unit_combo.setToolTip("Wind speed display unit for weather forecast")
        wind_unit_layout.addWidget(wind_unit_label)
        wind_unit_layout.addWidget(self.wind_unit_combo)
        wind_unit_layout.addStretch()
        weather_units_layout.addLayout(wind_unit_layout)

        # Precipitation/visibility unit setting
        precip_unit_layout = QHBoxLayout()
        precip_unit_label = QLabel("Precipitation/Visibility:")
        precip_unit_label.setMinimumWidth(120)
        self.precip_unit_combo = QComboBox()
        self.precip_unit_combo.addItems(["Metric (mm, km)", "Imperial (in, mi)"])
        self.precip_unit_combo.setToolTip("Units for precipitation and visibility in weather forecast")
        precip_unit_layout.addWidget(precip_unit_label)
        precip_unit_layout.addWidget(self.precip_unit_combo)
        precip_unit_layout.addStretch()
        weather_units_layout.addLayout(precip_unit_layout)

        app_settings_layout.addWidget(weather_units_group)

        # Performance Settings group
        perf_settings_group = QGroupBox("Performance Settings")
        perf_settings_layout = QVBoxLayout(perf_settings_group)

        # Thread count setting
        thread_layout = QHBoxLayout()
        thread_label = QLabel("Maximum Threads:")
        thread_label.setMinimumWidth(120)
        self.thread_count_spinbox = QSpinBox()
        self.thread_count_spinbox.setMinimum(1)
        self.thread_count_spinbox.setMaximum(128)
        import os
        default_threads = max(1, (os.cpu_count() or 4) - 2)
        self.thread_count_spinbox.setValue(default_threads)
        self.thread_count_spinbox.setToolTip(
            f"Number of threads to use for parallel operations.\n"
            f"Default: {default_threads} (CPU cores - 2)\n"
            f"Higher values may improve performance but increase CPU usage."
        )
        thread_layout.addWidget(thread_label)
        thread_layout.addWidget(self.thread_count_spinbox)
        thread_layout.addStretch()
        perf_settings_layout.addLayout(thread_layout)

        # Thumbnail disk cache setting
        self.cache_thumbnails_checkbox = QCheckBox("Cache thumbnails to disk")
        self.cache_thumbnails_checkbox.setToolTip(
            "When enabled, generated thumbnails are saved alongside original images.\n"
            "Speeds up gallery loading on subsequent launches."
        )
        perf_settings_layout.addWidget(self.cache_thumbnails_checkbox)

        # Thumbnail cache help text
        cache_help = QLabel(
            "Thumbnails will be saved as <i>{filename}_{size}_Thumbnail.jpg</i> in the same folder as the original image.<br>"
            "This speeds up gallery loading on subsequent launches. Cache files can be safely deleted."
        )
        cache_help.setWordWrap(True)
        cache_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(cache_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 20px; }}")
        perf_settings_layout.addWidget(cache_help)

        app_settings_layout.addWidget(perf_settings_group)
        app_settings_layout.addStretch()

        # Integrations tab: list of integrations on the left, that integration's
        # settings on the right (QStackedWidget swapped via list selection).
        from PySide6.QtWidgets import QStackedWidget

        integrations_tab = QWidget()
        integrations_layout = QHBoxLayout(integrations_tab)

        # --- Left: list of integrations ---
        integrations_left_panel = QWidget()
        integrations_left_layout = QVBoxLayout(integrations_left_panel)
        integrations_left_layout.setContentsMargins(0, 0, 0, 0)

        integrations_label = QLabel("Integrations:")
        integrations_label.setStyleSheet("font-weight: bold;")
        integrations_left_layout.addWidget(integrations_label)

        self.integrations_list = QListWidget()
        self.integrations_list.setMinimumWidth(180)
        self.integrations_list.addItems(["NINA", "Plate Solving", "ASTAP", "Astrometry.net", "PixInsight", "Siril",
                                         "OpenWeather", "VisualCrossing", "WeatherAPI"])
        integrations_left_layout.addWidget(self.integrations_list)
        integrations_layout.addWidget(integrations_left_panel)

        # --- Right: stacked settings pages, one per integration ---
        self.integrations_stack = QStackedWidget()
        integrations_layout.addWidget(self.integrations_stack, stretch=1)

        self.integrations_list.currentRowChanged.connect(self.integrations_stack.setCurrentIndex)

        # NINA page
        nina_page = QWidget()
        nina_layout = QVBoxLayout(nina_page)

        nina_title = QLabel("NINA Integration")
        themed_style(nina_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        nina_layout.addWidget(nina_title)

        # Enable checkbox
        self.nina_enabled_checkbox = QCheckBox("Enable NINA Integration")
        self.nina_enabled_checkbox.setToolTip(
            "When enabled, 'Send to NINA Framing Assistant' options will appear\n"
            "in context menus throughout the application."
        )
        nina_layout.addWidget(self.nina_enabled_checkbox)

        nina_ip_layout = QHBoxLayout()
        nina_ip_label = QLabel("API Host:")
        nina_ip_label.setMinimumWidth(120)
        self.nina_ip_input = QLineEdit()
        self.nina_ip_input.setText("localhost")
        self.nina_ip_input.setPlaceholderText("localhost or IP address")
        self.nina_ip_input.setToolTip(
            "IP address or hostname of the machine running NINA.\n"
            "Use 'localhost' if NINA is running on this machine."
        )
        nina_ip_layout.addWidget(nina_ip_label)
        nina_ip_layout.addWidget(self.nina_ip_input)
        nina_layout.addLayout(nina_ip_layout)

        nina_port_layout = QHBoxLayout()
        nina_port_label = QLabel("API Port:")
        nina_port_label.setMinimumWidth(120)
        self.nina_port_spinbox = QSpinBox()
        self.nina_port_spinbox.setMinimum(1)
        self.nina_port_spinbox.setMaximum(65535)
        self.nina_port_spinbox.setValue(1888)
        self.nina_port_spinbox.setToolTip(
            "Port number for NINA's Advanced API.\n"
            "Default: 1888"
        )
        nina_port_layout.addWidget(nina_port_label)
        nina_port_layout.addWidget(self.nina_port_spinbox)
        nina_port_layout.addStretch()
        nina_layout.addLayout(nina_port_layout)

        # Test connection button
        nina_test_layout = QHBoxLayout()
        nina_test_spacer = QLabel("")
        nina_test_spacer.setMinimumWidth(120)
        self.nina_test_btn = QPushButton("Test Connection")
        self.nina_test_btn.setToolTip("Test the connection to NINA's Advanced API")
        self.nina_test_btn.clicked.connect(self._test_nina_connection)
        nina_test_layout.addWidget(nina_test_spacer)
        nina_test_layout.addWidget(self.nina_test_btn)
        nina_test_layout.addStretch()
        nina_layout.addLayout(nina_test_layout)

        nina_help = QLabel(
            "Configure the port used by NINA's Advanced API plugin.<br>"
            "Requires the <b>Advanced API</b> plugin to be installed and enabled in NINA."
        )
        nina_help.setWordWrap(True)
        themed_style(nina_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; }}")
        nina_layout.addWidget(nina_help)

        nina_layout.addStretch()
        self.integrations_stack.addWidget(nina_page)

        # Plate solving page - which solvers are tried, in what order
        from cosmoscollection.solving.PlateSolver import SOLVER_ORDERS
        solving_page = QWidget()
        solving_page_layout = QVBoxLayout(solving_page)

        solving_title = QLabel("Plate Solving")
        themed_style(solving_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        solving_page_layout.addWidget(solving_title)

        solver_order_layout = QHBoxLayout()
        solver_order_label = QLabel("Try solvers:")
        solver_order_label.setMinimumWidth(120)
        self.solver_order_combo = QComboBox()
        for value, label in SOLVER_ORDERS:
            self.solver_order_combo.addItem(label, value)
        solver_order_layout.addWidget(solver_order_label)
        solver_order_layout.addWidget(self.solver_order_combo)
        solver_order_layout.addStretch()
        solving_page_layout.addLayout(solver_order_layout)

        solving_help = QLabel(
            "Plate solving finds where an image is in the sky, so its objects can be annotated. "
            "Each solver is tried in turn until one solves the image.<br><br>"
            "<b>Built-in</b> needs nothing installed: it matches the image's stars with an online star "
            "catalog (Gaia, or Tycho-2 for wide fields) near the image's object. Usually seconds.<br>"
            "<b>ASTAP</b> works offline once installed (see its page), and doesn't need to know the object.<br>"
            "<b>Astrometry.net</b> is online and needs a free API key (see its page). It can take minutes."
        )
        solving_help.setWordWrap(True)
        solving_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(solving_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        solving_page_layout.addWidget(solving_help)

        solving_page_layout.addStretch()
        self.integrations_stack.addWidget(solving_page)

        # ASTAP page
        astap_page = QWidget()
        astap_page_layout = QVBoxLayout(astap_page)

        astap_title = QLabel("ASTAP")
        themed_style(astap_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        astap_page_layout.addWidget(astap_title)

        # ASTAP path setting
        astap_layout = QHBoxLayout()
        astap_label = QLabel("ASTAP Path:")
        astap_label.setMinimumWidth(120)
        self.astap_path_input = QLineEdit()
        self.astap_path_input.setPlaceholderText("Path to astap_cli executable (auto-detected if empty)")
        astap_browse_btn = QPushButton("Browse...")
        astap_browse_btn.setFixedWidth(80)
        astap_browse_btn.clicked.connect(self._browse_astap_path)
        astap_layout.addWidget(astap_label)
        astap_layout.addWidget(self.astap_path_input)
        astap_layout.addWidget(astap_browse_btn)
        astap_page_layout.addLayout(astap_layout)

        # ASTAP help text
        astap_help = QLabel(
            "ASTAP is a free, fast local plate solver. Download from: "
            f"<a href='https://www.hnsky.org/astap.htm' style='color: {COLORS['link']};'>hnsky.org/astap.htm</a><br>"
            "Point to <b>astap_cli.exe</b> (command-line version), not astap.exe (GUI). "
            "Leave empty to auto-detect."
        )
        astap_help.setOpenExternalLinks(True)
        astap_help.setWordWrap(True)
        astap_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(astap_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        astap_page_layout.addWidget(astap_help)

        astap_page_layout.addStretch()
        self.integrations_stack.addWidget(astap_page)

        # Astrometry.net page
        astrometry_page = QWidget()
        astrometry_page_layout = QVBoxLayout(astrometry_page)

        astrometry_title = QLabel("Astrometry.net")
        themed_style(astrometry_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        astrometry_page_layout.addWidget(astrometry_title)

        # Astrometry.net API key setting
        api_key_layout = QHBoxLayout()
        api_key_label = QLabel("API Key:")
        api_key_label.setMinimumWidth(120)
        self.astrometry_api_key_input = QLineEdit()
        self.astrometry_api_key_input.setPlaceholderText("Required for online plate solving")
        self.astrometry_api_key_input.setEchoMode(QLineEdit.Password)
        show_key_btn = QPushButton("Show")
        show_key_btn.setFixedWidth(50)
        show_key_btn.setCheckable(True)
        show_key_btn.clicked.connect(lambda checked: self.astrometry_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.astrometry_api_key_input)
        api_key_layout.addWidget(show_key_btn)
        astrometry_page_layout.addLayout(api_key_layout)

        # Test API key button
        astrometry_test_layout = QHBoxLayout()
        astrometry_test_spacer = QLabel("")
        astrometry_test_spacer.setMinimumWidth(120)
        self.astrometry_test_btn = QPushButton("Test API Key")
        self.astrometry_test_btn.setToolTip("Verify the entered Astrometry.net API key is valid")
        self.astrometry_test_btn.clicked.connect(self._test_astrometry_key)
        astrometry_test_layout.addWidget(astrometry_test_spacer)
        astrometry_test_layout.addWidget(self.astrometry_test_btn)
        astrometry_test_layout.addStretch()
        astrometry_page_layout.addLayout(astrometry_test_layout)

        # API key help text
        api_key_help = QLabel(
            "Get a free API key: Register at "
            f"<a href='https://nova.astrometry.net/' style='color: {COLORS['link']};'>nova.astrometry.net</a>, "
            "then find your key in My Account > API.<br>"
            "An API key is required for online plate solving (used when ASTAP fails or is unavailable)."
        )
        api_key_help.setOpenExternalLinks(True)
        api_key_help.setWordWrap(True)
        api_key_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(api_key_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        astrometry_page_layout.addWidget(api_key_help)

        astrometry_page_layout.addStretch()
        self.integrations_stack.addWidget(astrometry_page)

        # PixInsight and Siril pages (stacking handoff when a session is completed)
        self.processing_app_widgets = {}
        self.integrations_stack.addWidget(self._build_processing_app_page(
            "pixinsight", "PixInsight",
            "Path to PixInsight.exe (auto-detected if empty)",
            "When a session in the Session Manager is marked Completed, you can hand its frames to "
            "PixInsight's WeightedBatchPreprocessing (WBPP): the subs are linked into a workspace folder "
            "and WBPP opens with them loaded. Your original subs are never moved or changed."))
        self.integrations_stack.addWidget(self._build_processing_app_page(
            "siril", "Siril",
            "Path to siril-cli (auto-detected if empty)",
            "When a session in the Session Manager is marked Completed, you can stack it in Siril: "
            "the subs are linked into a workspace folder and a generated script calibrates, registers and "
            "stacks them per filter. Requires Siril 1.3.4 or newer (1.4+ for XISF subs). Download from "
            f"<a href='https://siril.org/download/' style='color: {COLORS['link']};'>siril.org</a>."))

        # OpenWeather page
        openweather_page = QWidget()
        openweather_page_layout = QVBoxLayout(openweather_page)

        openweather_title = QLabel("OpenWeather")
        themed_style(openweather_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        openweather_page_layout.addWidget(openweather_title)

        # Enable checkbox
        self.openweather_enabled_checkbox = QCheckBox("Enable OpenWeather Integration")
        self.openweather_enabled_checkbox.setToolTip(
            "When enabled (and an API key is provided below), OpenWeather forecast\n"
            "data is blended with Open-Meteo data to improve forecast accuracy."
        )
        openweather_page_layout.addWidget(self.openweather_enabled_checkbox)

        # OpenWeather API key setting
        openweather_key_layout = QHBoxLayout()
        openweather_key_label = QLabel("API Key:")
        openweather_key_label.setMinimumWidth(120)
        self.openweather_api_key_input = QLineEdit()
        self.openweather_api_key_input.setPlaceholderText("Required to enable this integration")
        self.openweather_api_key_input.setEchoMode(QLineEdit.Password)
        openweather_show_key_btn = QPushButton("Show")
        openweather_show_key_btn.setFixedWidth(50)
        openweather_show_key_btn.setCheckable(True)
        openweather_show_key_btn.clicked.connect(lambda checked: self.openweather_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        openweather_key_layout.addWidget(openweather_key_label)
        openweather_key_layout.addWidget(self.openweather_api_key_input)
        openweather_key_layout.addWidget(openweather_show_key_btn)
        openweather_page_layout.addLayout(openweather_key_layout)

        # Test API key button
        openweather_test_layout = QHBoxLayout()
        openweather_test_spacer = QLabel("")
        openweather_test_spacer.setMinimumWidth(120)
        self.openweather_test_btn = QPushButton("Test API Key")
        self.openweather_test_btn.setToolTip("Verify the entered OpenWeather API key is valid")
        self.openweather_test_btn.clicked.connect(self._test_openweather_key)
        openweather_test_layout.addWidget(openweather_test_spacer)
        openweather_test_layout.addWidget(self.openweather_test_btn)
        openweather_test_layout.addStretch()
        openweather_page_layout.addLayout(openweather_test_layout)

        # OpenWeather help text
        openweather_help = QLabel(
            "Get a free API key: Register at "
            f"<a href='https://openweathermap.org/api' style='color: {COLORS['link']};'>openweathermap.org/api</a>, "
            "then find your key under My API Keys.<br>"
            "When enabled, OpenWeather's forecast is averaged with Open-Meteo's data "
            "(cloud cover, temperature, humidity, wind, precipitation chance) to reduce single-source forecast bias."
        )
        openweather_help.setOpenExternalLinks(True)
        openweather_help.setWordWrap(True)
        openweather_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(openweather_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        openweather_page_layout.addWidget(openweather_help)

        openweather_page_layout.addStretch()
        self.integrations_stack.addWidget(openweather_page)

        # VisualCrossing page
        visualcrossing_page = QWidget()
        visualcrossing_page_layout = QVBoxLayout(visualcrossing_page)

        visualcrossing_title = QLabel("VisualCrossing")
        themed_style(visualcrossing_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        visualcrossing_page_layout.addWidget(visualcrossing_title)

        # Enable checkbox
        self.visualcrossing_enabled_checkbox = QCheckBox("Enable VisualCrossing Integration")
        self.visualcrossing_enabled_checkbox.setToolTip(
            "When enabled (and an API key is provided below), VisualCrossing forecast\n"
            "data is blended with Open-Meteo (and OpenWeather, if also enabled) data\n"
            "to improve forecast accuracy."
        )
        visualcrossing_page_layout.addWidget(self.visualcrossing_enabled_checkbox)

        # VisualCrossing API key setting
        visualcrossing_key_layout = QHBoxLayout()
        visualcrossing_key_label = QLabel("API Key:")
        visualcrossing_key_label.setMinimumWidth(120)
        self.visualcrossing_api_key_input = QLineEdit()
        self.visualcrossing_api_key_input.setPlaceholderText("Required to enable this integration")
        self.visualcrossing_api_key_input.setEchoMode(QLineEdit.Password)
        visualcrossing_show_key_btn = QPushButton("Show")
        visualcrossing_show_key_btn.setFixedWidth(50)
        visualcrossing_show_key_btn.setCheckable(True)
        visualcrossing_show_key_btn.clicked.connect(lambda checked: self.visualcrossing_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        visualcrossing_key_layout.addWidget(visualcrossing_key_label)
        visualcrossing_key_layout.addWidget(self.visualcrossing_api_key_input)
        visualcrossing_key_layout.addWidget(visualcrossing_show_key_btn)
        visualcrossing_page_layout.addLayout(visualcrossing_key_layout)

        # Test API key button
        visualcrossing_test_layout = QHBoxLayout()
        visualcrossing_test_spacer = QLabel("")
        visualcrossing_test_spacer.setMinimumWidth(120)
        self.visualcrossing_test_btn = QPushButton("Test API Key")
        self.visualcrossing_test_btn.setToolTip("Verify the entered VisualCrossing API key is valid")
        self.visualcrossing_test_btn.clicked.connect(self._test_visualcrossing_key)
        visualcrossing_test_layout.addWidget(visualcrossing_test_spacer)
        visualcrossing_test_layout.addWidget(self.visualcrossing_test_btn)
        visualcrossing_test_layout.addStretch()
        visualcrossing_page_layout.addLayout(visualcrossing_test_layout)

        # VisualCrossing help text
        visualcrossing_help = QLabel(
            "Get a free API key: Register at "
            f"<a href='https://www.visualcrossing.com/sign-up' style='color: {COLORS['link']};'>visualcrossing.com/sign-up</a>, "
            "then find your key on your Account page.<br>"
            "When enabled, VisualCrossing's forecast is averaged with Open-Meteo's data "
            "(cloud cover, temperature, humidity, wind, precipitation chance) to reduce single-source forecast bias. "
            "The free tier has a daily record budget, so avoid refreshing excessively."
        )
        visualcrossing_help.setOpenExternalLinks(True)
        visualcrossing_help.setWordWrap(True)
        visualcrossing_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(visualcrossing_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        visualcrossing_page_layout.addWidget(visualcrossing_help)

        visualcrossing_page_layout.addStretch()
        self.integrations_stack.addWidget(visualcrossing_page)

        # WeatherAPI page
        weatherapi_page = QWidget()
        weatherapi_page_layout = QVBoxLayout(weatherapi_page)

        weatherapi_title = QLabel("WeatherAPI")
        themed_style(weatherapi_title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        weatherapi_page_layout.addWidget(weatherapi_title)

        # Enable checkbox
        self.weatherapi_enabled_checkbox = QCheckBox("Enable WeatherAPI Integration")
        self.weatherapi_enabled_checkbox.setToolTip(
            "When enabled (and an API key is provided below), WeatherAPI forecast\n"
            "data is blended with Open-Meteo (and OpenWeather/VisualCrossing, if also\n"
            "enabled) data to improve forecast accuracy."
        )
        weatherapi_page_layout.addWidget(self.weatherapi_enabled_checkbox)

        # WeatherAPI API key setting
        weatherapi_key_layout = QHBoxLayout()
        weatherapi_key_label = QLabel("API Key:")
        weatherapi_key_label.setMinimumWidth(120)
        self.weatherapi_api_key_input = QLineEdit()
        self.weatherapi_api_key_input.setPlaceholderText("Required to enable this integration")
        self.weatherapi_api_key_input.setEchoMode(QLineEdit.Password)
        weatherapi_show_key_btn = QPushButton("Show")
        weatherapi_show_key_btn.setFixedWidth(50)
        weatherapi_show_key_btn.setCheckable(True)
        weatherapi_show_key_btn.clicked.connect(lambda checked: self.weatherapi_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        weatherapi_key_layout.addWidget(weatherapi_key_label)
        weatherapi_key_layout.addWidget(self.weatherapi_api_key_input)
        weatherapi_key_layout.addWidget(weatherapi_show_key_btn)
        weatherapi_page_layout.addLayout(weatherapi_key_layout)

        # Test API key button
        weatherapi_test_layout = QHBoxLayout()
        weatherapi_test_spacer = QLabel("")
        weatherapi_test_spacer.setMinimumWidth(120)
        self.weatherapi_test_btn = QPushButton("Test API Key")
        self.weatherapi_test_btn.setToolTip("Verify the entered WeatherAPI API key is valid")
        self.weatherapi_test_btn.clicked.connect(self._test_weatherapi_key)
        weatherapi_test_layout.addWidget(weatherapi_test_spacer)
        weatherapi_test_layout.addWidget(self.weatherapi_test_btn)
        weatherapi_test_layout.addStretch()
        weatherapi_page_layout.addLayout(weatherapi_test_layout)

        # WeatherAPI help text
        weatherapi_help = QLabel(
            "Get a free API key: Register at "
            f"<a href='https://www.weatherapi.com/signup.aspx' style='color: {COLORS['link']};'>weatherapi.com/signup.aspx</a>, "
            "then find your key on your Account page.<br>"
            "When enabled, WeatherAPI's forecast is averaged with Open-Meteo's data "
            "(cloud cover, temperature, humidity, wind, precipitation chance) to reduce single-source forecast bias. "
            "The free tier is limited to a 3-day forecast."
        )
        weatherapi_help.setOpenExternalLinks(True)
        weatherapi_help.setWordWrap(True)
        weatherapi_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(weatherapi_help, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        weatherapi_page_layout.addWidget(weatherapi_help)

        weatherapi_page_layout.addStretch()
        self.integrations_stack.addWidget(weatherapi_page)

        self.integrations_list.setCurrentRow(0)

        # Backup/Restore tab
        backup_restore_tab = QWidget()
        backup_restore_layout = QVBoxLayout(backup_restore_tab)

        # Backup group
        backup_group = QGroupBox("Backup")
        backup_group_layout = QVBoxLayout(backup_group)

        backup_description = QLabel(
            "Create a backup of all your user data including:\n"
            "• User image metadata\n"
            "• Target list\n"
            "• Telescope profiles\n"
            "• Equipment (cameras, eyepieces, barlows)\n"
            "• Collage projects\n"
            "• Observing sessions\n"
            "• Application settings (location, timezone)"
        )
        backup_description.setWordWrap(True)
        backup_description.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        backup_group_layout.addWidget(backup_description)

        backup_btn_layout = QHBoxLayout()
        self.backup_button = QPushButton("Backup Now...")
        self.backup_button.setToolTip("Save all user data to a backup file")
        self.backup_button.clicked.connect(self._perform_backup)
        backup_btn_layout.addWidget(self.backup_button)
        backup_btn_layout.addStretch()
        backup_group_layout.addLayout(backup_btn_layout)

        backup_restore_layout.addWidget(backup_group)

        # Restore group
        restore_group = QGroupBox("Restore")
        restore_group_layout = QVBoxLayout(restore_group)

        restore_description = QLabel(
            "Restore your user data from a previously created backup file.<br><br>"
            "<b>Warning:</b> Restoring will replace your current data with the backup data. "
            "It is recommended to create a backup of your current data first."
        )
        restore_description.setTextFormat(Qt.RichText)
        restore_description.setWordWrap(True)
        restore_description.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        restore_group_layout.addWidget(restore_description)

        restore_btn_layout = QHBoxLayout()
        self.restore_button = QPushButton("Restore from Backup...")
        self.restore_button.setToolTip("Restore user data from a backup file")
        self.restore_button.clicked.connect(self._perform_restore)
        restore_btn_layout.addWidget(self.restore_button)
        restore_btn_layout.addStretch()
        restore_group_layout.addLayout(restore_btn_layout)

        backup_restore_layout.addWidget(restore_group)
        backup_restore_layout.addStretch()

        # Theme tab
        self.theme_tab = ThemeSettingsTab()

        # Add tabs
        tab_widget.addTab(app_settings_scroll, "Application Settings")
        tab_widget.addTab(self.theme_tab, "Theme")
        tab_widget.addTab(integrations_tab, "Integrations")
        tab_widget.addTab(location_tab, "Location Manager")
        tab_widget.addTab(backup_restore_tab, "Backup && Restore")

        layout.addWidget(tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Standard dialog buttons
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_settings)
        # Don't set as default to prevent accidental saves when map picker closes
        # self.save_button.setDefault(True)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def _update_reminder_status(self):
        from cosmoscollection.core.AppUpdater import update_reminder_status
        status = update_reminder_status()
        self.update_reminder_label.setText(status or "")
        self.update_reminder_row.setVisible(bool(status))

    def _reset_update_reminders(self):
        from cosmoscollection.core.AppUpdater import reset_update_reminders
        reset_update_reminders()
        self._update_reminder_status()

    def _on_offline_mode_toggled(self, offline):
        """Gray out the settings that only matter online (their values are kept)."""
        for checkbox in (self.check_updates_checkbox, self.simbad_lookup_checkbox):
            checkbox.setEnabled(not offline)

    def _load_current_settings(self):
        """Load current settings from database and QSettings"""
        # Populate location list
        self._refresh_location_list()

        # Load UI preferences from QSettings
        try:
            settings = QSettings("CosmosCollection", "CosmosCollection")
            from cosmoscollection.core.OnlineLimits import OFFLINE_SETTING
            offline_mode = settings.value(OFFLINE_SETTING, False, type=bool)
            self.offline_mode_checkbox.setChecked(offline_mode)
            self._on_offline_mode_toggled(offline_mode)

            show_observer_location = settings.value("show_observer_location", True, type=bool)
            self.show_observer_location_checkbox.setChecked(show_observer_location)

            check_updates = settings.value("check_updates_on_startup", True, type=bool)
            self.check_updates_checkbox.setChecked(check_updates)

            minimize_to_tray = settings.value("minimize_to_tray", True, type=bool)
            self.minimize_to_tray_checkbox.setChecked(minimize_to_tray)

            simbad_lookup = settings.value("simbad_lookup_on_search_failure", True, type=bool)
            self.simbad_lookup_checkbox.setChecked(simbad_lookup)

            enable_logfile = settings.value("enable_logfile", False, type=bool)
            self.enable_logfile_checkbox.setChecked(enable_logfile)

            # Load time format setting
            time_format = settings.value("time_format", "12-hour", type=str)
            index = self.time_format_combo.findText(time_format)
            if index >= 0:
                self.time_format_combo.setCurrentIndex(index)

            # Load thread count setting
            import os
            default_threads = max(1, (os.cpu_count() or 4) - 2)
            thread_count = settings.value("max_threads", default_threads, type=int)
            self.thread_count_spinbox.setValue(thread_count)

            # Load thumbnail cache setting
            cache_thumbnails = settings.value("cache_thumbnails_to_disk", True, type=bool)
            self.cache_thumbnails_checkbox.setChecked(cache_thumbnails)

            # Load weather unit settings
            temp_unit = settings.value("temperature_unit", "Celsius", type=str)
            index = self.temp_unit_combo.findText(temp_unit)
            if index >= 0:
                self.temp_unit_combo.setCurrentIndex(index)

            wind_unit = settings.value("wind_speed_unit", "km/h", type=str)
            index = self.wind_unit_combo.findText(wind_unit)
            if index >= 0:
                self.wind_unit_combo.setCurrentIndex(index)

            precip_unit = settings.value("precip_visibility_unit", "Metric (mm, km)", type=str)
            index = self.precip_unit_combo.findText(precip_unit)
            if index >= 0:
                self.precip_unit_combo.setCurrentIndex(index)

            # Load plate solving settings
            from cosmoscollection.solving.PlateSolver import solver_order
            order_index = self.solver_order_combo.findData(",".join(solver_order()))
            self.solver_order_combo.setCurrentIndex(max(0, order_index))
            astap_path = settings.value("astap_path", "", type=str)
            self.astap_path_input.setText(astap_path)

            astrometry_api_key = settings.value("astrometry_api_key", "", type=str)
            self.astrometry_api_key_input.setText(astrometry_api_key)

            # Load NINA settings
            nina_enabled = settings.value("nina_integration_enabled", False, type=bool)
            self.nina_enabled_checkbox.setChecked(nina_enabled)
            nina_ip = settings.value("nina_api_host", "localhost", type=str)
            self.nina_ip_input.setText(nina_ip)
            nina_port = settings.value("nina_api_port", 1888, type=int)
            self.nina_port_spinbox.setValue(nina_port)

            # Load PixInsight / Siril settings
            for app, widgets in self.processing_app_widgets.items():
                widgets["enabled"].setChecked(settings.value(f"{app}_integration_enabled", False, type=bool))
                widgets["path"].setText(settings.value(f"{app}_path", "", type=str))
                widgets["workspace"].setText(settings.value(f"{app}_workspace_dir", "", type=str))

            # Load OpenWeather settings
            openweather_enabled = settings.value("openweather_integration_enabled", False, type=bool)
            self.openweather_enabled_checkbox.setChecked(openweather_enabled)
            openweather_api_key = settings.value("openweather_api_key", "", type=str)
            self.openweather_api_key_input.setText(openweather_api_key)

            # Load VisualCrossing settings
            visualcrossing_enabled = settings.value("visualcrossing_integration_enabled", False, type=bool)
            self.visualcrossing_enabled_checkbox.setChecked(visualcrossing_enabled)
            visualcrossing_api_key = settings.value("visualcrossing_api_key", "", type=str)
            self.visualcrossing_api_key_input.setText(visualcrossing_api_key)

            # Load WeatherAPI settings
            weatherapi_enabled = settings.value("weatherapi_integration_enabled", False, type=bool)
            self.weatherapi_enabled_checkbox.setChecked(weatherapi_enabled)
            weatherapi_api_key = settings.value("weatherapi_api_key", "", type=str)
            self.weatherapi_api_key_input.setText(weatherapi_api_key)

            # Load theme settings
            self.theme_tab.load(load_theme_settings())

        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            
    def _refresh_location_list(self, keep_selection_id=None):
        """Refresh the location list widget from database

        Args:
            keep_selection_id: Re-select this location without reloading the
                form (so unsaved edits survive). By default the active
                location is selected and loaded into the form.
        """
        if keep_selection_id is not None:
            self.location_list.blockSignals(True)
        self.location_list.clear()
        select_item = None
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, location_lat, location_lon, location_name, timezone, is_active,
                           COALESCE(horizon_points, '') != ''
                    FROM usersettings ORDER BY id
                """)
                rows = cursor.fetchall()
                for row in rows:
                    loc_id, lat, lon, name, tz, is_active, has_horizon = row
                    if name:
                        display = name
                    else:
                        display = f"{lat:.4f}, {lon:.4f}"
                    if is_active:
                        display = "(Active) " + display
                    if tz:
                        display += f" [{tz}]"
                    if has_horizon:
                        display += " ⛰"
                    item = QListWidgetItem(display)
                    item.setData(Qt.UserRole, loc_id)
                    if has_horizon:
                        item.setToolTip("Has a custom horizon")
                    if is_active:
                        item.setBackground(QColor(tint(COLORS['accent'], 0.35)))
                    if (loc_id == keep_selection_id) if keep_selection_id is not None else is_active:
                        select_item = item
                    self.location_list.addItem(item)
            if select_item:
                self.location_list.setCurrentItem(select_item)
        except Exception as e:
            logger.error(f"Error loading location list: {str(e)}")
        finally:
            self.location_list.blockSignals(False)

    def _on_location_selected(self, current, previous):
        """Handle location list selection - populate form for editing"""
        if not current:
            return
        loc_id = current.data(Qt.UserRole)
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT location_lat, location_lon, location_name, timezone, is_active, horizon_points, horizon_name
                    FROM usersettings WHERE id = ?
                """, (loc_id,))
                row = cursor.fetchone()
                if row:
                    lat, lon, name, tz, is_active, horizon_points, horizon_name = row
                    self.pending_horizon = None
                    try:
                        from cosmoscollection.planning.HorizonProfile import HorizonProfile
                        self._show_horizon_status(HorizonProfile.from_json(horizon_points, name=horizon_name))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ignoring unreadable custom horizon for location {loc_id}: {e}")
                        self._show_horizon_status(None)
                    self.set_active_btn.setEnabled(not is_active)
                    self.latitude_input.setText(str(lat) if lat is not None else "")
                    self.longitude_input.setText(str(lon) if lon is not None else "")
                    self.location_name_input.setText(name if name else "")
                    if tz:
                        index = self.timezone_combo.findText(tz)
                        if index >= 0:
                            self.timezone_combo.setCurrentIndex(index)
                        else:
                            self.timezone_combo.setEditText(tz)
                    else:
                        self.timezone_combo.setCurrentIndex(0)
                    self.editing_location_id = loc_id
                    self.save_location_btn.setText("Update Location")
        except Exception as e:
            logger.error(f"Error loading location details: {str(e)}")

    def _clear_location_form(self):
        """Clear the location form and reset to add mode"""
        self.latitude_input.clear()
        self.longitude_input.clear()
        self.location_name_input.clear()
        self.timezone_combo.setCurrentIndex(0)
        self.editing_location_id = None
        self.pending_horizon = None
        self._show_horizon_status(None)
        self.save_location_btn.setText("Save Location")
        # Reset the current item too, not just the selection, so clicking the
        # same location again fires currentItemChanged and reloads the form
        self.location_list.setCurrentRow(-1)
        self.location_list.clearSelection()

    def _show_horizon_status(self, profile, pending=False):
        """Show a location's custom horizon (or None) in the Location Details form"""
        if profile is None:
            self.horizon_status_label.setText("None")
            self.horizon_status_label.setToolTip("")
        else:
            text = profile.summary()
            if pending:
                text += " (saved with the location)"
            self.horizon_status_label.setText(text)
            self.horizon_status_label.setToolTip(text)
        self.clear_horizon_btn.setEnabled(profile is not None)
        self.horizon_preview.set_profile(profile)
        self.horizon_preview.setVisible(profile is not None)

    def _import_horizon(self):
        """Import a NINA/Stellarium horizon file for the location in the form.

        An existing location is updated immediately; a new location keeps the
        horizon until Save Location."""
        from cosmoscollection.planning.HorizonProfile import HorizonParseError, parse_horizon_file

        settings = QSettings("CosmosCollection", "CosmosCollection")
        start_dir = settings.value("last_horizon_dir", "", type=str)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Custom Horizon",
            start_dir,
            "Horizon Files (*.hrz *.ini *.zip *.txt *.csv);;"
            "NINA Horizon (*.hrz);;"
            "Stellarium Landscape (*.ini *.zip);;"
            "All Files (*.*)"
        )
        if not file_path:
            return
        settings.setValue("last_horizon_dir", os.path.dirname(file_path))

        try:
            profile = parse_horizon_file(file_path)
        except HorizonParseError as e:
            QMessageBox.warning(self, "Invalid Horizon File", str(e))
            return

        if self.editing_location_id:
            self._store_horizon(self.editing_location_id, profile)
        else:
            self.pending_horizon = profile
            self._show_horizon_status(profile, pending=True)

    def _clear_horizon(self):
        """Remove the custom horizon from the location in the form"""
        if not self.editing_location_id:
            self.pending_horizon = None
            self._show_horizon_status(None)
            return

        reply = QMessageBox.question(self, "Remove Custom Horizon",
            "Remove the custom horizon from this location?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._store_horizon(self.editing_location_id, None)

    def _store_horizon(self, loc_id, profile):
        """Save (or with profile=None, remove) a location's custom horizon"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE usersettings SET horizon_points = ?, horizon_name = ? WHERE id = ?",
                    (profile.to_json() if profile else None, profile.name if profile else None, loc_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving custom horizon: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save custom horizon: {str(e)}")
            return

        self._show_horizon_status(profile)
        self._refresh_location_list(keep_selection_id=loc_id)
        logger.debug(f"Custom horizon for location {loc_id}: {profile.summary() if profile else 'removed'}")

    def _save_location(self):
        """Save or update a location in the database"""
        try:
            lat_text = self.latitude_input.text().strip()
            lon_text = self.longitude_input.text().strip()

            if not lat_text or not lon_text:
                QMessageBox.warning(self, "Missing Information",
                    "Please enter both latitude and longitude.")
                return

            lat = float(lat_text)
            lon = float(lon_text)

            if not (-90 <= lat <= 90):
                QMessageBox.warning(self, "Invalid Latitude",
                    "Latitude must be between -90 and 90 degrees.")
                return

            if not (-180 <= lon <= 180):
                QMessageBox.warning(self, "Invalid Longitude",
                    "Longitude must be between -180 and 180 degrees.")
                return

            location_name = self.location_name_input.text().strip() or None
            timezone = self.timezone_combo.currentText().strip() or "America/New_York"

            try:
                import pytz
                pytz.timezone(timezone)
            except Exception:
                QMessageBox.warning(self, "Invalid Timezone",
                    f"'{timezone}' is not a valid timezone identifier.")
                return

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if self.editing_location_id:
                    # Update existing location
                    cursor.execute("""
                        UPDATE usersettings SET location_lat = ?, location_lon = ?, location_name = ?, timezone = ?
                        WHERE id = ?
                    """, (lat, lon, location_name, timezone, self.editing_location_id))
                else:
                    # Insert new location, with any horizon imported before saving
                    horizon = self.pending_horizon
                    cursor.execute("""
                        INSERT INTO usersettings (location_lat, location_lon, location_name, timezone, is_active,
                                                  horizon_points, horizon_name)
                        VALUES (?, ?, ?, ?, 0, ?, ?)
                    """, (lat, lon, location_name, timezone,
                          horizon.to_json() if horizon else None, horizon.name if horizon else None))

                    # If it's the only location, auto-set as active
                    cursor.execute("SELECT COUNT(*) FROM usersettings")
                    count = cursor.fetchone()[0]
                    if count == 1:
                        cursor.execute("UPDATE usersettings SET is_active = 1 WHERE id = (SELECT id FROM usersettings LIMIT 1)")

                conn.commit()

            self._clear_location_form()
            self._refresh_location_list()
            logger.debug(f"Location saved: lat={lat}, lon={lon}, name={location_name}, tz={timezone}")

        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                "Please enter valid numeric values for latitude and longitude.")
        except Exception as e:
            logger.error(f"Error saving location: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save location: {str(e)}")

    def _delete_location(self):
        """Delete the selected location"""
        current = self.location_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a location to delete.")
            return

        loc_id = current.data(Qt.UserRole)
        reply = QMessageBox.question(self, "Confirm Delete",
            "Are you sure you want to delete this location?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Check if we're deleting the active location
                cursor.execute("SELECT is_active FROM usersettings WHERE id = ?", (loc_id,))
                was_active = cursor.fetchone()[0]

                cursor.execute("DELETE FROM usersettings WHERE id = ?", (loc_id,))

                if was_active:
                    # Set the most recent remaining location as active
                    cursor.execute("""
                        UPDATE usersettings SET is_active = 1
                        WHERE id = (SELECT id FROM usersettings ORDER BY id DESC LIMIT 1)
                    """)

                conn.commit()

            self._clear_location_form()
            self._refresh_location_list()
        except Exception as e:
            logger.error(f"Error deleting location: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to delete location: {str(e)}")

    def _set_active_location(self):
        """Set the selected location as the active one"""
        current = self.location_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a location to set as active.")
            return

        loc_id = current.data(Qt.UserRole)
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE usersettings SET is_active = 0 WHERE is_active = 1")
                cursor.execute("UPDATE usersettings SET is_active = 1 WHERE id = ?", (loc_id,))
                conn.commit()

            self._refresh_location_list()
        except Exception as e:
            logger.error(f"Error setting active location: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to set active location: {str(e)}")

    def _open_log_folder(self):
        """Open the folder where log files are stored"""
        # Route through UrlOpener instead of calling QDesktopServices directly:
        # on Linux (e.g. CachyOS), QDesktopServices.openUrl() shells out to
        # xdg-open, and in a PyInstaller-frozen build that subprocess inherits
        # our bundled LD_LIBRARY_PATH, which makes it fail to launch the file
        # manager. UrlOpener retries with gio/xdg-open under a cleaned
        # environment first (same fix already used for opening web links).
        from cosmoscollection.core.UrlOpener import open_url
        log_dir = ResourceManager.get_data_dir()
        if not open_url(QUrl.fromLocalFile(str(log_dir))):
            logger.warning(f"Failed to open log folder: {log_dir}")
            QMessageBox.warning(
                self, "Open Log Folder",
                f"Could not open the log folder automatically.\n\nLocation:\n{log_dir}"
            )

    def _save_settings(self):
        """Save application settings (QSettings only - locations are saved separately)"""
        try:
            settings = QSettings("CosmosCollection", "CosmosCollection")
            from cosmoscollection.core.OnlineLimits import OFFLINE_SETTING
            settings.setValue(OFFLINE_SETTING, self.offline_mode_checkbox.isChecked())
            settings.setValue("show_observer_location", self.show_observer_location_checkbox.isChecked())
            settings.setValue("check_updates_on_startup", self.check_updates_checkbox.isChecked())
            settings.setValue("minimize_to_tray", self.minimize_to_tray_checkbox.isChecked())
            settings.setValue("simbad_lookup_on_search_failure", self.simbad_lookup_checkbox.isChecked())
            settings.setValue("enable_logfile", self.enable_logfile_checkbox.isChecked())
            settings.setValue("time_format", self.time_format_combo.currentText())
            settings.setValue("max_threads", self.thread_count_spinbox.value())
            settings.setValue("cache_thumbnails_to_disk", self.cache_thumbnails_checkbox.isChecked())

            # Save weather unit settings
            settings.setValue("temperature_unit", self.temp_unit_combo.currentText())
            settings.setValue("wind_speed_unit", self.wind_unit_combo.currentText())
            settings.setValue("precip_visibility_unit", self.precip_unit_combo.currentText())

            # Save plate solving settings
            settings.setValue("plate_solver_order", self.solver_order_combo.currentData())
            settings.setValue("astap_path", self.astap_path_input.text().strip())
            settings.setValue("astrometry_api_key", self.astrometry_api_key_input.text().strip())

            # Save NINA settings
            settings.setValue("nina_integration_enabled", self.nina_enabled_checkbox.isChecked())
            settings.setValue("nina_api_host", self.nina_ip_input.text().strip() or "localhost")
            settings.setValue("nina_api_port", self.nina_port_spinbox.value())

            # Save PixInsight / Siril settings
            for app, widgets in self.processing_app_widgets.items():
                settings.setValue(f"{app}_integration_enabled", widgets["enabled"].isChecked())
                settings.setValue(f"{app}_path", widgets["path"].text().strip())
                settings.setValue(f"{app}_workspace_dir", widgets["workspace"].text().strip())

            # Save OpenWeather settings
            settings.setValue("openweather_integration_enabled", self.openweather_enabled_checkbox.isChecked())
            settings.setValue("openweather_api_key", self.openweather_api_key_input.text().strip())

            # Save VisualCrossing settings
            settings.setValue("visualcrossing_integration_enabled", self.visualcrossing_enabled_checkbox.isChecked())
            settings.setValue("visualcrossing_api_key", self.visualcrossing_api_key_input.text().strip())

            # Save WeatherAPI settings
            settings.setValue("weatherapi_integration_enabled", self.weatherapi_enabled_checkbox.isChecked())
            settings.setValue("weatherapi_api_key", self.weatherapi_api_key_input.text().strip())

            # Save and apply the theme only if it changed, to avoid restyling the app needlessly
            theme_settings = self.theme_tab.theme_settings()
            if theme_settings != load_theme_settings():
                save_theme_settings(theme_settings)
                apply_theme(QApplication.instance(), theme_settings)

            self.accept()

        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def _open_map_picker(self):
        """Open map dialog to select location coordinates"""
        from cosmoscollection.core.OnlineLimits import is_offline, notify_offline
        if is_offline():
            notify_offline(self, "The map")
            return
        try:
            # Get current coordinates as starting point (or default to NYC)
            current_lat = 40.7128
            current_lon = -74.0060

            try:
                lat_text = self.latitude_input.text().strip()
                lon_text = self.longitude_input.text().strip()
                if lat_text and lon_text:
                    current_lat = float(lat_text)
                    current_lon = float(lon_text)
            except ValueError:
                # Use default coordinates if current values are invalid
                pass

            # Open dialog - parent=None to avoid event propagation issues between nested modal dialogs
            dialog = MapLocationPickerDialog(current_lat, current_lon, parent=None)
            result_code = dialog.exec()

            if result_code == QDialog.Accepted:
                result = dialog.get_selected_coordinates()
                if result:
                    lat, lon, location_name = result
                    self.latitude_input.setText(f"{lat:.6f}")
                    self.longitude_input.setText(f"{lon:.6f}")
                    if location_name:
                        self.location_name_input.setText(location_name)

        except Exception as e:
            logger.error(f"Error opening map picker: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Error",
                f"Failed to open map picker: {str(e)}\n\n"
                "Please enter coordinates manually.")

    def _build_processing_app_page(self, app, label, path_placeholder, help_html):
        """Integrations page for a stacking app (PixInsight / Siril): enable
        checkbox, executable path, test button and default workspace folder.
        Widgets are kept in self.processing_app_widgets[app] for load/save."""
        page = QWidget()
        page_layout = QVBoxLayout(page)

        title = QLabel(label)
        themed_style(title, lambda: f"font-weight: bold; font-size: {font_size(11)};")
        page_layout.addWidget(title)

        enabled_checkbox = QCheckBox(f"Enable {label} Integration")
        enabled_checkbox.setToolTip(f"Offer {label} as a next step when a session is marked Completed.")
        page_layout.addWidget(enabled_checkbox)

        path_layout = QHBoxLayout()
        path_label = QLabel("Executable:")
        path_label.setMinimumWidth(120)
        path_input = QLineEdit()
        path_input.setPlaceholderText(path_placeholder)
        path_browse_btn = QPushButton("Browse...")
        path_browse_btn.setFixedWidth(80)
        path_browse_btn.clicked.connect(lambda: self._browse_processing_app_path(app, label))
        path_layout.addWidget(path_label)
        path_layout.addWidget(path_input)
        path_layout.addWidget(path_browse_btn)
        page_layout.addLayout(path_layout)

        workspace_layout = QHBoxLayout()
        workspace_label = QLabel("Workspace Folder:")
        workspace_label.setMinimumWidth(120)
        workspace_input = QLineEdit()
        workspace_input.setPlaceholderText("Default: next to the session's lights folder")
        workspace_input.setToolTip("Each completed session gets its own subfolder here.")
        workspace_browse_btn = QPushButton("Browse...")
        workspace_browse_btn.setFixedWidth(80)
        workspace_browse_btn.clicked.connect(lambda: self._browse_processing_workspace(app))
        workspace_layout.addWidget(workspace_label)
        workspace_layout.addWidget(workspace_input)
        workspace_layout.addWidget(workspace_browse_btn)
        page_layout.addLayout(workspace_layout)

        test_layout = QHBoxLayout()
        test_spacer = QLabel("")
        test_spacer.setMinimumWidth(120)
        test_btn = QPushButton("Test")
        test_btn.setToolTip(f"Check that {label} can be found")
        test_btn.clicked.connect(lambda: self._test_processing_app(app, label))
        test_layout.addWidget(test_spacer)
        test_layout.addWidget(test_btn)
        test_layout.addStretch()
        page_layout.addLayout(test_layout)

        help_label = QLabel(help_html)
        help_label.setOpenExternalLinks(True)
        help_label.setWordWrap(True)
        help_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        themed_style(help_label, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; margin-left: 120px; }}")
        page_layout.addWidget(help_label)

        page_layout.addStretch()
        self.processing_app_widgets[app] = {
            "enabled": enabled_checkbox, "path": path_input, "workspace": workspace_input,
        }
        return page

    def _browse_processing_app_path(self, app, label):
        path_input = self.processing_app_widgets[app]["path"]
        start_dir = os.path.dirname(path_input.text().strip()) or (
            "C:/Program Files" if sys.platform == 'win32' else "/usr/bin")
        filter_str = "Executable Files (*.exe);;All Files (*.*)" if sys.platform == 'win32' else "All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {label} Executable", start_dir, filter_str)
        if file_path:
            path_input.setText(os.path.normpath(file_path))

    def _browse_processing_workspace(self, app):
        workspace_input = self.processing_app_widgets[app]["workspace"]
        folder = QFileDialog.getExistingDirectory(
            self, "Select Default Workspace Folder", workspace_input.text().strip() or os.path.expanduser("~"))
        if folder:
            workspace_input.setText(os.path.normpath(folder))

    def _test_processing_app(self, app, label):
        from cosmoscollection.sessions import ProcessingHandoff
        configured = self.processing_app_widgets[app]["path"].text().strip()
        if app == "siril":
            exe = ProcessingHandoff.find_siril_cli(configured)
            version = ProcessingHandoff.siril_version(exe) if exe else None
            if not exe:
                QMessageBox.warning(self, label, "siril-cli wasn't found. Set its path above.")
            elif not version:
                QMessageBox.warning(self, label, f"Found {exe}, but couldn't read its version.")
            elif version < ProcessingHandoff.SIRIL_MIN_VERSION:
                QMessageBox.warning(self, label, f"Found Siril {'.'.join(map(str, version))} at:\n{exe}\n\n"
                                                 "Version 1.3.4 or newer is required.")
            else:
                QMessageBox.information(self, label, f"Found Siril {'.'.join(map(str, version))} at:\n{exe}")
        else:
            exe = ProcessingHandoff.find_pixinsight(configured)
            wbpp = ProcessingHandoff.find_wbpp_script(exe) if exe else None
            if not exe:
                QMessageBox.warning(self, label, "PixInsight wasn't found. Set its path above.")
            elif not wbpp:
                QMessageBox.warning(self, label, f"Found {exe}, but not the WBPP script in that installation.")
            else:
                QMessageBox.information(self, label, f"Found PixInsight at:\n{exe}\n\nWBPP:\n{wbpp}")

    def _browse_astap_path(self):
        """Browse for ASTAP executable"""
        import sys
        import os
        if sys.platform == 'win32':
            filter_str = "ASTAP CLI (astap_cli.exe);;Executable Files (*.exe);;All Files (*.*)"
            # Start in ASTAP folder if it exists
            if os.path.isdir("C:/Program Files/astap"):
                start_dir = "C:/Program Files/astap"
            elif os.path.isdir("C:/Program Files (x86)/astap"):
                start_dir = "C:/Program Files (x86)/astap"
            else:
                start_dir = "C:/Program Files"
        else:
            filter_str = "All Files (*)"
            start_dir = "/usr/bin"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASTAP CLI Executable (astap_cli.exe)",
            start_dir,
            filter_str
        )
        if file_path:
            self.astap_path_input.setText(file_path)

    def _test_nina_connection(self):
        """Test the connection to NINA's Advanced API"""
        nina_host = self.nina_ip_input.text().strip() or "localhost"
        nina_port = self.nina_port_spinbox.value()

        try:
            self.nina_test_btn.setEnabled(False)
            self.nina_test_btn.setText("Testing...")
            QApplication.processEvents()

            success, message, version = NINAIntegration.test_connection(nina_host, nina_port)

            if success:
                QMessageBox.information(self, "Connection Successful", message)
            else:
                QMessageBox.warning(self, "Connection Failed", message)

        finally:
            self.nina_test_btn.setEnabled(True)
            self.nina_test_btn.setText("Test Connection")

    def _test_astrometry_key(self):
        """Test that the entered Astrometry.net API key is valid"""
        api_key = self.astrometry_api_key_input.text().strip()

        try:
            self.astrometry_test_btn.setEnabled(False)
            self.astrometry_test_btn.setText("Testing...")
            QApplication.processEvents()

            from cosmoscollection.solving.PlateSolver import test_astrometry_key
            success, message = test_astrometry_key(api_key)

            if success:
                QMessageBox.information(self, "Connection Successful", message)
            else:
                QMessageBox.warning(self, "Connection Failed", message)

        finally:
            self.astrometry_test_btn.setEnabled(True)
            self.astrometry_test_btn.setText("Test API Key")

    def _test_openweather_key(self):
        """Test that the entered OpenWeather API key is valid"""
        api_key = self.openweather_api_key_input.text().strip()

        try:
            self.openweather_test_btn.setEnabled(False)
            self.openweather_test_btn.setText("Testing...")
            QApplication.processEvents()

            from cosmoscollection.planning.WeatherForecast import test_openweather_key
            success, message = test_openweather_key(api_key)

            if success:
                QMessageBox.information(self, "Connection Successful", message)
            else:
                QMessageBox.warning(self, "Connection Failed", message)

        finally:
            self.openweather_test_btn.setEnabled(True)
            self.openweather_test_btn.setText("Test API Key")

    def _test_visualcrossing_key(self):
        """Test that the entered VisualCrossing API key is valid"""
        api_key = self.visualcrossing_api_key_input.text().strip()

        try:
            self.visualcrossing_test_btn.setEnabled(False)
            self.visualcrossing_test_btn.setText("Testing...")
            QApplication.processEvents()

            from cosmoscollection.planning.WeatherForecast import test_visualcrossing_key
            success, message = test_visualcrossing_key(api_key)

            if success:
                QMessageBox.information(self, "Connection Successful", message)
            else:
                QMessageBox.warning(self, "Connection Failed", message)

        finally:
            self.visualcrossing_test_btn.setEnabled(True)
            self.visualcrossing_test_btn.setText("Test API Key")

    def _test_weatherapi_key(self):
        """Test that the entered WeatherAPI API key is valid"""
        api_key = self.weatherapi_api_key_input.text().strip()

        try:
            self.weatherapi_test_btn.setEnabled(False)
            self.weatherapi_test_btn.setText("Testing...")
            QApplication.processEvents()

            from cosmoscollection.planning.WeatherForecast import test_weatherapi_key
            success, message = test_weatherapi_key(api_key)

            if success:
                QMessageBox.information(self, "Connection Successful", message)
            else:
                QMessageBox.warning(self, "Connection Failed", message)

        finally:
            self.weatherapi_test_btn.setEnabled(True)
            self.weatherapi_test_btn.setText("Test API Key")

    def _perform_backup(self):
        """Perform a backup of all user data to a JSON file"""
        import json
        from datetime import datetime

        # Get save file path from user
        default_filename = f"CosmosCollection_Backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Backup File",
            default_filename,
            "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Collect all user data from database
            backup_data = {
                'backup_version': 1,
                'backup_date': datetime.now().isoformat(),
                'tables': {}
            }

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                for table in BACKUP_TABLES:
                    try:
                        cursor.execute(f"SELECT * FROM {table}")
                    except sqlite3.OperationalError:
                        continue  # Table not created yet on this install
                    backup_data['tables'][table] = {
                        'columns': [description[0] for description in cursor.description],
                        'rows': [list(row) for row in cursor.fetchall()]
                    }

            # Backup QSettings (only JSON-serializable values)
            settings = QSettings("CosmosCollection", "CosmosCollection")
            qsettings_data = {}
            for key in settings.allKeys():
                value = settings.value(key)
                # Only include basic JSON-serializable types
                if isinstance(value, (str, int, float, bool, type(None))):
                    qsettings_data[key] = value
            backup_data['qsettings'] = qsettings_data

            # Write the JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)

            QMessageBox.information(
                self,
                "Backup Complete",
                f"Backup created successfully!\n\nLocation: {file_path}"
            )
            logger.info(f"Backup completed successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Backup Error", f"Failed to create backup: {str(e)}")

    def _perform_restore(self):
        """Restore user data from a backup JSON file"""
        import json

        # Confirm with user
        confirm = QMessageBox.warning(
            self,
            "Confirm Restore",
            "Restoring from a backup will replace your current data.\n\n"
            "It is strongly recommended to create a backup of your current data first.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        # Get backup file from user
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backup File to Restore",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Validate backup file
            if 'backup_version' not in backup_data or 'tables' not in backup_data:
                QMessageBox.critical(self, "Invalid Backup", "This file does not appear to be a valid Cosmos Collection backup.")
                return

            # Verify backup version
            if backup_data.get('backup_version', 0) > 1:
                QMessageBox.warning(
                    self,
                    "Newer Backup Version",
                    "This backup was created with a newer version of Cosmos Collection.\n"
                    "Some data may not be restored correctly."
                )

            # Restore database tables
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                try:
                    for table in BACKUP_TABLES:
                        if table in backup_data['tables']:
                            self._restore_table(cursor, table, backup_data['tables'][table])
                    conn.commit()
                except Exception:
                    # Leave the current data intact rather than half-restored
                    conn.rollback()
                    raise

            # Restore QSettings if present
            if 'qsettings' in backup_data:
                settings = QSettings("CosmosCollection", "CosmosCollection")
                for key, value in backup_data['qsettings'].items():
                    settings.setValue(key, value)

            # Reload the settings dialog with restored data
            self._load_current_settings()

            QMessageBox.information(
                self,
                "Restore Complete",
                f"Backup restored successfully from:\n{file_path}\n\n"
                "Please restart the application for all changes to take effect."
            )
            logger.info(f"Backup restored successfully from {file_path}")

            # Check that every restored image still points at a real file on disk.
            # A backup made on another machine (or a moved/renamed image folder) can
            # leave userimages.image_path pointing nowhere; give the user a chance to
            # fix that right away instead of discovering broken thumbnails later.
            invalid_images = self._find_invalid_images()
            if invalid_images:
                fixup_dialog = RestoreImageFixupDialog(invalid_images, self.db_manager, self)
                fixup_dialog.exec()

        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Restore Error", f"Failed to restore backup: {str(e)}")

    @staticmethod
    def _restore_table(cursor, table, table_data):
        """Replace a table's rows with those from a backup.

        Only columns that exist in the current schema are restored, so backups
        from other versions still load and column names from the file are never
        interpolated into SQL unchecked."""
        cursor.execute(f"PRAGMA table_info({table})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if not existing_columns:
            logger.warning(f"Skipping restore of {table}: table does not exist")
            return

        columns = table_data['columns']
        keep = [i for i, column in enumerate(columns) if column in existing_columns]
        skipped = [column for column in columns if column not in existing_columns]
        if skipped:
            logger.warning(f"Restoring {table} without unknown columns: {skipped}")

        cursor.execute(f"DELETE FROM {table}")
        if not keep:
            return
        column_list = ", ".join(f'"{columns[i]}"' for i in keep)
        placeholders = ", ".join("?" for _ in keep)
        cursor.executemany(
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})",
            ([row[i] for i in keep] for row in table_data['rows'])
        )

    def _find_invalid_images(self):
        """Return restored userimages rows whose image_path no longer exists on disk.

        Each entry is a dict: {'id': userimage id, 'path': last known path, 'name': DSO name}.
        """
        invalid = []
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ui.id, ui.image_path,
                           COALESCE(GROUP_CONCAT(DISTINCT c.catalogue || ' ' || c.designation), 'Unknown') AS name
                    FROM userimages ui
                    LEFT JOIN dsodetail d ON d.id = ui.dsodetailid
                    LEFT JOIN cataloguenr c ON c.dsodetailid = d.id
                    WHERE ui.image_path IS NOT NULL AND ui.image_path != ''
                    GROUP BY ui.id
                """)
                rows = cursor.fetchall()
            for row_id, path, name in rows:
                if not os.path.exists(path):
                    invalid.append({'id': row_id, 'path': path, 'name': name})
        except Exception as e:
            logger.error(f"Error checking restored image paths: {str(e)}", exc_info=True)
        return invalid


# --- Folder Scan Worker (for restore image fixup) ---
class FolderScanWorker(QThread):
    """Recursively walks a folder tree in the background, indexing files by lowercase basename.

    Each directory is scanned as its own task on a thread pool, so many directories are
    listed concurrently instead of one os.walk() step at a time. This mainly pays off on
    network shares or spinning disks where each directory listing has real latency; the
    threads release the GIL while blocked on that I/O, so they run in parallel."""

    progress = Signal(int)  # number of files scanned so far
    finished_scan = Signal(dict)  # {lowercase basename: [full paths]}

    def __init__(self, root_folder, parent=None, max_workers=None):
        super().__init__(parent)
        self.root_folder = root_folder
        self.max_workers = max_workers or max(1, (os.cpu_count() or 4) - 2)
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def _scan_one_dir(self, path):
        """List a single directory (non-recursive). Returns (files, subdirs)."""
        files, subdirs = [], []
        if self._cancel_event.is_set():
            return files, subdirs
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if self._cancel_event.is_set():
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            files.append(entry.path)
                    except OSError:
                        continue  # e.g. broken symlink/permission race - skip that entry
        except OSError as e:
            logger.error(f"Error scanning directory {path}: {str(e)}")
        return files, subdirs

    def run(self):
        basename_map = {}
        lock = threading.Lock()
        count = 0
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                pending = {executor.submit(self._scan_one_dir, self.root_folder)}
                while pending and not self._cancel_event.is_set():
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        files, subdirs = future.result()
                        with lock:
                            for file_path in files:
                                key = os.path.basename(file_path).lower()
                                basename_map.setdefault(key, []).append(file_path)
                            count += len(files)
                        if not self._cancel_event.is_set():
                            for subdir in subdirs:
                                pending.add(executor.submit(self._scan_one_dir, subdir))
                    self.progress.emit(count)
        except Exception as e:
            logger.error(f"Error scanning folder {self.root_folder}: {str(e)}", exc_info=True)
        self.progress.emit(count)
        self.finished_scan.emit(basename_map)


# --- Scan Match Confirmation Dialog (for restore image fixup) ---
class ScanMatchConfirmationDialog(QDialog):
    """Lets the user review and approve filename matches found by a folder scan before anything is written"""

    def __init__(self, proposals, db_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Matched Images")
        self.setModal(True)
        self.resize(800, 400)

        self.proposals = proposals  # list of {'id', 'name', 'old_path', 'new_path'}
        self.db_manager = db_manager
        self.applied = []  # proposals actually written to the database

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        count = len(self.proposals)
        info_label = QLabel(
            f"Found {count} possible match{'es' if count != 1 else ''} by filename. "
            "These are only suggestions — review the paths below and uncheck anything "
            "that doesn't look right before applying."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.table = QTableWidget(count, 4)
        self.table.setHorizontalHeaderLabels(["Use", "DSO", "Missing Path", "Found Match"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 150)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        for row, proposal in enumerate(self.proposals):
            use_item = QTableWidgetItem()
            use_item.setFlags((use_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            use_item.setCheckState(Qt.Checked)
            self.table.setItem(row, 0, use_item)

            for col, value in enumerate(
                (proposal['name'], proposal['old_path'], proposal['new_path']), start=1
            ):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)

        # Size the path columns to fit their (typically long) contents; still
        # draggable afterward since the header stays in Interactive mode.
        self.table.resizeColumnToContents(2)
        self.table.resizeColumnToContents(3)

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Selected")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

    def _apply(self):
        checked_proposals = [
            proposal for row, proposal in enumerate(self.proposals)
            if self.table.item(row, 0).checkState() == Qt.Checked
        ]

        if not checked_proposals:
            QMessageBox.information(self, "Nothing Selected", "Select at least one match to apply, or Cancel.")
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                for proposal in checked_proposals:
                    cursor.execute(
                        "UPDATE userimages SET image_path = ? WHERE id = ?",
                        (proposal['new_path'], proposal['id'])
                    )
                conn.commit()
            self.applied = checked_proposals
            self.accept()
        except Exception as e:
            logger.error(f"Error applying scanned image matches: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to update image paths: {str(e)}")


# --- Restore Image Fixup Dialog ---
class RestoreImageFixupDialog(QDialog):
    """Lets the user resolve restored images whose files could not be found on disk,
    either by locating each file manually or by scanning a folder tree for filename matches"""

    def __init__(self, invalid_images, db_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Locate Missing Images")
        self.setModal(True)
        self.resize(750, 450)

        self.db_manager = db_manager
        # Each entry: {'id', 'path', 'name', 'fixed', 'note', 'scan_folder'}
        self.images = [
            {**img, 'fixed': False, 'note': '', 'scan_folder': None}
            for img in invalid_images
        ]
        self._scan_worker = None
        self._progress_dialog = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.header_label = QLabel()
        self.header_label.setWordWrap(True)
        layout.addWidget(self.header_label)

        self.table = QTableWidget(len(self.images), 4)
        self.table.setHorizontalHeaderLabels(["Status", "DSO", "Last Known Path", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(0, 110)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 320)
        self.table.setColumnWidth(3, 100)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        for row, image in enumerate(self.images):
            self.table.setItem(row, 1, self._readonly_item(image['name']))
            locate_btn = QPushButton("Locate...")
            locate_btn.clicked.connect(lambda _checked=False, r=row: self._locate_single(r))
            self.table.setCellWidget(row, 3, locate_btn)
            self._refresh_row(row)

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        scan_btn = QPushButton("Scan Folder (and Subfolders)...")
        scan_btn.clicked.connect(self._scan_folder)
        btn_layout.addWidget(scan_btn)
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._update_header()

    @staticmethod
    def _readonly_item(text):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _refresh_row(self, row):
        image = self.images[row]
        if image['fixed']:
            status_text = "✅ Fixed"
        elif image['note']:
            status_text = "⚠ " + image['note']
        else:
            status_text = "⚠ Missing"
        self.table.setItem(row, 0, self._readonly_item(status_text))
        self.table.setItem(row, 2, self._readonly_item(image['path']))

        locate_btn = self.table.cellWidget(row, 3)
        if locate_btn is not None:
            locate_btn.setText("Change..." if image['fixed'] else "Locate...")

    def _update_header(self):
        total = len(self.images)
        remaining = sum(1 for img in self.images if not img['fixed'])
        fixed = total - remaining
        if remaining == 0:
            self.header_label.setText(
                f"All {total} restored image{'s' if total != 1 else ''} have been located. You can close this window."
            )
        else:
            self.header_label.setText(
                f"{fixed} of {total} restored image{'s' if total != 1 else ''} located so far. "
                f"{remaining} still missing. Locate them individually, or scan a folder to find matches automatically."
            )

    def _apply_fix(self, row, new_path, write_db=True):
        image = self.images[row]
        if write_db:
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE userimages SET image_path = ? WHERE id = ?",
                        (new_path, image['id'])
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error updating image path: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to update image path: {str(e)}")
                return
        image['path'] = new_path
        image['fixed'] = True
        image['note'] = ''
        self._refresh_row(row)
        self._update_header()

    def _locate_single(self, row):
        image = self.images[row]
        start_dir = image['scan_folder'] or ''
        if not start_dir:
            last_known_dir = os.path.dirname(image['path'])
            start_dir = last_known_dir if os.path.isdir(last_known_dir) else ''

        try:
            from cosmoscollection.imaging.DSOGallery import SUPPORTED_IMAGE_EXTENSIONS
            ext_filter = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_EXTENSIONS))
            name_filter = f"Image Files ({ext_filter});;All Files (*.*)"
        except ImportError:
            name_filter = "All Files (*.*)"

        new_path, _ = QFileDialog.getOpenFileName(
            self, f"Locate Image for {image['name']}", start_dir, name_filter
        )
        if new_path:
            self._apply_fix(row, new_path)

    def _scan_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Scan")
        if not folder:
            return

        self._progress_dialog = QProgressDialog("Scanning folder for images...", "Cancel", 0, 0, self)
        self._progress_dialog.setWindowTitle("Scanning")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)

        # Match the user's configured worker-thread count (Settings > max_threads),
        # the same setting used for parallel data loading elsewhere in the app.
        settings = QSettings("CosmosCollection", "CosmosCollection")
        default_threads = max(1, (os.cpu_count() or 4) - 2)
        max_threads = max(1, min(settings.value("max_threads", default_threads, type=int), 128))

        self._scan_worker = FolderScanWorker(folder, max_workers=max_threads)
        self._scan_worker.progress.connect(
            lambda count: self._progress_dialog.setLabelText(f"Scanning folder for images...\n{count} files scanned")
        )
        self._scan_worker.finished_scan.connect(lambda basename_map: self._on_scan_finished(folder, basename_map))
        self._progress_dialog.canceled.connect(self._scan_worker.cancel)

        self._scan_worker.start()
        self._progress_dialog.show()

    def _on_scan_finished(self, folder, basename_map):
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

        proposals = []
        for row, image in enumerate(self.images):
            if image['fixed']:
                continue
            candidates = basename_map.get(os.path.basename(image['path']).lower(), [])
            if len(candidates) == 1:
                proposals.append({
                    'row': row,
                    'id': image['id'],
                    'name': image['name'],
                    'old_path': image['path'],
                    'new_path': candidates[0],
                })
            elif len(candidates) > 1:
                image['note'] = 'Multiple matches found - locate manually'
                image['scan_folder'] = folder
                self._refresh_row(row)

        if not proposals:
            QMessageBox.information(
                self, "No Matches Found",
                "No filenames in the selected folder (or its subfolders) matched the missing images."
            )
            return

        confirm_dialog = ScanMatchConfirmationDialog(proposals, self.db_manager, self)
        if confirm_dialog.exec() == QDialog.Accepted:
            applied_ids = {applied['id'] for applied in confirm_dialog.applied}
            for proposal in proposals:
                if proposal['id'] in applied_ids:
                    self._apply_fix(proposal['row'], proposal['new_path'], write_db=False)
