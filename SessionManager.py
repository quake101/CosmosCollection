#!/usr/bin/env python3
"""
Session Manager
Plan future observing sessions and log past ones: DSO target, date/time, location,
equipment, and (via drag-and-drop) the FITS/XISF subs a session produced.
"""

import os
import calendar
import logging
from collections import Counter
from datetime import datetime, date as date_cls, timedelta

from PySide6.QtCore import (Qt, QDate, QDateTime, QEvent, QTime, QTimer, QPoint, QSettings, Signal, QThread,
                            QStringListModel, QUrl)
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QGroupBox, QMessageBox,
                               QHeaderView, QTextEdit, QDialog, QComboBox,
                               QLineEdit, QCheckBox, QDateEdit, QTimeEdit, QMenu,
                               QCompleter, QSplitter, QFormLayout, QRadioButton,
                               QListWidget, QListWidgetItem, QProgressDialog,
                               QCalendarWidget, QTabWidget, QTableView, QDateTimeEdit,
                               QSpinBox, QDoubleSpinBox, QFileDialog, QAbstractItemView)

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from TimeFormatHelper import format_time
from Theme import COLORS
import SessionFileScanner
import SessionObservations

logger = logging.getLogger(__name__)

COMMON_TIMEZONES = [
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "America/Phoenix", "America/Anchorage", "Pacific/Honolulu", "UTC",
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Asia/Tokyo", "Australia/Sydney",
]


class LocationOverrideWidget(QWidget):
    """Per-session location picker: defaults to the active saved location, with a
    custom lat/lon/name/timezone entry (and map picker) for a one-off override."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_manager = DatabaseManager()
        self._locations = {}
        self._setup_ui()
        self._populate_locations()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Location:"))
        self.location_combo = QComboBox()
        self.location_combo.currentIndexChanged.connect(self._on_combo_changed)
        combo_row.addWidget(self.location_combo, 1)
        layout.addLayout(combo_row)

        self.custom_widget = QWidget()
        custom_layout = QVBoxLayout(self.custom_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit)
        custom_layout.addLayout(name_row)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("Lat:"))
        self.lat_edit = QLineEdit()
        coord_row.addWidget(self.lat_edit)
        coord_row.addWidget(QLabel("Lon:"))
        self.lon_edit = QLineEdit()
        coord_row.addWidget(self.lon_edit)
        custom_layout.addLayout(coord_row)

        tz_row = QHBoxLayout()
        tz_row.addWidget(QLabel("Timezone:"))
        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
        self.timezone_combo.addItems(COMMON_TIMEZONES)
        tz_row.addWidget(self.timezone_combo)
        map_btn = QPushButton("Pick on Map...")
        map_btn.clicked.connect(self._pick_on_map)
        tz_row.addWidget(map_btn)
        custom_layout.addLayout(tz_row)

        layout.addWidget(self.custom_widget)
        self.custom_widget.setVisible(False)

    def _populate_locations(self):
        self.location_combo.blockSignals(True)
        self.location_combo.clear()
        self._locations = {}
        active_index = 0
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, location_lat, location_lon, location_name, timezone, is_active
                    FROM usersettings ORDER BY id
                """)
                for row in cursor.fetchall():
                    loc_id, lat, lon, name, tz, is_active = row
                    display = name or (f"{lat:.4f}, {lon:.4f}" if lat is not None else "Unnamed Location")
                    if is_active:
                        display = f"(Active) {display}"
                    self.location_combo.addItem(display, loc_id)
                    self._locations[loc_id] = {"lat": lat, "lon": lon, "name": name, "timezone": tz}
                    if is_active:
                        active_index = self.location_combo.count() - 1
        except Exception as e:
            logger.error(f"Error loading locations: {e}")

        self.location_combo.addItem("Custom / Other Location...", -1)
        if self.location_combo.count() > 1:
            self.location_combo.setCurrentIndex(active_index)
        self.location_combo.blockSignals(False)
        self._on_combo_changed(self.location_combo.currentIndex())

    def _on_combo_changed(self, _index):
        loc_id = self.location_combo.currentData()
        is_custom = (loc_id == -1)
        self.custom_widget.setVisible(is_custom)
        if is_custom and not self.timezone_combo.currentText():
            active_tz = next((v["timezone"] for v in self._locations.values() if v.get("timezone")), None)
            if active_tz:
                self.timezone_combo.setCurrentText(active_tz)

    def _pick_on_map(self):
        from main import MapLocationPickerDialog
        try:
            lat0 = float(self.lat_edit.text()) if self.lat_edit.text().strip() else 40.7128
            lon0 = float(self.lon_edit.text()) if self.lon_edit.text().strip() else -74.0060
        except ValueError:
            lat0, lon0 = 40.7128, -74.0060

        dialog = MapLocationPickerDialog(lat0, lon0, parent=self)
        if dialog.exec() == QDialog.Accepted:
            result = dialog.get_selected_coordinates()
            if result:
                lat, lon, name = result
                self.lat_edit.setText(f"{lat:.6f}")
                self.lon_edit.setText(f"{lon:.6f}")
                if name:
                    self.name_edit.setText(name)

    def get_location(self):
        """Returns (lat, lon, name, timezone, source)."""
        loc_id = self.location_combo.currentData()
        if loc_id == -1:
            try:
                lat = float(self.lat_edit.text()) if self.lat_edit.text().strip() else None
            except ValueError:
                lat = None
            try:
                lon = float(self.lon_edit.text()) if self.lon_edit.text().strip() else None
            except ValueError:
                lon = None
            return (lat, lon, self.name_edit.text().strip() or None,
                    self.timezone_combo.currentText().strip() or None, "custom")

        loc = self._locations.get(loc_id)
        if not loc:
            return (None, None, None, None, "active")
        current_text = self.location_combo.itemText(self.location_combo.currentIndex())
        source = "active" if current_text.startswith("(Active)") else "saved"
        return (loc["lat"], loc["lon"], loc["name"], loc["timezone"], source)

    def set_location(self, lat, lon, name, timezone, source):
        if source != "custom" and lat is not None and lon is not None:
            for loc_id, loc in self._locations.items():
                if loc["lat"] is None or loc["lon"] is None:
                    continue
                if abs(loc["lat"] - lat) < 0.0001 and abs(loc["lon"] - lon) < 0.0001:
                    index = self.location_combo.findData(loc_id)
                    if index >= 0:
                        self.location_combo.setCurrentIndex(index)
                        return

        for i in range(self.location_combo.count()):
            if self.location_combo.itemData(i) == -1:
                self.location_combo.setCurrentIndex(i)
                break
        self.lat_edit.setText(f"{lat:.6f}" if lat is not None else "")
        self.lon_edit.setText(f"{lon:.6f}" if lon is not None else "")
        self.name_edit.setText(name or "")
        if timezone:
            self.timezone_combo.setCurrentText(timezone)


def _rollback(db_manager):
    try:
        with db_manager.get_connection() as conn:
            conn.rollback()
    except Exception as e:
        logger.error(f"Error rolling back: {e}")


class SessionFormWidget(QWidget):
    """The session's own fields (DSO, schedule, location, equipment, notes), shared
    by AddEditSessionDialog and SessionDetailsDialog's Overview tab. write() never
    commits, so a caller can save it in the same transaction as other changes."""

    def __init__(self, session_data=None, target_data=None, parsed_metadata=None,
                 initial_status="Planned", parent=None):
        super().__init__(parent)

        self.db_manager = DatabaseManager()
        self.is_edit_mode = False
        self.session_id = None
        self.saved_session_data = None
        self._dso_cache = {}
        self._ra_deg = None
        self._dec_deg = None
        self._target_id = None
        self._scanned_aggregate = None
        self._existing_aggregate = None

        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._do_dso_search)
        self._pending_search_text = ""

        self._setup_ui()
        self.status_combo.setCurrentText(initial_status)

        if target_data:
            self._populate_from_target_data(target_data)
        if parsed_metadata:
            self._populate_from_parsed_metadata(parsed_metadata)
        if session_data:
            self._populate_from_session_data(session_data)

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        schedule_group = QGroupBox("DSO && Schedule")
        schedule_layout = QVBoxLayout()

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("DSO Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., M 31, NGC 7000, IC 1396")
        self._completer_model = QStringListModel()
        self._completer = QCompleter()
        self._completer.setModel(self._completer_model)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self.name_edit.setCompleter(self._completer)
        self._completer.activated.connect(self._on_dso_selected)
        self.name_edit.textChanged.connect(self._on_name_text_changed)
        name_row.addWidget(self.name_edit)
        schedule_layout.addLayout(name_row)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Planned", "In Progress", "Completed", "Cancelled"])
        status_row.addWidget(self.status_combo)
        status_row.addWidget(QLabel("Date:"))
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        status_row.addWidget(self.date_edit)
        schedule_layout.addLayout(status_row)

        time_row = QHBoxLayout()
        self.time_checkbox = QCheckBox("Specify start/end time")
        self.time_checkbox.toggled.connect(self._on_time_checkbox_toggled)
        time_row.addWidget(self.time_checkbox)
        time_row.addWidget(QLabel("Start:"))
        self.start_time_edit = QTimeEdit(QTime(20, 0))
        self.start_time_edit.setEnabled(False)
        time_row.addWidget(self.start_time_edit)
        time_row.addWidget(QLabel("End:"))
        self.end_time_edit = QTimeEdit(QTime(23, 0))
        self.end_time_edit.setEnabled(False)
        time_row.addWidget(self.end_time_edit)
        schedule_layout.addLayout(time_row)

        schedule_group.setLayout(schedule_layout)
        layout.addWidget(schedule_group)

        location_group = QGroupBox("Location")
        location_layout = QVBoxLayout()
        self.location_widget = LocationOverrideWidget()
        location_layout.addWidget(self.location_widget)
        location_group.setLayout(location_layout)
        layout.addWidget(location_group)

        equipment_group = QGroupBox("Equipment")
        equipment_layout = QVBoxLayout()

        telescope_row = QHBoxLayout()
        telescope_row.addWidget(QLabel("Telescope:"))
        self.telescope_combo = QComboBox()
        self._populate_telescope_combo()
        telescope_row.addWidget(self.telescope_combo)
        equipment_layout.addLayout(telescope_row)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self.camera_edit = QLineEdit()
        camera_row.addWidget(self.camera_edit)
        equipment_layout.addLayout(camera_row)

        filters_row = QHBoxLayout()
        filters_row.addWidget(QLabel("Filters:"))
        self.filters_edit = QLineEdit()
        self.filters_edit.setPlaceholderText("e.g., Ha, OIII, SII")
        filters_row.addWidget(self.filters_edit)
        equipment_layout.addLayout(filters_row)

        equipment_group.setLayout(equipment_layout)
        layout.addWidget(equipment_group)

        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout()
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)
        layout.addStretch()

        self.setLayout(layout)

    def _on_time_checkbox_toggled(self, checked):
        self.start_time_edit.setEnabled(checked)
        self.end_time_edit.setEnabled(checked)

    def _populate_telescope_combo(self):
        self.telescope_combo.clear()
        self.telescope_combo.addItem("Any", None)
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length
                    FROM usertelescopes WHERE is_active = 1 ORDER BY name
                """)
                for tel_id, name, aperture, focal_length in cursor.fetchall():
                    if aperture and focal_length and aperture > 0:
                        display_text = f"{name} ({int(aperture)}mm f/{focal_length / aperture:.1f})"
                    elif aperture:
                        display_text = f"{name} ({int(aperture)}mm)"
                    else:
                        display_text = name
                    self.telescope_combo.addItem(display_text, tel_id)
        except Exception as e:
            logger.error(f"Error loading telescopes: {e}")

    def _on_name_text_changed(self, text):
        text = text.strip()
        if len(text) < 2:
            self._completer_model.setStringList([])
            self._dso_cache.clear()
            return
        self._pending_search_text = text
        self._search_timer.start()

    def _do_dso_search(self):
        if len(self._pending_search_text) < 2:
            return
        self._search_dso_catalog(self._pending_search_text)

    def _search_dso_catalog(self, text):
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                catalogue = None
                designation_part = None
                text_upper = text.upper().strip()
                for prefix in ("NGC", "IC", "M"):
                    if text_upper.startswith(prefix):
                        remainder = text_upper[len(prefix):]
                        if remainder == "" or remainder[0] in (" ", "-") or remainder[0].isdigit():
                            catalogue = prefix
                            designation_part = remainder.strip()
                            break

                if catalogue and designation_part is not None:
                    cursor.execute("""
                        SELECT c.catalogue || ' ' || c.designation as name, d.ra, d.dec
                        FROM cataloguenr c
                        JOIN dsodetail d ON d.id = c.dsodetailid
                        WHERE c.catalogue = ? AND c.designation LIKE ?
                        ORDER BY CAST(c.designation AS INTEGER), c.designation
                        LIMIT 20
                    """, (catalogue, designation_part + "%"))
                else:
                    cursor.execute("""
                        SELECT c.catalogue || ' ' || c.designation as name, d.ra, d.dec
                        FROM cataloguenr c
                        JOIN dsodetail d ON d.id = c.dsodetailid
                        WHERE c.catalogue || ' ' || c.designation LIKE ?
                        ORDER BY c.catalogue, CAST(c.designation AS INTEGER), c.designation
                        LIMIT 20
                    """, ("%" + text + "%",))

                self._dso_cache.clear()
                names = []
                for name, ra, dec in cursor.fetchall():
                    names.append(name)
                    self._dso_cache[name] = {"ra": ra, "dec": dec}
                self._completer_model.setStringList(names)
        except Exception as e:
            logger.error(f"Error searching DSO catalog: {e}")

    def _on_dso_selected(self, text):
        data = self._dso_cache.get(text)
        if not data:
            return
        if data.get("ra") is not None:
            self._ra_deg = round(data["ra"], 6)
        if data.get("dec") is not None:
            self._dec_deg = round(data["dec"], 6)

    def _populate_from_target_data(self, target_data):
        self.name_edit.setText(target_data.get("name", "") or "")
        self._ra_deg = target_data.get("ra_deg")
        self._dec_deg = target_data.get("dec_deg")
        self._target_id = target_data.get("id")
        telescope_id = target_data.get("telescope_id")
        if telescope_id is not None:
            index = self.telescope_combo.findData(telescope_id)
            if index >= 0:
                self.telescope_combo.setCurrentIndex(index)

    def _populate_from_parsed_metadata(self, summary):
        self._scanned_aggregate = summary
        dso_name = summary.get("dso_name")
        if dso_name:
            self.name_edit.setText(dso_name)
        camera = summary.get("camera")
        if camera:
            self.camera_edit.setText(camera)
        filters = summary.get("filters_used") or []
        if filters:
            self.filters_edit.setText(", ".join(filters))
        earliest = summary.get("earliest_sub_date")
        if earliest:
            try:
                dt = datetime.fromisoformat(str(earliest).replace('Z', '+00:00'))
                self.date_edit.setDate(QDate(dt.year, dt.month, dt.day))
            except ValueError:
                pass

    def _populate_from_session_data(self, session_data, edit_mode=True):
        if edit_mode:
            self.is_edit_mode = True
            self.session_id = session_data.get("id")
            self._existing_aggregate = {
                "sub_count": session_data.get("sub_count", 0),
                "integration_seconds": session_data.get("integration_seconds", 0),
                "earliest_sub_date": session_data.get("earliest_sub_date"),
                "latest_sub_date": session_data.get("latest_sub_date"),
            }

        self.name_edit.setText(session_data.get("dso_name", "") or "")
        self.status_combo.setCurrentText(session_data.get("status") or "Planned")
        self._ra_deg = session_data.get("ra_deg")
        self._dec_deg = session_data.get("dec_deg")
        self._target_id = session_data.get("target_id")

        session_date = session_data.get("session_date")
        if session_date:
            try:
                y, m, d = (int(p) for p in session_date.split("-"))
                self.date_edit.setDate(QDate(y, m, d))
            except Exception:
                pass

        start_time = session_data.get("start_time")
        end_time = session_data.get("end_time")
        if start_time or end_time:
            self.time_checkbox.setChecked(True)
            if start_time:
                h, mnt = (int(p) for p in start_time.split(":")[:2])
                self.start_time_edit.setTime(QTime(h, mnt))
            if end_time:
                h, mnt = (int(p) for p in end_time.split(":")[:2])
                self.end_time_edit.setTime(QTime(h, mnt))

        telescope_id = session_data.get("telescope_id")
        if telescope_id is not None:
            index = self.telescope_combo.findData(telescope_id)
            if index >= 0:
                self.telescope_combo.setCurrentIndex(index)

        self.camera_edit.setText(session_data.get("camera") or "")
        self.filters_edit.setText(session_data.get("filters_used") or "")
        self.notes_edit.setPlainText(session_data.get("notes") or "")

        self.location_widget.set_location(
            session_data.get("location_lat"), session_data.get("location_lon"),
            session_data.get("location_name"), session_data.get("location_timezone"),
            session_data.get("location_source") or "active",
        )

    def load_as_duplicate(self, session_data):
        """Pre-fill this (new, non-edit) dialog from an existing session's fields."""
        self._populate_from_session_data(session_data, edit_mode=False)
        self.status_combo.setCurrentText("Planned")

    def _resolve_target_id_by_name(self, name):
        """Auto-link to a Target List entry whose name matches the session's DSO
        name, so a target doesn't have to be planned via "Plan a Session" to get
        linked - typing/scanning in a name that's already on the target list is
        enough. Matches loosely (case/whitespace-insensitive) since FITS OBJECT
        headers and target list names often differ in spacing (e.g. "M31" vs "M 31")."""
        normalized = name.replace(" ", "").strip().upper() if name else ""
        if not normalized:
            return None
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM usertargetlist")
                for target_id, target_name in cursor.fetchall():
                    if target_name and target_name.replace(" ", "").strip().upper() == normalized:
                        return target_id
        except Exception as e:
            logger.error(f"Error resolving target list link for '{name}': {e}")
        return None

    def validate(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Validation Error", "DSO name is required.")
            return False
        return True

    def collect(self):
        """The form's values as a usersessions row dict (not validated)."""
        name = self.name_edit.text().strip()
        lat, lon, loc_name, tz, source = self.location_widget.get_location()
        has_times = self.time_checkbox.isChecked()
        aggregate = self._scanned_aggregate or self._existing_aggregate or {}
        resolved_target_id = self._resolve_target_id_by_name(name)
        return {
            "target_id": resolved_target_id if resolved_target_id is not None else self._target_id,
            "dso_name": name,
            "ra_deg": self._ra_deg,
            "dec_deg": self._dec_deg,
            "status": self.status_combo.currentText(),
            "session_date": self.date_edit.date().toString("yyyy-MM-dd"),
            "start_time": self.start_time_edit.time().toString("HH:mm") if has_times else None,
            "end_time": self.end_time_edit.time().toString("HH:mm") if has_times else None,
            "location_lat": lat,
            "location_lon": lon,
            "location_name": loc_name,
            "location_timezone": tz,
            "location_source": source,
            "telescope_id": self.telescope_combo.currentData(),
            "camera": self.camera_edit.text().strip(),
            "filters_used": self.filters_edit.text().strip(),
            "sub_count": aggregate.get("sub_count", 0),
            "integration_seconds": aggregate.get("integration_seconds", 0),
            "earliest_sub_date": aggregate.get("earliest_sub_date"),
            "latest_sub_date": aggregate.get("latest_sub_date"),
            "notes": self.notes_edit.toPlainText().strip(),
            "modified_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def snapshot(self):
        """Comparable form state, for detecting unsaved edits."""
        data = self.collect()
        data.pop("modified_date", None)
        return data

    def write(self, conn):
        """Insert or update the session row. Does not commit."""
        data = self.collect()
        cursor = conn.cursor()
        if self.is_edit_mode and self.session_id:
            cursor.execute("""
                UPDATE usersessions SET
                    target_id=?, dso_name=?, ra_deg=?, dec_deg=?, status=?, session_date=?,
                    start_time=?, end_time=?, location_lat=?, location_lon=?, location_name=?,
                    location_timezone=?, location_source=?, telescope_id=?, camera=?, filters_used=?,
                    sub_count=?, integration_seconds=?, earliest_sub_date=?, latest_sub_date=?,
                    notes=?, modified_date=?
                WHERE id=?
            """, (
                data["target_id"], data["dso_name"], data["ra_deg"], data["dec_deg"], data["status"],
                data["session_date"], data["start_time"], data["end_time"], data["location_lat"],
                data["location_lon"], data["location_name"], data["location_timezone"],
                data["location_source"], data["telescope_id"], data["camera"], data["filters_used"],
                data["sub_count"], data["integration_seconds"], data["earliest_sub_date"],
                data["latest_sub_date"], data["notes"], data["modified_date"], self.session_id,
            ))
        else:
            cursor.execute("""
                INSERT INTO usersessions (
                    target_id, dso_name, ra_deg, dec_deg, status, session_date, start_time, end_time,
                    location_lat, location_lon, location_name, location_timezone, location_source,
                    telescope_id, camera, filters_used, sub_count, integration_seconds,
                    earliest_sub_date, latest_sub_date, notes, modified_date
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                data["target_id"], data["dso_name"], data["ra_deg"], data["dec_deg"], data["status"],
                data["session_date"], data["start_time"], data["end_time"], data["location_lat"],
                data["location_lon"], data["location_name"], data["location_timezone"],
                data["location_source"], data["telescope_id"], data["camera"], data["filters_used"],
                data["sub_count"], data["integration_seconds"], data["earliest_sub_date"],
                data["latest_sub_date"], data["notes"], data["modified_date"],
            ))
            self.session_id = cursor.lastrowid
            self.is_edit_mode = True
        self.saved_session_data = data
        return data


class AddEditSessionDialog(WindowPositionMixin, QDialog):
    """Add dialog for a planned or logged session (also used to duplicate one)."""

    WINDOW_POSITION_KEY = "AddEditSessionDialog"

    def __init__(self, session_data=None, target_data=None, parsed_metadata=None,
                 initial_status="Planned", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Session" if session_data else "New Session")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(520, 680)  # default size the first time this dialog is ever opened

        self.db_manager = DatabaseManager()
        self.form = SessionFormWidget(session_data=session_data, target_data=target_data,
                                      parsed_metadata=parsed_metadata, initial_status=initial_status)

        layout = QVBoxLayout(self)
        layout.addWidget(self.form)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        self.save_btn = QPushButton("Save Changes" if session_data else "Save Session")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save_session)
        buttons_layout.addWidget(self.save_btn)
        layout.addLayout(buttons_layout)

        self.setup_window_position()

    @property
    def session_id(self):
        return self.form.session_id

    @property
    def saved_session_data(self):
        return self.form.saved_session_data

    def load_as_duplicate(self, session_data):
        self.form.load_as_duplicate(session_data)

    def _save_session(self):
        if not self.form.validate():
            return
        try:
            with self.db_manager.get_connection() as conn:
                self.form.write(conn)
                conn.commit()
            self.accept()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error saving session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session: {e}")


class DropMatchDialog(WindowPositionMixin, QDialog):
    """Always-shown confirmation for drag-and-dropped FITS/XISF metadata: create a
    new session or attach to an existing one - never auto-attached silently."""

    WINDOW_POSITION_KEY = "SessionDropMatchDialog"

    def __init__(self, summary, db_manager, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.db_manager = db_manager
        self.result_session_id = None
        self.setWindowTitle("Attach Scanned Files")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(520, 480)  # default size the first time this dialog is ever opened
        try:
            with self.db_manager.get_connection() as conn:
                self._already_attached = SessionObservations.count_already_attached(conn, summary.get("files", []))
        except Exception as e:
            logger.error(f"Error checking for already-attached files: {e}")
            self._already_attached = Counter()
        self._already_attached_total = sum(self._already_attached.values())
        self._setup_ui()
        self._load_existing_sessions()
        self.setup_window_position()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        summary_group = QGroupBox("Scanned Files")
        form = QFormLayout()
        frame_counts = self.summary.get("frame_type_counts", {})
        counts_text = ", ".join(f"{v} {k}" for k, v in frame_counts.items()) or "0 files"
        form.addRow("DSO Name:", QLabel(self.summary.get("dso_name") or "(unknown)"))
        form.addRow("Date Range:", QLabel(self._format_date_range()))
        form.addRow("Files:", QLabel(f"{self.summary.get('sub_count', 0)} ({counts_text})"))
        form.addRow("Integration Time:", QLabel(self._format_integration()))
        form.addRow("Camera:", QLabel(self.summary.get("camera") or "(unknown)"))
        form.addRow("Telescope:", QLabel(self.summary.get("telescope") or "(unknown)"))
        form.addRow("Filters:", QLabel(", ".join(self.summary.get("filters_used", [])) or "(none)"))
        if self._already_attached_total:
            already_label = QLabel(
                f"{self._already_attached_total} file(s) already belong to a session and will be skipped:\n"
                + self._already_attached_text()
            )
            already_label.setWordWrap(True)
            already_label.setStyleSheet(f"color: {COLORS['warning']};")
            form.addRow("Already Attached:", already_label)
        summary_group.setLayout(form)
        layout.addWidget(summary_group)

        attach_group = QGroupBox("Attach to")
        attach_layout = QVBoxLayout()
        self.create_radio = QRadioButton("Create New Session")
        self.create_radio.setChecked(True)
        self.existing_radio = QRadioButton("Add to Existing Session")
        attach_layout.addWidget(self.create_radio)
        attach_layout.addWidget(self.existing_radio)

        self.sessions_list = QListWidget()
        self.sessions_list.setEnabled(False)
        attach_layout.addWidget(self.sessions_list)
        self.create_radio.toggled.connect(lambda checked: self.sessions_list.setEnabled(not checked))

        attach_group.setLayout(attach_layout)
        layout.addWidget(attach_group)

        buttons_row = QHBoxLayout()
        buttons_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_row.addWidget(cancel_btn)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_confirm)
        buttons_row.addWidget(ok_btn)
        layout.addLayout(buttons_row)

    def _format_date_range(self):
        earliest = self.summary.get("earliest_sub_date")
        latest = self.summary.get("latest_sub_date")
        if not earliest:
            return "(unknown)"
        if latest and latest[:10] != earliest[:10]:
            return f"{earliest[:10]} to {latest[:10]}"
        return earliest[:10]

    def _format_integration(self):
        seconds = self.summary.get("integration_seconds", 0)
        return f"{seconds / 3600.0:.2f} h ({int(seconds)} s)"

    def _load_existing_sessions(self):
        normalized_dso = (self.summary.get("dso_name") or "").replace(" ", "").strip().upper()
        suggested_item = None
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, dso_name, status, session_date FROM usersessions
                    ORDER BY session_date DESC, id DESC
                """)
                for session_id, dso_name, status, session_date in cursor.fetchall():
                    label = f"{dso_name} — {status} — {session_date}"
                    is_ongoing_match = (
                        status == "In Progress" and normalized_dso
                        and (dso_name or "").replace(" ", "").strip().upper() == normalized_dso
                    )
                    if is_ongoing_match:
                        label += "  (ongoing — suggested match)"
                    item = QListWidgetItem(label)
                    item.setData(Qt.UserRole, session_id)
                    self.sessions_list.addItem(item)
                    if is_ongoing_match and suggested_item is None:
                        suggested_item = item
        except Exception as e:
            logger.error(f"Error loading sessions for drop match: {e}")

        # Still always requires the user to click OK - this only changes the
        # dialog's *default* selection so continuing an already-ongoing session
        # for the same DSO is the path of least resistance, per the "always
        # confirm, never silently auto-attach" rule.
        if suggested_item is not None:
            self.sessions_list.setCurrentItem(suggested_item)
            self.existing_radio.setChecked(True)

    def _on_confirm(self):
        if self.create_radio.isChecked():
            files = self.summary.get("files", [])
            if files and self._already_attached_total >= len(files):
                QMessageBox.information(
                    self, "Already Attached",
                    "Every scanned file already belongs to an existing session, so a new session "
                    "would have no data.\n\n" + self._already_attached_text(),
                )
                return
            dialog = AddEditSessionDialog(parsed_metadata=self.summary, initial_status="In Progress", parent=self)
            if dialog.exec() == QDialog.Accepted and dialog.session_id:
                if self._attach_to(dialog.session_id):
                    self.accept()
        else:
            item = self.sessions_list.currentItem()
            if not item:
                QMessageBox.warning(self, "No Selection", "Please select an existing session.")
                return
            if self._attach_to(item.data(Qt.UserRole)):
                self.accept()

    def _already_attached_text(self):
        return "\n".join(f"• {count} in {label}" for label, count in self._already_attached.most_common())

    def _attach_to(self, session_id):
        """Attach the scanned files (skipping duplicates), place them on their
        observing nights and recompute the session's totals, in one transaction."""
        try:
            with self.db_manager.get_connection() as conn:
                report = SessionObservations.attach_files(conn, session_id, self.summary.get("files", []))
                SessionObservations.assign_files_to_observations(conn, session_id)
                SessionObservations.recompute_session_aggregates(conn, session_id)
                conn.commit()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error attaching session files: {e}")
            QMessageBox.critical(self, "Error", f"Failed to attach files: {e}")
            return False

        message = SessionObservations.format_attach_report(report)
        if report["inserted"] == 0:
            QMessageBox.information(self, "Nothing New Attached",
                                    message or "No new files were attached.")
        elif message:
            QMessageBox.information(self, "Files Attached", message)
        self.result_session_id = session_id
        return True


class LinkTargetDialog(WindowPositionMixin, QDialog):
    """Pick a Target List entry to link a session to (sets usersessions.target_id -
    doesn't touch the session's own dso_name/ra_deg/dec_deg, same as the existing
    automatic name-match linking in AddEditSessionDialog._resolve_target_id_by_name)."""

    WINDOW_POSITION_KEY = "LinkTargetDialog"

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.selected_target = None
        self.setWindowTitle("Link to Target List")
        self.setModal(True)
        self.resize(420, 480)  # default size the first time this dialog is ever opened
        self._setup_ui()
        self._load_targets()
        self.setup_window_position()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._filter_targets)
        search_row.addWidget(self.search_box)
        layout.addLayout(search_row)

        self.targets_list = QListWidget()
        self.targets_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.targets_list.itemDoubleClicked.connect(lambda _: self._on_confirm())
        layout.addWidget(self.targets_list, 1)

        buttons_row = QHBoxLayout()
        buttons_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_row.addWidget(cancel_btn)
        self.ok_btn = QPushButton("Link")
        self.ok_btn.setDefault(True)
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self._on_confirm)
        buttons_row.addWidget(self.ok_btn)
        layout.addLayout(buttons_row)

    def _load_targets(self):
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, dso_type, constellation, magnitude, ra_deg, dec_deg
                    FROM usertargetlist ORDER BY name
                """)
                for target_id, name, dso_type, constellation, magnitude, ra_deg, dec_deg in cursor.fetchall():
                    details = ", ".join(p for p in (dso_type, constellation) if p)
                    label = f"{name} ({details})" if details else name
                    if magnitude is not None:
                        label += f" - Mag {magnitude:.1f}"
                    item = QListWidgetItem(label)
                    item.setData(Qt.UserRole, {
                        "id": target_id, "name": name, "ra_deg": ra_deg, "dec_deg": dec_deg,
                    })
                    self.targets_list.addItem(item)
        except Exception as e:
            logger.error(f"Error loading target list for linking: {e}")

    def _filter_targets(self, text):
        text = text.strip().lower()
        for i in range(self.targets_list.count()):
            item = self.targets_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _on_selection_changed(self):
        self.ok_btn.setEnabled(bool(self.targets_list.selectedItems()))

    def _on_confirm(self):
        items = self.targets_list.selectedItems()
        if not items:
            return
        self.selected_target = items[0].data(Qt.UserRole)
        self.accept()


def format_duration(seconds):
    """'2h 05m', '45m', '30s' - integration times read better than decimal hours."""
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    total_minutes = int(round(seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def format_exposure(exposure_seconds):
    return f"{float(exposure_seconds or 0):g}s"


def dropped_sub_paths(mime_data):
    """Local folders and FITS/XISF files in a drag's mime data (empty if none)."""
    if not mime_data.hasUrls():
        return []
    paths = []
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        if os.path.isdir(path) or os.path.splitext(path)[1].lower() in SessionFileScanner.SUPPORTED_EXTENSIONS:
            paths.append(path)
    return paths


def _make_table(columns, stretch_last=True):
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    header.setStretchLastSection(stretch_last)
    return table


def _table_item(text, align=None, user_data=None, tooltip=None):
    item = QTableWidgetItem(text)
    if align is not None:
        item.setTextAlignment(align)
    if user_data is not None:
        item.setData(Qt.UserRole, user_data)
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _to_qdatetime(dt):
    return QDateTime(QDate(dt.year, dt.month, dt.day), QTime(dt.hour, dt.minute))


class ObservationDialog(WindowPositionMixin, QDialog):
    """Add/edit one observation (one observing night): its time span, notes and
    hand-logged subs. The night's scanned files are shown read-only - files always
    follow their own timestamps, so they can't be moved or edited here."""

    WINDOW_POSITION_KEY = "ObservationDialog"

    def __init__(self, observation=None, default_date=None, filter_choices=(), taken_nights=(), parent=None):
        super().__init__(parent)
        self.observation = observation or {}
        self.filter_choices = sorted({f for f in filter_choices if f}, key=str.lower)
        self.taken_nights = set(taken_nights)
        self.result = None
        self.setWindowTitle("Edit Observation" if observation else "Add Observation")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(580, 600)  # default size the first time this dialog is ever opened
        self._setup_ui(default_date or date_cls.today())
        self.setup_window_position()

    def _setup_ui(self, default_date):
        layout = QVBoxLayout(self)

        time_group = QGroupBox("Night")
        time_form = QFormLayout()
        self.start_edit = QDateTimeEdit()
        self.end_edit = QDateTimeEdit()
        for edit in (self.start_edit, self.end_edit):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")

        start = SessionObservations.parse_obs_datetime(self.observation.get("start_datetime"))
        end = SessionObservations.parse_obs_datetime(self.observation.get("end_datetime"))
        if start is None:
            night = self.observation.get("night_date")
            base = datetime.strptime(night, "%Y-%m-%d") if night else datetime.combine(default_date, datetime.min.time())
            start = base.replace(hour=21, minute=0)
        if end is None or end <= start:
            end = start + timedelta(hours=3)
        self.start_edit.setDateTime(_to_qdatetime(start))
        self.end_edit.setDateTime(_to_qdatetime(end))
        self._last_start = self.start_edit.dateTime()

        self.start_edit.dateTimeChanged.connect(self._on_start_changed)
        self.end_edit.editingFinished.connect(self._on_end_edited)
        self.end_edit.dateTimeChanged.connect(lambda _dt: self._update_night_label())

        time_form.addRow("Start:", self.start_edit)
        time_form.addRow("End:", self.end_edit)
        self.night_label = QLabel()
        time_form.addRow("Observing night:", self.night_label)
        hint = QLabel("A night that runs past midnight is one observation. An end time earlier "
                      "than the start time moves to the next morning.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        time_form.addRow(hint)
        time_group.setLayout(time_form)
        layout.addWidget(time_group)

        file_groups = self.observation.get("file_groups") or []
        if self.observation.get("file_count"):
            files_group = QGroupBox(f"Scanned Files ({self.observation['file_count']} attached, read-only)")
            files_layout = QVBoxLayout()
            files_table = _make_table(["Filter", "Exposure", "Subs", "Integration"])
            files_table.setRowCount(len(file_groups))
            for row, group in enumerate(sorted(file_groups, key=lambda g: (
                    SessionObservations.normalize_filter(g["filter_name"]), g["exposure_seconds"] or 0))):
                files_table.setItem(row, 0, _table_item(group["filter_name"] or "(no filter)"))
                files_table.setItem(row, 1, _table_item(format_exposure(group["exposure_seconds"]), Qt.AlignCenter))
                files_table.setItem(row, 2, _table_item(str(group["count"]), Qt.AlignCenter))
                files_table.setItem(row, 3, _table_item(format_duration(group["seconds"]), Qt.AlignCenter))
            files_table.setMaximumHeight(140)
            files_layout.addWidget(files_table)
            files_group.setLayout(files_layout)
            layout.addWidget(files_group)

        logged_group = QGroupBox("Logged Subs (entered by hand)")
        logged_layout = QVBoxLayout()
        logged_hint = QLabel("For subs you didn't import. Where these overlap scanned files with the same "
                             "filter and exposure, the larger count is used - they are never added together.")
        logged_hint.setWordWrap(True)
        logged_hint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        logged_layout.addWidget(logged_hint)
        self.logged_table = _make_table(["Filter", "Exposure (s)", "Subs", "Integration"])
        self.logged_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.logged_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        logged_layout.addWidget(self.logged_table)

        row_buttons = QHBoxLayout()
        add_row_btn = QPushButton("Add Filter")
        add_row_btn.clicked.connect(lambda: self._add_logged_row())
        row_buttons.addWidget(add_row_btn)
        remove_row_btn = QPushButton("Remove Filter")
        remove_row_btn.clicked.connect(self._remove_logged_row)
        row_buttons.addWidget(remove_row_btn)
        row_buttons.addStretch()
        self.logged_total_label = QLabel()
        row_buttons.addWidget(self.logged_total_label)
        logged_layout.addLayout(row_buttons)
        logged_group.setLayout(logged_layout)
        layout.addWidget(logged_group, 1)

        for logged in self.observation.get("logged") or []:
            self._add_logged_row(logged["filter_name"] or "", logged["exposure_seconds"] or 0, logged["sub_count"] or 0)

        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout()
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(70)
        self.notes_edit.setPlainText(self.observation.get("notes") or "")
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        self._update_night_label()
        self._update_logged_totals()

    def _on_start_changed(self, new_start):
        """Moving the start moves the end by the same amount, keeping the span."""
        delta_secs = self._last_start.secsTo(new_start)
        self._last_start = new_start
        self.end_edit.blockSignals(True)
        self.end_edit.setDateTime(self.end_edit.dateTime().addSecs(delta_secs))
        self.end_edit.blockSignals(False)
        self._update_night_label()

    def _on_end_edited(self):
        """An end earlier than the start on the same date means the next morning."""
        start, end = self.start_edit.dateTime(), self.end_edit.dateTime()
        if end <= start and end.date() == start.date():
            self.end_edit.setDateTime(end.addDays(1))
        self._update_night_label()

    def _night_date(self):
        return SessionObservations.night_date_for(self.start_edit.dateTime().toPython()).isoformat()

    def _update_night_label(self):
        start = self.start_edit.dateTime().toPython()
        end = self.end_edit.dateTime().toPython()
        text = f"Night of {self._night_date()}"
        if end > start:
            text += f"  ({format_duration((end - start).total_seconds())})"
        self.night_label.setText(text)

    def _add_logged_row(self, filter_name="", exposure=None, count=None):
        row = self.logged_table.rowCount()
        if exposure is None:
            exposure = self.logged_table.cellWidget(row - 1, 1).value() if row else 300
        self.logged_table.insertRow(row)

        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self.filter_choices)
        combo.setCurrentText(filter_name)
        exposure_spin = QDoubleSpinBox()
        exposure_spin.setRange(0, 36000)
        exposure_spin.setDecimals(1)
        exposure_spin.setValue(float(exposure))
        count_spin = QSpinBox()
        count_spin.setRange(0, 100000)
        count_spin.setValue(int(count if count is not None else 1))
        exposure_spin.valueChanged.connect(self._update_logged_totals)
        count_spin.valueChanged.connect(self._update_logged_totals)

        self.logged_table.setCellWidget(row, 0, combo)
        self.logged_table.setCellWidget(row, 1, exposure_spin)
        self.logged_table.setCellWidget(row, 2, count_spin)
        self.logged_table.setItem(row, 3, _table_item("", Qt.AlignCenter))
        self._update_logged_totals()

    def _remove_logged_row(self):
        row = self.logged_table.currentRow()
        if row < 0:
            row = self.logged_table.rowCount() - 1
        if row >= 0:
            self.logged_table.removeRow(row)
            self._update_logged_totals()

    def _update_logged_totals(self):
        total = 0.0
        for row in range(self.logged_table.rowCount()):
            seconds = self.logged_table.cellWidget(row, 1).value() * self.logged_table.cellWidget(row, 2).value()
            total += seconds
            item = self.logged_table.item(row, 3)
            if item:
                item.setText(format_duration(seconds))
        self.logged_total_label.setText(f"Logged total: {format_duration(total)}")

    def _on_ok(self):
        start = self.start_edit.dateTime().toPython().replace(second=0, microsecond=0)
        end = self.end_edit.dateTime().toPython().replace(second=0, microsecond=0)
        if end <= start:
            QMessageBox.warning(self, "Invalid Times", "The end must be after the start.")
            return
        if end - start > SessionObservations.MAX_OBSERVATION_SPAN:
            QMessageBox.warning(self, "Invalid Times",
                                "One observation covers a single night (at most 24 hours). "
                                "Add another observation for the next night.")
            return

        night = self._night_date()
        original_night = self.observation.get("night_date")
        if self.observation.get("file_count") and night != original_night:
            QMessageBox.warning(
                self, "Night Can't Change",
                f"This observation has {self.observation['file_count']} attached file(s) taken on the "
                f"night of {original_night}, so it can't move to another night. Adjust the times only.",
            )
            return
        if night != original_night and night in self.taken_nights:
            QMessageBox.warning(self, "Night Already Logged",
                                f"There is already an observation for the night of {night}. Edit that one instead.")
            return

        logged = []
        for row in range(self.logged_table.rowCount()):
            filter_name = self.logged_table.cellWidget(row, 0).currentText().strip()
            exposure = self.logged_table.cellWidget(row, 1).value()
            count = self.logged_table.cellWidget(row, 2).value()
            if count == 0:
                continue
            if exposure <= 0:
                QMessageBox.warning(self, "Invalid Exposure", f"Row {row + 1}: the exposure must be more than 0 seconds.")
                return
            logged.append({"filter_name": filter_name, "exposure_seconds": exposure, "sub_count": count})

        self.result = {
            "night_date": night,
            "start_datetime": SessionObservations.format_datetime(start),
            "end_datetime": SessionObservations.format_datetime(end),
            "notes": self.notes_edit.toPlainText().strip(),
            "logged": logged,
        }
        self.accept()


class SessionDetailsDialog(WindowPositionMixin, QDialog):
    """Everything about one session: its editable fields (Overview), its
    observations with per-filter counted totals (Observations), whole-session
    per-filter totals (Filter Summary) and every attached file (Files).

    Observation adds/edits/deletes are held in memory and written together with
    the session's fields in one transaction on Save. Attaching or removing files
    scans/changes data on disk-backed rows, so those are written immediately
    (saving any pending edits first)."""

    WINDOW_POSITION_KEY = "SessionDetailsDialog"
    OVERVIEW_TAB, OBSERVATIONS_TAB, FILTERS_TAB, FILES_TAB = range(4)

    def __init__(self, session_data, start_tab=0, start_add=False, parent=None):
        super().__init__(parent)
        self.db_manager = DatabaseManager()
        self.session_data = dict(session_data)
        self.session_id = session_data["id"]
        self.data_changed = False
        self._observations = []
        self._deleted_ids = []
        self._files = []
        self._session_meta = {}
        self._obs_dirty = False
        self._added_any = False
        self._scan_thread = None

        self.setWindowTitle(f"{session_data.get('dso_name', '')} — {session_data.get('session_date', '')}")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        self.setModal(True)
        self.resize(1000, 720)  # default size the first time this dialog is ever opened

        self._setup_ui()
        self._enable_file_drops()
        self._reload_from_db()
        self._form_snapshot = self.form.snapshot()
        self.tabs.setCurrentIndex(start_tab)
        self.setup_window_position()
        if start_add:
            QTimer.singleShot(0, self._add_observation)

    # ---- UI -------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_overview_tab(), "Overview")
        self.tabs.addTab(self._build_observations_tab(), "Observations")
        self.tabs.addTab(self._build_filters_tab(), "Filter Summary")
        self.tabs.addTab(self._build_files_tab(), "Files")
        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        note = QLabel("Drop FITS/XISF files or folders anywhere here to attach them. "
                      "Attaching or removing files is saved immediately.")
        note.setStyleSheet(f"color: {COLORS['text_secondary']};")
        buttons.addWidget(note)
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        save_btn = QPushButton("Save Changes")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)

    def _build_overview_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.form = SessionFormWidget(session_data=self.session_data)
        layout.addWidget(self.form, 3)

        summary_group = QGroupBox("Summary")
        summary_form = QFormLayout()
        self.summary_labels = {}
        for key, label in (("subs", "Light Subs:"), ("integration", "Integration:"), ("nights", "Nights:"),
                           ("filters", "Filters:"), ("range", "Date Range:"), ("files", "Attached Files:"),
                           ("target", "Linked Target:"), ("created", "Created:"), ("modified", "Modified:")):
            value_label = QLabel()
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            summary_form.addRow(label, value_label)
            self.summary_labels[key] = value_label
        summary_group.setLayout(summary_form)

        summary_column = QVBoxLayout()
        summary_column.addWidget(summary_group)
        summary_column.addStretch()
        layout.addLayout(summary_column, 2)
        return tab

    def _build_observations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hint = QLabel("Each observation is one observing night - a night that runs past midnight stays one "
                      "observation. Hand-logged subs and scanned files with the same filter and exposure are the "
                      "same subs, so the larger count is used, never both.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add Observation...")
        add_btn.clicked.connect(self._add_observation)
        buttons.addWidget(add_btn)
        self.edit_obs_btn = QPushButton("Edit...")
        self.edit_obs_btn.clicked.connect(self._edit_observation)
        buttons.addWidget(self.edit_obs_btn)
        self.delete_obs_btn = QPushButton("Delete")
        self.delete_obs_btn.clicked.connect(self._delete_observation)
        buttons.addWidget(self.delete_obs_btn)
        buttons.addStretch()
        attach_btn = QPushButton("Attach Files ▾")
        attach_menu = QMenu(attach_btn)
        attach_menu.addAction("Sub Files...").triggered.connect(lambda: self._attach_files(folder=False))
        attach_menu.addAction("Folder...").triggered.connect(lambda: self._attach_files(folder=True))
        attach_btn.setMenu(attach_menu)
        buttons.addWidget(attach_btn)
        layout.addLayout(buttons)

        splitter = QSplitter(Qt.Vertical)
        self.obs_table = _make_table(["Night", "Filters", "Subs", "Integration", "Source", "Notes"])
        self.obs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.obs_table.itemSelectionChanged.connect(self._on_observation_selected)
        self.obs_table.itemDoubleClicked.connect(lambda _item: self._edit_observation())
        splitter.addWidget(self.obs_table)

        breakdown_widget = QWidget()
        breakdown_layout = QVBoxLayout(breakdown_widget)
        breakdown_layout.setContentsMargins(0, 0, 0, 0)
        self.breakdown_label = QLabel("Select an observation to see its per-filter breakdown.")
        breakdown_layout.addWidget(self.breakdown_label)
        self.breakdown_table = _make_table(["Filter", "Exposure", "Files", "Logged", "Counted", "Integration"])
        self.breakdown_table.setSelectionMode(QAbstractItemView.NoSelection)
        breakdown_layout.addWidget(self.breakdown_table)
        splitter.addWidget(breakdown_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return tab

    def _build_filters_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.filter_table = _make_table(["Filter", "Subs", "Integration", "Nights"])
        self.filter_table.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.filter_table)
        return tab

    def _build_files_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.files_count_label = QLabel()
        layout.addWidget(self.files_count_label)
        self.files_table = _make_table(["File", "Night", "Frame", "Filter", "Exposure", "Gain", "Temp", "Binning"])
        self.files_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.files_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.files_table.customContextMenuRequested.connect(self._show_files_menu)
        layout.addWidget(self.files_table)
        return tab

    # ---- Data -----------------------------------------------------------

    def _reload_from_db(self):
        try:
            with self.db_manager.get_connection() as conn:
                self._observations = SessionObservations.load_observations(conn, self.session_id)
                self._files = SessionObservations.load_session_files(conn, self.session_id)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.created_date, s.modified_date, t.name
                    FROM usersessions s LEFT JOIN usertargetlist t ON t.id = s.target_id
                    WHERE s.id = ?
                """, (self.session_id,))
                row = cursor.fetchone()
                self._session_meta = {"created": row[0], "modified": row[1], "target": row[2]} if row else {}
        except Exception as e:
            logger.error(f"Error loading session details: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load session details: {e}")
        for obs in self._observations:
            obs["state"] = None
        self._deleted_ids = []
        self._obs_dirty = False
        self._added_any = False
        self._refresh_all()

    def _visible_files(self):
        """Files not belonging to an observation that's pending deletion."""
        deleted = set(self._deleted_ids)
        return [f for f in self._files if f["observation_id"] not in deleted]

    def _loose_light_files(self):
        """Light files that couldn't be placed on a night (no DATE-OBS)."""
        return [f for f in self._files if f["observation_id"] is None and f["frame_type"] == "Light"]

    @staticmethod
    def _breakdown(obs):
        return SessionObservations.combine_breakdown(obs["file_groups"], obs["logged"])

    def _filter_totals(self):
        """[(display name, subs, seconds, nights)] across the session, counted values."""
        totals = {}
        for obs in self._observations:
            for row in self._breakdown(obs):
                if not row["counted_subs"]:
                    continue
                key = SessionObservations.normalize_filter(row["filter_name"])
                entry = totals.setdefault(key, {"name": row["filter_name"] or "(no filter)",
                                                "subs": 0, "seconds": 0.0, "nights": set()})
                entry["subs"] += row["counted_subs"]
                entry["seconds"] += row["counted_seconds"]
                entry["nights"].add(obs["night_date"])
        for f in self._loose_light_files():
            key = SessionObservations.normalize_filter(f["filter_name"])
            entry = totals.setdefault(key, {"name": f["filter_name"] or "(no filter)",
                                            "subs": 0, "seconds": 0.0, "nights": set()})
            entry["subs"] += 1
            entry["seconds"] += f["exptime_seconds"] or 0
        return sorted(((e["name"], e["subs"], e["seconds"], len(e["nights"])) for e in totals.values()),
                      key=lambda t: t[0].lower())

    def _filter_choices(self):
        choices = {f.strip() for f in (self.form.filters_edit.text() or "").split(",") if f.strip()}
        for obs in self._observations:
            choices.update(g["filter_name"] for g in obs["file_groups"] if g["filter_name"])
            choices.update(l["filter_name"] for l in obs["logged"] if l["filter_name"])
        return choices

    def _has_unsaved_changes(self):
        return self._obs_dirty or self.form.snapshot() != self._form_snapshot

    # ---- Refresh --------------------------------------------------------

    def _refresh_all(self):
        self._observations.sort(key=lambda o: o["night_date"])
        self._refresh_observations_table()
        self._refresh_filter_table()
        self._refresh_files_table()
        self._refresh_summary()
        has_observations = bool(self._observations)
        self.form.filters_edit.setReadOnly(has_observations)
        self.form.filters_edit.setToolTip(
            "Calculated from this session's observations." if has_observations else "")

    def _refresh_observations_table(self, select_obs=None):
        if select_obs is None:
            select_obs = self._selected_observation()
        self.obs_table.setRowCount(len(self._observations))
        select_row = -1
        for row, obs in enumerate(self._observations):
            subs, seconds, filters = SessionObservations.summarize_breakdown(self._breakdown(obs))
            has_files = obs["file_count"] > 0
            has_logged = any(l["sub_count"] for l in obs["logged"])
            source = ("Files + Logged" if has_files and has_logged else
                      "Files" if has_files else "Logged" if has_logged else "—")
            night_text = SessionObservations.format_night_span(
                obs["night_date"], obs["start_datetime"], obs["end_datetime"])
            if obs["state"]:
                night_text += "  *"
            notes = (obs["notes"] or "").splitlines()[0] if obs["notes"] else ""
            self.obs_table.setItem(row, 0, _table_item(night_text, tooltip="* unsaved" if obs["state"] else None))
            self.obs_table.setItem(row, 1, _table_item(", ".join(filters)))
            self.obs_table.setItem(row, 2, _table_item(str(subs), Qt.AlignCenter))
            self.obs_table.setItem(row, 3, _table_item(format_duration(seconds), Qt.AlignCenter))
            self.obs_table.setItem(row, 4, _table_item(source, Qt.AlignCenter))
            self.obs_table.setItem(row, 5, _table_item(notes, tooltip=obs["notes"] or None))
            if obs is select_obs:
                select_row = row
        if select_row < 0 and self._observations:
            select_row = 0
        if select_row >= 0:
            self.obs_table.selectRow(select_row)
        self._on_observation_selected()

    def _selected_observation(self):
        row = self.obs_table.currentRow()
        if 0 <= row < len(self._observations):
            return self._observations[row]
        return None

    def _on_observation_selected(self):
        obs = self._selected_observation()
        self.edit_obs_btn.setEnabled(obs is not None)
        self.delete_obs_btn.setEnabled(obs is not None)
        if obs is None:
            self.breakdown_label.setText("Select an observation to see its per-filter breakdown.")
            self.breakdown_table.setRowCount(0)
            return

        rows = self._breakdown(obs)
        self.breakdown_label.setText(
            f"<b>Night of {obs['night_date']}</b> — {obs['file_count']} attached file(s)")
        self.breakdown_table.setRowCount(len(rows) + (1 if rows else 0))
        for row, r in enumerate(rows):
            self.breakdown_table.setItem(row, 0, _table_item(r["filter_name"] or "(no filter)"))
            self.breakdown_table.setItem(row, 1, _table_item(format_exposure(r["exposure_seconds"]), Qt.AlignCenter))
            self.breakdown_table.setItem(row, 2, _table_item(str(r["file_subs"]), Qt.AlignCenter))
            self.breakdown_table.setItem(row, 3, _table_item(str(r["logged_subs"]), Qt.AlignCenter))
            self.breakdown_table.setItem(row, 4, _table_item(str(r["counted_subs"]), Qt.AlignCenter))
            self.breakdown_table.setItem(row, 5, _table_item(format_duration(r["counted_seconds"]), Qt.AlignCenter))
        if rows:
            subs, seconds, _filters = SessionObservations.summarize_breakdown(rows)
            self._set_total_row(self.breakdown_table, len(rows),
                                ["Total", "", str(sum(r["file_subs"] for r in rows)),
                                 str(sum(r["logged_subs"] for r in rows)), str(subs), format_duration(seconds)])

    @staticmethod
    def _set_total_row(table, row, values):
        bold = QFont()
        bold.setBold(True)
        for col, value in enumerate(values):
            item = _table_item(value, Qt.AlignCenter if col else None)
            item.setFont(bold)
            table.setItem(row, col, item)

    def _refresh_filter_table(self):
        totals = self._filter_totals()
        self.filter_table.setRowCount(len(totals) + (1 if totals else 0))
        for row, (name, subs, seconds, nights) in enumerate(totals):
            self.filter_table.setItem(row, 0, _table_item(name))
            self.filter_table.setItem(row, 1, _table_item(str(subs), Qt.AlignCenter))
            self.filter_table.setItem(row, 2, _table_item(format_duration(seconds), Qt.AlignCenter))
            self.filter_table.setItem(row, 3, _table_item(str(nights), Qt.AlignCenter))
        if totals:
            self._set_total_row(self.filter_table, len(totals), [
                "Total", str(sum(t[1] for t in totals)), format_duration(sum(t[2] for t in totals)),
                str(len(self._observations)),
            ])

    def _refresh_files_table(self):
        files = self._visible_files()
        self.files_table.setRowCount(len(files))
        for row, f in enumerate(files):
            path = f["file_path"] or ""
            self.files_table.setItem(row, 0, _table_item(os.path.basename(path), user_data=f["id"], tooltip=path))
            night = f["night_date"] or ("—" if f["date_obs"] else "(no date)")
            self.files_table.setItem(row, 1, _table_item(night, Qt.AlignCenter))
            self.files_table.setItem(row, 2, _table_item(f["frame_type"] or "", Qt.AlignCenter))
            self.files_table.setItem(row, 3, _table_item(f["filter_name"] or "", Qt.AlignCenter))
            exposure = format_exposure(f["exptime_seconds"]) if f["exptime_seconds"] is not None else ""
            self.files_table.setItem(row, 4, _table_item(exposure, Qt.AlignCenter))
            gain = f"{f['gain']:g}" if f["gain"] is not None else ""
            self.files_table.setItem(row, 5, _table_item(gain, Qt.AlignCenter))
            temp = f"{f['ccd_temp']:.1f}°C" if f["ccd_temp"] is not None else ""
            self.files_table.setItem(row, 6, _table_item(temp, Qt.AlignCenter))
            binning = f"{f['xbinning']}x{f['ybinning'] or f['xbinning']}" if f["xbinning"] else ""
            self.files_table.setItem(row, 7, _table_item(binning, Qt.AlignCenter))

        frame_counts = Counter(f["frame_type"] or "Unknown" for f in files)
        breakdown = ", ".join(f"{count} {frame}" for frame, count in frame_counts.most_common())
        self.files_count_label.setText(
            f"{len(files)} attached file(s)" + (f" — {breakdown}" if breakdown else "")
            + ("   (right-click to open a folder or remove files)" if files else ""))

    def _refresh_summary(self):
        totals = self._filter_totals()
        subs = sum(t[1] for t in totals)
        seconds = sum(t[2] for t in totals)
        starts = [o["start_datetime"] or o["night_date"] for o in self._observations]
        ends = [o["end_datetime"] or o["night_date"] for o in self._observations]
        if starts:
            first, last = min(starts).replace("T", " ")[:16], max(ends).replace("T", " ")[:16]
            date_range = first if first == last else f"{first} → {last}"
        else:
            date_range = "(no observations yet)"

        labels = self.summary_labels
        labels["subs"].setText(str(subs))
        labels["integration"].setText(f"{format_duration(seconds)}  ({seconds / 3600.0:.2f} h)")
        labels["nights"].setText(str(len(self._observations)))
        labels["filters"].setText(", ".join(t[0] for t in totals) or "(none)")
        labels["range"].setText(date_range)
        labels["files"].setText(str(len(self._visible_files())))
        labels["target"].setText(self._session_meta.get("target") or "(not linked)")
        labels["created"].setText(self._session_meta.get("created") or "")
        labels["modified"].setText(self._session_meta.get("modified") or "")

    # ---- Observation edits (held in memory until Save) -------------------

    def _mark_dirty(self, select_obs=None):
        self._obs_dirty = True
        self._observations.sort(key=lambda o: o["night_date"])
        self._refresh_observations_table(select_obs=select_obs)
        self._refresh_filter_table()
        self._refresh_files_table()
        self._refresh_summary()

    def _add_observation(self):
        self.tabs.setCurrentIndex(self.OBSERVATIONS_TAB)
        taken = {o["night_date"] for o in self._observations}
        default_date = self.form.date_edit.date().toPython()
        if default_date.isoformat() in taken:
            default_date = date_cls.today()
        dialog = ObservationDialog(default_date=default_date, filter_choices=self._filter_choices(),
                                   taken_nights=taken, parent=self)
        if dialog.exec() != QDialog.Accepted or not dialog.result:
            return
        obs = {"id": None, "state": "new", "is_manual": True, "file_groups": [], "file_count": 0}
        obs.update(dialog.result)
        self._observations.append(obs)
        self._added_any = True
        self._mark_dirty(select_obs=obs)

    def _edit_observation(self):
        obs = self._selected_observation()
        if obs is None:
            return
        taken = {o["night_date"] for o in self._observations if o is not obs}
        dialog = ObservationDialog(observation=obs, filter_choices=self._filter_choices(),
                                   taken_nights=taken, parent=self)
        if dialog.exec() != QDialog.Accepted or not dialog.result:
            return
        obs.update(dialog.result)
        if obs["id"] is not None:
            obs["state"] = "modified"
        self._mark_dirty(select_obs=obs)

    def _delete_observation(self):
        obs = self._selected_observation()
        if obs is None:
            return
        message = f"Delete the observation for the night of {obs['night_date']}?"
        if obs["file_count"]:
            message += (f"\n\nIts {obs['file_count']} attached file(s) will also be removed from this session "
                        f"(the files on disk are not touched).")
        message += "\n\nThis takes effect when you click Save Changes."
        if QMessageBox.question(self, "Delete Observation", message,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._observations.remove(obs)
        if obs["id"] is not None:
            self._deleted_ids.append(obs["id"])
        self._mark_dirty()

    # ---- Saving ---------------------------------------------------------

    @staticmethod
    def _insert_logged(cursor, observation_id, logged_rows):
        for logged in logged_rows:
            cursor.execute("""
                INSERT INTO usersessionobservationfilters (observation_id, filter_name, exposure_seconds, sub_count)
                VALUES (?, ?, ?, ?)
            """, (observation_id, logged["filter_name"], logged["exposure_seconds"], logged["sub_count"]))

    def _commit(self):
        """Write the form and every pending observation change in one transaction."""
        if not self.form.validate():
            self.tabs.setCurrentIndex(self.OVERVIEW_TAB)
            return False
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                self.form.write(conn)
                for obs_id in self._deleted_ids:
                    SessionObservations.delete_observation(conn, obs_id)
                for obs in self._observations:
                    if obs["state"] == "modified":
                        cursor.execute("""
                            UPDATE usersessionobservations SET night_date = ?, start_datetime = ?, end_datetime = ?,
                                notes = ?, is_manual = 1, modified_date = ?
                            WHERE id = ?
                        """, (obs["night_date"], obs["start_datetime"], obs["end_datetime"], obs["notes"],
                              now, obs["id"]))
                        cursor.execute("DELETE FROM usersessionobservationfilters WHERE observation_id = ?",
                                       (obs["id"],))
                        self._insert_logged(cursor, obs["id"], obs["logged"])
                for obs in self._observations:
                    if obs["state"] == "new":
                        cursor.execute("""
                            INSERT INTO usersessionobservations
                                (session_id, night_date, start_datetime, end_datetime, notes, is_manual, modified_date)
                            VALUES (?, ?, ?, ?, ?, 1, ?)
                        """, (self.session_id, obs["night_date"], obs["start_datetime"], obs["end_datetime"],
                              obs["notes"], now))
                        self._insert_logged(cursor, cursor.lastrowid, obs["logged"])
                SessionObservations.assign_files_to_observations(conn, self.session_id)
                SessionObservations.recompute_session_aggregates(conn, self.session_id, promote=self._added_any)
                conn.commit()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error saving session details: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session: {e}")
            return False
        self.data_changed = True
        return True

    def _save(self):
        if self._commit():
            self.accept()

    def reject(self):
        if self._has_unsaved_changes():
            reply = QMessageBox.question(self, "Discard Changes?", "Discard your unsaved changes to this session?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        super().reject()

    def _save_pending_before(self, action):
        """Immediate writes (attach/remove files) first save any pending edits,
        so the dialog never mixes committed and uncommitted state."""
        if not self._has_unsaved_changes():
            return True
        reply = QMessageBox.question(
            self, "Save Changes First?",
            f"{action} is saved immediately, so your other unsaved changes will be saved first. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes or not self._commit():
            return False
        self._after_immediate_write()
        return True

    def _after_immediate_write(self):
        self.data_changed = True
        self._reload_from_db()
        self._form_snapshot = self.form.snapshot()

    # ---- Files (written immediately) -------------------------------------

    def _attach_files(self, folder):
        if not self._save_pending_before("Attaching files"):
            return
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Attach a Folder of Subs")
            paths = [path] if path else []
        else:
            patterns = " ".join(f"*{ext}" for ext in sorted(SessionFileScanner.SUPPORTED_EXTENSIONS))
            paths, _ = QFileDialog.getOpenFileNames(self, "Attach Sub Files", "", f"FITS/XISF Files ({patterns})")
        if paths:
            self._scan_and_attach(paths)

    def _enable_file_drops(self):
        """Accept dropped files/folders anywhere in the dialog. Text fields accept
        drops themselves (a dropped file would be pasted in as a path), so they're
        filtered: a drop carrying sub files or folders goes to attaching instead."""
        self.setAcceptDrops(True)
        for widget in self.findChildren(QWidget):
            if widget.acceptDrops():
                widget.installEventFilter(self)

    def _scan_running(self):
        return self._scan_thread is not None and self._scan_thread.isRunning()

    def _accept_file_drag(self, event):
        if not self._scan_running() and dropped_sub_paths(event.mimeData()):
            event.acceptProposedAction()
            return True
        return False

    def _handle_file_drop(self, event):
        paths = dropped_sub_paths(event.mimeData())
        if not paths or self._scan_running():
            return False
        event.acceptProposedAction()
        # Defer: showing dialogs inside dropEvent would keep the source app's
        # drag operation (e.g. Explorer) blocked until they close.
        QTimer.singleShot(0, lambda: self._attach_dropped(paths))
        return True

    def _attach_dropped(self, paths):
        if self._save_pending_before("Attaching files"):
            self._scan_and_attach(paths)

    def dragEnterEvent(self, event):
        if not self._accept_file_drag(event):
            event.ignore()

    def dragMoveEvent(self, event):
        if not self._accept_file_drag(event):
            event.ignore()

    def dropEvent(self, event):
        if not self._handle_file_drop(event):
            event.ignore()

    def eventFilter(self, obj, event):
        event_type = event.type()
        if event_type in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if self._accept_file_drag(event):
                return True
        elif event_type == QEvent.Type.Drop:
            if self._handle_file_drop(event):
                return True
        return super().eventFilter(obj, event)

    def _scan_and_attach(self, paths):
        self._progress_dialog = QProgressDialog("Scanning files...", "Cancel", 0, 0, self)
        self._progress_dialog.setWindowTitle("Attach Files")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.show()

        self._scan_thread = SessionScanThread(paths)
        self._scan_thread.progress.connect(self._on_scan_progress)
        self._scan_thread.scan_finished.connect(self._on_scan_finished)
        self._scan_thread.scan_error.connect(self._on_scan_error)
        self._progress_dialog.canceled.connect(self._scan_thread.terminate)
        self._scan_thread.start()

    def _on_scan_progress(self, current, total):
        if total > 0:
            self._progress_dialog.setMaximum(total)
            self._progress_dialog.setValue(current)

    def _on_scan_error(self, message):
        self._progress_dialog.close()
        QMessageBox.critical(self, "Scan Error", f"Failed to scan files: {message}")

    def _on_scan_finished(self, files):
        self._progress_dialog.close()
        if not files:
            QMessageBox.information(self, "No Files Found", "No supported FITS/XISF files were found.")
            return
        try:
            with self.db_manager.get_connection() as conn:
                report = SessionObservations.attach_files(conn, self.session_id, files)
                SessionObservations.assign_files_to_observations(conn, self.session_id)
                SessionObservations.recompute_session_aggregates(conn, self.session_id)
                conn.commit()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error attaching files: {e}")
            QMessageBox.critical(self, "Error", f"Failed to attach files: {e}")
            return
        self._after_immediate_write()
        QMessageBox.information(self, "Attach Files",
                                SessionObservations.format_attach_report(report)
                                or f"{report['inserted']} new file(s) attached.")

    def _selected_file_ids(self):
        rows = sorted({index.row() for index in self.files_table.selectedIndexes()})
        return [self.files_table.item(row, 0).data(Qt.UserRole) for row in rows if self.files_table.item(row, 0)]

    def _show_files_menu(self, position):
        item = self.files_table.itemAt(position)
        if not item:
            return
        if not self.files_table.item(item.row(), 0).isSelected():
            self.files_table.selectRow(item.row())
        path = self.files_table.item(item.row(), 0).toolTip()
        count = len(self._selected_file_ids())

        menu = QMenu(self)
        menu.addAction("Open Containing Folder").triggered.connect(lambda: self._open_folder(path))
        menu.addSeparator()
        menu.addAction(f"Remove {count} File(s) from Session...").triggered.connect(self._remove_selected_files)
        menu.exec(self.files_table.viewport().mapToGlobal(position))

    def _open_folder(self, path):
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Folder Not Found", f"The folder no longer exists:\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _remove_selected_files(self):
        file_ids = self._selected_file_ids()
        if not file_ids:
            return
        if QMessageBox.question(
                self, "Remove Files",
                f"Remove {len(file_ids)} file(s) from this session? The files on disk are not touched.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if not self._save_pending_before("Removing files"):
            return
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                for i in range(0, len(file_ids), 500):
                    chunk = file_ids[i:i + 500]
                    cursor.execute(f"DELETE FROM usersessionfiles WHERE session_id = ? AND id IN "
                                   f"({','.join('?' * len(chunk))})", [self.session_id] + chunk)
                SessionObservations.assign_files_to_observations(conn, self.session_id)
                SessionObservations.recompute_session_aggregates(conn, self.session_id, promote=False)
                conn.commit()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error removing session files: {e}")
            QMessageBox.critical(self, "Error", f"Failed to remove files: {e}")
            return
        self._after_immediate_write()


class SessionScanThread(QThread):
    """Background scan of dropped files/folders so large folders don't block the UI."""
    progress = Signal(int, int)
    scan_finished = Signal(list)
    scan_error = Signal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        try:
            all_files = []
            for path in self.paths:
                if os.path.isdir(path):
                    all_files.extend(SessionFileScanner.scan_folder(
                        path, progress_callback=lambda i, total: self.progress.emit(i, total)
                    ))
                elif os.path.splitext(path)[1].lower() in SessionFileScanner.SUPPORTED_EXTENSIONS:
                    data = SessionFileScanner.scan_file(path)
                    if data is not None:
                        all_files.append(data)
            self.scan_finished.emit(all_files)
        except Exception as e:
            logger.error(f"Error scanning dropped files: {e}")
            self.scan_error.emit(str(e))


class VisibilityCalcThread(QThread):
    """Runs a DSOVisibilityCalculator calculation off the GUI thread. main.py's
    startup pre-warm already exercises DSOVisibilityCalculator once, single-threaded,
    before any window exists - so by the time this thread ever runs, the one-time
    astropy/matplotlib setup cost has already been paid and there's no risk of racing
    that same cold-import machinery against another thread (e.g. the weather worker)."""
    result_ready = Signal(dict)

    def __init__(self, ra_deg, dec_deg, dso_name, location_lat, location_lon, location_timezone, visibility_date):
        super().__init__()
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.dso_name = dso_name
        self.location_lat = location_lat
        self.location_lon = location_lon
        self.location_timezone = location_timezone
        self.visibility_date = visibility_date

    def run(self):
        try:
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            import pytz

            calc = DSOVisibilityCalculator(
                location_lat=self.location_lat,
                location_lon=self.location_lon,
                timezone=self.location_timezone,
            )

            coord = None
            if self.ra_deg is not None and self.dec_deg is not None:
                coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
                result = calc.calculate_visibility_for_coordinates(coord, self.visibility_date, dso_name=self.dso_name)
            elif self.dso_name:
                result = calc.calculate_visibility_for_date(self.dso_name, self.visibility_date)
            else:
                self.result_ready.emit({"error": "No coordinates or name available."})
                return

            if result.get("error"):
                self.result_ready.emit({"error": result["error"]})
                return

            max_alt_time_local = result["max_alt_time"].to_datetime(timezone=pytz.UTC).astimezone(calc.timezone)
            lines = [f"Max altitude: {result['max_altitude']:.1f}° at {format_time(max_alt_time_local)}"]

            windows = result.get("viewing_windows", [])
            if windows:
                best_window = max(windows, key=lambda w: w["duration_hours"])
                start_local = best_window["start_time"].to_datetime(timezone=pytz.UTC).astimezone(calc.timezone)
                end_local = best_window["end_time"].to_datetime(timezone=pytz.UTC).astimezone(calc.timezone)
                lines.append(
                    f"Best window: {format_time(start_local)} - {format_time(end_local)} "
                    f"({best_window['duration_hours']:.1f}h above 30°, dark sky)"
                )
            else:
                lines.append("No window above 30° during dark sky on this date.")

            obs_time = result["max_alt_time"]
            illumination = DSOVisibilityCalculator.get_moon_illumination(obs_time)
            lines.append(f"Moon illumination: {illumination * 100:.0f}%")
            if coord is not None:
                separation = DSOVisibilityCalculator.get_moon_separation(coord, obs_time)
                if separation is not None:
                    lines.append(f"Moon separation: {separation:.0f}°")

            self.result_ready.emit({"lines": lines})
        except Exception as e:
            logger.error(f"Error calculating visibility: {e}")
            self.result_ready.emit({"error": f"Visibility calculation failed: {e}"})


class SessionMonthlyVisibilityThread(QThread):
    """Computes per-day visibility hours across a date range spanning one or more
    displayed calendar months, for a specific session's DSO/location. Modeled on
    DSOVisibilityCalculator.MonthlyVisibilityThread, but takes explicit location
    (a session's own location override, not the DB's default active location) and
    an arbitrary start/end date range instead of a single calendar month, since the
    calendar view can show 1-3 months at once. Runs off the GUI thread - a 3-month
    span is ~90 sequential per-day calls, the same per-call cost already proven
    acceptable for the existing single-month DSOVisibilityCalculator feature."""
    progress = Signal(int, int)  # days completed, total days
    # object, not dict: Signal(dict) marshals through QVariantMap across threads,
    # which requires string keys and silently drops non-string keys (our keys are
    # datetime.date objects) - see the same fix already applied in
    # DSOVisibilityCalculator.MonthlyVisibilityThread.finished.
    finished_calc = Signal(object)  # {date: hours}
    error = Signal(str)

    def __init__(self, ra_deg, dec_deg, dso_name, location_lat, location_lon, location_timezone,
                 start_date, end_date, min_altitude=30):
        super().__init__()
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.dso_name = dso_name
        self.location_lat = location_lat
        self.location_lon = location_lon
        self.location_timezone = location_timezone
        self.start_date = start_date
        self.end_date = end_date
        self.min_altitude = min_altitude

    def run(self):
        try:
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            from datetime import timedelta

            calc = DSOVisibilityCalculator(
                location_lat=self.location_lat,
                location_lon=self.location_lon,
                timezone=self.location_timezone,
            )
            if calc.location is None:
                self.error.emit("Observer location not configured.")
                return

            if self.ra_deg is not None and self.dec_deg is not None:
                coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
            elif self.dso_name:
                coord, err = calc.get_dso_coordinates_enhanced(self.dso_name)
                if coord is None:
                    self.error.emit(f"Could not find coordinates: {err}")
                    return
            else:
                self.error.emit("No coordinates or name available.")
                return

            total_days = (self.end_date - self.start_date).days + 1
            results = {}
            current = self.start_date
            completed = 0
            while current <= self.end_date:
                hours = calc.calculate_visibility_hours_for_day(coord, current.strftime("%Y-%m-%d"), self.min_altitude)
                results[current] = hours
                completed += 1
                self.progress.emit(completed, total_days)
                current += timedelta(days=1)

            self.finished_calc.emit(results)
        except Exception as e:
            logger.error(f"Error calculating monthly visibility: {e}")
            self.error.emit(str(e))


class SessionMonthCalendar(QCalendarWidget):
    """One month of the Session Manager calendar view. Combines three independent,
    date-keyed layers on the same cells: DSO visibility hours (background fill),
    weather astro score (a stripe across the top, only for the ~7 dates with a live
    forecast), and session markers (small status-colored dots). Reuses the
    paintCell()-override technique from DSOVisibilityCalculator.VisibilityCalendar
    rather than setDateTextFormat, which that class already found doesn't suit
    per-cell custom painting."""

    STATUS_COLOR_KEYS = {
        "Planned": "info",
        "In Progress": "warning",
        "Completed": "success",
        "Cancelled": "text_disabled",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.visibility_hours = {}   # date -> hours
        self.weather_scores = {}     # date -> astro_score (0-100)
        self.weather_details = {}    # date -> {"cloud_cover": float, "seeing": str}
        self.session_markers = {}    # date -> [(session_id, dso_name, status, night_seconds, night_filters), ...]
        self.setGridVisible(True)
        self.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self.setMouseTracking(True)

        self.tooltip_label = QLabel(self)
        self.tooltip_label.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.tooltip_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['background_lighter']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10pt;
            }}
        """)
        self.tooltip_label.hide()

        # QCalendarWidget renders its day grid via a private internal QTableView
        # subclass - overriding mouseMoveEvent/leaveEvent on the QCalendarWidget
        # itself never fires while hovering over day cells, since those mouse
        # events are consumed by that internal view and don't propagate up to us.
        # Install an event filter on the view's viewport instead, which is where
        # the events actually land.
        self._day_view = self.findChild(QTableView)
        if self._day_view is not None:
            self._day_view.setMouseTracking(True)
            self._day_view.viewport().setMouseTracking(True)
            self._day_view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if self._day_view is not None and obj is self._day_view.viewport():
            if event.type() == QEvent.Type.MouseMove:
                py_date = self._date_from_view_pos(event.pos())
                self._show_tooltip_for_date(py_date, self._day_view.viewport().mapToGlobal(event.pos()))
            elif event.type() == QEvent.Type.Leave:
                self.tooltip_label.hide()
        return super().eventFilter(obj, event)

    def _date_from_view_pos(self, pos_in_view):
        """QCalendarWidget has no public dateAt()/similar for this - its day grid
        is rendered by an internal QTableView whose model is private/undocumented.
        Resolve the date ourselves from the cell's (row, col) instead: the model's
        row 0 is a non-clickable weekday-name header (confirmed empirically - its
        cells are text like "Sun"/"Mon", not day numbers), so the real day grid
        starts at row 1, which is always firstDayOfWeek()-aligned - walk backward
        from the 1st of the shown month by however many leading days that needs."""
        index = self._day_view.indexAt(pos_in_view)
        if not index.isValid() or index.row() < 1:
            return None
        first_of_month = date_cls(self.yearShown(), self.monthShown(), 1)
        qt_first_day = self.firstDayOfWeek().value  # Qt.Monday=1 .. Qt.Sunday=7
        first_weekday = first_of_month.isoweekday()  # Python: Monday=1 .. Sunday=7
        # Qt always shows at least one full leading week from the previous month,
        # even when the 1st already falls exactly on the configured first day of
        # the week (confirmed empirically: e.g. Feb 2026 starts on a Sunday with
        # Sunday as the first day of the week, yet row 1 still shows the last week
        # of January, not Feb 1st) - so this must land in [1, 7], never 0.
        leading_days = ((first_weekday - qt_first_day - 1) % 7) + 1
        first_visible_date = first_of_month - timedelta(days=leading_days)
        cell_offset = (index.row() - 1) * 7 + index.column()
        return first_visible_date + timedelta(days=cell_offset)

    def set_data(self, visibility_hours=None, weather_scores=None, weather_details=None, session_markers=None):
        if visibility_hours is not None:
            self.visibility_hours = visibility_hours
        if weather_scores is not None:
            self.weather_scores = weather_scores
        if weather_details is not None:
            self.weather_details = weather_details
        if session_markers is not None:
            self.session_markers = session_markers
        self.updateCells()
        self.update()

    def clear_visibility(self):
        self.visibility_hours = {}
        self.updateCells()
        self.update()

    def paintCell(self, painter, rect, date):
        from PySide6.QtGui import QPen, QBrush
        from DSOVisibilityCalculator import visibility_hours_to_color
        from WeatherForecast import get_rating_color

        if date.month() != self.monthShown() or date.year() != self.yearShown():
            super().paintCell(painter, rect, date)
            return

        py_date = date.toPython()

        if py_date in self.visibility_hours:
            bg_color, fg_color = visibility_hours_to_color(self.visibility_hours[py_date])
        else:
            bg_color, fg_color = QColor(64, 64, 64), QColor(200, 200, 200)

        painter.fillRect(rect, QBrush(bg_color))

        weather_score = self.weather_scores.get(py_date)
        if weather_score is not None:
            stripe_rect = rect.adjusted(0, 0, 0, -(rect.height() - 6))
            painter.fillRect(stripe_rect, QBrush(QColor(get_rating_color(weather_score))))

        painter.setPen(QPen(fg_color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(date.day()))

        markers = self.session_markers.get(py_date)
        if markers:
            statuses = sorted({status for _, _, status, _, _ in markers})
            dot_radius = 3
            spacing = 8
            start_x = rect.right() - 6 - spacing * (len(statuses) - 1)
            dot_y = rect.bottom() - 8
            painter.setPen(Qt.NoPen)
            for i, status in enumerate(statuses):
                color_key = self.STATUS_COLOR_KEYS.get(status, "text_disabled")
                painter.setBrush(QBrush(QColor(COLORS[color_key])))
                painter.drawEllipse(QPoint(start_x + i * spacing, dot_y), dot_radius, dot_radius)

        if date == self.selectedDate():
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        elif py_date == date_cls.today():
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.tooltip_label.hide()

    def _show_tooltip_for_date(self, py_date, global_pos):
        """py_date is a plain datetime.date (or None); global_pos is the screen
        position to anchor the tooltip near."""
        from WeatherForecast import get_rating_label

        if py_date is None or py_date.month != self.monthShown() or py_date.year != self.yearShown():
            self.tooltip_label.hide()
            return

        lines = []
        if py_date in self.visibility_hours:
            lines.append(f"{self.visibility_hours[py_date]:.1f}h visible")
        if py_date in self.weather_scores:
            score = self.weather_scores[py_date]
            weather_line = f"Weather: {score}/100 ({get_rating_label(score)})"
            detail = self.weather_details.get(py_date)
            if detail:
                if detail.get("cloud_cover") is not None:
                    weather_line += f", {detail['cloud_cover']:.0f}% clouds"
                if detail.get("seeing"):
                    weather_line += f", seeing {detail['seeing']}"
            lines.append(weather_line)
        for _, dso_name, status, night_seconds, night_filters in self.session_markers.get(py_date, []):
            marker_line = f"{dso_name} - {status}"
            if night_seconds:
                marker_line += f" - {night_seconds / 3600.0:.1f}h"
            if night_filters:
                marker_line += f" ({night_filters})"
            lines.append(marker_line)

        if not lines:
            self.tooltip_label.hide()
            return

        self.tooltip_label.setText("\n".join(lines))
        self.tooltip_label.adjustSize()
        self.tooltip_label.move(global_pos.x() + 15, global_pos.y() + 15)
        self.tooltip_label.show()
        self.tooltip_label.raise_()


# Superseded worker threads that are still running. quit() can't interrupt a
# QThread whose work is in run() (no event loop), and wait()-ing for one froze
# the UI until it finished - e.g. clicking a second session mid-fetch. So they're
# abandoned instead: kept referenced here (PySide6 can hard-crash, natively, if a
# QThread is garbage-collected while its OS thread is still running) with their
# signals blocked so late results are dropped, and pruned once they've ended.
_retired_threads = set()


def _retire_thread(thread):
    _retired_threads.difference_update([t for t in _retired_threads if not t.isRunning()])
    if thread is not None and thread.isRunning():
        thread.blockSignals(True)
        _retired_threads.add(thread)


class SharedWeatherFetch:
    """At most one WeatherWorker per location, shared by the calendar's weather
    stripe and the details panel - selecting a session used to start two
    identical fetches. A running worker is referenced here until it ends (see
    _retired_threads for why), and its result is cached before callbacks run."""

    _workers = {}  # (lat, lon) rounded -> running WeatherWorker

    @classmethod
    def key(cls, lat, lon):
        return round(float(lat), 2), round(float(lon), 2)

    @classmethod
    def request(cls, lat, lon, timezone, on_loaded, on_error):
        from WeatherForecast import WeatherCache, WeatherWorker
        key = cls.key(lat, lon)
        worker = cls._workers.get(key)
        is_new = worker is None or not worker.isRunning()
        if is_new:
            worker = WeatherWorker(lat, lon, timezone)
            worker.weather_loaded.connect(lambda data: WeatherCache().set(lat, lon, data))
            worker.finished.connect(lambda w=worker: cls._workers.pop(key, None) if cls._workers.get(key) is w else None)
            cls._workers[key] = worker
        worker.weather_loaded.connect(on_loaded)
        worker.error_occurred.connect(on_error)
        if is_new:
            worker.start()


class SessionCalendarWidget(QWidget):
    """The Session Manager 'Calendar' tab: 1-3 side-by-side SessionMonthCalendar
    instances kept chronologically chained, with weather/visibility layers that
    follow the currently selected session. Nothing is fetched until a session
    is selected."""

    MONTH_COUNT_SETTING = "session_calendar_month_count"

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self._current_session = None
        self.calendars = []
        self.weather_scores = {}
        self.weather_details = {}
        self.visibility_hours = {}
        self.session_markers = {}
        self.on_date_with_session_clicked = None
        self._weather_key = None
        self._visibility_thread = None

        today = QDate.currentDate()
        self.anchor_year, self.anchor_month = today.year(), today.month()
        self.month_count = self._load_month_count()

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_controls())

        self.calendars_layout = QHBoxLayout()
        layout.addLayout(self.calendars_layout, 1)

        self._rebuild_calendars()

    def _build_controls(self):
        row = QHBoxLayout()

        row.addWidget(QLabel("Months:"))
        self.month_count_combo = QComboBox()
        self.month_count_combo.addItems(["1 Month", "2 Months", "3 Months"])
        self.month_count_combo.setCurrentIndex(self.month_count - 1)
        self.month_count_combo.currentIndexChanged.connect(self._on_month_count_changed)
        row.addWidget(self.month_count_combo)

        prev_btn = QPushButton("◀ Previous")
        prev_btn.clicked.connect(self._go_previous)
        row.addWidget(prev_btn)

        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._go_next)
        row.addWidget(next_btn)

        row.addStretch()

        row.addWidget(QLabel("Background: DSO visibility • Top stripe: weather • Dot: session"))
        for label, color in (
            ("Excellent", COLORS['success']), ("Good", COLORS['info']),
            ("Moderate", COLORS['warning']), ("Poor", COLORS['error']),
        ):
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid {COLORS['border']};")
            row.addWidget(swatch)
            row.addWidget(QLabel(label))

        return row

    def _load_month_count(self):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        return max(1, min(3, settings.value(self.MONTH_COUNT_SETTING, 1, type=int)))

    def _save_month_count(self, count):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue(self.MONTH_COUNT_SETTING, count)

    def _on_month_count_changed(self, index):
        self.month_count = index + 1
        self._save_month_count(self.month_count)
        self._rebuild_calendars()

    @staticmethod
    def _next_month(year, month):
        return (year + 1, 1) if month == 12 else (year, month + 1)

    @staticmethod
    def _prev_month(year, month):
        return (year - 1, 12) if month == 1 else (year, month - 1)

    def _display_range(self):
        start = date_cls(self.anchor_year, self.anchor_month, 1)
        last_year, last_month = self.anchor_year, self.anchor_month
        for _ in range(self.month_count - 1):
            last_year, last_month = self._next_month(last_year, last_month)
        last_day = calendar.monthrange(last_year, last_month)[1]
        end = date_cls(last_year, last_month, last_day)
        return start, end

    def _rebuild_calendars(self):
        for cal in self.calendars:
            cal.setParent(None)
            cal.deleteLater()
        self.calendars = []

        while self.calendars_layout.count():
            self.calendars_layout.takeAt(0)

        year, month = self.anchor_year, self.anchor_month
        for i in range(self.month_count):
            cal = SessionMonthCalendar()
            cal.setCurrentPage(year, month)
            if i == 0:
                cal.currentPageChanged.connect(self._on_calendar0_page_changed)
            else:
                cal.setNavigationBarVisible(False)
            cal.clicked.connect(self._on_date_clicked)
            self.calendars_layout.addWidget(cal)
            self.calendars.append(cal)
            year, month = self._next_month(year, month)

        self._trigger_recalculation()

    def _set_anchor(self, year, month):
        self.anchor_year, self.anchor_month = year, month
        y, m = year, month
        for cal in self.calendars:
            cal.blockSignals(True)
            cal.setCurrentPage(y, m)
            cal.blockSignals(False)
            y, m = self._next_month(y, m)
        self._trigger_recalculation()

    def _on_calendar0_page_changed(self, year, month):
        self._set_anchor(year, month)

    def _go_previous(self):
        year, month = self._prev_month(self.anchor_year, self.anchor_month)
        self._set_anchor(year, month)

    def _go_next(self):
        year, month = self._next_month(self.anchor_year, self.anchor_month)
        self._set_anchor(year, month)

    def _on_date_clicked(self, qdate):
        py_date = qdate.toPython()
        markers = self.session_markers.get(py_date)
        if not markers or not self.on_date_with_session_clicked:
            return
        session_id = markers[0][0]
        self.on_date_with_session_clicked(session_id)

    def set_active_session(self, session):
        self._current_session = session
        self._trigger_recalculation()

    def _trigger_recalculation(self):
        self.refresh_session_markers()
        self._refresh_weather_layer()
        self._refresh_visibility_layer()

    def _refresh_weather_layer(self):
        # Weather is only fetched for a selected session - opening the Session
        # Manager doesn't start a download.
        session = self._current_session
        lat = session.get("location_lat") if session else None
        lon = session.get("location_lon") if session else None
        if lat is None or lon is None:
            self._weather_key = None
            self._apply_weather_scores({}, {})
            return

        from WeatherForecast import WeatherCache

        cached = WeatherCache().get(lat, lon)
        self._weather_key = SharedWeatherFetch.key(lat, lon)
        if cached:
            self._apply_weather_list(cached)
            return
        key = self._weather_key
        SharedWeatherFetch.request(
            lat, lon, session.get("location_timezone"),
            lambda data: self._on_weather_loaded(data, key),
            lambda msg: logger.debug(f"Calendar weather fetch failed: {msg}"))

    def _on_weather_loaded(self, data, key):
        if key != self._weather_key:
            return  # a different session/location has been selected since
        try:
            self._apply_weather_list(data)
        except RuntimeError:
            pass  # the window was closed while the fetch ran

    def _apply_weather_list(self, daily_summaries):
        scores = {}
        details = {}
        for summary in daily_summaries:
            d = summary.date.date()
            scores[d] = summary.astro_score
            details[d] = {
                "cloud_cover": summary.tonight_avg_cloud_cover,
                "seeing": summary.seeing_estimate,
            }
        self._apply_weather_scores(scores, details)

    def _apply_weather_scores(self, scores, details=None):
        self.weather_scores = scores
        self.weather_details = details if details is not None else {}
        for cal in self.calendars:
            cal.set_data(weather_scores=scores, weather_details=self.weather_details)

    def _refresh_visibility_layer(self):
        session = self._current_session
        if not session:
            self.visibility_hours = {}
            for cal in self.calendars:
                cal.clear_visibility()
            return

        start_date, end_date = self._display_range()

        _retire_thread(self._visibility_thread)
        self._visibility_thread = SessionMonthlyVisibilityThread(
            session.get("ra_deg"), session.get("dec_deg"), session.get("dso_name"),
            session.get("location_lat"), session.get("location_lon"), session.get("location_timezone"),
            start_date, end_date,
        )
        self._visibility_thread.finished_calc.connect(self._on_visibility_calc_finished)
        self._visibility_thread.error.connect(
            lambda msg: logger.debug(f"Calendar visibility calc failed: {msg}")
        )
        self._visibility_thread.start()

    def _on_visibility_calc_finished(self, hours_by_date):
        self.visibility_hours = hours_by_date
        for cal in self.calendars:
            cal.set_data(visibility_hours=hours_by_date)

    def refresh_session_markers(self):
        start_date, end_date = self._display_range()
        self.session_markers = self._query_session_markers(start_date, end_date)
        for cal in self.calendars:
            cal.set_data(session_markers=self.session_markers)

    def _query_session_markers(self, start_date, end_date):
        """Returns date -> [(session_id, dso_name, status, night_integration_seconds,
        night_filters), ...], one entry per observation (observing night). A night
        that runs past midnight (e.g. Friday 21:00 to Saturday 04:00) is one
        observation, and its complete counted totals are shown on EVERY calendar
        date it touches (both Friday and Saturday) rather than split across them."""
        markers = {}
        try:
            # A night starting the evening before start_date can reach into it.
            padded_start = start_date - timedelta(days=1)

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT o.id, o.session_id, s.status, s.dso_name, o.night_date,
                           o.start_datetime, o.end_datetime
                    FROM usersessionobservations o
                    JOIN usersessions s ON s.id = o.session_id
                    WHERE o.night_date BETWEEN ? AND ?
                """, (padded_start.isoformat(), end_date.isoformat()))
                observations = cursor.fetchall()
                breakdowns = SessionObservations.load_breakdowns(conn, [row[0] for row in observations])

                for obs_id, session_id, status, dso_name, night_date, start_dt, end_dt in observations:
                    _subs, seconds, filters = SessionObservations.summarize_breakdown(breakdowns[obs_id])
                    marker = (session_id, dso_name, status, seconds, ",".join(filters) or None)
                    try:
                        night = datetime.strptime(night_date, "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        continue
                    start = SessionObservations.parse_obs_datetime(start_dt)
                    end = SessionObservations.parse_obs_datetime(end_dt)
                    first_day = start.date() if start else night
                    last_day = end.date() if start and end and end >= start else first_day
                    day = first_day
                    while day <= last_day:
                        if start_date <= day <= end_date:
                            markers.setdefault(day, []).append(marker)
                        day += timedelta(days=1)

                cursor.execute("SELECT DISTINCT session_id FROM usersessionobservations")
                sessions_with_observations = {row[0] for row in cursor.fetchall()}

                cursor.execute("""
                    SELECT id, session_date, status, dso_name FROM usersessions
                    WHERE session_date BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                for session_id, session_date, status, dso_name in cursor.fetchall():
                    if session_id in sessions_with_observations:
                        continue
                    try:
                        day = datetime.strptime(session_date, "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        continue
                    markers.setdefault(day, []).append((session_id, dso_name, status, 0, None))
        except Exception as e:
            logger.error(f"Error querying session markers: {e}")
        return markers


class SessionManagerWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "SessionManager"
    COLUMNS_SETTING = "session_manager_table_columns_hidden"
    """Main window for planning and logging observing sessions"""

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("Session Manager - Cosmos Collection")
        self.resize(1200, 800)
        self.setup_window_position()

        self.db_manager = DatabaseManager()
        self.sessions_data = []
        self._current_session = None
        self._visibility_thread = None
        self._weather_key = None  # location of the forecast the details panel is waiting for
        self._init_database()
        self._init_ui()
        self._load_sessions()

    def _init_database(self):
        """Belt-and-suspenders table creation, mirroring DSOTargetListWindow's own
        _init_database() on top of DatabaseManager's copy."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usersessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        target_id INTEGER REFERENCES usertargetlist(id) ON DELETE SET NULL,
                        dso_name TEXT NOT NULL,
                        ra_deg REAL,
                        dec_deg REAL,
                        status TEXT NOT NULL DEFAULT 'Planned',
                        session_date TEXT NOT NULL,
                        start_time TEXT,
                        end_time TEXT,
                        location_lat REAL,
                        location_lon REAL,
                        location_name TEXT,
                        location_timezone TEXT,
                        location_source TEXT DEFAULT 'active',
                        telescope_id INTEGER REFERENCES usertelescopes(id) ON DELETE SET NULL,
                        camera TEXT,
                        filters_used TEXT,
                        sub_count INTEGER DEFAULT 0,
                        integration_seconds REAL DEFAULT 0,
                        earliest_sub_date TEXT,
                        latest_sub_date TEXT,
                        notes TEXT,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        modified_date TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usersessionfiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL REFERENCES usersessions(id) ON DELETE CASCADE,
                        file_path TEXT NOT NULL,
                        file_type TEXT,
                        frame_type TEXT,
                        object_name TEXT,
                        date_obs TEXT,
                        exptime_seconds REAL,
                        filter_name TEXT,
                        camera TEXT,
                        telescope TEXT,
                        gain REAL,
                        offset_value REAL,
                        ccd_temp REAL,
                        xbinning INTEGER,
                        ybinning INTEGER,
                        header_json TEXT,
                        added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(session_id, file_path)
                    )
                """)
                conn.commit()
                # Observation tables, then place any not-yet-grouped files on their nights.
                SessionObservations.migrate(conn)
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error initializing session manager database: {e}")

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout(central_widget)

        header_label = QLabel("Session Manager")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(header_label)

        control_group = QGroupBox("Session Management")
        control_layout = QVBoxLayout()

        search_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search DSO name...")
        self.search_box.setFixedWidth(280)
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._filter_sessions)
        search_row.addWidget(QLabel("Search:"))
        search_row.addWidget(self.search_box)
        search_row.addStretch()
        control_layout.addLayout(search_row)

        buttons_row = QHBoxLayout()
        new_planned_btn = QPushButton("New Planned Session")
        new_planned_btn.clicked.connect(self._new_planned_session)
        buttons_row.addWidget(new_planned_btn)

        log_past_btn = QPushButton("Log Past Session")
        log_past_btn.clicked.connect(self._log_past_session)
        buttons_row.addWidget(log_past_btn)

        self.edit_session_btn = QPushButton("Edit Selected")
        self.edit_session_btn.clicked.connect(self._edit_selected_session)
        self.edit_session_btn.setEnabled(False)
        buttons_row.addWidget(self.edit_session_btn)

        self.duplicate_session_btn = QPushButton("Duplicate")
        self.duplicate_session_btn.clicked.connect(self._duplicate_selected_session)
        self.duplicate_session_btn.setEnabled(False)
        buttons_row.addWidget(self.duplicate_session_btn)

        self.delete_session_btn = QPushButton("Delete Selected")
        self.delete_session_btn.clicked.connect(self._delete_selected_session)
        self.delete_session_btn.setEnabled(False)
        buttons_row.addWidget(self.delete_session_btn)

        buttons_row.addStretch()

        buttons_row.addWidget(QLabel("Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Planned", "In Progress", "Completed", "Cancelled"])
        self.status_filter.currentTextChanged.connect(self._filter_sessions)
        buttons_row.addWidget(self.status_filter)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_sessions)
        buttons_row.addWidget(refresh_btn)

        control_layout.addLayout(buttons_row)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        splitter = QSplitter(Qt.Horizontal)

        self.sessions_table = QTableWidget()
        self._column_labels = [
            "DSO Name", "Status", "Date", "Time", "Location", "Telescope",
            "Filters", "Subs", "Integration", "Linked Target"
        ]
        self.sessions_table.setColumnCount(len(self._column_labels))
        self.sessions_table.setHorizontalHeaderLabels(self._column_labels)
        self.sessions_table.setSortingEnabled(True)
        header = self.sessions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, 10):
            header.setSectionResizeMode(col, QHeaderView.Interactive)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_column_menu)
        self.sessions_table.setAlternatingRowColors(True)
        self.sessions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sessions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sessions_table.itemDoubleClicked.connect(self._edit_selected_session)
        self.sessions_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sessions_table.customContextMenuRequested.connect(self._show_context_menu)
        self._apply_saved_column_visibility()
        splitter.addWidget(self.sessions_table)

        self.detail_panel = self._build_detail_panel()
        splitter.addWidget(self.detail_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        sessions_tab = QWidget()
        sessions_tab_layout = QVBoxLayout(sessions_tab)
        sessions_tab_layout.setContentsMargins(0, 0, 0, 0)
        sessions_tab_layout.addWidget(splitter)

        self.calendar_widget = SessionCalendarWidget(self.db_manager)
        self.calendar_widget.on_date_with_session_clicked = self._on_calendar_date_clicked

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(sessions_tab, "Sessions")
        self.tab_widget.addTab(self.calendar_widget, "Calendar")
        main_layout.addWidget(self.tab_widget, 1)

        self.sessions_table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def _build_detail_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        info_group = QGroupBox("Session Info")
        info_layout = QVBoxLayout()
        self.detail_info_label = QLabel("Select a session to view details.")
        self.detail_info_label.setWordWrap(True)
        info_layout.addWidget(self.detail_info_label)
        self.detail_equipment_label = QLabel("")
        self.detail_equipment_label.setWordWrap(True)
        info_layout.addWidget(self.detail_equipment_label)
        self.view_target_btn = QPushButton("View in Target List")
        self.view_target_btn.clicked.connect(self._view_linked_target)
        self.view_target_btn.setVisible(False)
        info_layout.addWidget(self.view_target_btn)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        location_group = QGroupBox("Location")
        location_layout = QVBoxLayout()
        self.detail_location_label = QLabel("")
        self.detail_location_label.setWordWrap(True)
        location_layout.addWidget(self.detail_location_label)
        location_group.setLayout(location_layout)
        layout.addWidget(location_group)

        visibility_group = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout()
        self.detail_visibility_label = QLabel("")
        self.detail_visibility_label.setWordWrap(True)
        visibility_layout.addWidget(self.detail_visibility_label)
        open_best_dso_btn = QPushButton("Open Best DSO Tonight")
        open_best_dso_btn.clicked.connect(self._open_best_dso_tonight_for_current)
        visibility_layout.addWidget(open_best_dso_btn)
        visibility_group.setLayout(visibility_layout)
        layout.addWidget(visibility_group)

        weather_group = QGroupBox("Weather Forecast")
        weather_layout = QVBoxLayout()
        self.detail_weather_label = QLabel("")
        self.detail_weather_label.setWordWrap(True)
        weather_layout.addWidget(self.detail_weather_label)
        refresh_weather_btn = QPushButton("Refresh Weather")
        refresh_weather_btn.clicked.connect(self._refresh_weather_clicked)
        weather_layout.addWidget(refresh_weather_btn)
        weather_group.setLayout(weather_layout)
        layout.addWidget(weather_group)

        layout.addStretch()

        return panel

    def _load_sessions(self):
        try:
            columns = [
                "id", "target_id", "dso_name", "ra_deg", "dec_deg", "status",
                "session_date", "start_time", "end_time",
                "location_lat", "location_lon", "location_name", "location_timezone", "location_source",
                "telescope_id", "telescope_name",
                "camera", "filters_used", "sub_count", "integration_seconds",
                "earliest_sub_date", "latest_sub_date", "notes",
                "created_date", "modified_date",
            ]
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.id, s.target_id, s.dso_name, s.ra_deg, s.dec_deg, s.status,
                           s.session_date, s.start_time, s.end_time,
                           s.location_lat, s.location_lon, s.location_name, s.location_timezone, s.location_source,
                           s.telescope_id, tel.name as telescope_name,
                           s.camera, s.filters_used, s.sub_count, s.integration_seconds,
                           s.earliest_sub_date, s.latest_sub_date, s.notes,
                           s.created_date, s.modified_date
                    FROM usersessions s
                    LEFT JOIN usertelescopes tel ON s.telescope_id = tel.id
                    ORDER BY s.session_date DESC, s.id DESC
                """)
                self.sessions_data = [dict(zip(columns, row)) for row in cursor.fetchall()]

            self._populate_table()
            self._filter_sessions()
            self.status_label.setText(f"Loaded {len(self.sessions_data)} sessions")
            self.calendar_widget.refresh_session_markers()
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load sessions: {e}")

    def _populate_table(self):
        self.sessions_table.setSortingEnabled(False)
        self.sessions_table.setRowCount(len(self.sessions_data))

        for row, session in enumerate(self.sessions_data):
            name_item = QTableWidgetItem(session.get("dso_name", ""))
            name_item.setData(Qt.UserRole, session)
            self.sessions_table.setItem(row, 0, name_item)

            self.sessions_table.setItem(row, 1, QTableWidgetItem(session.get("status", "")))
            self.sessions_table.setItem(row, 2, QTableWidgetItem(session.get("session_date", "")))

            start_time = session.get("start_time")
            end_time = session.get("end_time")
            if start_time and end_time:
                time_text = f"{start_time}-{end_time}"
            else:
                time_text = start_time or ""
            self.sessions_table.setItem(row, 3, QTableWidgetItem(time_text))

            loc_name = session.get("location_name")
            lat, lon = session.get("location_lat"), session.get("location_lon")
            if loc_name:
                loc_text = loc_name
            elif lat is not None and lon is not None:
                loc_text = f"{lat:.2f}, {lon:.2f}"
            else:
                loc_text = ""
            self.sessions_table.setItem(row, 4, QTableWidgetItem(loc_text))

            self.sessions_table.setItem(row, 5, QTableWidgetItem(session.get("telescope_name") or "Any"))
            self.sessions_table.setItem(row, 6, QTableWidgetItem(session.get("filters_used") or ""))

            subs_item = QTableWidgetItem(str(session.get("sub_count") or 0))
            subs_item.setTextAlignment(Qt.AlignCenter)
            self.sessions_table.setItem(row, 7, subs_item)

            integration_seconds = session.get("integration_seconds") or 0
            integration_item = QTableWidgetItem(f"{integration_seconds / 3600.0:.1f} h" if integration_seconds else "")
            integration_item.setData(Qt.UserRole, integration_seconds)
            integration_item.setTextAlignment(Qt.AlignCenter)
            self.sessions_table.setItem(row, 8, integration_item)

            linked_item = QTableWidgetItem("Yes" if session.get("target_id") else "")
            linked_item.setTextAlignment(Qt.AlignCenter)
            self.sessions_table.setItem(row, 9, linked_item)

        self.sessions_table.setSortingEnabled(True)

    def _filter_sessions(self):
        search_text = self.search_box.text().strip().lower()
        status_filter = self.status_filter.currentText()

        for row in range(self.sessions_table.rowCount()):
            show_row = True
            if status_filter != "All":
                status_item = self.sessions_table.item(row, 1)
                if not status_item or status_item.text() != status_filter:
                    show_row = False
            if search_text and show_row:
                name_item = self.sessions_table.item(row, 0)
                if not name_item or search_text not in name_item.text().lower():
                    show_row = False
            self.sessions_table.setRowHidden(row, not show_row)

    def _on_selection_changed(self):
        row = self.sessions_table.currentRow()
        has_selection = row >= 0
        self.edit_session_btn.setEnabled(has_selection)
        self.delete_session_btn.setEnabled(has_selection)
        self.duplicate_session_btn.setEnabled(has_selection)

        session_data = None
        if has_selection:
            name_item = self.sessions_table.item(row, 0)
            session_data = name_item.data(Qt.UserRole) if name_item else None
        self._update_detail_panel(session_data)

    def _select_session_by_id(self, session_id):
        for row in range(self.sessions_table.rowCount()):
            name_item = self.sessions_table.item(row, 0)
            if name_item:
                data = name_item.data(Qt.UserRole)
                if data and data.get("id") == session_id:
                    self.sessions_table.selectRow(row)
                    self.sessions_table.scrollToItem(name_item)
                    # selectRow() is a no-op if this row index was already the
                    # current selection (Qt tracks selection by row index, not
                    # item identity - _populate_table() replaces every row's
                    # QTableWidgetItem in place via setItem(), so editing a
                    # session or attaching new files to it doesn't move it to a
                    # different row and Qt never re-fires selectionChanged, even
                    # though the underlying data just changed). Refresh the
                    # detail panel/calendar unconditionally so they never show a
                    # stale dict from before the reload.
                    self._on_selection_changed()
                    return

    def _on_calendar_date_clicked(self, session_id):
        """Jump to the session in the Sessions tab when a marked calendar date is clicked."""
        self.tab_widget.setCurrentIndex(0)
        self._select_session_by_id(session_id)

    def _update_detail_panel(self, session):
        self._current_session = session
        if not session:
            self.detail_info_label.setText("Select a session to view details.")
            self.detail_location_label.setText("")
            self.detail_visibility_label.setText("")
            self.detail_weather_label.setText("")
            self._weather_key = None
            self.detail_equipment_label.setText("")
            self.view_target_btn.setVisible(False)
            self.calendar_widget.set_active_session(None)
            return

        info_lines = [
            f"<b>{session.get('dso_name', '')}</b>",
            f"Status: {session.get('status', '')}",
            f"Date: {session.get('session_date', '')}",
        ]
        time_text = self._format_time_range(session)
        if time_text:
            info_lines.append(f"Time: {time_text}")
        if session.get("notes"):
            info_lines.append(f"Notes: {session['notes']}")
        self.detail_info_label.setText("<br>".join(info_lines))

        loc_name = session.get("location_name")
        lat, lon, tz = session.get("location_lat"), session.get("location_lon"), session.get("location_timezone")
        if lat is not None and lon is not None:
            loc_text = f"{loc_name or ''} ({lat:.4f}, {lon:.4f})"
            if tz:
                loc_text += f" — {tz}"
        else:
            loc_text = "No location set"
        self.detail_location_label.setText(loc_text)

        self.view_target_btn.setVisible(bool(session.get("target_id")))

        nights_count = self._count_observation_nights(session["id"])
        equipment_lines = [
            f"Telescope: {session.get('telescope_name') or 'Any'}",
            f"Camera: {session.get('camera') or '(unknown)'}",
            f"Filters: {session.get('filters_used') or '(none)'}",
            f"Subs: {session.get('sub_count') or 0}",
            f"Integration: {(session.get('integration_seconds') or 0) / 3600.0:.2f} h",
            f"Observation Nights: {nights_count}",
        ]
        self.detail_equipment_label.setText("<br>".join(equipment_lines))

        self._refresh_visibility_panel(session)
        self._refresh_weather_panel(session)
        self.calendar_widget.set_active_session(session)

    def _format_time_range(self, session):
        start, end = session.get("start_time"), session.get("end_time")
        if start and end:
            return f"{start} - {end}"
        return start or ""

    def _count_observation_nights(self, session_id):
        """Number of observations (observing nights - one that runs past midnight
        is still one night) recorded for this session."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM usersessionobservations WHERE session_id = ?", (session_id,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting observation nights: {e}")
            return 0

    def _refresh_visibility_panel(self, session):
        ra_deg, dec_deg = session.get("ra_deg"), session.get("dec_deg")
        dso_name = session.get("dso_name")
        status = session.get("status")

        # Planned/In Progress sessions aren't tied to one fixed night any more (an
        # ongoing session may span many nights past its original planned date), so
        # show tonight's numbers - what matters day-to-day is "how does it look
        # right now," not the date it was first planned/started. Completed/Cancelled
        # sessions keep showing their recorded session_date for historical reference.
        showing_tonight = status in ("Planned", "In Progress")
        visibility_date = date_cls.today().strftime("%Y-%m-%d") if showing_tonight else session.get("session_date")

        if not visibility_date:
            self.detail_visibility_label.setText("No date set.")
            return

        self.detail_visibility_label.setText("Calculating visibility...")
        header = f"<b>Tonight</b> ({visibility_date}):" if showing_tonight else f"On {visibility_date}:"
        session_id = session.get("id")

        _retire_thread(self._visibility_thread)

        # Runs off the GUI thread - the first calculation per app run can otherwise
        # take several seconds (see VisibilityCalcThread) and would freeze the whole
        # window if done inline here, as selecting a row used to do.
        self._visibility_thread = VisibilityCalcThread(
            ra_deg, dec_deg, dso_name,
            session.get("location_lat"), session.get("location_lon"), session.get("location_timezone"),
            visibility_date,
        )
        self._visibility_thread.result_ready.connect(
            lambda result: self._on_visibility_result(result, header, session_id)
        )
        self._visibility_thread.start()

    def _on_visibility_result(self, result, header, session_id):
        # Discard a result for a session the user has since navigated away from.
        if not self._current_session or self._current_session.get("id") != session_id:
            return

        if result.get("error"):
            self.detail_visibility_label.setText(result["error"])
            return

        lines = [header] + result.get("lines", [])
        self.detail_visibility_label.setText("<br>".join(lines))

    def _refresh_weather_clicked(self):
        if self._current_session:
            self._refresh_weather_panel(self._current_session, force=True)

    def _refresh_weather_panel(self, session, force=False):
        status = session.get("status")
        lat, lon = session.get("location_lat"), session.get("location_lon")

        # Like the Visibility panel, Planned/In Progress sessions show tonight's
        # forecast (index 0 of the 7-day array) rather than the session's original
        # planned date, since an ongoing session isn't tied to one fixed night.
        if status not in ("Planned", "In Progress") or lat is None or lon is None:
            self._weather_key = None
            self.detail_weather_label.setText(
                "Weather forecast is only shown for planned or ongoing sessions with a location set."
            )
            return

        from WeatherForecast import WeatherCache

        cached = None if force else WeatherCache().get(lat, lon)
        self._weather_key = SharedWeatherFetch.key(lat, lon)
        if cached:
            self._apply_weather_summary(cached[0])
            return

        self.detail_weather_label.setText("Loading weather forecast...")
        key = self._weather_key
        SharedWeatherFetch.request(
            lat, lon, session.get("location_timezone"),
            lambda data: self._on_weather_loaded(data, key),
            lambda msg: self._on_weather_error(msg, key))

    def _on_weather_loaded(self, data, key):
        if key != self._weather_key:
            return  # a different session/location has been selected since
        try:
            if data:
                self._apply_weather_summary(data[0])
            else:
                self.detail_weather_label.setText("No forecast data available.")
        except RuntimeError:
            pass  # the window was closed while the fetch ran

    def _on_weather_error(self, message, key):
        if key != self._weather_key:
            return
        try:
            self.detail_weather_label.setText(f"Weather fetch failed: {message}")
        except RuntimeError:
            pass

    def _apply_weather_summary(self, day_summary):
        from WeatherForecast import get_rating_label
        lines = [
            f"Astro score: {day_summary.astro_score}/100 ({get_rating_label(day_summary.astro_score)})",
            f"Avg cloud cover tonight: {day_summary.tonight_avg_cloud_cover:.0f}%",
            f"Seeing estimate: {day_summary.seeing_estimate}",
        ]
        if day_summary.moon_phase:
            lines.append(f"Moon: {day_summary.moon_phase.phase_name} ({day_summary.moon_phase.illumination:.0f}%)")
        if day_summary.dark_hours_start and day_summary.dark_hours_end:
            lines.append(
                f"Dark hours: {format_time(day_summary.dark_hours_start)} - {format_time(day_summary.dark_hours_end)}"
            )
        self.detail_weather_label.setText("<br>".join(lines))

    def _open_best_dso_tonight_for_current(self):
        if not self._current_session:
            return
        try:
            from BestDSOTonight import BestDSOTonightWindow
            self.best_dso_window = BestDSOTonightWindow(use_target_list=bool(self._current_session.get("target_id")))
            self.best_dso_window.show()
            self.best_dso_window.raise_()
            self.best_dso_window.activateWindow()
        except Exception as e:
            logger.error(f"Error opening Best DSO Tonight: {e}")
            QMessageBox.warning(self, "Error", f"Could not open Best DSO Tonight: {e}")

    def _view_linked_target(self):
        if not self._current_session or not self._current_session.get("target_id"):
            return
        try:
            from DSOTargetList import DSOTargetListWindow
            if not hasattr(self, "target_list_window") or not self.target_list_window.isVisible():
                self.target_list_window = DSOTargetListWindow()
            self.target_list_window.open_and_select_target(self._current_session.get("dso_name"))
        except Exception as e:
            logger.error(f"Error opening target list: {e}")
            QMessageBox.warning(self, "Error", f"Could not open Target List: {e}")

    def _new_planned_session(self):
        dialog = AddEditSessionDialog(initial_status="Planned", parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_sessions()
            if dialog.session_id:
                self._select_session_by_id(dialog.session_id)

    def _log_past_session(self):
        dialog = AddEditSessionDialog(initial_status="Completed", parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_sessions()
            if dialog.session_id:
                self._select_session_by_id(dialog.session_id)

    def _edit_selected_session(self, _item=None, start_tab=0, start_add=False):
        row = self.sessions_table.currentRow()
        if row < 0:
            return
        name_item = self.sessions_table.item(row, 0)
        session_data = name_item.data(Qt.UserRole)
        dialog = SessionDetailsDialog(session_data, start_tab=start_tab, start_add=start_add, parent=self)
        # Attaching/removing files writes immediately, so reload even on Cancel.
        if dialog.exec() == QDialog.Accepted or dialog.data_changed:
            self._load_sessions()
            self._select_session_by_id(session_data["id"])
            # Compared against the status the dialog opened with, so a change
            # committed early (attaching files saves pending edits) still counts.
            updated = next((s for s in self.sessions_data if s["id"] == session_data["id"]), None)
            if updated and updated["status"] == "Completed" and session_data.get("status") != "Completed":
                from SessionCompletion import SessionCompletionDialog
                SessionCompletionDialog(updated, parent=self).exec()
                self._load_sessions()  # the target list entry may have been marked Completed
                self._select_session_by_id(session_data["id"])

    def _add_observation_to_selected(self):
        self._edit_selected_session(start_tab=SessionDetailsDialog.OBSERVATIONS_TAB, start_add=True)

    def _duplicate_selected_session(self):
        row = self.sessions_table.currentRow()
        if row < 0:
            return
        name_item = self.sessions_table.item(row, 0)
        session_data = name_item.data(Qt.UserRole)
        dialog = AddEditSessionDialog(parent=self)
        dialog.load_as_duplicate(session_data)
        if dialog.exec() == QDialog.Accepted:
            self._load_sessions()
            if dialog.session_id:
                self._select_session_by_id(dialog.session_id)

    def _delete_selected_session(self):
        row = self.sessions_table.currentRow()
        if row < 0:
            return
        name_item = self.sessions_table.item(row, 0)
        session_data = name_item.data(Qt.UserRole)

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete the session for '{session_data.get('dso_name')}' on {session_data.get('session_date')}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                SessionObservations.delete_session_children(conn, session_data["id"])
                cursor.execute("DELETE FROM usersessions WHERE id = ?", (session_data["id"],))
                conn.commit()
            self._load_sessions()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error deleting session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete session: {e}")

    def _show_context_menu(self, position):
        item = self.sessions_table.itemAt(position)
        if not item:
            return
        self.sessions_table.selectRow(item.row())
        session_data = self.sessions_table.item(item.row(), 0).data(Qt.UserRole)

        menu = QMenu(self)
        edit_action = menu.addAction("Session Details...")
        edit_action.triggered.connect(lambda: self._edit_selected_session())
        add_obs_action = menu.addAction("Add Observation...")
        add_obs_action.triggered.connect(self._add_observation_to_selected)
        duplicate_action = menu.addAction("Duplicate Session")
        duplicate_action.triggered.connect(self._duplicate_selected_session)
        menu.addSeparator()
        if session_data and session_data.get("target_id"):
            change_link_action = menu.addAction("Change Linked Target...")
            change_link_action.triggered.connect(self._link_session_to_target)
            unlink_action = menu.addAction("Unlink Target")
            unlink_action.triggered.connect(self._unlink_session_target)
        else:
            link_action = menu.addAction("Link to Target List...")
            link_action.triggered.connect(self._link_session_to_target)
        menu.addSeparator()
        delete_action = menu.addAction("Delete Session")
        delete_action.triggered.connect(self._delete_selected_session)
        menu.exec(self.sessions_table.mapToGlobal(position))

    def _link_session_to_target(self):
        row = self.sessions_table.currentRow()
        if row < 0:
            return
        session_data = self.sessions_table.item(row, 0).data(Qt.UserRole)

        dialog = LinkTargetDialog(self.db_manager, parent=self)
        if dialog.exec() == QDialog.Accepted and dialog.selected_target:
            target = dialog.selected_target
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE usersessions SET target_id = ? WHERE id = ?",
                                   (target["id"], session_data["id"]))
                    conn.commit()
                self._load_sessions()
                self._select_session_by_id(session_data["id"])
            except Exception as e:
                logger.error(f"Error linking session to target: {e}")
                QMessageBox.critical(self, "Error", f"Failed to link target: {e}")

    def _unlink_session_target(self):
        row = self.sessions_table.currentRow()
        if row < 0:
            return
        session_data = self.sessions_table.item(row, 0).data(Qt.UserRole)

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE usersessions SET target_id = NULL WHERE id = ?", (session_data["id"],))
                conn.commit()
            self._load_sessions()
            self._select_session_by_id(session_data["id"])
        except Exception as e:
            logger.error(f"Error unlinking target: {e}")
            QMessageBox.critical(self, "Error", f"Failed to unlink target: {e}")

    def _show_column_menu(self, position):
        """Right-click menu on the sessions table's header - toggle which columns
        are shown. Selection is persisted via QSettings and restored on next launch."""
        header = self.sessions_table.horizontalHeader()
        menu = QMenu(self)
        for col, label in enumerate(self._column_labels):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(not self.sessions_table.isColumnHidden(col))
            action.toggled.connect(lambda checked, c=col: self._toggle_column_visibility(c, checked))
        menu.exec(header.mapToGlobal(position))

    def _toggle_column_visibility(self, col, visible):
        self.sessions_table.setColumnHidden(col, not visible)
        self._save_column_visibility()

    def _save_column_visibility(self):
        hidden = [str(col) for col in range(self.sessions_table.columnCount())
                  if self.sessions_table.isColumnHidden(col)]
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue(self.COLUMNS_SETTING, ",".join(hidden))

    def _apply_saved_column_visibility(self):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        saved = settings.value(self.COLUMNS_SETTING, "", type=str)
        if not saved:
            return
        for part in saved.split(","):
            part = part.strip()
            if part.isdigit():
                col = int(part)
                if 0 <= col < self.sessions_table.columnCount():
                    self.sessions_table.setColumnHidden(col, True)

    def create_session_from_dso(self, dso_data):
        """Public entry point: open a pre-filled planned session for a Target List DSO."""
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        dialog = AddEditSessionDialog(target_data=dso_data, initial_status="Planned", parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_sessions()
            if dialog.session_id:
                self._select_session_by_id(dialog.session_id)

    def open_and_select_session(self, session_id):
        """Public entry point: show/raise the window and select a session by id."""
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        self._load_sessions()
        self._select_session_by_id(session_id)

    def dragEnterEvent(self, event):
        if dropped_sub_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        dropped_paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        event.acceptProposedAction()
        if dropped_paths:
            self._start_scan(dropped_paths)

    def _start_scan(self, paths):
        self._progress_dialog = QProgressDialog("Scanning files...", "Cancel", 0, 0, self)
        self._progress_dialog.setWindowTitle("Session Manager")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.show()

        self._scan_thread = SessionScanThread(paths)
        self._scan_thread.progress.connect(self._on_scan_progress)
        self._scan_thread.scan_finished.connect(self._on_scan_finished)
        self._scan_thread.scan_error.connect(self._on_scan_error)
        self._progress_dialog.canceled.connect(self._scan_thread.terminate)
        self._scan_thread.start()

    def _on_scan_progress(self, current, total):
        if total > 0:
            self._progress_dialog.setMaximum(total)
            self._progress_dialog.setValue(current)

    def _on_scan_finished(self, files):
        self._progress_dialog.close()
        if not files:
            QMessageBox.information(self, "No Files Found", "No supported FITS/XISF files were found.")
            return
        summary = SessionFileScanner.summarize_files(files)
        dialog = DropMatchDialog(summary, self.db_manager, parent=self)
        if dialog.exec() == QDialog.Accepted and dialog.result_session_id:
            self._load_sessions()
            self._select_session_by_id(dialog.result_session_id)

    def _on_scan_error(self, message):
        self._progress_dialog.close()
        QMessageBox.critical(self, "Scan Error", f"Failed to scan dropped files: {message}")
