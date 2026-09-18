#!/usr/bin/env python3
"""
Session Manager
Plan future observing sessions and log past ones: DSO target, date/time, location,
equipment, and (via drag-and-drop) the FITS/XISF subs a session produced.
"""

import os
import calendar
import logging
from datetime import datetime, date as date_cls, timedelta

from PySide6.QtCore import Qt, QDate, QEvent, QTime, QTimer, QPoint, QSettings, Signal, QThread, QStringListModel
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QGroupBox, QMessageBox,
                               QHeaderView, QTextEdit, QDialog, QComboBox,
                               QLineEdit, QCheckBox, QDateEdit, QTimeEdit, QMenu,
                               QCompleter, QSplitter, QFormLayout, QRadioButton,
                               QListWidget, QListWidgetItem, QProgressDialog,
                               QCalendarWidget, QTabWidget, QTableView)

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from TimeFormatHelper import format_time
from Theme import COLORS
import SessionFileScanner

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


class AddEditSessionDialog(WindowPositionMixin, QDialog):
    """Add/edit dialog for a planned or logged session."""

    WINDOW_POSITION_KEY = "AddEditSessionDialog"

    def __init__(self, session_data=None, target_data=None, parsed_metadata=None,
                 initial_status="Planned", parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(520, 680)  # default size the first time this dialog is ever opened

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

        self.setup_window_position()

    def _setup_ui(self):
        layout = QVBoxLayout()

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

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        self.save_btn = QPushButton("Save Session")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save_session)
        buttons_layout.addWidget(self.save_btn)
        layout.addLayout(buttons_layout)

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
            self.save_btn.setText("Save Changes")
            self.setWindowTitle("Edit Session")
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

    def _save_session(self):
        try:
            name = self.name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Validation Error", "DSO name is required.")
                return

            lat, lon, loc_name, tz, source = self.location_widget.get_location()
            session_date = self.date_edit.date().toString("yyyy-MM-dd")
            has_times = self.time_checkbox.isChecked()
            start_time = self.start_time_edit.time().toString("HH:mm") if has_times else None
            end_time = self.end_time_edit.time().toString("HH:mm") if has_times else None

            aggregate = self._scanned_aggregate or self._existing_aggregate or {}
            resolved_target_id = self._resolve_target_id_by_name(name)
            target_id = resolved_target_id if resolved_target_id is not None else self._target_id

            data = {
                "target_id": target_id,
                "dso_name": name,
                "ra_deg": self._ra_deg,
                "dec_deg": self._dec_deg,
                "status": self.status_combo.currentText(),
                "session_date": session_date,
                "start_time": start_time,
                "end_time": end_time,
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

            with self.db_manager.get_connection() as conn:
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
                conn.commit()

            self.saved_session_data = data
            self.accept()

        except Exception as e:
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
            dialog = AddEditSessionDialog(parsed_metadata=self.summary, initial_status="In Progress", parent=self)
            if dialog.exec() == QDialog.Accepted and dialog.session_id:
                self._insert_file_rows(dialog.session_id)
                self._recompute_session_aggregates(dialog.session_id)
                self.result_session_id = dialog.session_id
                self.accept()
        else:
            item = self.sessions_list.currentItem()
            if not item:
                QMessageBox.warning(self, "No Selection", "Please select an existing session.")
                return
            session_id = item.data(Qt.UserRole)
            self._insert_file_rows(session_id)
            self._recompute_session_aggregates(session_id)
            self.result_session_id = session_id
            self.accept()

    def _insert_file_rows(self, session_id):
        files = self.summary.get("files", [])
        if not files:
            return
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                for f in files:
                    row = SessionFileScanner.file_dict_to_row(f)
                    cursor.execute("""
                        INSERT OR IGNORE INTO usersessionfiles (
                            session_id, file_path, file_type, frame_type, object_name, date_obs,
                            exptime_seconds, filter_name, camera, telescope, gain, offset_value,
                            ccd_temp, xbinning, ybinning, header_json
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        session_id, row["file_path"], row["file_type"], row["frame_type"],
                        row["object_name"], row["date_obs"], row["exptime_seconds"], row["filter_name"],
                        row["camera"], row["telescope"], row["gain"], row["offset_value"],
                        row["ccd_temp"], row["xbinning"], row["ybinning"], row["header_json"],
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting session files: {e}")

    def _recompute_session_aggregates(self, session_id):
        """Recompute sub_count/integration_seconds/filters/date-range from the
        usersessionfiles rows actually stored for this session, rather than adding
        the new drop's totals on top of the old ones - since INSERT OR IGNORE
        (keyed on session_id + file_path) silently skips files already attached,
        recomputing from that table is what makes re-dropping the same folder a
        true no-op instead of double-counting integration time.

        Also promotes a 'Planned' session to 'In Progress' the first time data is
        actually attached to it - a session stays 'In Progress' (accumulating
        across as many nights/drops as needed) until the user manually marks it
        'Completed'. 'Completed'/'Cancelled' sessions are left alone."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM usersessions WHERE id = ?", (session_id,))
                status_row = cursor.fetchone()
                current_status = status_row[0] if status_row else None
                new_status = "In Progress" if current_status == "Planned" else current_status

                cursor.execute("""
                    SELECT COUNT(*),
                           COALESCE(SUM(CASE WHEN frame_type = 'Light' THEN exptime_seconds ELSE 0 END), 0),
                           MIN(date_obs), MAX(date_obs)
                    FROM usersessionfiles WHERE session_id = ?
                """, (session_id,))
                sub_count, integration_seconds, earliest, latest = cursor.fetchone()

                cursor.execute("""
                    SELECT DISTINCT filter_name FROM usersessionfiles
                    WHERE session_id = ? AND frame_type = 'Light'
                          AND filter_name IS NOT NULL AND filter_name != ''
                """, (session_id,))
                filters = sorted(row[0] for row in cursor.fetchall())

                cursor.execute("""
                    UPDATE usersessions SET
                        sub_count = ?, integration_seconds = ?, earliest_sub_date = ?, latest_sub_date = ?,
                        filters_used = ?, status = ?, modified_date = ?
                    WHERE id = ?
                """, (
                    sub_count, integration_seconds, earliest, latest,
                    ", ".join(filters), new_status, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id,
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recomputing session aggregates: {e}")


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


class SessionCalendarWidget(QWidget):
    """The Session Manager 'Calendar' tab: 1-3 side-by-side SessionMonthCalendar
    instances kept chronologically chained, with weather/visibility layers that
    follow the currently selected session (falling back to the app's active
    location for weather when nothing is selected)."""

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
        self._weather_worker = None
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

    def _get_active_location(self):
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon, timezone FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT location_lat, location_lon, timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                if row:
                    return row[0], row[1], row[2]
        except Exception as e:
            logger.error(f"Error loading active location for calendar: {e}")
        return None, None, None

    def _refresh_weather_layer(self):
        session = self._current_session
        if session and session.get("location_lat") is not None:
            lat, lon, tz = session.get("location_lat"), session.get("location_lon"), session.get("location_timezone")
        else:
            lat, lon, tz = self._get_active_location()

        if lat is None or lon is None:
            self._apply_weather_scores({}, {})
            return

        from WeatherForecast import WeatherCache, WeatherWorker

        cache = WeatherCache()
        cached = cache.get(lat, lon)
        if cached:
            self._apply_weather_list(cached)
            return

        # Never replace a still-running thread's reference - PySide6 can hard-crash
        # (native, uncatchable) if a QThread object is garbage-collected while its
        # underlying OS thread is still executing. Same guard already used by
        # DSOVisibilityCalculator.start_monthly_visibility_calculation().
        if self._weather_worker and self._weather_worker.isRunning():
            self._weather_worker.quit()
            self._weather_worker.wait()

        self._weather_worker = WeatherWorker(lat, lon, tz)
        self._weather_worker.weather_loaded.connect(lambda data: self._on_weather_loaded(data, lat, lon))
        self._weather_worker.error_occurred.connect(
            lambda msg: logger.debug(f"Calendar weather fetch failed: {msg}")
        )
        self._weather_worker.start()

    def _on_weather_loaded(self, data, lat, lon):
        from WeatherForecast import WeatherCache
        WeatherCache().set(lat, lon, data)
        self._apply_weather_list(data)

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

        if self._visibility_thread and self._visibility_thread.isRunning():
            self._visibility_thread.quit()
            self._visibility_thread.wait()

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
        night_filters), ...]. Groups files into "observing nights" rather than raw
        calendar dates: a sub taken before noon belongs to the PREVIOUS evening's
        night (e.g. a 2 AM sub on Saturday is part of "Friday night"), so a session
        that runs past midnight is one continuous night, not two fragments - and
        that night's combined totals are shown on EVERY calendar date it actually
        touches (both Friday and Saturday get the same, complete night data),
        rather than splitting the totals across the two dates."""
        markers = {}
        try:
            # Pad by a day on each side so a night starting the evening before
            # start_date, or ending the morning after end_date, is fully captured
            # before grouping - only dates inside [start_date, end_date] get
            # returned, but the night's totals reflect all of its subs.
            padded_start = start_date - timedelta(days=1)
            padded_end = end_date + timedelta(days=1)

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT f.session_id, s.status, s.dso_name, f.date_obs, f.frame_type,
                           f.exptime_seconds, f.filter_name
                    FROM usersessionfiles f
                    JOIN usersessions s ON s.id = f.session_id
                    WHERE f.date_obs IS NOT NULL
                      AND date(f.date_obs) BETWEEN ? AND ?
                """, (padded_start.isoformat(), padded_end.isoformat()))

                nights = {}  # (session_id, night_date) -> aggregate dict
                for session_id, status, dso_name, date_obs, frame_type, exptime, filter_name in cursor.fetchall():
                    try:
                        dt = datetime.fromisoformat(str(date_obs).replace('Z', ''))
                    except ValueError:
                        continue
                    calendar_date = dt.date()
                    night_date = calendar_date - timedelta(days=1) if dt.hour < 12 else calendar_date

                    night = nights.setdefault((session_id, night_date), {
                        "status": status, "dso_name": dso_name, "seconds": 0.0,
                        "filters": set(), "calendar_dates": set(),
                    })
                    night["calendar_dates"].add(calendar_date)
                    if frame_type == "Light":
                        night["seconds"] += exptime or 0
                        if filter_name:
                            night["filters"].add(filter_name)

                for (session_id, _night_date), night in nights.items():
                    filters_str = ",".join(sorted(night["filters"])) if night["filters"] else None
                    marker = (session_id, night["dso_name"], night["status"], night["seconds"], filters_str)
                    for calendar_date in night["calendar_dates"]:
                        if start_date <= calendar_date <= end_date:
                            markers.setdefault(calendar_date, []).append(marker)

                cursor.execute("SELECT DISTINCT session_id FROM usersessionfiles")
                sessions_with_files = {row[0] for row in cursor.fetchall()}

                cursor.execute("""
                    SELECT id, session_date, status, dso_name FROM usersessions
                    WHERE session_date BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                for session_id, session_date, status, dso_name in cursor.fetchall():
                    if session_id in sessions_with_files:
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
        self._weather_worker = None
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
        except Exception as e:
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
        self.sessions_table.setColumnCount(10)
        self.sessions_table.setHorizontalHeaderLabels([
            "DSO Name", "Status", "Date", "Time", "Location", "Telescope",
            "Filters", "Subs", "Integration", "Linked Target"
        ])
        self.sessions_table.setSortingEnabled(True)
        header = self.sessions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, 10):
            header.setSectionResizeMode(col, QHeaderView.Interactive)
        self.sessions_table.setAlternatingRowColors(True)
        self.sessions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sessions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sessions_table.itemDoubleClicked.connect(self._edit_selected_session)
        self.sessions_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sessions_table.customContextMenuRequested.connect(self._show_context_menu)
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
        """Counts distinct observing nights (not calendar dates) with logged subs
        for this session, using the same before-noon-belongs-to-the-previous-
        evening convention as the calendar's session markers, so a night that
        runs past midnight counts once, not twice."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT date_obs FROM usersessionfiles WHERE session_id = ? AND date_obs IS NOT NULL",
                    (session_id,),
                )
                nights = set()
                for (date_obs,) in cursor.fetchall():
                    try:
                        dt = datetime.fromisoformat(str(date_obs).replace('Z', ''))
                    except ValueError:
                        continue
                    night_date = dt.date() - timedelta(days=1) if dt.hour < 12 else dt.date()
                    nights.add(night_date)
                return len(nights)
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

        # Never replace a still-running thread's reference - PySide6 can hard-crash
        # (native, uncatchable) if a QThread object is garbage-collected while its
        # underlying OS thread is still executing. Same guard already used by
        # DSOVisibilityCalculator.start_monthly_visibility_calculation().
        if self._visibility_thread and self._visibility_thread.isRunning():
            self._visibility_thread.quit()
            self._visibility_thread.wait()

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
            self.detail_weather_label.setText(
                "Weather forecast is only shown for planned or ongoing sessions with a location set."
            )
            return

        from WeatherForecast import WeatherCache, WeatherWorker

        cache = WeatherCache()
        cached = None if force else cache.get(lat, lon)
        if cached:
            self._apply_weather_summary(cached[0])
            return

        # Never replace a still-running thread's reference - PySide6 can hard-crash
        # (native, uncatchable) if a QThread object is garbage-collected while its
        # underlying OS thread is still executing.
        if self._weather_worker and self._weather_worker.isRunning():
            self._weather_worker.quit()
            self._weather_worker.wait()

        self.detail_weather_label.setText("Loading weather forecast...")
        self._weather_worker = WeatherWorker(lat, lon, session.get("location_timezone"))
        self._weather_worker.weather_loaded.connect(
            lambda data: self._on_weather_loaded(data, lat, lon)
        )
        self._weather_worker.error_occurred.connect(
            lambda msg: self.detail_weather_label.setText(f"Weather fetch failed: {msg}")
        )
        self._weather_worker.start()

    def _on_weather_loaded(self, data, lat, lon):
        from WeatherForecast import WeatherCache
        WeatherCache().set(lat, lon, data)
        if data:
            self._apply_weather_summary(data[0])
        else:
            self.detail_weather_label.setText("No forecast data available.")

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

    def _edit_selected_session(self):
        row = self.sessions_table.currentRow()
        if row < 0:
            return
        name_item = self.sessions_table.item(row, 0)
        session_data = name_item.data(Qt.UserRole)
        dialog = AddEditSessionDialog(session_data=session_data, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_sessions()
            self._select_session_by_id(session_data["id"])

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
                cursor.execute("DELETE FROM usersessionfiles WHERE session_id = ?", (session_data["id"],))
                cursor.execute("DELETE FROM usersessions WHERE id = ?", (session_data["id"],))
                conn.commit()
            self._load_sessions()
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete session: {e}")

    def _show_context_menu(self, position):
        item = self.sessions_table.itemAt(position)
        if not item:
            return
        self.sessions_table.selectRow(item.row())

        menu = QMenu(self)
        edit_action = menu.addAction("Edit Session")
        edit_action.triggered.connect(self._edit_selected_session)
        duplicate_action = menu.addAction("Duplicate Session")
        duplicate_action.triggered.connect(self._duplicate_selected_session)
        menu.addSeparator()
        delete_action = menu.addAction("Delete Session")
        delete_action.triggered.connect(self._delete_selected_session)
        menu.exec(self.sessions_table.mapToGlobal(position))

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
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if not url.isLocalFile():
                    continue
                path = url.toLocalFile()
                if os.path.isdir(path) or os.path.splitext(path)[1].lower() in SessionFileScanner.SUPPORTED_EXTENSIONS:
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
