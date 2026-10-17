"""
EquipmentManager.py - The Equipment window: telescopes, cameras, eyepieces and
barlows/reducers.

Each kind has a tab: a list (Add, Duplicate, Delete) with the selected item's
details beside it. Changes save as you type - there's no Save button - and
deleting first shows what uses the item, keeping those records working.

Equipment links to telescopes through telescope_equipment, edited from either
side: a telescope's "Used with" lists, or an item's "Used with telescopes".

The Matches tab lists the telescope and camera names in the user's files and
gallery images and the equipment each one is (see EquipmentMapper.py), and
offers to link images that could be.
"""

import html
import json
import logging
import re
from collections import Counter

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QPlainTextEdit,
    QComboBox, QCheckBox, QPushButton, QListWidget, QListWidgetItem, QTabWidget, QSplitter,
    QScrollArea, QGroupBox, QDoubleSpinBox, QSpinBox, QMenu, QMessageBox, QCompleter, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QDialogButtonBox
)

import EquipmentMapper
import LibraryScanner
from LibraryScanner import TELESCOPE, CAMERA
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS, font_size, themed_style

logger = logging.getLogger(__name__)

MOUNT_TYPES = ("Alt-Az", "Equatorial (German)", "Equatorial (Fork)", "Dobsonian", "Tripod", "Pier", "Other")

EYEPIECE_PRESETS = {
    "32mm Plossl (52 AFOV)": (32, 52), "25mm Plossl (52 AFOV)": (25, 52), "20mm Plossl (50 AFOV)": (20, 50),
    "15mm Plossl (50 AFOV)": (15, 50), "10mm Plossl (50 AFOV)": (10, 50), "6mm Plossl (50 AFOV)": (6, 50),
    "Televue 21mm Ethos": (21, 100), "Televue 17mm Ethos": (17, 100), "Televue 13mm Ethos": (13, 100),
    "Televue 10mm Ethos": (10, 100), "Televue 8mm Ethos": (8, 100), "Televue 6mm Ethos": (6, 100),
    "Televue 4.7mm Ethos": (4.7, 100), "Televue 3.7mm Ethos": (3.7, 100),
    "Televue 31mm Nagler": (31, 82), "Televue 22mm Nagler": (22, 82), "Televue 17mm Nagler": (17, 82),
    "Televue 16mm Nagler": (16, 82), "Televue 13mm Nagler": (13, 82), "Televue 12mm Nagler": (12, 82),
    "Televue 11mm Nagler": (11, 82), "Televue 9mm Nagler": (9, 82), "Televue 7mm Nagler": (7, 82),
    "Televue 5mm Nagler": (5, 82), "Televue 3.5mm Nagler": (3.5, 82),
    "Televue 41mm Panoptic": (41, 68), "Televue 35mm Panoptic": (35, 68), "Televue 27mm Panoptic": (27, 68),
    "Televue 24mm Panoptic": (24, 68), "Televue 19mm Panoptic": (19, 68), "Televue 15mm Panoptic": (15, 68),
    "ES 30mm 82": (30, 82), "ES 24mm 82": (24, 82), "ES 18mm 82": (18, 82), "ES 14mm 82": (14, 82),
    "ES 11mm 82": (11, 82), "ES 8.8mm 82": (8.8, 82), "ES 6.7mm 82": (6.7, 82), "ES 4.7mm 82": (4.7, 82),
    "ES 40mm 68": (40, 68), "ES 34mm 68": (34, 68), "ES 28mm 68": (28, 68), "ES 24mm 68": (24, 68),
    "ES 20mm 68": (20, 68), "ES 16mm 68": (16, 68),
}

BARLOW_PRESETS = {
    "1.25x Barlow": (1.25, "barlow"), "1.5x Barlow": (1.5, "barlow"), "2x Barlow": (2.0, "barlow"),
    "2.5x Barlow": (2.5, "barlow"), "3x Barlow": (3.0, "barlow"), "4x Barlow": (4.0, "barlow"),
    "5x Barlow": (5.0, "barlow"),
    "Televue 2x Powermate": (2.0, "barlow"), "Televue 2.5x Powermate": (2.5, "barlow"),
    "Televue 4x Powermate": (4.0, "barlow"), "Televue 5x Powermate": (5.0, "barlow"),
    "0.5x Reducer": (0.5, "reducer"), "0.6x Reducer": (0.6, "reducer"), "0.63x Reducer": (0.63, "reducer"),
    "0.67x Reducer": (0.67, "reducer"), "0.7x Reducer": (0.7, "reducer"), "0.75x Reducer": (0.75, "reducer"),
    "0.8x Reducer": (0.8, "reducer"),
    "Starizona SCT Corrector 0.63x": (0.63, "reducer"), "Starizona Hyperstar": (0.33, "reducer"),
}


# ---------------------------------------------------------------------------
# Pixel sizes from the user's own subs
# ---------------------------------------------------------------------------

def _plain(name):
    """'ZWO ASI2600MC Pro' -> 'zwoasi2600mcpro', for matching camera names."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def pixel_sizes_from_subs(conn):
    """{plain camera name: (pixel size µm, subs)} from the XPIXSZ in the headers of
    the user's session subs (NINA and most capture software write it). The
    unbinned size - XPIXSZ includes any binning."""
    counts = Counter()
    try:
        rows = conn.execute("SELECT camera, header_json FROM usersessionfiles "
                            "WHERE header_json LIKE '%XPIXSZ%'").fetchall()
    except Exception as e:
        logger.debug(f"No sub headers to read pixel sizes from: {e}")
        return {}
    for camera, header in rows:
        try:
            values = json.loads(header)
            pixel = float(values["XPIXSZ"]) / max(float(values.get("XBINNING") or 1), 1.0)
        except (ValueError, KeyError, TypeError):
            continue
        name = _plain(values.get("INSTRUME") or camera)
        if name and 0.5 < pixel < 50:
            counts[(name, round(pixel, 2))] += 1
    found = {}
    for (name, pixel), subs in counts.most_common():
        found.setdefault(name, (pixel, subs))  # Each camera's most common size
    return found


def pixel_size_for(name, sub_pixels):
    """(pixel size, subs) recorded for a camera named like 'name' in the user's subs, or None."""
    plain = _plain(name)
    if len(plain) < 4:
        return None
    matches = [value for recorded, value in sub_pixels.items() if plain in recorded or recorded in plain]
    return max(matches, key=lambda value: value[1]) if matches else None


def fill_pixel_sizes(conn, sub_pixels=None):
    """Give cameras without a pixel size the one their subs record. Returns how many."""
    sub_pixels = pixel_sizes_from_subs(conn) if sub_pixels is None else sub_pixels
    filled = 0
    for camera_id, name in conn.execute("SELECT id, name FROM userequipment WHERE equipment_type = 'camera' "
                                        "AND (pixel_size IS NULL OR pixel_size <= 0)").fetchall():
        found = pixel_size_for(name, sub_pixels)
        if found:
            conn.execute("UPDATE userequipment SET pixel_size = ? WHERE id = ?", (found[0], camera_id))
            filled += 1
    if filled:
        conn.commit()
    return filled


def select_telescope(combo, db_manager, telescope_id):
    """Select a saved record's telescope in a picker that lists enabled telescopes.
    A disabled one is added (marked) so it stays selected - and saving keeps it."""
    if telescope_id is None:
        return False
    index = combo.findData(telescope_id)
    if index < 0:
        try:
            with db_manager.get_connection() as conn:
                row = conn.execute("SELECT name FROM usertelescopes WHERE id = ?", (telescope_id,)).fetchone()
        except Exception as e:
            logger.error(f"Error loading telescope {telescope_id}: {e}")
            row = None
        if not row:
            return False  # Deleted
        combo.addItem(f"{row[0]} (disabled)", telescope_id)
        index = combo.count() - 1
    combo.setCurrentIndex(index)
    return True


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def _number_box(maximum, decimals, suffix, unset="Not set"):
    """A number field where 0 means 'not set'."""
    box = QDoubleSpinBox()
    box.setRange(0, maximum)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setSpecialValueText(unset)
    box.setMinimumWidth(110)
    return box


def _value(box):
    return box.value() or None


def _set(box, value):
    box.setValue(float(value) if value else 0.0)


def _fmt(number):
    """12.0 -> '12', 3.76 -> '3.76'."""
    return f"{number:g}" if number is not None else ""


def _count(n, word):
    """'1 session', '3 sessions'."""
    return f"{n} {word}{'s' if n != 1 else ''}"


def _and(parts):
    """'a', 'a and b', 'a, b and c'."""
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _images_keep_name(images):
    return (f"{_count(images, 'gallery image')} taken with it {'keeps' if images == 1 else 'keep'} its name."
            if images else "")


class _CheckList(QListWidget):
    """Checkable items (id in UserRole); toggled(id, checked) when the user ticks one."""

    toggled = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemChanged.connect(self._on_item_changed)
        self._filling = False

    def fill(self, entries, checked_ids):
        """entries: [(id, text)]."""
        self._filling = True
        self.clear()
        for item_id, text in entries:
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if item_id in checked_ids else Qt.Unchecked)
            item.setData(Qt.UserRole, item_id)
            self.addItem(item)
        self._filling = False

    def _on_item_changed(self, item):
        if not self._filling:
            self.toggled.emit(item.data(Qt.UserRole), item.checkState() == Qt.Checked)


# ---------------------------------------------------------------------------
# One tab: a list and the selected item's details
# ---------------------------------------------------------------------------

class _EquipmentPage(QWidget):
    """A kind of equipment: its list (Add, Duplicate, Delete) and the selected
    item's details, saved as they're edited. Subclasses describe the kind."""

    changed = Signal()  # Something was saved - the other tabs refresh what they show of it
    KIND = "item"       # For messages: "Delete the camera ...?"
    NEW_NAME = "New item"
    TABLE = "userequipment"  # Where the item's is_active (Enabled) lives
    ENABLED_TIP = ("Offered when choosing equipment in the FOV Simulator.\n"
                   "Turn off what you no longer use - nothing about it is lost.")

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.current_id = None
        self._loading = False
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(400)
        self.save_timer.timeout.connect(self.save_now)

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setToolTip("Untick to disable - it stays here, but isn't offered when choosing equipment")
        self.list.currentItemChanged.connect(self._on_current_changed)
        self.list.itemChanged.connect(self._on_item_checked)
        left_layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(lambda: self.add_new())
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setToolTip(f"A copy of this {self.KIND} to change")
        self.duplicate_button.clicked.connect(self.duplicate)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete)
        for button in (self.add_button, self.duplicate_button, self.delete_button):
            buttons.addWidget(button)
        left_layout.addLayout(buttons)
        splitter.addWidget(left)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.form_widget = QWidget()
        form_column = QVBoxLayout(self.form_widget)
        form_column.setContentsMargins(12, 0, 0, 0)
        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form_column.addLayout(self.form)
        self.build_form(form_column)
        form_column.addStretch()
        scroll.setWidget(self.form_widget)
        self.empty_label = QLabel(self.empty_text())
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        themed_style(self.empty_label, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(11)};")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(scroll, 1)
        right_layout.addWidget(self.empty_label, 1)
        self.scroll = scroll
        splitter.addWidget(right)
        splitter.setSizes([280, 620])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    # --- For subclasses ---
    def build_form(self, column):
        raise NotImplementedError

    def empty_text(self):
        return f"No {self.KIND}s yet. Use Add to create one."

    def load_rows(self, conn):
        """[(id, list text, tooltip or None, dimmed)]"""
        raise NotImplementedError

    def fill_form(self, conn, item_id):
        raise NotImplementedError

    def read_form(self):
        """The form's values as a dict, or None when something needed is missing."""
        raise NotImplementedError

    def insert(self, conn, values):
        raise NotImplementedError

    def update(self, conn, item_id, values):
        raise NotImplementedError

    def usage(self, conn, item_id):
        """Sentences on what uses the item, for the delete question."""
        return []

    def remove(self, conn, item_id):
        raise NotImplementedError

    def refresh_links(self):
        """Another tab changed: refresh anything shown from it."""

    # --- Shared ---
    def connect_field(self, widget):
        """Save when this field changes."""
        for signal in ("textChanged", "valueChanged", "currentIndexChanged", "toggled"):
            if hasattr(widget, signal):
                getattr(widget, signal).connect(self.field_changed)
                return

    def field_changed(self, *_args):
        if self._loading or self.current_id is None:
            return
        self.save_timer.start()

    def mark_invalid(self, widget, invalid):
        themed_style(widget, (lambda: f"border: 1px solid {COLORS['error']};") if invalid else (lambda: ""))

    def save_now(self):
        """Write the form now (when it's valid)."""
        self.save_timer.stop()
        if self.current_id is None or self._loading:
            return
        values = self.read_form()
        if values is None:
            return
        try:
            with self.db_manager.get_connection() as conn:
                self.update(conn, self.current_id, values)
                conn.commit()
                rows = {row[0]: row for row in self._rows(conn)}
        except Exception as e:
            logger.error(f"Error saving the {self.KIND}: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't save the {self.KIND}: {e}")
            return
        # The saved item's own row - when switching items, the list has already moved on
        for index in range(self.list.count()):
            item = self.list.item(index)
            if item.data(Qt.UserRole) == self.current_id and self.current_id in rows:
                self._style_item(item, rows[self.current_id])
                break
        self.changed.emit()

    def _rows(self, conn):
        """load_rows() with whether each is enabled: [(id, text, tooltip, enabled)]."""
        enabled = {row[0] for row in conn.execute(
            f"SELECT id FROM {self.TABLE} WHERE COALESCE(is_active, 1) = 1")}
        return [(item_id, text, tooltip, item_id in enabled) for item_id, text, tooltip, _dim in self.load_rows(conn)]

    def _style_item(self, item, row):
        _id, text, tooltip, enabled = row
        self.list.blockSignals(True)
        item.setText(text)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
        item.setToolTip(tooltip if enabled else "Disabled - kept here, but not offered when choosing equipment")
        item.setForeground(QColor(COLORS["text"] if enabled else COLORS["text_disabled"]))
        self.list.blockSignals(False)

    def set_enabled(self, item_id, enabled):
        """Enable or disable an item (from its list checkbox or its details)."""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(f"UPDATE {self.TABLE} SET is_active = ? WHERE id = ?", (1 if enabled else 0, item_id))
                conn.commit()
                rows = {row[0]: row for row in self._rows(conn)}
        except Exception as e:
            logger.error(f"Error enabling the {self.KIND}: {e}", exc_info=True)
            return
        for index in range(self.list.count()):
            item = self.list.item(index)
            if item.data(Qt.UserRole) == item_id and item_id in rows:
                self._style_item(item, rows[item_id])
        if item_id == self.current_id:
            self.enabled_check.blockSignals(True)
            self.enabled_check.setChecked(enabled)
            self.enabled_check.blockSignals(False)
        self.changed.emit()

    def _on_item_checked(self, item):
        self.set_enabled(item.data(Qt.UserRole), item.checkState() == Qt.Checked)

    def _on_enabled_toggled(self, checked):
        if self.current_id is not None and not self._loading:
            self.set_enabled(self.current_id, checked)

    def reload(self, select_id=None):
        """Refill the list, keeping (or moving to) a selection."""
        self.save_now()
        select_id = select_id if select_id is not None else self.current_id
        try:
            with self.db_manager.get_connection() as conn:
                rows = self._rows(conn)
        except Exception as e:
            logger.error(f"Error loading {self.KIND}s: {e}", exc_info=True)
            rows = []
        self.list.blockSignals(True)
        self.list.clear()
        select_row = 0 if rows else -1
        for index, row in enumerate(rows):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, row[0])
            self._style_item(item, row)
            self.list.addItem(item)
            if row[0] == select_id:
                select_row = index
        self.list.blockSignals(False)
        self.current_id = None
        if select_row >= 0:
            self.list.setCurrentRow(select_row)
            self._show(self.list.item(select_row).data(Qt.UserRole))
        else:
            self._show(None)

    def _on_current_changed(self, current, _previous):
        self.save_now()  # The one being left
        self._show(current.data(Qt.UserRole) if current is not None else None)

    def _show(self, item_id):
        self.current_id = item_id
        has_item = item_id is not None
        self.scroll.setVisible(has_item)
        self.empty_label.setVisible(not has_item)
        self.duplicate_button.setEnabled(has_item)
        self.delete_button.setEnabled(has_item)
        if not has_item:
            return
        self._loading = True
        try:
            with self.db_manager.get_connection() as conn:
                self.fill_form(conn, item_id)
                row = conn.execute(f"SELECT COALESCE(is_active, 1) FROM {self.TABLE} WHERE id = ?",
                                   (item_id,)).fetchone()
                self.enabled_check.setChecked(bool(row and row[0]))
        except Exception as e:
            logger.error(f"Error loading the {self.KIND}: {e}", exc_info=True)
        finally:
            self._loading = False

    def add_new(self, values=None):
        """A new item, selected with its name ready to type over."""
        self.save_now()
        try:
            with self.db_manager.get_connection() as conn:
                new_id = self.insert(conn, values or {"name": self.NEW_NAME})
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding a {self.KIND}: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't add the {self.KIND}: {e}")
            return None
        self.reload(select_id=new_id)
        self.name_edit.setFocus()
        self.name_edit.selectAll()
        self.changed.emit()
        return new_id

    def duplicate(self):
        values = self.read_form()
        if values is None or self.current_id is None:
            return
        source = self.current_id
        values = dict(values, name=f"{values['name']} (copy)")
        new_id = self.add_new(values)
        if new_id is not None:
            self.copy_links(source, new_id)
            self.reload(select_id=new_id)
            self.changed.emit()

    def copy_links(self, source_id, new_id):
        """Give a copy the same telescope/equipment links."""
        with self.db_manager.get_connection() as conn:
            column, other = self.link_columns()
            for (linked,) in conn.execute(f"SELECT {other} FROM telescope_equipment WHERE {column} = ?",
                                          (source_id,)).fetchall():
                conn.execute(f"INSERT OR IGNORE INTO telescope_equipment ({column}, {other}) VALUES (?, ?)",
                             (new_id, linked))
            conn.commit()

    def link_columns(self):
        return "equipment_id", "telescope_id"

    def delete(self):
        if self.current_id is None:
            return
        name = self.name_edit.text().strip() or f"this {self.KIND}"
        try:
            with self.db_manager.get_connection() as conn:
                usage = self.usage(conn, self.current_id)
        except Exception as e:
            logger.error(f"Error checking what uses the {self.KIND}: {e}")
            usage = []
        text = f"Delete the {self.KIND} \"{name}\"?"
        if usage:
            text += "\n\n" + "\n".join(usage)
        if QMessageBox.question(self, f"Delete {self.KIND.title()}", text,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.save_timer.stop()
        try:
            with self.db_manager.get_connection() as conn:
                self.remove(conn, self.current_id)
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting the {self.KIND}: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't delete the {self.KIND}: {e}")
            return
        row = self.list.currentRow()
        self.current_id = None
        self.reload()
        if self.list.count():
            self.list.setCurrentRow(min(row, self.list.count() - 1))
        self.changed.emit()

    def add_name_field(self, placeholder, presets=None, on_preset=None):
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(placeholder)
        if presets:
            completer = QCompleter(sorted(presets, key=str.lower), self.name_edit)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            completer.activated.connect(on_preset)
            self.name_edit.setCompleter(completer)
            self.name_edit.setToolTip("Start typing to pick from known models - their details fill in for you")
        self.connect_field(self.name_edit)
        self.form.addRow("Name:", self.name_edit)
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setToolTip(self.ENABLED_TIP)
        self.enabled_check.toggled.connect(self._on_enabled_toggled)
        self.form.addRow("", self.enabled_check)

    def read_name(self):
        name = self.name_edit.text().strip()
        self.mark_invalid(self.name_edit, not name)
        return name or None

    def add_telescope_links(self, column):
        """'Used with telescopes' for an equipment item."""
        group = QGroupBox("Used with telescopes")
        group.setToolTip("Ticked telescopes offer this in the FOV Simulator and pick it automatically")
        group_layout = QVBoxLayout(group)
        self.telescope_links = _CheckList()
        self.telescope_links.setMaximumHeight(150)
        self.telescope_links.toggled.connect(self._on_telescope_link)
        group_layout.addWidget(self.telescope_links)
        column.addWidget(group)

    def fill_telescope_links(self, conn, item_id):
        telescopes = [(item_id, name if active else f"{name} (disabled)") for item_id, name, active in conn.execute(
            "SELECT id, name, COALESCE(is_active, 1) FROM usertelescopes ORDER BY name COLLATE NOCASE")]
        linked = {row[0] for row in conn.execute(
            "SELECT telescope_id FROM telescope_equipment WHERE equipment_id = ?", (item_id,))}
        self.telescope_links.fill(telescopes, linked)

    def _on_telescope_link(self, telescope_id, checked):
        if self.current_id is not None:
            _set_link(self.db_manager, telescope_id, self.current_id, checked)
            self.changed.emit()

    def linked_telescope_names(self, conn, item_id):
        return [name for (name,) in conn.execute(
            "SELECT t.name FROM telescope_equipment te JOIN usertelescopes t ON t.id = te.telescope_id "
            "WHERE te.equipment_id = ? ORDER BY t.name COLLATE NOCASE", (item_id,))]

    def remove_equipment(self, conn, item_id):
        conn.execute("DELETE FROM telescope_equipment WHERE equipment_id = ?", (item_id,))
        conn.execute("DELETE FROM userequipment WHERE id = ?", (item_id,))


def _set_link(db_manager, telescope_id, equipment_id, linked):
    try:
        with db_manager.get_connection() as conn:
            if linked:
                conn.execute("INSERT OR IGNORE INTO telescope_equipment (telescope_id, equipment_id) VALUES (?, ?)",
                             (telescope_id, equipment_id))
            else:
                conn.execute("DELETE FROM telescope_equipment WHERE telescope_id = ? AND equipment_id = ?",
                             (telescope_id, equipment_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving the equipment link: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Telescopes
# ---------------------------------------------------------------------------

class TelescopesPage(_EquipmentPage):
    KIND = "telescope"
    NEW_NAME = "New Telescope"
    TABLE = "usertelescopes"
    ENABLED_TIP = ("Offered when choosing a telescope - in the FOV Simulator, sessions and the target list.\n"
                   "Turn off ones you no longer use: they stay in your sessions and history.")

    def build_form(self, column):
        self.add_name_field("e.g. My Celestron 8SE")
        self.aperture_box = _number_box(2000, 1, " mm")
        self.focal_box = _number_box(20000, 1, " mm")
        self.ratio_label = QLabel()
        self.mount_combo = QComboBox()
        self.mount_combo.addItems(MOUNT_TYPES)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Anything to remember about it")
        self.notes_edit.setMaximumHeight(70)
        self.usage_label = QLabel()
        themed_style(self.usage_label, lambda: f"color: {COLORS['text_secondary']};")
        self.form.addRow("Aperture:", self.aperture_box)
        self.form.addRow("Focal length:", self.focal_box)
        self.form.addRow("Focal ratio:", self.ratio_label)
        self.form.addRow("Mount:", self.mount_combo)
        self.form.addRow("Notes:", self.notes_edit)
        self.form.addRow("", self.usage_label)
        for widget in (self.aperture_box, self.focal_box, self.mount_combo):
            self.connect_field(widget)
        self.notes_edit.textChanged.connect(self.field_changed)
        self.aperture_box.valueChanged.connect(self._update_ratio)
        self.focal_box.valueChanged.connect(self._update_ratio)

        group = QGroupBox("Used with this telescope")
        group.setToolTip("Ticked equipment is offered with this telescope in the FOV Simulator")
        links = QHBoxLayout(group)
        self.link_lists = {}
        for key, title in (("camera", "Cameras"), ("eyepiece", "Eyepieces"), ("barlow", "Barlows/Reducers")):
            box = QVBoxLayout()
            box.addWidget(QLabel(title))
            check_list = _CheckList()
            check_list.setMinimumHeight(120)
            check_list.toggled.connect(self._on_equipment_link)
            box.addWidget(check_list)
            links.addLayout(box)
            self.link_lists[key] = check_list
        column.addWidget(group)

        # Add: a blank telescope, or a smart telescope with its camera
        from FOVSimulator import AladinLiteWindow
        menu = QMenu(self.add_button)
        menu.addAction("New Telescope").triggered.connect(lambda: self.add_new())
        smart = menu.addMenu("Smart Telescope")
        for preset in AladinLiteWindow.SMART_TELESCOPES:
            smart.addAction(f"{preset['name']} ({preset['focal_length']}mm f/"
                            f"{preset['focal_length'] / preset['aperture']:.1f})").triggered.connect(
                lambda _=False, p=preset: self.add_smart_telescope(p))
        self._smart_menu = smart  # Kept: PySide can drop a submenu only Python refers to
        self.add_button.setText("Add ▾")
        self.add_button.setMenu(menu)

    def empty_text(self):
        return "No telescopes yet. Use Add to create one - or pick a smart telescope like a Seestar."

    def _update_ratio(self, *_args):
        aperture, focal = self.aperture_box.value(), self.focal_box.value()
        self.ratio_label.setText(f"f/{focal / aperture:.1f}" if aperture and focal else "—")

    def load_rows(self, conn):
        rows = []
        for item_id, name, aperture, focal, active in conn.execute(
                "SELECT id, name, aperture, focal_length, is_active FROM usertelescopes ORDER BY name COLLATE NOCASE"):
            text = name
            if focal:
                text += f"  ·  {_fmt(focal)} mm" + (f" f/{focal / aperture:.1f}" if aperture else "")
            rows.append((item_id, text, None, not active))
        return rows

    def fill_form(self, conn, item_id):
        name, aperture, focal, mount, notes, active = conn.execute(
            "SELECT name, aperture, focal_length, mount_type, notes, is_active FROM usertelescopes WHERE id = ?",
            (item_id,)).fetchone()
        self.name_edit.setText(name or "")
        self.mark_invalid(self.name_edit, False)
        _set(self.aperture_box, aperture)
        _set(self.focal_box, focal)
        self._update_ratio()
        index = self.mount_combo.findText(mount or "")
        self.mount_combo.setCurrentIndex(index if index >= 0 else 0)
        self.notes_edit.setPlainText(notes or "")
        self.usage_label.setText(" ".join(self.usage(conn, item_id, short=True)))
        self.refresh_links(conn)

    def refresh_links(self, conn=None):
        if self.current_id is None:
            return
        if conn is None:
            with self.db_manager.get_connection() as conn:
                return self.refresh_links(conn)
        linked = {row[0] for row in conn.execute(
            "SELECT equipment_id FROM telescope_equipment WHERE telescope_id = ?", (self.current_id,))}
        for key, check_list in self.link_lists.items():
            types = ("barlow", "reducer") if key == "barlow" else (key,)
            rows = [(item_id, name if active else f"{name} (disabled)") for item_id, name, active in conn.execute(
                f"SELECT id, name, COALESCE(is_active, 1) FROM userequipment "
                f"WHERE equipment_type IN ({','.join('?' * len(types))}) ORDER BY name COLLATE NOCASE", types)]
            check_list.fill(rows, linked)

    def _on_equipment_link(self, equipment_id, checked):
        if self.current_id is not None:
            _set_link(self.db_manager, self.current_id, equipment_id, checked)
            self.changed.emit()

    def read_form(self):
        name = self.read_name()
        if name is None:
            return None
        return {"name": name, "aperture": _value(self.aperture_box), "focal_length": _value(self.focal_box),
                "mount_type": self.mount_combo.currentText(), "notes": self.notes_edit.toPlainText().strip(),
                "is_active": 1 if self.enabled_check.isChecked() else 0}

    def insert(self, conn, values):
        values = {"aperture": None, "focal_length": None, "mount_type": MOUNT_TYPES[0], "notes": "",
                  "is_active": 1, **values}
        return conn.execute(
            "INSERT INTO usertelescopes (name, aperture, focal_length, mount_type, notes, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (values["name"], values["aperture"], values["focal_length"], values["mount_type"], values["notes"],
             values["is_active"])).lastrowid

    def update(self, conn, item_id, values):
        conn.execute("UPDATE usertelescopes SET name = ?, aperture = ?, focal_length = ?, mount_type = ?, "
                     "notes = ?, is_active = ? WHERE id = ?",
                     (values["name"], values["aperture"], values["focal_length"], values["mount_type"],
                      values["notes"], values["is_active"], item_id))

    def link_columns(self):
        return "telescope_id", "equipment_id"

    def usage(self, conn, item_id, short=False):
        sessions = conn.execute("SELECT COUNT(*) FROM usersessions WHERE telescope_id = ?", (item_id,)).fetchone()[0]
        targets = conn.execute("SELECT COUNT(*) FROM usertargetlist WHERE telescope_id = ?",
                               (item_id,)).fetchone()[0]
        linked = conn.execute("SELECT COUNT(*) FROM telescope_equipment WHERE telescope_id = ?",
                              (item_id,)).fetchone()[0]
        images = EquipmentMapper.images_using(conn, TELESCOPE, item_id)
        uses = [_count(n, word) for n, word in ((sessions, "session"), (targets, "target")) if n]
        if short:
            uses += [_count(images, "gallery image")] if images else []
            return [f"Used by {_and(uses)}."] if uses else []
        lines = []
        if uses:
            lines.append(f"It's used by {_and(uses)}. They keep everything else, "
                         "but won't show a telescope any more.")
        if images:
            lines.append(_images_keep_name(images))
        if linked:
            lines.append("Its cameras, eyepieces and barlows stay in your equipment.")
        return lines

    def remove(self, conn, item_id):
        EquipmentMapper.keep_names_of(conn, TELESCOPE, item_id)
        conn.execute("UPDATE usersessions SET telescope_id = NULL WHERE telescope_id = ?", (item_id,))
        conn.execute("UPDATE usertargetlist SET telescope_id = NULL WHERE telescope_id = ?", (item_id,))
        conn.execute("UPDATE userequipment SET telescope_id = NULL WHERE telescope_id = ?", (item_id,))
        conn.execute("DELETE FROM telescope_equipment WHERE telescope_id = ?", (item_id,))
        conn.execute("DELETE FROM usertelescopes WHERE id = ?", (item_id,))

    def add_smart_telescope(self, preset):
        """A smart telescope and its built-in camera, already linked."""
        from FOVSimulator import camera_presets, preset_camera_name
        camera_name = preset_camera_name(preset["camera_text"])
        width, height, pixel, _text = camera_presets().get(camera_name, (None, None, None, None))
        try:
            with self.db_manager.get_connection() as conn:
                telescope_id = self.insert(conn, {"name": preset["name"], "aperture": preset["aperture"],
                                                  "focal_length": preset["focal_length"], "mount_type": "Alt-Az"})
                # Reuse a matching camera if it's already there
                camera_id = None
                for existing_id, w, h in conn.execute(
                        "SELECT id, sensor_width, sensor_height FROM userequipment "
                        "WHERE equipment_type = 'camera' AND name = ? COLLATE NOCASE", (camera_name,)):
                    if w and h and width and abs(w - width) < 0.05 and abs(h - height) < 0.05:
                        camera_id = existing_id
                        break
                if camera_id is None:
                    camera_id = conn.execute(
                        "INSERT INTO userequipment (equipment_type, name, sensor_width, sensor_height, pixel_size) "
                        "VALUES ('camera', ?, ?, ?, ?)", (camera_name, width, height, pixel)).lastrowid
                conn.execute("INSERT OR IGNORE INTO telescope_equipment (telescope_id, equipment_id) VALUES (?, ?)",
                             (telescope_id, camera_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding {preset['name']}: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't add {preset['name']}: {e}")
            return
        self.reload(select_id=telescope_id)
        self.changed.emit()


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

class CamerasPage(_EquipmentPage):
    KIND = "camera"
    NEW_NAME = "New Camera"

    def __init__(self, db_manager, parent=None):
        self._sub_pixels = None
        self._linking = False
        super().__init__(db_manager, parent)

    def build_form(self, column):
        from FOVSimulator import camera_presets
        self.presets = camera_presets()
        self.add_name_field("e.g. ASI2600MC Pro", self.presets, self._apply_preset)
        size = QHBoxLayout()
        self.width_box = _number_box(100, 2, " mm")
        self.height_box = _number_box(100, 2, " mm")
        size.addWidget(self.width_box)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height_box)
        size.addStretch()
        self.form.addRow("Sensor size:", size)
        resolution = QHBoxLayout()
        self.res_x_box = QSpinBox()
        self.res_y_box = QSpinBox()
        for box in (self.res_x_box, self.res_y_box):
            box.setRange(0, 100000)
            box.setSuffix(" px")
            box.setSpecialValueText("Not set")
            box.setMinimumWidth(110)
        resolution.addWidget(self.res_x_box)
        resolution.addWidget(QLabel("×"))
        resolution.addWidget(self.res_y_box)
        resolution.addStretch()
        self.form.addRow("Resolution:", resolution)
        self.pixel_box = _number_box(50, 2, " µm", unset="Unknown")
        self.pixel_box.setToolTip("Optional - for the pixel scale (\"/px) in the FOV Simulator")
        self.form.addRow("Pixel size:", self.pixel_box)
        self.pixel_hint = QLabel()
        self.pixel_hint.setWordWrap(True)
        self.pixel_hint.setToolTip("Capture software records the camera's pixel size in each sub it saves.\n"
                                   "\"Verified\" means your subs taken with this camera agree on it.")
        themed_style(self.pixel_hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        self.form.addRow("", self.pixel_hint)
        hint = QLabel("Fill in any two of sensor size, resolution and pixel size - the third is worked out.")
        hint.setWordWrap(True)
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        self.form.addRow("", hint)
        for box in (self.width_box, self.height_box, self.pixel_box):
            self.connect_field(box)
        self.width_box.valueChanged.connect(lambda *_: self._link_sizes("sensor"))
        self.height_box.valueChanged.connect(lambda *_: self._link_sizes("sensor"))
        self.res_x_box.valueChanged.connect(lambda *_: self._link_sizes("resolution"))
        self.res_y_box.valueChanged.connect(lambda *_: self._link_sizes("resolution"))
        self.pixel_box.valueChanged.connect(lambda *_: self._link_sizes("pixel"))
        self.name_edit.editingFinished.connect(self._pixel_from_subs)
        self.add_telescope_links(column)

    def _link_sizes(self, edited):
        """Sensor size, resolution and pixel size: any two give the third."""
        if self._linking or self._loading:
            return
        self._linking = True
        try:
            width, height = self.width_box.value(), self.height_box.value()
            res_x, res_y = self.res_x_box.value(), self.res_y_box.value()
            pixel = self.pixel_box.value()

            def resolution_from_sensor():
                self.res_x_box.setValue(round(width * 1000.0 / pixel) if width else 0)
                self.res_y_box.setValue(round(height * 1000.0 / pixel) if height else 0)

            def sensor_from_resolution():
                if res_x:
                    self.width_box.setValue(res_x * pixel / 1000.0)
                if res_y:
                    self.height_box.setValue(res_y * pixel / 1000.0)

            if edited == "resolution":
                if width and res_x:
                    self.pixel_box.setValue(width * 1000.0 / res_x)
                    self.pixel_hint.setText("Worked out from the sensor size and resolution.")
                    if res_y and not height:
                        self.height_box.setValue(res_y * self.pixel_box.value() / 1000.0)
                elif pixel:
                    sensor_from_resolution()
            elif edited == "pixel":
                self.pixel_hint.setText("")
                if pixel and (width or height):
                    resolution_from_sensor()
                elif pixel and (res_x or res_y):
                    sensor_from_resolution()
            else:  # The sensor size
                if pixel:
                    resolution_from_sensor()
                elif width and res_x:
                    self.pixel_box.setValue(width * 1000.0 / res_x)
                    self.pixel_hint.setText("Worked out from the sensor size and resolution.")
        finally:
            self._linking = False
        self.field_changed()

    def _apply_preset(self, name):
        preset = self.presets.get(name)
        if preset is None:
            return
        width, height, pixel, _text = preset
        self._loading = True
        try:
            self.width_box.setValue(width)
            self.height_box.setValue(height)
            self.pixel_box.setValue(pixel or 0)
            self.res_x_box.setValue(round(width * 1000 / pixel) if pixel else 0)
            self.res_y_box.setValue(round(height * 1000 / pixel) if pixel else 0)
        finally:
            self._loading = False
        self.pixel_hint.setText("From the camera's specifications." if pixel else "")
        self.save_now()

    def _subs(self):
        if self._sub_pixels is None:
            with self.db_manager.get_connection() as conn:
                self._sub_pixels = pixel_sizes_from_subs(conn)
        return self._sub_pixels

    def _pixel_from_subs(self):
        """A name was typed: its pixel size from the user's subs, if it isn't known yet."""
        if self.current_id is None or self.pixel_box.value():
            return
        found = pixel_size_for(self.name_edit.text(), self._subs())
        if found:
            self.pixel_box.setValue(found[0])  # Links the resolution, saves
            self.pixel_hint.setText(f"Verified from {found[1]} of your subs.")

    def load_rows(self, conn):
        rows = []
        for item_id, name, width, height, pixel in conn.execute(
                "SELECT id, name, sensor_width, sensor_height, pixel_size FROM userequipment "
                "WHERE equipment_type = 'camera' ORDER BY name COLLATE NOCASE"):
            text = name
            if width and height:
                text += f"  ·  {_fmt(width)} × {_fmt(height)} mm"
            rows.append((item_id, text, f"{_fmt(pixel)} µm pixels" if pixel else None, False))
        return rows

    def fill_form(self, conn, item_id):
        name, width, height, pixel = conn.execute(
            "SELECT name, sensor_width, sensor_height, pixel_size FROM userequipment WHERE id = ?",
            (item_id,)).fetchone()
        self.name_edit.setText(name or "")
        self.mark_invalid(self.name_edit, False)
        _set(self.width_box, width)
        _set(self.height_box, height)
        _set(self.pixel_box, pixel)
        self.res_x_box.setValue(round(width * 1000 / pixel) if width and pixel else 0)
        self.res_y_box.setValue(round(height * 1000 / pixel) if height and pixel else 0)
        found = pixel_size_for(name, self._subs())
        preset = self.presets.get(name)
        if pixel and found and abs(found[0] - pixel) < 0.01:
            self.pixel_hint.setText(f"Verified from {found[1]} of your subs.")
        elif pixel and preset and preset[2] and abs(preset[2] - pixel) < 0.01:
            self.pixel_hint.setText("From the camera's specifications.")
        elif not pixel:
            self.pixel_hint.setText("Not known - pick the camera from the list as you type its name, "
                                    "or enter its resolution.")
        else:
            self.pixel_hint.setText("")
        self.fill_telescope_links(conn, item_id)

    def refresh_links(self):
        if self.current_id is not None:
            with self.db_manager.get_connection() as conn:
                self.fill_telescope_links(conn, self.current_id)

    def read_form(self):
        name = self.read_name()
        if name is None:
            return None
        return {"name": name, "sensor_width": _value(self.width_box), "sensor_height": _value(self.height_box),
                "pixel_size": _value(self.pixel_box)}

    def insert(self, conn, values):
        return conn.execute(
            "INSERT INTO userequipment (equipment_type, name, sensor_width, sensor_height, pixel_size) "
            "VALUES ('camera', ?, ?, ?, ?)",
            (values["name"], values.get("sensor_width"), values.get("sensor_height"),
             values.get("pixel_size"))).lastrowid

    def update(self, conn, item_id, values):
        conn.execute("UPDATE userequipment SET name = ?, sensor_width = ?, sensor_height = ?, pixel_size = ? "
                     "WHERE id = ?", (values["name"], values["sensor_width"], values["sensor_height"],
                                      values["pixel_size"], item_id))

    def usage(self, conn, item_id):
        names = self.linked_telescope_names(conn, item_id)
        lines = [f"It's used with {', '.join(names)}."] if names else []
        lines.append("Sessions keep the camera name they were saved with.")
        images = EquipmentMapper.images_using(conn, CAMERA, item_id)
        if images:
            lines.append(_images_keep_name(images))
        return lines

    def remove(self, conn, item_id):
        EquipmentMapper.keep_names_of(conn, CAMERA, item_id)
        self.remove_equipment(conn, item_id)


# ---------------------------------------------------------------------------
# Eyepieces
# ---------------------------------------------------------------------------

class EyepiecesPage(_EquipmentPage):
    KIND = "eyepiece"
    NEW_NAME = "New Eyepiece"

    def build_form(self, column):
        self.add_name_field("e.g. Televue 13mm Ethos", EYEPIECE_PRESETS, self._apply_preset)
        self.focal_box = _number_box(100, 1, " mm")
        self.afov_box = _number_box(120, 0, "°")
        self.afov_box.setToolTip("Apparent field of view, from the eyepiece's specifications")
        self.form.addRow("Focal length:", self.focal_box)
        self.form.addRow("Apparent field:", self.afov_box)
        for box in (self.focal_box, self.afov_box):
            self.connect_field(box)
        self.add_telescope_links(column)

    def _apply_preset(self, name):
        if name in EYEPIECE_PRESETS:
            focal, afov = EYEPIECE_PRESETS[name]
            self._loading = True
            self.focal_box.setValue(focal)
            self.afov_box.setValue(afov)
            self._loading = False
            self.save_now()

    def load_rows(self, conn):
        rows = []
        for item_id, name, focal, afov in conn.execute(
                "SELECT id, name, focal_length, apparent_fov FROM userequipment "
                "WHERE equipment_type = 'eyepiece' ORDER BY name COLLATE NOCASE"):
            text = name + (f"  ·  {_fmt(focal)} mm" if focal else "") + (f", {_fmt(afov)}°" if afov else "")
            rows.append((item_id, text, None, False))
        return rows

    def fill_form(self, conn, item_id):
        name, focal, afov = conn.execute(
            "SELECT name, focal_length, apparent_fov FROM userequipment WHERE id = ?", (item_id,)).fetchone()
        self.name_edit.setText(name or "")
        self.mark_invalid(self.name_edit, False)
        _set(self.focal_box, focal)
        _set(self.afov_box, afov)
        self.fill_telescope_links(conn, item_id)

    def refresh_links(self):
        if self.current_id is not None:
            with self.db_manager.get_connection() as conn:
                self.fill_telescope_links(conn, self.current_id)

    def read_form(self):
        name = self.read_name()
        if name is None:
            return None
        return {"name": name, "focal_length": _value(self.focal_box), "apparent_fov": _value(self.afov_box)}

    def insert(self, conn, values):
        return conn.execute(
            "INSERT INTO userequipment (equipment_type, name, focal_length, apparent_fov) VALUES ('eyepiece', ?, ?, ?)",
            (values["name"], values.get("focal_length"), values.get("apparent_fov"))).lastrowid

    def update(self, conn, item_id, values):
        conn.execute("UPDATE userequipment SET name = ?, focal_length = ?, apparent_fov = ? WHERE id = ?",
                     (values["name"], values["focal_length"], values["apparent_fov"], item_id))

    def usage(self, conn, item_id):
        names = self.linked_telescope_names(conn, item_id)
        return [f"It's used with {', '.join(names)}."] if names else []

    def remove(self, conn, item_id):
        self.remove_equipment(conn, item_id)


# ---------------------------------------------------------------------------
# Barlows and reducers
# ---------------------------------------------------------------------------

class BarlowsPage(_EquipmentPage):
    KIND = "barlow/reducer"
    NEW_NAME = "New Barlow"

    def build_form(self, column):
        self.add_name_field("e.g. Televue 2x Powermate", BARLOW_PRESETS, self._apply_preset)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Barlow", "Reducer"])
        self.factor_box = _number_box(10, 2, "x")
        self.factor_box.setToolTip("How it changes the focal length: 2x for a barlow, 0.63x for a reducer")
        self.form.addRow("Type:", self.type_combo)
        self.form.addRow("Factor:", self.factor_box)
        self.connect_field(self.type_combo)
        self.connect_field(self.factor_box)
        self.add_telescope_links(column)

    def empty_text(self):
        return "No barlows or reducers yet. Use Add to create one."

    def _apply_preset(self, name):
        if name in BARLOW_PRESETS:
            factor, kind = BARLOW_PRESETS[name]
            self._loading = True
            self.factor_box.setValue(factor)
            self.type_combo.setCurrentText(kind.title())
            self._loading = False
            self.save_now()

    def load_rows(self, conn):
        rows = []
        for item_id, name, factor, kind in conn.execute(
                "SELECT id, name, factor, equipment_type FROM userequipment "
                "WHERE equipment_type IN ('barlow', 'reducer') ORDER BY name COLLATE NOCASE"):
            rows.append((item_id, name + (f"  ·  {_fmt(factor)}x" if factor else ""), kind.title(), False))
        return rows

    def fill_form(self, conn, item_id):
        name, factor, kind = conn.execute(
            "SELECT name, factor, equipment_type FROM userequipment WHERE id = ?", (item_id,)).fetchone()
        self.name_edit.setText(name or "")
        self.mark_invalid(self.name_edit, False)
        _set(self.factor_box, factor)
        self.type_combo.setCurrentText((kind or "barlow").title())
        self.fill_telescope_links(conn, item_id)

    def refresh_links(self):
        if self.current_id is not None:
            with self.db_manager.get_connection() as conn:
                self.fill_telescope_links(conn, self.current_id)

    def read_form(self):
        name = self.read_name()
        if name is None:
            return None
        return {"name": name, "factor": _value(self.factor_box), "equipment_type": self.type_combo.currentText().lower()}

    def insert(self, conn, values):
        kind = values.get("equipment_type") or ("reducer" if (values.get("factor") or 2) < 1 else "barlow")
        return conn.execute("INSERT INTO userequipment (equipment_type, name, factor) VALUES (?, ?, ?)",
                            (kind, values["name"], values.get("factor"))).lastrowid

    def update(self, conn, item_id, values):
        conn.execute("UPDATE userequipment SET name = ?, factor = ?, equipment_type = ? WHERE id = ?",
                     (values["name"], values["factor"], values["equipment_type"], item_id))

    def usage(self, conn, item_id):
        names = self.linked_telescope_names(conn, item_id)
        return [f"It's used with {', '.join(names)}."] if names else []

    def remove(self, conn, item_id):
        self.remove_equipment(conn, item_id)


# ---------------------------------------------------------------------------
# Matches: the names in files and images, and the equipment each one is
# ---------------------------------------------------------------------------

# Combo item data (strings: Qt can't compare Python tuples in findData)
NOT_MATCHED, ADD_NEW = "none", "new"


def _equipment_combo(kind, info, chosen_id, not_matched="Not matched - keep the name", add_new=True):
    """A picker of the user's telescopes or cameras (enabled ones first),
    with chosen_id selected - else the 'not matched' item."""
    combo = QComboBox()
    if not_matched:
        combo.addItem(not_matched, NOT_MATCHED)
    items = info.items(kind)
    for item_id in sorted(items, key=lambda i: (not items[i]["is_active"], (items[i]["name"] or "").lower())):
        item = items[item_id]
        combo.addItem(item["name"] + ("" if item["is_active"] else " (disabled)"), f"id:{item_id}")
    if add_new:
        combo.addItem(f"Add as a new {kind}", ADD_NEW)
    index = combo.findData(f"id:{chosen_id}") if chosen_id is not None else -1
    combo.setCurrentIndex(index if index >= 0 else 0)
    return combo


def _combo_choice(combo):
    """An _equipment_combo's choice: an equipment id, None (not matched) or ADD_NEW."""
    choice = combo.currentData()
    return int(choice[3:]) if isinstance(choice, str) and choice.startswith("id:") else \
        (ADD_NEW if choice == ADD_NEW else None)


def _add_equipment(conn, kind, name, header=None):
    """A new telescope or camera named so (the one already named so, if
    there is one), with what a file's header says about it. Its id."""
    measured = LibraryScanner.equipment_from_header(kind, header)
    if kind == TELESCOPE:
        return LibraryScanner.add_telescope(conn, name, measured["focal_length"], measured["aperture"])
    width, height = measured["sensor_width"], measured["sensor_height"]
    if width and height and width < height:  # A portrait frame: the sensor is landscape
        width, height = height, width
    return LibraryScanner.add_camera(conn, name, width, height, measured["pixel_size"])


def _sized_rows(table):
    """Even rows, and no taller than they need (the space left goes below)."""
    row_height = max(QComboBox().sizeHint().height() + 6, 26)
    table.verticalHeader().setDefaultSectionSize(row_height)
    for row in range(table.rowCount()):
        table.setRowHeight(row, row_height)
    table.setMaximumHeight(table.horizontalHeader().sizeHint().height() + row_height * table.rowCount()
                           + 2 * table.frameWidth() + 2)


def refresh_image_views():
    """An open gallery and the main window show images' new equipment."""
    try:
        from DSOGallery import refresh_open_image_views
        refresh_open_image_views()
    except Exception as e:
        logger.debug(f"Couldn't refresh the image views: {e}")


def refresh_equipment_window():
    """An open Equipment window shows equipment added elsewhere."""
    try:
        if _window is not None and _window.isVisible():
            for page in _window.pages:
                page.reload()
    except RuntimeError:
        pass  # Closed and deleted


class MatchesPage(QWidget):
    """The telescope and camera names in the user's files and gallery images,
    and which equipment each one is: change a match (the images given that
    name follow), forget one, or review images that could be linked."""

    changed = Signal()

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self._suggestions = []
        self._info = LibraryScanner.EquipmentInfo()

        layout = QVBoxLayout(self)
        intro = QLabel("The telescope and camera names in your files, gallery images and sessions, and which of "
                       "your equipment each one is. Changing a match updates the images and sessions given that "
                       "name, and the Import Wizard uses it for files you import later.")
        intro.setWordWrap(True)
        themed_style(intro, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(intro)

        # Images that could be linked to equipment
        self.fix_bar = QFrame()
        self.fix_bar.setObjectName("fixBar")
        themed_style(self.fix_bar, lambda: f"QFrame#fixBar {{ background: {COLORS['background_light']}; "
                                           f"border: 1px solid {COLORS['warning']}; border-radius: 4px; }}")
        bar = QHBoxLayout(self.fix_bar)
        bar.setContentsMargins(8, 4, 8, 4)
        self.fix_label = QLabel()
        self.fix_label.setWordWrap(True)
        bar.addWidget(self.fix_label, 1)
        fix_button = QPushButton("Review and Fix...")
        fix_button.setAutoDefault(False)
        fix_button.setToolTip("See which images could be linked to your equipment, and link them")
        fix_button.clicked.connect(self._review)
        bar.addWidget(fix_button)
        layout.addWidget(self.fix_bar)

        self.tables = {}
        for kind, heading in ((TELESCOPE, "Telescopes"), (CAMERA, "Cameras")):
            label = QLabel(heading)
            themed_style(label, lambda: f"font-weight: bold; font-size: {font_size(11)};")
            table = QTableWidget(0, 4)
            table.setHorizontalHeaderLabels(["Name", "Used by", "Is", ""])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.NoSelection)
            table.setWordWrap(False)
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            table.setColumnWidth(2, 320)
            layout.addWidget(label)
            layout.addWidget(table, 1)
            self.tables[kind] = (label, table)
        self.empty_label = QLabel("No names yet. The names in your files appear here once you've used the "
                                  "Import Wizard, and names on gallery images and sessions that aren't in "
                                  "your equipment.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        themed_style(self.empty_label, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(11)};")
        layout.addWidget(self.empty_label, 1)
        layout.addStretch()
        self.status = QLabel()
        themed_style(self.status, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.status)

    # The window's calls to every tab
    def save_now(self):
        pass

    def refresh_links(self):
        self.reload()

    def reload(self, select_id=None):
        try:
            with self.db_manager.get_connection() as conn:
                self._info = LibraryScanner.load_equipment_info(conn)
                rows = EquipmentMapper.matches(conn, self._info)
                self._suggestions = EquipmentMapper.suggestions(conn, self._info)
        except Exception as e:
            logger.error(f"Error loading the equipment matches: {e}", exc_info=True)
            rows, self._suggestions = [], []
        for kind, (label, table) in self.tables.items():
            kind_rows = [r for r in rows if r["kind"] == kind]
            label.setVisible(bool(kind_rows))
            table.setVisible(bool(kind_rows))
            table.setRowCount(len(kind_rows))
            for row, entry in enumerate(kind_rows):
                self._fill_row(table, row, entry)
            _sized_rows(table)
        self.empty_label.setVisible(not rows)
        could = {table: len({i for s in self._suggestions for i in s["ids"].get(table, [])})
                 for table in (EquipmentMapper.IMAGES, EquipmentMapper.SESSIONS)}
        self.fix_bar.setVisible(any(could.values()))
        self.fix_label.setText(f"<b>{EquipmentMapper.describe(could)}</b> could be linked to your equipment.")

    def _fill_row(self, table, row, entry):
        name_item = QTableWidgetItem(entry["name"])
        name_item.setToolTip("Matched in the Import Wizard or here" if entry["saved"] else
                             "Given to gallery images or sessions, but not matched to your equipment")
        table.setItem(row, 0, name_item)
        table.setItem(row, 1, QTableWidgetItem(EquipmentMapper.describe(entry["used"]) or "-"))
        combo = _equipment_combo(entry["kind"], self._info, entry["equipment_id"])
        combo.currentIndexChanged.connect(lambda _i, e=entry, c=combo: self._on_match_changed(e, c))
        table.setCellWidget(row, 2, combo)
        forget = QPushButton("Forget")
        forget.setAutoDefault(False)
        forget.setEnabled(entry["saved"])
        forget.setToolTip("Forget this match: the Import Wizard asks about the name again. "
                          "Images and sessions keep their equipment." if entry["saved"] else "Not a saved match")
        forget.clicked.connect(lambda _checked=False, e=entry: self._forget(e))
        table.setCellWidget(row, 3, forget)

    def _on_match_changed(self, entry, combo):
        kind, name = entry["kind"], entry["name"]
        choice = _combo_choice(combo)
        try:
            with self.db_manager.get_connection() as conn:
                if choice == ADD_NEW:
                    choice = _add_equipment(conn, kind, name)
                    conn.commit()
                changed = EquipmentMapper.save_matches(conn, {(kind, name): choice})
                item = LibraryScanner.load_equipment_info(conn).items(kind).get(choice)
        except Exception as e:
            logger.error(f"Error saving the match for {name}: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't save the match: {e}")
            changed = {}
        else:
            text = f"“{name}” is the {item['name']} now" if item else f"“{name}” isn't matched to your equipment now"
            self.status.setText(text + (f" - {EquipmentMapper.describe(changed)} updated." if any(changed.values())
                                        else "."))
        QTimer.singleShot(0, self.reload)  # Not from inside the combo's own signal
        self.changed.emit()
        if any(changed.values()):
            refresh_image_views()

    def _forget(self, entry):
        try:
            with self.db_manager.get_connection() as conn:
                EquipmentMapper.forget_match(conn, entry["kind"], entry["name"])
        except Exception as e:
            logger.error(f"Error forgetting the match for {entry['name']}: {e}", exc_info=True)
            return
        self.status.setText(f"Forgot the match for “{entry['name']}”.")
        QTimer.singleShot(0, self.reload)

    def _review(self):
        dialog = FixImagesDialog(self._suggestions, self._info, self)
        if dialog.exec() != QDialog.Accepted:
            return
        changed = Counter()
        try:
            with self.db_manager.get_connection() as conn:
                for suggestion, equipment_id in dialog.choices():
                    changed += EquipmentMapper.apply_suggestion(conn, suggestion, equipment_id)
        except Exception as e:
            logger.error(f"Error linking images and sessions to equipment: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Couldn't link them: {e}")
        self.status.setText(f"Linked {EquipmentMapper.describe(changed)} to your equipment." if changed
                            else "Nothing was linked.")
        self.reload()
        self.changed.emit()
        if changed:
            refresh_image_views()


class FixImagesDialog(QDialog):
    """Review EquipmentMapper.suggestions(): tick the ones to link, to the
    suggested equipment or another."""

    def __init__(self, suggestions, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Link to Your Equipment")
        self.resize(940, 420)
        self.suggestions = suggestions
        layout = QVBoxLayout(self)
        intro = QLabel("These gallery images and sessions could be linked to your equipment. Untick any that "
                       "aren't right, or pick other equipment. A name is remembered, so files given it later "
                       "are linked too.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(len(suggestions), 4)
        self.table.setHorizontalHeaderLabels(["What", "Used by", "Link to", "Why"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(2, 220)
        for row, suggestion in enumerate(suggestions):
            if suggestion["name"] is None:
                text = f"{info.telescopes[suggestion['telescope_id']]['name']} images with no camera"
            else:
                text = f"{suggestion['kind'].title()} named “{suggestion['name']}”"
            what = QTableWidgetItem(text)
            what.setToolTip(text)
            what.setFlags(what.flags() | Qt.ItemIsUserCheckable)
            what.setCheckState(Qt.Checked)
            self.table.setItem(row, 0, what)
            used = {table: len(ids) for table, ids in suggestion["ids"].items()}
            self.table.setItem(row, 1, QTableWidgetItem(EquipmentMapper.describe(used)))
            self.table.setCellWidget(row, 2, _equipment_combo(suggestion["kind"], info, suggestion["equipment_id"],
                                                              not_matched=None, add_new=False))
            self.table.setItem(row, 3, QTableWidgetItem(suggestion["reason"]))
        _sized_rows(self.table)
        self.table.setMaximumHeight(16777215)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        link = buttons.addButton("Link Ticked", QDialogButtonBox.AcceptRole)
        link.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choices(self):
        """[(suggestion, equipment id)] for the ticked rows."""
        chosen = []
        for row, suggestion in enumerate(self.suggestions):
            equipment_id = _combo_choice(self.table.cellWidget(row, 2))
            if self.table.item(row, 0).checkState() == Qt.Checked and isinstance(equipment_id, int):
                chosen.append((suggestion, equipment_id))
        return chosen


def choose_equipment(parent, telescope, camera, header=None, record="image"):
    """The telescope and camera to save on a gallery image or session, from
    the names typed or filled in: linked to the user's Equipment where they
    name it, or are matched to it. A name that isn't asks first - use the
    user's equipment it's like (saved as a match), add it to Equipment, or
    keep it as typed (remembered, so it isn't asked about again).
    telescope: None when there's none to save (a session's is picked from
    Equipment). header: the image file's, for a new item's details. record:
    'image' or 'session', for the question. {"telescope_id",
    "telescope_name", "camera_id", "camera_name"}, or None when the user cancels."""
    db_manager = DatabaseManager()
    try:
        with db_manager.get_connection() as conn:
            info = LibraryScanner.load_equipment_info(conn)
    except Exception as e:
        logger.error(f"Couldn't load the equipment: {e}", exc_info=True)
        info = LibraryScanner.EquipmentInfo()
    chosen, unlisted = {}, []  # unlisted: [(kind, name, the equipment it's like, or None)]
    for kind, name in ((TELESCOPE, telescope), (CAMERA, camera)):
        name = (name or "").strip()
        equipment_id = EquipmentMapper.resolve(kind, name, info)
        chosen[kind] = (equipment_id, name or None)
        if name and equipment_id is None and not info.decided(kind, name):
            unlisted.append((kind, name, EquipmentMapper.suggest(kind, name, info, header)[0]))
    if unlisted:
        similar = [(kind, name, like) for kind, name, like in unlisted if like is not None]
        lines = []
        for kind, name, like in unlisted:
            line = f"{kind.title()}: <b>{html.escape(name)}</b>"
            if like is not None:
                line += f" - like your <b>{html.escape(info.items(kind)[like]['name'])}</b>"
            lines.append(line)
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Not in Your Equipment")
        box.setText(f"{'This isn' if len(unlisted) == 1 else 'These aren'}'t in your Equipment yet:"
                    f"<br><br>{'<br>'.join(lines)}")
        box.setInformativeText(
            ("Use your own if it's the same - the name is remembered for next time. " if similar else "")
            + f"Add to your Equipment to link the {record} to it, and have it offered in the FOV Simulator, "
            f"sessions and the target list. Or keep the name on the {record} as typed.")
        use_button = None
        if similar:
            kind, _name, like = similar[0]
            use_button = box.addButton(f"Use {info.items(kind)[like]['name']}" if len(similar) == 1
                                       else "Use Yours", QMessageBox.AcceptRole)
        add_button = box.addButton("Add to Equipment", QMessageBox.ActionRole if similar else QMessageBox.AcceptRole)
        keep_button = box.addButton("Keep as Typed", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(use_button or add_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (use_button, add_button, keep_button) or clicked is None:
            return None
        try:
            with db_manager.get_connection() as conn:
                if clicked is add_button:
                    for kind, name, _like in unlisted:
                        chosen[kind] = (_add_equipment(conn, kind, name, header), name)
                    conn.commit()
                elif clicked is use_button:
                    # Matched to the user's own (any without one stay as typed) - other
                    # images and sessions given the name link too
                    EquipmentMapper.save_matches(conn, {(kind, name): like for kind, name, like in unlisted})
                    for kind, name, like in similar:
                        chosen[kind] = (like, name)
                else:
                    LibraryScanner.save_equipment_aliases(conn, {(kind, name): None for kind, name, _like in unlisted})
        except Exception as e:
            logger.error(f"Couldn't save the equipment: {e}", exc_info=True)
            QMessageBox.critical(parent, "Error", f"Couldn't save the equipment: {e}")
            return None
        if clicked is add_button:
            refresh_equipment_window()
    return {"telescope_id": chosen[TELESCOPE][0], "telescope_name": chosen[TELESCOPE][1],
            "camera_id": chosen[CAMERA][0], "camera_name": chosen[CAMERA][1]}


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class EquipmentDialog(WindowPositionMixin, QDialog):
    """Telescopes, cameras, eyepieces and barlows/reducers - add, edit, delete -
    and the names in files and images matched to them."""

    WINDOW_POSITION_KEY = "EquipmentDialog"
    TELESCOPES_TAB, CAMERAS_TAB, EYEPIECES_TAB, BARLOWS_TAB, MATCHES_TAB = range(5)

    def __init__(self, parent=None, start_tab=0):
        super().__init__(parent)
        self.setWindowTitle("Equipment - Cosmos Collection")
        # Its own window: the rest of the app stays usable while it's open
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(940, 640)
        self.db_manager = DatabaseManager()

        # Cameras whose subs record their pixel size get it
        try:
            with self.db_manager.get_connection() as conn:
                fill_pixel_sizes(conn)
        except Exception as e:
            logger.debug(f"Couldn't fill pixel sizes from subs: {e}")

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.pages = [TelescopesPage(self.db_manager), CamerasPage(self.db_manager),
                      EyepiecesPage(self.db_manager), BarlowsPage(self.db_manager), MatchesPage(self.db_manager)]
        for page, title in zip(self.pages, ("Telescopes", "Cameras", "Eyepieces", "Barlows && Reducers", "Matches")):
            self.tabs.addTab(page, title)  # "&&" shows as "&"
            page.changed.connect(lambda source=page: self._on_changed(source))
        layout.addWidget(self.tabs, 1)

        bottom = QHBoxLayout()
        note = QLabel("Changes are saved as you go.")
        themed_style(note, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        bottom.addWidget(note)
        bottom.addStretch()
        close_button = QPushButton("Close")
        close_button.setAutoDefault(False)
        close_button.clicked.connect(self.accept)
        bottom.addWidget(close_button)
        layout.addLayout(bottom)

        for page in self.pages:
            page.reload()
        self.tabs.setCurrentIndex(start_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setup_window_position()

    def _on_changed(self, source):
        """Saved on one tab: the others show the change."""
        for page in self.pages:
            if page is not source:
                page._dirty_from_other = True

    def _on_tab_changed(self, index):
        for page in self.pages:
            page.save_now()
        page = self.pages[index]
        if getattr(page, "_dirty_from_other", False):
            page._dirty_from_other = False
            page.reload()

    def done(self, result):
        for page in self.pages:
            page.save_now()
        super().done(result)

    def closeEvent(self, event):
        for page in self.pages:
            page.save_now()
        super().closeEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        # Back from another window (equipment may have changed - e.g. a session's telescope): stay current
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ActivationChange and self.isActiveWindow() and hasattr(self, "pages"):
            page = self.pages[self.tabs.currentIndex()]
            page.refresh_links()


_window = None


def open_equipment_window(parent=None, start_tab=0):
    """The Equipment window - one at a time; opening it again brings it to the front."""
    global _window
    try:
        if _window is not None and _window.isVisible():
            _window.tabs.setCurrentIndex(start_tab)
            _window.showNormal()
            _window.raise_()
            _window.activateWindow()
            return _window
    except RuntimeError:
        pass  # Closed and deleted
    _window = EquipmentDialog(parent, start_tab)
    _window.show()
    return _window
