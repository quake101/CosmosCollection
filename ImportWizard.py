#!/usr/bin/env python3
"""
Import Wizard
Brings a whole astrophotography library into the collection: scans the folders
the user picks for imaging sessions (subs) and finished images (LibraryScanner),
then shows everything found grouped by object, with its equipment, dates and
folders, so the user decides what is imported. Files are never moved or changed.

Pages: Choose folders -> Scanning -> Review.
"""

import os
import re
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from PySide6.QtCore import Qt, QEvent, QObject, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QProgressBar, QPushButton, QToolTip,
    QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import LibraryScanner
import MountDetection
from LibraryScanner import ScanRules, scan_and_build, ScanIndex
from Theme import COLORS, font_size, themed_style
from WindowPositionManager import WindowPositionMixin

logger = logging.getLogger(__name__)

SETTINGS_PREFIX = "import_wizard"
THUMBNAIL_SIZE = 480

# Tree columns
COL_NAME, COL_STATUS, COL_DATES, COL_NIGHTS, COL_SUBS, COL_INTEGRATION, COL_CAMERA, COL_TELESCOPE, \
    COL_FILTERS, COL_FOLDER = range(10)
COLUMN_TITLES = ["Object / item", "Status", "Dates", "Nights", "Subs", "Integration", "Camera",
                 "Telescope", "Filters", "Folder"]
COLUMN_WIDTHS = [300, 190, 170, 55, 70, 85, 150, 130, 130, 300]
ITEM_ROLE = Qt.UserRole + 1     # The Session / FinalImage / DSOGroup / ... a row shows

FILTER_ALL, FILTER_NEW, FILTER_NOT_IMPORTED, FILTER_UNKNOWN = range(4)
FINAL_TYPES = {1: "Finished image", 2: "Stack", 3: "Possible image"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _settings():
    return QSettings("CosmosCollection", "CosmosCollection")


def saved_folders():
    folders = _settings().value(f"{SETTINGS_PREFIX}/folders", [], type=list)
    return [f for f in folders if f]


def save_folders(folders):
    _settings().setValue(f"{SETTINGS_PREFIX}/folders", list(folders))


def _split_words(text):
    return tuple(w.strip() for w in re.split(r"[,;]", text or "") if w.strip())


def load_rules():
    """The scan rules, with the user's Advanced changes."""
    settings = _settings()
    rules = ScanRules()
    rules.session_gap_days = settings.value(f"{SETTINGS_PREFIX}/session_gap_days", rules.session_gap_days, type=int)
    finals = _split_words(settings.value(f"{SETTINGS_PREFIX}/final_keywords", "", type=str))
    work = _split_words(settings.value(f"{SETTINGS_PREFIX}/work_folders", "", type=str))
    if finals:
        rules.final_keywords = finals
    if work:
        rules.work_folders = work
    return rules


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_duration(seconds):
    from SessionManager import format_duration as session_format_duration
    return session_format_duration(seconds)


def format_night_range(nights):
    """'Sep 19, 2024', 'Sep 19 – Oct 4, 2024', 'Dec 28, 2024 – Jan 3, 2025'."""
    if not nights:
        return ""
    first, last = nights[0], nights[-1]
    if first == last:
        return f"{first:%b} {first.day}, {first.year}"
    if first.year == last.year:
        return f"{first:%b} {first.day} – {last:%b} {last.day}, {last.year}"
    return f"{first:%b} {first.day}, {first.year} – {last:%b} {last.day}, {last.year}"


def _count(n, word, bold=False):
    """'1 session', '3 sessions' - with the number in bold for rich text."""
    number = f"<b>{n:,}</b>" if bold else f"{n:,}"
    return f"{number} {word}{'' if n == 1 else 's'}"


def natural_key(text):
    """'M 2' before 'M 13'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text or "")]


def _common_folder(folders):
    try:
        return os.path.commonpath(folders) if folders else ""
    except ValueError:  # Different drives
        return folders[0]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class ScanWorker(QThread):
    """Runs LibraryScanner.scan_and_build off the UI thread."""
    progress = Signal(object)       # dict of counts
    status = Signal(str)
    scanned = Signal(object, object)    # ScanResult, Library
    failed = Signal(str)

    def __init__(self, roots, rules, reread=False):
        super().__init__()
        self.roots = roots
        self.rules = rules
        self.reread = reread
        self.cancel = threading.Event()

    def stop(self):
        self.cancel.set()

    def _progress(self, scan):
        self.progress.emit({
            "folders": scan.folders_listed, "folders_queued": scan.folders_queued,
            "files": scan.files_seen, "to_read": scan.headers_queued, "read": scan.headers_read,
            "known": scan.headers_cached, "new": scan.new_files, "seconds": scan.seconds,
        })

    def run(self):
        try:
            scan, library = scan_and_build(self.roots, self.rules, progress=self._progress, cancel=self.cancel,
                                           reread=self.reread, scan_index=ScanIndex(), status=self.status.emit)
        except Exception as e:
            logger.error(f"Library scan failed: {e}", exc_info=True)
            self.failed.emit(str(e))
            return
        self._save_known_mounts(library)
        self.scanned.emit(scan, library)

    def _save_known_mounts(self, library):
        """Sessions already in the Session Manager get the mounts just detected."""
        import sqlite3
        from ResourceManager import ResourceManager
        try:
            conn = sqlite3.connect(str(ResourceManager.get_database_path()), timeout=30)
            try:
                LibraryScanner.save_known_mounts(conn, library)
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Couldn't save the detected mounts: {e}", exc_info=True)


class ImportWorker(QThread):
    """Runs LibraryScanner.import_library off the UI thread, with its own connection."""
    progress = Signal(int, int, str)
    imported = Signal(object)       # ImportReport
    failed = Signal(str)

    def __init__(self, library):
        super().__init__()
        self.library = library
        self.cancel = threading.Event()

    def stop(self):
        self.cancel.set()

    def run(self):
        import sqlite3
        from ResourceManager import ResourceManager
        try:
            conn = sqlite3.connect(str(ResourceManager.get_database_path()))
            conn.row_factory = sqlite3.Row
            try:
                report = LibraryScanner.import_library(conn, self.library, progress=self.progress.emit,
                                                       cancelled=self.cancel.is_set)
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Library import failed: {e}", exc_info=True)
            self.failed.emit(str(e))
            return
        self.imported.emit(report)


class ThumbnailLoader(QThread):
    """Loads one image's preview as a QImage (safe off the UI thread)."""
    loaded = Signal(str, object)    # path, QImage or None

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        from ImageLoader import load_image_qimage
        try:
            image, _error = load_image_qimage(self.path, max_dim=THUMBNAIL_SIZE)
        except Exception as e:
            logger.debug(f"No preview for {self.path}: {e}")
            image = None
        self.loaded.emit(self.path, image)


# ---------------------------------------------------------------------------
# Page 1: choose folders
# ---------------------------------------------------------------------------

class FoldersPage(QWidget):
    scan_requested = Signal(list, bool)    # folders, read every file again

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Import your library")
        themed_style(title, lambda: f"font-size: {font_size(15)}; font-weight: bold;")
        layout.addWidget(title)
        intro = QLabel(
            "Choose the folders where you keep your astrophotography. Cosmos Collection looks through "
            "them for imaging sessions (your subs) and finished images, and shows you everything it "
            "finds before anything is imported. Your files stay where they are.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.folder_list = QTreeWidget()
        self.folder_list.setHeaderLabels(["Folder", "Last scanned"])
        self.folder_list.setRootIsDecorated(False)
        self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.folder_list.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.folder_list.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.folder_list.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.folder_list, 1)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add Folder...")
        add_button.clicked.connect(self._add_folder)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_folders)
        buttons.addWidget(add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        # Advanced: tucked away, most people never need it
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setStyleSheet("QToolButton { border: none; }")
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        form = QFormLayout(self.advanced)
        form.setContentsMargins(20, 0, 0, 0)
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(1, 3650)
        self.gap_spin.setSuffix(" days")
        self.gap_spin.setMaximumWidth(120)
        self.gap_spin.setToolTip("Nights of the same object with the same camera and telescope are one session "
                                 "unless they are further apart than this.")
        form.addRow("New session after a gap of:", self.gap_spin)
        self.final_words = QLineEdit()
        self.final_words.setToolTip("A file or folder name with one of these words is a finished image. "
                                    "Separate words with commas.")
        form.addRow("Words that mark a finished image:", self.final_words)
        self.work_folders = QLineEdit()
        self.work_folders.setToolTip("Folders with these names hold copies your stacking software made; "
                                     "they are noted but not looked through. Separate names with commas.")
        form.addRow("Stacking work folders to skip:", self.work_folders)
        reset = QPushButton("Reset to Defaults")
        reset.setAutoDefault(False)
        reset.clicked.connect(self._reset_advanced)
        self.reread_check = QCheckBox("Read every file again (slower - only needed if results look out of date)")
        form.addRow(self.reread_check)
        reset_row = QHBoxLayout()
        reset_row.addWidget(reset)
        reset_row.addStretch()
        form.addRow(reset_row)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        bottom = QHBoxLayout()
        self.hint = QLabel("")
        themed_style(self.hint, lambda: f"color: {COLORS['text_secondary']};")
        bottom.addWidget(self.hint, 1)
        self.scan_button = QPushButton("Scan")
        self.scan_button.setDefault(True)
        self.scan_button.clicked.connect(self._scan)
        bottom.addWidget(self.scan_button)
        layout.addLayout(bottom)

        self._load()

    def _load(self):
        rules = load_rules()
        self.gap_spin.setValue(rules.session_gap_days)
        self.final_words.setText(", ".join(rules.final_keywords))
        self.work_folders.setText(", ".join(rules.work_folders))
        for folder in saved_folders():
            self._add_row(folder)
        self._update_buttons()

    def refresh_scan_dates(self):
        index = ScanIndex()
        for row in range(self.folder_list.topLevelItemCount()):
            item = self.folder_list.topLevelItem(row)
            item.setText(1, _format_scan_date(index.last_scan(item.text(0))))

    def _add_row(self, folder):
        item = QTreeWidgetItem([folder, _format_scan_date(ScanIndex().last_scan(folder))])
        item.setToolTip(0, folder)
        self.folder_list.addTopLevelItem(item)

    def folders(self):
        return [self.folder_list.topLevelItem(i).text(0) for i in range(self.folder_list.topLevelItemCount())]

    def add_folders(self, paths):
        existing = {os.path.normcase(os.path.normpath(f)) for f in self.folders()}
        for path in paths:
            path = os.path.normpath(path)
            if os.path.isdir(path) and os.path.normcase(path) not in existing:
                self._add_row(path)
                existing.add(os.path.normcase(path))
        save_folders(self.folders())
        self._update_buttons()

    def _add_folder(self):
        start = self.folders()[-1] if self.folders() else ""
        folder = QFileDialog.getExistingDirectory(self, "Choose a Folder to Scan", start)
        if folder:
            self.add_folders([folder])

    def _remove_folders(self):
        for item in self.folder_list.selectedItems():
            self.folder_list.takeTopLevelItem(self.folder_list.indexOfTopLevelItem(item))
        save_folders(self.folders())
        self._update_buttons()

    def _update_buttons(self):
        self.remove_button.setEnabled(bool(self.folder_list.selectedItems()))
        has_folders = bool(self.folders())
        self.scan_button.setEnabled(has_folders)
        self.hint.setText("" if has_folders else "Add a folder to start - you can drop folders here too.")

    def _toggle_advanced(self, shown):
        self.advanced.setVisible(shown)
        self.advanced_toggle.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)

    def _reset_advanced(self):
        defaults = ScanRules()
        self.gap_spin.setValue(defaults.session_gap_days)
        self.final_words.setText(", ".join(defaults.final_keywords))
        self.work_folders.setText(", ".join(defaults.work_folders))
        self.reread_check.setChecked(False)

    def _save_advanced(self):
        settings = _settings()
        defaults = ScanRules()
        settings.setValue(f"{SETTINGS_PREFIX}/session_gap_days", self.gap_spin.value())
        finals, work = _split_words(self.final_words.text()), _split_words(self.work_folders.text())
        # Stored only when changed, so improved defaults reach everyone else
        settings.setValue(f"{SETTINGS_PREFIX}/final_keywords",
                          "" if finals == defaults.final_keywords else ", ".join(finals))
        settings.setValue(f"{SETTINGS_PREFIX}/work_folders",
                          "" if work == defaults.work_folders else ", ".join(work))

    def _scan(self):
        if not self.folders():
            return
        self._save_advanced()
        save_folders(self.folders())
        self.scan_requested.emit(self.folders(), self.reread_check.isChecked())


def _format_scan_date(value):
    if not value:
        return "Not scanned yet"
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    from TimeFormatHelper import format_time
    return f"{dt:%b} {dt.day}, {dt.year} {format_time(dt)}"


# ---------------------------------------------------------------------------
# Page 2: scanning
# ---------------------------------------------------------------------------

class ScanningPage(QWidget):
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        middle = QHBoxLayout()
        content = QWidget()
        content.setMaximumWidth(760)
        middle.addStretch()
        middle.addWidget(content, 1)
        middle.addStretch()
        outer.addLayout(middle, 1)
        layout = QVBoxLayout(content)
        layout.addStretch(1)
        self.title = QLabel("Scanning your folders")
        self.title.setAlignment(Qt.AlignCenter)
        themed_style(self.title, lambda: f"font-size: {font_size(15)}; font-weight: bold;")
        layout.addWidget(self.title)
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

        grid = QGridLayout()
        self.counts = {}
        for column, (key, label) in enumerate((("folders", "Folders"), ("files", "Files"),
                                                ("new", "New since last scan"), ("time", "Time"))):
            value = QLabel("0")
            value.setAlignment(Qt.AlignCenter)
            themed_style(value, lambda: f"font-size: {font_size(14)}; font-weight: bold;")
            caption = QLabel(label)
            caption.setAlignment(Qt.AlignCenter)
            themed_style(caption, lambda: f"color: {COLORS['text_secondary']};")
            grid.addWidget(value, 0, column)
            grid.addWidget(caption, 1, column)
            self.counts[key] = value
        layout.addLayout(grid)

        self.note = QLabel("The first scan of a large library, especially on a network drive, can take a while. "
                           "Later scans only read files that are new or changed.")
        self.note.setWordWrap(True)
        self.note.setAlignment(Qt.AlignCenter)
        themed_style(self.note, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.note)
        layout.addStretch(2)

        bottom = QHBoxLayout()
        bottom.addStretch()
        outer.addLayout(bottom)
        self.stop_button = QPushButton("Stop and Review")
        self.stop_button.setToolTip("Stop scanning and show what was found so far. What was read is kept, "
                                    "so scanning again carries on where this one stopped.")
        self.stop_button.clicked.connect(self._stop)
        bottom.addWidget(self.stop_button)

    def reset(self):
        self.status.setText("Starting...")
        self.bar.setRange(0, 0)
        self.stop_button.setEnabled(True)
        for value in self.counts.values():
            value.setText("0")

    def _stop(self):
        self.stop_button.setEnabled(False)
        self.status.setText("Stopping...")
        self.stop_requested.emit()

    def set_status(self, text):
        if self.stop_button.isEnabled():
            self.status.setText(text)
        if not text.startswith("Looking"):
            self.bar.setRange(0, 0)

    def set_progress(self, counts):
        self.counts["folders"].setText(f"{counts['folders']:,}")
        self.counts["files"].setText(f"{counts['files']:,}")
        self.counts["new"].setText(f"{counts['new']:,}")
        seconds = int(counts["seconds"])
        self.counts["time"].setText(f"{seconds // 60}:{seconds % 60:02d}")
        if not self.stop_button.isEnabled():
            return
        walking = counts["folders"] < counts["folders_queued"]
        if counts["to_read"]:
            self.bar.setRange(0, counts["to_read"])
            self.bar.setValue(counts["read"])
            more = " (still finding files)" if walking else ""
            self.status.setText(f"Reading file details: {counts['read']:,} of {counts['to_read']:,}{more}")
        else:
            self.bar.setRange(0, 0)
            self.status.setText("Looking through your folders...")


# ---------------------------------------------------------------------------
# Page 3: equipment
# ---------------------------------------------------------------------------

# Combo item data (strings: Qt can't compare Python tuples in findData)
NOT_MATCHED, ADD_NEW = "none", "new"


def _matched(item_id):
    return f"id:{item_id}"


class EquipmentPage(QWidget):
    """Match the telescope and camera names in the files (TELESCOP / INSTRUME)
    to the user's Equipment, so sessions are grouped and filed by what they
    were really taken with. Suggestions come ready-picked; choices are kept."""
    back_requested = Signal()
    # {(kind, name in files): equipment id or None}, [(kind, name, header)] to add - "object", as Qt would
    # turn a dict into a map with string keys
    finished = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        layout = QVBoxLayout(self)
        title = QLabel("Match your equipment")
        themed_style(title, lambda: f"font-size: {font_size(15)}; font-weight: bold;")
        layout.addWidget(title)
        intro = QLabel("Your files name the telescope and camera they were taken with. Match each name to your "
                       "equipment, so sessions taken with the same gear stay together and are filed under it. "
                       "Likely matches are already picked, and your choices are remembered for next time.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.tables = {}
        for kind, heading in ((LibraryScanner.TELESCOPE, "Telescopes"), (LibraryScanner.CAMERA, "Cameras")):
            label = QLabel(heading)
            themed_style(label, lambda: f"font-weight: bold; font-size: {font_size(11)};")
            layout.addWidget(label)
            table = QTableWidget(0, 5)
            table.setHorizontalHeaderLabels(["Name in your files", "Subs", "Nights", "Matched to", "Why"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.NoSelection)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
            table.horizontalHeader().setStretchLastSection(True)
            table.setColumnWidth(3, 300)
            layout.addWidget(table, 1)
            self.tables[kind] = (label, table)
        layout.addStretch()
        bottom = QHBoxLayout()
        back = QPushButton("Back")
        back.setAutoDefault(False)
        back.clicked.connect(self.back_requested.emit)
        bottom.addWidget(back)
        note = QLabel("You can change these any time from the review's Equipment button.")
        themed_style(note, lambda: f"color: {COLORS['text_secondary']};")
        bottom.addWidget(note, 1)
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self._finish)
        bottom.addWidget(self.next_button)
        layout.addLayout(bottom)

    def show_equipment(self, rows, info, revisiting=False):
        """rows: LibraryScanner.equipment_in_files(); info: EquipmentInfo."""
        self.rows = rows
        self.next_button.setText("Done" if revisiting else "Next")
        for kind, (label, table) in self.tables.items():
            kind_rows = [r for r in rows if r["kind"] == kind]
            label.setVisible(bool(kind_rows))
            table.setVisible(bool(kind_rows))
            table.setRowCount(len(kind_rows))
            for row, entry in enumerate(kind_rows):
                self._fill_row(table, row, entry, info)
            # Even rows, and no taller than they need (the space left goes below)
            row_height = max(QComboBox().sizeHint().height() + 6, 26)
            table.verticalHeader().setDefaultSectionSize(row_height)
            for row in range(table.rowCount()):
                table.setRowHeight(row, row_height)
            table.setMaximumHeight(table.horizontalHeader().sizeHint().height() + row_height * len(kind_rows)
                                   + 2 * table.frameWidth() + 2)

    def _fill_row(self, table, row, entry, info):
        kind, value = entry["kind"], entry["value"]
        name_item = QTableWidgetItem(value)
        name_item.setData(Qt.UserRole, entry)
        table.setItem(row, 0, name_item)
        for column, text in ((1, f"{entry['subs']:,}"),
                             (2, format_night_range([entry["first"], entry["last"]]) if entry["first"] else "")):
            cell = QTableWidgetItem(text)
            cell.setTextAlignment((Qt.AlignRight if column == 1 else Qt.AlignLeft) | Qt.AlignVCenter)
            table.setItem(row, column, cell)

        combo = QComboBox()
        combo.addItem("Not matched - keep the name from the files", NOT_MATCHED)
        items = info.items(kind)
        for item_id in sorted(items, key=lambda i: (not items[i]["is_active"], (items[i]["name"] or "").lower())):
            item = items[item_id]
            combo.addItem(item["name"] + ("" if item["is_active"] else " (disabled)"), _matched(item_id))
        measured = LibraryScanner.equipment_from_header(kind, entry["header"])
        if kind == LibraryScanner.TELESCOPE:
            detail = f" ({measured['focal_length']:g} mm)" if measured["focal_length"] else ""
            combo.addItem(f"Add as a new telescope{detail}", ADD_NEW)
        else:
            detail = f" ({measured['pixel_size']:g} µm pixels)" if measured["pixel_size"] else ""
            combo.addItem(f"Add as a new camera{detail}", ADD_NEW)

        if info.decided(kind, value):
            chosen = info.aliases[(kind, LibraryScanner._alias_key(value))]
            why = "Chosen before"
        else:
            chosen, reason = LibraryScanner.suggest_equipment(kind, value, entry["header"], info)
            why = f"Suggested: {reason}" if chosen is not None else "No match found"
        index = combo.findData(_matched(chosen)) if chosen is not None else 0
        combo.setCurrentIndex(max(index, 0))
        why_item = QTableWidgetItem(why)
        table.setItem(row, 4, why_item)
        combo.currentIndexChanged.connect(lambda _i, item=why_item: item.setText("Your choice"))
        table.setCellWidget(row, 3, combo)

    def _finish(self):
        decisions, new = {}, []
        for kind, (_label, table) in self.tables.items():
            for row in range(table.rowCount()):
                entry = table.item(row, 0).data(Qt.UserRole)
                choice = table.cellWidget(row, 3).currentData()
                if choice == ADD_NEW:
                    new.append((kind, entry["value"], entry["header"]))
                else:
                    decisions[(kind, entry["value"])] = int(choice[3:]) if choice.startswith("id:") else None
        self.finished.emit(decisions, new)


# ---------------------------------------------------------------------------
# Page 4: review
# ---------------------------------------------------------------------------

class ReviewPage(QWidget):
    back_requested = Signal()
    rescan_requested = Signal()
    equipment_requested = Signal()
    import_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scan = None
        self.library = None
        self._expanded = set()
        self._thumbnail_loader = None
        self._pending_thumbnail = None
        self._shown_path = None
        self._viewers = []
        self._layout_edited = False     # Sessions merged/split or images re-paired since the scan
        self._identifier = None
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(400)
        self._rebuild_timer.timeout.connect(self._rebuild)

        layout = QVBoxLayout(self)
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        themed_style(self.banner, lambda: f"background: {COLORS['background_light']}; "
                                          f"border: 1px solid {COLORS['warning']}; border-radius: 4px; padding: 6px;")
        self.banner.setVisible(False)
        layout.addWidget(self.banner)

        top = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        top.addWidget(self.summary, 1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search objects, files, folders, equipment")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(240)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(250)
        self._search_timer.timeout.connect(self._rebuild)
        self.search.textChanged.connect(lambda _text: self._search_timer.start())
        top.addWidget(self.search)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Everything", "New since the last scan", "Not yet in your collection",
                                    "Needs an object"])
        self.filter_combo.currentIndexChanged.connect(self._rebuild)
        top.addWidget(self.filter_combo)
        layout.addLayout(top)

        # Finding objects by plate solving: offered when some files have none
        self.identify_bar = QFrame()
        identify_layout = QHBoxLayout(self.identify_bar)
        identify_layout.setContentsMargins(8, 4, 8, 4)
        themed_style(self.identify_bar, lambda: f"QFrame {{ background: {COLORS['background_light']}; "
                                                f"border-radius: 4px; }}")
        self.identify_label = QLabel("")
        self.identify_label.setWordWrap(True)
        identify_layout.addWidget(self.identify_label, 1)
        self.identify_button = QPushButton("Identify by Plate Solving")
        self.identify_button.setAutoDefault(False)
        self.identify_button.setToolTip("Plate solve the images (and subs with no position) that nothing named, "
                                        "to find the objects they show")
        self.identify_button.clicked.connect(self.identify_unknown)
        identify_layout.addWidget(self.identify_button)
        self.identify_stop = QPushButton("Stop")
        self.identify_stop.setAutoDefault(False)
        self.identify_stop.clicked.connect(self._stop_identifying)
        identify_layout.addWidget(self.identify_stop)
        self.identify_bar.setVisible(False)
        layout.addWidget(self.identify_bar)

        splitter = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(COLUMN_TITLES)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setStretchLastSection(True)
        for column, width in enumerate(COLUMN_WIDTHS):
            self.tree.setColumnWidth(column, width)
        state = _settings().value(f"{SETTINGS_PREFIX}/tree_header")
        if state is not None:
            self.tree.header().restoreState(state)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.viewport().installEventFilter(self)   # Check box help, worked out when hovered
        self.tree.currentItemChanged.connect(self._on_current_changed)
        self.tree.itemExpanded.connect(lambda item: self._remember_expanded(item, True))
        self.tree.itemCollapsed.connect(lambda item: self._remember_expanded(item, False))
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_menu)
        splitter.addWidget(self.tree)
        self.details = DetailsPane()
        self.details.view_requested.connect(self.open_image)
        self.details.change_requested.connect(lambda: self.change_object(*self._selection()))
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 340])
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        back = QPushButton("Back")
        back.setAutoDefault(False)
        back.setToolTip("Change the folders to scan")
        back.clicked.connect(self.back_requested.emit)
        bottom.addWidget(back)
        rescan = QPushButton("Scan Again")
        rescan.setAutoDefault(False)
        rescan.setToolTip("Look through the folders again for anything new or changed")
        rescan.clicked.connect(self._request_rescan)
        bottom.addWidget(rescan)
        equipment = QPushButton("Equipment...")
        equipment.setAutoDefault(False)
        equipment.setToolTip("Match the telescope and camera names in your files to your equipment")
        equipment.clicked.connect(self._request_equipment)
        bottom.addWidget(equipment)
        bottom.addStretch()
        self.selection_label = QLabel("")
        bottom.addWidget(self.selection_label)
        self.import_button = QPushButton("Import Selected")
        self.import_button.setDefault(True)
        self.import_button.setEnabled(False)
        self.import_button.setToolTip("Add the checked sessions to the Session Manager and the checked images "
                                      "to the gallery. Files stay where they are.")
        self.import_button.clicked.connect(self.import_requested.emit)
        bottom.addWidget(self.import_button)
        layout.addLayout(bottom)

    # ---- Data ----

    def show_library(self, scan, library):
        self._stop_identifying()
        self.scan, self.library = scan, library
        self._layout_edited = False
        stopped = scan.cancelled
        self.banner.setVisible(stopped)
        self.banner.setText("The scan was stopped before it finished, so this is only part of your library. "
                            "Scan Again carries on where it stopped." if stopped else "")
        self._rebuild()

    def _rel(self, path):
        root = LibraryScanner._root_of(path, self.scan.roots)
        rel = os.path.relpath(path, root)
        return os.path.join(os.path.basename(root), rel) if rel != "." else os.path.basename(root) or root

    def _matches(self, texts):
        needle = self.search.text().strip().lower()
        return not needle or any(needle in (t or "").lower() for t in texts)

    # ---- Tree ----

    def _rebuild(self):
        if not self.library:
            return
        mode = self.filter_combo.currentIndex()
        current = self.tree.currentItem()
        current_object = current.data(COL_NAME, ITEM_ROLE) if current else None
        self.tree.blockSignals(True)
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()

        groups = list(self.library.groups.values())
        known = sorted((g for g in groups if g.dsodetailid), key=lambda g: natural_key(g.name))
        unknown = sorted((g for g in groups if not g.dsodetailid), key=lambda g: natural_key(g.name))
        if mode != FILTER_UNKNOWN:
            for group in known:
                self._add_group(self.tree.invisibleRootItem(), group, mode)
        unknown_items = []
        if unknown:
            header = QTreeWidgetItem(["Needs an object"])
            header.setData(COL_NAME, ITEM_ROLE, ("unknown_header", None))
            header.setToolTip(COL_NAME, "Cosmos Collection couldn't tell which object these are")
            self._bold(header)
            for group in unknown:
                item = self._add_group(header, group, mode, importable=False)
                if item:
                    unknown_items.append(item)
            if unknown_items:
                header.setText(COL_STATUS, _count(len(unknown_items), "group"))
                self.tree.addTopLevelItem(header)
                self._restore_expanded(header, "unknown_header")
        if mode == FILTER_ALL:
            self._add_calibration_libraries()
            self._add_not_imported()

        self.tree.setUpdatesEnabled(True)
        self.tree.blockSignals(False)
        if not self._reselect(current_object):
            self.details.clear()
        self._update_summary()
        self._update_identify_bar()

    def _bold(self, item):
        font = item.font(COL_NAME)
        font.setBold(True)
        item.setFont(COL_NAME, font)

    def _remember_expanded(self, item, expanded):
        key = self._expand_key(item)
        if key:
            (self._expanded.add if expanded else self._expanded.discard)(key)

    def _expand_key(self, item):
        data = item.data(COL_NAME, ITEM_ROLE)
        if not data:
            return None
        kind, obj = data
        if kind == "group":
            return ("group", obj.key)
        if kind == "more":
            return ("more", obj.key)
        return (kind, None) if obj is None else None

    def _restore_expanded(self, item, kind, key=None):
        if (kind, key) in self._expanded:
            item.setExpanded(True)

    def _add_group(self, parent, group, mode, importable=True):
        sessions = [s for s in group.sessions if self._session_visible(s, group, mode)]
        finals = [f for f in group.finals if self._final_visible(f, group, mode)]
        if not sessions and not finals:
            return None
        item = QTreeWidgetItem([group.name])
        item.setData(COL_NAME, ITEM_ROLE, ("group", group))
        self._bold(item)
        parts = []
        if group.sessions:
            parts.append(_count(len(group.sessions), "session"))
        if group.finals:
            parts.append(_count(len(group.finals), "image"))
        item.setText(COL_STATUS, " · ".join(parts))
        if not importable:
            item.setToolTip(COL_NAME, "Choose an object for these first (right-click > Change Object...), "
                                      "then they can be imported")
        if group.mismatches:
            item.setToolTip(COL_NAME, "Some files are named for this object but were pointed elsewhere - "
                                      "check them in the details.")
        if importable:
            # Its check box shows (and sets) its children's: set before they're added
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item.setCheckState(COL_NAME, Qt.Unchecked)
        parent.addChild(item)

        for session in sessions:
            self._add_session(item, session, importable)
        shown = [f for f in finals if f.record.tier == 1 or f.checked]
        more = [f for f in finals if f not in shown]
        for final in shown:
            self._add_final(item, final, importable)
        if more:
            more_item = QTreeWidgetItem([f"More images ({len(more)})"])
            more_item.setData(COL_NAME, ITEM_ROLE, ("more", group))
            more_item.setText(COL_STATUS, "Stacks, by-products and other images")
            if importable:
                more_item.setFlags(more_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                more_item.setCheckState(COL_NAME, Qt.Unchecked)
            item.addChild(more_item)
            for final in more:
                self._add_final(more_item, final, importable)
            self._restore_expanded(more_item, "more", group.key)
        self._restore_expanded(item, "group", group.key)
        return item

    def _session_visible(self, session, group, mode):
        if mode == FILTER_NEW and not session.new_lights:
            return False
        if mode == FILTER_NOT_IMPORTED and session.imported >= len(session.lights):
            return False
        return self._matches([group.name, session.camera, session.telescope, *session.filters,
                              *(self._rel(f) for f in session.folders)])

    def _final_visible(self, final, group, mode):
        if mode == FILTER_NEW and not final.is_new:
            return False
        if mode == FILTER_NOT_IMPORTED and final.imported:
            return False
        return self._matches([group.name, final.record.name, self._rel(final.record.folder)])

    def _add_session(self, parent, session, importable):
        nights = session.nights
        item = QTreeWidgetItem()
        item.setData(COL_NAME, ITEM_ROLE, ("session", session))
        item.setText(COL_NAME, "Session")
        status = {"New": "New", "Already imported": "Already in Session Manager"}.get(
            session.status, f"Partly imported ({session.imported:,} of {len(session.lights):,})")
        if session.new_lights and session.imported:
            status += f" · {session.new_lights:,} new"
        item.setText(COL_STATUS, status)
        item.setText(COL_DATES, format_night_range(nights))
        item.setText(COL_NIGHTS, str(len(nights)))
        item.setText(COL_SUBS, f"{len(session.lights):,}")
        item.setText(COL_INTEGRATION, format_duration(session.integration_seconds))
        item.setText(COL_CAMERA, session.camera or "Unknown")
        item.setText(COL_TELESCOPE, session.telescope or "Unknown")
        item.setText(COL_FILTERS, ", ".join(f for f in session.filters if f != "(none)") or "None")
        folder = self._rel(_common_folder(session.folders))
        item.setText(COL_FOLDER, folder)
        item.setToolTip(COL_FOLDER, "\n".join(session.folders[:20]))
        for column in (COL_NIGHTS, COL_SUBS, COL_INTEGRATION):
            item.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)
        self._set_check(item, session.checked, importable,
                        "Already in the Session Manager" if session.status == "Already imported" else None)
        parent.addChild(item)

    def _add_final(self, parent, final, importable):
        record = final.record
        item = QTreeWidgetItem()
        item.setData(COL_NAME, ITEM_ROLE, ("final", final))
        item.setText(COL_NAME, record.name)
        item.setToolTip(COL_NAME, f"{record.path}\nDouble-click to open it in the Image Viewer")
        kind = FINAL_TYPES.get(record.tier, "Image")
        item.setText(COL_STATUS, "In your gallery" if final.imported else kind + (" · new" if final.is_new else ""))
        item.setText(COL_DATES, f"{final.date:%b} {final.date.day}, {final.date.year}")
        if final.session:
            item.setText(COL_CAMERA, final.session.camera or "")
            item.setText(COL_TELESCOPE, final.session.telescope or "")
        item.setText(COL_FOLDER, self._rel(record.folder))
        item.setToolTip(COL_FOLDER, record.folder)
        self._set_check(item, final.checked, importable, "Already in your gallery" if final.imported else None)
        parent.addChild(item)

    def _set_check(self, item, checked, importable, disabled_reason):
        if not importable:
            item.setToolTip(COL_NAME, "Choose an object for these first (right-click > Change Object...), "
                                      "then they can be imported")
            return
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(COL_NAME, Qt.Checked if checked else Qt.Unchecked)
        if disabled_reason:
            item.setToolTip(COL_STATUS, disabled_reason)

    def _add_calibration_libraries(self):
        libraries = [c for c in self.library.calibration_libraries
                     if self._matches([c.folder, c.camera, c.frame_type])]
        if not libraries:
            return
        header = QTreeWidgetItem([f"Calibration libraries ({len(libraries)})"])
        header.setData(COL_NAME, ITEM_ROLE, ("calibration_header", None))
        header.setText(COL_STATUS, "Darks, flats and bias frames shared by several objects")
        self._bold(header)
        for library in libraries:
            exposures = sorted({e for e in (LibraryScanner.exposure_of(r.header) for r in library.files) if e})
            item = QTreeWidgetItem([f"{library.frame_type} frames"])
            item.setData(COL_NAME, ITEM_ROLE, ("calibration", library))
            item.setText(COL_STATUS, ", ".join(f"{e:g}s" for e in exposures[:4]) + ("..." if len(exposures) > 4 else ""))
            item.setText(COL_SUBS, f"{len(library.files):,}")
            item.setTextAlignment(COL_SUBS, Qt.AlignRight | Qt.AlignVCenter)
            item.setText(COL_CAMERA, library.camera or "Unknown")
            item.setText(COL_FOLDER, self._rel(library.folder))
            item.setToolTip(COL_FOLDER, library.folder)
            header.addChild(item)
        self.tree.addTopLevelItem(header)
        self._restore_expanded(header, "calibration_header")

    def _add_not_imported(self):
        """What the scan found but leaves out - a note, nothing to check."""
        library, scan = self.library, self.scan
        rows = []
        processing = [p for p, r in scan.skipped_folders if r == "processing"]
        rejected = [p for p, r in scan.skipped_folders if r == "rejected"]
        if processing:
            rows.append((_count(len(processing), "stacking work folder"),
                         "Copies your stacking software made - not looked through", processing))
        if rejected:
            rows.append((_count(len(rejected), "Rejected folder"), "Subs the light quality review set aside", rejected))
        if library.counts.get("duplicate"):
            rows.append((f"{library.counts['duplicate']:,} copies of subs", "The same subs in another folder", []))
        if library.counts.get("preview"):
            rows.append((f"{library.counts['preview']:,} previews and thumbnails",
                         "Pictures your capture software saved beside the subs", []))
        if library.counts.get("processing"):
            rows.append((f"{library.counts['processing']:,} processing files",
                         "Calibration masters and intermediate files", []))
        if library.raw_folders:
            count = sum(library.raw_folders.values())
            rows.append((f"{count:,} camera raw files", "DSLR raw files (CR2, NEF...) can't be read yet",
                         sorted(library.raw_folders)))
        if library.unreadable:
            rows.append((f"{len(library.unreadable):,} unreadable files", "Their headers couldn't be read",
                         [r.path for r in library.unreadable]))
        if not rows:
            return
        header = QTreeWidgetItem(["Not imported"])
        header.setData(COL_NAME, ITEM_ROLE, ("skipped_header", None))
        header.setText(COL_STATUS, "Found, but left out on purpose")
        self._bold(header)
        for title, reason, paths in rows:
            item = QTreeWidgetItem([title])
            item.setData(COL_NAME, ITEM_ROLE, ("skipped", (title, reason, paths)))
            item.setText(COL_STATUS, reason)
            header.addChild(item)
        self.tree.addTopLevelItem(header)
        self._restore_expanded(header, "skipped_header")

    def _reselect(self, obj):
        """Select the row showing obj again after a rebuild; False if it's gone."""
        if obj is None:
            return False
        iterator = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while iterator:
            item = iterator.pop()
            data = item.data(COL_NAME, ITEM_ROLE)
            if data and data[0] == obj[0] and data[1] is obj[1]:
                self.tree.setCurrentItem(item)
                return True
            iterator.extend(item.child(i) for i in range(item.childCount()))
        return False

    def save_state(self):
        _settings().setValue(f"{SETTINGS_PREFIX}/tree_header", self.tree.header().saveState())

    # ---- Checks ----

    def _on_item_changed(self, item, column):
        if column != COL_NAME:
            return
        data = item.data(COL_NAME, ITEM_ROLE)
        if not data or data[0] not in ("session", "final"):
            return
        data[1].checked = item.checkState(COL_NAME) == Qt.Checked
        self._update_selection_label()

    def eventFilter(self, obj, event):
        if obj is self.tree.viewport() and event.type() == QEvent.ToolTip:
            item = self.tree.itemAt(event.pos())
            if item and self.tree.columnAt(event.pos().x()) == COL_NAME:
                help_text = self._check_help(item)
                if help_text:
                    extra = item.toolTip(COL_NAME)
                    QToolTip.showText(event.globalPos(), help_text + (f"\n\n{extra}" if extra else ""),
                                      self.tree.viewport(), self.tree.visualItemRect(item))
                    return True
        return super().eventFilter(obj, event)

    def _check_help(self, item):
        """What a row's check box means right now: what will be imported, and
        why something is left unchecked. None for rows without a check box."""
        data = item.data(COL_NAME, ITEM_ROLE)
        # Every row is user-checkable by default in Qt: only the ones given a check state show a box
        if not data or item.data(COL_NAME, Qt.CheckStateRole) is None:
            return None
        kind, obj = data
        if kind == "session":
            return self._session_help(obj)
        if kind == "final":
            return self._final_help(obj)
        if kind in ("group", "more"):
            return self._group_help(item, kind)
        return None

    def _group_help(self, item, kind):
        """Counts of the rows under an object (or its "More images") - what its box sets."""
        sessions, finals = [], []
        below = [item.child(i) for i in range(item.childCount())]
        while below:
            child = below.pop()
            data = child.data(COL_NAME, ITEM_ROLE)
            if data and data[0] == "session":
                sessions.append(data[1])
            elif data and data[0] == "final":
                finals.append(data[1])
            below.extend(child.child(i) for i in range(child.childCount()))
        new_sessions = [s for s in sessions if s.imported < len(s.lights)]
        new_finals = [f for f in finals if not f.imported]
        parts = []
        if new_sessions:
            parts.append(f"{sum(s.checked for s in new_sessions):,} of {_count(len(new_sessions), 'session')}")
        if new_finals:
            parts.append(f"{sum(f.checked for f in new_finals):,} of {_count(len(new_finals), 'image')}")
        if not parts:
            lines = ["Everything here is already in your collection."]
        elif not any(s.checked for s in new_sessions) and not any(f.checked for f in new_finals):
            lines = ["Nothing here will be imported."]
        else:
            lines = [f"{' and '.join(parts)} will be imported."]
        if kind == "more":
            lines.append("Stacks, by-products and other images that aren't checked on their own.")
        lines.append("Click the box to check or uncheck everything under it.")
        if self.search.text().strip() or self.filter_combo.currentIndex() != FILTER_ALL:
            lines.append("Rows hidden by the search or filter keep their own check boxes.")
        return "\n".join(lines)

    def _session_help(self, session):
        missing = len(session.lights) - session.imported
        if session.status == "Already imported":
            return ("All its subs are already in the Session Manager, so nothing will be added."
                    if session.checked else "Unchecked: all its subs are already in the Session Manager.")
        if session.status == "Partly imported":
            missing, imported = _count(missing, "sub"), _count(session.imported, "sub")
            if not session.checked:
                return (f"Unchecked: the {missing} not yet in the Session Manager won't be added.\n"
                        f"Check it to add them to the session that has the other {imported}.")
            return (f"Checked: the {missing} not yet in the Session Manager will be added to the session "
                    f"that has the other {imported}. That session keeps its status.")
        what = _count(len(session.lights), "sub")
        if session.calibration:
            what += f" and {_count(len(session.calibration), 'calibration frame')}"
        if not session.checked:
            return f"Unchecked: it won't be imported.\nCheck it to add its {what} to the Session Manager."
        all_finals = [f for g in self.library.groups.values() for f in g.finals]
        today = datetime.now().date()
        gap = self.library.session_gap_days
        status = LibraryScanner._status(session, all_finals, gap, today)
        nights = session.nights
        if status == "In Progress":
            why = f"its last night was in the last {gap} days, so you may still be imaging it"
        elif nights and nights[-1] < today - timedelta(days=gap):
            why = f"its last night was more than {gap} days ago"
        else:
            why = "an image made from it is checked or already in your gallery"
        return (f"Checked: its {what} will be added to the Session Manager as a new session.\n"
                f"It will be marked {status}: {why}.")

    def _final_help(self, final):
        group = self.library.groups.get(final.dso_key)
        gallery = f"the {group.name} gallery" if group else "the gallery"
        if final.imported:
            return f"Already in {gallery}, so it won't be added again."
        if final.checked:
            text = f"Checked: it will be added to {gallery}"
            if final.session:
                text += f", linked to its session ({format_night_range(final.session.nights)})"
            return text + "."
        reason = self._final_reason(final, group)
        return f"Unchecked: it won't be added to {gallery}." + \
            (f"\nNot checked by default: {reason}." if reason else "")

    def _final_reason(self, final, group):
        """Why the defaults leave a final image unchecked (LibraryScanner._choose_default_finals),
        or None when they'd have checked it."""
        record = final.record
        if final.variant_words:
            return f"\"{', '.join(sorted(final.variant_words))}\" in its name marks a by-product of processing"
        if record.tier == 3:
            if record.reason == "software's default output name":
                return "its name is one your processing software gives by default, so it's likely a step along the way"
            return "nothing in its name or folder marks it as a finished image"
        finals = group.finals if group else []
        if record.tier == 2:
            if any(f.record.tier == 1 and (f.checked or f.imported) for f in finals):
                return "it's a stack, and this object has a finished image"
            if any(f.record.tier <= 2 and f.imported for f in finals):
                return "a stack of this object is already in your gallery"
            if any(f.record.tier == 2 and not f.variant_words and f.record.mtime_ns > record.mtime_ns
                   for f in finals):
                return "it's a stack, and a newer one is offered instead"
            return None
        family = LibraryScanner._family_key(final)
        versions = [f for f in finals if f is not final and LibraryScanner._family_key(f) == family]
        if any(f.imported for f in versions):
            return "another version of it is already in your gallery"
        newer = [f for f in versions if f.record.mtime_ns > record.mtime_ns]
        if newer:
            return f"it's an older version of {max(newer, key=lambda f: f.record.mtime_ns).record.name}"
        return None

    def _update_summary(self):
        library = self.library
        groups = [g for g in library.groups.values() if g.dsodetailid]
        sessions = sum(len(g.sessions) for g in groups)
        finals = sum(len(g.finals) for g in groups)
        unknown = sum(1 for g in library.groups.values() if not g.dsodetailid)
        text = (f"Found {_count(len(groups), 'object', bold=True)} with {_count(sessions, 'session', bold=True)} "
                f"and {_count(finals, 'image', bold=True)}")
        if unknown:
            text += f" · {_count(unknown, 'group')} {'needs' if unknown == 1 else 'need'} an object"
        if self.scan.new_files and self.scan.headers_cached:
            text += f" · {self.scan.new_files:,} files new since the last scan"
        self.summary.setText(text)
        self._update_selection_label()

    def _update_selection_label(self):
        groups = [g for g in self.library.groups.values() if g.dsodetailid]
        sessions = sum(s.checked for g in groups for s in g.sessions)
        finals = sum(f.checked for g in groups for f in g.finals)
        self.import_button.setEnabled(bool(sessions or finals) and self._identifier is None)
        self.selection_label.setText(f"Selected: {sessions:,} sessions, {finals:,} images")

    # ---- Details ----

    def _on_current_changed(self, current, _previous):
        data = current.data(COL_NAME, ITEM_ROLE) if current else None
        if not data:
            self.details.clear()
            return
        kind, obj = data
        self.details.set_viewable(obj if kind == "final" else None)
        self.details.set_changeable(kind in ("group", "session", "final"))
        if kind == "group":
            self.details.show_group(obj, self._rel)
        elif kind == "session":
            self.details.show_session(obj, self._rel)
        elif kind == "final":
            self.details.show_final(obj, self._rel)
            self._load_thumbnail(obj.record.path)
        elif kind == "calibration":
            self.details.show_calibration(obj, self._rel)
        elif kind == "skipped":
            self.details.show_skipped(*obj, self._rel)
        else:
            self.details.clear()

    # ---- Editing ----

    def _selection(self):
        """(sessions, finals, groups) selected - a group or "More images" row
        brings everything in it."""
        sessions, finals, groups = [], [], []

        def add(seq, obj):
            if not any(o is obj for o in seq):
                seq.append(obj)
        for item in self.tree.selectedItems():
            data = item.data(COL_NAME, ITEM_ROLE)
            if not data:
                continue
            kind, obj = data
            if kind == "session":
                add(sessions, obj)
            elif kind == "final":
                add(finals, obj)
            elif kind == "group":
                add(groups, obj)
                for session in obj.sessions:
                    add(sessions, session)
                for final in obj.finals:
                    add(finals, final)
            elif kind == "more":
                for i in range(item.childCount()):
                    child = item.child(i).data(COL_NAME, ITEM_ROLE)
                    if child and child[0] == "final":
                        add(finals, child[1])
        return sessions, finals, groups

    def _show_menu(self, position):
        if not self.library or not self.tree.itemAt(position):
            return
        sessions, finals, groups = self._selection()
        menu = QMenu(self)
        if sessions or finals:
            menu.addAction("Change Object...", lambda: self.change_object(sessions, finals, groups))
        same_group = len({s.dso_key for s in sessions}) == 1
        if len(sessions) > 1 and same_group and not finals and not groups:
            menu.addAction("Merge Sessions", lambda: self.merge(sessions))
        if len(sessions) == 1 and not finals and not groups and len(sessions[0].nights) > 1:
            menu.addAction("Split Session...", lambda: self.split(sessions[0]))
        if finals and not sessions and len({f.dso_key for f in finals}) == 1:
            group = self.library.groups.get(finals[0].dso_key)
            if group is not None and group.sessions:
                pair_menu = menu.addMenu("From Session")
                for session in group.sessions:
                    action = pair_menu.addAction(f"{format_night_range(session.nights)} · {session.camera or 'Unknown camera'}",
                                                 lambda session=session: self.pair(finals, session))
                    action.setCheckable(True)
                    action.setChecked(all(f.session is session for f in finals))
                pair_menu.addSeparator()
                none_action = pair_menu.addAction("None", lambda: self.pair(finals, None))
                none_action.setCheckable(True)
                none_action.setChecked(all(f.session is None for f in finals))
        if len(finals) == 1 and not sessions:
            menu.addSeparator()
            menu.addAction("View Image", lambda: self.open_image(finals[0]))
        solvable = [entry for entry in self._solvable_items(sessions, finals)]
        if solvable and (self._identifier is None):
            menu.addSeparator()
            menu.addAction("Identify by Plate Solving", lambda: self.identify(solvable))
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(position))

    def _suggestions(self, sessions, finals):
        """Objects near where the files were pointed, for the object picker."""
        coords = []
        for session in sessions:
            clusters = LibraryScanner.position_clusters(
                [c for c in (LibraryScanner._coords(r.header) for r in session.lights[::max(1, len(session.lights) // 50)]) if c])
            if clusters:
                coords.append(clusters[0][0])
        for final in finals:
            c = LibraryScanner._coords(final.record.header)
            if c:
                coords.append(c)
        suggestions = []
        for ra, dec in coords[:5]:
            for dsodetailid in self.library.catalog.near(ra, dec):
                if dsodetailid not in suggestions:
                    suggestions.append(dsodetailid)
        return suggestions[:15]

    def change_object(self, sessions, finals, groups=()):
        if not self.library or not (sessions or finals):
            return
        parts = []
        if sessions:
            parts.append(_count(len(sessions), "session"))
        if finals:
            parts.append(_count(len(finals), "image"))
        keys = {s.dso_key for s in sessions} | {f.dso_key for f in finals}
        current = next(iter(keys)) if len(keys) == 1 else None
        dialog = ObjectPickerDialog(self.library.catalog, self._suggestions(sessions, finals),
                                    " and ".join(parts), current, self)
        if dialog.exec() != QDialog.Accepted or not dialog.chosen:
            return
        self._move(sessions, finals, dialog.chosen, "you")

    def _move(self, sessions, finals, dsodetailid, source):
        catalog = self.library.catalog
        remember, target = LibraryScanner.move_to_object(self.library, sessions, finals, dsodetailid,
                                                         catalog.name_of(dsodetailid), source)
        self._save_assignments(remember)
        self._expanded.add(("group", target.key))
        self._rebuild()
        self._select_group(target.key)

    def _save_assignments(self, remember):
        if not remember:
            return
        try:
            from DatabaseManager import DatabaseManager
            with DatabaseManager().get_connection() as conn:
                LibraryScanner.save_assignments(conn, remember)
        except Exception as e:
            logger.error(f"Couldn't save the chosen objects: {e}", exc_info=True)

    def _select_group(self, key):
        for item in self._all_items():
            data = item.data(COL_NAME, ITEM_ROLE)
            if data and data[0] == "group" and data[1].key == key:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                return

    def _all_items(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def merge(self, sessions):
        merged = LibraryScanner.merge_sessions(self.library, sessions)
        self._layout_edited = True
        self._rebuild()
        self._reselect(("session", merged))

    def split(self, session):
        dialog = SplitSessionDialog(session, self)
        if dialog.exec() != QDialog.Accepted:
            return
        new = LibraryScanner.split_session(self.library, session, dialog.chosen_nights())
        if new is not None:
            self._layout_edited = True
            self._rebuild()
            self._reselect(("session", new))

    def pair(self, finals, session):
        for final in finals:
            final.session = session
        self._layout_edited = True
        self._rebuild()

    def _request_equipment(self):
        if self._layout_edited and QMessageBox.question(
                self, "Equipment",
                "Changing equipment matches builds the sessions again, so merged or split sessions and changed "
                "image sessions go back to how they were found. The objects you chose are kept.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.equipment_requested.emit()

    def _request_rescan(self):
        if self._layout_edited and QMessageBox.question(
                self, "Scan Again",
                "Scanning again puts merged or split sessions and changed image sessions back to how "
                "they were found. The objects you chose are kept.\n\nScan again?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._stop_identifying()
        self.rescan_requested.emit()

    # ---- Identifying by plate solving ----

    def _solvable_items(self, sessions, finals):
        """[(item, path, hints)] that plate solving could name: in a group
        with no object (and not a solar-system one - those move across the
        sky), images, and sessions whose subs say nothing of where they point."""
        entries = []
        for final in finals:
            group = self.library.groups.get(final.dso_key)
            if group is None or group.dsodetailid or LibraryScanner.SOLAR_SYSTEM.search(group.name):
                continue
            coords = LibraryScanner._coords(final.record.header)
            hints = {"ra": coords[0], "dec": coords[1], "radius": 15} if coords else None
            entries.append((final, final.record.path, hints))
        for session in sessions:
            group = self.library.groups.get(session.dso_key)
            if group is None or group.dsodetailid or LibraryScanner.SOLAR_SYSTEM.search(group.name):
                continue
            if any(LibraryScanner._coords(r.header) for r in session.lights):
                continue  # Its position was already looked up - nothing there
            entries.append((session, session.lights[len(session.lights) // 2].path, None))
        return entries

    def _all_solvable(self):
        sessions = [s for g in self.library.groups.values() if not g.dsodetailid for s in g.sessions]
        finals = [f for g in self.library.groups.values() if not g.dsodetailid for f in g.finals]
        return self._solvable_items(sessions, finals)

    def _update_identify_bar(self):
        if self._identifier is not None:
            return
        count = len(self._all_solvable()) if self.library else 0
        self.identify_bar.setVisible(bool(count))
        self.identify_button.setVisible(True)
        self.identify_stop.setVisible(False)
        self.identify_label.setText(
            f"{count} file{'s' if count != 1 else ''} with no object could be identified by plate solving.")

    def identify_unknown(self):
        self.identify(self._all_solvable())

    def identify(self, entries):
        if not entries or self._identifier is not None:
            return
        blind = can_solve_blind()
        if not blind and not any(hints for _obj, _path, hints in entries):
            QMessageBox.information(
                self, "Identify by Plate Solving",
                "These files don't say where they were pointed, so plate solving them needs ASTAP. "
                "Install ASTAP (Settings > Plate Solving) to identify them.")
            return
        if QMessageBox.question(
                self, "Identify by Plate Solving",
                f"Plate solve {len(entries)} file{'s' if len(entries) != 1 else ''} to find the objects they show?"
                "\n\nThey are solved one at a time in the background - you can keep reviewing. Each solution is "
                "saved beside its file (.wcs), as plate solving does everywhere in Cosmos Collection.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        self._identifier = ObjectIdentifier(entries, self)
        self._identifier.identified.connect(self._on_identified)
        self._identifier.progress.connect(self._on_identify_progress)
        self._identifier.done.connect(self._on_identify_done)
        self.identify_bar.setVisible(True)
        self.identify_button.setVisible(False)
        self.identify_stop.setVisible(True)
        self._identifier.start()

    def _on_identified(self, obj, dsodetailid):
        if self.library is None:
            return
        sessions = [obj] if isinstance(obj, LibraryScanner.Session) else []
        finals = [obj] if isinstance(obj, LibraryScanner.FinalImage) else []
        remember, _target = LibraryScanner.move_to_object(self.library, sessions, finals, dsodetailid,
                                                          self.library.catalog.name_of(dsodetailid), "plate solve")
        self._save_assignments(remember)
        self._rebuild_timer.start()

    def _on_identify_progress(self, checked, found, total):
        self.identify_label.setText(f"Identifying by plate solving: {checked} of {total} checked, {found} found...")

    def _on_identify_done(self, found, total):
        self._identifier = None
        self._rebuild_timer.stop()
        self._rebuild()
        missed = total - found
        text = f"Plate solving found the object of {found} of {total} file{'s' if total != 1 else ''}."
        if missed:
            text += f" {missed} couldn't be solved - choose their object with Change Object."
        self.identify_bar.setVisible(True)
        self.identify_button.setVisible(bool(self._all_solvable()))
        self.identify_stop.setVisible(False)
        self.identify_label.setText(text)

    def _stop_identifying(self):
        if self._identifier is not None:
            self._identifier.stop()

    def _on_double_clicked(self, item, _column):
        data = item.data(COL_NAME, ITEM_ROLE)
        if data and data[0] == "final":
            self.open_image(data[1])

    def open_image(self, final):
        """Show a found image full size in the Image Viewer, with its object's
        position so its annotations work."""
        from DSOGallery import open_image_viewer
        ra = dec = None
        dsodetailid = final.resolution.dsodetailid
        if dsodetailid:
            try:
                from DatabaseManager import DatabaseManager
                with DatabaseManager().get_connection() as conn:
                    row = conn.execute("SELECT ra, dec FROM dsodetail WHERE id = ?", (dsodetailid,)).fetchone()
                if row:
                    ra, dec = row[0], row[1]
            except Exception as e:
                logger.debug(f"No position for {dsodetailid}: {e}")
        group = self.library.groups.get(final.dso_key) if self.library else None
        name = group.name if group else (final.resolution.name or final.record.stem)
        viewer = open_image_viewer(self, name, final.record.path, ra, dec)
        if viewer is not None:
            # Keep the viewers alive while they're open
            self._viewers = [v for v in self._viewers if _alive(v)] + [viewer]

    def _load_thumbnail(self, path):
        """One preview load at a time; the latest selection wins."""
        self._shown_path = path
        if self._thumbnail_loader is not None and self._thumbnail_loader.isRunning():
            self._pending_thumbnail = path
            return
        self._start_thumbnail(path)

    def _start_thumbnail(self, path):
        self._pending_thumbnail = None
        loader = ThumbnailLoader(path)
        loader.loaded.connect(self._on_thumbnail)
        loader.finished.connect(lambda: self._thumbnail_done(loader))
        self._thumbnail_loader = loader
        loader.start()

    def _thumbnail_done(self, loader):
        if self._thumbnail_loader is loader:
            self._thumbnail_loader = None
        loader.deleteLater()

    def _on_thumbnail(self, path, image):
        if path == self._shown_path:
            self.details.set_thumbnail(image)
        if self._pending_thumbnail and self._pending_thumbnail != path:
            self._start_thumbnail(self._pending_thumbnail)

    def shutdown(self):
        self._stop_identifying()
        if self._thumbnail_loader is not None and self._thumbnail_loader.isRunning():
            self._thumbnail_loader.wait(5000)


def _alive(widget):
    try:
        return widget.isVisible()
    except RuntimeError:    # Closed and deleted
        return False


class DetailsPane(QWidget):
    """What the selected row is: a session's nights and folders, an image's
    preview, how its object was worked out."""
    view_requested = Signal(object)     # FinalImage to show in the Image Viewer
    change_requested = Signal()         # Change Object... for what's selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 0, 0, 0)
        self.thumbnail = QLabel()
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setMinimumHeight(200)
        self.thumbnail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        themed_style(self.thumbnail, lambda: f"background: {COLORS['background_light']}; border-radius: 4px;")
        layout.addWidget(self.thumbnail)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        self.text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.text)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.table, 1)
        folders_label = QLabel("Folders")
        themed_style(folders_label, lambda: "font-weight: bold;")
        self.folders_label = folders_label
        layout.addWidget(folders_label)
        self.folders = QListWidget()
        self.folders.setToolTip("Double-click to open the folder")
        self.folders.itemDoubleClicked.connect(lambda item: self._open(item.data(Qt.UserRole)))
        self.folders.currentItemChanged.connect(lambda *_: self.open_button.setEnabled(
            self.folders.currentItem() is not None))
        layout.addWidget(self.folders, 1)
        buttons = QHBoxLayout()
        self.change_button = QPushButton("Change Object...")
        self.change_button.setAutoDefault(False)
        self.change_button.setToolTip("Choose which object this is (also on the right-click menu)")
        self.change_button.clicked.connect(self.change_requested.emit)
        buttons.addWidget(self.change_button)
        buttons.addStretch()
        self.view_button = QPushButton("View Image")
        self.view_button.setAutoDefault(False)
        self.view_button.setToolTip("Open it full size in the Image Viewer")
        self.view_button.clicked.connect(lambda: self._final is not None and self.view_requested.emit(self._final))
        buttons.addWidget(self.view_button)
        self.open_button = QPushButton("Open Folder")
        self.open_button.setAutoDefault(False)
        self.open_button.clicked.connect(lambda: self.folders.currentItem() and self._open(
            self.folders.currentItem().data(Qt.UserRole)))
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self._pixmap = None
        self._final = None
        self.clear()

    def set_changeable(self, changeable):
        self.change_button.setVisible(changeable)

    def set_viewable(self, final):
        """The image View Image (and a double-click on the preview) opens, or None."""
        self._final = final
        self.view_button.setVisible(final is not None)
        self.thumbnail.setCursor(Qt.PointingHandCursor if final is not None else Qt.ArrowCursor)
        self.thumbnail.setToolTip("Double-click to open it in the Image Viewer" if final is not None else "")

    def mouseDoubleClickEvent(self, event):
        if self._final is not None and self.thumbnail.geometry().contains(event.position().toPoint()):
            self.view_requested.emit(self._final)
            return
        super().mouseDoubleClickEvent(event)

    def clear(self):
        self.set_viewable(None)
        self.set_changeable(False)
        self._pixmap = None
        self.thumbnail.clear()
        self.thumbnail.setVisible(False)
        self.text.setText("<i>Select an item to see its details.</i>")
        self.table.setVisible(False)
        self._set_folders([])

    def _set_folders(self, folders, rel=None):
        self.folders.clear()
        for folder in folders:
            item = QListWidgetItem(rel(folder) if rel else folder)
            item.setData(Qt.UserRole, folder)
            item.setToolTip(folder)
            self.folders.addItem(item)
        shown = bool(folders)
        self.folders_label.setVisible(shown)
        self.folders.setVisible(shown)
        self.open_button.setVisible(shown)
        if shown:
            self.folders.setCurrentRow(0)

    def _open(self, folder):
        if not folder:
            return
        from ResourceManager import ResourceManager
        ResourceManager.open_file_manager(folder)

    def _set_table(self, headers, rows, numeric=()):
        """rows of values - a (text, tooltip) pair for a cell with a tooltip."""
        self.table.setVisible(bool(rows))
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                value, tooltip = value if isinstance(value, tuple) else (value, None)
                cell = QTableWidgetItem(str(value))
                if tooltip:
                    cell.setToolTip(tooltip)
                if c in numeric:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, cell)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def show_group(self, group, rel):
        self._pixmap = None
        self.thumbnail.setVisible(False)
        lines = [f"<h3>{_html(group.name)}</h3>"]
        if not group.dsodetailid:
            lines.append("Cosmos Collection couldn't tell which object these files are. Choose one with "
                         "Change Object, or let plate solving find it.")
        lines.append(f"{_count(len(group.sessions), 'session')} · {_count(len(group.finals), 'image')}")
        integration = sum(s.integration_seconds for s in group.sessions)
        if integration:
            lines.append(f"Total integration: {format_duration(integration)}")
        for folder, degrees in group.mismatches:
            lines.append(f"<span style='color:{COLORS['warning']}'>Named {_html(group.name)} but pointed "
                         f"{degrees:.0f}° away: {_html(rel(folder))}</span>")
        self.text.setText("<br>".join(lines))
        rows = [(format_night_range(s.nights), f"{len(s.lights):,}", format_duration(s.integration_seconds))
                for s in group.sessions]
        self._set_table(["Session", "Subs", "Integration"], rows, numeric=(1, 2))
        self._set_folders(sorted({_common_folder(s.folders) for s in group.sessions}), rel)

    def show_session(self, session, rel):
        self._pixmap = None
        self.thumbnail.setVisible(False)
        nights = session.nights
        lines = [f"<h3>Session · {format_night_range(nights)}</h3>",
                 f"{len(nights)} night{'s' if len(nights) != 1 else ''} · {len(session.lights):,} subs · "
                 f"{format_duration(session.integration_seconds)}",
                 f"Camera: {_html(session.camera or 'Unknown')}",
                 f"Telescope: {_html(session.telescope or 'Unknown')}"]
        mounts = session.night_mounts
        if mounts:
            lines.append(f"Mount: {MountDetection.count_text([m.mount for m in mounts.values()])}")
            if len(mounts) == 1:
                lines.append(f"<span style='color:{COLORS['text_secondary']}'>"
                             f"{_html(next(iter(mounts.values())).evidence)}</span>")
        calibration = defaultdict(int)
        for record in session.calibration:
            calibration[record.frame_type] += 1
        if calibration:
            lines.append("Calibration found with it: " + ", ".join(
                f"{n:,} {kind.lower()}{'' if kind == 'Bias' else 's'}" for kind, n in sorted(calibration.items())))
        if session.imported:
            lines.append(f"{session.imported:,} of these subs are already in the Session Manager.")
        if session.new_lights:
            lines.append(f"{session.new_lights:,} subs are new since the last scan.")
        self.text.setText("<br>".join(lines))
        per_night = defaultdict(lambda: [0, 0.0])
        for record in session.lights:
            key = (LibraryScanner._night(record), LibraryScanner._clean(record.header.get('FILTER')) or "None",
                   LibraryScanner.exposure_of(record.header) or 0)
            per_night[key][0] += 1
            per_night[key][1] += key[2]
        rows = []
        for (night, filt, exposure), (count, total) in sorted(per_night.items()):
            mount = mounts.get(night)
            rows.append((f"{night:%b} {night.day}, {night.year}", filt, f"{count:,} × {exposure:g}s",
                         format_duration(total),
                         ((mount.mount or "Unknown"), mount.evidence) if mount else "Not checked"))
        self._set_table(["Night", "Filter", "Subs", "Integration", "Mount"], rows, numeric=(2, 3))
        folders = session.folders + sorted({r.folder for r in session.calibration})
        self._set_folders(folders, rel)

    def show_final(self, final, rel):
        record = final.record
        self.thumbnail.setVisible(True)
        self.thumbnail.setText("Loading preview...")
        source = {"header": "its FITS header", "file name": "its file name", "folder name": "its folder's name",
                  "position": "where it was pointed", "you": "your choice",
                  "plate solve": "plate solving"}.get(final.resolution.source, "its folder")
        lines = [f"<h3>{_html(record.name)}</h3>",
                 f"{FINAL_TYPES.get(record.tier, 'Image')} · found by {_html(_reason_text(record))}",
                 f"Object: {_html(final.resolution.name or '')} (from {source})"]
        if record.variants:
            lines.append("Also saved as: " + ", ".join(sorted({v.ext.lstrip('.').upper() for v in record.variants})))
        if final.session:
            lines.append(f"From the session of {format_night_range(final.session.nights)} "
                         f"({_html(final.session.camera or 'unknown camera')})")
        if final.imported:
            lines.append("Already in your gallery.")
        if final.variant_words:
            lines.append("Not selected by default: looks like a by-product (" +
                         ", ".join(sorted(final.variant_words)) + ").")
        self.text.setText("<br>".join(lines))
        self.table.setVisible(False)
        self._set_folders([record.folder], rel)

    def show_calibration(self, library, rel):
        self._pixmap = None
        self.thumbnail.setVisible(False)
        exposures = sorted({e for e in (LibraryScanner.exposure_of(r.header) for r in library.files) if e})
        self.text.setText("<br>".join([
            f"<h3>{library.frame_type} frames</h3>",
            f"{len(library.files):,} frames · {_html(library.camera or 'Unknown camera')}",
            "Exposures: " + ", ".join(f"{e:g}s" for e in exposures),
            "These sit in a folder shared by several objects, so they aren't attached to a session. "
            "Choosing sessions for them isn't available yet."]))
        self.table.setVisible(False)
        self._set_folders([library.folder], rel)

    def show_skipped(self, title, reason, paths, rel):
        self._pixmap = None
        self.thumbnail.setVisible(False)
        self.text.setText(f"<h3>{_html(title)}</h3>{_html(reason)}.")
        self.table.setVisible(False)
        self._set_folders(sorted(set(paths))[:500], rel)

    def set_thumbnail(self, image):
        if image is None or image.isNull():
            self._pixmap = None
            self.thumbnail.setText("No preview available")
            return
        self._pixmap = QPixmap.fromImage(image)
        self._fit_thumbnail()

    def _fit_thumbnail(self):
        if self._pixmap is not None:
            width = max(self.thumbnail.width(), 100)
            self.thumbnail.setPixmap(self._pixmap.scaled(width, min(320, int(width * 0.8)),
                                                         Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_thumbnail()


def _html(text):
    from html import escape
    return escape(str(text or ""))


def _reason_text(record):
    if record.reason == "name":
        return "its file name"
    if record.reason == "folder name":
        return "its folder's name"
    if record.reason == "software's default output name":
        return "its name - one your processing software gives by default, so likely a step along the way"
    return record.reason


# ---------------------------------------------------------------------------
# Editing: choosing an object, splitting a session, plate solving
# ---------------------------------------------------------------------------

class ObjectPickerDialog(QDialog):
    """Which object some sessions/images show: search the catalog by name or
    designation, or pick from the objects near where they were pointed."""

    def __init__(self, catalog, suggestions, what, current=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose the Object")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.resize(460, 520)
        self.catalog = catalog
        self.suggestions = [i for i in suggestions if i != current]
        self.chosen = None
        layout = QVBoxLayout(self)
        intro = QLabel(f"Choose the object for {_html(what)}.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Type a name or designation - M 31, NGC 7000, Pelican...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill)
        layout.addWidget(self.search)
        self.hint = QLabel("")
        themed_style(self.hint, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.hint)
        self.results = QListWidget()
        self.results.itemDoubleClicked.connect(lambda _item: self._accept())
        self.results.currentItemChanged.connect(lambda *_: self.ok_button.setEnabled(self.results.currentItem() is not None))
        layout.addWidget(self.results, 1)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self.ok_button)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._fill("")
        self.search.setFocus()

    def _fill(self, text):
        self.results.clear()
        if text.strip():
            ids = self.catalog.search(text, limit=200)
            self.hint.setText("" if ids else "Nothing in the catalog matches that.")
        else:
            ids = self.suggestions
            self.hint.setText("Near where they were pointed:" if ids else "Type to search the catalog.")
        for dsodetailid in ids:
            item = QListWidgetItem(self.catalog.label_of(dsodetailid))
            item.setData(Qt.UserRole, dsodetailid)
            self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)
        self.ok_button.setEnabled(self.results.currentItem() is not None)

    def _accept(self):
        item = self.results.currentItem()
        if item is not None:
            self.chosen = item.data(Qt.UserRole)
            self.accept()


class SplitSessionDialog(QDialog):
    """Pick the nights of a session to make a session of their own."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split Session")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.resize(420, 420)
        layout = QVBoxLayout(self)
        intro = QLabel("Check the nights to move into a new session:")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        counts = defaultdict(lambda: [0, 0.0])
        for record in session.lights:
            night = LibraryScanner._night(record)
            counts[night][0] += 1
            counts[night][1] += LibraryScanner.exposure_of(record.header) or 0
        self.nights = QListWidget()
        for night in sorted(counts):
            subs, seconds = counts[night]
            item = QListWidgetItem(f"{night:%b} {night.day}, {night.year}  ·  {subs:,} subs  ·  {format_duration(seconds)}")
            item.setData(Qt.UserRole, night)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.nights.addItem(item)
        self.nights.itemChanged.connect(self._update)
        layout.addWidget(self.nights, 1)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.ok_button = QPushButton("Split")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self.ok_button)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._update()

    def chosen_nights(self):
        return [self.nights.item(i).data(Qt.UserRole) for i in range(self.nights.count())
                if self.nights.item(i).checkState() == Qt.Checked]

    def _update(self, *_):
        chosen = len(self.chosen_nights())
        self.ok_button.setEnabled(0 < chosen < self.nights.count())


def can_solve_blind():
    """Whether an image can be plate solved without knowing where it points
    (ASTAP; astrometry.net could, but uploads the image)."""
    try:
        from PlateSolver import PlateSolver, solver_order
        return "astap" in solver_order() and PlateSolver().is_astap_available()
    except Exception as e:
        logger.debug(f"Couldn't check for ASTAP: {e}")
        return False


class ObjectIdentifier(QObject):
    """Finds what files nothing named show, by plate solving them in the app's
    shared queue (one at a time, in the background): the most notable object
    in each solved image. Files already solved (.wcs) are just read."""
    identified = Signal(object, str)    # the Session or FinalImage, dsodetailid
    progress = Signal(int, int, int)    # checked, found, total
    done = Signal(int, int)             # found, total

    def __init__(self, items, parent=None):
        """items: [(Session or FinalImage, file path to solve, hints or None)]"""
        super().__init__(parent)
        from AnnotationOverlay import annotation_store, plate_solve_queue
        self._queue, self._store = plate_solve_queue(), annotation_store()
        self.items = {path: (obj, hints) for obj, path, hints in items}
        self.pending = set(self.items)
        self.tried = set()
        self.found = 0
        self.blind = can_solve_blind()
        self._queue.changed.connect(self._check)
        self._store.ready.connect(self._check)

    def start(self):
        for path in list(self.pending):
            self._check(path)
        self._report()

    def _check(self, path):
        if path not in self.pending:
            return
        state = self._queue.state(path)
        if state == "solved":
            data = self._store.get(path)
            if data is not None:
                self._finish(path, data)
            return  # Being read - ready() comes back here
        if state in ("unsolved", "failed"):
            if path in self.tried:
                self._finish(path, None)
                return
            self.tried.add(path)
            _obj, hints = self.items[path]
            if not (hints or self.blind) or not self._queue.add(path, hints, os.path.basename(path)):
                self._finish(path, None)

    def _finish(self, path, data):
        self.pending.discard(path)
        obj, _hints = self.items[path]
        best = next((o for o in data.dsos if o.dsodetailid), None) if data is not None else None
        if best is not None:
            self.found += 1
            self.identified.emit(obj, str(best.dsodetailid))
        self._report()

    def _report(self):
        total = len(self.items)
        self.progress.emit(total - len(self.pending), self.found, total)
        if not self.pending:
            self._disconnect()
            self.done.emit(self.found, total)

    def stop(self):
        if self.pending:
            self._queue.stop(list(self.pending))
            self.pending.clear()
            self._report()

    def _disconnect(self):
        for signal, slot in ((self._queue.changed, self._check), (self._store.ready, self._check)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class ImportWizardWindow(WindowPositionMixin, QDialog):
    WINDOW_POSITION_KEY = "ImportWizard"
    FOLDERS_PAGE, SCANNING_PAGE, EQUIPMENT_PAGE, REVIEW_PAGE = range(4)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Wizard - Cosmos Collection")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.resize(1280, 760)
        self.worker = None
        self._last_folders = []
        self.scan = self.library = None
        self._revisiting_equipment = False

        layout = QVBoxLayout(self)
        self.pages = QStackedWidget()
        self.folders_page = FoldersPage()
        self.scanning_page = ScanningPage()
        self.equipment_page = EquipmentPage()
        self.review_page = ReviewPage()
        for page in (self.folders_page, self.scanning_page, self.equipment_page, self.review_page):
            self.pages.addWidget(page)
        layout.addWidget(self.pages)

        self.folders_page.scan_requested.connect(self.start_scan)
        self.scanning_page.stop_requested.connect(self._stop_scan)
        self.review_page.back_requested.connect(self._show_folders)
        self.review_page.rescan_requested.connect(lambda: self.start_scan(self.folders_page.folders(), False))
        self.review_page.equipment_requested.connect(lambda: self._show_equipment(revisiting=True))
        self.equipment_page.finished.connect(self._on_equipment_matched)
        self.review_page.import_requested.connect(self.start_import)
        self.import_worker = None
        self._import_progress = None
        self.equipment_page.back_requested.connect(self._equipment_back)
        self.setup_window_position()

    def start_scan(self, folders, reread):
        if self.worker is not None and self.worker.isRunning():
            return
        if not folders:
            self._show_folders()
            return
        self._last_folders = folders
        self.scanning_page.reset()
        self.pages.setCurrentIndex(self.SCANNING_PAGE)
        self.worker = ScanWorker(folders, load_rules(), reread)
        self.worker.progress.connect(self.scanning_page.set_progress)
        self.worker.status.connect(self.scanning_page.set_status)
        self.worker.scanned.connect(self._on_scanned)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _stop_scan(self):
        if self.worker is not None:
            self.worker.stop()

    def _on_scanned(self, scan, library):
        self.worker = None
        self.scan, self.library = scan, library
        self.folders_page.refresh_scan_dates()
        _refresh_session_manager()  # Sessions it already had may have been given their mounts
        # Names in the files nobody has matched to equipment yet: ask first
        equipment = library.equipment or LibraryScanner.EquipmentInfo()
        if any(not equipment.decided(row["kind"], row["value"]) for row in LibraryScanner.equipment_in_files(library)):
            self._show_equipment(revisiting=False)
        else:
            self._show_review()

    def _show_review(self):
        self.review_page.show_library(self.scan, self.library)
        self.pages.setCurrentIndex(self.REVIEW_PAGE)

    def _show_equipment(self, revisiting):
        if self.library is None:
            return
        self._revisiting_equipment = revisiting
        self.equipment_page.show_equipment(LibraryScanner.equipment_in_files(self.library),
                                           self.library.equipment or LibraryScanner.EquipmentInfo(), revisiting)
        self.pages.setCurrentIndex(self.EQUIPMENT_PAGE)

    def _equipment_back(self):
        if self._revisiting_equipment:
            self.pages.setCurrentIndex(self.REVIEW_PAGE)
        else:
            self._show_folders()

    def _on_equipment_matched(self, decisions, new):
        """Save the matches (adding the new equipment, linking gallery images given
        those names), then group the sessions by them."""
        from DatabaseManager import DatabaseManager
        try:
            with DatabaseManager().get_connection() as conn:
                for kind, value, header in new:
                    measured = LibraryScanner.equipment_from_header(kind, header)
                    if kind == LibraryScanner.TELESCOPE:
                        item_id = LibraryScanner.add_telescope(conn, value, measured["focal_length"], measured["aperture"])
                    else:
                        width, height = measured["sensor_width"], measured["sensor_height"]
                        if width and height and width < height:   # A portrait frame: the sensor is landscape
                            width, height = height, width
                        item_id = LibraryScanner.add_camera(conn, value, width, height, measured["pixel_size"])
                    decisions[(kind, value)] = item_id
                # Gallery images given these names link to the equipment too
                import EquipmentMapper
                EquipmentMapper.save_matches(conn, decisions)
                equipment = LibraryScanner.load_equipment_info(conn)
        except Exception as e:
            logger.error(f"Couldn't save the equipment matches: {e}", exc_info=True)
            QMessageBox.critical(self, "Equipment", f"The equipment matches couldn't be saved:\n{e}")
            return
        LibraryScanner.regroup_sessions(self.library, equipment)
        if new:
            _refresh_equipment_window()
        self._show_review()

    # ---- Importing ----

    def start_import(self):
        from PySide6.QtWidgets import QProgressDialog
        if self.library is None or self.import_worker is not None:
            return
        sessions, finals = LibraryScanner.plan(self.library)
        if not sessions and not finals:
            return
        lights = sum(len(s.lights) for s in sessions)
        calibration = sum(len(s.calibration) for s in sessions)
        lines = []
        if sessions:
            line = f"{_count(len(sessions), 'session')} with {_count(lights, 'sub')}"
            if calibration:
                line += f" and {_count(calibration, 'calibration frame')}"
            lines.append(line + " into the Session Manager")
        if finals:
            lines.append(f"{_count(len(finals), 'image')} into the gallery")
        gap = self.library.session_gap_days
        if QMessageBox.question(
                self, "Import",
                "Import " + " and ".join(lines) + "?\n\n"
                "Your files stay where they are - only references to them are added. Sessions whose last night "
                f"is more than {gap} days ago, or that have a finished image, are marked Completed; recent "
                "ones In Progress. Subs already in a session are added to it.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        self._import_progress = QProgressDialog("Importing...", "Stop", 0, len(sessions) + (1 if finals else 0), self)
        self._import_progress.setWindowTitle("Import")
        self._import_progress.setWindowModality(Qt.WindowModal)
        self._import_progress.setMinimumDuration(0)
        self._import_progress.setAutoClose(False)
        self._import_progress.setAutoReset(False)
        self.import_worker = ImportWorker(self.library)
        self._import_progress.canceled.connect(self._stop_import)
        self.import_worker.progress.connect(self._on_import_progress)
        self.import_worker.imported.connect(self._on_imported)
        self.import_worker.failed.connect(self._on_import_failed)
        self.import_worker.finished.connect(self.import_worker.deleteLater)
        self.import_worker.start()

    def _stop_import(self):
        if self.import_worker is not None:
            self.import_worker.stop()
            self._import_progress.setLabelText("Stopping after the current session...")

    def _on_import_progress(self, done, total, text):
        if self._import_progress is not None and not self._import_progress.wasCanceled():
            self._import_progress.setMaximum(max(total, 1))
            self._import_progress.setValue(done)
            self._import_progress.setLabelText(text)

    def _close_import_progress(self):
        self.import_worker = None
        if self._import_progress is not None:
            self._import_progress.close()
            self._import_progress.deleteLater()
            self._import_progress = None

    def _on_import_failed(self, message):
        self._close_import_progress()
        QMessageBox.critical(self, "Import", f"The import couldn't be done:\n{message}")

    def _on_imported(self, report):
        self._close_import_progress()
        # What's in the collection now: marked imported, nothing left checked
        for group in self.library.groups.values():
            for session in group.sessions:
                if id(session) in report.session_ids:
                    session.checked = False
                    for record in session.lights:
                        record.imported = True
        for final in report.imported_finals:
            final.imported, final.checked = True, False
        self.review_page.show_library(self.scan, self.library)
        try:
            from DSOGallery import refresh_open_image_views
            refresh_open_image_views()
        except Exception as e:
            logger.debug(f"Couldn't refresh the gallery: {e}")
        _refresh_session_manager()

        lines = []
        if report.sessions_created:
            lines.append(f"{_count(report.sessions_created, 'new session')} "
                         f"({report.completed} Completed, {report.in_progress} In Progress)")
        if report.sessions_extended:
            lines.append(f"{_count(report.sessions_extended, 'existing session')} got more subs")
        if report.subs_added or report.calibration_added:
            lines.append(f"{_count(report.subs_added, 'sub')} and {_count(report.calibration_added, 'calibration frame')}")
        if report.images_added:
            lines.append(f"{_count(report.images_added, 'image')} added to the gallery")
        if report.subs_skipped:
            lines.append(f"{_count(report.subs_skipped, 'sub')} skipped - already in another session")
        if report.images_skipped:
            lines.append(f"{_count(report.images_skipped, 'image')} skipped - already in the gallery")
        if report.images_linked:
            lines.append(f"{_count(report.images_linked, 'gallery image')} linked to "
                         f"{'its session' if report.images_linked == 1 else 'their sessions'}")
        text = "\n".join("• " + line for line in lines) or "Nothing new was imported."
        if report.cancelled:
            text = "The import was stopped part way. What was imported is kept:\n\n" + text
        if report.errors:
            text += "\n\nThese couldn't be imported:\n" + "\n".join(report.errors[:10])
            QMessageBox.warning(self, "Import", text)
        else:
            QMessageBox.information(self, "Import Done", text)

    def _on_failed(self, message):
        from PySide6.QtWidgets import QMessageBox
        self.worker = None
        QMessageBox.critical(self, "Scan Failed", f"The folders couldn't be scanned:\n{message}")
        self._show_folders()

    def _show_folders(self):
        self.folders_page.refresh_scan_dates()
        self.pages.setCurrentIndex(self.FOLDERS_PAGE)

    def dragEnterEvent(self, event):
        if self.pages.currentIndex() == self.FOLDERS_PAGE and event.mimeData().hasUrls() and any(
                url.isLocalFile() and os.path.isdir(url.toLocalFile()) for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        folders = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile() and os.path.isdir(u.toLocalFile())]
        if folders:
            self.folders_page.add_folders(folders)
            event.acceptProposedAction()

    def closeEvent(self, event):
        if self.import_worker is not None and self.import_worker.isRunning():
            # Each session is its own transaction - stop after the current one
            self.import_worker.stop()
            self.import_worker.wait(60000)
        if self.worker is not None and self.worker.isRunning():
            # What was read so far is saved by the worker as it stops
            self.worker.stop()
            self.worker.wait(30000)
        self.review_page.shutdown()
        self.review_page.save_state()
        super().closeEvent(event)


def _refresh_session_manager():
    """An open Session Manager shows the sessions just imported."""
    from PySide6.QtWidgets import QApplication
    for widget in QApplication.topLevelWidgets():
        if type(widget).__name__ == "SessionManagerWindow" and widget.isVisible():
            try:
                widget._load_sessions()
            except Exception as e:
                logger.debug(f"Couldn't refresh the Session Manager: {e}")


def _refresh_equipment_window():
    """An open Equipment window shows the telescopes/cameras just added."""
    try:
        import EquipmentManager
        window = getattr(EquipmentManager, "_window", None)
        if window is not None and window.isVisible():
            for page in window.pages:
                page.reload()
    except Exception as e:
        logger.debug(f"Couldn't refresh the Equipment window: {e}")


_window = None


def open_import_wizard(parent=None):
    """The Import Wizard window - one at a time; opening it again brings it to the front."""
    global _window
    try:
        if _window is not None and _window.isVisible():
            _window.showNormal()
            _window.raise_()
            _window.activateWindow()
            return _window
    except RuntimeError:
        pass  # Closed and deleted
    _window = ImportWizardWindow(parent)
    _window.show()
    return _window
