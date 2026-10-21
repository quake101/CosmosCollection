#!/usr/bin/env python3
"""
Session File Check
Finds a session's attached files that are no longer on disk, or no longer on
any observing night, and lets the user tidy them up: missing files are looked
for beside where they were (a renamed or moved night folder) or in a folder the
user picks, and relinked when their headers match; the rest can be removed from
the session. Files on disk are never changed.
"""

import os
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QHeaderView, QAbstractItemView, QProgressBar, QFileDialog,
                               QMessageBox)

import SessionFileScanner
import SessionObservations
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS, themed_style
from SessionManager import _retire_thread

logger = logging.getLogger(__name__)

# A missing folder's renamed copy is looked for under its nearest existing
# parent, at most this many levels up (a night folder inside a target folder).
SEARCH_LEVELS_UP = 3
# Most folders one search lists - keeps a mistaken pick of a whole drive bounded.
MAX_SEARCH_FOLDERS = 20000


# ---- Disk checks (worker threads) ---------------------------------------------

def missing_paths(paths, cancelled=None):
    """(missing, unreachable): the paths not on disk, and of those, the ones on a
    drive or share that isn't there at all (offline NAS, unplugged disk) - those
    may well come back. One folder listing per folder instead of a check per file,
    which is much faster on a network drive."""
    by_folder = defaultdict(list)
    for path in paths:
        by_folder[os.path.dirname(path)].append(path)
    missing, unreachable = set(), set()
    for folder, files in by_folder.items():
        if cancelled and cancelled():
            break
        try:
            names = {os.path.normcase(name) for name in os.listdir(folder)}
        except OSError:
            missing.update(files)
            if not _existing_ancestor(folder):
                unreachable.update(files)
            continue
        missing.update(p for p in files if os.path.normcase(os.path.basename(p)) not in names)
    return missing, unreachable


def _existing_ancestor(folder, levels=None):
    """(nearest existing folder above folder, levels climbed), or None - within
    levels, or up to the drive or share root."""
    current, climbed = folder, 0
    while True:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return None
        current, climbed = parent, climbed + 1
        if levels is not None and climbed > levels:
            return None
        if os.path.isdir(current):
            return current, climbed


def _list_tree(root, depth, wanted, found, budget, cancelled):
    """Add files under root (up to depth folders down) whose normcased name is in
    wanted to found {name: [paths]}. Skips Rejected folders. budget: [folders left]."""
    stack = [(root, 0)]
    while stack and budget[0] > 0:
        if cancelled and cancelled():
            return
        folder, level = stack.pop()
        budget[0] -= 1
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if level < depth and entry.name.lower() != SessionFileScanner.REJECTED_FOLDER.lower():
                                stack.append((entry.path, level + 1))
                        elif os.path.normcase(entry.name) in wanted:
                            found[os.path.normcase(entry.name)].append(os.path.normpath(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue


def _fingerprint(path):
    data = SessionFileScanner.scan_file(path)
    if data is None:
        return None
    row = SessionFileScanner.file_dict_to_row(data)
    return SessionObservations.make_sub_key(row["date_obs"], row["exptime_seconds"], row["filter_name"],
                                           row["frame_type"], row["camera"])


def find_moved(files, search_root=None, progress=None, cancelled=None):
    """{file id: new path} for missing files found again by name, and confirmed
    by their headers' fingerprint when the file has one (a name-only match must
    be the only file of that name). files: file dicts with id, file_path, sub_key.
    With search_root None each file is looked for near where it was - under its
    nearest existing parent folder, about as deep as it was (a renamed night
    folder); otherwise anywhere under search_root."""
    progress = progress or (lambda text: None)
    wanted = {os.path.normcase(os.path.basename(f["file_path"])) for f in files}
    found = defaultdict(list)
    budget = [MAX_SEARCH_FOLDERS]
    if search_root:
        progress(f"Looking through {search_root}...")
        _list_tree(search_root, 64, wanted, found, budget, cancelled)
    else:
        searches = {}
        for folder in {os.path.dirname(f["file_path"]) for f in files}:
            anchor = _existing_ancestor(folder, SEARCH_LEVELS_UP) if not os.path.isdir(folder) else (folder, 0)
            if anchor:
                root, climbed = anchor
                key = os.path.normcase(root)
                searches[key] = (root, max(climbed + 1, searches.get(key, (root, 0))[1]))
        for root, depth in searches.values():
            progress(f"Looking for moved files in {root}...")
            _list_tree(root, depth, wanted, found, budget, cancelled)

    moves = {}
    for n, f in enumerate(files):
        if cancelled and cancelled():
            break
        old = os.path.normcase(os.path.normpath(f["file_path"]))
        candidates = [p for p in found.get(os.path.normcase(os.path.basename(f["file_path"])), [])
                      if os.path.normcase(os.path.normpath(p)) != old]
        if not candidates:
            continue
        progress(f"Checking the files found: {n + 1:,} of {len(files):,}")
        if f.get("sub_key"):
            match = next((p for p in candidates if _fingerprint(p) == f["sub_key"]), None)
        else:
            match = candidates[0] if len(candidates) == 1 else None
        if match:
            moves[f["id"]] = match
    return moves


@dataclass
class CheckResult:
    missing: set = field(default_factory=set)       # file ids not on disk
    unreachable: set = field(default_factory=set)   # of those, on a drive or share that isn't there
    moved: dict = field(default_factory=dict)       # file id -> where it was found


class FileCheckWorker(QThread):
    """Checks which files are on disk and, with find, looks for the missing ones."""

    status = Signal(str)
    checked = Signal(object)  # CheckResult

    def __init__(self, files, find=False, search_root=None, parent=None):
        super().__init__(parent)
        self.files = [dict(f) for f in files]
        self.find = find
        self.search_root = search_root
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        result = CheckResult()
        try:
            cancelled = lambda: self._stop
            self.status.emit(f"Checking {len(self.files):,} file(s) on disk...")
            by_path = defaultdict(list)
            for f in self.files:
                by_path[f["file_path"]].append(f["id"])
            missing, unreachable = missing_paths(list(by_path), cancelled)
            result.missing = {i for p in missing for i in by_path[p]}
            result.unreachable = {i for p in unreachable for i in by_path[p]}
            lost = [f for f in self.files if f["id"] in result.missing and f["id"] not in result.unreachable]
            if self.find and lost:
                result.moved = find_moved(lost, self.search_root, self.status.emit, cancelled)
        except Exception as e:
            logger.error(f"Checking session files failed: {e}", exc_info=True)
        self.checked.emit(result)


# ---- Summaries ------------------------------------------------------------------

def stranded_calibration(files):
    """Calibration frames on no night - a night with no lights left, or taken on
    a night the session has no lights for."""
    return [f for f in files if f["observation_id"] is None and f["frame_type"] != "Light"]


def problem_summary(files, result):
    """(text, is_warning) for the Files tab's line, or ('', False) when all's well."""
    by_id = {f["id"]: f for f in files}
    missing = Counter(by_id[i]["frame_type"] for i in result.missing if i in by_id)
    alone = Counter(f["frame_type"] for f in stranded_calibration(files) if f["id"] not in result.missing)
    parts = []
    if missing:
        unreachable = len(result.unreachable)
        where = " (their drive isn't available)" if unreachable and unreachable == sum(missing.values()) else ""
        parts.append(f"{SessionObservations.describe_frame_counts(missing)} "
                     f"{'is' if sum(missing.values()) == 1 else 'are'} missing on disk{where}")
    if alone:
        parts.append(f"{SessionObservations.describe_frame_counts(alone)} "
                     f"{'is' if sum(alone.values()) == 1 else 'are'}n't on any night")
    if not parts:
        return "", False
    return "; ".join(parts) + ".", bool(missing)


# ---- Asking before a night's last lights go ------------------------------------

def ask_stranded(parent, db_manager, session_id, file_ids):
    """When removing file_ids empties a night of its lights, ask whether its
    flats/darks/bias go too. Returns the extra file ids to remove ([] for no)."""
    try:
        with db_manager.get_connection() as conn:
            stranded = SessionObservations.calibration_left_without_night(conn, session_id, file_ids)
    except Exception as e:
        logger.error(f"Couldn't check for calibration frames left on no night: {e}")
        return []
    if not stranded:
        return []
    extra = [file_id for files in stranded.values() for file_id, _type in files]
    reply = QMessageBox.question(
        parent, "Remove the Night's Calibration Frames?",
        f"That leaves {SessionObservations.stranded_calibration_text(stranded)} with no lights, so the night "
        f"is removed.\n\nAlso remove its calibration frames from the session? Kept, they're still used when "
        f"stacking. The files on disk aren't touched.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    return extra if reply == QMessageBox.Yes else []


# ---- The clean up dialog ----------------------------------------------------------

MOVED, MISSING, UNREACHABLE, NO_NIGHT = "moved", "missing", "unreachable", "no_night"


class SessionFileCheckDialog(WindowPositionMixin, QDialog):
    """A session's files that need attention, grouped by folder, with what
    Apply does to each group: relink moved files, remove missing ones (and, when
    checked, calibration frames on no night) from the session."""

    WINDOW_POSITION_KEY = "SessionFileCheckDialog"
    FIND_DIR_SETTING = "SessionFileCheck/find_dir"

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = dict(session)
        self.session_id = session["id"]
        self.db_manager = DatabaseManager()
        self.changed = False
        self._files = []
        self._result = CheckResult()
        self._worker = None
        self._searching_root = None
        self._groups = []

        from DSONames import display_name
        self.setWindowTitle(f"Check Files — {display_name(session.get('dso_name', ''))}")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        self.setModal(True)
        self.resize(900, 460)
        self._setup_ui()
        self.setup_window_position()
        self._check()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel("Files this session points to that are no longer on disk, or no longer on any night. "
                       "Moved files are found by name and confirmed by their headers. Apply changes only the "
                       "session - files on disk are never touched.")
        intro.setWordWrap(True)
        themed_style(intro, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(intro)

        status_row = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(160)
        status_row.addWidget(self.busy)
        layout.addLayout(status_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Problem", "Files", "Folder", "Apply will"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        for column in (0, 1, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.itemChanged.connect(lambda _item: self._update_buttons())
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.find_btn = QPushButton("Find in Folder...")
        self.find_btn.setToolTip("Look for the missing files anywhere under a folder you choose")
        self.find_btn.clicked.connect(self._find_in_folder)
        buttons.addWidget(self.find_btn)
        self.again_btn = QPushButton("Check Again")
        self.again_btn.clicked.connect(self._check)
        buttons.addWidget(self.again_btn)
        buttons.addStretch()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._apply)
        buttons.addWidget(self.apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    # ---- Checking ----

    def _load_files(self):
        try:
            with self.db_manager.get_connection() as conn:
                self._files = SessionObservations.load_session_files(conn, self.session_id)
        except Exception as e:
            logger.error(f"Error loading session {self.session_id}'s files: {e}")
            self._files = []

    def _check(self, search_root=None):
        if self._worker is not None and self._worker.isRunning():
            return
        self._load_files()
        files = self._files
        if search_root:  # Only the ones still missing
            files = [f for f in files if f["id"] in self._result.missing and f["id"] not in self._result.moved]
        self._searching_root = search_root
        # No parent: a search still running when the dialog closes is retired, not destroyed
        self._worker = FileCheckWorker(files, find=True, search_root=search_root)
        self._worker.status.connect(self.status_label.setText)
        self._worker.checked.connect(self._on_checked)
        self._set_busy(True)
        self._worker.start()

    def _set_busy(self, busy):
        self.busy.setVisible(busy)
        for button in (self.find_btn, self.again_btn, self.apply_btn):
            button.setEnabled(not busy)
        if not busy:
            self._update_buttons()

    def _on_checked(self, result):
        if self._searching_root:
            # A folder search: keep the first check's missing files, add what it found
            result = CheckResult(self._result.missing, self._result.unreachable,
                                 {**self._result.moved, **result.moved})
        result.moved = self._unattached(result.moved)
        self._result = result
        self._set_busy(False)
        self._fill_table()

    def _unattached(self, moved):
        """Drops found paths some session already has - relinking would attach a sub twice."""
        if not moved:
            return moved
        try:
            with self.db_manager.get_connection() as conn:
                attached = {os.path.normcase(os.path.normpath(row[0]))
                            for row in conn.execute("SELECT file_path FROM usersessionfiles")}
        except Exception as e:
            logger.error(f"Error reading attached files: {e}")
            return {}
        return {i: p for i, p in moved.items() if os.path.normcase(os.path.normpath(p)) not in attached}

    # ---- Table ----

    def _fill_table(self):
        result = self._result
        groups = defaultdict(list)  # (kind, folder, new folder) -> files
        for f in self._files:
            folder = os.path.normpath(os.path.dirname(f["file_path"]))
            if f["id"] in result.moved:
                groups[(MOVED, folder, os.path.dirname(result.moved[f["id"]]))].append(f)
            elif f["id"] in result.unreachable:
                groups[(UNREACHABLE, folder, None)].append(f)
            elif f["id"] in result.missing:
                groups[(MISSING, folder, None)].append(f)
        for f in stranded_calibration(self._files):
            if f["id"] not in result.missing:
                groups[(NO_NIGHT, os.path.normpath(os.path.dirname(f["file_path"])), None)].append(f)
        order = {MOVED: 0, MISSING: 1, UNREACHABLE: 2, NO_NIGHT: 3}
        self._groups = sorted(groups.items(), key=lambda item: (order[item[0][0]], item[0][1].lower()))

        problem = {MOVED: "Moved", MISSING: "Not on disk", UNREACHABLE: "Drive not available",
                   NO_NIGHT: "Not on any night"}
        action = {MOVED: "Point to the new folder", MISSING: "Remove from the session",
                  UNREACHABLE: "Nothing - connect the drive and Check Again",
                  NO_NIGHT: "Remove from the session"}
        tips = {
            MOVED: "Found by name in another folder, with matching headers - a renamed or moved folder.",
            MISSING: "Not found on disk, or near where it was. Find in Folder... looks somewhere else.",
            UNREACHABLE: "The drive or network share this folder is on isn't there right now.",
            NO_NIGHT: "Calibration frames from a night this session has no lights for - often a night whose "
                      "lights were removed. They're still used when stacking; check to remove them.",
        }
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._groups))
        for row, ((kind, folder, new_folder), files) in enumerate(self._groups):
            item = QTableWidgetItem(problem[kind])
            item.setToolTip(tips[kind])
            if kind == UNREACHABLE:
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            else:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked if kind == NO_NIGHT else Qt.Checked)
            if kind in (MISSING, UNREACHABLE):
                item.setForeground(QColor(COLORS["warning"]))
            self.table.setItem(row, 0, item)
            counts = Counter(f["frame_type"] for f in files)
            self.table.setItem(row, 1, QTableWidgetItem(SessionObservations.describe_frame_counts(counts)))
            where = folder if new_folder is None else f"{folder}  →  {new_folder}"
            folder_item = QTableWidgetItem(where)
            folder_item.setToolTip(where)
            self.table.setItem(row, 2, folder_item)
            self.table.setItem(row, 3, QTableWidgetItem(action[kind]))
        self.table.blockSignals(False)

        if not self._groups:
            self.status_label.setText(f"All {len(self._files):,} attached file(s) are on disk and on a night.")
        else:
            text, _warning = problem_summary(self._files, result)
            if result.moved:
                text += f" {len(result.moved):,} of the missing were found in another folder."
            self.status_label.setText(text.strip())
        self._update_buttons()

    def _checked_groups(self):
        return [group for row, group in enumerate(self._groups)
                if self.table.item(row, 0) and self.table.item(row, 0).checkState() == Qt.Checked
                and group[0][0] != UNREACHABLE]

    def _update_buttons(self):
        busy = self._worker is not None and self._worker.isRunning()
        lost = any(i not in self._result.moved and i not in self._result.unreachable for i in self._result.missing)
        self.find_btn.setEnabled(not busy and lost)
        self.apply_btn.setEnabled(not busy and bool(self._checked_groups()))

    # ---- Actions ----

    def _find_in_folder(self):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        start = settings.value(self.FIND_DIR_SETTING, "", type=str)
        if not start:
            lost = [f["file_path"] for f in self._files if f["id"] in self._result.missing]
            anchor = _existing_ancestor(os.path.dirname(lost[0])) if lost else None
            start = anchor[0] if anchor else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Find the Missing Files In", start)
        if not folder:
            return
        settings.setValue(self.FIND_DIR_SETTING, folder)
        self._check(search_root=folder)

    def _apply(self):
        moves, remove = [], []
        for (kind, _folder, _new), files in self._checked_groups():
            if kind == MOVED:
                moves.extend((f["id"], f["file_path"], self._result.moved[f["id"]]) for f in files)
            else:
                remove.extend(f["id"] for f in files)
        if not moves and not remove:
            return
        if remove:
            counts = Counter(f["frame_type"] for f in self._files if f["id"] in set(remove))
            if QMessageBox.question(
                    self, "Remove from Session",
                    f"Remove {SessionObservations.describe_frame_counts(counts)} from this session?"
                    + (f"\n\n{len(moves):,} moved file(s) will also be pointed to their new folder." if moves else "")
                    + "\n\nThe files on disk aren't touched.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
                return
        try:
            with self.db_manager.get_connection() as conn:
                SessionObservations.relink_files(conn, moves)
                if remove:
                    SessionObservations.remove_files(conn, self.session_id, remove)
                conn.commit()
        except Exception as e:
            try:
                with self.db_manager.get_connection() as conn:
                    conn.rollback()
            except Exception:
                pass
            logger.error(f"Error cleaning up session {self.session_id}'s files: {e}")
            QMessageBox.critical(self, "Error", f"The session couldn't be updated: {e}")
            return
        self.changed = True
        self._result = CheckResult()
        self._check()

    def done(self, result):
        if self._worker is not None:
            self._worker.stop()
            _retire_thread(self._worker)
        super().done(result)
