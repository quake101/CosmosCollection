#!/usr/bin/env python3
"""
Filter Corrections
Set the filter for a night's subs whose headers don't record it (a manual
filter) or record the wrong one. The night's lights and flats are split into
runs - a new run starts after a break in the subs or when the exposure (or the
files' own filter) changes, which is where a manual filter usually gets swapped -
and each run gets a filter. Saved with the session through
SessionObservations.set_file_filters; clearing goes back to the files' own filters.
"""

import logging
from datetime import timedelta

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
                               QSpinBox, QHeaderView, QAbstractItemView)

from SessionManager import _make_table, _table_item, format_exposure
from Theme import COLORS, themed_style
from WindowPositionManager import WindowPositionMixin
import ProcessingHandoff
import SessionObservations

logger = logging.getLogger(__name__)

RUN_FRAME_TYPES = ("Light", "Flat")
DEFAULT_GAP_MINUTES = 10
GAP_SETTING = "filter_corrections/gap_minutes"

FROM_FILE_TEXT = "From the files"
KEEP_TEXT = "(as set now - mixed)"
_FROM_FILE, _KEEP = "from_file", "keep"


def current_correction(f):
    """The filter the user set for a file dict, or None for the file's own."""
    return f["filter_name"] if f.get("filter_corrected") else None


def split_runs(files, tz_name, gap_minutes):
    """[{"frame_type", "files", "start", "end", "exposure"}] for one night's
    lights and flats (each frame type on its own), in time order. A run ends
    where the break from one sub's end to the next one's start is longer than
    gap_minutes, or where the exposure or the files' own filter changes."""
    gap = timedelta(minutes=gap_minutes)
    runs = []
    for frame_type in RUN_FRAME_TYPES:
        timed = []
        for f in files:
            if f["frame_type"] != frame_type:
                continue
            local = SessionObservations.to_local(f["date_obs"], tz_name)
            if local is not None:
                timed.append((local, f))
        timed.sort(key=lambda pair: (pair[0], pair[1]["file_path"] or ""))
        run = None
        for local, f in timed:
            exposure = SessionObservations.exposure_bucket(f["exptime_seconds"])
            end = local + timedelta(seconds=f["exptime_seconds"] or 0)
            if (run is None or local - run["end"] > gap or exposure != run["bucket"]
                    or (f["filter_from_file"] or "") != run["file_filter"]):
                run = {"frame_type": frame_type, "files": [], "start": local, "end": end,
                       "exposure": f["exptime_seconds"] or 0, "bucket": exposure,
                       "file_filter": f["filter_from_file"] or ""}
                runs.append(run)
            run["files"].append(f)
            run["end"] = max(run["end"], end)
    return runs


def flat_match_problems(night_files, chosen):
    """Warnings about this night's lights and flats after the choices in
    `chosen` ({file id: correction or None}): light filters with no flats of
    the exact same name (PixInsight's WBPP only pairs exact names), and names
    PixInsight will read differently."""
    def effective(f):
        correction = chosen.get(f["id"], current_correction(f))
        return correction if correction is not None else (f["filter_from_file"] or "")

    lights = {effective(f) for f in night_files if f["frame_type"] == "Light"}
    flats = {effective(f) for f in night_files if f["frame_type"] == "Flat"}
    problems = []
    if flats:
        for name in sorted(lights - flats, key=str.lower):
            close = [flat for flat in flats if flat and SessionObservations.normalize_filter(flat)
                     == SessionObservations.normalize_filter(name)]
            shown = name or "(no filter)"
            if close:
                problems.append(f"The {shown} lights and the {close[0]} flats don't share the exact same name - "
                                "PixInsight only pairs flats with lights of the same filter name.")
            else:
                problems.append(f"No flats on this night have the filter {shown}. In PixInsight these lights use "
                                "another night's flats of that filter, or none.")
    corrected = {c for c in chosen.values() if c}
    for name in sorted(corrected, key=str.lower):
        read_as = ProcessingHandoff.wbpp_filter_name(name)
        if read_as != name:
            problems.append(f"PixInsight will read the filter {name} as {read_as}.")
    return problems


class NightFiltersDialog(WindowPositionMixin, QDialog):
    """Pick a filter for each run of one night's lights and flats. On Save,
    `changes` holds {file id: correction or None (the file's own filter)} for
    the files whose correction changed."""

    WINDOW_POSITION_KEY = "NightFiltersDialog"
    COLUMNS = ["Frame", "Time", "Subs", "Exposure", "In the files", "Filter"]
    FILTER_COLUMN = 5

    def __init__(self, night_date, files, tz_name, filter_choices=(), parent=None):
        super().__init__(parent)
        self.night_files = [f for f in files if f["frame_type"] in RUN_FRAME_TYPES]
        self.tz_name = tz_name
        self.filter_choices = sorted({f for f in filter_choices if f}
                                     | {f["filter_from_file"] for f in self.night_files if f["filter_from_file"]},
                                     key=str.lower)
        # The pending correction per file id - starts as what's saved
        self._chosen = {f["id"]: current_correction(f) for f in self.night_files}
        self._runs = []
        self.changes = {}

        self.setWindowTitle(f"Set Filters - Night of {night_date}")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(720, 460)  # default size the first time this dialog is ever opened
        self._setup_ui()
        self._fill_runs()
        self.setup_window_position()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        hint = QLabel("Use this when the files don't record the filter (a manual filter) or record the wrong one. "
                      "The night's subs are split into runs - a new run starts after a break or when the exposure "
                      "changes, usually where the filter was swapped. Your filters are saved with the session and "
                      "used everywhere, including the Siril and PixInsight handoffs. Flats need the same filter "
                      "names as their lights.")
        hint.setWordWrap(True)
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(hint)

        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel("Start a new run after a break of more than"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(1, 240)
        self.gap_spin.setSuffix(" min")
        self.gap_spin.setValue(QSettings("CosmosCollection", "CosmosCollection").value(
            GAP_SETTING, DEFAULT_GAP_MINUTES, type=int))
        self.gap_spin.valueChanged.connect(self._on_gap_changed)
        gap_row.addWidget(self.gap_spin)
        gap_row.addStretch()
        layout.addLayout(gap_row)

        self.table = _make_table(self.COLUMNS)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(self.FILTER_COLUMN, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.problems_label = QLabel()
        self.problems_label.setWordWrap(True)
        themed_style(self.problems_label, lambda: f"color: {COLORS['warning']};")
        self.problems_label.hide()
        layout.addWidget(self.problems_label)

        buttons = QHBoxLayout()
        clear_btn = QPushButton("Clear All")
        clear_btn.setToolTip("Use the files' own filters for every run on this night")
        clear_btn.clicked.connect(self._clear_all)
        buttons.addWidget(clear_btn)
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)

    def _fill_runs(self):
        self._runs = split_runs(self.night_files, self.tz_name, self.gap_spin.value())
        self.table.setRowCount(len(self._runs))
        for row, run in enumerate(self._runs):
            self.table.setItem(row, 0, _table_item(run["frame_type"], Qt.AlignCenter))
            span = f"{run['start']:%H:%M} – {run['end']:%H:%M}"
            self.table.setItem(row, 1, _table_item(span, Qt.AlignCenter))
            self.table.setItem(row, 2, _table_item(str(len(run["files"])), Qt.AlignCenter))
            self.table.setItem(row, 3, _table_item(format_exposure(run["exposure"]), Qt.AlignCenter))
            self.table.setItem(row, 4, _table_item(run["file_filter"] or "(none)", Qt.AlignCenter))
            self.table.setCellWidget(row, self.FILTER_COLUMN, self._make_combo(run))
        self._update_problems()

    def _make_combo(self, run):
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItem(FROM_FILE_TEXT, _FROM_FILE)
        corrections = {self._chosen[f["id"]] for f in run["files"]}
        if len(corrections) > 1:
            combo.addItem(KEEP_TEXT, _KEEP)
        combo.addItems(self.filter_choices)
        if len(corrections) > 1:
            combo.setCurrentIndex(1)
        else:
            correction = next(iter(corrections))
            if correction is None:
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentText(correction)
        combo.setToolTip(f"'{FROM_FILE_TEXT}' uses the filter the subs' headers record")
        combo.currentTextChanged.connect(lambda _text: self._update_problems())
        return combo

    def _combo_choice(self, row):
        """_FROM_FILE, _KEEP or the filter name typed or picked in a row."""
        text = self.table.cellWidget(row, self.FILTER_COLUMN).currentText().strip()
        if not text or text == FROM_FILE_TEXT:
            return _FROM_FILE
        if text == KEEP_TEXT:
            return _KEEP
        return text

    def _apply_rows(self):
        """The table's choices into self._chosen."""
        for row, run in enumerate(self._runs):
            choice = self._combo_choice(row)
            if choice == _KEEP:
                continue
            for f in run["files"]:
                self._chosen[f["id"]] = None if choice == _FROM_FILE else choice

    def _on_gap_changed(self, value):
        QSettings("CosmosCollection", "CosmosCollection").setValue(GAP_SETTING, value)
        self._apply_rows()
        self._fill_runs()

    def _update_problems(self):
        self._apply_rows()
        problems = flat_match_problems(self.night_files, self._chosen)
        self.problems_label.setText("\n".join(f"⚠ {p}" for p in problems))
        self.problems_label.setVisible(bool(problems))

    def _clear_all(self):
        for row in range(self.table.rowCount()):
            self.table.cellWidget(row, self.FILTER_COLUMN).setCurrentIndex(0)

    def _on_save(self):
        self._apply_rows()
        self.changes = {f["id"]: self._chosen[f["id"]] for f in self.night_files
                        if self._chosen[f["id"]] != current_correction(f)}
        self.accept()
