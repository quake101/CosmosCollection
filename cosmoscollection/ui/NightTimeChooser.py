#!/usr/bin/env python3
"""
The "Time:" / "Direction at:" picker of Best DSO Tonight and the Target List:
Now, Dusk, Midnight, Dawn, Tonight (dusk -> dawn), a custom time or a custom
time frame (NightTimes.DIRECTION_MODES), with the time boxes the choice needs
and a note beside them (e.g. the actual dusk and dawn times).

    self.time_chooser = NightTimeChooser("Time:", SETTING_KEY, DEFAULT, tooltip, "Time tonight")
    self.time_chooser.changed.connect(self._on_time_choice_changed)
    row.addWidget(self.time_chooser)

The choice is saved under the settings key as soon as it changes. changed is
sent right away for the dropdown and once the time boxes stop changing.
"""

from PySide6.QtCore import QTime, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTimeEdit, QWidget

from cosmoscollection.planning.NightTimes import DIRECTION_MODES, load_night_choice, save_night_choice
from cosmoscollection.ui.Theme import COLORS, themed_style

TIME_FORMAT = "HH:mm"
TYPING_PAUSE_MS = 500


class NightTimeChooser(QWidget):
    """A night time choice: mode() and choice() = (mode, at, from, to)."""

    changed = Signal()

    def __init__(self, label, settings_key, default, tooltip, at_tooltip, parent=None):
        super().__init__(parent)
        self.settings_key = settings_key
        mode, at, start, end = load_night_choice(settings_key, default)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        self.mode_combo = QComboBox()
        for key, text in DIRECTION_MODES:
            self.mode_combo.addItem(text, key)
        self.mode_combo.setToolTip(tooltip)
        layout.addWidget(self.mode_combo)

        # The time boxes' changes are passed on once they stop, not on every step
        self._pause_timer = QTimer(self)
        self._pause_timer.setSingleShot(True)
        self._pause_timer.setInterval(TYPING_PAUSE_MS)
        self._pause_timer.timeout.connect(self.changed.emit)

        def time_edit(hhmm, edit_tooltip):
            edit = QTimeEdit(QTime.fromString(hhmm, TIME_FORMAT))
            edit.setDisplayFormat(TIME_FORMAT)
            edit.setToolTip(edit_tooltip)
            edit.timeChanged.connect(lambda _time: self._on_time_changed())
            layout.addWidget(edit)
            return edit

        self.at_edit = time_edit(at, at_tooltip)
        self.from_edit = time_edit(start, "Start of the time frame")
        self.to_label = QLabel("to")
        layout.addWidget(self.to_label)
        self.to_edit = time_edit(end, "End of the time frame (after midnight is fine)")
        self.note_label = QLabel("")
        themed_style(self.note_label, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.note_label)

        self.mode_combo.setCurrentIndex(max(self.mode_combo.findData(mode), 0))
        self.mode_combo.currentIndexChanged.connect(lambda _index: self._on_mode_changed())
        self._show_time_edits()

    def mode(self):
        return self.mode_combo.currentData()

    def choice(self):
        """(mode, at, from, to), times as HH:MM."""
        return (self.mode(), self.at_edit.time().toString(TIME_FORMAT),
                self.from_edit.time().toString(TIME_FORMAT), self.to_edit.time().toString(TIME_FORMAT))

    def set_note(self, text):
        self.note_label.setText(text)

    def _show_time_edits(self):
        mode = self.mode()
        self.at_edit.setVisible(mode == "time")
        for widget in (self.from_edit, self.to_label, self.to_edit):
            widget.setVisible(mode == "frame")

    def _on_mode_changed(self):
        self._pause_timer.stop()
        self._show_time_edits()
        save_night_choice(self.settings_key, self.choice())
        self.changed.emit()

    def _on_time_changed(self):
        save_night_choice(self.settings_key, self.choice())
        self._pause_timer.start()
