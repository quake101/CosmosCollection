#!/usr/bin/env python3
"""
Catalog suggestions while an object's name is typed into a QLineEdit (the
Target List's Add Target dialog, the Session Manager's session form):

    self.name_completer = DSONameCompleter(self.name_edit)
    self.name_completer.picked.connect(self._on_dso_picked)  # gets the details dict

Suggestions come from DSONames.search_catalog once typing pauses, so a
designation or a common name ("Andromeda") both find the object.
"""

import logging

from PySide6.QtCore import QObject, QStringListModel, Qt, QTimer, Signal
from PySide6.QtWidgets import QCompleter

from cosmoscollection.catalog.DSONames import search_catalog
from cosmoscollection.core.DatabaseManager import DatabaseManager

logger = logging.getLogger(__name__)

MIN_CHARACTERS = 2
TYPING_PAUSE_MS = 300


class DSONameCompleter(QObject):
    """Suggests catalog objects for a name field and says which one was picked."""

    picked = Signal(object)  # The chosen object's details (see DSONames.search_catalog)

    def __init__(self, line_edit, db_manager=None):
        super().__init__(line_edit)
        self.db_manager = db_manager or DatabaseManager()
        self._found = {}  # label -> details, for the suggestions shown

        self._model = QStringListModel(self)
        self.completer = QCompleter(self._model, line_edit)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.activated.connect(self._on_activated)
        line_edit.setCompleter(self.completer)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(TYPING_PAUSE_MS)
        self._timer.timeout.connect(self._search)
        self._text = ""
        line_edit.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self, text):
        self._text = text.strip()
        if len(self._text) < MIN_CHARACTERS:
            self._timer.stop()
            self._found = {}
            self._model.setStringList([])
            return
        self._timer.start()

    def _search(self):
        try:
            with self.db_manager.get_connection() as conn:
                found = search_catalog(conn, self._text)
        except Exception as e:
            logger.error(f"Error searching the DSO catalog: {e}")
            return
        self._found = dict(found)
        self._model.setStringList([label for label, _ in found])

    def _on_activated(self, label):
        details = self._found.get(label)
        if details:
            self.picked.emit(details)
