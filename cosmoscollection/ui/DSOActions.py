#!/usr/bin/env python3
"""
What can be done with a deep-sky object from any window that lists one: open
it in DSO Details, the Visibility Calculator or the FOV Simulator, plan a
session, add it to (or find it on) the Target List, or send it to NINA - and
the right-click menu that offers all of these.

Every opener takes an object as a plain dict. The keys used, all optional but
name and position:
    name, ra_deg, dec_deg, dsodetailid (its catalog id), designations,
    size_min, size_max (arcminutes), magnitude, constellation, dso_type,
    dso_class, notes, best_months, and for a Target List row its entry's
    target_id and telescope_id.
"""

import functools
import logging

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QWidget

from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.nina.NINAIntegration import NINAIntegration
from cosmoscollection.ui.WindowLifetime import window_is_open

logger = logging.getLogger(__name__)


def _reported(what):
    """Show what went wrong (failed to <what>) instead of raising."""
    def wrap(function):
        @functools.wraps(function)
        def run(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to {what}: {e}", exc_info=True)
                # A function's parent argument, or the window a method belongs to
                parent = kwargs.get("parent", args[1] if len(args) > 1 else None)
                if parent is None and args and isinstance(args[0], QWidget):
                    parent = args[0]
                QMessageBox.critical(parent, "Error", f"Failed to {what}: {e}")
                return None
        return run
    return wrap


def tool_window(class_name):
    """The open window of a one-at-a-time tool ("DSOTargetListWindow",
    "SessionManagerWindow") wherever it was opened from, else a new one - so
    the toolbar, the menus and other windows share one."""
    for widget in QApplication.topLevelWidgets():
        if type(widget).__name__ == class_name and window_is_open(widget):
            return widget
    if class_name == "DSOTargetListWindow":
        from cosmoscollection.planning.DSOTargetList import DSOTargetListWindow
        return DSOTargetListWindow()
    from cosmoscollection.sessions.SessionManager import SessionManagerWindow
    return SessionManagerWindow()


def _has_position(obj):
    return obj.get("ra_deg") is not None and obj.get("dec_deg") is not None


# ---- Looking up ----------------------------------------------------------------

def target_list_entry(obj):
    """The object's Target List entry - the same name, another designation of
    the same object, or the same position - or None."""
    name = (obj.get("name") or "").strip()
    if not name:
        return None
    from cosmoscollection.planning.TargetRows import find_existing_target
    try:
        with DatabaseManager().get_connection() as conn:
            return find_existing_target(conn, name, obj.get("ra_deg"), obj.get("dec_deg"))
    except Exception as e:
        logger.error(f"Error checking the Target List for {name}: {e}")
        return None


def object_sessions(obj):
    """The object's Session Manager sessions (see SessionRows.sessions_for_object)."""
    from cosmoscollection.sessions.SessionRows import sessions_for_object
    try:
        with DatabaseManager().get_connection() as conn:
            return sessions_for_object(conn, obj.get("name", ""), obj.get("ra_deg"), obj.get("dec_deg"),
                                       obj.get("dsodetailid"))
    except Exception as e:
        logger.error(f"Error finding sessions for {obj.get('name')}: {e}")
        return []


def catalog_data(obj):
    """The object's catalog entry (DSOData.load_dso's dict) - by its catalog id,
    shown under the name obj has; else found by designation or position (a
    Target List entry, a session), under the catalog's name. None if not found."""
    from cosmoscollection.app.DSOData import find_full_dso_data, load_dso
    db_manager = DatabaseManager()
    if obj.get("dsodetailid") not in (None, ""):
        with db_manager.get_connection() as conn:
            data = load_dso(conn, obj["dsodetailid"])
        if data is not None:
            if obj.get("name"):
                data["name"] = obj["name"]
            return data
    return find_full_dso_data(db_manager, obj.get("name", ""), obj)


# ---- Opening -------------------------------------------------------------------

@_reported("open DSO Details")
def open_dso_details(obj, parent=None):
    """DSO Details for the object (see catalog_data), or a note that it isn't
    in the catalog. parent: the window it stays in front of, or None."""
    data = catalog_data(obj)
    if data is None:
        QMessageBox.warning(parent, "Object Not Found",
                            f"Could not find {obj.get('name') or 'this object'} in the DSO catalog.")
        return None
    from cosmoscollection.catalog.DSODetail import DSODetailWindow
    window = DSODetailWindow(data, parent)
    window.show()
    window.raise_()
    window.activateWindow()
    return window


@_reported("open the Visibility Calculator")
def open_visibility(obj, parent=None):
    """The Visibility Calculator, calculating for the object."""
    if not _has_position(obj):
        QMessageBox.warning(parent, "No Coordinates", f"{obj.get('name') or 'This object'} has no coordinates.")
        return None
    from cosmoscollection.planning.DSOVisibilityCalculator import DSOVisibilityApp
    window = DSOVisibilityApp()  # Kept alive until closed (WindowLifetime)
    window.dso_input.setText(obj.get("name") or "")
    window.set_dso_coordinates(obj["ra_deg"], obj["dec_deg"])
    window.show()
    window.raise_()
    window.activateWindow()
    QTimer.singleShot(500, window.calculate_visibility)  # Once the window is fully up
    return window


@_reported("open the FOV Simulator")
def open_fov(obj, parent=None, framing=None, session_id=None):
    """The FOV Simulator on the object - with a session's saved framing. Given
    its catalog entry when it has one, for its size and other designations."""
    from cosmoscollection.planning.FOVSimulator import AladinLiteWindow
    window = AladinLiteWindow(catalog_data(obj) or dict(obj), parent, framing=framing, session_id=session_id)
    window.show()
    return window


@_reported("open the Session Manager")
def plan_session(obj, parent=None, target_id=None):
    """A new planned session for the object, linked to its Target List entry."""
    tool_window("SessionManagerWindow").create_session_from_dso({**obj, "id": target_id})


@_reported("open the Session Manager")
def open_session(session_id, parent=None):
    tool_window("SessionManagerWindow").open_and_select_session(session_id)


@_reported("add to the Target List")
def add_to_target_list(obj, parent=None):
    """The Target List, adding the object."""
    window = tool_window("DSOTargetListWindow")
    window.show()
    window.raise_()
    window.activateWindow()
    window.add_target_from_dso(dict(obj))


@_reported("open the Target List")
def open_target_entry(entry, parent=None):
    """The Target List with an entry (target_list_entry's) selected and open."""
    if not tool_window("DSOTargetListWindow").open_and_select_target(entry["name"]):
        QMessageBox.warning(parent, "Not Found", f"Could not find '{entry['name']}' in your target list.")


def send_to_nina(obj, parent=None):
    NINAIntegration.send_to_framing_assistant(obj.get("ra_deg"), obj.get("dec_deg"), obj.get("name") or "Unknown",
                                              parent)


def slew_to(obj, parent=None):
    NINAIntegration.slew_to_coordinates(obj.get("ra_deg"), obj.get("dec_deg"), obj.get("name") or "Unknown", parent)


# ---- The right-click menu --------------------------------------------------------

_SAME = object()


def build_dso_context_menu(parent, obj, extras=None, on_target_list=False, details_parent=_SAME):
    """The right-click menu for an object in a list: DSO Details, Visibility
    Calculator, FOV Simulator, NINA; its sessions and Plan a Session; then Open
    in / Add to Target List (left out on the Target List itself, whose rows
    carry target_id). extras(menu): adds the list's own actions at the end.
    details_parent: the window DSO Details stays in front of (default parent)."""
    menu = QMenu(parent)
    details_parent = parent if details_parent is _SAME else details_parent
    menu.addAction("View DSO Details").triggered.connect(lambda: open_dso_details(obj, details_parent))
    menu.addAction("Visibility Calculator").triggered.connect(lambda: open_visibility(obj, parent))
    menu.addAction("FOV Simulator").triggered.connect(lambda: open_fov(obj, parent))

    if NINAIntegration.is_enabled():
        menu.addSeparator()
        nina_menu = menu.addMenu("NINA")
        nina_menu.addAction("Send to Framing Assistant").triggered.connect(lambda: send_to_nina(obj, parent))
        nina_menu.addAction("Slew to Target").triggered.connect(lambda: slew_to(obj, parent))

    menu.addSeparator()
    if obj.get("target_id") is not None:
        entry = {"id": obj["target_id"], "name": obj.get("name", "")}
    else:
        entry = target_list_entry(obj)
    target_id = entry["id"] if entry else None
    sessions = object_sessions(obj)
    if sessions:
        sessions_menu = menu.addMenu(f"Sessions ({len(sessions)})")
        for session_id, session_name, date, status, _ in sessions:
            sessions_menu.addAction(f"{date or 'No date'} – {session_name} ({status})").triggered.connect(
                lambda checked=False, sid=session_id: open_session(sid, parent))
        sessions_menu.addSeparator()
        plan_action = sessions_menu.addAction("Plan a New Session")
    else:
        plan_action = menu.addAction("Plan a Session")
    plan_action.triggered.connect(lambda: plan_session(obj, parent, target_id))

    if not on_target_list:
        if entry:
            menu.addAction("Open in Target List").triggered.connect(lambda: open_target_entry(entry, parent))
        else:
            menu.addAction("Add to Target List").triggered.connect(lambda: add_to_target_list(obj, parent))
    if extras:
        menu.addSeparator()
        extras(menu)
    return menu


# ---- Target List actions for a window showing one object -----------------------------

class TargetListActionsMixin:
    """Add to / Remove from / Open in Target List for a window about one object
    (DSO Details, the FOV Simulator). The window has self.data (the object's
    dict), calls add_target_list_actions(menu) once, and
    update_target_list_actions() when the object or the list may have changed.
    target_list_data() may add to what the Add Target dialog is given."""

    def add_target_list_actions(self, menu):
        self.add_target_action = QAction("Add to Target List", self)
        self.add_target_action.setToolTip("Add this object to your observing target list")
        self.add_target_action.triggered.connect(lambda: self._add_to_target_list())
        menu.addAction(self.add_target_action)
        self.remove_target_action = QAction("Remove from Target List", self)
        self.remove_target_action.setToolTip("Remove this object from your target list")
        self.remove_target_action.triggered.connect(lambda: self._remove_from_target_list())
        menu.addAction(self.remove_target_action)
        self.open_target_action = QAction("Open in Target List", self)
        self.open_target_action.setToolTip("Open this object's entry in your target list")
        self.open_target_action.triggered.connect(lambda: self._open_in_target_list())
        menu.addAction(self.open_target_action)
        self._show_target_list_actions(False)

    def target_list_data(self):
        return dict(self.data)

    def _show_target_list_actions(self, listed):
        self.add_target_action.setVisible(not listed)
        self.remove_target_action.setVisible(listed)
        self.open_target_action.setVisible(listed)

    def update_target_list_actions(self):
        self._show_target_list_actions(target_list_entry(self.data) is not None)

    @_reported("add to the Target List")
    def _add_to_target_list(self):
        from cosmoscollection.planning.DSOTargetList import AddTargetDialog
        if AddTargetDialog(dso_data=self.target_list_data(), parent=self).exec():
            self.update_target_list_actions()

    @_reported("remove from the Target List")
    def _remove_from_target_list(self):
        # Named as the list has it ("Barnard 33 (Horsehead)"), which may differ from the catalog's name
        entry = target_list_entry(self.data)
        name = (entry or {}).get("name") or self.data.get("name") or "this object"
        if QMessageBox.question(self, "Remove from Target List", f"Remove '{name}' from the target list?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if entry:
            from cosmoscollection.planning.TargetRows import delete_target
            with DatabaseManager().get_connection() as conn:
                delete_target(conn, entry["id"])  # Just this object's entry
        self.update_target_list_actions()
        QMessageBox.information(self, "Removed", f"'{name}' removed from the target list.")

    def _open_in_target_list(self):
        entry = target_list_entry(self.data)
        if entry is None:
            QMessageBox.warning(self, "Not Found", f"Could not find '{self.data.get('name', '')}' in your target list.")
            return
        open_target_entry(entry, self)
