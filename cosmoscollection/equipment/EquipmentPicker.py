"""
EquipmentPicker.py - Dropdowns of the user's equipment, for any window:

- fill_telescope_combo / select_telescope: a telescope for a target or
  session ("Any" or an enabled telescope; a saved disabled one stays shown)
- equipment_combo / combo_choice: what a telescope or camera name (from files,
  images or sessions) is - a piece of the user's equipment, not matched, or
  added as new (the Equipment window's Matches tab, the Import Wizard)
- equipment_filter_combo / fill_equipment_filter: a list's Telescope / Camera
  filter (the gallery, the Session Manager)
- size_rows: even table rows that fit those dropdowns
"""

import logging

from PySide6.QtWidgets import QComboBox

from cosmoscollection.sessions import LibraryScanner

logger = logging.getLogger(__name__)


# ---- A telescope for a target or session ---------------------------------------

def telescope_label(name, aperture, focal_length):
    """'Name (80mm f/6.0)' for a telescope picker."""
    if aperture and focal_length and aperture > 0:
        return f"{name} ({int(aperture)}mm f/{focal_length / aperture:.1f})"
    if aperture:
        return f"{name} ({int(aperture)}mm)"
    return name


def fill_telescope_combo(combo, db_manager):
    """List the enabled telescopes in a picker, after 'Any' (no telescope).
    Each item's data is the telescope's id; select_telescope picks a saved one."""
    combo.clear()
    combo.addItem("Any", None)
    try:
        with db_manager.get_connection() as conn:
            for telescope_id, name, aperture, focal_length in conn.execute(
                    "SELECT id, name, aperture, focal_length FROM usertelescopes WHERE is_active = 1 "
                    "ORDER BY name").fetchall():
                combo.addItem(telescope_label(name, aperture, focal_length), telescope_id)
    except Exception as e:
        logger.error(f"Error loading telescopes: {e}")


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


# ---- What a name is: one of the user's telescopes or cameras ----------------------

# Item data (strings: Qt can't compare Python tuples in findData)
NOT_MATCHED, ADD_NEW = "none", "new"


def _item_data(item_id):
    return f"id:{item_id}"


def equipment_combo(kind, info, chosen_id, not_matched="Not matched - keep the name", add_new=True):
    """A picker of the user's telescopes or cameras (enabled ones first; see
    LibraryScanner.equipment_choices), with chosen_id selected - else the
    first item. info: LibraryScanner.EquipmentInfo.
    not_matched: text of the first item (None: no such item)
    add_new: True for 'Add as a new telescope', other text for that item, or None"""
    combo = QComboBox()
    if not_matched:
        combo.addItem(not_matched, NOT_MATCHED)
    for item_id, name in LibraryScanner.equipment_choices(info, kind):
        combo.addItem(name, _item_data(item_id))
    if add_new:
        combo.addItem(f"Add as a new {kind}" if add_new is True else add_new, ADD_NEW)
    index = combo.findData(_item_data(chosen_id)) if chosen_id is not None else -1
    combo.setCurrentIndex(max(index, 0))
    return combo


def combo_choice(combo):
    """An equipment_combo's choice: an equipment id, None (not matched) or ADD_NEW."""
    choice = combo.currentData()
    if isinstance(choice, str) and choice.startswith("id:"):
        return int(choice[3:])
    return ADD_NEW if choice == ADD_NEW else None


# ---- Filtering a list by its equipment ------------------------------------------

ALL = "All"


def equipment_filter_combo():
    """A telescope or camera filter for a list ('All' until filled)."""
    combo = QComboBox()
    combo.addItem(ALL)
    combo.setMinimumWidth(140)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    return combo


def fill_equipment_filter(combo, names, current):
    """List 'All' and the names in use (sorted, blanks left out) without
    sending a change; current stays chosen if it's still listed. Returns the
    name chosen ('All' if current isn't listed)."""
    names = sorted({name for name in names if name}, key=str.casefold)
    chosen = current if current in names else ALL
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(ALL)
    combo.addItems(names)
    combo.setCurrentText(chosen)
    combo.blockSignals(False)
    combo.view().setMinimumWidth(combo.minimumSizeHint().width())  # The dropdown shows full names
    return chosen


def size_rows(table):
    """Even rows that fit an equipment_combo, and the table no taller than they
    need (the space left goes below)."""
    row_height = max(QComboBox().sizeHint().height() + 6, 26)
    table.verticalHeader().setDefaultSectionSize(row_height)
    for row in range(table.rowCount()):
        table.setRowHeight(row, row_height)
    table.setMaximumHeight(table.horizontalHeader().sizeHint().height() + row_height * table.rowCount()
                           + 2 * table.frameWidth() + 2)
