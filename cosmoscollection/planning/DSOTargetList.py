#!/usr/bin/env python3
"""
DSO Target List Manager
Allows users to manage their observing target list for deep sky objects
"""

import sys
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal, QStringListModel, QThread, QTime
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QGroupBox, QMessageBox,
                               QHeaderView, QTextEdit, QDialog, QComboBox,
                               QLineEdit, QCheckBox, QDateEdit, QSpinBox, QMenu,
                               QCompleter, QListWidget, QListWidgetItem, QSplitter,
                               QPlainTextEdit, QTimeEdit)
from PySide6.QtGui import QFont, QColor

from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.core.Settings import app_settings
from cosmoscollection.catalog.DSONames import (designation_candidates, display_name, preferred_designation,
                                              suggestion_names, type_code,
                                              type_name)
from cosmoscollection.ui.WindowPositionManager import WindowPositionMixin
from cosmoscollection.ui.WindowLifetime import release_on_close
from cosmoscollection.ui.Theme import COLORS, font_px, theme_manager, themed_style, themed_text
from cosmoscollection.ui import DSOActions
from cosmoscollection.sessions import SessionFileScanner
# The Target List's rows (moved to TargetRows.py, which has no UI imports; kept
# importable from here)
from cosmoscollection.planning.TargetRows import (  # noqa: F401
    SAME_POSITION_DEG, TARGET_COLUMNS, TARGET_PRIORITIES, TARGET_STATUSES, TargetListIndex,
    _standard_name, catalog_object_ids, catalog_object_ids_many, delete_target,
    find_existing_target, has_position, load_target_rows, name_key)
from cosmoscollection.planning.Observer import load_observer, local_zone  # noqa: F401
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PriorityTableWidgetItem(QTableWidgetItem):
    """Custom QTableWidgetItem that sorts priorities correctly"""

    PRIORITY_ORDER = {"Urgent": 4, "High": 3, "Medium": 2, "Low": 1}

    def __init__(self, priority_text):
        super().__init__(priority_text)
        self.priority_value = self.PRIORITY_ORDER.get(priority_text, 0)
        self.setTextAlignment(Qt.AlignCenter)

    def __lt__(self, other):
        """Override less-than operator for proper sorting"""
        if isinstance(other, PriorityTableWidgetItem):
            return self.priority_value < other.priority_value
        return super().__lt__(other)


class AddTargetDialog(QDialog):
    """Dialog for adding a new target to the list"""
    
    def __init__(self, dso_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Target to List")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(500, 400)
        
        self.dso_data = dso_data
        self.db_manager = DatabaseManager()
        self.is_edit_mode = False  # Track if we're editing an existing target
        self.target_id = None  # Store the ID of the target being edited
        self._dso_cache = {}  # Cache for autocomplete results

        # Debounce timer for DSO catalog search
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._do_dso_search)
        self._pending_search_text = ""

        self._setup_ui()
        
        # Pre-fill with DSO data if provided
        if self.dso_data:
            self._populate_from_dso_data()
    
    def _setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()
        
        # DSO Information Group
        dso_group = QGroupBox("DSO Information")
        dso_layout = QVBoxLayout()
        
        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., M 31, NGC 7000, IC 1396")

        # Set up autocomplete for DSO catalog
        self._completer_model = QStringListModel()
        self._completer = QCompleter()
        self._completer.setModel(self._completer_model)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self.name_edit.setCompleter(self._completer)
        self._completer.activated.connect(self._on_dso_selected)
        self.name_edit.textChanged.connect(self._on_name_text_changed)

        name_layout.addWidget(self.name_edit)
        dso_layout.addLayout(name_layout)
        
        # Type and Constellation
        type_constellation_layout = QHBoxLayout()
        type_constellation_layout.addWidget(QLabel("Type:"))
        self.type_edit = QLineEdit()
        type_constellation_layout.addWidget(self.type_edit)
        
        type_constellation_layout.addWidget(QLabel("Constellation:"))
        self.constellation_edit = QLineEdit()
        type_constellation_layout.addWidget(self.constellation_edit)
        dso_layout.addLayout(type_constellation_layout)
        
        # Coordinates
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("RA (deg):"))
        self.ra_edit = QLineEdit()
        coord_layout.addWidget(self.ra_edit)
        
        coord_layout.addWidget(QLabel("Dec (deg):"))
        self.dec_edit = QLineEdit()
        coord_layout.addWidget(self.dec_edit)
        dso_layout.addLayout(coord_layout)
        
        # Magnitude and Size
        mag_size_layout = QHBoxLayout()
        mag_size_layout.addWidget(QLabel("Magnitude:"))
        self.magnitude_edit = QLineEdit()
        coord_layout.addWidget(self.magnitude_edit)
        
        mag_size_layout.addWidget(QLabel("Size ('):"))
        self.size_edit = QLineEdit()
        mag_size_layout.addWidget(self.size_edit)
        dso_layout.addLayout(mag_size_layout)
        
        dso_group.setLayout(dso_layout)
        layout.addWidget(dso_group)
        
        # Target Information Group
        target_group = QGroupBox("Target Information")
        target_layout = QVBoxLayout()
        
        # Priority
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("Priority:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Low", "Medium", "High", "Urgent"])
        self.priority_combo.setCurrentText("Medium")
        priority_layout.addWidget(self.priority_combo)

        # Status
        priority_layout.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Not Observed", "Observed", "Imaged", "Completed"])
        self.status_combo.setCurrentText("Not Observed")
        priority_layout.addWidget(self.status_combo)
        target_layout.addLayout(priority_layout)

        # Telescope
        telescope_layout = QHBoxLayout()
        telescope_layout.addWidget(QLabel("Telescope:"))
        self.telescope_combo = QComboBox()
        self._populate_telescope_combo()
        telescope_layout.addWidget(self.telescope_combo)
        telescope_layout.addStretch()
        target_layout.addLayout(telescope_layout)
        
        # Best months for observing
        months_layout = QHBoxLayout()
        months_layout.addWidget(QLabel("Best Months:"))
        self.months_edit = QLineEdit()
        self.months_edit.setPlaceholderText("e.g., Nov-Feb, Mar-Jun")
        months_layout.addWidget(self.months_edit)
        target_layout.addLayout(months_layout)
        
        # Notes
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setPlaceholderText("Observing notes, equipment recommendations, etc.")
        notes_layout.addWidget(self.notes_edit)
        target_layout.addLayout(notes_layout)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        self.save_btn = QPushButton("Add to Target List")
        self.save_btn.clicked.connect(self._save_target)
        self.save_btn.setDefault(True)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def set_edit_mode(self, target_id):
        """Set the dialog to edit mode, changing the button text"""
        self.is_edit_mode = True
        self.target_id = target_id
        self.save_btn.setText("Save Changes")
    
    def _populate_telescope_combo(self):
        """Populate telescope dropdown with active telescopes"""
        self.telescope_combo.clear()
        self.telescope_combo.addItem("Any", None)  # First item for unassigned

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length
                    FROM usertelescopes
                    WHERE is_active = 1
                    ORDER BY name
                """)
                telescopes = cursor.fetchall()

                for telescope in telescopes:
                    tel_id, name, aperture, focal_length = telescope
                    # Calculate f/ratio if we have both values
                    if aperture and focal_length and aperture > 0:
                        f_ratio = focal_length / aperture
                        display_text = f"{name} ({int(aperture)}mm f/{f_ratio:.1f})"
                    elif aperture:
                        display_text = f"{name} ({int(aperture)}mm)"
                    else:
                        display_text = name
                    self.telescope_combo.addItem(display_text, tel_id)
        except Exception as e:
            logger.error(f"Error loading telescopes: {str(e)}")

    def _populate_from_dso_data(self):
        """Populate dialog fields with DSO data"""
        if not self.dso_data:
            return

        # A new entry is named with its common name, "M 31 (Andromeda Galaxy)"
        # (editing puts the saved name back afterwards)
        self.name_edit.setText(display_name(self.dso_data.get("name", ""), self.dso_data.get("dsodetailid"),
                                            self.dso_data.get("designations")))
        self.type_edit.setText(type_name(self.dso_data.get("dso_type")))
        self.constellation_edit.setText(self.dso_data.get("constellation", ""))

        # Handle numeric fields - only set if value is not None
        ra_deg = self.dso_data.get("ra_deg")
        if ra_deg is not None:
            self.ra_edit.setText(str(ra_deg))

        dec_deg = self.dso_data.get("dec_deg")
        if dec_deg is not None:
            self.dec_edit.setText(str(dec_deg))

        magnitude = self.dso_data.get("magnitude")
        if magnitude is not None:
            self.magnitude_edit.setText(str(magnitude))

        # Format size
        size_min = self.dso_data.get("size_min", 0)
        size_max = self.dso_data.get("size_max", 0)
        if size_min > 0 or size_max > 0:
            self.size_edit.setText(f"{size_min:.1f} x {size_max:.1f}")

        # Populate best months if available
        self.months_edit.setText(self.dso_data.get("best_months", ""))

    def _on_name_text_changed(self, text):
        """Handle text changes in the name field — debounce before searching"""
        text = text.strip()
        if len(text) < 2:
            self._completer_model.setStringList([])
            self._dso_cache.clear()
            return
        self._pending_search_text = text
        self._search_timer.start()

    def _do_dso_search(self):
        """Execute the DSO catalog search after debounce"""
        text = self._pending_search_text
        if len(text) < 2:
            return
        self._search_dso_catalog(text)

    def _search_dso_catalog(self, text):
        """Search the DSO catalog and update completer suggestions"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Try to parse a catalogue prefix (e.g., "M 3", "NGC 70", "IC 13")
                catalogue = None
                designation_part = None
                text_upper = text.upper().strip()

                for prefix in ("NGC", "IC", "M"):
                    if text_upper.startswith(prefix):
                        remainder = text_upper[len(prefix):]
                        # Ensure remainder is empty, starts with space, or a digit
                        if remainder == "" or remainder[0] in (" ", "-") or remainder[0].isdigit():
                            catalogue = prefix
                            designation_part = remainder.strip()
                            break

                if catalogue and designation_part is not None:
                    # Search within a specific catalogue
                    cursor.execute("""
                        SELECT c.catalogue || ' ' || c.designation as name,
                               d.ra, d.dec, d.magnitude,
                               d.sizemin / 60.0 as sizemin,
                               d.sizemax / 60.0 as sizemax,
                               d.constellation, d.dsotype,
                               c.dsodetailid, c.catalogue, c.designation
                        FROM cataloguenr c
                        JOIN dsodetail d ON d.id = c.dsodetailid
                        WHERE c.catalogue = ? AND c.designation LIKE ?
                        ORDER BY CAST(c.designation AS INTEGER), c.designation
                        LIMIT 20
                    """, (catalogue, designation_part + "%"))
                else:
                    # Search across all catalogues
                    cursor.execute("""
                        SELECT c.catalogue || ' ' || c.designation as name,
                               d.ra, d.dec, d.magnitude,
                               d.sizemin / 60.0 as sizemin,
                               d.sizemax / 60.0 as sizemax,
                               d.constellation, d.dsotype,
                               c.dsodetailid, c.catalogue, c.designation
                        FROM cataloguenr c
                        JOIN dsodetail d ON d.id = c.dsodetailid
                        WHERE c.catalogue || ' ' || c.designation LIKE ?
                        ORDER BY c.catalogue, CAST(c.designation AS INTEGER), c.designation
                        LIMIT 20
                    """, ("%" + text + "%",))

                results = cursor.fetchall()
                self._dso_cache.clear()
                names = []
                # One suggestion per object, with its common name; a common
                # name typed in ("Andromeda") finds the object too
                row_of = {str(row[8]): row for row in results}
                for name, dso in suggestion_names(conn, [tuple(row)[8:11] for row in results], text):
                    row = row_of[dso]
                    names.append(name)
                    self._dso_cache[name] = {
                        "ra": row[1],
                        "dec": row[2],
                        "magnitude": row[3],
                        "sizemin": row[4],
                        "sizemax": row[5],
                        "constellation": row[6],
                        "dsotype": row[7],
                    }

                self._completer_model.setStringList(names)

        except Exception as e:
            logger.error(f"Error searching DSO catalog: {str(e)}")

    def _on_dso_selected(self, text):
        """Auto-fill fields when a DSO suggestion is selected"""
        data = self._dso_cache.get(text)
        if not data:
            return

        self.type_edit.setText(type_name(data.get("dsotype")))
        self.constellation_edit.setText(data.get("constellation") or "")

        ra = data.get("ra")
        if ra is not None:
            self.ra_edit.setText(str(round(ra, 6)))

        dec = data.get("dec")
        if dec is not None:
            self.dec_edit.setText(str(round(dec, 6)))

        mag = data.get("magnitude")
        if mag is not None:
            self.magnitude_edit.setText(str(round(mag, 2)))
        else:
            self.magnitude_edit.setText("")

        sizemin = data.get("sizemin") or 0
        sizemax = data.get("sizemax") or 0
        if sizemin > 0 or sizemax > 0:
            self.size_edit.setText(f"{sizemin:.1f} x {sizemax:.1f}")
        else:
            self.size_edit.setText("")

    def _save_target(self):
        """Save the target to the database"""
        try:
            # Validate required fields
            if not self.name_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Name is required.")
                return

            # Helper function to safely convert to float
            def safe_float(text):
                """Convert text to float, handling empty strings and 'None'"""
                text = text.strip()
                if not text or text.lower() == 'none':
                    return 0.0
                return float(text)

            # Create target data
            target_data = {
                "name": self.name_edit.text().strip(),
                "dso_type": type_code(self.type_edit.text()) or self.type_edit.text().strip(),
                "constellation": self.constellation_edit.text().strip(),
                "ra_deg": safe_float(self.ra_edit.text()),
                "dec_deg": safe_float(self.dec_edit.text()),
                "magnitude": safe_float(self.magnitude_edit.text()),
                "size_info": self.size_edit.text().strip(),
                "priority": self.priority_combo.currentText(),
                "status": self.status_combo.currentText(),
                "best_months": self.months_edit.text().strip(),
                "notes": self.notes_edit.toPlainText().strip(),
                "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "telescope_id": self.telescope_combo.currentData()
            }

            # Save to database - either INSERT new or UPDATE existing
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if not (self.is_edit_mode and self.target_id):
                    # Same name/object/position already on the list? (how duplicates
                    # like "M17" and "M 17" got in)
                    existing = find_existing_target(
                        conn, target_data["name"], target_data["ra_deg"], target_data["dec_deg"])
                    if existing:
                        reply = QMessageBox.question(
                            self, "Already on Your Target List",
                            f"\u201c{target_data['name']}\u201d is already on your target list as "
                            f"\u201c{existing['name']}\u201d.\n\nAdd it again anyway?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if reply != QMessageBox.Yes:
                            return

                if self.is_edit_mode and self.target_id:
                    # Update existing record
                    cursor.execute("""
                        UPDATE usertargetlist SET
                            name = ?, dso_type = ?, constellation = ?, ra_deg = ?, dec_deg = ?,
                            magnitude = ?, size_info = ?, priority = ?, status = ?,
                            best_months = ?, notes = ?, telescope_id = ?
                        WHERE id = ?
                    """, (
                        target_data["name"], target_data["dso_type"], target_data["constellation"],
                        target_data["ra_deg"], target_data["dec_deg"], target_data["magnitude"],
                        target_data["size_info"], target_data["priority"], target_data["status"],
                        target_data["best_months"], target_data["notes"], target_data["telescope_id"],
                        self.target_id
                    ))
                    success_message = f"{target_data['name']} has been updated in your target list."
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO usertargetlist (
                            name, dso_type, constellation, ra_deg, dec_deg, magnitude,
                            size_info, priority, status, best_months, notes, date_added, telescope_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        target_data["name"], target_data["dso_type"], target_data["constellation"],
                        target_data["ra_deg"], target_data["dec_deg"], target_data["magnitude"],
                        target_data["size_info"], target_data["priority"], target_data["status"],
                        target_data["best_months"], target_data["notes"], target_data["date_added"],
                        target_data["telescope_id"]
                    ))
                    success_message = f"{target_data['name']} has been added to your target list."
                
                conn.commit()
            
            QMessageBox.information(self, "Success", success_message)
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", "Please enter valid numeric values for coordinates and magnitude.")
        except Exception as e:
            logger.error(f"Error saving target: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save target: {str(e)}")



# ---- Duplicate entries --------------------------------------------------------
# Finds Target List entries that are probably the same object, suggests how to
# merge them, and performs merges/deletes (moving linked sessions along). Also
# the shared "is this already on the list?" check used wherever a target can be
# added (DSODetail, FOVSimulator, BestDSOTonight, the main window), so new duplicates
# aren't created.
#
# - Duplicate: names match once spacing, punctuation and parenthetical common
#   names are ignored (M17 / "M 17"), or the names are two catalog designations
#   of one object (IC 4725 / M 25).
# - Possible: entries within POSSIBLE_MAX_SEPARATION_ARCMIN of each other. Some
#   are genuinely separate objects (NGC 2244 inside the Rosette), so a possible
#   group can be dismissed; it's flagged again only if another entry joins it.

POSSIBLE_MAX_SEPARATION_ARCMIN = 2.0
DISMISSED_SETTING = "target_duplicates_dismissed"


@dataclass
class DuplicateGroup:
    kind: str  # 'duplicate' (confident) or 'possible' (position only)
    reason: str  # e.g. "Same name", "Same object (IC 4725 = M 25)", "1′ apart"
    targets: list = field(default_factory=list)  # target row dicts, oldest first

    @property
    def ids(self):
        return frozenset(t["id"] for t in self.targets)


# ---------------------------------------------------------------- detection

def find_duplicate_groups(conn, targets=None, dismissed=None):
    """Groups of Target List entries that look like the same object.

    Confident groups (same name / same catalog object) come first, then
    'possible' ones (close positions) that haven't been dismissed.
    """
    if targets is None:
        targets = load_target_rows(conn)
    if dismissed is None:
        dismissed = load_dismissed()
    cursor = conn.cursor()
    by_id = {t["id"]: t for t in targets}

    # Union-find over the confident links
    parent = {t["id"]: t["id"] for t in targets}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons = {}  # root pair -> reason, filled as links are made

    def link(a, b, reason):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        reasons.setdefault((min(a, b), max(a, b)), reason)

    keys = {}
    for t in targets:
        key = name_key(t["name"])
        if key:
            if key in keys:
                link(keys[key], t["id"], "Same name")
            else:
                keys[key] = t["id"]

    target_objects = catalog_object_ids_many(cursor, [t["name"] for t in targets])
    objects = {t["id"]: target_objects[t["name"]] for t in targets}
    seen_objects = {}
    for t in targets:
        for obj in objects[t["id"]]:
            other = seen_objects.get(obj)
            if other is not None and name_key(by_id[other]["name"]) != name_key(t["name"]):
                link(other, t["id"], f"Same object ({by_id[other]['name']} = {t['name']})")
            seen_objects.setdefault(obj, t["id"])

    components = {}
    for t in targets:
        components.setdefault(find(t["id"]), []).append(t)
    groups = []
    for members in components.values():
        if len(members) < 2:
            continue
        member_ids = {m["id"] for m in members}
        group_reasons = [r for (a, b), r in reasons.items() if a in member_ids and b in member_ids]
        reason = "Same name" if group_reasons and all(r == "Same name" for r in group_reasons) \
            else next((r for r in group_reasons if r != "Same name"), "Same name")
        groups.append(DuplicateGroup("duplicate", reason, sorted(members, key=lambda m: m["id"])))

    # Possible: close pairs that aren't already one confident group
    positioned = [t for t in targets if has_position(t)]
    possible_pairs = []
    for i, a in enumerate(positioned):
        for b in positioned[i + 1:]:
            if find(a["id"]) == find(b["id"]):
                continue
            separation = SessionFileScanner.angular_separation_deg(
                a["ra_deg"], a["dec_deg"], b["ra_deg"], b["dec_deg"]) * 60
            if separation <= POSSIBLE_MAX_SEPARATION_ARCMIN:
                possible_pairs.append((a, b, separation))
    for a, b, separation in possible_pairs:
        ids = frozenset((a["id"], b["id"]))
        if any(ids <= d for d in dismissed):
            continue
        groups.append(DuplicateGroup("possible", f"{separation:.0f}′ apart" if separation >= 0.5
                                     else "Same position", [a, b]))
    return groups


# ---------------------------------------------------------------- dismissals

def load_dismissed():
    """Dismissed 'possible' groups, as frozensets of target ids."""
    settings = app_settings()
    saved = settings.value(DISMISSED_SETTING, "", type=str) or ""
    dismissed = []
    for part in saved.split(";"):
        ids = {int(i) for i in part.split(",") if i.strip().isdigit()}
        if len(ids) >= 2:
            dismissed.append(frozenset(ids))
    return dismissed


def dismiss_group(group):
    """Remember that a group's entries are different objects."""
    dismissed = load_dismissed()
    dismissed.append(group.ids)
    settings = app_settings()
    settings.setValue(DISMISSED_SETTING, ";".join(",".join(str(i) for i in sorted(d)) for d in dismissed))


# ---------------------------------------------------------------- merging

# Preferred catalogs for a merged name, as when adding targets (DSOTargetList)
PREFERRED_CATALOGS = ("M", "NGC", "IC")


def preferred_duplicate_name(cursor, names, fallback):
    """The best name among duplicates: one carrying a common name ("vdB 152
    (Wolf's Cave Nebula)"), else an M / NGC / IC designation in that order, else
    fallback - in the catalog's own spelling ('M17' -> 'M 17')."""
    names = [n for n in names if n]
    with_common = [n for n in names if "(" in n]
    if with_common:
        return with_common[0]

    def catalog_rank(name):
        candidates = designation_candidates(name)
        catalogue = candidates[0][0].upper() if candidates else ""
        return PREFERRED_CATALOGS.index(catalogue) if catalogue in PREFERRED_CATALOGS else len(PREFERRED_CATALOGS)

    ranked = sorted(names, key=catalog_rank)
    best = ranked[0] if ranked and catalog_rank(ranked[0]) < len(PREFERRED_CATALOGS) else fallback
    standard = _standard_name(cursor, best)
    return standard if standard and name_key(standard) == name_key(best) else best


def _rank(value, order):
    return order.index(value) if value in order else -1


def _filled(value):
    return value not in (None, "", 0, 0.0)


def suggest_merge(conn, group):
    """(keep_id, merged field values) - a best guess the user can change.

    Keeps the entry with the most linked sessions (then the most filled-in,
    then the oldest); takes the most advanced status, the highest priority,
    every entry's notes, and the earliest dates.
    """
    targets = group.targets
    keep = max(targets, key=lambda t: (t["session_count"],
                                       sum(_filled(t.get(c)) for c in TARGET_COLUMNS),
                                       -t["id"]))
    merged = {c: keep.get(c) for c in TARGET_COLUMNS if c not in ("id", "created_date")}

    # Blank fields filled from the other entries
    for column in ("dso_type", "constellation", "magnitude", "size_info", "best_months", "telescope_id"):
        if not _filled(merged.get(column)):
            merged[column] = next((t[column] for t in targets if _filled(t.get(column))), merged.get(column))
    if not has_position(keep):
        source = next((t for t in targets if has_position(t)), None)
        if source:
            merged["ra_deg"], merged["dec_deg"] = source["ra_deg"], source["dec_deg"]

    merged["name"] = preferred_duplicate_name(conn.cursor(), [t["name"] for t in targets], keep["name"])

    merged["status"] = max((t["status"] for t in targets), key=lambda s: _rank(s, TARGET_STATUSES))
    merged["priority"] = max((t["priority"] for t in targets), key=lambda p: _rank(p, TARGET_PRIORITIES))
    notes = [t["notes"].strip() for t in targets if (t.get("notes") or "").strip()]
    merged["notes"] = "\n\n".join(dict.fromkeys(notes))
    dates_added = [t["date_added"] for t in targets if t.get("date_added")]
    merged["date_added"] = min(dates_added) if dates_added else keep.get("date_added")
    observed = [t["date_observed"] for t in targets if t.get("date_observed")]
    merged["date_observed"] = min(observed) if observed else None
    return keep["id"], merged


def merge_targets(conn, keep_id, other_ids, merged):
    """Save the merged values on keep_id, move the others' linked sessions to it
    and delete the others - all or nothing. Returns the number of sessions moved."""
    cursor = conn.cursor()
    try:
        columns = [c for c in TARGET_COLUMNS if c not in ("id", "created_date") and c in merged]
        cursor.execute(f"UPDATE usertargetlist SET {', '.join(f'{c} = ?' for c in columns)} WHERE id = ?",
                       [merged[c] for c in columns] + [keep_id])
        placeholders = ",".join("?" * len(other_ids))
        moved = 0
        try:
            cursor.execute(f"UPDATE usersessions SET target_id = ? WHERE target_id IN ({placeholders})",
                           [keep_id, *other_ids])
            moved = cursor.rowcount
        except Exception as e:
            if "no such table" not in str(e):
                raise
        cursor.execute(f"DELETE FROM usertargetlist WHERE id IN ({placeholders})", list(other_ids))
        conn.commit()
        return moved
    except Exception:
        conn.rollback()
        raise


# ================================================================ review dialog

class DuplicateReviewDialog(WindowPositionMixin, QDialog):
    """Review possible duplicate Target List entries: merge them (pre-filled,
    every field changeable), delete one, edit one, or - for position-only
    matches - mark them as different objects."""

    WINDOW_POSITION_KEY = "TargetDuplicateReview"
    targets_changed = Signal()

    # (label, column) rows of the comparison; "position" is RA + Dec together
    ROWS = (("Name", "name"), ("Linked sessions", "session_count"),
            ("Type", "dso_type"), ("Constellation", "constellation"),
            ("Position", "position"), ("Magnitude", "magnitude"), ("Size", "size_info"),
            ("Priority", "priority"), ("Status", "status"), ("Telescope", "telescope_id"),
            ("Best months", "best_months"), ("Date added", "date_added"),
            ("Date observed", "date_observed"), ("Notes", "notes"))

    def __init__(self, db_manager, edit_callback=None, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.edit_callback = edit_callback  # callable(target dict) that opens the Edit Target dialog
        self.groups = []
        self._group = None
        self._keep_id = None
        self._merged_widgets = {}
        self.setWindowTitle("Possible Duplicate Targets")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        self.resize(980, 640)  # default size the first time this dialog is ever opened
        self._telescopes = self._load_telescopes()
        self._setup_ui()
        self.setup_window_position()
        self._refresh()

    def _load_telescopes(self):
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM usertelescopes")
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.debug(f"Couldn't load telescopes: {e}")
            return {}

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel("These entries look like the same object. Merge them into one entry (the merged "
                       "values are pre-filled; change any of them first), delete one, or edit one.")
        intro.setWordWrap(True)
        themed_style(intro, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(intro)

        splitter = QSplitter(Qt.Horizontal)
        self.groups_list = QListWidget()
        self.groups_list.currentRowChanged.connect(self._show_group)
        splitter.addWidget(self.groups_list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.reason_label = QLabel("")
        self.reason_label.setWordWrap(True)
        bold = QFont(self.reason_label.font())
        bold.setBold(True)
        self.reason_label.setFont(bold)
        right_layout.addWidget(self.reason_label)

        self.compare_table = QTableWidget()
        self.compare_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.compare_table.setSelectionMode(QTableWidget.NoSelection)
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.compare_table, 1)

        actions = QHBoxLayout()
        self.merge_btn = QPushButton("Merge")
        self.merge_btn.setDefault(True)
        self.merge_btn.setToolTip("Combine these entries into one, using the Merged result column")
        self.merge_btn.clicked.connect(self._merge)
        actions.addWidget(self.merge_btn)
        self.delete_btn = QPushButton("Delete...")
        self.delete_btn.setToolTip("Delete one of these entries")
        actions.addWidget(self.delete_btn)
        self.edit_btn = QPushButton("Edit...")
        self.edit_btn.setToolTip("Edit one of these entries, e.g. if its name or position is wrong")
        actions.addWidget(self.edit_btn)
        self.dismiss_btn = QPushButton("Not a Duplicate")
        self.dismiss_btn.setToolTip("These are different objects that happen to be close together - "
                                    "don't flag them again")
        self.dismiss_btn.clicked.connect(self._dismiss)
        actions.addWidget(self.dismiss_btn)
        actions.addStretch()
        right_layout.addLayout(actions)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        bottom.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    # ---------------------------------------------------------------- groups

    def _refresh(self, keep_row=0):
        """Re-detect duplicates (after a change) and show the group at keep_row."""
        try:
            with self.db_manager.get_connection() as conn:
                self.groups = find_duplicate_groups(conn)
        except Exception as e:
            logger.error(f"Error finding duplicate targets: {e}")
            self.groups = []
        self.groups_list.blockSignals(True)
        self.groups_list.clear()
        for group in self.groups:
            names = " / ".join(t["name"] for t in group.targets)
            text = names if group.kind == "duplicate" else f"{names}  (maybe)"
            item = QListWidgetItem(text)
            item.setToolTip(group.reason)
            if group.kind == "possible":
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
            self.groups_list.addItem(item)
        self.groups_list.blockSignals(False)
        if self.groups:
            self.groups_list.setCurrentRow(min(max(keep_row, 0), len(self.groups) - 1))
            self._show_group(self.groups_list.currentRow())
        else:
            self._group = None
            self.reason_label.setText("No duplicates left - your target list is clean.")
            self.compare_table.clear()
            self.compare_table.setRowCount(0)
            self.compare_table.setColumnCount(0)
            for button in (self.merge_btn, self.delete_btn, self.edit_btn, self.dismiss_btn):
                button.setEnabled(False)
            self.dismiss_btn.setVisible(False)

    def _show_group(self, row):
        if row < 0 or row >= len(self.groups):
            return
        group = self.groups[row]
        self._group = group
        if group.kind == "possible":
            self.reason_label.setText(f"Maybe the same object: {group.reason}. Close objects can be "
                                      "genuinely different (e.g. a cluster inside a nebula).")
        else:
            self.reason_label.setText(f"Duplicate: {group.reason}")
        with self.db_manager.get_connection() as conn:
            self._keep_id, merged = suggest_merge(conn, group)
        self._fill_table(group, merged)
        for button in (self.merge_btn, self.delete_btn, self.edit_btn):
            button.setEnabled(True)
        self.dismiss_btn.setVisible(group.kind == "possible")
        self.dismiss_btn.setEnabled(True)
        self._build_entry_menus(group)

    def _build_entry_menus(self, group):
        delete_menu = QMenu(self)
        edit_menu = QMenu(self)
        for target in group.targets:
            sessions = target["session_count"]
            suffix = f"  ({sessions} linked session{'s' if sessions != 1 else ''})" if sessions else ""
            delete_menu.addAction(f"Delete “{target['name']}”{suffix}",
                                  lambda t=target: self._delete(t))
            edit_menu.addAction(f"Edit “{target['name']}”", lambda t=target: self._edit(t))
        self.delete_btn.setMenu(delete_menu)
        self.edit_btn.setMenu(edit_menu)

    # ---------------------------------------------------------------- comparison

    def _display(self, target, column):
        if column == "position":
            if not has_position(target):
                return ""
            return f"RA {target['ra_deg']:.4f}°, Dec {target['dec_deg']:+.4f}°"
        value = target.get(column)
        if column == "telescope_id":
            return self._telescopes.get(value, "Any") if value is not None else "Any"
        if column == "magnitude":
            return f"{value:g}" if isinstance(value, (int, float)) and value else ""
        if column == "session_count":
            return str(value or 0)
        return "" if value is None else str(value)

    def _fill_table(self, group, merged):
        targets = group.targets
        table = self.compare_table
        table.clear()
        table.setRowCount(len(self.ROWS))
        table.setColumnCount(len(targets) + 1)
        table.setHorizontalHeaderLabels([t["name"] for t in targets] + ["Merged result"])
        table.setVerticalHeaderLabels([label for label, _ in self.ROWS])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # (reset by setColumnCount)
        self._merged_widgets = {}
        merged_col = len(targets)

        for row, (_label, column) in enumerate(self.ROWS):
            values = [self._display(t, column) for t in targets]
            differs = len(set(values)) > 1  # Differences in bold
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if differs:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(row, col, item)
            widget = self._merged_widget(column, targets, merged)
            if widget is not None:
                table.setCellWidget(row, merged_col, widget)
                self._merged_widgets[column] = widget
            else:
                total = sum(t["session_count"] for t in targets)
                table.setItem(row, merged_col, QTableWidgetItem(f"All {total} move here" if total else "0"))
        notes_row = [c for _, c in self.ROWS].index("notes")
        table.setRowHeight(notes_row, 60)

    def _merged_widget(self, column, targets, merged):
        """Editable control for one merged field, set to the suggestion."""
        if column == "session_count":
            return None
        if column == "notes":
            return QPlainTextEdit(merged.get("notes") or "")
        combo = QComboBox()
        if column == "priority":
            options = [(p, p) for p in TARGET_PRIORITIES]
            current = merged.get("priority")
        elif column == "status":
            options = [(s, s) for s in TARGET_STATUSES]
            current = merged.get("status")
        elif column == "telescope_id":
            ids = list(dict.fromkeys([t["telescope_id"] for t in targets] + [merged.get("telescope_id")]))
            options = [(self._telescopes.get(i, "Any") if i is not None else "Any", i) for i in ids]
            current = merged.get("telescope_id")
        elif column == "position":
            options = [(self._display(t, "position"), (t["ra_deg"], t["dec_deg"]))
                       for t in targets if has_position(t)]
            current = (merged.get("ra_deg"), merged.get("dec_deg"))
            if not options:
                options = [("", current)]
        else:
            values = [t.get(column) for t in targets] + [merged.get(column)]
            if column == "name":
                with self.db_manager.get_connection() as conn:
                    standard = _standard_name(conn.cursor(), merged.get("name"))
                if standard:
                    values.append(standard)
                combo.setEditable(True)  # A name of your own is fine too
            options = [(self._display({column: v}, column), v) for v in dict.fromkeys(values)]
            current = merged.get(column)
        seen = set()
        for text, data in options:
            if (text, repr(data)) in seen:
                continue
            seen.add((text, repr(data)))
            combo.addItem(text, data)
        for i in range(combo.count()):
            if combo.itemData(i) == current:
                combo.setCurrentIndex(i)
                break
        return combo

    def _merged_values(self):
        """The Merged result column's current values, as usertargetlist columns."""
        values = {}
        for column, widget in self._merged_widgets.items():
            if isinstance(widget, QPlainTextEdit):
                values[column] = widget.toPlainText().strip()
            elif column == "position":
                ra_dec = widget.currentData()
                values["ra_deg"], values["dec_deg"] = ra_dec if ra_dec else (None, None)
            elif column == "name":
                values["name"] = widget.currentText().strip()
            else:
                values[column] = widget.currentData()
        return values

    # ---------------------------------------------------------------- actions

    def _merge(self):
        group = self._group
        if not group:
            return
        values = self._merged_values()
        if not values.get("name"):
            QMessageBox.warning(self, "Name Required", "The merged entry needs a name.")
            return
        with self.db_manager.get_connection() as conn:
            _keep, merged = suggest_merge(conn, group)
        merged.update(values)
        others = [t for t in group.targets if t["id"] != self._keep_id]
        sessions = sum(t["session_count"] for t in others)
        names = ", ".join(f"“{t['name']}”" for t in group.targets)
        message = f"Merge {names} into one entry named “{values['name']}”?"
        if sessions:
            message += f"\n\n{sessions} linked session{'s' if sessions != 1 else ''} will move to it."
        message += "\n\nThe other entr" + ("ies" if len(others) > 1 else "y") + " will be deleted."
        if QMessageBox.question(self, "Merge Targets", message,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        try:
            with self.db_manager.get_connection() as conn:
                merge_targets(conn, self._keep_id, [t["id"] for t in others], merged)
        except Exception as e:
            logger.error(f"Error merging targets: {e}")
            QMessageBox.critical(self, "Error", f"Failed to merge targets: {e}")
            return
        self.targets_changed.emit()
        self._refresh(self.groups_list.currentRow())

    def _delete(self, target):
        group = self._group
        others = [t for t in group.targets if t["id"] != target["id"]] if group else []
        move_to = None
        count = target["session_count"]
        if count and others:
            other = others[0]
            answer = QMessageBox.question(
                self, "Linked Sessions",
                f"“{target['name']}” has {count} linked session{'s' if count != 1 else ''}.\n\n"
                f"Move {'them' if count != 1 else 'it'} to “{other['name']}”? "
                "(No leaves them unlinked.)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes:
                move_to = other["id"]
        elif QMessageBox.question(self, "Delete Target",
                                  f"Delete “{target['name']}” from your target list?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            with self.db_manager.get_connection() as conn:
                delete_target(conn, target["id"], move_to)
        except Exception as e:
            logger.error(f"Error deleting target: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete target: {e}")
            return
        self.targets_changed.emit()
        self._refresh(self.groups_list.currentRow())

    def _edit(self, target):
        if not self.edit_callback:
            return
        self.edit_callback(dict(target))
        self.targets_changed.emit()
        self._refresh(self.groups_list.currentRow())

    def _dismiss(self):
        group = self._group
        if not group or group.kind != "possible":
            return
        dismiss_group(group)
        self.targets_changed.emit()
        self._refresh(self.groups_list.currentRow())


# ---- Direction column --------------------------------------------------------
# Where each target is in the sky at a chosen time (Now, astronomical dusk,
# midnight, dawn, or a custom time) or over a time frame (dusk to dawn, or a
# custom one, shown as start -> end). All targets are computed together in one
# astropy transform per time, in a worker thread, so the window opens first and
# the column fills in. Targets below the location's custom horizon (or 0°) are
# marked.

from cosmoscollection.planning.NightTimes import (  # noqa: E402,F401
    DIRECTION_MODES, _twilight, compass_direction, compute_directions,
    direction_cells, resolve_direction_times)
DIRECTION_SETTING = "target_list_direction"  # "mode|at|from|to", times as HH:MM
DIRECTION_DEFAULT = ("midnight", "22:00", "21:00", "01:00")
NOW_REFRESH_MS = 5 * 60 * 1000  # "Now" mode recalculates every 5 minutes


def load_direction_setting():
    """(mode, at, start, end) as saved for the Direction column."""
    settings = app_settings()
    parts = (settings.value(DIRECTION_SETTING, "", type=str) or "").split("|")
    if len(parts) != 4 or parts[0] not in dict(DIRECTION_MODES):
        return DIRECTION_DEFAULT
    return tuple(parts)


def save_direction_setting(mode, at, start, end):
    settings = app_settings()
    settings.setValue(DIRECTION_SETTING, "|".join((mode, at, start, end)))


class DirectionWorker(QThread):
    """Computes the Direction column off the UI thread."""

    result_ready = Signal(int, object)  # generation, compute_directions() result or {'error': str}

    def __init__(self, generation, args):
        super().__init__()
        self._generation = generation
        self._args = args

    def run(self):
        try:
            result = compute_directions(*self._args)
        except Exception as e:
            logger.error(f"Error calculating target directions: {e}", exc_info=True)
            result = {"error": str(e)}
        self.result_ready.emit(self._generation, result)


# Running workers, referenced until they finish (a running QThread that's
# garbage collected - e.g. with its window - aborts the app)
_direction_workers = set()


class DSOTargetListWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "DSOTargetList"
    """Main window for DSO target list management"""
    
    def __init__(self):
        super().__init__()
        release_on_close(self)  # Freed once closed (see WindowLifetime)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("DSO Target List - Cosmos Collection")
        self.resize(1210, 850)
        self.setup_window_position()

        self.db_manager = DatabaseManager()
        self.targets_data = []
        # Direction column: results by target id (text, tooltip, below throughout),
        # and a generation number so a stale worker's result is ignored
        self._direction_cells = {}
        self._direction_generation = 0
        self._init_ui()
        theme_manager().theme_changed.connect(self._apply_direction_cells)
        self._load_targets()
    
    def _populate_telescope_filter(self):
        """Populate telescope filter dropdown with all telescopes (including inactive)"""
        self.telescope_filter.clear()
        self.telescope_filter.addItem("All", "all")
        self.telescope_filter.addItem("Unassigned", "unassigned")

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Include all telescopes (even inactive) since targets may reference them
                cursor.execute("""
                    SELECT id, name, is_active
                    FROM usertelescopes
                    ORDER BY name
                """)
                telescopes = cursor.fetchall()

                for telescope in telescopes:
                    tel_id, name, is_active = telescope
                    display_text = name if is_active else f"{name} (inactive)"
                    self.telescope_filter.addItem(display_text, tel_id)
        except Exception as e:
            logger.error(f"Error loading telescopes for filter: {str(e)}")

    def _init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        #header_label = QLabel("DSO Target List")
        #header_label.setAlignment(Qt.AlignCenter)
        #themed_style(header_label, lambda: f"font-size: {font_px(18)}; font-weight: bold; margin: 10px;")
        #main_layout.addWidget(header_label)
        
        # Control panel
        control_group = QGroupBox("Target List Management")
        control_layout = QVBoxLayout()

        # Search row
        search_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search name, type, constellation...")
        self.search_box.setFixedWidth(280)
        self.search_box.textChanged.connect(self._filter_targets)
        self.search_box.setClearButtonEnabled(True)
        search_row.addWidget(QLabel("Search:"))
        search_row.addWidget(self.search_box)
        search_row.addSpacing(20)
        self._build_direction_controls(search_row)
        search_row.addStretch()
        control_layout.addLayout(search_row)

        # Buttons and filter row
        buttons_row = QHBoxLayout()

        # Add target button
        add_target_btn = QPushButton("Add New Target")
        add_target_btn.clicked.connect(self._add_new_target)
        buttons_row.addWidget(add_target_btn)

        # Edit target button
        self.edit_target_btn = QPushButton("Edit Selected")
        self.edit_target_btn.clicked.connect(self._edit_selected_target)
        self.edit_target_btn.setEnabled(False)
        buttons_row.addWidget(self.edit_target_btn)

        # View details button
        self.view_details_btn = QPushButton("View Details")
        self.view_details_btn.clicked.connect(self._view_target_details)
        self.view_details_btn.setEnabled(False)
        self.view_details_btn.setToolTip("Open detailed view of selected target")
        buttons_row.addWidget(self.view_details_btn)

        # Remove target button
        self.remove_target_btn = QPushButton("Remove Selected")
        self.remove_target_btn.clicked.connect(self._remove_selected_target)
        self.remove_target_btn.setEnabled(False)
        buttons_row.addWidget(self.remove_target_btn)

        # Best DSO Tonight button
        best_tonight_btn = QPushButton("Best DSO Tonight")
        best_tonight_btn.clicked.connect(self._open_best_dso_tonight)
        best_tonight_btn.setToolTip("Open Best DSO Tonight window to find the best objects to observe tonight")
        buttons_row.addWidget(best_tonight_btn)

        buttons_row.addStretch()

        # Filter controls
        buttons_row.addWidget(QLabel("Filter by Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Not Observed", "Observed", "Imaged", "Completed"])
        self.status_filter.currentTextChanged.connect(self._filter_targets)
        buttons_row.addWidget(self.status_filter)

        buttons_row.addWidget(QLabel("Filter by Priority:"))
        self.priority_filter = QComboBox()
        self.priority_filter.addItems(["All", "Low", "Medium", "High", "Urgent"])
        self.priority_filter.currentTextChanged.connect(self._filter_targets)
        buttons_row.addWidget(self.priority_filter)

        buttons_row.addWidget(QLabel("Telescope:"))
        self.telescope_filter = QComboBox()
        self._populate_telescope_filter()
        self.telescope_filter.currentIndexChanged.connect(self._filter_targets)
        buttons_row.addWidget(self.telescope_filter)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_targets)
        buttons_row.addWidget(refresh_btn)

        control_layout.addLayout(buttons_row)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Targets table
        targets_group = QGroupBox("Target List")
        targets_layout = QVBoxLayout()
        
        self.targets_table = QTableWidget()
        self.targets_table.setColumnCount(11)
        self.targets_table.setHorizontalHeaderLabels([
            "Name", "Type", "Constellation", "Magnitude", "Size",
            "Priority", "Status", "Telescope", "Direction", "Best Months", "Date Added"
        ])

        # Enable sorting and disable editing
        self.targets_table.setSortingEnabled(True)

        # Set column widths - Allow manual resizing
        header = self.targets_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name column autosizes to content
        for col in range(1, 11):
            header.setSectionResizeMode(col, QHeaderView.Interactive)  # Other columns allow manual resizing

        # Set initial default widths for manually resizable columns
        self.targets_table.setColumnWidth(1, 120)  # Type
        self.targets_table.setColumnWidth(2, 100)  # Constellation
        self.targets_table.setColumnWidth(3, 90)   # Magnitude
        self.targets_table.setColumnWidth(4, 80)   # Size
        self.targets_table.setColumnWidth(5, 90)   # Priority
        self.targets_table.setColumnWidth(6, 100)  # Status
        self.targets_table.setColumnWidth(7, 120)  # Telescope
        self.targets_table.setColumnWidth(8, 170)  # Direction (room for "E (below) → SE")
        self.targets_table.setColumnWidth(9, 150)  # Best Months
        self.targets_table.setColumnWidth(10, 100) # Date Added
        
        self.targets_table.setAlternatingRowColors(True)
        self.targets_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.targets_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Disable cell editing
        self.targets_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.targets_table.itemDoubleClicked.connect(self._edit_selected_target)

        # Enable context menu
        self.targets_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.targets_table.customContextMenuRequested.connect(self._show_context_menu)
        
        targets_layout.addWidget(self.targets_table)
        targets_group.setLayout(targets_layout)
        main_layout.addWidget(targets_group)
        
        # Status bar, with a link to review possible duplicate entries when there are any
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_row.addWidget(self.status_label)
        self.duplicates_label = QLabel("")
        self.duplicates_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.duplicates_label.setToolTip("Some entries look like the same object - review them to merge, "
                                         "delete or fix")
        self.duplicates_label.linkActivated.connect(lambda _link: self._review_duplicates())
        self.duplicates_label.hide()
        status_row.addWidget(self.duplicates_label)
        status_row.addStretch()
        main_layout.addLayout(status_row)
    
    def _build_direction_controls(self, row):
        """"Direction at:" - which time (or time frame) the Direction column shows."""
        mode, at, start, end = load_direction_setting()
        row.addWidget(QLabel("Direction at:"))
        self.direction_mode_combo = QComboBox()
        for key, label in DIRECTION_MODES:
            self.direction_mode_combo.addItem(label, key)
        self.direction_mode_combo.setToolTip(
            "When the Direction column shows each target's compass direction.\n"
            "Dusk and dawn are astronomical twilight (sun 18° below the horizon).\n"
            "Tonight means the coming night - or, before dawn, the night you're in.")
        row.addWidget(self.direction_mode_combo)

        def time_edit(hhmm, tooltip):
            edit = QTimeEdit(QTime.fromString(hhmm, "HH:mm"))
            edit.setDisplayFormat("HH:mm")
            edit.setToolTip(tooltip)
            edit.timeChanged.connect(lambda _t: self._direction_timer.start())
            return edit

        self.direction_at_edit = time_edit(at, "Time tonight to show directions for")
        row.addWidget(self.direction_at_edit)
        self.direction_from_edit = time_edit(start, "Start of the time frame")
        row.addWidget(self.direction_from_edit)
        self.direction_to_label = QLabel("to")
        row.addWidget(self.direction_to_label)
        self.direction_to_edit = time_edit(end, "End of the time frame (after midnight is fine)")
        row.addWidget(self.direction_to_edit)
        self.direction_note_label = QLabel("")
        themed_style(self.direction_note_label, lambda: f"color: {COLORS['text_secondary']};")
        row.addWidget(self.direction_note_label)

        # Time edits are applied once they stop changing, not on every step
        self._direction_timer = QTimer(self)
        self._direction_timer.setSingleShot(True)
        self._direction_timer.setInterval(500)
        self._direction_timer.timeout.connect(self._on_direction_setting_changed)
        # "Now" moves on: recalculate while the window is open
        self._direction_now_timer = QTimer(self)
        self._direction_now_timer.setInterval(NOW_REFRESH_MS)
        self._direction_now_timer.timeout.connect(self._update_directions)

        index = self.direction_mode_combo.findData(mode)
        self.direction_mode_combo.setCurrentIndex(max(index, 0))
        self.direction_mode_combo.currentIndexChanged.connect(lambda _i: self._on_direction_setting_changed())
        self._show_direction_time_edits()

    def _direction_choice(self):
        """(mode, at, from, to) currently chosen."""
        return (self.direction_mode_combo.currentData(),
                self.direction_at_edit.time().toString("HH:mm"),
                self.direction_from_edit.time().toString("HH:mm"),
                self.direction_to_edit.time().toString("HH:mm"))

    def _show_direction_time_edits(self):
        mode = self.direction_mode_combo.currentData()
        self.direction_at_edit.setVisible(mode == "time")
        for widget in (self.direction_from_edit, self.direction_to_label, self.direction_to_edit):
            widget.setVisible(mode == "frame")
        if mode == "now":
            self._direction_now_timer.start()
        else:
            self._direction_now_timer.stop()

    def _on_direction_setting_changed(self):
        self._show_direction_time_edits()
        save_direction_setting(*self._direction_choice())
        self._update_directions()

    def _update_directions(self):
        """Recalculate the Direction column in the background."""
        self._direction_generation += 1
        try:
            with self.db_manager.get_connection() as conn:
                observer = load_observer(conn)
                horizon = None
                if observer:
                    from cosmoscollection.planning.HorizonProfile import load_active_horizon
                    horizon = load_active_horizon(conn)
        except Exception as e:
            logger.error(f"Error loading location for directions: {e}")
            observer = None
        if not observer:
            self._direction_cells = {t["id"]: ("Location not set", "Set your location in Settings", False)
                                     for t in self.targets_data}
            self.direction_note_label.setText("")
            self._apply_direction_cells()
            return
        targets = [(t["id"], t["ra_deg"], t["dec_deg"]) for t in self.targets_data if has_position(t)]
        self.direction_note_label.setText("Calculating...")
        lat, lon, tz_name = observer
        worker = DirectionWorker(self._direction_generation,
                                 (targets, lat, lon, tz_name, horizon, *self._direction_choice()))
        worker.result_ready.connect(self._on_directions_ready)
        _direction_workers.add(worker)
        worker.finished.connect(lambda: _direction_workers.discard(worker))
        worker.start()

    def _on_directions_ready(self, generation, result):
        if generation != self._direction_generation:
            return  # A newer calculation is on its way
        if "error" in result:
            self.direction_note_label.setText("Couldn't calculate directions")
            return
        self._direction_cells = result["cells"]
        times = result["times"]
        when = " \u2192 ".join(f"{t:%H:%M}" for _label, t in times)
        mode = self.direction_mode_combo.currentData()
        note = {"now": f"(as of {when})", "dusk": f"({when})", "dawn": f"({when})",
                "night": f"({when})"}.get(mode, "")
        if result.get("note"):
            note = f"{note} - {result['note']}" if note else result["note"]
        self.direction_note_label.setText(note)
        self._apply_direction_cells()

    def _apply_direction_cells(self):
        """Fill the Direction column from the latest results."""
        table = self.targets_table
        sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)  # Rows would move while being updated
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            item = table.item(row, 8)
            target = name_item.data(Qt.UserRole) if name_item else None
            if not target or not item:
                continue
            if not has_position(target):
                text, tooltip, below = "No coordinates", "", False
            else:
                text, tooltip, below = self._direction_cells.get(target["id"], ("\u2026", "Calculating...", False))
            item.setText(text)
            item.setToolTip(tooltip)
            item.setForeground(QColor(COLORS['text_disabled'] if below else COLORS['text']))
        table.setSortingEnabled(sorting)

    def _add_new_target(self):
        """Add a new target to the list"""
        dialog = AddTargetDialog(parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_targets()
    
    def _edit_selected_target(self):
        """Edit the selected target"""
        current_row = self.targets_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a target to edit.")
            return

        # Get target data from the name item (column 0) to handle sorting
        name_item = self.targets_table.item(current_row, 0)
        if not name_item:
            return

        target_data = name_item.data(Qt.UserRole)
        dialog = AddTargetDialog(dso_data=target_data, parent=self)
        dialog.setWindowTitle("Edit Target")
        dialog.set_edit_mode(target_data["id"])  # Change button text to "Save Changes" and set target ID
        
        # Pre-populate with target data
        dialog.name_edit.setText(target_data.get("name", ""))
        dialog.type_edit.setText(type_name(target_data.get("dso_type")))
        dialog.constellation_edit.setText(target_data.get("constellation", ""))

        # Handle numeric fields - only set if value is not None
        ra_deg = target_data.get("ra_deg")
        if ra_deg is not None:
            dialog.ra_edit.setText(str(ra_deg))

        dec_deg = target_data.get("dec_deg")
        if dec_deg is not None:
            dialog.dec_edit.setText(str(dec_deg))

        magnitude = target_data.get("magnitude")
        if magnitude is not None:
            dialog.magnitude_edit.setText(str(magnitude))

        dialog.size_edit.setText(target_data.get("size_info", ""))
        dialog.priority_combo.setCurrentText(target_data.get("priority", "Medium"))
        dialog.status_combo.setCurrentText(target_data.get("status", "Not Observed"))
        dialog.months_edit.setText(target_data.get("best_months", ""))
        dialog.notes_edit.setPlainText(target_data.get("notes", ""))

        # Set telescope selection (a disabled one is kept)
        from cosmoscollection.equipment.EquipmentManager import select_telescope
        if not select_telescope(dialog.telescope_combo, self.db_manager, target_data.get("telescope_id")):
            dialog.telescope_combo.setCurrentIndex(0)  # "Any"

        if dialog.exec() == QDialog.Accepted:
            # Store the target ID to re-select after reload
            edited_target_id = target_data["id"]

            # Reload targets to reflect the changes (dialog already handles the database update)
            self._load_targets()

            # Re-select the edited target
            self._select_target_by_id(edited_target_id)

    def _select_target_by_id(self, target_id):
        """Find and select a target row by its ID"""
        for row in range(self.targets_table.rowCount()):
            name_item = self.targets_table.item(row, 0)
            if name_item:
                row_data = name_item.data(Qt.UserRole)
                if row_data and row_data.get("id") == target_id:
                    self.targets_table.selectRow(row)
                    self.targets_table.scrollToItem(name_item)
                    return

    def _remove_selected_target(self):
        """Remove the selected target from the list"""
        current_row = self.targets_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a target to remove.")
            return

        # Get target data from the name item (column 0) to handle sorting
        name_item = self.targets_table.item(current_row, 0)
        if not name_item:
            return

        target_data = name_item.data(Qt.UserRole)
        target_name = target_data.get("name", "Unknown")
        
        reply = QMessageBox.question(
            self, "Confirm Removal", 
            f"Are you sure you want to remove '{target_name}' from your target list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM usertargetlist WHERE id = ?", (target_data["id"],))
                    conn.commit()
                
                QMessageBox.information(self, "Success", f"'{target_name}' has been removed from your target list.")
                self._load_targets()
                
            except Exception as e:
                logger.error(f"Error removing target: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to remove target: {str(e)}")
    
    def _on_selection_changed(self):
        """Handle selection changes in the table"""
        has_selection = self.targets_table.currentRow() >= 0
        self.edit_target_btn.setEnabled(has_selection)
        self.view_details_btn.setEnabled(has_selection)
        self.remove_target_btn.setEnabled(has_selection)

    def _open_best_dso_tonight(self):
        """Open the Best DSO Tonight window"""
        try:
            # Imported here: Best DSO Tonight pulls in astropy's table code, which other
            # tools importing this module's shared helpers shouldn't have to load
            from cosmoscollection.planning.BestDSOTonight import BestDSOTonightWindow
            # Create and show the Best DSO Tonight window with target list auto-selected
            self.best_dso_window = BestDSOTonightWindow(use_target_list=True)
            self.best_dso_window.show()
            self.best_dso_window.raise_()
            self.best_dso_window.activateWindow()
            logger.debug("Best DSO Tonight window opened successfully with target list selected")
        except Exception as e:
            logger.error(f"Error opening Best DSO Tonight window: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Best DSO Tonight window: {str(e)}")

    def _row_object(self, row):
        """The target a table row shows, as DSOActions takes it - or None."""
        name_item = self.targets_table.item(row, 0)
        target = name_item.data(Qt.UserRole) if name_item else None
        return {**target, "target_id": target.get("id")} if target else None

    def _view_target_details(self):
        """DSO Details for the selected target."""
        obj = self._row_object(self.targets_table.currentRow())
        if obj is None:
            QMessageBox.warning(self, "No Selection", "Please select a target to view details.")
            return
        DSOActions.open_dso_details(obj, self)

    def _filter_targets(self):
        """Apply filters to the targets table"""
        status_filter = self.status_filter.currentText()
        priority_filter = self.priority_filter.currentText()
        telescope_filter_data = self.telescope_filter.currentData()  # Can be "all", "unassigned", or telescope_id
        search_text = self.search_box.text().strip().lower()

        for row in range(self.targets_table.rowCount()):
            show_row = True

            if status_filter != "All":
                status_item = self.targets_table.item(row, 6)  # Status column
                if not status_item or status_item.text() != status_filter:
                    show_row = False

            if priority_filter != "All" and show_row:
                priority_item = self.targets_table.item(row, 5)  # Priority column
                if not priority_item or priority_item.text() != priority_filter:
                    show_row = False

            if telescope_filter_data != "all" and show_row:
                telescope_item = self.targets_table.item(row, 7)  # Telescope column
                if telescope_item:
                    telescope_id = telescope_item.data(Qt.UserRole)
                    if telescope_filter_data == "unassigned":
                        # Show only targets with no telescope assigned
                        if telescope_id is not None:
                            show_row = False
                    else:
                        # Filter by specific telescope ID
                        if telescope_id != telescope_filter_data:
                            show_row = False

            if search_text and show_row:
                name_text = (self.targets_table.item(row, 0).text() if self.targets_table.item(row, 0) else "").lower()
                type_text = (self.targets_table.item(row, 1).text() if self.targets_table.item(row, 1) else "").lower()
                const_text = (self.targets_table.item(row, 2).text() if self.targets_table.item(row, 2) else "").lower()
                if search_text not in name_text and search_text not in type_text and search_text not in const_text:
                    show_row = False

            self.targets_table.setRowHidden(row, not show_row)
    
    def _load_targets(self):
        """Load targets from the database and populate the table"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.id, t.name, t.dso_type, t.constellation, t.ra_deg, t.dec_deg, t.magnitude,
                           t.size_info, t.priority, t.status, t.best_months, t.notes, t.date_added,
                           t.telescope_id, tel.name as telescope_name
                    FROM usertargetlist t
                    LEFT JOIN usertelescopes tel ON t.telescope_id = tel.id
                    ORDER BY t.priority DESC, t.date_added DESC
                """)

                rows = cursor.fetchall()
                self.targets_data = []

                for row in rows:
                    target_data = {
                        "id": row[0],
                        "name": row[1],
                        "dso_type": row[2],
                        "constellation": row[3],
                        "ra_deg": row[4],
                        "dec_deg": row[5],
                        "magnitude": row[6],
                        "size_info": row[7],
                        "priority": row[8],
                        "status": row[9],
                        "best_months": row[10],
                        "notes": row[11],
                        "date_added": row[12],
                        "telescope_id": row[13],
                        "telescope_name": row[14]
                    }
                    self.targets_data.append(target_data)
            
            self._populate_table()
            self._filter_targets()
            
            self.status_label.setText(f"Loaded {len(self.targets_data)} targets")
            self._update_duplicates_link()
            
        except Exception as e:
            logger.error(f"Error loading targets: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load targets: {str(e)}")

    def _update_duplicates_link(self):
        """Show "N possible duplicates - Review..." when entries look like the same object."""
        try:
            with self.db_manager.get_connection() as conn:
                count = len(find_duplicate_groups(conn))
        except Exception as e:
            logger.error(f"Error checking for duplicate targets: {e}")
            count = 0
        if not count:
            self.duplicates_label.hide()
            return
        text = f"{count} possible duplicate{'s' if count != 1 else ''}"
        themed_text(self.duplicates_label, lambda: (
            f"<span style='color: {COLORS['warning']};'>⚠</span> "
            f"<a href='review' style='color: {COLORS['link']};'>{text} — Review…</a>"))
        self.duplicates_label.show()

    def _review_duplicates(self):
        """Open the review dialog for possible duplicate entries."""
        dialog = DuplicateReviewDialog(self.db_manager, edit_callback=self._edit_target_with_data,
                                                        parent=self)
        dialog.targets_changed.connect(self._load_targets)
        dialog.exec()
        self._load_targets()
    
    def _populate_table(self):
        """Populate the targets table with loaded data"""
        # Disable sorting temporarily while populating
        self.targets_table.setSortingEnabled(False)

        self.targets_table.setRowCount(len(self.targets_data))

        for row, target in enumerate(self.targets_data):
            # Name - store target data in item
            name_item = QTableWidgetItem(target.get("name", ""))
            name_item.setData(Qt.UserRole, target)  # Store full target data
            self.targets_table.setItem(row, 0, name_item)

            # Type - use friendly name
            dso_type = target.get("dso_type", "")
            friendly_type = type_name(dso_type)
            self.targets_table.setItem(row, 1, QTableWidgetItem(friendly_type))

            # Constellation
            self.targets_table.setItem(row, 2, QTableWidgetItem(target.get("constellation", "")))

            # Magnitude - use numeric sorting
            magnitude = target.get("magnitude", 0)
            mag_item = QTableWidgetItem()
            mag_item.setData(Qt.DisplayRole, f"{magnitude:.1f}" if magnitude > 0 else "")
            mag_item.setData(Qt.UserRole, magnitude if magnitude > 0 else 999)  # Store numeric value for sorting
            mag_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 3, mag_item)

            # Size
            self.targets_table.setItem(row, 4, QTableWidgetItem(target.get("size_info", "")))

            # Priority - use custom PriorityTableWidgetItem for proper sorting
            priority = target.get("priority", "")
            priority_item = PriorityTableWidgetItem(priority)
            self.targets_table.setItem(row, 5, priority_item)

            # Status - use status order for sorting
            status = target.get("status", "")
            status_item = QTableWidgetItem(status)
            status_order = {"Not Observed": 1, "Observed": 2, "Imaged": 3, "Completed": 4}
            status_item.setData(Qt.UserRole, status_order.get(status, 0))  # Store numeric value for sorting
            status_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 6, status_item)

            # Telescope - display name or "Any" for unassigned
            telescope_name = target.get("telescope_name", "")
            telescope_id = target.get("telescope_id")
            telescope_display = telescope_name if telescope_name else "Any"
            telescope_item = QTableWidgetItem(telescope_display)
            telescope_item.setData(Qt.UserRole, telescope_id)  # Store telescope_id for filtering
            telescope_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 7, telescope_item)

            # Direction - filled in by _update_directions (in the background)
            direction_item = QTableWidgetItem("")
            direction_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 8, direction_item)

            # Best Months
            self.targets_table.setItem(row, 9, QTableWidgetItem(target.get("best_months", "")))

            # Date Added - use date object for sorting
            date_added = target.get("date_added", "")
            if date_added:
                try:
                    # Format date for display
                    date_obj = datetime.strptime(date_added, "%Y-%m-%d %H:%M:%S")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    date_timestamp = date_obj.timestamp()
                except:
                    formatted_date = date_added
                    date_timestamp = 0
            else:
                formatted_date = ""
                date_timestamp = 0

            date_item = QTableWidgetItem(formatted_date)
            date_item.setData(Qt.UserRole, date_timestamp)  # Store timestamp for sorting
            date_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 10, date_item)

        # Re-enable sorting
        self.targets_table.setSortingEnabled(True)

        # Set default sort by Priority (column 5) in descending order (Urgent first)
        self.targets_table.sortItems(5, Qt.DescendingOrder)

        # Last results meanwhile (no flicker), then recalculate for any new targets
        self._apply_direction_cells()
        self._update_directions()
    
    def _show_context_menu(self, position):
        """Right-click menu for a target (see DSOActions), with Edit and Remove."""
        item = self.targets_table.itemAt(position)
        obj = self._row_object(item.row()) if item else None
        if obj is None:
            return
        row = item.row()

        def extras(menu):
            menu.addAction("Edit Target").triggered.connect(lambda: self._context_edit_target(row))
            menu.addAction("Remove Target").triggered.connect(lambda: self._context_remove_target(row))

        menu = DSOActions.build_dso_context_menu(self, obj, extras=extras, on_target_list=True)
        menu.exec(self.targets_table.mapToGlobal(position))

    def _context_edit_target(self, row):
        """Edit target from context menu"""
        # Set the table selection to this row and call existing method
        self.targets_table.selectRow(row)
        self._edit_selected_target()

    def _context_remove_target(self, row):
        """Remove target from context menu"""
        # Set the table selection to this row and call existing method
        self.targets_table.selectRow(row)
        self._remove_selected_target()

    def add_target_from_dso(self, dso_data):
        """Add a target from DSO data (called from DSODetailWindow)"""
        # If designations are available, use the preferred catalog name
        if "designations" in dso_data and dso_data["designations"]:
            preferred_name = preferred_designation(
                [d.strip().split(" ", 1) for d in dso_data["designations"].split(",") if " " in d.strip()])
            if preferred_name:
                dso_data["name"] = preferred_name

        dialog = AddTargetDialog(dso_data=dso_data, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_targets()

    def open_and_select_target(self, dso_name):
        """Open the target list window and select the target with the given name

        Args:
            dso_name: Name of the DSO to select

        Returns:
            bool: True if target was found and selected, False otherwise
        """
        try:
            # Ensure the window is visible
            if not self.isVisible():
                self.show()
            self.raise_()
            self.activateWindow()

            # Reload targets to ensure we have current data
            self._load_targets()

            # The entry for this object, even if it's listed under another name
            with self.db_manager.get_connection() as conn:
                match = find_existing_target(conn, dso_name)
            if not match:
                return False

            for row in range(self.targets_table.rowCount()):
                name_item = self.targets_table.item(row, 0)
                row_data = name_item.data(Qt.UserRole) if name_item else None
                if row_data and row_data.get("id") == match["id"]:
                    # Select the row
                    self.targets_table.selectRow(row)
                    self.targets_table.scrollToItem(name_item)

                    # Open the edit dialog to show the notes
                    target_data = name_item.data(Qt.UserRole)
                    if target_data:
                        self._edit_target_with_data(target_data)
                    return True

            return False
        except Exception as e:
            logger.error(f"Error opening and selecting target: {str(e)}")
            return False

    def _edit_target_with_data(self, target_data):
        """Open the edit target dialog with the given target data

        Args:
            target_data: Dictionary containing target information
        """
        try:
            dialog = AddTargetDialog(dso_data=target_data, parent=self)
            dialog.setWindowTitle("Edit Target")
            dialog.set_edit_mode(target_data["id"])  # Enable edit mode and set target ID

            # Pre-populate with target data
            dialog.name_edit.setText(target_data.get("name", ""))
            dialog.type_edit.setText(type_name(target_data.get("dso_type")))
            dialog.constellation_edit.setText(target_data.get("constellation", ""))

            # Handle numeric fields - only set if value is not None
            ra_deg = target_data.get("ra_deg")
            if ra_deg is not None:
                dialog.ra_edit.setText(str(ra_deg))

            dec_deg = target_data.get("dec_deg")
            if dec_deg is not None:
                dialog.dec_edit.setText(str(dec_deg))

            magnitude = target_data.get("magnitude")
            if magnitude is not None:
                dialog.magnitude_edit.setText(str(magnitude))

            dialog.size_edit.setText(target_data.get("size_info", ""))
            dialog.priority_combo.setCurrentText(target_data.get("priority", "Medium"))
            dialog.status_combo.setCurrentText(target_data.get("status", "Not Observed"))
            dialog.months_edit.setText(target_data.get("best_months", ""))
            dialog.notes_edit.setPlainText(target_data.get("notes", ""))

            # Set telescope selection (a disabled one is kept)
            from cosmoscollection.equipment.EquipmentManager import select_telescope
            if not select_telescope(dialog.telescope_combo, self.db_manager, target_data.get("telescope_id")):
                dialog.telescope_combo.setCurrentIndex(0)  # "Any"

            if dialog.exec() == QDialog.Accepted:
                # Store the target ID to re-select after reload
                edited_target_id = target_data["id"]

                # Reload targets to reflect the changes
                self._load_targets()

                # Re-select the edited target
                self._select_target_by_id(edited_target_id)
        except Exception as e:
            logger.error(f"Error opening edit target dialog: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open target details: {str(e)}")


def main():
    """Main entry point for the application"""
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = DSOTargetListWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()