"""
The Target List's rows in the database: loading, adding, deleting, and the
shared "is this object already on the list?" check.

No UI imports - used by the Target List window (DSOTargetList.py, which
re-exports these names) and the command line.
"""

import logging
from datetime import datetime

from cosmoscollection.catalog.DSONames import designation_candidates
from cosmoscollection.core.Formatting import known_magnitude
from cosmoscollection.sessions import SessionFileScanner

logger = logging.getLogger(__name__)

# Order from least to most advanced / important (merges keep the highest)
TARGET_STATUSES = ["Not Observed", "Observed", "Imaged", "Completed"]
TARGET_PRIORITIES = ["Low", "Medium", "High", "Urgent"]

TARGET_COLUMNS = ("id", "name", "dso_type", "constellation", "ra_deg", "dec_deg", "magnitude",
                  "size_info", "priority", "status", "best_months", "notes", "date_added",
                  "date_observed", "created_date", "telescope_id")


# ---------------------------------------------------------------- name rules

def name_key(name):
    """Comparable key for a target name: spacing, punctuation and parenthetical
    common names ignored ('M 17', 'M17', 'm-17 (Omega)' all give 'M17')."""
    return SessionFileScanner.normalize_object_name(name)


_ASCII_UPPER = str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def catalog_object_ids(cursor, name):
    """The catalog objects (dsodetail ids) a target name designates; empty if
    the name isn't a known designation."""
    return catalog_object_ids_many(cursor, [name])[name]


def catalog_object_ids_many(cursor, names):
    """{name: catalog_object_ids(name)} for many names in one pass over the
    catalog - cataloguenr is a view over the update catalogs that no index
    serves case-insensitively, so each lookup scans it all."""
    def fold(text):  # Case folded as COLLATE NOCASE does it (ASCII letters only)
        return (text or "").translate(_ASCII_UPPER)

    wanted = {}  # (CATALOGUE, DESIGNATION) -> the names that could be it
    for name in names:
        for catalogue, designation in designation_candidates(name):
            wanted.setdefault((fold(catalogue), fold(designation)), set()).add(name)
    ids = {name: set() for name in names}
    designations = list({designation for _, designation in wanted})
    try:
        for start in range(0, len(designations), 500):
            chunk = designations[start:start + 500]
            cursor.execute(f"""
                SELECT dsodetailid, catalogue, designation FROM cataloguenr
                WHERE designation COLLATE NOCASE IN ({",".join("?" * len(chunk))})
            """, chunk)
            for obj, catalogue, designation in cursor.fetchall():
                for name in wanted.get((fold(catalogue), fold(designation)), ()):
                    ids[name].add(obj)
    except Exception as e:
        logger.debug(f"Catalog lookup failed for {len(names)} names: {e}")
    return {name: frozenset(found) for name, found in ids.items()}


def _standard_name(cursor, name):
    """The catalog's own spelling of a target's designation ('M17' -> 'M 17'), or None."""
    for catalogue, designation in designation_candidates(name):
        cursor.execute("""
            SELECT catalogue || ' ' || designation FROM cataloguenr
            WHERE catalogue = ? COLLATE NOCASE AND designation = ? COLLATE NOCASE LIMIT 1
        """, (catalogue, designation))
        row = cursor.fetchone()
        if row:
            return row[0]
    return None


# ---------------------------------------------------------------- loading

def load_target_rows(conn):
    """All Target List rows as dicts (oldest first) and their linked session counts."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT {', '.join(TARGET_COLUMNS)} FROM usertargetlist ORDER BY id")
    targets = [dict(zip(TARGET_COLUMNS, row)) for row in cursor.fetchall()]
    session_counts = {}
    try:
        cursor.execute("SELECT target_id, COUNT(*) FROM usersessions WHERE target_id IS NOT NULL GROUP BY target_id")
        session_counts = dict(cursor.fetchall())
    except Exception as e:
        logger.debug(f"Couldn't count linked sessions: {e}")  # No sessions table yet
    for target in targets:
        target["session_count"] = session_counts.get(target["id"], 0)
    return targets


def has_position(target):
    """Whether a target has real coordinates (the add dialog stores blanks as 0, 0)."""
    ra, dec = target.get("ra_deg"), target.get("dec_deg")
    return ra is not None and dec is not None and not (ra == 0 and dec == 0)


# ---------------------------------------------------------------- changes

def size_info_text(size_min, size_max):
    """The Target List's size text ("35.0 x 95.0", arcminutes), or "" when unknown."""
    size_min, size_max = size_min or 0, size_max or 0
    return f"{size_min:.1f} x {size_max:.1f}" if (size_min or size_max) else ""


def add_target(conn, name, dso=None, priority="Medium", status="Not Observed", telescope_id=None, notes=""):
    """Add a Target List entry - from a catalog object's data (dso: the main
    window's dict, see DSOData.dso_row_to_dict) like Bulk Add does. Commits;
    returns the new entry's id. Doesn't check for duplicates (find_existing_target)."""
    dso = dso or {}
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO usertargetlist (
            name, dso_type, constellation, ra_deg, dec_deg,
            magnitude, size_info, priority, status, notes, date_added, telescope_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name, dso.get("dso_type", ""), dso.get("constellation", ""),
        dso.get("ra_deg", 0), dso.get("dec_deg", 0), known_magnitude(dso.get("magnitude")),
        size_info_text(dso.get("size_min"), dso.get("size_max")),
        priority, status, notes or "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), telescope_id,
    ))
    conn.commit()
    return cursor.lastrowid


def delete_target(conn, target_id, move_sessions_to=None):
    """Delete one entry; its linked sessions move to move_sessions_to (or are
    unlinked). Returns the number of sessions moved."""
    cursor = conn.cursor()
    try:
        moved = 0
        try:
            # Done here rather than left to ON DELETE SET NULL, which SQLite only
            # enforces when foreign keys are switched on for the connection
            cursor.execute("UPDATE usersessions SET target_id = ? WHERE target_id = ?",
                           (move_sessions_to, target_id))
            moved = cursor.rowcount if move_sessions_to is not None else 0
        except Exception as e:
            if "no such table" not in str(e):
                raise
        cursor.execute("DELETE FROM usertargetlist WHERE id = ?", (target_id,))
        conn.commit()
        return moved
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------- prevention

# A target within this of a catalog object's position is that object (an entry
# added from the catalog keeps its coordinates), matching DSODetail's old check
SAME_POSITION_DEG = 0.001


class TargetListIndex:
    """The Target List indexed for "is this object already on the list?"
    checks - same name (ignoring case, spacing, punctuation and parenthetical
    common names), another designation of the same catalog object, or
    (optionally) the same position. Build once to check many objects."""

    def __init__(self, conn, exclude_id=None):
        self._cursor = conn.cursor()
        self._cursor.execute("SELECT id, name, ra_deg, dec_deg FROM usertargetlist ORDER BY id")
        self.entries = [{"id": row[0], "name": row[1], "ra_deg": row[2], "dec_deg": row[3]}
                        for row in self._cursor.fetchall() if row[0] != exclude_id and row[1]]
        self._exact, self._keys = {}, {}
        objects = {}
        entry_objects = catalog_object_ids_many(self._cursor, [entry["name"] for entry in self.entries])
        for entry in self.entries:
            self._exact.setdefault(entry["name"].strip().upper(), entry)
            self._keys.setdefault(name_key(entry["name"]), entry)
            for obj in entry_objects[entry["name"]]:
                objects.setdefault(obj, entry)
        # Every catalog designation of the listed objects ("NGC 6618" for M 17), so
        # checking a name is a lookup rather than a catalog query per name
        self._designations = {}
        object_ids = list(objects)
        for start in range(0, len(object_ids), 500):
            chunk = object_ids[start:start + 500]
            self._cursor.execute(f"""
                SELECT dsodetailid, catalogue || ' ' || designation FROM cataloguenr
                WHERE dsodetailid IN ({",".join("?" * len(chunk))})
            """, chunk)
            for obj, designation in self._cursor.fetchall():
                self._designations.setdefault(name_key(designation), objects[obj])

    def match(self, name, ra_deg=None, dec_deg=None):
        """The entry that's the same object as name (and, if given, position), or None."""
        name = (name or "").strip()
        if name:
            key = name_key(name)
            entry = self._exact.get(name.upper()) or self._keys.get(key) or self._designations.get(key)
            if entry:
                return entry
        if ra_deg is not None and dec_deg is not None and not (ra_deg == 0 and dec_deg == 0):
            for entry in self.entries:
                if (has_position(entry) and abs(entry["ra_deg"] - ra_deg) < SAME_POSITION_DEG
                        and abs(entry["dec_deg"] - dec_deg) < SAME_POSITION_DEG):
                    return entry
        return None


def find_existing_target(conn, name, ra_deg=None, dec_deg=None, exclude_id=None):
    """The Target List entry that's the same object as name (or at the same
    position, if given), as {'id', 'name', 'ra_deg', 'dec_deg'}; None if it isn't
    on the list. exclude_id leaves out the entry being edited."""
    return TargetListIndex(conn, exclude_id).match(name, ra_deg, dec_deg)
