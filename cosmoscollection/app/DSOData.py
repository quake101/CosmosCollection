"""Loading the DSO catalog for the main table, and the search matching helpers."""

import logging
import re

from PySide6.QtCore import QThread, Signal

from cosmoscollection.core.ResourceManager import ResourceManager

logger = logging.getLogger(__name__)


# --- DSO data loading ---
# One row per catalogued object. Designations and the first user image come from
# subqueries rather than joins, so an object with several images doesn't repeat
# every designation once per image.
DSO_QUERY = """
    SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness,
           CAST(d.sizemin/60.0 AS REAL) as sizemin,
           CAST(d.sizemax/60.0 AS REAL) as sizemax,
           d.constellation, d.dsotype, d.dsoclass,
           (SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY
                       CASE c.catalogue
                           WHEN 'M' THEN 1
                           WHEN 'NGC' THEN 2
                           WHEN 'IC' THEN 3
                           ELSE 4
                       END, c.designation)
            FROM cataloguenr c
            WHERE c.dsodetailid = d.id) as designations,
           ui.image_path, ui.integration_time, ui.date_taken, ui.notes,
           COALESCE(img.image_count, 0) as image_count
    FROM dsodetail d
    LEFT JOIN (SELECT dsodetailid, MIN(id) AS first_id, COUNT(*) AS image_count
               FROM userimages
               GROUP BY dsodetailid) img ON img.dsodetailid = d.id
    LEFT JOIN userimages ui ON ui.id = img.first_id
    WHERE EXISTS (SELECT 1 FROM cataloguenr c WHERE c.dsodetailid = d.id)
"""


def format_ra(ra_deg):
    """Convert RA in degrees to hms format"""
    if ra_deg is None:
        return ""
    ra_hours = ra_deg / 15.0
    ra_h = int(ra_hours)
    ra_remaining = (ra_hours - ra_h) * 60
    ra_m = int(ra_remaining)
    ra_s = (ra_remaining - ra_m) * 60
    return f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s"


def format_dec(dec_deg):
    """Convert Dec in degrees to dms format"""
    if dec_deg is None:
        return ""
    dec_sign = '-' if dec_deg < 0 else '+'
    dec_abs = abs(dec_deg)
    dec_d = int(dec_abs)
    dec_remaining = (dec_abs - dec_d) * 60
    dec_m = int(dec_remaining)
    dec_s = (dec_remaining - dec_m) * 60
    return f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""


def dso_row_to_dict(row):
    """Convert a DSO_QUERY row into the dict used by the table model and detail window"""
    obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
        constellation, dso_type, dso_class, designations, image_path, integration_time, \
        date_taken, notes, image_count = row

    designations = designations or ""
    # Primary designation, e.g. "M 31"; a few objects have no number ("Coalsack")
    catalogue, _, designation = designations.split(", ")[0].strip().partition(" ")

    return {
        "dsodetailid": obj_id,
        "id": designation,
        "ra_deg": ra,
        "dec_deg": dec,
        "catalogue": catalogue,
        "name": f"{catalogue} {designation}",
        "magnitude": magnitude,
        "surface_brightness": surface_brightness,
        "size_min": float(size_min) if size_min is not None else 0.0,
        "size_max": float(size_max) if size_max is not None else 0.0,
        "constellation": constellation,
        "dso_type": dso_type,
        "dso_class": dso_class,
        "designations": designations,
        "image_path": image_path,
        "integration_time": integration_time,
        "date_taken": date_taken,
        "notes": notes,
        "image_count": image_count
    }


def load_all_dso_data(conn):
    """Load every catalogued DSO in a single query. Returns (dso_data, catalogs)."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT catalogue FROM cataloguenr ORDER BY catalogue")
    catalogs = [row[0] for row in cursor.fetchall()]
    cursor.execute(DSO_QUERY)
    return [dso_row_to_dict(row) for row in cursor.fetchall()], catalogs


# --- Startup Data Loader Thread ---
class InitialDataLoadWorker(QThread):
    """Loads the whole DSO catalog on startup without blocking the UI.

    A single query takes well under a second for the full catalog, so everything
    is loaded up front and filtering/sorting happen in memory."""
    data_loaded = Signal(list, list)  # dso_data, catalogs
    load_failed = Signal(str)  # error message

    def run(self):
        conn = None
        try:
            import sqlite3
            from cosmoscollection.core.ResourceManager import attach_update_catalogs

            conn = sqlite3.connect(str(ResourceManager.get_database_path()))
            attach_update_catalogs(conn)
            dso_data, catalogs = load_all_dso_data(conn)
            logger.debug(f"Loaded {len(dso_data)} DSOs in background thread")
            self.data_loaded.emit(dso_data, catalogs)

        except Exception as e:
            logger.error(f"Error loading initial data in background: {e}", exc_info=True)
            self.load_failed.emit(str(e))
        finally:
            if conn is not None:
                conn.close()


# --- Search matching helpers ---
SEARCH_RANK_EXACT = 0
SEARCH_RANK_PREFIX = 1
SEARCH_RANK_SUBSTRING = 2

_SEARCH_SEPARATORS_RE = re.compile(r"[\s\-_]+")
# A separator only carries meaning between two numbers ("M 3-1" is not "M 31")
_NON_NUMERIC_SEPARATOR_RE = re.compile(r"(?<!\d)-|-(?!\d)")
# Queries that look like RA ("00h42m") or Dec ("41°16'", "+41") coordinates
_COORDINATE_QUERY_RE = re.compile(r"\d\s*h|°|^[+-]\s*\d")


def normalize_search_text(text):
    """Normalize a designation for comparison.

    "M31", "M 31" and "M-31" all become "m31"; "Sh2 129" and "sh2-129" both
    become "sh2-129", while "M 3-1" stays distinct as "m3-1".
    """
    text = _SEARCH_SEPARATORS_RE.sub("-", text.strip().lower())
    return _NON_NUMERIC_SEPARATOR_RE.sub("", text)


def _rank_search_key(key, query_norm):
    """Rank how well a normalized designation matches a normalized query, or None"""
    if key == query_norm:
        return SEARCH_RANK_EXACT
    if key.startswith(query_norm):
        return SEARCH_RANK_PREFIX
    if query_norm in key:
        return SEARCH_RANK_SUBSTRING
    return None


def prepare_search_query(search_text, selected_catalog=None):
    """Precompute the query forms used by DSOTableModel._match_item, or None for no search"""
    raw = (search_text or "").strip().lower()
    norm = normalize_search_text(raw)
    is_coordinate = bool(_COORDINATE_QUERY_RE.search(raw))
    if not norm and not is_coordinate:
        return None

    catalog = selected_catalog if selected_catalog and selected_catalog != "All Catalogs" else None
    return {
        "raw": raw,
        "norm": norm,
        # With a catalog filter, "31" should match "M 31" exactly
        "catalog_prefix": f"{catalog.lower()} " if catalog else None,
        "catalog_norm": normalize_search_text(f"{catalog} {raw}") if catalog else None,
        "is_coordinate": is_coordinate,
    }


def designation_in_catalog(designation, catalog):
    """Whether a "CATALOG DESIGNATION" string belongs to catalog.

    The M catalogue entries also hold Minkowski planetaries ("M 1-65"), so only
    M 1 - M 110 count as Messier objects."""
    if not designation.startswith(catalog + " "):
        return False
    if catalog == "M":
        number = designation[2:].strip()
        return number.isdecimal() and 1 <= int(number) <= 110
    return True


_NATURAL_SPLIT_RE = re.compile(r"(\d+)")


def _natural_key(text):
    """Sort key that orders embedded numbers numerically ("M 2" before "M 10")"""
    return tuple((0, int(part)) if part.isdecimal() else (1, part.lower())
                 for part in _NATURAL_SPLIT_RE.split(text or "") if part)
