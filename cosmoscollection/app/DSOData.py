"""Loading the DSO catalog for the main table, and the search matching helpers.

No Qt: the command line uses these too (the startup loading thread is in Startup.py)."""

import logging
import re

from cosmoscollection.core.Formatting import format_dec, format_ra, natural_key

logger = logging.getLogger(__name__)


# --- DSO data loading ---
# An object's designations, best known catalog first ("M 31, NGC 224, UGC 454"),
# for a query whose dsodetail table is "d". A subquery rather than a join, so it
# never repeats a designation once per joined row (e.g. per image).
DESIGNATIONS_SQL = """(SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY
                           CASE c.catalogue WHEN 'M' THEN 1 WHEN 'NGC' THEN 2 WHEN 'IC' THEN 3 ELSE 4 END,
                           c.designation)
                       FROM cataloguenr c WHERE c.dsodetailid = d.id)"""

# One row per catalogued object, with its first user image (also a subquery).
# Add conditions with " AND ...".
DSO_QUERY = f"""
    SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness,
           CAST(d.sizemin/60.0 AS REAL) as sizemin,
           CAST(d.sizemax/60.0 AS REAL) as sizemax,
           d.constellation, d.dsotype, d.dsoclass,
           {DESIGNATIONS_SQL} as designations,
           ui.image_path, ui.integration_time, ui.date_taken, ui.notes,
           COALESCE(img.image_count, 0) as image_count
    FROM dsodetail d
    LEFT JOIN (SELECT dsodetailid, MIN(id) AS first_id, COUNT(*) AS image_count
               FROM userimages
               GROUP BY dsodetailid) img ON img.dsodetailid = d.id
    LEFT JOIN userimages ui ON ui.id = img.first_id
    WHERE EXISTS (SELECT 1 FROM cataloguenr c WHERE c.dsodetailid = d.id)
"""


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


def load_dso(conn, dsodetailid):
    """One object as the main window loads it (dso_row_to_dict), or None."""
    row = conn.execute(DSO_QUERY + " AND d.id = ?", (dsodetailid,)).fetchone()
    return dso_row_to_dict(tuple(row)) if row else None


def _split_designation(name):
    """('M', '31') from 'M 31' or 'M31'; ('NGC', '7000') from 'NGC7000'; else None.
    Session names often come from FITS OBJECT headers, which usually drop the space."""
    parts = (name or "").split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    match = re.fullmatch(r"([A-Za-z]+)[-_ ]?(\d[\w.+-]*)", (name or "").strip())
    return (match.group(1), match.group(2)) if match else None


def find_full_dso_data(db_manager, target_name, target_data):
    """An object's data (load_dso's dict, the one DSODetailWindow and the FOV
    Simulator take) for a Target List entry or session: looked up by catalog
    designation, else the nearest object within 0.1 degrees of target_data's
    ra_deg/dec_deg. Its notes fall back to target_data's. None if not found."""
    try:
        with db_manager.get_connection() as conn:
            row = None
            split = _split_designation(target_name)
            if split:
                row = conn.execute(DSO_QUERY + """
                    AND d.id = (SELECT c2.dsodetailid FROM cataloguenr c2
                                WHERE c2.catalogue = ? COLLATE NOCASE AND c2.designation = ? COLLATE NOCASE
                                LIMIT 1)""", split).fetchone()
            ra, dec = target_data.get("ra_deg"), target_data.get("dec_deg")
            if row is None and ra is not None and dec is not None:
                tolerance = 0.1  # degrees
                row = conn.execute(DSO_QUERY + """
                    AND ABS(d.ra - ?) < ? AND ABS(d.dec - ?) < ?
                    ORDER BY ABS(d.ra - ?) + ABS(d.dec - ?) LIMIT 1""",
                                   (ra, tolerance, dec, tolerance, ra, dec)).fetchone()
        if row is None:
            return None
        data = dso_row_to_dict(tuple(row))
        data["notes"] = data["notes"] or target_data.get("notes", "")
        return data
    except Exception as e:
        logger.error(f"Error looking up {target_name!r} in the catalog: {e}")
        return None


def load_all_dso_data(conn):
    """Load every catalogued DSO in a single query. Returns (dso_data, catalogs)."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT catalogue FROM cataloguenr ORDER BY catalogue")
    catalogs = [row[0] for row in cursor.fetchall()]
    cursor.execute(DSO_QUERY)
    return [dso_row_to_dict(row) for row in cursor.fetchall()], catalogs


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


class DSOSearch:
    """The main window's DSO list filters and search ranking (DSOTableModel),
    without Qt, so the command line finds the same objects. Keeps a cache of
    each designations string's search keys."""

    def __init__(self):
        self._key_cache = {}  # designations string -> [(designation, normalized)]

    def search_keys(self, designations):
        """[(designation, normalized designation)], cached by designations string"""
        keys = self._key_cache.get(designations)
        if keys is None:
            keys = [(designation, normalize_search_text(designation))
                    for designation in designations.split(", ")]
            self._key_cache[designations] = keys
        return keys

    def filter(self, dso_data, search_text, selected_catalog=None, show_images_only=False,
               selected_type=None, show_no_images_only=False):
        """The entries passing all filters, search matches tagged with their rank
        ("search_rank") and the designation that matched ("matched_designation")"""
        catalog = selected_catalog if selected_catalog and selected_catalog != "All Catalogs" else None
        type_code = selected_type if selected_type and selected_type != "All Types" else None
        query = prepare_search_query(search_text, catalog)

        if not (query or catalog or type_code or show_images_only or show_no_images_only):
            return list(dso_data)

        matches = []
        for item in dso_data:
            if catalog and not any(designation_in_catalog(designation, catalog)
                                   for designation, _ in self.search_keys(item["designations"])):
                continue

            if type_code and item.get("dso_type", "") != type_code:
                continue

            if show_images_only and item["image_count"] == 0:
                continue

            if show_no_images_only and item["image_count"] > 0:
                continue

            if query is None:
                matches.append(item)
                continue

            match = self.match_item(item, query)
            if match is None:
                continue

            rank, matched_designation = match
            item_copy = item.copy()
            item_copy["search_rank"] = rank
            if matched_designation:
                item_copy["matched_designation"] = matched_designation
            matches.append(item_copy)

        return matches

    def match_item(self, item, query):
        """Match an item against a prepared search query.

        Returns (rank, matched_designation) or None. On equal rank, a designation
        from the selected catalog is preferred so it is the one displayed.
        """
        best = None  # (rank, outside_catalog, designation)
        for designation, key in self.search_keys(item["designations"]):
            in_catalog = bool(query["catalog_prefix"]) and designation.lower().startswith(query["catalog_prefix"])

            ranks = []
            if query["norm"]:
                ranks.append(_rank_search_key(key, query["norm"]))
            if in_catalog:
                ranks.append(_rank_search_key(key, query["catalog_norm"]))
            ranks = [rank for rank in ranks if rank is not None]
            if not ranks:
                continue

            candidate = (min(ranks), not in_catalog, designation)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        if best is not None:
            return best[0], best[2]

        # Coordinate search: typed fragment of an RA/Dec, or a pasted full RA/Dec string
        if query["is_coordinate"] and item.get("ra_deg") is not None and item.get("dec_deg") is not None:
            raw = query["raw"].replace(" ", "")  # '00h42m' and '00h 42m' both match
            for formatted in (format_ra(item["ra_deg"]).lower().replace(" ", ""),
                              format_dec(item["dec_deg"]).lower().replace(" ", "")):
                if raw in formatted or formatted in raw:
                    return SEARCH_RANK_SUBSTRING, None

        return None

    def display_designation(self, entry, selected_catalog=None):
        """(catalog, designation) shown for an entry: the search match if any, else
        its designation in the selected catalog, else its primary designation"""
        matched_designation = entry.get("matched_designation")
        if matched_designation:
            catalog, sep, designation = matched_designation.partition(" ")
            return catalog, designation if sep else matched_designation

        if selected_catalog:
            for designation, _ in self.search_keys(entry["designations"]):
                if designation_in_catalog(designation, selected_catalog):
                    return selected_catalog, designation.split(" ", 1)[1]
            return selected_catalog, entry["id"]
        return entry["catalogue"], entry["id"]

    def sort_key(self, selected_catalog=None):
        """Sort key for entries by their shown designation (natural number order)"""
        def key(entry):
            catalog, designation = self.display_designation(entry, selected_catalog)
            return natural_key(catalog), natural_key(designation)
        return key
