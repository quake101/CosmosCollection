#!/usr/bin/env python3
"""
DSO_Name_Updater.py (v1.5)
==========================

Generates a supplementary SQLite database of missing "NAME" catalog records
for deep-sky objects by cross-referencing the user's DSO database against
one or more of:

  * OpenNGC   (https://github.com/mattiaverga/OpenNGC)   -- CC-BY-SA-4.0
  * SIMBAD    (https://simbad.cds.unistra.fr/simbad/)    -- acknowledgement
  * Wikidata  (https://www.wikidata.org/)                -- CC0 (public domain)

The source DSO database is never modified. A separate "update" database is
produced that mirrors the relevant schema so the consuming project can
attach it and union the results at query time.

USAGE
-----
    # All sources (default):
    python3 DSO_Name_Updater.py --source-db DSO.sqlite

    # One source:
    python3 DSO_Name_Updater.py --source-db DSO.sqlite --source openngc
    python3 DSO_Name_Updater.py --source-db DSO.sqlite --source simbad
    python3 DSO_Name_Updater.py --source-db DSO.sqlite --source wikidata

    # Combinations:
    python3 DSO_Name_Updater.py --source-db DSO.sqlite \
        --source openngc,wikidata

    # Offline OpenNGC:
    python3 DSO_Name_Updater.py --source-db DSO.sqlite --source openngc \
        --openngc-dir ./opennfc_cache

    # Review what was added and skipped:
    python3 DSO_Name_Updater.py --source-db DSO.sqlite \
        --review-csv review.csv

v1.5 CHANGES
------------
* Wikidata names are cleaned and vetted. Rejected: catalog numbers in
  words ("New General Catalogue 1123", "Messier Object 36"), any one or
  two word name ending in a number ("Mrk 590", "Haro 30", "LH-41"), lists
  of designations, notes, and star designations ("R Mon", "19 Puppis")
  unless the object is a star or globular cluster. Wiki markup and a
  leading "NAME " are stripped; all-lowercase names are capitalized.
* Names on name_exclusions.txt (next to this script) are never added,
  for every object or ("dsoid | Name") just one.
* --skip-shared-names skips a name that would also belong to another
  object (so each name finds one object). Off by default: real names are
  often shared (the source DB has Lagoon Nebula on NGC 6533, so M 8
  couldn't get it), and junk shared ones are caught by the filters above.
  The exclusion list can block a name on one wrong object.
* Names already in other update databases in the output folder are not
  added again (the app attaches every .sqlite there and unions them).
* A batch that fails after its retries is queried again in six smaller
  pieces, so one heavy batch doesn't sink the run.
* A run with failed batches no longer replaces the output unless
  --allow-partial is given, and the output is written to a temp file
  and swapped in, so a crash can't leave a missing or half-written DB.
* --review-csv lists every added and skipped name with the reason.
* User-Agent names the CosmosCollection project, per Wikimedia's policy.
* Names compare ignoring a leading "The", spaces, and hyphens
  ("The Sombrero Galaxy" = "Sombrero Galaxy", "Bow-Tie" = "Bowtie").

v1.4 CHANGES
------------
* Fixed Wikidata SPARQL 504 Gateway Timeouts. The previous query used
  REPLACE() inside a BIND/FILTER, which forced the engine to compute a
  function over every P528 statement before filtering -- too expensive
  for the public endpoint's 60-second budget. v1.4 sends each (catalog,
  code) pair as multiple literal stored-form variants in the VALUES
  clause, so the query becomes a pure equality lookup against the P528
  index. This is dramatically faster and avoids 504s entirely.
* Default --wikidata-batch-size lowered from 100 to 30 since each pair
  now expands to ~3 stored-form variants in the VALUES clause.

v1.3 CHANGES
------------
* Added Wikidata source via SPARQL queries against query.wikidata.org.
* `--source` now accepts a comma-separated list (e.g., "openngc,wikidata")
  in addition to "all" / "both".
* Per-name provenance now supports any combination of the three sources.

ATTRIBUTION (SUMMARY)
---------------------
OpenNGC:
  Mattia Verga, https://github.com/mattiaverga/OpenNGC
  CC-BY-SA-4.0 -- REQUIRES credit, share-alike for redistribution.

SIMBAD:
  CDS, Strasbourg. Standard ack: "This research has made use of the
  SIMBAD database, operated at CDS, Strasbourg, France"
  Reference: 2000A&AS..143....9W (Wenger et al.)
  No share-alike, but academic/publication use should cite.

Wikidata:
  Wikimedia Foundation. All structured data is released under CC0 1.0
  (Public Domain Dedication). No share-alike obligation; attribution
  is courteous but not required.
  https://www.wikidata.org/wiki/Wikidata:Licensing

Each generated update database carries an `attribution` table that
records the sources actually used in that run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# The app's own "is this a catalog number, not a name" rule, so the names
# this script adds pass the same test the app uses to pick a common name.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from DSONames import CATALOG_NUMBER_NAME, CONSTELLATIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "1.5"

OPENNGC_NGC_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/"
    "master/database_files/NGC.csv"
)
OPENNGC_ADDENDUM_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/"
    "master/database_files/addendum.csv"
)
OPENNGC_DELIMITER = ";"

SIMBAD_TAP_SYNC_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Wikidata Q-IDs for the catalogs we look up
WIKIDATA_Q_MESSIER = "Q14530"   # Messier catalog
WIKIDATA_Q_NGC = "Q14534"       # New General Catalogue
WIKIDATA_Q_IC = "Q741672"       # Index Catalogue

USER_AGENT = (
    f"CosmosCollection-DSO-Name-Updater/{VERSION} "
    "(https://github.com/quake101/CosmosCollection) Python-urllib"
)

DEFAULT_EXCLUDE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "name_exclusions.txt"
)

# OpenNGC CSV column headers we care about, looked up case-insensitively.
COL_NAME = "name"
COL_M = "m"
COL_NGC = "ngc"
COL_IC = "ic"
COL_COMMON = "common names"

NAME_CATALOGUE = "NAME"

# Source identifiers
SOURCE_OPENNGC = "openngc"
SOURCE_SIMBAD = "simbad"
SOURCE_WIKIDATA = "wikidata"
SOURCE_ALL_ALIAS = ("all", "both")  # convenience aliases
ALL_SOURCES = (SOURCE_OPENNGC, SOURCE_SIMBAD, SOURCE_WIKIDATA)

# Pretty labels written into the provenance table.
LABEL_OPENNGC = "OpenNGC"
LABEL_SIMBAD = "SIMBAD"
LABEL_WIKIDATA = "Wikidata"
LABEL_BY_SOURCE = {
    SOURCE_OPENNGC: LABEL_OPENNGC,
    SOURCE_SIMBAD: LABEL_SIMBAD,
    SOURCE_WIKIDATA: LABEL_WIKIDATA,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _norm_name(n: str) -> str:
    """Normalize a friendly name for dedup: ignores case, a leading "The",
    spaces and hyphens ("The Bow-Tie Nebula" = "Bowtie Nebula")."""
    s = n.strip().casefold()
    s = re.sub(r"^the\s+", "", s)
    return re.sub(r"[\s\-]+", "", s)


def _ro_uri(path: str) -> str:
    """Read-only SQLite URI for a path (safe with spaces, '#', '?').

    Built from abspath, not Path.resolve().as_uri(): resolving a mapped
    network drive gives a \\\\host\\share path, and SQLite rejects a host
    in a file: URI. An empty host plus a //host/share path works.
    """
    p = os.path.abspath(path).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p  # "C:/x" -> "/C:/x"
    return "file://" + urllib.parse.quote(p, safe="/:") + "?mode=ro"


def _norm_designation(raw: str) -> str:
    """Normalize a designation: strip leading zeros, uppercase letter suffix."""
    s = raw.strip().upper()
    mo = re.match(r"^(\d+)([A-Z]*)$", s)
    if not mo:
        return s
    return str(int(mo.group(1))) + mo.group(2)


def _simbad_norm_key(s: str) -> str:
    """Whitespace-insensitive uppercase key for SIMBAD identifier matching."""
    return re.sub(r"\s+", "", s.strip()).upper()


# Wikidata labels are crowd-edited and full of designations, translations
# and notes. These rules decide which ones are friendly names.

# Catalog numbers in words that DSONames.CATALOG_NUMBER_NAME doesn't know:
# "New General Galaxy 149", "Nuevo Catálogo General 55".
_CATALOG_WORDS_RE = re.compile(r"(?i)(?=.*\d).*\b(?:new general|catálogo|katalog)\b")

_GREEK_LETTERS = (
    "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|"
    "xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    "alf|bet|gam|del|eps|zet|tet|iot|kap|lam|omi|sig|ups|ome"
)
_CONSTELLATION_GENITIVES = (
    "Andromedae|Antliae|Apodis|Aquarii|Aquilae|Arae|Arietis|Aurigae|Bootis|"
    "Boötis|Caeli|Camelopardalis|Cancri|Canum Venaticorum|Canis Majoris|"
    "Canis Minoris|Capricorni|Carinae|Cassiopeiae|Centauri|Cephei|Ceti|"
    "Chamaeleontis|Circini|Columbae|Comae Berenices|Coronae Australis|"
    "Coronae Borealis|Corvi|Crateris|Crucis|Cygni|Delphini|Doradus|Draconis|"
    "Equulei|Eridani|Fornacis|Geminorum|Gruis|Herculis|Horologii|Hydrae|"
    "Hydri|Indi|Lacertae|Leonis|Leonis Minoris|Leporis|Librae|Lupi|Lyncis|"
    "Lyrae|Mensae|Microscopii|Monocerotis|Muscae|Normae|Octantis|Ophiuchi|"
    "Orionis|Pavonis|Pegasi|Persei|Phoenicis|Pictoris|Piscium|"
    "Piscis Austrini|Puppis|Pyxidis|Reticuli|Sagittae|Sagittarii|Scorpii|"
    "Sculptoris|Scuti|Serpentis|Sextantis|Tauri|Telescopii|Trianguli|"
    "Trianguli Australis|Tucanae|Ursae Majoris|Ursae Minoris|Velorum|"
    "Virginis|Volantis|Vulpeculae"
)
# Flamsteed, Bayer and variable-star names: "19 Puppis", "R Mon",
# "Omega Centauri", "V1059 Sagittarii", "ZZ Leporis".
_STAR_DESIGNATION_RE = re.compile(
    r"(?i)^(?:\d{1,3}|[A-Z]{1,2}|V\d{3,4}|[α-ω]|" + _GREEK_LETTERS + r")\s+"
    r"(?:" + _CONSTELLATION_GENITIVES + "|" + "|".join(CONSTELLATIONS) + r")$"
)
# Objects whose name may be a star designation ("47 Tucanae" is a globular
# cluster, "19 Puppis" is the star NGC 2542).
_STAR_NAMED_TYPES = {"1STAR", "2STAR", "3STAR", "4STAR", "GLOCL", "LMCGC", "SMCGC"}

_MAX_NAME_LENGTH = 50


def _clean_name(raw: str) -> str:
    """Tidy a crowd-edited label: wiki markup ('''Cat's Eye Nebula'''), a
    SIMBAD-style "NAME " prefix, extra spaces; capitalizes an all-lowercase
    name ("skull nebula" -> "Skull Nebula")."""
    s = re.sub(r"\s+", " ", raw).strip()
    s = re.sub(r"^'{2,}|'{2,}$", "", s).strip().strip('"').strip()
    if s.startswith("NAME "):
        s = s[len("NAME "):].strip()
    if s and s == s.lower():
        s = " ".join(w[:1].upper() + w[1:] for w in s.split(" "))
    return s


def _is_code_like(name: str) -> bool:
    """One or two words ending in a number ("Mrk 590", "Haro 30", "LH-41",
    "Δ252", "MITG J2132+4435"), or only uppercase abbreviations and numbers
    ("GCRV 128 E"). Longer names ("Robert's Quartet 1") and names without
    digits ("Andromeda II", "Orion B") aren't."""
    words = name.split()
    if not any(re.search(r"\d", w) for w in words):
        return False
    if all(w.isupper() or re.search(r"\d", w) for w in words):
        return True
    if len(words) > 2 or not re.search(r"\d", words[-1]):
        return False
    return "'" not in words[0] and "’" not in words[0]


def _reject_reason(name: str, dsotype: str = "") -> Optional[str]:
    """Why `name` isn't a friendly name for an object of `dsotype`, or None
    when it is one."""
    if len(name) < 4:
        return "too short"
    if "," in name:
        return "list of designations"
    if (
        "(" in name
        or name.casefold().startswith("aka ")
        or len(name) > _MAX_NAME_LENGTH
    ):
        return "note, not a name"
    if name.startswith("["):
        return "catalog code"
    if CATALOG_NUMBER_NAME.search(name) or _CATALOG_WORDS_RE.search(name):
        return "catalog number"
    if _is_code_like(name):
        return "catalog code"
    if _STAR_DESIGNATION_RE.match(name) and dsotype not in _STAR_NAMED_TYPES:
        return "star designation"
    return None


# ---------------------------------------------------------------------------
# Source DB loading (shared by all providers)
# ---------------------------------------------------------------------------


def load_source_info(
    source_db_path: str,
) -> Tuple[
    Dict[str, Set[str]],               # dsoid -> set of normalized existing NAMEs
    Dict[str, List[Tuple[str, str]]],  # dsoid -> list of (cat, norm_desig)
    Dict[str, str],                    # dsoid -> dsotype ("GALXY", "GLOCL"...)
]:
    """Read the source DB once (read-only). See module docstring."""
    conn = sqlite3.connect(_ro_uri(source_db_path), uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT dsodetailid, catalogue, designation FROM cataloguenr"
        )
        existing_names: Dict[str, Set[str]] = defaultdict(set)
        xrefs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for dsoid, cat, desig in cur.fetchall():
            if cat == NAME_CATALOGUE:
                existing_names[dsoid].add(_norm_name(desig))
            elif cat in ("NGC", "IC", "M"):
                xrefs[dsoid].append((cat, _norm_designation(desig)))
        types = {
            dsoid: (dsotype or "")
            for dsoid, dsotype in cur.execute("SELECT id, dsotype FROM dsodetail")
        }
        return existing_names, xrefs, types
    finally:
        conn.close()


def load_other_update_names(
    output_path: str,
    source_db_path: str,
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Normalized NAMEs already in the other update DBs beside the output.

    The app attaches every .sqlite in catalogs/Updates and unions them, so a
    name in one of them must not be added again by this run. The output file
    itself is skipped (it is being regenerated), as is the source DB.
    Returns (dsoid -> names, file names read).
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    skip = {
        os.path.normcase(os.path.abspath(p))
        for p in (output_path, source_db_path)
    }
    names: Dict[str, Set[str]] = defaultdict(set)
    used: List[str] = []
    for fn in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, fn)
        if not fn.lower().endswith(".sqlite") or os.path.normcase(path) in skip:
            continue
        conn = sqlite3.connect(_ro_uri(path), uri=True)
        try:
            rows = conn.execute(
                "SELECT dsodetailid, designation FROM cataloguenr "
                "WHERE catalogue = ?",
                (NAME_CATALOGUE,),
            ).fetchall()
        except sqlite3.DatabaseError:
            continue  # not an update DB
        finally:
            conn.close()
        for dsoid, desig in rows:
            names[dsoid].add(_norm_name(desig))
        used.append(fn)
    return names, used


def load_exclusions(path: Optional[str]) -> Set[Tuple[str, str]]:
    """Names never to add, as (dsoid or "", normalized name).

    One per line, '#' starts a comment. "Name" blocks the name for every
    object; "dsoid | Name" blocks it for that object only (a name Wikidata
    puts on the wrong object, "NGC246 | Pac-man Nebula").
    """
    if not path or not os.path.isfile(path):
        return set()
    out: Set[Tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            dsoid, _, name = line.rpartition("|")
            out.add((dsoid.strip(), _norm_name(name)))
    return out


# ---------------------------------------------------------------------------
# OpenNGC provider
# ---------------------------------------------------------------------------


def _http_get_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def load_openngc_csv(
    ngc_url: str = OPENNGC_NGC_URL,
    addendum_url: str = OPENNGC_ADDENDUM_URL,
    local_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load OpenNGC rows from the canonical CSVs (URL or local dir)."""
    sources: List[Tuple[str, str]] = []
    if local_dir is not None:
        ngc_path = os.path.join(local_dir, "NGC.csv")
        add_path = os.path.join(local_dir, "addendum.csv")
        if not os.path.isfile(ngc_path):
            raise FileNotFoundError(f"OpenNGC file not found: {ngc_path}")
        with open(ngc_path, "r", encoding="utf-8") as fh:
            sources.append(("NGC.csv", fh.read()))
        if os.path.isfile(add_path):
            with open(add_path, "r", encoding="utf-8") as fh:
                sources.append(("addendum.csv", fh.read()))
    else:
        sources.append(("NGC.csv", _http_get_text(ngc_url)))
        try:
            sources.append(("addendum.csv", _http_get_text(addendum_url)))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] could not download addendum.csv: {exc}",
                file=sys.stderr,
            )

    rows: List[Dict[str, str]] = []
    for label, text in sources:
        reader = csv.DictReader(io.StringIO(text), delimiter=OPENNGC_DELIMITER)
        if reader.fieldnames is None:
            raise RuntimeError(f"{label}: no CSV header found")
        lower_map = {h.lower().strip(): h for h in reader.fieldnames}

        def need(col: str) -> str:
            try:
                return lower_map[col]
            except KeyError as exc:
                raise RuntimeError(
                    f"{label}: missing expected column '{col}' "
                    f"(have: {reader.fieldnames})"
                ) from exc

        name_k = need(COL_NAME)
        common_k = need(COL_COMMON)
        m_k = lower_map.get(COL_M)
        ngc_k = lower_map.get(COL_NGC)
        ic_k = lower_map.get(COL_IC)

        for row in reader:
            rows.append(
                {
                    "name": (row.get(name_k) or "").strip(),
                    "common": (row.get(common_k) or "").strip(),
                    "m": (row.get(m_k) or "").strip() if m_k else "",
                    "ngc": (row.get(ngc_k) or "").strip() if ngc_k else "",
                    "ic": (row.get(ic_k) or "").strip() if ic_k else "",
                    "_source_file": label,
                }
            )
    return rows


_OPENNGC_NAME_RE = re.compile(
    r"^(?P<prefix>NGC|IC)(?P<num>\d+)(?P<suffix>[A-Z]?)(?:_\d+)?$"
)


def _parse_openngc_name(name: str) -> Optional[Tuple[str, str]]:
    """Parse 'NGC0001' -> ('NGC', '1'); None for components/non-NGC."""
    m = _OPENNGC_NAME_RE.match(name)
    if not m or "_" in name:
        return None
    return m.group("prefix"), str(int(m.group("num"))) + m.group("suffix")


def _split_common_names(raw: str) -> List[str]:
    if not raw:
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


def _build_openngc_index(
    rows: Iterable[Dict[str, str]],
) -> Dict[Tuple[str, str], List[str]]:
    """(catalogue, designation) -> [common_names]."""
    index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    seen: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def add(key: Tuple[str, str], names: List[str]) -> None:
        bucket = index[key]
        s = seen[key]
        for n in names:
            k = _norm_name(n)
            if k and k not in s:
                s.add(k)
                bucket.append(n)

    for row in rows:
        commons = _split_common_names(row["common"])
        if not commons:
            continue
        parsed = _parse_openngc_name(row["name"])
        if parsed is not None:
            add(parsed, commons)
        if row["m"]:
            add(("M", _norm_designation(row["m"])), commons)
        if row["ngc"]:
            add(("NGC", _norm_designation(row["ngc"])), commons)
        if row["ic"]:
            add(("IC", _norm_designation(row["ic"])), commons)
    return index


def openngc_additions(
    existing_by_dso: Dict[str, Set[str]],
    xrefs_by_dso: Dict[str, List[Tuple[str, str]]],
    ngc_url: str = OPENNGC_NGC_URL,
    addendum_url: str = OPENNGC_ADDENDUM_URL,
    local_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """Return dsoid -> [new names not already in existing_by_dso]."""
    print("  loading OpenNGC CSVs...")
    rows = load_openngc_csv(ngc_url, addendum_url, local_dir)
    print(f"  loaded {len(rows):,} rows")

    index = _build_openngc_index(rows)
    print(f"  index keys: {len(index):,}")

    out: Dict[str, List[str]] = {}
    for dsoid, keys in xrefs_by_dso.items():
        proposed: List[str] = []
        proposed_seen: Set[str] = set()
        existing = existing_by_dso.get(dsoid, set())
        for key in keys:
            for name in index.get(key, ()):
                k = _norm_name(name)
                if k in existing or k in proposed_seen:
                    continue
                proposed_seen.add(k)
                proposed.append(name)
        if proposed:
            out[dsoid] = proposed
            if verbose:
                print(f"  [OpenNGC] {dsoid}: +{proposed}")
    return out


# ---------------------------------------------------------------------------
# SIMBAD provider
# ---------------------------------------------------------------------------


def _simbad_candidate_ids(xrefs: List[Tuple[str, str]]) -> List[str]:
    """Build all candidate SIMBAD identifiers for a DSO from its cross-refs."""
    out: List[str] = []
    seen_keys: Set[str] = set()
    for cat in ("M", "NGC", "IC"):
        for c, d in xrefs:
            if c == cat:
                sid = f"{cat} {d}"
                k = _simbad_norm_key(sid)
                if k not in seen_keys:
                    seen_keys.add(k)
                    out.append(sid)
    return out


# A batch that fails after its retries is queried again in this many pieces.
_SPLIT_PIECES = 6


def _run_batch(batch: List, run, sleep_between: float, what: str):
    """Query `batch` with run(items) -> result.

    If that fails after its retries, query the batch again in _SPLIT_PIECES
    smaller pieces (each with its own retries), so one heavy batch - many
    labels, or a busy moment on the server - doesn't lose the whole batch.
    Only one level of splitting, so a server that is down costs a bounded
    number of extra requests.

    Returns (results, failed pieces, whether it was split).
    """
    try:
        return [run(batch)], 0, False
    except Exception as exc:  # noqa: BLE001
        if len(batch) == 1:
            print(f" FAILED: {exc}")
            return [], 1, False
        size = -(-len(batch) // _SPLIT_PIECES)
        pieces = [batch[i:i + size] for i in range(0, len(batch), size)]
        print(f" failed ({exc}); retrying in {len(pieces)} pieces")

    results: List = []
    failed = 0
    for i, piece in enumerate(pieces, 1):
        if sleep_between > 0:
            time.sleep(sleep_between)
        print(
            f"    piece {i}/{len(pieces)} ({len(piece)} {what})...",
            end="",
            flush=True,
        )
        try:
            results.append(run(piece))
            print(" ok")
        except Exception as exc:  # noqa: BLE001
            print(f" FAILED: {exc}")
            failed += 1
    return results, failed, True


def _simbad_tap_csv(adql: str, timeout: float = 90.0) -> str:
    """POST an ADQL query to SIMBAD's sync TAP endpoint; return CSV body."""
    body = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": adql,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        SIMBAD_TAP_SYNC_URL,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/csv",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _simbad_tap_csv_with_retry(
    adql: str,
    attempts: int = 3,
    initial_backoff: float = 5.0,
    timeout: float = 90.0,
) -> str:
    """Retry wrapper for SIMBAD TAP."""
    last_exc: Optional[Exception] = None
    backoff = initial_backoff
    for i in range(attempts):
        try:
            return _simbad_tap_csv(adql, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i == attempts - 1:
                break
            print(
                f"  [warn] SIMBAD request failed ({exc}); "
                f"retry {i + 1}/{attempts - 1} in {backoff:.0f}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"SIMBAD query failed after {attempts} attempts"
    ) from last_exc


def simbad_additions(
    existing_by_dso: Dict[str, Set[str]],
    xrefs_by_dso: Dict[str, List[Tuple[str, str]]],
    batch_size: int = 200,
    sleep_between: float = 1.0,
    verbose: bool = False,
    _query_fn=None,
) -> Tuple[Dict[str, List[str]], int]:
    """Query SIMBAD for NAME identifiers; return (dsoid -> [new names],
    number of failed batches)."""
    if _query_fn is None:
        _query_fn = _simbad_tap_csv_with_retry

    key_to_dsoids: Dict[str, Set[str]] = defaultdict(set)
    send_ids: List[str] = []
    send_ids_seen: Set[str] = set()

    for dsoid, xrefs in xrefs_by_dso.items():
        for sid in _simbad_candidate_ids(xrefs):
            k = _simbad_norm_key(sid)
            key_to_dsoids[k].add(dsoid)
            if sid not in send_ids_seen:
                send_ids_seen.add(sid)
                send_ids.append(sid)

    send_ids.sort()
    print(
        f"  built {len(send_ids):,} candidate SIMBAD IDs covering "
        f"{len({d for ds in key_to_dsoids.values() for d in ds}):,} DSOs"
    )
    if not send_ids:
        return {}, 0

    total_batches = (len(send_ids) + batch_size - 1) // batch_size
    print(
        f"  querying SIMBAD in {total_batches} batch(es) of up to "
        f"{batch_size} IDs..."
    )

    names_by_key: Dict[str, List[str]] = defaultdict(list)
    names_by_key_seen: Dict[str, Set[str]] = defaultdict(set)
    sample_returned_forms: List[str] = []
    failed = 0

    def run(ids: List[str]) -> str:
        in_list = ", ".join("'" + s.replace("'", "''") + "'" for s in ids)
        return _query_fn(
            "SELECT i1.id AS input_id, i2.id AS name_id "
            "FROM ident AS i1 "
            "JOIN ident AS i2 ON i1.oidref = i2.oidref "
            f"WHERE i1.id IN ({in_list}) "
            "AND i2.id LIKE 'NAME %'"
        )

    for b_idx in range(total_batches):
        batch = send_ids[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(
            f"  batch {b_idx + 1}/{total_batches} ({len(batch)} IDs)...",
            end="",
            flush=True,
        )
        bodies, batch_failed, split = _run_batch(batch, run, sleep_between, "IDs")
        failed += batch_failed
        if not bodies:
            continue

        rows = [r for body in bodies for r in csv.DictReader(io.StringIO(body))]
        batch_rows = 0
        for row in rows:
            input_id_raw = (row.get("input_id") or "").strip()
            name_id = (row.get("name_id") or "").strip()
            if not input_id_raw or not name_id.startswith("NAME "):
                continue
            friendly = name_id[len("NAME "):].strip()
            if not friendly:
                continue
            key = _simbad_norm_key(input_id_raw)
            if len(sample_returned_forms) < 5:
                sample_returned_forms.append(input_id_raw)
            nkey = _norm_name(friendly)
            if nkey in names_by_key_seen[key]:
                continue
            names_by_key_seen[key].add(nkey)
            names_by_key[key].append(friendly)
            batch_rows += 1
        if split:
            print(f"    batch {b_idx + 1} done ({batch_rows} NAME rows)")
        else:
            print(f" ok ({batch_rows} NAME rows)")

        if sleep_between > 0 and b_idx + 1 < total_batches:
            time.sleep(sleep_between)

    matched_keys = set(names_by_key)
    total_keys = len(key_to_dsoids)
    ratio = len(matched_keys) / max(total_keys, 1)
    if ratio < 0.01 and total_keys > 100:
        print(
            f"  [warn] only {len(matched_keys)} of {total_keys:,} "
            "candidate IDs matched any NAME row -- this is unusually low. "
            "Sample raw input_id values returned by SIMBAD:"
        )
        for s in sample_returned_forms:
            print(f"    {s!r}  (normalized: {_simbad_norm_key(s)!r})")

    per_dso: Dict[str, List[str]] = defaultdict(list)
    per_dso_seen: Dict[str, Set[str]] = defaultdict(set)
    for key, names in names_by_key.items():
        for dsoid in key_to_dsoids.get(key, ()):
            existing = existing_by_dso.get(dsoid, set())
            for n in names:
                k = _norm_name(n)
                if k in existing or k in per_dso_seen[dsoid]:
                    continue
                per_dso_seen[dsoid].add(k)
                per_dso[dsoid].append(n)

    out: Dict[str, List[str]] = {}
    for dsoid in sorted(per_dso):
        names = per_dso[dsoid]
        if names:
            out[dsoid] = names
            if verbose:
                print(f"  [SIMBAD] {dsoid}: +{names}")
    return out, failed


# ---------------------------------------------------------------------------
# Wikidata provider
# ---------------------------------------------------------------------------
#
# Strategy
# --------
# * Each DSO has zero or more (catalogue, designation) cross-refs in the
#   user's DB. For each such pair we know the catalogue's Wikidata Q-ID
#   (Messier=Q14530, NGC=Q14534, IC=Q741672).
# * We send batched SPARQL queries with a VALUES (?catalog ?codeNorm) {...}
#   block. Each binding identifies a (catalog, code) pair we want.
# * Wikidata's P528 "catalog code" data is INCONSISTENT: some items store
#   bare numbers like "4736", others store prefixed forms like "NGC 4736".
#   The query uses a server-side normalization that strips spaces and the
#   catalog prefix from the stored value before comparing to ours.
# * For each matched item we collect rdfs:label and skos:altLabel in the
#   chosen language, then clean them and drop anything that isn't a
#   friendly name (_reject_reason), so "M 94", "New General Catalogue 4736"
#   and "Mrk 590" don't end up as NAMEs.
#
# Output: dsoid -> [friendly names]


def _wikidata_catalog_qid(catalogue: str) -> Optional[str]:
    if catalogue == "M":
        return WIKIDATA_Q_MESSIER
    if catalogue == "NGC":
        return WIKIDATA_Q_NGC
    if catalogue == "IC":
        return WIKIDATA_Q_IC
    return None


def _wikidata_code_variants(catalogue: str, designation: str) -> List[str]:
    """
    Build the strings that might appear as P528 values for this object.

    Wikidata is inconsistent: some items store bare numbers ("4736"), others
    have prefixed forms ("NGC 4736") or no-space forms ("NGC4736"). To get
    fast index-backed lookups, we send all three forms as literal values in
    the SPARQL VALUES clause and let the query engine do direct equality
    matching instead of computing REPLACE() over the whole P528 table.

    Letter suffixes (e.g. "67A") are uppercased; the prefix is the catalog
    prefix as commonly stored in Wikidata (M for Messier).
    """
    desig = designation.strip()
    prefix = catalogue  # "M", "NGC", or "IC"
    return [
        desig,                       # "4736"
        f"{prefix} {desig}",         # "NGC 4736"
        f"{prefix}{desig}",          # "NGC4736"
    ]


def _sparql_post_json(
    sparql: str,
    timeout: float = 90.0,
) -> Dict:
    """POST a SPARQL query to query.wikidata.org; return parsed JSON."""
    body = urllib.parse.urlencode(
        {"query": sparql, "format": "json"}
    ).encode("utf-8")
    req = urllib.request.Request(
        WIKIDATA_SPARQL_URL,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _sparql_post_json_with_retry(
    sparql: str,
    attempts: int = 3,
    initial_backoff: float = 5.0,
    timeout: float = 90.0,
) -> Dict:
    last_exc: Optional[Exception] = None
    backoff = initial_backoff
    for i in range(attempts):
        try:
            return _sparql_post_json(sparql, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i == attempts - 1:
                break
            print(
                f"  [warn] Wikidata request failed ({exc}); "
                f"retry {i + 1}/{attempts - 1} in {backoff:.0f}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"Wikidata query failed after {attempts} attempts"
    ) from last_exc


def _wikidata_sparql(values_block: str, language: str) -> str:
    """
    Build a SPARQL query optimized for Wikidata's TAP service.

    The VALUES clause binds (?catalog, ?codeNorm, ?codeStored) triples,
    where ?codeStored is one of the literal forms a P528 value might take
    (bare, prefixed-with-space, prefixed-without-space). The triple pattern
    `?stmt ps:P528 ?codeStored` then becomes a direct equality lookup
    against the indexed P528 triples -- no REPLACE/BIND, no full-column
    scan, no 504 timeouts.

    ?codeNorm is a label we carry through to the result so the Python
    side can map each row back to the (catalog, code-we-asked-for) pair.
    """
    return f"""
SELECT DISTINCT ?catalog ?codeNorm ?item ?label ?altLabel WHERE {{
  VALUES (?catalog ?codeNorm ?codeStored) {{
    {values_block}
  }}
  ?item p:P528 ?stmt .
  ?stmt ps:P528 ?codeStored .
  ?stmt pq:P972 ?catalog .
  OPTIONAL {{
    ?item rdfs:label ?label .
    FILTER(LANG(?label) = "{language}")
  }}
  OPTIONAL {{
    ?item skos:altLabel ?altLabel .
    FILTER(LANG(?altLabel) = "{language}")
  }}
}}
""".strip()


def wikidata_additions(
    existing_by_dso: Dict[str, Set[str]],
    xrefs_by_dso: Dict[str, List[Tuple[str, str]]],
    types_by_dso: Optional[Dict[str, str]] = None,
    language: str = "en",
    batch_size: int = 30,
    sleep_between: float = 1.0,
    verbose: bool = False,
    rejected: Optional[List[Tuple[str, str, str, str]]] = None,
    _query_fn=None,
) -> Tuple[Dict[str, List[str]], int]:
    """
    Query Wikidata for friendly names; return (dsoid -> [new names],
    number of failed batches).

    Names come from rdfs:label and skos:altLabel, cleaned by _clean_name.
    Ones that aren't friendly names (_reject_reason: "M 94", "PGC 43495",
    "New General Catalogue 4736", "R Mon"...) are appended to `rejected` as
    (dsoid, name, source label, reason) instead.
    """
    types_by_dso = types_by_dso or {}
    if _query_fn is None:
        _query_fn = _sparql_post_json_with_retry

    # Map (catalog Q, code-normalized) -> set of dsoids that need it.
    # codeNorm is an uppercase, space-stripped form (e.g. "4736", "94", "67A").
    key_to_dsoids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for dsoid, xrefs in xrefs_by_dso.items():
        for cat, desig in xrefs:
            qid = _wikidata_catalog_qid(cat)
            if qid is None:
                continue
            code_norm = re.sub(r"\s+", "", desig).upper()
            key_to_dsoids[(qid, code_norm)].add(dsoid)

    keys = sorted(key_to_dsoids)
    print(
        f"  built {len(keys):,} (catalog, code) pairs covering "
        f"{len({d for ds in key_to_dsoids.values() for d in ds}):,} DSOs"
    )
    if not keys:
        return {}, 0

    total_batches = (len(keys) + batch_size - 1) // batch_size
    print(
        f"  querying Wikidata in {total_batches} batch(es) of up to "
        f"{batch_size} pairs (lang={language})..."
    )

    # matched_key -> ordered list of cleaned labels (vetted per DSO below)
    names_by_key: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    names_by_key_seen: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    failed = 0

    def _add_name(key: Tuple[str, str], n: str) -> bool:
        n = _clean_name(n)
        nkey = _norm_name(n)
        if not nkey or nkey in names_by_key_seen[key]:
            return False
        names_by_key_seen[key].add(nkey)
        names_by_key[key].append(n)
        return True

    cat_to_prefix = {
        WIKIDATA_Q_MESSIER: "M",
        WIKIDATA_Q_NGC: "NGC",
        WIKIDATA_Q_IC: "IC",
    }

    def run(pairs: List[Tuple[str, str]]) -> Dict:
        # Build VALUES block. Each (catalog Q, codeNorm) pair from the user's
        # DB expands into N rows -- one per literal stored-form variant we
        # want to match against P528. The query engine then does direct
        # equality lookups, which are index-backed and fast.
        lines = []
        for qid, code in pairs:
            prefix = cat_to_prefix.get(qid, "")
            for variant in _wikidata_code_variants(prefix, code):
                esc_code = code.replace("\\", "\\\\").replace('"', '\\"')
                esc_var = variant.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(
                    f'    (wd:{qid} "{esc_code}" "{esc_var}")'
                )
        values_block = "\n".join(lines).strip()
        return _query_fn(_wikidata_sparql(values_block, language))

    for b_idx in range(total_batches):
        batch = keys[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(
            f"  batch {b_idx + 1}/{total_batches} ({len(batch)} pairs)...",
            end="",
            flush=True,
        )
        payloads, batch_failed, split = _run_batch(batch, run, sleep_between, "pairs")
        failed += batch_failed
        if not payloads:
            continue

        bindings = [
            b for p in payloads for b in p.get("results", {}).get("bindings", [])
        ]
        labels_added = 0
        for b in bindings:
            try:
                cat_uri = b["catalog"]["value"]  # http://www.wikidata.org/entity/Q14534
                code = b["codeNorm"]["value"]
            except KeyError:
                continue
            qid = cat_uri.rsplit("/", 1)[-1]
            key = (qid, code)
            for field in ("label", "altLabel"):
                if field in b and _add_name(key, b[field]["value"]):
                    labels_added += 1
        if split:
            print(f"    batch {b_idx + 1} done ({labels_added} labels)")
        else:
            print(f" ok ({labels_added} labels)")

        if sleep_between > 0 and b_idx + 1 < total_batches:
            time.sleep(sleep_between)

    # Map matched keys back to DSOs, dedup against existing NAMEs, and keep
    # only friendly names (whether a star designation is one depends on the
    # object's type, so this happens per DSO).
    per_dso: Dict[str, List[str]] = defaultdict(list)
    per_dso_seen: Dict[str, Set[str]] = defaultdict(set)
    for key, names in names_by_key.items():
        for dsoid in key_to_dsoids.get(key, ()):
            existing = existing_by_dso.get(dsoid, set())
            for n in names:
                k = _norm_name(n)
                if k in existing or k in per_dso_seen[dsoid]:
                    continue
                per_dso_seen[dsoid].add(k)
                reason = _reject_reason(n, types_by_dso.get(dsoid, ""))
                if reason:
                    if rejected is not None:
                        rejected.append((dsoid, n, LABEL_WIKIDATA, reason))
                    continue
                per_dso[dsoid].append(n)

    out: Dict[str, List[str]] = {}
    for dsoid in sorted(per_dso):
        names = per_dso[dsoid]
        if names:
            out[dsoid] = names
            if verbose:
                print(f"  [Wikidata] {dsoid}: +{names}")
    return out, failed


# ---------------------------------------------------------------------------
# Merge + provenance
# ---------------------------------------------------------------------------


def merge_additions(
    by_source: Dict[str, Dict[str, List[str]]],
) -> List[Tuple[str, str, str]]:
    """
    Combine multiple provider maps into rows: (dsoid, name, source_label).

    by_source keys are LABEL_OPENNGC / LABEL_SIMBAD / LABEL_WIKIDATA in the
    order they should be ingested (first source wins on spelling).

    source_label joins all contributing sources with '+', e.g.
    "OpenNGC+Wikidata" or "SIMBAD+Wikidata" or "OpenNGC+SIMBAD+Wikidata".
    """
    combined: Dict[str, Dict[str, Tuple[str, Set[str]]]] = defaultdict(dict)

    for label, m in by_source.items():
        for dsoid, names in m.items():
            for n in names:
                key = _norm_name(n)
                if key not in combined[dsoid]:
                    combined[dsoid][key] = (n, {label})
                else:
                    combined[dsoid][key][1].add(label)

    # Stable order of sources for the source_label string
    order = [LABEL_OPENNGC, LABEL_SIMBAD, LABEL_WIKIDATA]

    rows: List[Tuple[str, str, str]] = []
    for dsoid in sorted(combined):
        for _, (name, sources) in combined[dsoid].items():
            ordered = [s for s in order if s in sources]
            rows.append((dsoid, name, "+".join(ordered)))
    return rows


def drop_unwanted(
    rows: List[Tuple[str, str, str]],
    existing_by_dso: Dict[str, Set[str]],
    excluded: Set[Tuple[str, str]],
    skip_shared: bool = False,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]]:
    """
    Split merged rows into (kept, rejected); rejected rows carry a reason.

    Drops names on the exclusion list (load_exclusions), and - with
    skip_shared - names that would also belong to another object, already
    (in the source or another update DB) or in this run, so a name looked
    up finds one object. Excluded rows don't count as owners, so blocking a
    name on the wrong object lets the right one keep it.
    """
    candidates: List[Tuple[str, str, str]] = []
    rejected: List[Tuple[str, str, str, str]] = []
    for dsoid, name, sources in rows:
        k = _norm_name(name)
        if ("", k) in excluded or (dsoid, k) in excluded:
            rejected.append((dsoid, name, sources, "on exclusion list"))
        else:
            candidates.append((dsoid, name, sources))

    owners: Dict[str, Set[str]] = defaultdict(set)
    for dsoid, names in existing_by_dso.items():
        for k in names:
            owners[k].add(dsoid)
    for dsoid, name, _ in candidates:
        owners[_norm_name(name)].add(dsoid)

    kept: List[Tuple[str, str, str]] = []
    for dsoid, name, sources in candidates:
        k = _norm_name(name)
        others = owners[k] - {dsoid}
        if others and skip_shared:
            rejected.append(
                (dsoid, name, sources, "also names " + ", ".join(sorted(others)))
            )
            continue
        kept.append((dsoid, name, sources))
    return kept, rejected


def write_review_csv(
    path: str,
    kept: List[Tuple[str, str, str]],
    rejected: List[Tuple[str, str, str, str]],
) -> None:
    """Every added and skipped name, with the reason, for a quick look over."""
    lines = [(d, n, s, "added", "") for d, n, s in kept]
    lines += [(d, n, s, "skipped", r) for d, n, s, r in rejected]
    lines.sort(key=lambda r: (r[0], r[3], r[1].casefold()))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dsodetailid", "name", "sources", "status", "reason"])
        w.writerows(lines)


# ---------------------------------------------------------------------------
# Output database
# ---------------------------------------------------------------------------

UPDATE_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS cataloguenr (
    dsodetailid TEXT,
    catalogue   TEXT,
    designation TEXT,
    PRIMARY KEY (dsodetailid, catalogue, designation)
);

CREATE TABLE IF NOT EXISTS name_provenance (
    dsodetailid TEXT,
    designation TEXT,
    source      TEXT NOT NULL,
    PRIMARY KEY (dsodetailid, designation)
);

CREATE TABLE IF NOT EXISTS attribution (
    source TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (source, key)
);

CREATE TABLE IF NOT EXISTS import_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


OPENNGC_ATTRIBUTION: List[Tuple[str, str]] = [
    ("name", "OpenNGC"),
    ("author", "Mattia Verga"),
    ("url", "https://github.com/mattiaverga/OpenNGC"),
    ("license", "CC-BY-SA-4.0"),
    ("license_url", "https://creativecommons.org/licenses/by-sa/4.0/"),
    (
        "modifications",
        "Only the 'Common names' field was extracted. Names were "
        "re-associated to the target database's DSO identifiers via NGC, "
        "IC, and M cross-references, and de-duplicated against existing "
        "NAME entries.",
    ),
    (
        "upstream_acknowledgement",
        "OpenNGC incorporates data from NED (NASA/IPAC Extragalactic "
        "Database), HyperLEDA, SIMBAD (CDS, Strasbourg), and HEASARC.",
    ),
    (
        "redistribution_notice",
        "OpenNGC is licensed CC-BY-SA-4.0. Redistributions of content "
        "derived from OpenNGC must preserve this attribution and license "
        "the derived portion under CC-BY-SA-4.0 or a compatible license.",
    ),
]

SIMBAD_ATTRIBUTION: List[Tuple[str, str]] = [
    ("name", "SIMBAD"),
    ("author", "CDS, Strasbourg"),
    ("url", "https://simbad.cds.unistra.fr/simbad/"),
    ("license", "Academic / acknowledgement"),
    (
        "acknowledgement_text",
        "This research has made use of the SIMBAD database, operated at "
        "CDS, Strasbourg, France.",
    ),
    ("reference", "2000A&AS..143....9W (Wenger et al. 2000)"),
    (
        "modifications",
        "Only the vernacular identifiers in the NAME pseudo-catalog "
        "(SIMBAD `ident.id LIKE 'NAME %'`) were extracted, via batched "
        "ADQL queries against the TAP sync endpoint, and re-associated "
        "to the target database's DSO identifiers.",
    ),
    (
        "redistribution_notice",
        "SIMBAD does not impose a share-alike clause, but reuse in "
        "publications should include the acknowledgement text above "
        "and cite the Wenger et al. (2000) reference.",
    ),
]

WIKIDATA_ATTRIBUTION: List[Tuple[str, str]] = [
    ("name", "Wikidata"),
    ("author", "Wikidata contributors / Wikimedia Foundation"),
    ("url", "https://www.wikidata.org/"),
    ("license", "CC0 1.0 (Public Domain Dedication)"),
    (
        "license_url",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    (
        "modifications",
        "Items were located via P528 (catalog code) qualified by P972 "
        "(catalog). For each matched item, rdfs:label and skos:altLabel "
        "in the requested language were collected; values that look like "
        "catalog designations were filtered out so only friendly/common "
        "names remain.",
    ),
    (
        "redistribution_notice",
        "Wikidata structured data is released under CC0; no attribution "
        "is legally required, but acknowledgement is courteous and "
        "standard practice.",
    ),
]

ATTRIBUTION_BY_LABEL = {
    LABEL_OPENNGC: OPENNGC_ATTRIBUTION,
    LABEL_SIMBAD: SIMBAD_ATTRIBUTION,
    LABEL_WIKIDATA: WIKIDATA_ATTRIBUTION,
}


def write_update_db(
    output_path: str,
    rows: List[Tuple[str, str, str]],
    sources_used: List[str],
    run_meta: Dict[str, str],
) -> None:
    """Create (or replace) the update DB and populate it.

    Written to a temp file and swapped in, so a crash leaves the previous
    DB in place. The temp name doesn't end in .sqlite, so the app never
    attaches a half-written file.
    """
    tmp_path = output_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(UPDATE_DB_SCHEMA)

        conn.executemany(
            "INSERT INTO cataloguenr (dsodetailid, catalogue, designation) "
            "VALUES (?, ?, ?)",
            [(dsoid, NAME_CATALOGUE, name) for dsoid, name, _ in rows],
        )
        conn.executemany(
            "INSERT INTO name_provenance "
            "(dsodetailid, designation, source) VALUES (?, ?, ?)",
            rows,
        )

        attr_rows: List[Tuple[str, str, str]] = []
        for label in sources_used:
            for k, v in ATTRIBUTION_BY_LABEL.get(label, ()):
                attr_rows.append((label, k, v))
        conn.executemany(
            "INSERT OR REPLACE INTO attribution (source, key, value) "
            "VALUES (?, ?, ?)",
            attr_rows,
        )

        conn.executemany(
            "INSERT OR REPLACE INTO import_meta (key, value) VALUES (?, ?)",
            list(run_meta.items()),
        )
        conn.commit()
    except BaseException:
        conn.close()
        os.remove(tmp_path)
        raise
    conn.close()
    try:
        os.replace(tmp_path, output_path)
    except OSError:
        os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_sources(raw: str) -> List[str]:
    """Parse --source value; supports lists, 'all', and 'both' (= all)."""
    raw = raw.strip().lower()
    if raw in SOURCE_ALL_ALIAS or raw == "":
        return list(ALL_SOURCES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return list(ALL_SOURCES)
    invalid = [p for p in parts if p not in ALL_SOURCES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown source(s): {invalid}. Valid: "
            f"{ALL_SOURCES + SOURCE_ALL_ALIAS}"
        )
    # de-dup, preserve order, normalize to canonical sequence
    seen: Set[str] = set()
    out: List[str] = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a supplementary SQLite database of missing DSO NAME "
            "records from OpenNGC, SIMBAD, and/or Wikidata."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sources:\n"
            "  openngc   OpenNGC (CC-BY-SA-4.0)  -- requires share-alike\n"
            "  simbad    SIMBAD (academic ack)\n"
            "  wikidata  Wikidata (CC0, public domain)\n"
            "  all       All three (default)\n"
            "  Multiple: --source openngc,wikidata\n"
        ),
    )
    p.add_argument("--source-db", required=True,
                   help="Path to the existing DSO SQLite database (read-only).")
    p.add_argument("--output-db", default="DSO_update.sqlite",
                   help="Path for the generated update database "
                        "(default: DSO_update.sqlite)")
    p.add_argument("--source", default="all", type=_parse_sources,
                   help="Source(s) to use: openngc, simbad, wikidata, all, "
                        "or a comma-separated list (default: all).")
    # OpenNGC
    p.add_argument("--openngc-dir", default=None,
                   help="Load OpenNGC CSVs from a local directory instead "
                        "of downloading.")
    p.add_argument("--ngc-url", default=OPENNGC_NGC_URL,
                   help="Override URL for OpenNGC NGC.csv.")
    p.add_argument("--addendum-url", default=OPENNGC_ADDENDUM_URL,
                   help="Override URL for OpenNGC addendum.csv.")
    # SIMBAD
    p.add_argument("--simbad-batch-size", type=int, default=200,
                   help="IDs per SIMBAD TAP request (default: 200).")
    p.add_argument("--simbad-sleep", type=float, default=1.0,
                   help="Sleep between SIMBAD batches (default: 1.0).")
    # Wikidata
    p.add_argument("--wikidata-language", default="en",
                   help="Language code for Wikidata labels (default: en).")
    p.add_argument("--wikidata-batch-size", type=int, default=30,
                   help="(catalog,code) pairs per SPARQL request "
                        "(default: 30). Each pair expands to ~3 stored-form "
                        "variants. Lower this further if you hit 504s.")
    p.add_argument("--wikidata-sleep", type=float, default=1.0,
                   help="Sleep between Wikidata batches (default: 1.0).")
    # Vetting
    p.add_argument("--exclude-file", default=None,
                   help="Names never to add, one per line (default: "
                        "name_exclusions.txt next to this script, if any).")
    p.add_argument("--skip-shared-names", action="store_true",
                   help="Skip names that another object already has or "
                        "gets in this run, so each name finds one object.")
    p.add_argument("--review-csv", default=None,
                   help="Write every added and skipped name, with the "
                        "reason, to this CSV.")
    # Run control
    p.add_argument("--allow-partial", action="store_true",
                   help="Write the output even if some query batches "
                        "failed (by default the old output is kept).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute the diff and print a summary; no output.")
    p.add_argument("--verbose", action="store_true",
                   help="Print every addition as it is found.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.isfile(args.source_db):
        print(f"error: source DB not found: {args.source_db}", file=sys.stderr)
        return 2

    exclude_file = args.exclude_file
    if exclude_file is None:
        exclude_file = DEFAULT_EXCLUDE_FILE
    elif not os.path.isfile(exclude_file):
        print(f"error: exclude file not found: {exclude_file}", file=sys.stderr)
        return 2
    excluded = load_exclusions(exclude_file)

    sources: List[str] = args.source  # already a list
    print(
        f"DSO_Name_Updater.py v{VERSION} -- sources: "
        f"{', '.join(LABEL_BY_SOURCE[s] for s in sources)}"
    )

    print("[1/N] Reading source DB cross-references...")
    existing_by_dso, xrefs_by_dso, types_by_dso = load_source_info(args.source_db)
    print(
        f"      {len(existing_by_dso):,} DSOs with existing NAME rows; "
        f"{len(xrefs_by_dso):,} DSOs with NGC/IC/M cross-refs"
    )
    other_names, other_files = load_other_update_names(
        args.output_db, args.source_db
    )
    for dsoid, names in other_names.items():
        existing_by_dso[dsoid] |= names
    if other_files:
        print(
            f"      also skipping names already in: {', '.join(other_files)}"
        )
    if excluded:
        print(f"      {len(excluded):,} names on the exclusion list")

    by_source: Dict[str, Dict[str, List[str]]] = {}
    rejected: List[Tuple[str, str, str, str]] = []
    failed_batches = 0

    step = 2
    if SOURCE_OPENNGC in sources:
        print(f"[{step}/N] OpenNGC lookup:")
        by_source[LABEL_OPENNGC] = openngc_additions(
            existing_by_dso, xrefs_by_dso,
            ngc_url=args.ngc_url,
            addendum_url=args.addendum_url,
            local_dir=args.openngc_dir,
            verbose=args.verbose,
        )
        print(
            f"      OpenNGC found new names for "
            f"{len(by_source[LABEL_OPENNGC]):,} DSOs"
        )
        step += 1

    if SOURCE_SIMBAD in sources:
        print(f"[{step}/N] SIMBAD lookup:")
        by_source[LABEL_SIMBAD], failed = simbad_additions(
            existing_by_dso, xrefs_by_dso,
            batch_size=args.simbad_batch_size,
            sleep_between=args.simbad_sleep,
            verbose=args.verbose,
        )
        failed_batches += failed
        print(
            f"      SIMBAD found new names for "
            f"{len(by_source[LABEL_SIMBAD]):,} DSOs"
        )
        step += 1

    if SOURCE_WIKIDATA in sources:
        print(f"[{step}/N] Wikidata lookup:")
        by_source[LABEL_WIKIDATA], failed = wikidata_additions(
            existing_by_dso, xrefs_by_dso,
            types_by_dso=types_by_dso,
            language=args.wikidata_language,
            batch_size=args.wikidata_batch_size,
            sleep_between=args.wikidata_sleep,
            verbose=args.verbose,
            rejected=rejected,
        )
        failed_batches += failed
        print(
            f"      Wikidata found new names for "
            f"{len(by_source[LABEL_WIKIDATA]):,} DSOs"
        )
        step += 1

    merged = merge_additions(by_source)
    rows, dropped = drop_unwanted(
        merged, existing_by_dso, excluded, args.skip_shared_names
    )
    rejected += dropped
    affected = len({r[0] for r in rows})
    print(
        f"      merged: {len(rows):,} NAME rows across {affected:,} objects; "
        f"skipped {len(rejected):,} names"
    )
    if args.verbose:
        for dsoid, name, label, reason in rejected:
            print(f"  [skip] {dsoid}: {name!r} ({label}: {reason})")

    if args.review_csv:
        write_review_csv(args.review_csv, rows, rejected)
        print(f"      review list written to {args.review_csv}")

    if failed_batches:
        print(
            f"[warn] {failed_batches} query batch(es) failed, so names may be "
            "missing.",
            file=sys.stderr,
        )
        if not args.dry_run and not args.allow_partial:
            print(
                f"error: {args.output_db} left as it was. Re-run, or pass "
                "--allow-partial to write the incomplete result.",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print("Dry-run: no output written.")
        return 0

    sources_used = [LABEL_BY_SOURCE[s] for s in sources]

    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": f"DSO_Name_Updater.py v{VERSION}",
        "source_db": os.path.abspath(args.source_db),
        "sources_used": ", ".join(sources_used),
        "rows_inserted": str(len(rows)),
        "names_skipped": str(len(rejected)),
        "failed_batches": str(failed_batches),
        "openngc_ngc_url": (
            args.ngc_url if SOURCE_OPENNGC in sources else ""
        ),
        "openngc_addendum_url": (
            args.addendum_url if SOURCE_OPENNGC in sources else ""
        ),
        "openngc_local_dir": (
            os.path.abspath(args.openngc_dir)
            if SOURCE_OPENNGC in sources and args.openngc_dir
            else ""
        ),
        "simbad_tap_url": (
            SIMBAD_TAP_SYNC_URL if SOURCE_SIMBAD in sources else ""
        ),
        "wikidata_sparql_url": (
            WIKIDATA_SPARQL_URL if SOURCE_WIKIDATA in sources else ""
        ),
        "wikidata_language": (
            args.wikidata_language if SOURCE_WIKIDATA in sources else ""
        ),
    }

    print(f"Writing {args.output_db}...")
    try:
        write_update_db(args.output_db, rows, sources_used, run_meta)
    except PermissionError as exc:
        print(
            f"error: could not replace {args.output_db} ({exc}). Close "
            "CosmosCollection (it keeps update DBs open) and re-run.",
            file=sys.stderr,
        )
        return 1
    print(f"      done. {len(rows):,} rows inserted.")

    print()
    print("Attribution reminder:")
    if LABEL_OPENNGC in sources_used:
        print("  OpenNGC (Mattia Verga), CC-BY-SA-4.0 -- preserve "
              "attribution and share-alike on redistribution.")
        print("    https://github.com/mattiaverga/OpenNGC")
    if LABEL_SIMBAD in sources_used:
        print("  SIMBAD (CDS, Strasbourg) -- include the ack text in "
              "any publications.")
        print("    https://simbad.cds.unistra.fr/simbad/")
    if LABEL_WIKIDATA in sources_used:
        print("  Wikidata, CC0 (public domain) -- attribution courteous "
              "but not required.")
        print("    https://www.wikidata.org/")
    print("  Full details are stored in the 'attribution' table of the "
          "update DB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
