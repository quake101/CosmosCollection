#!/usr/bin/env python3
"""
Library Scanner
Finds the sessions (sub-exposures) and finished images in whole folder trees for
the Import Wizard and groups them by DSO. No Qt - it runs on worker threads, or
from the command line to check how a library is understood:

    python LibraryScanner.py "H:/Astrophotography" [--workers 32] [--full] [--new-only]

Stages:
  1. scan_library(): walk the folders and read FITS/XISF headers on thread pools.
     A network drive is latency-bound (~60 ms per file open), so many reads in
     flight matter far more than CPU. Stacking work folders (WBPP calibrated/
     registered/..., Siril process/) are noted but never entered.
  2. classify(): what every file is - a sub, a final image, a processing file,
     a preview of a sub, ...
  3. build_library(): name each group's DSO, split lights into sessions, attach
     nearby calibration, pair final images with sessions and choose what is
     checked by default.
  4. import_library(): write what the review leaves checked into the Session
     Manager and the gallery.

ScanIndex remembers what earlier scans found (each file's size, modified time
and header), so a rescan only reads new or changed files and can tell which
files are new since the last scan.
"""

import os
import re
import json
import math
import time
import logging
import sqlite3
import threading
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import MountDetection
import SessionFileScanner
import SessionObservations
from SessionObservations import make_sub_key, parse_obs_datetime, night_date_for
from DatabaseManager import add_user_image_row

logger = logging.getLogger(__name__)

FITS_EXTENSIONS = SessionFileScanner.FITS_EXTENSIONS
XISF_EXTENSIONS = SessionFileScanner.XISF_EXTENSIONS
HEADER_EXTENSIONS = SessionFileScanner.SUPPORTED_EXTENSIONS
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef'}

# Which of a final image's formats the collection points at (the others are listed as its variants)
FORMAT_PREFERENCE = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.xisf', '.fits', '.fit', '.fts']

# Words in a file name (or its folder's) that mark a finished image - tier 1
DEFAULT_FINAL_KEYWORDS = ("processed", "final", "export", "exported", "edit", "edited",
                          "finished", "stretched", "result")
# Words that mark a stack or integration - tier 2, only checked when nothing finished exists
DEFAULT_STACK_KEYWORDS = ("stacked", "stack", "integration", "masterlight", "autosave")
# Words that mark a by-product of processing - listed, never checked by default
VARIANT_WORDS = {"starless", "starmask", "stars", "mask", "annotated", "background", "settings",
                 "red", "green", "blue", "test", "resampled"}
VARIANT_KEYS = {re.sub(r"(.)\1+", r"\1", w): w for w in VARIANT_WORDS}   # As _words() writes them
# Words for a light edit of an image or a processing step it went through - left
# out when telling its versions apart, so 'M51_v1_Cropped' is a version of
# 'M51_v3' and 'M51_SIRIL_GraXpert' of 'M51_SIRIL', not images of their own.
# Letter boundaries, not \b: '_' is a word character
EDIT_WORDS = re.compile(r"(?<![a-z])(cropped|crop|resized|resize|downsized|downsampled|small|smaller|web|rotated|"
                        r"flipped|stretched|graxpert|astap|siril|seestar|blurxterminator|noisexterminator|"
                        r"starxterminator|bxt|nxt|ap|af)(?![a-z])")

# Stacking work folders that are never entered - they hold copies of subs. The
# app's own processing workspaces (<DSO>_<date>_PixInsight / _Siril) also skip frames/.
DEFAULT_WORK_FOLDERS = ("calibrated", "cosmetized", "debayered", "registered", "fastintegration",
                        "logs", "ln_reference_frame_data", "process")
WORKSPACE_FOLDER = re.compile(r"_(pixinsight|siril)(_\d+)?$", re.IGNORECASE)
WORKSPACE_WORK_FOLDERS = {"frames"}

# Pipeline copies of subs, recognized by name alone (headers aren't read):
# Siril's pp_/r_pp_/bkg_ prefixes, WBPP's _c/_cc/_d/_r/_a/_n suffixes and its
# local-normalization reference. Case-sensitive: both tools write lowercase.
PIPELINE_NAME = re.compile(r"^(?:(?:r|bkg)_)*pp_|^LN_Reference_|(?:_(?:c|cc|d|r|a|n))+$")
CALIBRATION_MASTER_NAME = re.compile(r"master[\s_-]?(dark|flat|bias|offset|darkflat)"
                                     r"|^(dark|flat|bias|offset|darkflat)s?[\s_-]?stacked", re.IGNORECASE)
# Calibration frames some capture drivers save as IMAGETYP=LIGHT, told apart by
# their folder or OBJECT name
CALIBRATION_WORDS = {"dark": "Dark", "darks": "Dark", "flat": "Flat", "flats": "Flat",
                     "bias": "Bias", "biases": "Bias", "offset": "Bias"}
THUMBNAIL_NAME = re.compile(r"(_thn|thumbnail)$", re.IGNORECASE)
# Software's own names for a stack: a capture app's stack of a night (Seestar
# 'Stacked_81_M 31_10.0s_...') and the Siril hand-off's result_<filter>.fit
# (ProcessingHandoff) - stacks, even though "result" otherwise marks a finished image
TOOL_STACK_NAME = re.compile(r"^(stacked_\d+_|result_)", re.IGNORECASE)
# Names software gives its output by default - a step along the way, not a
# finished image, whatever words follow: PixInsight's window never saved under a
# name of its own ('Unsaved star recomposition result') and Siril's starless +
# stars recombination ('Star recomposition result'). A name the user gave it
# ('Processed star recomposition result') doesn't start with these.
DEFAULT_OUTPUT_NAME = re.compile(r"^(unsaved\b|star[\s_-]*recomposition[\s_-]*result)", re.IGNORECASE)

# The header keywords kept (in memory and in the scan index): what classifying,
# grouping and importing use, not the whole header
HEADER_KEYWORDS = frozenset(SessionFileScanner.FITS_KEYWORDS) | {
    'EXPOSURE', 'STACKCNT', 'NCOMBINE', 'TOTALEXP', 'LIVETIME', 'DATE-LOC', 'DATE-END',
    'SITELONG', 'SITELAT', 'LONG-OBS', 'LAT-OBS', 'CRVAL1', 'CRVAL2', 'NAXIS1', 'NAXIS2', 'NAXIS3',
    'YPIXSZ', 'FOCRATIO', 'CREATOR', '_XISF_IMAGETYPE',
}
# Bump when HEADER_KEYWORDS (or anything stored per file) changes: indexed
# headers are then read again, while first-seen dates are kept
INDEX_VERSION = 1

WALK_WORKERS = 8
DEFAULT_READ_WORKERS = 32
DEFAULT_SESSION_GAP_DAYS = 60

# Objects whose pointing is this close to their center count as "pointed at"
# even when the catalog size is tiny (degrees)
POSITION_MATCH_RADIUS = 0.25
# A name-matched DSO this far from where the subs were pointed is flagged
POSITION_MISMATCH_DEG = 3.0

NOT_DSO_TYPES = {"1STAR", "2STAR", "3STAR", "4STAR", "NONEX"}
CATALOG_RANK = {"M": 0, "NGC": 1, "IC": 2, "Sh2": 3, "Barnard": 3, "LDN": 3, "vdB": 3, "LBN": 4,
                "Caldwell": 5, "Abell": 5, "Arp": 5, "Collinder": 5, "Melotte": 5, "Ced": 5}

# Catalog prefixes as people write them -> the catalog's own name
CATALOG_ALIASES = {
    "M": "M", "MESSIER": "M", "NGC": "NGC", "IC": "IC",
    "SH2": "Sh2", "SHARPLESS": "Sh2", "LDN": "LDN", "LBN": "LBN",
    "B": "Barnard", "BARNARD": "Barnard", "VDB": "vdB", "ABELL": "Abell", "ARP": "Arp",
    "CALDWELL": "Caldwell", "CR": "Collinder", "COLLINDER": "Collinder", "MEL": "Melotte",
    "MELOTTE": "Melotte", "CED": "Ced", "RCW": "RCW", "GUM": "Gum", "UGC": "UGC",
    "HCG": "Hickson", "HICKSON": "Hickson", "TRUMPLER": "Trumpler", "BERKELEY": "Berkeley",
    "STOCK": "Stock", "KING": "King", "HENIZE": "Henize",
}
_ALIAS_PATTERN = "|".join(sorted((re.escape(a) if a != "SH2" else r"SH\s*2" for a in CATALOG_ALIASES),
                                 key=len, reverse=True))
DESIGNATION = re.compile(rf"(?<![A-Z0-9])({_ALIAS_PATTERN})[\s_.\-]*(\d+)([A-Z]?)(?![0-9])")
# Solar-system targets move across the sky, so their pointing names whatever
# galaxy happened to be behind them - never look them up by position
# Letter boundaries, not \b: '_' is a word character, and '_Moon' is a folder name
SOLAR_SYSTEM = re.compile(r"(?<![a-z])(comets?|moon|lunar|sun|solar|planet|planets|planetary|mercury|venus|mars|"
                          r"jupiter|saturn|uranus|neptune)(?![a-z])|(?<![a-z0-9])[CP]/?\s?\d{4}\s?[A-Z]{1,2}\d*(?![a-z0-9])",
                          re.IGNORECASE)
# A share of a folder's subs pointed this close to a named DSO confirms the name
POSITION_CONFIRM_SHARE = 0.1
# OBJECT values that name nothing
PLACEHOLDER_OBJECTS = {"", "UNKNOWN", "OBJECT", "TARGET", "NONE", "FLATWIZARD", "FLAT", "DARK", "BIAS"}


@dataclass
class ScanRules:
    final_keywords: tuple = DEFAULT_FINAL_KEYWORDS
    stack_keywords: tuple = DEFAULT_STACK_KEYWORDS
    work_folders: tuple = DEFAULT_WORK_FOLDERS
    session_gap_days: int = DEFAULT_SESSION_GAP_DAYS
    workers: int = DEFAULT_READ_WORKERS


@dataclass(eq=False)   # Identity: the same file/session, not equal-looking ones
class FileRecord:
    path: str
    size: int = 0
    mtime_ns: int = 0
    header: Optional[dict] = None
    error: Optional[str] = None
    # Set by classify()
    kind: str = ""          # sub, final, variant, processing, preview, duplicate, raw, unreadable
    frame_type: str = ""    # subs: Light/Dark/Flat/Bias
    tier: int = 0           # finals: 1 finished, 2 stack/integration, 3 possible
    reason: str = ""
    variant_of: Optional["FileRecord"] = None
    variants: list = field(default_factory=list)
    is_new: bool = False    # Not in the scan index before this scan
    imported: bool = False  # A light already in a Session Manager session
    mount: Optional[object] = None  # Lights: their night's MountDetection.NightMount

    def __post_init__(self):
        self.folder, self.name = os.path.split(self.path)
        self.stem, ext = os.path.splitext(self.name)
        self.ext = ext.lower()


@dataclass
class ScanResult:
    roots: list
    records: list = field(default_factory=list)
    skipped_folders: list = field(default_factory=list)   # (path, reason)
    folders_listed: int = 0
    folders_queued: int = 0     # Listed or waiting to be: the walk is done when it equals folders_listed
    files_seen: int = 0
    headers_queued: int = 0     # Headers to read (not in the index): progress is headers_read of these
    headers_read: int = 0
    headers_cached: int = 0
    new_files: int = 0
    seconds: float = 0.0
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Stage 1: walk + read headers
# ---------------------------------------------------------------------------

def _list_dir(path):
    """(path, files [(path, size, mtime_ns)], subdirs [path], error) of one folder.
    On Windows the size/mtime come with the listing, so no extra round trips."""
    files, subdirs = [], []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        stat = entry.stat(follow_symlinks=False)
                        files.append((entry.path, stat.st_size, stat.st_mtime_ns))
                except OSError:
                    continue
    except OSError as e:
        return path, files, subdirs, str(e)
    return path, files, subdirs, None


def _read_header(record):
    try:
        if record.ext in FITS_EXTENSIONS:
            header = SessionFileScanner.read_fits_header_fast(record.path)
        else:
            header = SessionFileScanner.extract_xisf_header(record.path, keywords=None)
        return record, {k: v for k, v in header.items() if k in HEADER_KEYWORDS}, None
    except Exception as e:
        return record, None, str(e) or type(e).__name__


def skip_reason(folder_path, rules):
    """Why a folder isn't entered ('processing', 'rejected'), or None."""
    name = os.path.basename(folder_path).lower()
    if name == SessionFileScanner.REJECTED_FOLDER.lower():
        return "rejected"
    if name in {w.lower() for w in rules.work_folders}:
        return "processing"
    if name in WORKSPACE_WORK_FOLDERS and WORKSPACE_FOLDER.search(os.path.basename(os.path.dirname(folder_path))):
        return "processing"
    return None


def scan_library(roots, rules=None, progress=None, cancel=None, index=None, reread=False):
    """Walk roots and read every FITS/XISF header. progress(ScanResult) is called
    about four times a second from this thread; cancel is a threading.Event.
    index: ScanIndex.load() - an unchanged file's header isn't read again (unless
    reread), and files missing from it are marked is_new."""
    rules = rules or ScanRules()
    cancel = cancel or threading.Event()
    index = index or {}
    result = ScanResult(roots=[os.path.normpath(r) for r in roots])
    started = time.perf_counter()
    walk_pool = ThreadPoolExecutor(max_workers=WALK_WORKERS, thread_name_prefix="library-walk")
    read_pool = ThreadPoolExecutor(max_workers=rules.workers, thread_name_prefix="library-read")
    pending = {walk_pool.submit(_list_dir, root) for root in result.roots}
    result.folders_queued = len(pending)
    last_progress = 0.0
    try:
        while pending and not cancel.is_set():
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                outcome = future.result()
                if isinstance(outcome[0], FileRecord):
                    record, header, error = outcome
                    record.header, record.error = header, error
                    result.headers_read += 1
                    continue
                path, files, subdirs, error = outcome
                result.folders_listed += 1
                if error:
                    logger.warning(f"Could not list {path}: {error}")
                for subdir in subdirs:
                    reason = skip_reason(subdir, rules)
                    if reason:
                        result.skipped_folders.append((subdir, reason))
                    else:
                        pending.add(walk_pool.submit(_list_dir, subdir))
                        result.folders_queued += 1
                result.files_seen += len(files)
                for file_path, size, mtime_ns in files:
                    record = _new_record(file_path, size, mtime_ns)
                    if record is None:
                        continue
                    result.records.append(record)
                    indexed = index.get(os.path.normcase(file_path))
                    if indexed is None:
                        record.is_new = True
                        result.new_files += 1
                    if record.ext in HEADER_EXTENSIONS and not record.kind:
                        if (not reread and indexed and indexed[2] is not None
                                and indexed[0] == size and indexed[1] == mtime_ns):
                            record.header = indexed[2]
                            result.headers_cached += 1
                        else:
                            pending.add(read_pool.submit(_read_header, record))
                            result.headers_queued += 1
            now = time.perf_counter()
            if progress and now - last_progress >= 0.25:
                last_progress = now
                result.seconds = now - started
                progress(result)
    finally:
        result.cancelled = cancel.is_set()
        walk_pool.shutdown(wait=True, cancel_futures=True)
        read_pool.shutdown(wait=True, cancel_futures=True)
    result.seconds = time.perf_counter() - started
    if progress:
        progress(result)
    return result


def _new_record(path, size, mtime_ns):
    """A FileRecord for a file the scanner cares about, or None. Pipeline copies
    of subs are classified here by name, so their headers are never read."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in HEADER_EXTENSIONS and ext not in IMAGE_EXTENSIONS and ext not in RAW_EXTENSIONS:
        return None
    record = FileRecord(path, size, mtime_ns)
    if ext in RAW_EXTENSIONS:
        record.kind, record.reason = "raw", "camera raw (not supported yet)"
    elif ext in HEADER_EXTENSIONS and PIPELINE_NAME.search(record.stem):
        record.kind, record.reason = "processing", "stacking work file"
    return record


# ---------------------------------------------------------------------------
# Scan index
# ---------------------------------------------------------------------------

def _is_under(path_key, root_keys):
    """path_key (normcase) is root_keys' root itself or inside one of them."""
    return any(path_key == r or path_key.startswith(r.rstrip("\\/") + os.sep) for r in root_keys)


class ScanIndex:
    """What earlier scans found: each file's size, modified time, header (the
    HEADER_KEYWORDS) and when it was first seen, plus when each scanned folder
    was last scanned completely. Kept in the app's cache folder, not the
    collection database - it can always be rebuilt (a lost index only means
    the next scan reads every header again) and shouldn't grow the database or
    its backups. Compact, since a library can hold 100,000+ subs: each folder
    is stored once, and a header keeps only what differs from the values all
    of its folder's files share, compressed. Each call opens its own
    connection, so it's safe on a worker thread."""

    def __init__(self, path=None):
        if path is None:
            from ResourceManager import ResourceManager
            path = ResourceManager.get_cache_dir("library") / "scan_index.sqlite"
        self.path = str(path)

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()
        if row is None or row[0] != str(INDEX_VERSION):
            # Different headers kept (or compressed differently): start over
            for table in ("files", "folders", "roots"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('version', ?)", (str(INDEX_VERSION),))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                folder_id INTEGER PRIMARY KEY,
                folder_key TEXT NOT NULL UNIQUE,    -- os.path.normcase(folder)
                common BLOB                         -- header values all its files share
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                folder_id INTEGER NOT NULL,
                name_key TEXT NOT NULL,
                file_size INTEGER,
                file_mtime_ns INTEGER,
                header BLOB,                        -- NULL: not read (not FITS/XISF, unreadable, cancelled)
                first_seen TEXT,
                PRIMARY KEY (folder_id, name_key)
            ) WITHOUT ROWID
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS roots (
                root_key TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                last_scan TEXT,
                file_count INTEGER
            )
        """)
        conn.commit()
        return conn

    def load(self, roots):
        """{normcase path: (size, mtime_ns, header or None, first_seen)} of the
        indexed files in roots - pass it to scan_library()."""
        root_keys = [os.path.normcase(os.path.normpath(r)) for r in roots]
        found = {}
        conn = self._connect()
        try:
            folders = {}
            for folder_id, folder_key, common in conn.execute("SELECT folder_id, folder_key, common FROM folders"):
                if _is_under(folder_key, root_keys):
                    folders[folder_id] = (folder_key, _unpack(common) or {})
            for folder_id, name_key, size, mtime_ns, header, first_seen in conn.execute(
                    "SELECT folder_id, name_key, file_size, file_mtime_ns, header, first_seen FROM files"):
                folder = folders.get(folder_id)
                if folder is None:
                    continue
                own = _unpack(header)
                found[os.path.join(folder[0], name_key)] = (
                    size, mtime_ns, {**folder[1], **own} if own is not None else None, first_seen)
        finally:
            conn.close()
        return found

    def last_scan(self, root):
        """When root was last scanned completely ('2026-10-16T21:04:11'), or None."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT last_scan FROM roots WHERE root_key = ?",
                               (os.path.normcase(os.path.normpath(root)),)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def save(self, scan, previous=None):
        """Store what a scan found, keeping first-seen dates from previous (the
        load() it was given). Every folder the scan listed is rewritten; after a
        complete (not cancelled) scan, folders that are gone are forgotten too and
        the roots' scan date is set. A cancelled scan's progress is kept - only
        its unread headers are read next time."""
        previous = previous or {}
        now = datetime.now().isoformat(timespec="seconds")
        by_folder = defaultdict(list)
        for record in scan.records:
            by_folder[os.path.normcase(record.folder)].append(record)
        conn = self._connect()
        try:
            ids = {key: folder_id for folder_id, key in conn.execute("SELECT folder_id, folder_key FROM folders")}
            if not scan.cancelled:
                # Folders under the roots that this scan didn't list (deleted, emptied, skipped)
                root_keys = [os.path.normcase(r) for r in scan.roots]
                gone = [ids.pop(key) for key in list(ids) if key not in by_folder and _is_under(key, root_keys)]
                conn.executemany("DELETE FROM files WHERE folder_id = ?", [(i,) for i in gone])
                conn.executemany("DELETE FROM folders WHERE folder_id = ?", [(i,) for i in gone])
            for folder_key, records in by_folder.items():
                headers = [r.header for r in records if r.header is not None]   # An empty one shares nothing
                common = _common_values(headers) if len(headers) > 1 else {}
                if folder_key in ids:
                    folder_id = ids[folder_key]
                    conn.execute("UPDATE folders SET common = ? WHERE folder_id = ?", (_pack(common), folder_id))
                    conn.execute("DELETE FROM files WHERE folder_id = ?", (folder_id,))
                else:
                    folder_id = conn.execute("INSERT INTO folders (folder_key, common) VALUES (?, ?)",
                                             (folder_key, _pack(common))).lastrowid
                conn.executemany(
                    "INSERT OR REPLACE INTO files (folder_id, name_key, file_size, file_mtime_ns, header, first_seen) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [(folder_id, os.path.normcase(r.name), r.size, r.mtime_ns,
                      _pack({k: v for k, v in r.header.items() if common.get(k, _MISSING) != v})
                      if r.header is not None else None,
                      _first_seen(previous, r) or now)
                     for r in records])
            if not scan.cancelled:
                for root in scan.roots:
                    key = os.path.normcase(root)
                    count = sum(len(records) for folder, records in by_folder.items() if _is_under(folder, [key]))
                    conn.execute("INSERT OR REPLACE INTO roots (root_key, root_path, last_scan, file_count) "
                                 "VALUES (?, ?, ?, ?)", (key, root, now, count))
            conn.commit()
        finally:
            conn.close()


_MISSING = object()

# Preset dictionary for compressing headers: their keys, so even a short header
# compresses well. Changing it makes stored headers unreadable - bump INDEX_VERSION.
_HEADER_ZDICT = "".join(f'"{k}":' for k in sorted(HEADER_KEYWORDS)).encode("ascii")


def _first_seen(previous, record):
    indexed = previous.get(os.path.normcase(record.path))
    return indexed[3] if indexed and len(indexed) > 3 else None


def _pack(header):
    if header is None:
        return None
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, _HEADER_ZDICT)
    data = json.dumps(header, separators=(",", ":"), default=str).encode("utf-8")
    return compressor.compress(data) + compressor.flush()


def _unpack(blob):
    if blob is None:
        return None
    try:
        return json.loads(zlib.decompressobj(-15, _HEADER_ZDICT).decompress(blob))
    except (zlib.error, ValueError):
        return None


def _common_values(headers):
    """The {key: value} every header has the same."""
    common = dict(headers[0])
    for header in headers[1:]:
        for key in [k for k, v in common.items() if header.get(k, _MISSING) != v]:
            del common[key]
        if not common:
            break
    return common


# ---------------------------------------------------------------------------
# Stage 2: classify
# ---------------------------------------------------------------------------

def _words(text, collapse=True):
    """Lowercase words of a file/folder name, camelCase split ('RescreenedResult'
    -> rescreened, result) and, with collapse, doubled letters collapsed so a
    typo like 'Proccessed' still matches 'processed'."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    words = re.findall(r"[a-z]+", text.lower())
    return {re.sub(r"(.)\1+", r"\1", w) for w in words} if collapse else set(words)


def _keyword_tier(text, rules):
    """1 for a finished-image word, 2 for a stack word, 3 for software's
    default output name, else 0."""
    if DEFAULT_OUTPUT_NAME.match(text):
        return 3
    if TOOL_STACK_NAME.match(text):
        return 2
    words = _words(text)
    if words & {re.sub(r"(.)\1+", r"\1", k.lower()) for k in rules.final_keywords}:
        return 1
    if "masterlight" in text.lower():
        return 2
    if words & {re.sub(r"(.)\1+", r"\1", k.lower()) for k in rules.stack_keywords}:
        return 2
    return 0


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def exposure_of(header):
    return _number(header.get('EXPTIME', header.get('EXPOSURE'))) if header else None


def _calibration_hint(record):
    """Dark/Flat/Bias when a 'Light' sub's folder or OBJECT says it's calibration."""
    for text in (os.path.basename(record.folder), _clean((record.header or {}).get('OBJECT'))):
        for word in _words(text, collapse=False):
            if word in CALIBRATION_WORDS:
                return CALIBRATION_WORDS[word]
    return None


def _classify_header_file(record, rules):
    if record.kind:
        return
    header = record.header
    if header is None:
        record.kind, record.reason = "unreadable", record.error or "no header"
        return
    image_type = re.sub(r"[\s_-]", "", str(header.get('IMAGETYP') or header.get('FRAME')
                                             or header.get('_XISF_IMAGETYPE') or '')).upper()
    tier = _keyword_tier(record.stem, rules)
    # A stack that kept its first sub's header but made EXPTIME the total (Astro
    # Pixel Processor: EXPTIME 9660, EXPOSURE 30). A sub's two are the same.
    exptime, exposure = _number(header.get('EXPTIME')), _number(header.get('EXPOSURE'))
    summed = bool(exptime and exposure and exptime > exposure * 1.5)
    if (image_type.startswith("MASTER") and "LIGHT" not in image_type) or CALIBRATION_MASTER_NAME.search(record.stem):
        record.kind, record.reason = "processing", "calibration master"
    elif tier == 1:
        record.kind, record.tier, record.reason = "final", 1, "name"
    elif tier == 3:
        # Its header is often a light's, copied along - never take it for a sub
        record.kind, record.tier, record.reason = "final", 3, "software's default output name"
    elif tier == 2 or "MASTERLIGHT" in image_type or (_number(header.get('STACKCNT')) or 0) > 1 \
            or (_number(header.get('NCOMBINE')) or 0) > 1 or summed:
        count = header.get('STACKCNT') or header.get('NCOMBINE')
        record.kind = "final"
        if tier == 2 or summed:
            record.tier = 2
            if count:
                record.reason = f"integration of {count} subs"
            elif summed:
                record.reason = f"integration of about {round(exptime / exposure):,} subs (its exposure time is their total)"
            else:
                record.reason = "integration"
        else:
            # Stacking headers but not an integration's name: a channel
            # extraction or other image made from a master
            record.tier, record.reason = 3, "made from an integration"
    else:
        frame_type = SessionFileScanner._normalize_frame_type(header)
        if frame_type == "Light":
            frame_type = _calibration_hint(record) or frame_type
        if frame_type != "Unknown" and (exposure_of(header) is not None or frame_type == "Bias"):
            record.kind, record.frame_type = "sub", frame_type
        else:
            record.kind, record.reason = "processing", "processing file"


def _classify_image(record, rules):
    if THUMBNAIL_NAME.search(record.stem):
        record.kind, record.reason = "preview", "thumbnail"
    elif CALIBRATION_MASTER_NAME.search(record.stem):
        record.kind, record.reason = "processing", "calibration master"
    else:
        tier = _keyword_tier(record.stem, rules)
        reason = "software's default output name" if tier == 3 else "name"
        if not tier:
            parent = os.path.dirname(record.folder)
            folder_tiers = [_keyword_tier(os.path.basename(f), rules) for f in (record.folder, parent)]
            tier = min((t for t in folder_tiers if t), default=0)
            reason = "folder name"
        if not tier:
            tier, reason = 3, "image file"
        record.kind, record.tier, record.reason = "final", tier, reason


def _format_rank(record):
    return FORMAT_PREFERENCE.index(record.ext) if record.ext in FORMAT_PREFERENCE else len(FORMAT_PREFERENCE)


def classify(scan, rules=None):
    """Set kind (and frame_type/tier/reason) on every record. Files in one folder
    sharing a name are one item: a JPEG beside a sub is that sub's preview, and
    a final image's other formats become its variants."""
    rules = rules or ScanRules()
    stems = defaultdict(list)
    for record in scan.records:
        stems[(record.folder.lower(), record.stem.lower())].append(record)
    for group in stems.values():
        for record in group:
            if record.ext in HEADER_EXTENSIONS:
                _classify_header_file(record, rules)
        has_sub = any(r.kind == "sub" for r in group)
        for record in group:
            if record.ext in IMAGE_EXTENSIONS:
                if has_sub:
                    record.kind, record.reason = "preview", "preview of a sub"
                else:
                    _classify_image(record, rules)
        finals = [r for r in group if r.kind == "final"]
        if finals:
            lead = min(finals, key=lambda r: (r.tier, _format_rank(r)))
            for record in group:
                if record is not lead and record.kind in ("final", "processing", "unreadable"):
                    record.kind, record.variant_of = "variant", lead
                    lead.variants.append(record)


# ---------------------------------------------------------------------------
# DSO names
# ---------------------------------------------------------------------------

def _normalize_name(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


@dataclass
class Resolution:
    dsodetailid: Optional[str] = None
    name: str = ""
    source: str = ""            # header, file name, folder name, position
    others: list = field(default_factory=list)   # other DSOs named in the same text
    mismatch_deg: Optional[float] = None         # name-matched DSO far from the pointing


# Words that say what a common name is of - preferred for labels
COMMON_NAME_TYPE = re.compile(r"\b(nebula|galaxy|cluster|cloud|loop|remnant|pillars|group|triplet|quintet)\b",
                              re.IGNORECASE)


def natural_sort_key(text):
    """'M 2' before 'M 13'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text or "")]


class Catalog:
    """The DSO catalog in memory, for naming objects by designation, common name
    or position."""

    def __init__(self, conn):
        import numpy as np
        self._np = np
        self.designations = {}
        self.common = {}
        self.common_names = defaultdict(list)   # dsodetailid -> its common names, as written
        names = defaultdict(list)
        for dsodetailid, catalogue, designation in conn.execute(
                "SELECT dsodetailid, catalogue, designation FROM cataloguenr"):
            if not catalogue or designation is None:
                continue
            if catalogue == "NAME":
                key = _normalize_name(designation)
                if len(key) >= 6:
                    self.common.setdefault(key, dsodetailid)
                self.common_names[dsodetailid].append(str(designation))
                continue
            self.designations.setdefault((catalogue.upper(), str(designation).upper().replace(" ", "")), dsodetailid)
            names[dsodetailid].append((CATALOG_RANK.get(catalogue, 9), catalogue, str(designation)))
        # Best-known designation: M, NGC, IC ... and '7318A' over '7318-1'
        best = {i: min(n, key=lambda d: (d[0], "-" in d[2], d[1], d[2])) for i, n in names.items()}
        self.names = {i: f"{d[1]} {d[2]}" for i, d in best.items()}
        self.rank = {i: d[0] for i, d in best.items()}
        # Longest common names first, so 'Western Veil Nebula' wins over 'Veil Nebula'
        self._common_sorted = sorted(self.common.items(), key=lambda item: -len(item[0]))

        ids, ras, decs, sizes = [], [], [], []
        for dsodetailid, ra, dec, sizemax, sizemin, dsotype in conn.execute(
                "SELECT id, ra, dec, sizemax, sizemin, dsotype FROM dsodetail"):
            if ra is None or dec is None or (dsotype or "").upper() in NOT_DSO_TYPES or dsodetailid not in self.names:
                continue
            ids.append(dsodetailid)
            ras.append(float(ra))
            decs.append(float(dec))
            sizes.append(float(sizemax or sizemin or 0) / 3600.0)  # Arcseconds -> degrees
        self.ids = ids
        self.ra = np.radians(np.array(ras))
        self.dec = np.radians(np.array(decs))
        self.size = np.array(sizes)
        self._index = {i: n for n, i in enumerate(ids)}
        self._all_designations = names
        self._search_rows = None

    def name_of(self, dsodetailid):
        return self.names.get(dsodetailid, str(dsodetailid))

    def position(self, dsodetailid):
        """(ra_deg, dec_deg) of an object, or (None, None)."""
        n = self._index.get(dsodetailid)
        if n is None:
            return None, None
        return math.degrees(float(self.ra[n])), math.degrees(float(self.dec[n]))

    def label_of(self, dsodetailid):
        """'IC 443 - Jellyfish Nebula': its best designation and its best-known
        common name - preferably the shortest that says what it is ('Orion
        Nebula', not 'Ori Nebula' or radio-source codes like 'Gem A')."""
        name = self.name_of(dsodetailid)
        common = self.common_names.get(dsodetailid)
        if not common:
            return name

        def plain_words(n):  # No digits or abbreviations ('Ori', 'NEB')
            words = re.findall(r"[A-Za-z]+", n)
            return not re.search(r"\d", n) and words and all(
                len(w) >= 4 or w.lower() in ("of", "in", "the", "a") for w in words) and not n.isupper()
        described = [n for n in common if plain_words(n) and COMMON_NAME_TYPE.search(n)]
        nicknames = [n for n in common if plain_words(n)]
        return f"{name} - {min(described or nicknames or common, key=len)}"

    def search(self, text, limit=200):
        """[dsodetailid] matching what someone typed - a designation ('m31',
        'NGC 7000', 'sh2-129') or part of a name ('andromeda') - best first:
        exact designations, then names starting with it, then containing it;
        better-known catalogs first within each."""
        key = _normalize_name(text)
        if not key:
            return []
        if self._search_rows is None:
            rows = []
            for dsodetailid, designations in self._all_designations.items():
                rank = self.rank.get(dsodetailid, 9)
                for _, catalogue, designation in designations:
                    rows.append((_normalize_name(catalogue + designation), rank, dsodetailid))
                for name in self.common_names.get(dsodetailid, ()):
                    rows.append((_normalize_name(name), rank + 0.5, dsodetailid))
            self._search_rows = rows
        exact = [i for i, _ in self.find_in_text(text)]
        scored = {}
        for name_key, rank, dsodetailid in self._search_rows:
            if name_key == key:
                score = (0, rank)
            elif name_key.startswith(key):
                score = (1, rank, len(name_key))
            elif key in name_key:
                score = (2, rank, len(name_key))
            else:
                continue
            if dsodetailid not in scored or score < scored[dsodetailid]:
                scored[dsodetailid] = score
        ranked = sorted(scored, key=lambda i: (scored[i], natural_sort_key(self.name_of(i))))
        return (exact + [i for i in ranked if i not in exact])[:limit]

    def near(self, ra_deg, dec_deg, radius_deg=1.5, limit=12):
        """[dsodetailid] of the objects within radius_deg of a position -
        suggestions for where files were pointed: ones the position falls on
        first (as at_position counts them), then best-known, largest, nearest."""
        np = self._np
        ra, dec = math.radians(ra_deg), math.radians(dec_deg)
        cos_sep = np.sin(dec) * np.sin(self.dec) + np.cos(dec) * np.cos(self.dec) * np.cos(ra - self.ra)
        sep = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        on = sep <= np.maximum(self.size / 2.0, POSITION_MATCH_RADIUS)
        inside = np.nonzero(sep <= radius_deg)[0]
        best = sorted(inside, key=lambda n: (not on[n], self.rank.get(self.ids[n], 9), -self.size[n], sep[n]))
        return [self.ids[n] for n in best[:limit]]

    def find_in_text(self, text):
        """[(dsodetailid, name)] of the DSOs named in a piece of text: its
        designations in order (named as written - 'LDN 1164', not the catalog's
        'Barnard 174' for the same cloud), else the longest common name."""
        if not text:
            return []
        found = []
        # As written first ('vdB 152'), then camelCase split ('QuakeM42' -> 'Quake M42')
        for upper in (str(text).upper(), re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).upper()):
            for alias, number, suffix in DESIGNATION.findall(upper):
                catalogue = CATALOG_ALIASES.get(re.sub(r"\s", "", alias))
                if not catalogue:
                    continue
                key = catalogue.upper()
                for designation in (number + suffix, number, number + "A"):   # NGC 7318 -> 7318A
                    dsodetailid = self.designations.get((key, designation))
                    if dsodetailid:
                        if dsodetailid not in [i for i, _ in found]:
                            found.append((dsodetailid, f"{catalogue} {designation}"))
                        break
            if found:
                return found
        key = _normalize_name(text)
        for name, dsodetailid in self._common_sorted:
            if name in key:
                return [(dsodetailid, self.name_of(dsodetailid))]
        return []

    def separation(self, dsodetailid, ra_deg, dec_deg):
        n = self._index.get(dsodetailid)
        if n is None:
            return None
        np = self._np
        ra, dec = math.radians(ra_deg), math.radians(dec_deg)
        cos_sep = (math.sin(dec) * math.sin(self.dec[n]) + math.cos(dec) * math.cos(self.dec[n])
                   * math.cos(ra - self.ra[n]))
        return math.degrees(math.acos(max(-1.0, min(1.0, float(cos_sep)))))

    def at_position(self, ra_deg, dec_deg):
        """The most notable DSO the position points at, or None: within its own
        extent or POSITION_MATCH_RADIUS of its center, best catalog first."""
        np = self._np
        ra, dec = math.radians(ra_deg), math.radians(dec_deg)
        cos_sep = np.sin(dec) * np.sin(self.dec) + np.cos(dec) * np.cos(self.dec) * np.cos(ra - self.ra)
        sep = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        inside = np.nonzero(sep <= np.maximum(self.size / 2.0, POSITION_MATCH_RADIUS))[0]
        if not len(inside):
            return None
        best = min(inside, key=lambda n: (self.rank.get(self.ids[n], 9), -self.size[n], sep[n]))
        return self.ids[best]

    def resolve(self, texts, clusters=None):
        """Name a DSO from texts [(source, text)] in order of trust, else from
        where the files were pointed: clusters [((ra_deg, dec_deg), share)],
        most frames first (see position_clusters). A name no sizable cluster
        agrees with (a wrong OBJECT header) gives way to a later text that does;
        if none does, the first name stays, flagged. Solar-system targets are
        never named by position."""
        clusters = [c for c in (clusters or []) if c[1] >= POSITION_CONFIRM_SHARE]
        first = None
        for source, text in texts:
            found = self.find_in_text(text)
            if not found:
                continue
            dsodetailid, name = found[0]
            res = Resolution(dsodetailid, name, source, [i for i, _ in found[1:]])
            if clusters and dsodetailid in self._index:
                limit = POSITION_MISMATCH_DEG + self.size[self._index[dsodetailid]] / 2
                seps = [self.separation(dsodetailid, *center) for center, _ in clusters]
                if min(seps) > limit:
                    res.mismatch_deg = seps[0]
                    first = first or res
                    continue
            return res
        if first:
            return first
        if clusters and not any(SOLAR_SYSTEM.search(text or "") for _, text in texts):
            dsodetailid = self.at_position(*clusters[0][0])
            if dsodetailid:
                return Resolution(dsodetailid, self.name_of(dsodetailid), "position")
        return Resolution()


def load_catalog():
    import sqlite3
    from ResourceManager import ResourceManager, attach_update_catalogs
    conn = sqlite3.connect(str(ResourceManager.get_database_path()))
    try:
        attach_update_catalogs(conn)
        return Catalog(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stage 3: group into DSOs, sessions and final images
# ---------------------------------------------------------------------------

@dataclass(eq=False)   # Identity: the same file/session, not equal-looking ones
class Session:
    dso_key: str
    camera: str
    telescope: str
    lights: list = field(default_factory=list)
    calibration: list = field(default_factory=list)
    checked: bool = True
    camera_id: Optional[int] = None         # The user's Equipment the camera/telescope names were matched to
    telescope_id: Optional[int] = None

    @property
    def imported(self):
        """How many of its lights are already in a Session Manager session."""
        return sum(1 for r in self.lights if r.imported)

    @property
    def start(self):
        return _obs_time(self.lights[0])

    @property
    def end(self):
        return _obs_time(self.lights[-1])

    @property
    def nights(self):
        return sorted({_night(r) for r in self.lights if _night(r)})

    @property
    def integration_seconds(self):
        return sum(exposure_of(r.header) or 0 for r in self.lights)

    @property
    def filters(self):
        return Counter(_clean(r.header.get('FILTER')) or "(none)" for r in self.lights)

    @property
    def folders(self):
        return sorted({r.folder for r in self.lights})

    @property
    def new_lights(self):
        """Lights found since the previous scan."""
        return sum(1 for r in self.lights if r.is_new)

    @property
    def night_mounts(self):
        """{night: the strongest MountDetection.NightMount of its lights that
        night} - empty before detect_mounts()."""
        best = {}
        for record in self.lights:
            mount = record.mount
            if mount is not None:
                current = best.get(mount.night)
                if current is None or (mount.mount is not None, mount.strength) > \
                        (current.mount is not None, current.strength):
                    best[mount.night] = mount
        return dict(sorted(best.items()))

    @property
    def mounts(self):
        """Every NightMount its lights point to (to save with the session)."""
        return list({id(r.mount): r.mount for r in self.lights if r.mount is not None}.values())

    @property
    def status(self):
        if not self.imported:
            return "New"
        return "Already imported" if self.imported >= len(self.lights) else "Partly imported"


@dataclass(eq=False)   # Identity: the same file/session, not equal-looking ones
class FinalImage:
    record: FileRecord
    dso_key: str
    resolution: Resolution
    session: Optional[Session] = None
    imported: bool = False
    checked: bool = False

    @property
    def date(self):
        header = self.record.header or {}
        return (parse_obs_datetime(header.get('DATE-OBS'))
                or datetime.fromtimestamp(self.record.mtime_ns / 1e9))

    @property
    def variant_words(self):
        # _words() collapses doubled letters ('starless' -> 'starles'), so match on that form
        return {VARIANT_KEYS[w] for w in _words(self.record.stem) if w in VARIANT_KEYS}

    @property
    def is_new(self):
        """Found since the previous scan (any of its formats)."""
        return self.record.is_new or any(v.is_new for v in self.record.variants)


@dataclass(eq=False)   # Identity: the same file/session, not equal-looking ones
class CalibrationLibrary:
    folder: str
    frame_type: str
    camera: str
    files: list = field(default_factory=list)

    @property
    def is_new(self):
        return any(r.is_new for r in self.files)


@dataclass(eq=False)   # Identity: the same file/session, not equal-looking ones
class DSOGroup:
    key: str                    # dsodetailid, or 'unknown:<folder>'
    name: str
    dsodetailid: Optional[str] = None
    sessions: list = field(default_factory=list)
    finals: list = field(default_factory=list)
    mismatches: list = field(default_factory=list)   # (path, degrees) of name/position disagreements


@dataclass
class Library:
    groups: dict = field(default_factory=dict)              # key -> DSOGroup
    calibration_libraries: list = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)        # previews, duplicates, processing, ...
    raw_folders: Counter = field(default_factory=Counter)   # folder -> camera raw files
    unreadable: list = field(default_factory=list)
    catalog: Optional[object] = None                         # The Catalog it was named with
    equipment: Optional[object] = None                       # The EquipmentInfo sessions were grouped with
    session_gap_days: int = DEFAULT_SESSION_GAP_DAYS


def _clean(value):
    return SessionFileScanner._clean_str(value) or ""


def _obs_time(record):
    """UTC capture time (naive) from DATE-OBS, else the file's modified time."""
    cached = getattr(record, "_obs_time", None)
    if cached is not None:
        return cached
    dt = parse_obs_datetime((record.header or {}).get('DATE-OBS'))
    if dt is None:
        dt = datetime.fromtimestamp(record.mtime_ns / 1e9, timezone.utc)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    record._obs_time = dt
    return dt


def _night(record):
    """The observing night of a sub: local time from the site longitude
    (solar time, close enough to split nights at noon), else this computer's
    time zone."""
    header = record.header or {}
    utc = _obs_time(record)
    longitude = _number(header.get('SITELONG', header.get('LONG-OBS')))
    if longitude is not None:
        local = utc + timedelta(hours=longitude / 15.0)
    else:
        local = utc.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    return night_date_for(local)


def _coords(header):
    """(ra_deg, dec_deg) from a header: the target (OBJCTRA/OBJCTDEC - a mount's
    own RA/DEC can be far off after a bad sync), else the pointing (RA/DEC), else
    the plate solve's reference point (CRVAL1/2)."""
    if not header:
        return None
    if header.get('OBJCTRA') is not None and header.get('OBJCTDEC') is not None:
        ra_h = SessionFileScanner._parse_sexagesimal(header.get('OBJCTRA'))
        dec_d = SessionFileScanner._parse_sexagesimal(header.get('OBJCTDEC'))
        if ra_h is not None and dec_d is not None:
            return ra_h * 15.0, dec_d
    coords = SessionFileScanner._file_coordinates(header)
    if coords:
        return coords
    ra, dec = _number(header.get('CRVAL1')), _number(header.get('CRVAL2'))
    return (ra, dec) if ra is not None and dec is not None else None


def position_clusters(coords):
    """[((ra_deg, dec_deg), share of frames)] of where a folder's subs were
    pointed, most frames first: positions within 1 degree of each other are
    one cluster, so frames with bad coordinates (taken before a plate solve,
    or after a bad mount sync) form their own instead of dragging the mean."""
    if not coords:
        return []
    if len(coords) > 200:
        coords = coords[::len(coords) // 200 + 1]
    remaining, clusters = list(coords), []
    while remaining:
        best = max(remaining, key=lambda c: sum(
            1 for o in remaining if SessionFileScanner.angular_separation_deg(c[0], c[1], o[0], o[1]) <= 1.0))
        near = [o for o in remaining if SessionFileScanner.angular_separation_deg(best[0], best[1], o[0], o[1]) <= 1.0]
        clusters.append((SessionFileScanner._mean_coordinates(near), len(near) / len(coords)))
        remaining = [o for o in remaining
                     if SessionFileScanner.angular_separation_deg(best[0], best[1], o[0], o[1]) > 1.0]
    return clusters


def _folder_texts(folder, root):
    """[('folder name', name)] from a file's folder up to (and including) the scan root."""
    texts = []
    root_norm = os.path.normcase(root)
    while True:
        texts.append(("folder name", os.path.basename(folder)))
        if os.path.normcase(folder) == root_norm:
            break
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return texts


def _root_of(path, roots):
    norm = os.path.normcase(path)
    for root in sorted(roots, key=len, reverse=True):
        r = os.path.normcase(root)
        if norm == r or norm.startswith(r.rstrip("\\/") + os.sep):
            return root
    return roots[0]


def _unknown_label(folder, root):
    """The top folder below the scan root holding folder - 'Milky Way', '_Moon' ..."""
    rel = os.path.relpath(folder, root)
    return rel.split(os.sep)[0] if rel not in (".", "") else os.path.basename(root)


def _ancestors(folder, root):
    """folder, its parent, ... up to and including root."""
    out = []
    root_norm = os.path.normcase(root)
    while True:
        out.append(os.path.normcase(folder))
        if os.path.normcase(folder) == root_norm:
            break
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return out


def _shared_depth(a, b):
    a_parts = os.path.normcase(a).split(os.sep)
    b_parts = os.path.normcase(b).split(os.sep)
    n = 0
    for x, y in zip(a_parts, b_parts):
        if x != y:
            break
        n += 1
    return n


def _family_key(final):
    """A final image's versions share this: the same name, version numbers left
    out ('Processed_M51_V2' and '_V3') as are edit words ('_V1_Cropped'), from
    the same session (or folder)."""
    stem = final.record.stem.lower()
    stem = re.sub(r"\(\d+\)", "", stem)
    stem = EDIT_WORDS.sub("", stem)
    stem = re.sub(r"[\s_\-]*v\d+(\.\d+)*", "", stem)
    # A trailing copy number ('M51 final 2'), but not an object's ('Processed IC443')
    copy_number = re.search(r"[\s_\-]*\d+$", stem)
    if copy_number and not any(d.end() == len(stem) for d in DESIGNATION.finditer(stem.upper())):
        stem = stem[:copy_number.start()]
    stem = re.sub(r"[\s_\-]+", " ", stem)   # What removing words left behind: 'm51__final'
    origin = id(final.session) if final.session else os.path.normcase(final.record.folder)
    return origin, stem.strip()


@dataclass
class KnownItems:
    """What the collection already has, so the review can mark it imported."""
    image_paths: set = field(default_factory=set)
    session_paths: set = field(default_factory=set)
    sub_keys: set = field(default_factory=set)


def _norm_path(path):
    return os.path.normcase(os.path.normpath(path or ""))


def load_known(conn):
    known = KnownItems()
    for (path,) in conn.execute("SELECT image_path FROM userimages WHERE image_path IS NOT NULL"):
        known.image_paths.add(_norm_path(path))
    # Fingerprints made here, not read from sub_key, so they match _sub_key()
    # even in a database the app hasn't migrated yet
    for path, date_obs, exptime, filter_name, frame_type, camera in conn.execute(
            "SELECT file_path, date_obs, exptime_seconds, filter_name, frame_type, camera FROM usersessionfiles"):
        known.session_paths.add(_norm_path(path))
        sub_key = make_sub_key(date_obs, exptime, filter_name, frame_type, camera)
        if sub_key:
            known.sub_keys.add(sub_key)
    return known


# ---------------------------------------------------------------------------
# Equipment: the names in the files matched to the user's Equipment
# ---------------------------------------------------------------------------

TELESCOPE, CAMERA = "telescope", "camera"
HEADER_KEY_OF = {TELESCOPE: "TELESCOP", CAMERA: "INSTRUME"}
# A telescope's focal length within this share of the files' FOCALLEN matches
FOCAL_LENGTH_TOLERANCE = 0.03
PIXEL_SIZE_TOLERANCE = 0.05     # microns


def _alias_key(value):
    return (value or "").strip().casefold()


def _plain(name):
    """'ZWO ASI2600MC Pro' -> 'zwoasi2600mcpro', for comparing names."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@dataclass
class EquipmentInfo:
    """The user's telescopes and cameras, and which of them each name in the
    files (TELESCOP / INSTRUME) was matched to."""
    telescopes: dict = field(default_factory=dict)  # id -> {"name", "focal_length", "aperture", "is_active"}
    cameras: dict = field(default_factory=dict)     # id -> {"name", "pixel_size", "sensor_width", "sensor_height", "is_active"}
    aliases: dict = field(default_factory=dict)     # (kind, alias key) -> equipment id, or None: leave unmatched

    def items(self, kind):
        return self.telescopes if kind == TELESCOPE else self.cameras

    def decided(self, kind, value):
        """Whether the user has said what this name in the files is."""
        return (kind, _alias_key(value)) in self.aliases

    def match(self, kind, value):
        """(equipment id or None, the name to show) for a TELESCOP/INSTRUME value."""
        value = _clean(value)
        equipment_id = self.aliases.get((kind, _alias_key(value)))
        item = self.items(kind).get(equipment_id) if equipment_id is not None else None
        return (equipment_id, item["name"]) if item else (None, value)


def _ensure_alias_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equipmentaliases (
            kind TEXT NOT NULL,             -- 'telescope' or 'camera'
            alias_key TEXT NOT NULL,        -- the name in the files, trimmed and case-folded
            header_value TEXT,              -- as the files spell it
            equipment_id INTEGER,           -- usertelescopes.id / userequipment.id; NULL: leave unmatched
            PRIMARY KEY (kind, alias_key)
        )
    """)


def load_equipment_info(conn):
    _ensure_alias_table(conn)
    info = EquipmentInfo()
    for item_id, name, aperture, focal_length, is_active in conn.execute(
            "SELECT id, name, aperture, focal_length, is_active FROM usertelescopes"):
        info.telescopes[item_id] = {"name": name, "aperture": aperture, "focal_length": focal_length,
                                    "is_active": is_active if is_active is not None else 1}
    for item_id, name, width, height, pixel, is_active in conn.execute(
            "SELECT id, name, sensor_width, sensor_height, pixel_size, is_active FROM userequipment "
            "WHERE equipment_type = 'camera'"):
        info.cameras[item_id] = {"name": name, "sensor_width": width, "sensor_height": height,
                                 "pixel_size": pixel, "is_active": is_active if is_active is not None else 1}
    for kind, key, equipment_id in conn.execute("SELECT kind, alias_key, equipment_id FROM equipmentaliases"):
        # A match to equipment that's since been deleted is no match
        if equipment_id is None or equipment_id in info.items(kind):
            info.aliases[(kind, key)] = equipment_id
    return info


def save_equipment_aliases(conn, decisions):
    """Store {(kind, name in the files): equipment id or None}; commits."""
    _ensure_alias_table(conn)
    conn.executemany(
        "INSERT OR REPLACE INTO equipmentaliases (kind, alias_key, header_value, equipment_id) VALUES (?, ?, ?, ?)",
        [(kind, _alias_key(value), value, equipment_id) for (kind, value), equipment_id in decisions.items()])
    conn.commit()


def add_telescope(conn, name, focal_length=None, aperture=None):
    """A new telescope in the user's Equipment (or the one already named so); its id."""
    row = conn.execute("SELECT id FROM usertelescopes WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO usertelescopes (name, aperture, focal_length, is_active) VALUES (?, ?, ?, 1)",
                        (name, aperture, focal_length)).lastrowid


def add_camera(conn, name, sensor_width=None, sensor_height=None, pixel_size=None):
    """A new camera in the user's Equipment (or the one already named so); its id."""
    row = conn.execute("SELECT id FROM userequipment WHERE equipment_type = 'camera' AND name = ? COLLATE NOCASE",
                       (name,)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO userequipment (equipment_type, name, sensor_width, sensor_height, pixel_size) "
                        "VALUES ('camera', ?, ?, ?, ?)", (name, sensor_width, sensor_height, pixel_size)).lastrowid


def equipment_from_header(kind, header):
    """What a light's header says about its telescope or camera: {"focal_length",
    "aperture"} (mm) or {"pixel_size" (µm, unbinned), "sensor_width",
    "sensor_height" (mm)} - the values a new Equipment entry starts with."""
    header = header or {}
    if kind == TELESCOPE:
        focal = _number(header.get('FOCALLEN'))
        aperture = _number(header.get('APTDIA'))
        ratio = _number(header.get('FOCRATIO'))
        if not aperture and focal and ratio:
            aperture = round(focal / ratio, 1)
        if aperture and aperture < 1:   # Some software writes metres
            aperture *= 1000
        return {"focal_length": focal or None, "aperture": aperture or None}
    pixel = _number(header.get('XPIXSZ'))
    binning = _number(header.get('XBINNING')) or 1
    width, height = _number(header.get('NAXIS1')), _number(header.get('NAXIS2'))
    return {"pixel_size": round(pixel / max(binning, 1), 2) if pixel else None,
            # Binned pixels times the binned pixel size: the sensor's size either way
            "sensor_width": round(width * pixel / 1000, 2) if width and pixel else None,
            "sensor_height": round(height * pixel / 1000, 2) if height and pixel else None}


def suggest_equipment(kind, value, header, info):
    """(equipment id, why) for a name in the files nobody has matched yet - by
    name ('ZWO ASI2600MC Pro' ~ 'ASI2600MC Pro'), else by focal length or pixel
    size when exactly one item fits - or (None, "")."""
    items = info.items(kind)
    plain = _plain(value)
    if not plain or not items:
        return None, ""
    named = []
    for item_id, item in items.items():
        other = _plain(item["name"])
        if other and min(len(other), len(plain)) >= 3 and (other in plain or plain in other):
            named.append(item_id)
    measured = equipment_from_header(kind, header)
    if kind == TELESCOPE:
        focal = measured["focal_length"]
        fits = [i for i, item in items.items() if focal and item["focal_length"]
                and abs(item["focal_length"] - focal) <= focal * FOCAL_LENGTH_TOLERANCE]
        fit_reason = f"same focal length ({focal:g} mm)" if focal else ""
    else:
        pixel = measured["pixel_size"]
        fits = [i for i, item in items.items() if pixel and item["pixel_size"]
                and abs(item["pixel_size"] - pixel) <= PIXEL_SIZE_TOLERANCE]
        fit_reason = f"same pixel size ({pixel:g} µm)" if pixel else ""
    if named:
        both = [i for i in named if i in fits]
        candidates = both or named
        # The closest name, enabled equipment first
        best = min(candidates, key=lambda i: (not items[i]["is_active"], abs(len(_plain(items[i]["name"])) - len(plain))))
        return best, "same name" + (f" and {fit_reason}" if best in fits else "")
    if len(fits) == 1:
        return fits[0], fit_reason
    active = [i for i in fits if items[i]["is_active"]]
    if len(active) == 1:
        return active[0], fit_reason
    return None, ""


def equipment_in_files(library):
    """[{"kind", "value", "subs", "sessions", "header", "first", "last"}] - every
    telescope and camera name the lights' headers use, with how many subs and
    sessions, a light's header (for its focal length, pixel size ...) and the
    first and last nights."""
    found = {}
    for group in library.groups.values():
        for session in group.sessions:
            for record in session.lights:
                for kind in (TELESCOPE, CAMERA):
                    value = _clean(record.header.get(HEADER_KEY_OF[kind]))
                    if not value:
                        continue
                    entry = found.setdefault((kind, _alias_key(value)), {
                        "kind": kind, "spellings": Counter(), "subs": 0, "sessions": set(),
                        "header": record.header, "first": None, "last": None})
                    entry["spellings"][value] += 1
                    entry["subs"] += 1
                    entry["sessions"].add(id(session))
                    night = _night(record)
                    entry["first"] = min(entry["first"] or night, night)
                    entry["last"] = max(entry["last"] or night, night)
    rows = []
    for entry in found.values():
        rows.append({"kind": entry["kind"], "value": entry["spellings"].most_common(1)[0][0], "subs": entry["subs"],
                     "sessions": len(entry["sessions"]), "header": entry["header"],
                     "first": entry["first"], "last": entry["last"]})
    rows.sort(key=lambda r: (r["kind"] != TELESCOPE, -r["subs"]))
    return rows


def _ensure_assignments_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS importassignments (
            path_key TEXT PRIMARY KEY,      -- os.path.normcase of a light folder or an image file
            dsodetailid TEXT NOT NULL,
            name TEXT,
            source TEXT,                    -- 'you' (chosen in the review) or 'plate solve'
            assigned_date TEXT
        )
    """)


def load_assignments(conn):
    """{normcase path: (dsodetailid, name, source)} - the objects chosen in the
    Import Wizard's review (or found by plate solving) for light folders and
    images. Kept in the collection database: they are the user's decisions,
    unlike the disposable scan index."""
    _ensure_assignments_table(conn)
    return {key: (dsodetailid, name, source) for key, dsodetailid, name, source in conn.execute(
        "SELECT path_key, dsodetailid, name, source FROM importassignments")}


def save_assignments(conn, assignments):
    """Store {path: (dsodetailid, name, source)}; commits."""
    _ensure_assignments_table(conn)
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO importassignments (path_key, dsodetailid, name, source, assigned_date) "
        "VALUES (?, ?, ?, ?, ?)",
        [(os.path.normcase(path), dsodetailid, name, source, now)
         for path, (dsodetailid, name, source) in assignments.items()])
    conn.commit()


def _sub_key(record):
    h = record.header or {}
    return make_sub_key(h.get('DATE-OBS'), exposure_of(h), _clean(h.get('FILTER')), record.frame_type,
                        _clean(h.get('INSTRUME')))


def _rig(record, equipment):
    """((camera id or None, camera name), (telescope id or None, telescope name))."""
    return (equipment.match(CAMERA, record.header.get('INSTRUME')),
            equipment.match(TELESCOPE, record.header.get('TELESCOP')))


def _rig_key(record, equipment):
    """Lights with the same key were taken with the same camera and telescope:
    matched ones by their Equipment entry ('Seestar S50' and 'S50_5a91178b'
    alike), others by name."""
    return tuple(("id", equipment_id) if equipment_id is not None else ("name", _alias_key(name))
                 for equipment_id, name in _rig(record, equipment))


def _set_rig(session, equipment):
    """A session's camera and telescope: the ones most of its lights were taken with."""
    (camera, telescope) = Counter(_rig(r, equipment) for r in session.lights).most_common(1)[0][0]
    session.camera_id, session.camera = camera
    session.telescope_id, session.telescope = telescope


def _build_sessions(key, lights, gap, equipment):
    """One object's lights as sessions: by camera and telescope, a new session
    after a gap of more than gap (timedelta) between nights."""
    by_rig = defaultdict(list)
    for record in lights:
        by_rig[_rig_key(record, equipment)].append(record)
    sessions = []
    for rig_lights in by_rig.values():
        rig_lights.sort(key=_obs_time)
        session = None
        for record in rig_lights:
            if session is None or _obs_time(record) - _obs_time(session.lights[-1]) > gap:
                session = Session(key, "", "")
                sessions.append(session)
            session.lights.append(record)
    for session in sessions:
        _set_rig(session, equipment)
    return sessions


def _same_camera(session, camera_value, equipment):
    """Whether calibration frames whose INSTRUME is camera_value suit the session."""
    if not camera_value:
        return True
    camera_id, name = equipment.match(CAMERA, camera_value)
    if camera_id is not None or session.camera_id is not None:
        return camera_id == session.camera_id
    return _alias_key(name) == _alias_key(session.camera)


def build_library(scan, catalog, known=None, rules=None, assignments=None, equipment=None):
    """Group a classified scan by DSO. assignments: load_assignments() - the
    objects the user chose (or plate solving found) for folders and files,
    which win over what names and positions say. equipment: load_equipment_info()
    - sessions are split by the matched telescope and camera."""
    rules = rules or ScanRules()
    known = known or KnownItems()
    assignments = assignments or {}
    equipment = equipment or EquipmentInfo()
    library = Library(catalog=catalog, equipment=equipment, session_gap_days=rules.session_gap_days)
    roots = scan.roots

    def group_for(resolution, path):
        if resolution.dsodetailid:
            key, name = resolution.dsodetailid, resolution.name
        else:
            label = _unknown_label(path, _root_of(path, roots))
            key, name = f"unknown:{label.lower()}", label
        if key not in library.groups:
            library.groups[key] = DSOGroup(key, name, resolution.dsodetailid)
        return library.groups[key]

    # Duplicate copies of lights (a copied folder): keep the shallowest, oldest
    # copy. Calibration isn't deduplicated here: the same flats in a library and
    # a night's folder both belong.
    by_key = defaultdict(list)
    for record in scan.records:
        if record.kind == "sub" and record.frame_type == "Light":
            key = _sub_key(record)
            if key:
                by_key[key].append(record)
    for copies in by_key.values():
        if len(copies) > 1:
            copies.sort(key=lambda r: (r.path.count(os.sep), r.mtime_ns, r.path.lower()))
            for record in copies[1:]:
                record.kind, record.reason = "duplicate", f"copy of {copies[0].path}"

    for record in scan.records:
        if record.kind == "raw":
            library.raw_folders[record.folder] += 1
        elif record.kind == "unreadable":
            library.unreadable.append(record)
        elif record.kind in ("preview", "duplicate", "processing", "variant"):
            library.counts[record.kind] += 1
    library.counts["processing folders"] = sum(1 for _, r in scan.skipped_folders if r == "processing")
    library.counts["rejected folders"] = sum(1 for _, r in scan.skipped_folders if r == "rejected")

    # Lights: one DSO per folder + OBJECT, then sessions by DSO + camera + telescope
    buckets = defaultdict(list)
    for record in scan.records:
        if record.kind == "sub" and record.frame_type == "Light":
            obj = _clean((record.header or {}).get('OBJECT'))
            buckets[(record.folder, obj)].append(record)
    group_lights = defaultdict(list)
    for (folder, obj), records in buckets.items():
        root = _root_of(folder, roots)
        texts = []
        if _normalize_name(obj) not in PLACEHOLDER_OBJECTS:
            texts.append(("header", obj))
        texts.append(("file name", records[0].stem))
        texts += _folder_texts(folder, root)
        clusters = position_clusters([c for c in (_coords(r.header) for r in records) if c])
        assigned = assignments.get(os.path.normcase(folder))
        resolution = Resolution(*assigned) if assigned else catalog.resolve(texts, clusters)
        group = group_for(resolution, folder)
        if resolution.mismatch_deg:
            group.mismatches.append((folder, resolution.mismatch_deg))
        group_lights[group.key].extend(records)

    gap = timedelta(days=rules.session_gap_days)
    all_sessions = []
    for key, lights in group_lights.items():
        sessions = _build_sessions(key, lights, gap, equipment)
        library.groups[key].sessions.extend(sessions)
        all_sessions.extend(sessions)
    for session in all_sessions:
        for r in session.lights:
            r.imported = _norm_path(r.path) in known.session_paths or _sub_key(r) in known.sub_keys
        # Nothing can be imported without a DSO - unknown groups wait for the user to name them
        session.checked = session.imported < len(session.lights) and not session.dso_key.startswith("unknown:")
    for group in library.groups.values():
        group.sessions.sort(key=lambda s: s.start)

    # Calibration: attach to the session of the nearest folder holding lights of one DSO
    under = defaultdict(set)    # folder (normcase) -> indexes of sessions with lights below it
    for n, session in enumerate(all_sessions):
        for folder in session.folders:
            for ancestor in _ancestors(folder, _root_of(folder, roots)):
                under[ancestor].add(n)
    calibration = defaultdict(list)
    for record in scan.records:
        if record.kind == "sub" and record.frame_type != "Light":
            calibration[(record.folder, record.frame_type, _clean(record.header.get('INSTRUME')))].append(record)
    for (folder, frame_type, camera), records in calibration.items():
        records.sort(key=_obs_time)
        target = None
        for ancestor in _ancestors(folder, _root_of(folder, roots)):
            indexes = under.get(ancestor)
            if not indexes:
                continue
            if len({all_sessions[n].dso_key for n in indexes}) > 1:
                break   # Lights of several DSOs share this folder - a calibration library
            same_camera = [all_sessions[n] for n in indexes if _same_camera(all_sessions[n], camera, equipment)]
            if same_camera:
                middle = _obs_time(records[len(records) // 2])
                target = min(same_camera, key=lambda s: _time_distance(middle, s))
                break
        if target:
            target.calibration.extend(records)
        else:
            library.calibration_libraries.append(CalibrationLibrary(folder, frame_type, camera, records))
    library.calibration_libraries.sort(key=lambda c: c.folder.lower())

    # Final images
    for record in scan.records:
        if record.kind != "final":
            continue
        header = record.header or {}
        root = _root_of(record.folder, roots)
        texts = []
        obj = _clean(header.get('OBJECT'))
        if _normalize_name(obj) not in PLACEHOLDER_OBJECTS:
            texts.append(("header", obj))
        texts.append(("file name", record.stem))
        texts += _folder_texts(record.folder, root)
        coords = _coords(header)
        assigned = (assignments.get(os.path.normcase(record.path))
                    or assignments.get(os.path.normcase(record.folder)))
        resolution = Resolution(*assigned) if assigned else catalog.resolve(texts, [(coords, 1.0)] if coords else None)
        if resolution.others and not assigned:
            # 'Processed_M82_M81': the DSO whose session sits nearest the image
            candidates = [resolution.dsodetailid] + resolution.others
            depths = [max((_shared_depth(record.folder, f) for s in library.groups[c].sessions for f in s.folders),
                           default=0) if c in library.groups else 0 for c in candidates]
            best = candidates[depths.index(max(depths))]
            if best != resolution.dsodetailid:
                resolution.others = [c for c in candidates if c != best]
                resolution.dsodetailid, resolution.name = best, library.groups[best].name
        group = group_for(resolution, record.folder)
        final = FinalImage(record, group.key, resolution)
        final.imported = any(_norm_path(r.path) in known.image_paths for r in [record] + record.variants)
        final.session = _pair_session(final, group.sessions)
        group.finals.append(final)
    for group in library.groups.values():
        if group.dsodetailid:
            _choose_default_finals(group.finals)
        group.finals.sort(key=lambda f: (f.record.tier, not f.checked, f.record.path.lower()))
    return library


def _time_distance(moment, session):
    if session.start <= moment <= session.end:
        return timedelta(0)
    return min(abs(moment - session.start), abs(moment - session.end))


def _pair_session(final, sessions):
    """The session a final image most likely came from: sharing the most of its
    folder path, then the latest one finished before the image was made."""
    if not sessions:
        return None
    date = final.date
    if date.tzinfo is not None:
        date = date.astimezone(timezone.utc).replace(tzinfo=None)

    def score(session):
        depth = max(_shared_depth(final.record.folder, folder) for folder in session.folders)
        before = session.end <= date + timedelta(days=1)
        return depth, before, -abs((date - session.end).total_seconds())
    return max(sessions, key=score)


def _choose_default_finals(finals):
    """Check the newest finished image of each version family; a DSO with no
    finished image gets its newest stack instead. Possible images, by-products
    (starless, annotated ...) and ones already imported stay unchecked."""
    families = defaultdict(list)
    for final in finals:
        if final.record.tier == 1 and not final.variant_words:
            families[_family_key(final)].append(final)
    for versions in families.values():
        if not any(f.imported for f in versions):   # One version already in the collection is enough
            max(versions, key=lambda f: f.record.mtime_ns).checked = True
    # A stack already in the gallery counts too - a second import doesn't offer another
    if not any(f.checked or (f.imported and f.record.tier <= 2) for f in finals):
        stacks = [f for f in finals if f.record.tier == 2 and not f.variant_words and not f.imported]
        if stacks:
            max(stacks, key=lambda f: f.record.mtime_ns).checked = True


# ---------------------------------------------------------------------------
# The user's changes in the review
# ---------------------------------------------------------------------------

def _group_of(library, key):
    return library.groups.get(key)


def _drop_if_empty(library, group):
    if group is not None and not group.sessions and not group.finals:
        library.groups.pop(group.key, None)


def _repair_pairings(group):
    """Images whose session left the group pair with one of its own again."""
    for final in group.finals:
        if final.session is not None and final.session not in group.sessions:
            final.session = _pair_session(final, group.sessions)


def move_to_object(library, sessions, finals, dsodetailid, name, source="you"):
    """Move sessions and images to another object (made a group if it isn't
    one yet). Ones coming from a group without an object get the usual
    default checks. Returns ({path: (dsodetailid, name, source)} to remember -
    the light folders and image files - and the target group)."""
    target = library.groups.get(dsodetailid)
    if target is None:
        target = DSOGroup(dsodetailid, name, dsodetailid)
        library.groups[dsodetailid] = target
    remember = {}
    touched = set()
    for session in sessions:
        old = _group_of(library, session.dso_key)
        if old is target:
            continue
        if old is not None:
            old.sessions.remove(session)
            touched.add(old)
        if session.dso_key.startswith("unknown:"):
            session.checked = session.imported < len(session.lights)
        session.dso_key = target.key
        target.sessions.append(session)
        for folder in session.folders:
            remember[folder] = (dsodetailid, target.name, source)
    for final in finals:
        old = _group_of(library, final.dso_key)
        if old is not target and old is not None:
            old.finals.remove(final)
            touched.add(old)
        if old is not target:
            if final.dso_key.startswith("unknown:"):
                final.checked = final.record.tier == 1 and not final.variant_words and not final.imported
            final.dso_key = target.key
            target.finals.append(final)
        final.resolution = Resolution(dsodetailid, target.name, source)
        remember[final.record.path] = (dsodetailid, target.name, source)
    target.sessions.sort(key=lambda s: s.start)
    _repair_pairings(target)
    for final in target.finals:
        if final.session is None:
            final.session = _pair_session(final, target.sessions)
    for group in touched:
        _repair_pairings(group)
        _drop_if_empty(library, group)
    return remember, target


def merge_sessions(library, sessions):
    """Make one session of several of the same object; returns it. Its camera
    and telescope are the ones most of its lights name."""
    sessions = sorted(sessions, key=lambda s: s.start)
    merged = sessions[0]
    group = _group_of(library, merged.dso_key)
    for session in sessions[1:]:
        merged.lights.extend(session.lights)
        merged.calibration.extend(session.calibration)
        merged.checked = merged.checked or session.checked
        if group is not None and session in group.sessions:
            group.sessions.remove(session)
    merged.lights.sort(key=_obs_time)
    _set_rig(merged, library.equipment or EquipmentInfo())
    if group is not None:
        for final in group.finals:
            if final.session in sessions[1:]:
                final.session = merged
    return merged


def split_session(library, session, nights):
    """Move the given nights (dates) of a session into a new session of the
    same object; returns it, or None if that would leave either one empty.
    Calibration goes with whichever is nearer in time."""
    nights = set(nights)
    moving = [r for r in session.lights if _night(r) in nights]
    if not moving or len(moving) == len(session.lights):
        return None
    new = Session(session.dso_key, session.camera, session.telescope, checked=session.checked)
    new.lights = moving
    session.lights = [r for r in session.lights if _night(r) not in nights]
    for s in (session, new):
        _set_rig(s, library.equipment or EquipmentInfo())
    calibration, session.calibration = session.calibration, []
    for record in calibration:
        moment = _obs_time(record)
        (new if _time_distance(moment, new) < _time_distance(moment, session) else session).calibration.append(record)
    group = _group_of(library, session.dso_key)
    if group is not None:
        group.sessions.append(new)
        group.sessions.sort(key=lambda s: s.start)
        for final in group.finals:
            if final.session is session:
                final.session = _pair_session(final, [session, new])
    return new


def regroup_sessions(library, equipment):
    """Build every object's sessions again with new equipment matches (lights
    of one rig under two names become one session). Calibration goes to the
    nearest session with its camera; images pair again. Merges and splits made
    in the review are undone - like a new scan, without reading anything."""
    library.equipment = equipment
    gap = timedelta(days=library.session_gap_days)
    for group in library.groups.values():
        if not group.sessions:
            continue
        lights = [r for s in group.sessions for r in s.lights]
        calibration = [r for s in group.sessions for r in s.calibration]
        unchecked = {id(r) for s in group.sessions if not s.checked for r in s.lights}
        group.sessions = _build_sessions(group.key, lights, gap, equipment)
        group.sessions.sort(key=lambda s: s.start)
        for session in group.sessions:
            # Unchecked if most of its lights were in sessions the user (or the defaults) left unchecked
            left_out = sum(1 for r in session.lights if id(r) in unchecked)
            session.checked = left_out * 2 < len(session.lights) and session.imported < len(session.lights)
        for record in calibration:
            camera = _clean(record.header.get('INSTRUME'))
            candidates = [s for s in group.sessions if _same_camera(s, camera, equipment)] or group.sessions
            moment = _obs_time(record)
            min(candidates, key=lambda s: _time_distance(moment, s)).calibration.append(record)
        for final in group.finals:
            final.session = _pair_session(final, group.sessions)


def detect_mounts(library, site=None, cache=None, status=None, cancel=None):
    """Tell each session's nights' mount, equatorial or alt-az (MountDetection):
    every light's record.mount is its night's NightMount. site: (lat, lon) for
    subs that don't say where they were taken. Nights not checked when cancel
    is set keep None."""
    status = status or (lambda text: None)
    nights, records_of = [], {}
    for group in library.groups.values():
        for session in group.sessions:
            by_night = defaultdict(list)
            for record in session.lights:
                record.mount = None
                by_night[(_night(record), MountDetection.rig_of(record.header))].append(record)
            for (night, rig), records in sorted(by_night.items(), key=lambda item: item[0]):
                subs = []
                for record in records:
                    coords = SessionFileScanner._file_coordinates(record.header or {})
                    sub = MountDetection.Sub(record.path, _obs_time(record), *(coords or (None, None)),
                                             record.size, record.mtime_ns, _sub_key(record),
                                             *MountDetection.site_of(record.header))
                    records_of[id(sub)] = record
                    subs.append(sub)
                nights.append((night, rig, subs))
    if not nights:
        return
    status(f"Checking how the mount tracked each night: 0 of {len(nights):,}")
    mounts = MountDetection.detect(
        nights, site, cache, cancel=cancel,
        progress=lambda done, total: status(f"Checking how the mount tracked each night: {done:,} of {total:,}"))
    for mount in mounts:
        for sub in mount.subs:
            records_of[id(sub)].mount = mount


def save_known_mounts(conn, library):
    """Save the detected mounts of lights already in Session Manager sessions -
    a rescan fills in sessions imported before. Commits."""
    MountDetection.ensure_schema(conn)
    by_path, by_key = _owners(conn)
    for group in library.groups.values():
        for session in group.sessions:
            mounts = session.mounts
            if not mounts or not session.imported:
                continue
            owners = {by_path.get(os.path.normcase(os.path.normpath(r.path))) or by_key.get(_sub_key(r))
                      for r in session.lights} - {None}
            for session_id in owners:
                MountDetection.save(conn, session_id, mounts, tidy=False)
    MountDetection.tidy_saved(conn)
    conn.commit()


def scan_and_build(roots, rules=None, progress=None, cancel=None, reread=False, scan_index=None, status=None,
                   mounts=True):
    """The whole scan, for a worker thread: load the index, scan (reading only
    new or changed headers, or every one with reread), save the index, classify,
    group and (with mounts) tell each night's mount. scan_index=None skips the
    index and the mount cache kept beside it. status(text) is told each step.
    Returns (ScanResult, Library); a cancelled scan's Library covers what was
    found before cancelling."""
    rules = rules or ScanRules()
    status = status or (lambda text: None)
    status("Loading what earlier scans found...")
    known_files = scan_index.load(roots) if scan_index else {}
    status("Looking through your folders...")
    scan = scan_library(roots, rules, progress=progress, cancel=cancel, index=known_files, reread=reread)
    if scan_index:
        status("Saving the scan...")
        try:
            scan_index.save(scan, known_files)
        except sqlite3.Error as e:
            logger.warning(f"Could not save the library scan index: {e}")   # Only costs a slower next scan
    status("Sorting files into sessions and images...")
    classify(scan, rules)

    from ResourceManager import ResourceManager
    catalog = load_catalog()
    conn = sqlite3.connect(str(ResourceManager.get_database_path()))
    try:
        known = load_known(conn)
        assignments = load_assignments(conn)
        equipment = load_equipment_info(conn)
        locations = _saved_locations(conn)
    finally:
        conn.close()
    library = build_library(scan, catalog, known, rules, assignments, equipment)
    if mounts and not (cancel and cancel.is_set()):
        active = next((loc for loc in locations if loc[4]), locations[0] if locations else None)
        cache = None
        try:
            cache = MountDetection.MountCache(scan_index.path) if scan_index else None
        except sqlite3.Error as e:
            logger.warning(f"Could not open the mount cache: {e}")   # Only costs measuring again
        try:
            detect_mounts(library, (active[0], active[1]) if active else None, cache, status, cancel)
        finally:
            if cache:
                cache.close()
    return scan, library


# ---------------------------------------------------------------------------
# Import
# Writes what the review leaves checked into the collection: sessions - with
# their subs and calibration frames - into the Session Manager, and finished
# images into the gallery, linked to the session they came from. Runs on a
# worker thread with its own database connection. Files are never moved or
# changed; only references to them are stored.
#
# Session status (the user's rule): a new session is Completed when its last
# night is older than the session gap, or a finished image from it is imported
# or already in the gallery; otherwise In Progress (still being imaged). A
# session that already has some of the subs in the Session Manager gets the rest
# added and keeps its status. No completion hand-off is opened.
# ---------------------------------------------------------------------------

# Subs taken this close to a saved location use it
LOCATION_MATCH_KM = 50


@dataclass
class ImportReport:
    sessions_created: int = 0
    sessions_extended: int = 0
    completed: int = 0
    in_progress: int = 0
    subs_added: int = 0
    calibration_added: int = 0
    subs_skipped: int = 0           # Already in another session
    images_added: int = 0
    images_skipped: int = 0         # Already in the gallery for that object
    images_linked: int = 0          # Already in the gallery, now linked to their session
    cancelled: bool = False
    errors: list = field(default_factory=list)
    session_ids: dict = field(default_factory=dict)     # id(Session) -> usersessions.id
    imported_finals: list = field(default_factory=list)


def plan(library):
    """(sessions, images) the review leaves checked: of objects (not groups
    still needing one), images not already in the gallery."""
    groups = [g for g in library.groups.values() if g.dsodetailid]
    sessions = [s for g in groups for s in g.sessions if s.checked]
    finals = [f for g in groups for f in g.finals if f.checked and not f.imported]
    return sessions, finals


def _file_dict(record):
    """A light/calibration record in SessionFileScanner.scan_file()'s shape."""
    data = {k: v for k, v in (record.header or {}).items() if not k.startswith("_")}
    data["_file_path"] = record.path
    data["_file_type"] = "xisf" if record.ext in XISF_EXTENSIONS else "fits"
    data["_frame_type"] = record.frame_type
    return data


def _owners(conn):
    """{normcase path: session id}, {sub key: session id} of the subs already
    in Session Manager sessions."""
    by_path, by_key = {}, {}
    for session_id, path, date_obs, exptime, filter_name, frame_type, camera in conn.execute(
            "SELECT session_id, file_path, date_obs, exptime_seconds, filter_name, frame_type, camera "
            "FROM usersessionfiles"):
        by_path[os.path.normcase(os.path.normpath(path or ""))] = session_id
        key = SessionObservations.make_sub_key(date_obs, exptime, filter_name, frame_type, camera)
        if key:
            by_key[key] = session_id
    return by_path, by_key


def _existing_session(session, by_path, by_key):
    """The Session Manager session holding most of this session's subs already, or None."""
    owners = Counter()
    for record in session.lights:
        owner = by_path.get(os.path.normcase(os.path.normpath(record.path))) or by_key.get(_sub_key(record))
        if owner:
            owners[owner] += 1
    return owners.most_common(1)[0][0] if owners else None


def _distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _saved_locations(conn):
    try:
        return [row for row in conn.execute(
            "SELECT location_lat, location_lon, location_name, timezone, is_active FROM usersettings "
            "WHERE location_lat IS NOT NULL AND location_lon IS NOT NULL")]
    except Exception as e:
        logger.debug(f"No saved locations: {e}")
        return []


def _session_location(session, locations):
    """(lat, lon, name, timezone, source) for a new session: the saved location
    nearest where its subs say they were taken (within LOCATION_MATCH_KM), else
    that site as a custom location (with the active location's time zone), else
    the active location."""
    active = next((loc for loc in locations if loc[4]), locations[0] if locations else None)
    lats, lons = [], []
    for record in session.lights[::max(1, len(session.lights) // 50)]:
        header = record.header or {}
        lat = _number(header.get('SITELAT', header.get('LAT-OBS')))
        lon = _number(header.get('SITELONG', header.get('LONG-OBS')))
        if lat is not None and lon is not None and (lat or lon):
            lats.append(lat)
            lons.append(lon)
    if lats:
        lat, lon = sorted(lats)[len(lats) // 2], sorted(lons)[len(lons) // 2]
        near = [loc for loc in locations if _distance_km(lat, lon, loc[0], loc[1]) <= LOCATION_MATCH_KM]
        if near:
            # The active location if it's one of them (several saved ones can sit at home), else the nearest
            chosen = next((loc for loc in near if loc[4]), None) or \
                min(near, key=lambda loc: _distance_km(lat, lon, loc[0], loc[1]))
            return chosen[0], chosen[1], chosen[2], chosen[3], "active" if chosen[4] else "saved"
        return lat, lon, None, active[3] if active else None, "custom"
    if active:
        return active[0], active[1], active[2], active[3], "active"
    return None, None, None, None, "active"


def _target_id(dso_name, targets):
    """The Target List entry named like the session's object (as the Session Manager links them)."""
    for wanted in (2, 1):
        for target_id, target_name in targets:
            if SessionObservations.target_name_match(dso_name, target_name) == wanted:
                return target_id
    return None


def _status(session, finals, gap_days, today):
    """Completed if its last night is older than the gap, or a finished image
    from it is imported (or already in the gallery); else In Progress."""
    nights = session.nights
    if nights and nights[-1] < today - timedelta(days=gap_days):
        return "Completed"
    if any(f.session is session and (f.checked or f.imported) for f in finals):
        return "Completed"
    return "In Progress"


def _format_integration(seconds):
    """'2h 30m' - the gallery's integration time format."""
    total_minutes = round((seconds or 0) / 60.0)
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    return f"{hours}h" if hours else f"{minutes}m"


def import_library(conn, library, progress=None, cancelled=None, today=None):
    """Import the checked sessions and images; returns an ImportReport. Each
    session (and the images at the end) is its own transaction, so stopping part
    way keeps what was done. progress(done, total, text); cancelled() -> bool."""
    progress = progress or (lambda done, total, text: None)
    cancelled = cancelled or (lambda: False)
    today = today or datetime.now().date()
    report = ImportReport()
    sessions, finals = plan(library)
    all_finals = [f for g in library.groups.values() for f in g.finals]
    total = len(sessions) + (1 if finals else 0)

    by_path, by_key = _owners(conn)
    locations = _saved_locations(conn)
    targets = list(conn.execute("SELECT id, name FROM usertargetlist"))
    MountDetection.ensure_schema(conn)
    conn.commit()
    catalog = library.catalog

    for done, session in enumerate(sessions):
        if cancelled():
            report.cancelled = True
            return report
        group = library.groups.get(session.dso_key)
        progress(done, total, f"{group.name}: {_clean(session.camera) or 'session'}, "
                              f"{len(session.lights):,} subs")
        try:
            report.session_ids[id(session)] = _import_session(conn, session, group, all_finals, library, catalog,
                                                               by_path, by_key, locations, targets, today, report)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Importing a {group.name} session failed: {e}", exc_info=True)
            report.errors.append(f"{group.name}: {e}")

    if finals and not cancelled():
        progress(len(sessions), total, f"Adding {len(finals):,} images to the gallery")
        try:
            _import_images(conn, finals, library, by_path, by_key, report)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Adding images failed: {e}", exc_info=True)
            report.errors.append(f"Images: {e}")
    elif finals:
        report.cancelled = True
    if not cancelled():
        try:
            link_gallery_images(conn, library, by_path, by_key, report)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Linking gallery images failed: {e}", exc_info=True)
            report.errors.append(f"Linking images: {e}")
    try:
        MountDetection.tidy_saved(conn)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Tidying the detected mounts failed: {e}", exc_info=True)
    progress(total, total, "Done")
    return report


def _import_session(conn, session, group, all_finals, library, catalog, by_path, by_key, locations, targets,
                    today, report):
    session_id = _existing_session(session, by_path, by_key)
    if session_id is not None:
        report.sessions_extended += 1
    else:
        status = _status(session, all_finals, library.session_gap_days, today)
        ra, dec = catalog.position(group.dsodetailid) if catalog else (None, None)
        lat, lon, location_name, tz, source = _session_location(session, locations)
        nights = session.nights
        folder = os.path.commonpath(session.folders) if len(session.folders) > 1 else session.folders[0]
        cursor = conn.execute("""
            INSERT INTO usersessions (
                target_id, dso_name, ra_deg, dec_deg, status, session_date,
                location_lat, location_lon, location_name, location_timezone, location_source,
                telescope_id, camera_id, camera, notes, modified_date
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (_target_id(group.name, targets), group.name, ra, dec, status,
              nights[0].isoformat() if nights else today.isoformat(),
              lat, lon, location_name, tz, source, session.telescope_id, session.camera_id, session.camera or None,
              f"Imported from {folder}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        session_id = cursor.lastrowid
        report.sessions_created += 1
        if status == "Completed":
            report.completed += 1
        else:
            report.in_progress += 1

    lights = SessionObservations.attach_files(conn, session_id, [_file_dict(r) for r in session.lights])
    calibration = SessionObservations.attach_files(conn, session_id, [_file_dict(r) for r in session.calibration])
    SessionObservations.assign_files_to_observations(conn, session_id)
    SessionObservations.recompute_session_aggregates(conn, session_id, promote=False)
    MountDetection.save(conn, session_id, session.mounts, tidy=False)
    report.subs_added += lights["inserted"]
    report.calibration_added += calibration["inserted"]
    report.subs_skipped += sum(lights["dup_other_sessions"].values())
    if session.telescope_id is not None and session.camera_id is not None:
        # The camera was used with the telescope (the Equipment window's "Used with")
        conn.execute("INSERT OR IGNORE INTO telescope_equipment (telescope_id, equipment_id) VALUES (?, ?)",
                     (session.telescope_id, session.camera_id))
    for record in session.lights:
        key = os.path.normcase(os.path.normpath(record.path))
        by_path.setdefault(key, session_id)
    return session_id


def link_gallery_images(conn, library, by_path=None, by_key=None, report=None):
    """Link the gallery images already in the collection (and not linked yet) to
    the Session Manager session their paired session's subs are in - whether
    imported now or before. Matched by path (any of the image's formats), under
    whichever objects the gallery has it. Does not commit; returns how many."""
    if by_path is None or by_key is None:
        by_path, by_key = _owners(conn)
    unlinked = {}
    for image_id, path in conn.execute(
            "SELECT id, image_path FROM userimages WHERE session_id IS NULL AND image_path IS NOT NULL"):
        unlinked.setdefault(_norm_path(path), []).append(image_id)
    linked = 0
    for group in library.groups.values():
        for final in group.finals:
            if not final.imported or final.session is None:
                continue
            ids = [i for r in [final.record] + final.record.variants for i in unlinked.pop(_norm_path(r.path), [])]
            if not ids:
                continue
            session_id = (report.session_ids.get(id(final.session)) if report else None) or \
                _existing_session(final.session, by_path, by_key)
            if session_id is None:
                continue
            conn.execute(f"UPDATE userimages SET session_id = ? WHERE id IN ({','.join('?' * len(ids))})",
                         [session_id] + ids)
            linked += len(ids)
    if report:
        report.images_linked += linked
    return linked


def _import_images(conn, finals, library, by_path, by_key, report):
    for final in finals:
        group = library.groups.get(final.dso_key)
        exists = conn.execute("SELECT 1 FROM userimages WHERE dsodetailid = ? AND image_path = ?",
                              (group.dsodetailid, final.record.path)).fetchone()
        if exists:
            report.images_skipped += 1
            continue
        session = final.session
        session_id = None
        if session is not None:
            session_id = report.session_ids.get(id(session)) or _existing_session(session, by_path, by_key)
        equipment = {"telescope_id": session.telescope_id, "telescope_name": session.telescope or None,
                     "camera_id": session.camera_id, "camera_name": session.camera or None} if session else None
        integration = _format_integration(session.integration_seconds) if session else ""
        taken = final.date
        add_user_image_row(conn.cursor(), group.dsodetailid, final.record.path, integration,
                           taken.strftime("%Y-%m-%d") if taken else "", "", session_id, equipment=equipment)
        report.images_added += 1
        report.imported_finals.append(final)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _format_duration(seconds):
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def print_library(library, scan, out, new_only=False):
    roots = scan.roots

    def rel(path):
        return os.path.relpath(path, _root_of(path, roots))

    out.write(f"Scanned {', '.join(roots)} in {scan.seconds:.0f}s: {scan.folders_listed} folders, "
              f"{scan.files_seen} files, {len(scan.records)} considered, {scan.headers_read} headers read, "
              f"{scan.headers_cached} from the index, {scan.new_files} new since the last scan\n")
    kinds = Counter(r.kind for r in scan.records)
    out.write("Kinds: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()) + "\n")
    out.write("Notes: " + ", ".join(f"{v} {k}" for k, v in sorted(library.counts.items()) if v) + "\n")
    for path, reason in sorted(scan.skipped_folders):
        out.write(f"  skipped ({reason}): {rel(path)}\n")
    for folder, n in sorted(library.raw_folders.items()):
        out.write(f"  camera raw ({n}): {rel(folder)}\n")
    for record in library.unreadable[:20]:
        out.write(f"  unreadable: {rel(record.path)} ({record.reason})\n")

    known_groups = sorted((g for g in library.groups.values() if g.dsodetailid), key=lambda g: g.name)
    unknown_groups = sorted((g for g in library.groups.values() if not g.dsodetailid), key=lambda g: g.name)
    out.write(f"\n{len(known_groups)} DSOs, {len(unknown_groups)} unknown groups, "
              f"{sum(len(g.sessions) for g in library.groups.values())} sessions, "
              f"{sum(len(g.finals) for g in library.groups.values())} final images "
              f"({sum(f.checked for g in library.groups.values() for f in g.finals)} checked)\n")
    for group in known_groups + unknown_groups:
        sessions = [s for s in group.sessions if s.new_lights or not new_only]
        finals = [f for f in group.finals if f.is_new or not new_only]
        if new_only and not sessions and not finals:
            continue
        label = group.name if group.dsodetailid else f"UNKNOWN: {group.name}"
        out.write(f"\n== {label}  ({len(group.sessions)} sessions, {len(group.finals)} finals)\n")
        for folder, deg in group.mismatches:
            out.write(f"   ! named {group.name} but pointed {deg:.1f} deg away: {rel(folder)}\n")
        for s in sessions:
            n = group.sessions.index(s) + 1
            new = f" ({s.new_lights} new)" if s.new_lights else ""
            filters = ", ".join(f"{k} {v}" for k, v in s.filters.most_common())
            nights = s.nights
            out.write(f"   [{'x' if s.checked else ' '}] Session {n}: {nights[0]} -> {nights[-1]}, "
                      f"{len(s.nights)} nights, {len(s.lights)} lights, {_format_duration(s.integration_seconds)}, "
                      f"{s.camera or '?'} / {s.telescope or '?'}, {filters} [{s.status}]{new}\n")
            for night, mount in s.night_mounts.items():
                out.write(f"         {night} mount: {mount.mount or 'Unknown'} ({mount.source or '-'}) - "
                          f"{mount.evidence}\n")
            cal = Counter(r.frame_type for r in s.calibration)
            if cal:
                out.write(f"         + calibration: {', '.join(f'{v} {k}' for k, v in cal.items())}\n")
            folders = s.folders
            out.write(f"         folders: {rel(folders[0])}" + (f" (+{len(folders) - 1} more)" if len(folders) > 1 else "")
                      + "\n")
        for f in finals:
            r = f.record
            variants = f" (+{', '.join(sorted({v.ext for v in r.variants}))})" if r.variants else ""
            session = (f" -> session {group.sessions.index(f.session) + 1}" if f.session else "")
            flags = (" [imported]" if f.imported else "") + (" [new]" if f.is_new else "") + (f" {{{', '.join(sorted(f.variant_words))}}}"
                                                              if f.variant_words else "")
            out.write(f"   [{'x' if f.checked else ' '}] T{r.tier} {rel(r.path)}{variants} "
                      f"({r.reason}; DSO by {f.resolution.source or 'folder'}){session}{flags}\n")
    calibration = [c for c in library.calibration_libraries if c.is_new or not new_only]
    if calibration:
        out.write("\n== Calibration libraries\n")
        for c in calibration:
            exposures = sorted({exposure_of(r.header) for r in c.files if exposure_of(r.header) is not None})
            out.write(f"   {c.frame_type} x{len(c.files)} {c.camera or '?'} "
                      f"{', '.join(f'{e:g}s' for e in exposures)}: {rel(c.folder)}\n")


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Scan folders for astrophotography sessions and final images.")
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--workers", type=int, default=DEFAULT_READ_WORKERS)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_SESSION_GAP_DAYS)
    parser.add_argument("--full", action="store_true", help="Read every header again, ignoring the scan index")
    parser.add_argument("--no-index", action="store_true", help="Don't use or update the scan index")
    parser.add_argument("--new-only", action="store_true", help="Only list what's new since the last scan")
    parser.add_argument("--no-mounts", action="store_true", help="Don't tell each night's mount (no image reads)")
    parser.add_argument("--out", help="Write the report here instead of the console")
    args = parser.parse_args(argv)

    rules = ScanRules(workers=args.workers, session_gap_days=args.gap_days)
    scan_index = None if args.no_index else ScanIndex()
    if scan_index:
        for root in args.roots:
            sys.stderr.write(f"{root}: last scanned {scan_index.last_scan(root) or 'never'}\n")

    def progress(scan):
        sys.stderr.write(f"\r{scan.seconds:6.0f}s  folders {scan.folders_listed}  files {scan.files_seen}  "
                         f"headers {scan.headers_read + scan.headers_cached}   ")
        sys.stderr.flush()

    started = time.perf_counter()
    scan, library = scan_and_build(args.roots, rules, progress=progress, reread=args.full, scan_index=scan_index,
                                   status=lambda text: sys.stderr.write(f"\r{text}   "), mounts=not args.no_mounts)
    sys.stderr.write(f"\nDone in {time.perf_counter() - started:.1f}s\n")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            print_library(library, scan, out, args.new_only)
    else:
        print_library(library, scan, sys.stdout, args.new_only)


if __name__ == "__main__":
    main()
