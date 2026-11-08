"""
Checking a session's attached files on disk: which are missing, which are on a
drive that isn't there, and where missing ones went (a renamed or moved night
folder), plus calibration frames left on no night.

No UI imports - used by the Session File Check dialog and the Session Manager
(SessionFileCheck.py, which re-exports these names) and the command line's
'sessions check'. Files on disk are never changed.
"""

import os
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from cosmoscollection.sessions import SessionFileScanner
from cosmoscollection.core.Paths import path_key
from cosmoscollection.sessions import SessionObservations

logger = logging.getLogger(__name__)

# A missing folder's renamed copy is looked for under its nearest existing
# parent, at most this many levels up (a night folder inside a target folder).
SEARCH_LEVELS_UP = 3
# Most folders one search lists - keeps a mistaken pick of a whole drive bounded.
MAX_SEARCH_FOLDERS = 20000

# What's wrong with a group of files (group_problems)
MOVED, MISSING, UNREACHABLE, NO_NIGHT = "moved", "missing", "unreachable", "no_night"


# ---- Disk checks (worker threads) ---------------------------------------------

def missing_paths(paths, cancelled=None):
    """(missing, unreachable): the paths not on disk, and of those, the ones on a
    drive or share that isn't there at all (offline NAS, unplugged disk) - those
    may well come back. One folder listing per folder instead of a check per file,
    which is much faster on a network drive."""
    by_folder = defaultdict(list)
    for path in paths:
        by_folder[os.path.dirname(path)].append(path)
    missing, unreachable = set(), set()
    for folder, files in by_folder.items():
        if cancelled and cancelled():
            break
        try:
            names = {os.path.normcase(name) for name in os.listdir(folder)}
        except OSError:
            missing.update(files)
            if not _existing_ancestor(folder):
                unreachable.update(files)
            continue
        missing.update(p for p in files if os.path.normcase(os.path.basename(p)) not in names)
    return missing, unreachable


def _existing_ancestor(folder, levels=None):
    """(nearest existing folder above folder, levels climbed), or None - within
    levels, or up to the drive or share root."""
    current, climbed = folder, 0
    while True:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return None
        current, climbed = parent, climbed + 1
        if levels is not None and climbed > levels:
            return None
        if os.path.isdir(current):
            return current, climbed


def _list_tree(root, depth, wanted, found, budget, cancelled):
    """Add files under root (up to depth folders down) whose normcased name is in
    wanted to found {name: [paths]}. Skips Rejected folders. budget: [folders left]."""
    stack = [(root, 0)]
    while stack and budget[0] > 0:
        if cancelled and cancelled():
            return
        folder, level = stack.pop()
        budget[0] -= 1
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if level < depth and entry.name.lower() != SessionFileScanner.REJECTED_FOLDER.lower():
                                stack.append((entry.path, level + 1))
                        elif os.path.normcase(entry.name) in wanted:
                            found[os.path.normcase(entry.name)].append(os.path.normpath(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue


def _fingerprint(path):
    data = SessionFileScanner.scan_file(path)
    if data is None:
        return None
    row = SessionFileScanner.file_dict_to_row(data)
    return SessionObservations.make_sub_key(row["date_obs"], row["exptime_seconds"], row["filter_name"],
                                           row["frame_type"], row["camera"])


def find_moved(files, search_root=None, progress=None, cancelled=None):
    """{file id: new path} for missing files found again by name, and confirmed
    by their headers' fingerprint when the file has one (a name-only match must
    be the only file of that name). files: file dicts with id, file_path, sub_key.
    With search_root None each file is looked for near where it was - under its
    nearest existing parent folder, about as deep as it was (a renamed night
    folder); otherwise anywhere under search_root."""
    progress = progress or (lambda text: None)
    wanted = {os.path.normcase(os.path.basename(f["file_path"])) for f in files}
    found = defaultdict(list)
    budget = [MAX_SEARCH_FOLDERS]
    if search_root:
        progress(f"Looking through {search_root}...")
        _list_tree(search_root, 64, wanted, found, budget, cancelled)
    else:
        searches = {}
        for folder in {os.path.dirname(f["file_path"]) for f in files}:
            anchor = _existing_ancestor(folder, SEARCH_LEVELS_UP) if not os.path.isdir(folder) else (folder, 0)
            if anchor:
                root, climbed = anchor
                key = os.path.normcase(root)
                searches[key] = (root, max(climbed + 1, searches.get(key, (root, 0))[1]))
        for root, depth in searches.values():
            progress(f"Looking for moved files in {root}...")
            _list_tree(root, depth, wanted, found, budget, cancelled)

    moves = {}
    for n, f in enumerate(files):
        if cancelled and cancelled():
            break
        old = path_key(f["file_path"])
        candidates = [p for p in found.get(os.path.normcase(os.path.basename(f["file_path"])), [])
                      if path_key(p) != old]
        if not candidates:
            continue
        progress(f"Checking the files found: {n + 1:,} of {len(files):,}")
        if f.get("sub_key"):
            match = next((p for p in candidates if _fingerprint(p) == f["sub_key"]), None)
        else:
            match = candidates[0] if len(candidates) == 1 else None
        if match:
            moves[f["id"]] = match
    return moves


@dataclass
class CheckResult:
    missing: set = field(default_factory=set)       # file ids not on disk
    unreachable: set = field(default_factory=set)   # of those, on a drive or share that isn't there
    moved: dict = field(default_factory=dict)       # file id -> where it was found


def check_files(files, find=False, search_root=None, status=None, cancelled=None):
    """CheckResult for file dicts (SessionObservations.load_session_files): which
    are on disk and, with find, where the missing ones (not on an unavailable
    drive) are now. status(text) is told what it's doing."""
    status = status or (lambda text: None)
    result = CheckResult()
    status(f"Checking {len(files):,} file(s) on disk...")
    by_path = defaultdict(list)
    for f in files:
        by_path[f["file_path"]].append(f["id"])
    missing, unreachable = missing_paths(list(by_path), cancelled)
    result.missing = {i for p in missing for i in by_path[p]}
    result.unreachable = {i for p in unreachable for i in by_path[p]}
    lost = [f for f in files if f["id"] in result.missing and f["id"] not in result.unreachable]
    if find and lost:
        result.moved = find_moved(lost, search_root, status, cancelled)
    return result


def unattached(conn, moved):
    """moved without the found paths some session already has - relinking those
    would attach a sub twice."""
    if not moved:
        return moved
    attached = {path_key(row[0])
                for row in conn.execute("SELECT file_path FROM usersessionfiles")}
    return {i: p for i, p in moved.items() if path_key(p) not in attached}


# ---- Summaries ------------------------------------------------------------------

def stranded_calibration(files):
    """Calibration frames on no night - a night with no lights left, or taken on
    a night the session has no lights for."""
    return [f for f in files if f["observation_id"] is None and f["frame_type"] != "Light"]


def problem_summary(files, result):
    """(text, is_warning) for the Files tab's line, or ('', False) when all's well."""
    by_id = {f["id"]: f for f in files}
    missing = Counter(by_id[i]["frame_type"] for i in result.missing if i in by_id)
    alone = Counter(f["frame_type"] for f in stranded_calibration(files) if f["id"] not in result.missing)
    parts = []
    if missing:
        unreachable = len(result.unreachable)
        where = " (their drive isn't available)" if unreachable and unreachable == sum(missing.values()) else ""
        parts.append(f"{SessionObservations.describe_frame_counts(missing)} "
                     f"{'is' if sum(missing.values()) == 1 else 'are'} missing on disk{where}")
    if alone:
        parts.append(f"{SessionObservations.describe_frame_counts(alone)} "
                     f"{'is' if sum(alone.values()) == 1 else 'are'}n't on any night")
    if not parts:
        return "", False
    return "; ".join(parts) + ".", bool(missing)


def group_problems(files, result):
    """The files needing attention, grouped by folder and what's wrong:
    [((kind, folder, new folder or None), [file dicts])], moved first, then
    missing, unreachable and calibration on no night."""
    groups = defaultdict(list)  # (kind, folder, new folder) -> files
    for f in files:
        folder = os.path.normpath(os.path.dirname(f["file_path"]))
        if f["id"] in result.moved:
            groups[(MOVED, folder, os.path.dirname(result.moved[f["id"]]))].append(f)
        elif f["id"] in result.unreachable:
            groups[(UNREACHABLE, folder, None)].append(f)
        elif f["id"] in result.missing:
            groups[(MISSING, folder, None)].append(f)
    for f in stranded_calibration(files):
        if f["id"] not in result.missing:
            groups[(NO_NIGHT, os.path.normpath(os.path.dirname(f["file_path"])), None)].append(f)
    order = {MOVED: 0, MISSING: 1, UNREACHABLE: 2, NO_NIGHT: 3}
    return sorted(groups.items(), key=lambda item: (order[item[0][0]], item[0][1].lower()))
