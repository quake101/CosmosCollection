"""
Light frame quality for a session or a set of files, without windows: analyzing
lights (with the measurements cached per file), grading them with the saved
thresholds, the stacking choices (Good only, top % ...), and moving, deleting
or removing frames from a session.

Only PySide6.QtCore (QSettings) - used by the Light Quality review window
(LightQualityReview.py, which re-exports these names), session completion and
the command line's 'quality'. The measuring and grading themselves are in
FrameQuality.py.
"""

import json
import logging
import math
import os
import queue
import shutil
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor


from cosmoscollection.imaging import FrameQuality
from cosmoscollection.imaging.ImageArrays import read_file_bytes
from cosmoscollection.core.Settings import app_settings
from cosmoscollection.sessions import SessionFileScanner
from cosmoscollection.sessions import SessionObservations

logger = logging.getLogger(__name__)


def analysis_workers():
    """Parallel file reads + analyses: Settings → Maximum Threads, the same
    setting used for parallel work elsewhere in the app."""
    settings = app_settings()
    default_threads = max(1, (os.cpu_count() or 4) - 2)
    return max(1, min(settings.value("max_threads", default_threads, type=int), 128))


READ_THREADS = 16                    # files read at once - waiting on a disk or network, not the CPU
READ_AHEAD_BYTES = 512 * 1024 * 1024  # most file bytes held waiting for an analysis thread

SETTINGS_PREFIX = "light_quality/"

# GradeSettings field, label, spin box prefix/suffix, range, step, decimals
THRESHOLDS = [
    ("flux_drop_pct", "Dimmer stars (clouds, haze)", "-", "%", 5, 95, 1, 0),
    ("star_drop_pct", "Fewer stars (obstruction)", "-", "%", 10, 95, 1, 0),
    ("fwhm_rise_pct", "Bigger stars (seeing, focus)", "+", "%", 5, 300, 1, 0),
    ("eccentricity_rise", "More elongated (trailing, guiding)", "+", "", 0.02, 0.5, 0.01, 2),
    ("outside_rise_pct", "Light outside star cores (guiding jumps, halos)", "+", "%", 10, 300, 5, 0),
    ("background_rise_pct", "Brighter sky (moon, dawn)", "+", "%", 10, 500, 5, 0),
    ("signal_drop_pct", "Less signal (moon, dawn, haze; 0 = off)", "-", "%", 0, 95, 1, 0),
    ("trail_count", "Satellite trails in one frame (Marginal at, 0 = off)", "", "", 0, 50, 1, 0),
]


# ---- Cache ----------------------------------------------------------------------

def ensure_cache_table(conn):
    """Measurements are keyed by file path, so a sub attached to several sessions
    is only analyzed once."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS framequality (
            file_path TEXT PRIMARY KEY,
            file_size INTEGER,
            file_mtime_ns INTEGER,
            version INTEGER,
            metrics_json TEXT,
            analyzed_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)


def load_cached(conn, paths):
    """{path: (size, mtime_ns, version, FrameMetrics)} for the paths with stored measurements."""
    cached = {}
    paths = list(paths)
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        rows = conn.execute(
            f"SELECT file_path, file_size, file_mtime_ns, version, metrics_json FROM framequality "
            f"WHERE file_path IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        for path, size, mtime_ns, version, metrics_json in rows:
            try:
                # The path is the row's only: older saves also kept one in the JSON,
                # stale once a folder was renamed or moved (relinked)
                metrics = FrameQuality.FrameMetrics.from_dict({**json.loads(metrics_json), "path": path})
            except (TypeError, ValueError):
                continue
            cached[path] = (size, mtime_ns, version, metrics)
    return cached


def save_cached(conn, entries):
    """entries: [(FrameMetrics, size, mtime_ns)]. The path is stored only as the
    row's key, so moving a file is one update. Does not commit."""
    conn.executemany(
        "INSERT OR REPLACE INTO framequality (file_path, file_size, file_mtime_ns, version, metrics_json) "
        "VALUES (?, ?, ?, ?, ?)",
        [(m.path, size, mtime_ns, m.version, json.dumps({k: v for k, v in m.to_dict().items() if k != "path"}))
         for m, size, mtime_ns in entries])


# ---- Analyzing ------------------------------------------------------------------

def _stat(path):
    try:
        return os.stat(path)
    except OSError:
        return None


def _read(path):
    """A FITS file's bytes, read ahead on a thread of its own (on a network
    share most of a frame's time is waiting for its bytes) - None for XISF,
    which analyze_file reads itself."""
    if os.path.splitext(path)[1].lower() in SessionFileScanner.XISF_EXTENSIONS:
        return None
    try:
        return read_file_bytes(path)
    except Exception:
        return None  # analyze_file reads it again and reports why


def analyze(paths, cached, force=False, workers=1, on_checked=None, on_frame=None, cancelled=None):
    """Analyze the paths whose cached measurements are missing or out of date,
    several at a time. cached: load_cached()'s result; force: analyze them all.

    on_checked(count): how many frames need analyzing, once the files are checked.
    on_frame(FrameMetrics, (size, mtime_ns) to cache it under - or None): per
    frame analyzed, or not found. cancelled(): stop soon - frames already being
    analyzed still finish and are reported."""
    on_checked = on_checked or (lambda count: None)
    on_frame = on_frame or (lambda metrics, stamp: None)
    cancelled = cancelled or (lambda: False)

    # Files are read on threads of their own, ahead of the analyses: from a
    # network share a frame spends most of its time waiting for its bytes,
    # and with one pool doing both the reads were never hidden behind the
    # analyses (on a NAS, 8 threads analyzed about half as fast as the network).
    read_threads = max(READ_THREADS, workers)
    readers = ThreadPoolExecutor(max_workers=read_threads)
    analyzers = ThreadPoolExecutor(max_workers=workers)
    results = queue.Queue()  # (analysis future or None, (size, mtime_ns)) - one per frame started

    def read_then_analyze(path, stamp):
        if cancelled():
            results.put((None, stamp))
            return
        raw = _read(path)
        try:
            analysis = analyzers.submit(FrameQuality.analyze_file, path, raw)
        except RuntimeError:  # stopped - the analyzers are shut down
            results.put((None, stamp))
            return
        analysis.add_done_callback(lambda future: results.put((future, stamp)))

    try:
        # A few ms each on a network share - thousands of subs one at a time
        # would be a long wait before the first analysis starts
        todo = deque()
        for path, stat in zip(paths, readers.map(_stat, paths)):
            if cancelled():
                return
            if stat is None:
                on_frame(FrameQuality.FrameMetrics(path=path, error="File not found"), None)
                continue
            hit = None if force else cached.get(path)
            if (hit and hit[0] == stat.st_size and hit[1] == stat.st_mtime_ns
                    and hit[2] == FrameQuality.ANALYSIS_VERSION):
                continue  # the cached measurements still hold
            todo.append((path, stat.st_size, stat.st_mtime_ns))

        on_checked(len(todo))
        # Frames read or being read but not analyzed yet, kept to a few per
        # thread and to READ_AHEAD_BYTES so big subs can't fill memory
        max_frames = read_threads + 2 * workers
        started = started_bytes = 0
        stopping = False
        while True:
            if cancelled() and not stopping:
                stopping = True
                analyzers.shutdown(wait=False, cancel_futures=True)
            while (todo and not stopping and started < max_frames
                   and (not started or started_bytes + todo[0][1] <= READ_AHEAD_BYTES)):
                path, size, mtime_ns = todo.popleft()
                readers.submit(read_then_analyze, path, (size, mtime_ns))
                started += 1
                started_bytes += size
            if not started:
                break
            try:
                analysis, stamp = results.get(timeout=0.2)
            except queue.Empty:
                continue
            started -= 1
            started_bytes -= stamp[0]
            # Frames already being analyzed when stopped still finish - keep them too
            if analysis is not None and not analysis.cancelled():
                metrics = analysis.result()
                on_frame(metrics, stamp if metrics.ok else None)
    finally:
        readers.shutdown(wait=True, cancel_futures=True)
        analyzers.shutdown(wait=True, cancel_futures=True)


# ---- Grading --------------------------------------------------------------------

# Which lights a stacking handoff takes (SessionCompletionDialog's Lights choice)
LIGHTS_ALL, LIGHTS_GOOD, LIGHTS_GOOD_MARGINAL, LIGHTS_TOP, LIGHTS_PICKED = (
    "all", "good", "good_marginal", "top", "picked")
LIGHT_CHOICE_LABELS = {
    LIGHTS_ALL: "All lights",
    LIGHTS_GOOD: "Good only",
    LIGHTS_GOOD_MARGINAL: "Good + Marginal",
    LIGHTS_TOP: "Top {percent}% of each filter",
    LIGHTS_PICKED: "Picked in the quality review",
}
# Frames in a group too small to compare can't be judged, so they're kept
KEEP_GRADES = {
    LIGHTS_GOOD: {FrameQuality.GRADE_GOOD, FrameQuality.GRADE_UNGRADED},
    LIGHTS_GOOD_MARGINAL: {FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_UNGRADED},
}
TOP_PERCENT_SETTING = SETTINGS_PREFIX + "top_percent"
DEFAULT_TOP_PERCENT = 75
RANKED_GRADES = (FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_REJECT)


def light_choice_label(choice, percent):
    return LIGHT_CHOICE_LABELS[choice].format(percent=percent)


def saved_top_percent():
    value = app_settings().value(TOP_PERCENT_SETTING, DEFAULT_TOP_PERCENT, type=int)
    return min(max(value, 1), 100)


def save_top_percent(percent):
    app_settings().setValue(TOP_PERCENT_SETTING, int(percent))


def top_percent_lights(lights, grades, percent):
    """The best `percent` of each group's graded lights by score - always at
    least one. Per group, as scores only compare frames of the same filter,
    exposure and binning (which also keeps the stack's filter balance). Frames
    in groups too small to grade are kept; missing or unanalyzed ones aren't.
    lights: {path: file dict}, grades: {path: FrameGrade}."""
    keep, ranked = set(), {}
    for path, g in grades.items():
        if path not in lights:
            continue
        if g.grade == FrameQuality.GRADE_UNGRADED:
            keep.add(path)
        elif g.grade in RANKED_GRADES:
            ranked.setdefault(group_key(lights[path]), []).append((g.score, path))
    for members in ranked.values():
        members.sort(reverse=True)
        keep.update(path for _score, path in members[:max(1, math.ceil(len(members) * percent / 100))])
    return keep


def session_light_files(conn, session_id):
    """{path: file dict} for a session's lights, oldest first. Same rule as the
    stacking handoff: untyped files are lights only when the session has no typed lights."""
    files = SessionObservations.load_session_files(conn, session_id)
    has_typed = any(f["frame_type"] == "Light" for f in files)
    lights = {}
    for f in files:
        path = f["file_path"]
        if path and path not in lights and (f["frame_type"] == "Light"
                                            or (f["frame_type"] == "Unknown" and not has_typed)):
            lights[path] = f
    return lights


def group_key(light):
    """Frames are grouped by filter, exposure and binning - and graded within
    that by night (FrameQuality.grade_frames)."""
    return (light["filter_name"] or "No filter", light["exptime_seconds"], light["xbinning"] or 1)


def saved_grade_settings():
    """The thresholds last set in the review window."""
    settings = app_settings()
    defaults = FrameQuality.GradeSettings()
    return FrameQuality.GradeSettings(**{
        name: settings.value(SETTINGS_PREFIX + name, getattr(defaults, name), type=float)
        for name, *_rest in THRESHOLDS})


def grade_session_lights(conn, session_id):
    """(lights, {path: FrameGrade}) for a session from its saved measurements and
    the saved thresholds. Lights never analyzed have no grade; the grades are
    empty when none has been analyzed."""
    lights = session_light_files(conn, session_id)
    ensure_cache_table(conn)
    metrics = {path: m for path, (_size, _mtime, version, m) in load_cached(conn, lights).items()
               if version == FrameQuality.ANALYSIS_VERSION}
    if not metrics:
        return lights, {}
    return lights, grade_lights(lights, metrics, saved_grade_settings())


def grade_lights(lights, metrics, settings):
    """{path: FrameGrade} for the measured lights. lights: {path: file dict},
    metrics: {path: FrameMetrics} (paths not in lights are left out)."""
    paths = [p for p in metrics if p in lights]
    return FrameQuality.grade_frames(((group_key(lights[p]), metrics[p]) for p in paths), settings,
                                     gains={p: lights[p]["gain"] for p in paths},
                                     nights={p: lights[p]["night_date"] for p in paths})


def bad_nights(lights, grades):
    """{night date: (grade, issue, frames, Counter of their grades)} for the
    nights graded down as a whole (FrameQuality._check_nights) - Reject when
    any filter's was. Frames that hold up against a typical night on their
    own keep their grade."""
    nights = {}
    for path, g in grades.items():
        if g.night_grade and path in lights:
            night = lights[path]["night_date"] or ""
            grade, issue, count, counts = nights.get(night, (g.night_grade, g.night_issue, 0, Counter()))
            if g.night_grade == FrameQuality.GRADE_REJECT:
                grade = FrameQuality.GRADE_REJECT
            counts[g.grade] += 1
            nights[night] = (grade, issue, count + 1, counts)
    return nights


# ---- Moving, deleting and removing frames ---------------------------------------

MOVE, DELETE = "move", "delete"


def _unused_path(path):
    """path, or 'name (2).fits' etc. if something's already there."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    number = 2
    while os.path.exists(f"{stem} ({number}){ext}"):
        number += 1
    return f"{stem} ({number}){ext}"


def move_or_delete(mode, paths, destination=None, progress=None, cancelled=None):
    """Move (MOVE) or permanently delete (DELETE) files one at a time. A move
    goes into destination, or a SessionFileScanner.REJECTED_FOLDER beside each
    file when None. progress(done, total) after each.

    Returns ({path: new path, or None when deleted}, {path: error text}) - only
    files that succeeded are in the first."""
    finished, failed = {}, {}
    for i, path in enumerate(paths, 1):
        if cancelled and cancelled():
            break
        try:
            if mode == DELETE:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass  # already gone - still detach it
                finished[path] = None
            else:
                folder = destination or os.path.join(os.path.dirname(path), SessionFileScanner.REJECTED_FOLDER)
                if os.path.normcase(os.path.abspath(folder)) == os.path.normcase(os.path.dirname(os.path.abspath(path))):
                    failed[path] = "Already in that folder"
                else:
                    os.makedirs(folder, exist_ok=True)
                    target = _unused_path(os.path.join(folder, os.path.basename(path)))
                    shutil.move(path, target)
                    finished[path] = target
        except OSError as e:
            failed[path] = e.strerror or str(e)
        if progress:
            progress(i, len(paths))
    return finished, failed


def other_sessions(conn, session_id, paths):
    """{session_id: (label, [file ids])} for other sessions these files are also attached to."""
    others = {}
    paths = list(paths)
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        rows = conn.execute(
            f"SELECT f.session_id, f.id, s.dso_name, s.session_date FROM usersessionfiles f "
            f"JOIN usersessions s ON s.id = f.session_id "
            f"WHERE f.session_id != ? AND f.file_path IN ({','.join('?' * len(chunk))})",
            [session_id] + chunk).fetchall()
        for other_id, file_id, name, date in rows:
            others.setdefault(other_id, (f"{name} ({date})", []))[1].append(file_id)
    return others


def record_removed(conn, session_id, lights, changed, mode, extra_ids=(), others=None):
    """Detach frames from the session - changed maps each path to its new path
    when moved (MOVE), or None when deleted (DELETE); mode None: removed from
    the session only - and update other sessions and the saved measurements to
    match: a moved file stays attached to other sessions at its new place, a
    deleted one is removed from them too. lights: {path: file dict} (the
    session's). extra_ids: other session files to detach too (a night's
    calibration frames). others: other_sessions() for a delete, if already
    looked up. session_id None: frames graded from folders, in no session of
    their own (lights unused). Does not commit."""
    if session_id is not None:
        SessionObservations.remove_files(conn, session_id, [lights[p]["id"] for p in changed] + list(extra_ids))
    if mode == MOVE:
        for old, new in changed.items():
            conn.execute("UPDATE usersessionfiles SET file_path = ? WHERE file_path = ? AND session_id IS NOT ?",
                         (new, old, session_id))
            conn.execute("UPDATE OR REPLACE framequality SET file_path = ? WHERE file_path = ?", (new, old))
    elif mode == DELETE:
        if others is None:
            others = other_sessions(conn, session_id if session_id is not None else -1, changed)
        for other_id, (_label, ids) in others.items():
            SessionObservations.remove_files(conn, other_id, ids)
        paths = list(changed)
        for i in range(0, len(paths), 500):
            chunk = paths[i:i + 500]
            conn.execute(f"DELETE FROM framequality WHERE file_path IN ({','.join('?' * len(chunk))})", chunk)
