#!/usr/bin/env python3
"""
Session Observations
Data-layer helpers for the Session Manager's observations - one observing night
of a session. Groups a session's subs into nights (including nights that run past
midnight), keeps the same sub from being counted twice, and merges hand-logged
subs with scanned files into the "counted" totals every screen shows.

Double-count rules (all enforced here, so no screen can show inflated totals):
  A. The same sub attached twice to a session under a different path (copied
     folder, renamed or calibrated copy) is caught by its sub_key fingerprint.
  B. A sub already attached to another session is skipped, never attached twice.
  C. Hand-logged subs and scanned files for the same night/filter/exposure are
     never added together: counted = max(files, logged).
"""

import os
import re
import logging
from collections import Counter
from datetime import datetime, timedelta

import SessionFileScanner

logger = logging.getLogger(__name__)

# A sub taken before this local hour belongs to the previous evening's night.
NIGHT_CUTOFF_HOUR = 12

# Longest a single observation (one night) may span.
MAX_OBSERVATION_SPAN = timedelta(hours=24)

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Keys are already lowercased with spaces/hyphens/underscores stripped.
FILTER_ALIASES = {
    "halpha": "ha", "hydrogenalpha": "ha",
    "o3": "oiii", "oxygen3": "oiii", "oxygeniii": "oiii",
    "s2": "sii", "sulfur2": "sii", "sulfurii": "sii", "sulphurii": "sii",
    "lum": "l", "luminance": "l",
    "red": "r", "green": "g", "blue": "b",
}


# ---------------------------------------------------------------------------
# Schema / migration
# ---------------------------------------------------------------------------

def ensure_schema(conn):
    """Create the observation tables and the usersessionfiles columns they need,
    backfill sub_key fingerprints, and remove within-session duplicate subs before
    the unique fingerprint index is created. Idempotent. Commits."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usersessionobservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES usersessions(id) ON DELETE CASCADE,
            night_date TEXT NOT NULL,
            start_datetime TEXT,
            end_datetime TEXT,
            notes TEXT,
            is_manual INTEGER DEFAULT 0,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP,
            modified_date TEXT,
            UNIQUE(session_id, night_date)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessionobservations_session_id "
                   "ON usersessionobservations(session_id)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usersessionobservationfilters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_id INTEGER NOT NULL REFERENCES usersessionobservations(id) ON DELETE CASCADE,
            filter_name TEXT,
            exposure_seconds REAL NOT NULL DEFAULT 0,
            sub_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessionobservationfilters_observation_id "
                   "ON usersessionobservationfilters(observation_id)")

    for column_sql in ("ALTER TABLE usersessionfiles ADD COLUMN observation_id INTEGER",
                       "ALTER TABLE usersessionfiles ADD COLUMN sub_key TEXT"):
        try:
            cursor.execute(column_sql)
        except Exception:
            pass  # Column already exists
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessionfiles_observation_id "
                   "ON usersessionfiles(observation_id)")

    affected_sessions = _backfill_sub_keys(conn)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_usersessionfiles_session_sub_key "
                   "ON usersessionfiles(session_id, sub_key) WHERE sub_key IS NOT NULL")
    conn.commit()
    return affected_sessions


def _backfill_sub_keys(conn):
    """Fingerprint files attached before sub_key existed. A second copy of the
    same sub within one session was double-counted before - its row is removed
    (the file on disk is never touched). Returns the ids of sessions that lost rows."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_id, date_obs, exptime_seconds, filter_name, frame_type, camera
        FROM usersessionfiles WHERE sub_key IS NULL AND date_obs IS NOT NULL
    """)
    pending = [tuple(row) for row in cursor.fetchall()]  # sqlite3.Row isn't orderable
    if not pending:
        return set()

    cursor.execute("SELECT session_id, sub_key FROM usersessionfiles WHERE sub_key IS NOT NULL")
    seen = {(row[0], row[1]) for row in cursor.fetchall()}

    affected = set()
    removed = 0
    for file_id, session_id, date_obs, exptime, filter_name, frame_type, camera in sorted(pending):
        key = make_sub_key(date_obs, exptime, filter_name, frame_type, camera)
        if key is None:
            continue
        if (session_id, key) in seen:
            cursor.execute("DELETE FROM usersessionfiles WHERE id = ?", (file_id,))
            affected.add(session_id)
            removed += 1
            continue
        seen.add((session_id, key))
        cursor.execute("UPDATE usersessionfiles SET sub_key = ? WHERE id = ?", (key, file_id))

    if removed:
        logger.info(f"Removed {removed} duplicate session file row(s) while backfilling sub fingerprints")
    return affected


def migrate(conn):
    """Full startup migration: schema, then assign every session's files to
    observations and recompute totals for any session whose data changed. Commits."""
    affected = ensure_schema(conn)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT session_id FROM usersessionfiles
        WHERE observation_id IS NULL AND date_obs IS NOT NULL AND frame_type = 'Light'
    """)
    affected |= {row[0] for row in cursor.fetchall()}
    for session_id in sorted(affected):
        assign_files_to_observations(conn, session_id)
        recompute_session_aggregates(conn, session_id, promote=False)
    conn.commit()


# ---------------------------------------------------------------------------
# Time / night helpers
# ---------------------------------------------------------------------------

def parse_obs_datetime(value):
    """Parse a DATE-OBS style string ('2026-09-22T03:14:15.123Z', with or without
    fractional seconds/offset) into a datetime, or None."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    match = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})", text)
    if match:
        return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}")
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None


def to_local(value, tz_name):
    """DATE-OBS is UTC per the FITS standard - convert to the session's local
    time (naive) so nights split at local noon, not UTC noon. With no usable
    timezone the raw value is used, as it always was."""
    dt = parse_obs_datetime(value)
    if dt is None:
        return None
    if tz_name:
        try:
            import pytz
            tz = pytz.timezone(tz_name)
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            return dt.astimezone(tz).replace(tzinfo=None)
        except Exception:
            pass
    return dt.replace(tzinfo=None)


def night_date_for(dt_local):
    """The observing night a local datetime belongs to: its evening's date. A sub
    at 02:00 on Saturday belongs to Friday night, so a night that runs past
    midnight stays one night."""
    if dt_local.hour < NIGHT_CUTOFF_HOUR:
        return dt_local.date() - timedelta(days=1)
    return dt_local.date()


def session_timezone(conn, session_id):
    """The session's own timezone, else the active saved location's."""
    cursor = conn.cursor()
    cursor.execute("SELECT location_timezone FROM usersessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if row and row[0]:
        return row[0]
    cursor.execute("SELECT timezone FROM usersettings WHERE is_active = 1 AND timezone IS NOT NULL LIMIT 1")
    row = cursor.fetchone()
    return row[0] if row else None


def format_datetime(dt):
    return dt.strftime(DATETIME_FORMAT) if dt else None


def format_night_span(night_date, start_datetime=None, end_datetime=None):
    """'Sep 22' for a same-date night, 'Sep 22 → 23, 21:10 – 04:35' across
    midnight (month shown on both sides when it changes)."""
    try:
        night = datetime.strptime(str(night_date), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return str(night_date or "")
    start = parse_obs_datetime(start_datetime)
    end = parse_obs_datetime(end_datetime)
    if not (start and end):
        return f"{night:%Y-%m-%d}"
    if end.date() == start.date():
        label = f"{start:%Y-%m-%d}"
    elif end.month == start.month:
        label = f"{start:%Y-%m-%d} → {end.day:02d}"
    else:
        label = f"{start:%Y-%m-%d} → {end:%b %d}"
    return f"{label}, {start:%H:%M} – {end:%H:%M}"


# ---------------------------------------------------------------------------
# Sub identity / filter matching
# ---------------------------------------------------------------------------

def normalize_filter(name):
    """Canonical filter key, so 'Ha', 'H-alpha' and 'H_Alpha' all match."""
    if not name:
        return ""
    key = re.sub(r"[\s_\-]+", "", str(name)).lower()
    return FILTER_ALIASES.get(key, key)


def exposure_bucket(exposure_seconds):
    """Exposures within about half a second of each other are the same setting."""
    try:
        return int(round(float(exposure_seconds or 0)))
    except (TypeError, ValueError):
        return 0


def make_sub_key(date_obs, exptime, filter_name, frame_type, camera):
    """Fingerprint of one physical sub, independent of its path: a copied,
    renamed or calibrated copy of a sub keeps these header values. None when the
    sub has no DATE-OBS (those fall back to path-only matching)."""
    dt = parse_obs_datetime(date_obs)
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None) - (dt.utcoffset() or timedelta(0))
    try:
        exposure = f"{float(exptime):.1f}" if exptime is not None else ""
    except (TypeError, ValueError):
        exposure = ""
    return "|".join((
        dt.strftime(DATETIME_FORMAT), exposure, normalize_filter(filter_name),
        (frame_type or "").strip().lower(), (camera or "").strip().lower(),
    ))


# ---------------------------------------------------------------------------
# Counted totals (rule C)
# ---------------------------------------------------------------------------

def combine_breakdown(file_groups, logged_rows):
    """Merge one night's scanned Light files with its hand-logged rows.

    file_groups: [{"filter_name", "exposure_seconds", "count", "seconds"}]
    logged_rows: [{"filter_name", "exposure_seconds", "sub_count"}]

    Returns rows sorted by filter then exposure:
      {"filter_name", "exposure_seconds", "file_subs", "logged_subs",
       "counted_subs", "counted_seconds"}
    where counted = max(files, logged) - the two are evidence of the same subs,
    never added together."""
    merged = {}
    for group in file_groups:
        key = (normalize_filter(group["filter_name"]), exposure_bucket(group["exposure_seconds"]))
        row = merged.setdefault(key, {
            "filter_name": group["filter_name"] or "", "exposure_seconds": group["exposure_seconds"] or 0,
            "file_subs": 0, "file_seconds": 0.0, "logged_subs": 0,
        })
        row["file_subs"] += group["count"]
        row["file_seconds"] += group["seconds"] or 0

    for logged in logged_rows:
        key = (normalize_filter(logged["filter_name"]), exposure_bucket(logged["exposure_seconds"]))
        row = merged.setdefault(key, {
            "filter_name": logged["filter_name"] or "", "exposure_seconds": logged["exposure_seconds"] or 0,
            "file_subs": 0, "file_seconds": 0.0, "logged_subs": 0,
        })
        row["logged_subs"] += int(logged["sub_count"] or 0)

    result = []
    for row in merged.values():
        if row["file_subs"] >= row["logged_subs"]:
            counted_subs, counted_seconds = row["file_subs"], row["file_seconds"]
        else:
            counted_subs = row["logged_subs"]
            counted_seconds = counted_subs * float(row["exposure_seconds"] or 0)
        result.append({
            "filter_name": row["filter_name"], "exposure_seconds": row["exposure_seconds"],
            "file_subs": row["file_subs"], "logged_subs": row["logged_subs"],
            "counted_subs": counted_subs, "counted_seconds": counted_seconds,
        })
    result.sort(key=lambda r: (normalize_filter(r["filter_name"]), r["exposure_seconds"] or 0))
    return result


def summarize_breakdown(rows):
    """(counted_subs, counted_seconds, sorted display filter names) for a breakdown."""
    subs = sum(r["counted_subs"] for r in rows)
    seconds = sum(r["counted_seconds"] for r in rows)
    names = {}
    for r in rows:
        if r["counted_subs"] > 0 and r["filter_name"]:
            names.setdefault(normalize_filter(r["filter_name"]), r["filter_name"])
    return subs, seconds, sorted(names.values(), key=str.lower)


def _load_file_groups(conn, observation_ids):
    """{observation_id: [file_group, ...]} for Light files."""
    groups = {obs_id: [] for obs_id in observation_ids}
    if not observation_ids:
        return groups
    cursor = conn.cursor()
    for chunk in _chunks(list(observation_ids)):
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(f"""
            SELECT observation_id, filter_name, exptime_seconds, COUNT(*), COALESCE(SUM(exptime_seconds), 0)
            FROM usersessionfiles
            WHERE observation_id IN ({placeholders}) AND frame_type = 'Light'
            GROUP BY observation_id, filter_name, exptime_seconds
        """, chunk)
        for obs_id, filter_name, exposure, count, seconds in cursor.fetchall():
            groups[obs_id].append({
                "filter_name": filter_name, "exposure_seconds": exposure,
                "count": count, "seconds": seconds,
            })
    return groups


def _load_logged_rows(conn, observation_ids):
    """{observation_id: [logged_row, ...]}."""
    rows = {obs_id: [] for obs_id in observation_ids}
    if not observation_ids:
        return rows
    cursor = conn.cursor()
    for chunk in _chunks(list(observation_ids)):
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(f"""
            SELECT observation_id, filter_name, exposure_seconds, sub_count
            FROM usersessionobservationfilters
            WHERE observation_id IN ({placeholders}) ORDER BY id
        """, chunk)
        for obs_id, filter_name, exposure, sub_count in cursor.fetchall():
            rows[obs_id].append({
                "filter_name": filter_name, "exposure_seconds": exposure, "sub_count": sub_count,
            })
    return rows


def load_breakdowns(conn, observation_ids):
    """{observation_id: combine_breakdown(...)} in two queries."""
    file_groups = _load_file_groups(conn, observation_ids)
    logged = _load_logged_rows(conn, observation_ids)
    return {obs_id: combine_breakdown(file_groups[obs_id], logged[obs_id]) for obs_id in observation_ids}


def load_observations(conn, session_id):
    """All of a session's observations, oldest night first, each with its
    logged rows, Light file groups and total attached file count."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, night_date, start_datetime, end_datetime, notes, is_manual
        FROM usersessionobservations WHERE session_id = ? ORDER BY night_date
    """, (session_id,))
    observations = [{
        "id": row[0], "night_date": row[1], "start_datetime": row[2], "end_datetime": row[3],
        "notes": row[4] or "", "is_manual": bool(row[5]),
    } for row in cursor.fetchall()]

    ids = [o["id"] for o in observations]
    file_groups = _load_file_groups(conn, ids)
    logged = _load_logged_rows(conn, ids)
    file_counts = {}
    for chunk in _chunks(ids):
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(f"""
            SELECT observation_id, COUNT(*) FROM usersessionfiles
            WHERE observation_id IN ({placeholders}) GROUP BY observation_id
        """, chunk)
        file_counts.update(dict(cursor.fetchall()))

    for obs in observations:
        obs["file_groups"] = file_groups[obs["id"]]
        obs["logged"] = logged[obs["id"]]
        obs["file_count"] = file_counts.get(obs["id"], 0)
    return observations


def load_session_files(conn, session_id):
    """Every attached file (all frame types) as dicts, oldest first."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.id, f.observation_id, f.file_path, f.frame_type, f.filter_name, f.exptime_seconds, f.gain,
               f.ccd_temp, f.xbinning, f.ybinning, f.date_obs, o.night_date
        FROM usersessionfiles f
        LEFT JOIN usersessionobservations o ON o.id = f.observation_id
        WHERE f.session_id = ?
        ORDER BY f.date_obs, f.file_path
    """, (session_id,))
    columns = ["id", "observation_id", "file_path", "frame_type", "filter_name", "exptime_seconds", "gain",
               "ccd_temp", "xbinning", "ybinning", "date_obs", "night_date"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def recompute_session_aggregates(conn, session_id, promote=True):
    """Recompute the session's sub_count / integration_seconds / filters_used /
    date range from its observations' counted totals (plus any Light files that
    couldn't be placed on a night because they have no DATE-OBS). Does not commit.

    With promote, a 'Planned' session that now has data becomes 'In Progress';
    it then stays 'In Progress' until the user marks it 'Completed'. A session
    with no observations keeps its typed filters_used (a planned filter list)."""
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM usersessions WHERE id = ?", (session_id,))
    status_row = cursor.fetchone()
    if not status_row:
        return
    current_status = status_row[0]

    cursor.execute("SELECT id FROM usersessionobservations WHERE session_id = ?", (session_id,))
    obs_ids = [row[0] for row in cursor.fetchall()]
    rows = [r for bd in load_breakdowns(conn, obs_ids).values() for r in bd]
    sub_count, integration_seconds, filters = summarize_breakdown(rows)

    cursor.execute("""
        SELECT COUNT(*), COALESCE(SUM(exptime_seconds), 0) FROM usersessionfiles
        WHERE session_id = ? AND observation_id IS NULL AND frame_type = 'Light'
    """, (session_id,))
    loose_count, loose_seconds = cursor.fetchone()
    sub_count += loose_count
    integration_seconds += loose_seconds
    if loose_count:
        cursor.execute("""
            SELECT DISTINCT filter_name FROM usersessionfiles
            WHERE session_id = ? AND observation_id IS NULL AND frame_type = 'Light'
                  AND filter_name IS NOT NULL AND filter_name != ''
        """, (session_id,))
        known = {normalize_filter(f) for f in filters}
        for (name,) in cursor.fetchall():
            if normalize_filter(name) not in known:
                filters.append(name)
                known.add(normalize_filter(name))
        filters.sort(key=str.lower)

    cursor.execute("SELECT MIN(date_obs), MAX(date_obs), COUNT(*) FROM usersessionfiles WHERE session_id = ?",
                   (session_id,))
    earliest, latest, file_count = cursor.fetchone()
    if earliest is None and obs_ids:
        cursor.execute("""
            SELECT MIN(COALESCE(start_datetime, night_date)), MAX(COALESCE(end_datetime, night_date))
            FROM usersessionobservations WHERE session_id = ?
        """, (session_id,))
        earliest, latest = cursor.fetchone()

    has_data = bool(obs_ids) or file_count > 0
    new_status = "In Progress" if (promote and has_data and current_status == "Planned") else current_status
    modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if has_data:
        cursor.execute("""
            UPDATE usersessions SET
                sub_count = ?, integration_seconds = ?, earliest_sub_date = ?, latest_sub_date = ?,
                filters_used = ?, status = ?, modified_date = ?
            WHERE id = ?
        """, (sub_count, integration_seconds, earliest, latest, ", ".join(filters),
              new_status, modified, session_id))
    else:
        cursor.execute("""
            UPDATE usersessions SET
                sub_count = 0, integration_seconds = 0, earliest_sub_date = NULL, latest_sub_date = NULL,
                modified_date = ?
            WHERE id = ?
        """, (modified, session_id))


# ---------------------------------------------------------------------------
# Attaching files (rules A and B) and grouping them into nights
# ---------------------------------------------------------------------------

def _rows_from_file_dicts(file_dicts):
    rows = []
    for f in file_dicts:
        row = SessionFileScanner.file_dict_to_row(f)
        if not row.get("file_path"):
            continue
        row["sub_key"] = make_sub_key(row["date_obs"], row["exptime_seconds"], row["filter_name"],
                                      row["frame_type"], row["camera"])
        rows.append(row)
    return rows


def _normalized_path(path):
    return os.path.normcase(os.path.normpath(path or ""))


def _find_owners(conn, exclude_session_id, rows):
    """[owner label or None per row] - the session (other than exclude_session_id)
    that already holds each row's sub, matched by fingerprint or by path."""
    cursor = conn.cursor()
    owners_by_key, owners_by_path = {}, {}
    keys = sorted({r["sub_key"] for r in rows if r["sub_key"]})
    for chunk in _chunks(keys):
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(f"""
            SELECT f.sub_key, s.dso_name, s.session_date FROM usersessionfiles f
            JOIN usersessions s ON s.id = f.session_id
            WHERE f.session_id != ? AND f.sub_key IN ({placeholders})
        """, [exclude_session_id] + chunk)
        for key, dso_name, session_date in cursor.fetchall():
            owners_by_key.setdefault(key, f"{dso_name} — {session_date}")

    # Path-only check covers subs without DATE-OBS. Paths are compared
    # case-insensitively on Windows, so fetch the candidates and normalize here.
    cursor.execute("""
        SELECT f.file_path, s.dso_name, s.session_date FROM usersessionfiles f
        JOIN usersessions s ON s.id = f.session_id WHERE f.session_id != ?
    """, (exclude_session_id,))
    wanted = {_normalized_path(r["file_path"]) for r in rows}
    for file_path, dso_name, session_date in cursor.fetchall():
        normalized = _normalized_path(file_path)
        if normalized in wanted:
            owners_by_path.setdefault(normalized, f"{dso_name} — {session_date}")

    return [
        (owners_by_key.get(r["sub_key"]) if r["sub_key"] else None) or owners_by_path.get(_normalized_path(r["file_path"]))
        for r in rows
    ]


def count_already_attached(conn, file_dicts):
    """Counter({session label: n}) of scanned files that some session already holds."""
    rows = _rows_from_file_dicts(file_dicts)
    return Counter(owner for owner in _find_owners(conn, -1, rows) if owner)


def attach_files(conn, session_id, file_dicts):
    """Insert scanned files into a session, skipping every kind of duplicate.
    Does not commit. Returns a report dict:
      {"inserted": n, "dup_in_session": n, "dup_other_sessions": Counter({label: n})}"""
    report = {"inserted": 0, "dup_in_session": 0, "dup_other_sessions": Counter()}
    rows = _rows_from_file_dicts(file_dicts)
    if not rows:
        return report

    cursor = conn.cursor()
    for row, owner in zip(rows, _find_owners(conn, session_id, rows)):
        if owner:
            report["dup_other_sessions"][owner] += 1
            continue
        cursor.execute("""
            INSERT OR IGNORE INTO usersessionfiles (
                session_id, file_path, file_type, frame_type, object_name, date_obs,
                exptime_seconds, filter_name, camera, telescope, gain, offset_value,
                ccd_temp, xbinning, ybinning, header_json, sub_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            session_id, row["file_path"], row["file_type"], row["frame_type"],
            row["object_name"], row["date_obs"], row["exptime_seconds"], row["filter_name"],
            row["camera"], row["telescope"], row["gain"], row["offset_value"],
            row["ccd_temp"], row["xbinning"], row["ybinning"], row["header_json"], row["sub_key"],
        ))
        if cursor.rowcount == 1:
            report["inserted"] += 1
        else:
            report["dup_in_session"] += 1
    return report


def format_attach_report(report):
    """Human-readable summary of an attach_files() report, or '' if every file was new."""
    other_total = sum(report["dup_other_sessions"].values())
    if not report["dup_in_session"] and not other_total:
        return ""
    lines = [f"{report['inserted']} new file(s) attached."]
    if report["dup_in_session"]:
        lines.append(f"{report['dup_in_session']} file(s) were already in this session and were skipped.")
    if other_total:
        lines.append(f"{other_total} file(s) already belong to another session and were skipped:")
        for label, count in report["dup_other_sessions"].most_common():
            lines.append(f"    • {count} in {label}")
    return "\n".join(lines)


def assign_files_to_observations(conn, session_id):
    """Place every file with a DATE-OBS on its observing night (local time, noon
    cutoff), creating observations as needed. Files always follow their own
    timestamps, so this also re-homes files if the session's timezone changed.
    Auto-created observations that end up with no files are removed; observations
    the user created or edited are kept. Each observation's start/end is set to
    its subs' span (auto) or widened to include them (user-edited). Does not commit."""
    tz_name = session_timezone(conn, session_id)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, night_date, start_datetime, end_datetime, is_manual
        FROM usersessionobservations WHERE session_id = ?
    """, (session_id,))
    observations = {row[1]: {"id": row[0], "start": row[2], "end": row[3], "is_manual": bool(row[4])}
                    for row in cursor.fetchall()}

    cursor.execute("""
        SELECT id, date_obs, observation_id, frame_type FROM usersessionfiles
        WHERE session_id = ? AND date_obs IS NOT NULL
    """, (session_id,))
    # night_date str -> {"files": [(id, current_obs_id)], "min"/"max": Light sub span or None}
    nights = {}
    for file_id, date_obs, current_obs_id, frame_type in cursor.fetchall():
        local = to_local(date_obs, tz_name)
        if local is None:
            continue
        night = nights.setdefault(night_date_for(local).isoformat(), {"files": [], "min": None, "max": None})
        night["files"].append((file_id, current_obs_id))
        # Only Light subs define a night and its span - morning flats or darks
        # join the night they fall on but don't create or stretch one.
        if frame_type == "Light":
            night["min"] = local if night["min"] is None else min(night["min"], local)
            night["max"] = local if night["max"] is None else max(night["max"], local)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for night_date, night in list(nights.items()):
        obs = observations.get(night_date)
        if night["min"] is None and not (obs and (obs["is_manual"] or _has_user_content(cursor, obs["id"]))):
            # Calibration frames only, and no observation the user made or annotated.
            unplaced = [file_id for file_id, current in night["files"] if current is not None]
            for chunk in _chunks(unplaced):
                placeholders = ",".join("?" * len(chunk))
                cursor.execute(f"UPDATE usersessionfiles SET observation_id = NULL WHERE id IN ({placeholders})",
                               chunk)
            del nights[night_date]
            continue
        if obs is None:
            cursor.execute("""
                INSERT INTO usersessionobservations
                    (session_id, night_date, start_datetime, end_datetime, is_manual, modified_date)
                VALUES (?, ?, ?, ?, 0, ?)
            """, (session_id, night_date, format_datetime(night["min"]), format_datetime(night["max"]), now))
            obs = {"id": cursor.lastrowid, "is_manual": False}
            observations[night_date] = obs
        elif night["min"] is not None:
            if obs["is_manual"]:
                start = parse_obs_datetime(obs["start"])
                end = parse_obs_datetime(obs["end"])
                start = min(start, night["min"]) if start else night["min"]
                end = max(end, night["max"]) if end else night["max"]
            else:
                start, end = night["min"], night["max"]
            cursor.execute("UPDATE usersessionobservations SET start_datetime = ?, end_datetime = ? WHERE id = ?",
                           (format_datetime(start), format_datetime(end), obs["id"]))

        stale = [file_id for file_id, current in night["files"] if current != obs["id"]]
        for chunk in _chunks(stale):
            placeholders = ",".join("?" * len(chunk))
            cursor.execute(f"UPDATE usersessionfiles SET observation_id = ? WHERE id IN ({placeholders})",
                           [obs["id"]] + chunk)

    # Drop auto-created nights with no Light subs left and nothing the user added.
    for night_date, obs in observations.items():
        if obs["is_manual"] or night_date in nights or _has_user_content(cursor, obs["id"]):
            continue
        cursor.execute("UPDATE usersessionfiles SET observation_id = NULL WHERE observation_id = ?", (obs["id"],))
        cursor.execute("DELETE FROM usersessionobservations WHERE id = ?", (obs["id"],))


def _has_user_content(cursor, observation_id):
    """Logged subs or notes - an auto-created observation with either is kept."""
    cursor.execute("SELECT COUNT(*) FROM usersessionobservationfilters WHERE observation_id = ?", (observation_id,))
    if cursor.fetchone()[0]:
        return True
    cursor.execute("SELECT notes FROM usersessionobservations WHERE id = ?", (observation_id,))
    row = cursor.fetchone()
    return bool(row and (row[0] or "").strip())


# ---------------------------------------------------------------------------
# Deletes (SQLite foreign keys aren't enforced here, so cascade by hand)
# ---------------------------------------------------------------------------

def delete_observation(conn, observation_id):
    """Delete an observation, its logged rows and its attached file rows (files on
    disk are never touched). Does not commit."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM usersessionobservationfilters WHERE observation_id = ?", (observation_id,))
    cursor.execute("DELETE FROM usersessionfiles WHERE observation_id = ?", (observation_id,))
    cursor.execute("DELETE FROM usersessionobservations WHERE id = ?", (observation_id,))


def delete_session_children(conn, session_id):
    """Delete everything hanging off a session. Does not commit."""
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM usersessionobservationfilters WHERE observation_id IN
            (SELECT id FROM usersessionobservations WHERE session_id = ?)
    """, (session_id,))
    cursor.execute("DELETE FROM usersessionobservations WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM usersessionfiles WHERE session_id = ?", (session_id,))


def _chunks(items, size=500):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]
