"""
Session Manager sessions as rows: the list the Session Manager shows, and an
object's sessions. Shared with the command line, so no Qt here.
"""

from cosmoscollection.catalog.DSONames import NOT_DSO_TYPES
from cosmoscollection.equipment import EquipmentMapper
from cosmoscollection.planning.TargetRows import SAME_POSITION_DEG, catalog_object_ids, has_position, name_key
from cosmoscollection.sessions import MountDetection
from cosmoscollection.sessions.SessionFileScanner import angular_separation_deg

SESSION_COLUMNS = [
    "id", "target_id", "dso_name", "ra_deg", "dec_deg", "status",
    "session_date", "start_time", "end_time",
    "location_lat", "location_lon", "location_name", "location_timezone", "location_source",
    "telescope_id", "telescope_name",
    "camera_id", "camera", "filters_used", "sub_count", "integration_seconds",
    "earliest_sub_date", "latest_sub_date", "notes",
    "created_date", "modified_date",
    "mount_override", "mount_detected", "telescope_mount", "nights",
]


def load_sessions(conn, session_id=None):
    """Every session (or the one with session_id), newest first, as dicts of
    SESSION_COLUMNS plus "mount" and "mount_how" (MountDetection.summarize).
    camera is its Equipment's name, else the name it was given; nights is how
    many nights it has."""
    sql = f"""
        SELECT s.id, s.target_id, s.dso_name, s.ra_deg, s.dec_deg, s.status,
               s.session_date, s.start_time, s.end_time,
               s.location_lat, s.location_lon, s.location_name, s.location_timezone, s.location_source,
               s.telescope_id, tel.name,
               s.camera_id, {EquipmentMapper.session_camera_sql('s')},
               s.filters_used, s.sub_count, s.integration_seconds,
               s.earliest_sub_date, s.latest_sub_date, s.notes,
               s.created_date, s.modified_date,
               s.mount_override, {MountDetection.detected_sql('s')}, tel.mount_type,
               (SELECT COUNT(*) FROM usersessionobservations o WHERE o.session_id = s.id)
        FROM usersessions s
        LEFT JOIN usertelescopes tel ON s.telescope_id = tel.id
    """
    params = ()
    if session_id is not None:
        sql += " WHERE s.id = ?"
        params = (session_id,)
    sql += " ORDER BY s.session_date DESC, s.id DESC"
    sessions = [dict(zip(SESSION_COLUMNS, tuple(row))) for row in conn.execute(sql, params).fetchall()]
    for session in sessions:
        session["mount"], session["mount_how"] = MountDetection.summarize(
            session["mount_detected"], session["mount_override"], session["telescope_mount"])
    return sessions


# A session made from dropped subs stores where they were pointed, which can be
# off the catalog center by the framing - this far still counts as the object
POINTED_NEAR_DEG = 5 / 60


def sessions_for_object(conn, name, ra=None, dec=None, dsodetailid=None):
    """An object's Session Manager sessions, newest first: [(id, name, date, status,
    has framing)]. A session is the object's when it's linked to the object's Target
    List entry or named as it (any designation: "M 42 (Orion Nebula)", "NGC 1976"),
    is at its exact position, or was pointed within POINTED_NEAR_DEG of it with no
    other catalog object closer (so M 43's sessions aren't M 42's)."""
    cursor = conn.cursor()
    name = (name or "").strip()
    position = (float(ra), float(dec)) if has_position({"ra_deg": ra, "dec_deg": dec}) else None

    # The catalog objects the name designates, and every designation of them
    objects = set(catalog_object_ids(cursor, name)) if name else set()
    if dsodetailid:
        objects.add(dsodetailid)
    keys = {name_key(name)} if name else set()
    if objects:
        cursor.execute(f"SELECT catalogue || ' ' || designation FROM cataloguenr WHERE dsodetailid IN "
                       f"({','.join('?' * len(objects))})", list(objects))
        keys.update(name_key(row[0]) for row in cursor.fetchall())
    keys.discard("")

    def exact_position(s_ra, s_dec):
        return (position is not None and has_position({"ra_deg": s_ra, "dec_deg": s_dec})
                and abs(s_ra - position[0]) < SAME_POSITION_DEG and abs(s_dec - position[1]) < SAME_POSITION_DEG)

    # Its Target List entries (by any designation or the same position, as find_existing_target
    # matches them - without its whole-list catalog lookup)
    entries = {target_id for target_id, target_name, t_ra, t_dec in cursor.execute(
        "SELECT id, name, ra_deg, dec_deg FROM usertargetlist").fetchall()
        if name_key(target_name or "") in keys or exact_position(t_ra, t_dec)}

    found = []
    for session_id, session_name, date, status, s_ra, s_dec, target_id, framing in conn.execute(
            "SELECT id, dso_name, session_date, status, ra_deg, dec_deg, target_id, framing FROM usersessions "
            "ORDER BY session_date DESC, id DESC").fetchall():
        same = target_id in entries or name_key(session_name or "") in keys
        if not same and position and has_position({"ra_deg": s_ra, "dec_deg": s_dec}):
            distance = angular_separation_deg(position[0], position[1], s_ra, s_dec)
            same = exact_position(s_ra, s_dec) or (
                distance < POINTED_NEAR_DEG and not _other_object_closer(cursor, s_ra, s_dec, distance, objects))
        if same:
            found.append((session_id, session_name, date, status, bool(framing)))
    return found


def _other_object_closer(cursor, ra, dec, distance, objects):
    """Whether a catalog object other than objects is within distance of (ra, dec)."""
    cursor.execute(f"""
        SELECT id, ra, dec FROM dsodetail
        WHERE dec BETWEEN ? AND ? AND COALESCE(dsotype, '') NOT IN ({','.join('?' * len(NOT_DSO_TYPES))})
    """, (dec - distance, dec + distance, *NOT_DSO_TYPES))
    return any(obj not in objects and o_ra is not None and o_dec is not None
               and angular_separation_deg(ra, dec, float(o_ra), float(o_dec)) < distance
               for obj, o_ra, o_dec in cursor.fetchall())
