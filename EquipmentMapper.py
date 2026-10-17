"""
EquipmentMapper.py - maps the telescope and camera names on gallery images and
sessions (and in the files they came from) to the user's Equipment.

Images link through userimages.telescope_id / camera_id, sessions through
usersessions.telescope_id / camera_id, so renaming a telescope renames it
everywhere. Each keeps the name it was given (from its files, or typed in):
userimages.telescope_name / camera_name, usersessions.camera. That name is
shown when it isn't linked - gear that isn't in Equipment, or was deleted -
and matching it to equipment later (the Equipment window's Matches tab, the
Import Wizard) links everything given it. A session's telescope is picked
from Equipment, so it has no name of its own.

Images from before these columns kept "telescope, camera" text in
userimages.equipment, and sessions their files' camera name: migrate()
converts them at startup.
"""

import json
import logging
import re
import sqlite3
from collections import Counter, defaultdict

import LibraryScanner
from LibraryScanner import TELESCOPE, CAMERA

logger = logging.getLogger(__name__)

IMAGES, SESSIONS = "userimages", "usersessions"
# Where a name links to equipment: (table, kind) -> (id column, name column)
LINKS = {(IMAGES, TELESCOPE): ("telescope_id", "telescope_name"),
         (IMAGES, CAMERA): ("camera_id", "camera_name"),
         (SESSIONS, CAMERA): ("camera_id", "camera")}
COLUMNS = {kind: LINKS[(IMAGES, kind)] for kind in (TELESCOPE, CAMERA)}  # An image's
TABLES = {TELESCOPE: "usertelescopes", CAMERA: "userequipment"}


def _add_columns(cursor, table, columns):
    added = False
    for column in columns:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            added = True
        except sqlite3.OperationalError:
            pass  # Column already exists
    return added


def ensure_columns(cursor):
    """Add the image equipment columns. True when they're new - the
    database's first run with them."""
    return _add_columns(cursor, IMAGES, ("telescope_id INTEGER", "camera_id INTEGER",
                                         "telescope_name TEXT", "camera_name TEXT"))


def ensure_session_columns(cursor):
    """Add usersessions.camera_id. True when it's new."""
    return _add_columns(cursor, SESSIONS, ("camera_id INTEGER",))


def _key(name):
    return (name or "").strip().casefold()


def _clean(name):
    name = (name or "").strip()
    return name or None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def select_sql(alias="ui"):
    """SQL for the columns image_equipment() reads: the ids and names saved
    on the image, then its telescope and camera as shown - its Equipment's
    names, else the names it was given."""
    return (f"{alias}.telescope_id, {alias}.telescope_name, {alias}.camera_id, {alias}.camera_name, "
            f"COALESCE((SELECT name FROM usertelescopes WHERE id = {alias}.telescope_id), {alias}.telescope_name), "
            f"COALESCE((SELECT name FROM userequipment WHERE id = {alias}.camera_id), {alias}.camera_name)")


def session_camera_sql(alias="s"):
    """SQL for a session's camera as shown: its Equipment's name, else the
    name it was given."""
    return f"COALESCE((SELECT name FROM userequipment WHERE id = {alias}.camera_id), {alias}.camera)"


def equipment_text(telescope, camera):
    """'ZenithStar 61, ASI2600MC Pro' - an image's equipment, as shown."""
    return ", ".join(name.strip() for name in (telescope, camera) if name and name.strip())


def equipment_key(telescope_id, telescope_name, camera_id, camera_name):
    """The same for images taken with the same telescope and camera: linked
    ones by their Equipment entry, others by name."""
    def part(prefix, item_id, name):
        if item_id is not None:
            return f"{prefix}{item_id}"
        return f"n:{_key(name)}" if _key(name) else ""
    return f"{part('t', telescope_id, telescope_name)}|{part('c', camera_id, camera_name)}"


def image_equipment(values):
    """From select_sql()'s six columns: {"telescope", "camera" (as shown, ''
    when none), "equipment" (both, as text), "equipment_key", "telescope_id",
    "camera_id"}."""
    telescope_id, telescope_name, camera_id, camera_name, telescope, camera = values
    return {"telescope": (telescope or "").strip(), "camera": (camera or "").strip(),
            "equipment": equipment_text(telescope, camera),
            "equipment_key": equipment_key(telescope_id, telescope_name, camera_id, camera_name),
            "telescope_id": telescope_id, "camera_id": camera_id}


def describe(changed):
    """'3 images and 1 session' for {table: how many} - '' for none."""
    parts = [f"{changed[table]} {word}{'s' if changed[table] != 1 else ''}"
             for table, word in ((IMAGES, "image"), (SESSIONS, "session")) if changed.get(table)]
    return " and ".join(parts)


# ---------------------------------------------------------------------------
# Names -> equipment
# ---------------------------------------------------------------------------

def resolve(kind, name, info):
    """The id of the equipment a name means, or None: the equipment it's
    matched to (Import Wizard / Matches tab), else the user's telescope or
    camera of that name. info: LibraryScanner.EquipmentInfo."""
    if not _key(name):
        return None
    equipment_id, _shown = info.match(kind, name)
    if equipment_id is not None:
        return equipment_id
    items = info.items(kind)
    named = [item_id for item_id, item in items.items() if _key(item["name"]) == _key(name)]
    # Enabled equipment first, should two share the name
    return min(named, key=lambda i: (not items[i]["is_active"], i)) if named else None


def suggest(kind, name, info, header=None):
    """(equipment id, why) for a name that isn't the user's equipment or
    matched to it, as the Import Wizard suggests (a similar name, or the
    header's focal length / pixel size) - (None, "") when nothing fits, or
    the user chose to leave the name unmatched."""
    if not _key(name) or info.decided(kind, name):
        return None, ""
    return LibraryScanner.suggest_equipment(kind, name.strip(), header, info)


def display_name(kind, name, info, header=None):
    """A name from a file or session as it's best shown: the user's
    equipment it means, or would be suggested for it - else the name itself."""
    name = (name or "").strip()
    equipment_id = resolve(kind, name, info)
    if equipment_id is None:
        equipment_id, _why = suggest(kind, name, info, header)
    item = info.items(kind).get(equipment_id) if equipment_id is not None else None
    return item["name"] if item else name


_LABEL = re.compile(r"^\s*(telescope|scope|camera)\s*:\s*", re.IGNORECASE)


def parse_equipment_text(text, info):
    """(telescope_id, telescope_name, camera_id, camera_name) from an image's
    old equipment text: "telescope, camera" as the Add Image dialog joined
    them, or either alone. Split where the most of it names the user's
    equipment, else at the first comma - a name alone is a telescope, unless
    it names one of the user's cameras."""
    text = (text or "").strip()
    if not text:
        return None, None, None, None
    splits = [(text[:m.start()], text[m.end():]) for m in re.finditer(r",\s*", text)]
    best, best_score = (None, _LABEL.sub("", text).strip() or None, None, None), -1
    for index, (telescope, camera) in enumerate(splits + [(text, ""), ("", text)]):
        telescope, camera = _LABEL.sub("", telescope).strip(), _LABEL.sub("", camera).strip()
        if index < len(splits) and not (telescope and camera):
            continue
        telescope_id = resolve(TELESCOPE, telescope, info)
        camera_id = resolve(CAMERA, camera, info)
        if index == len(splits) + 1 and camera_id is None:
            continue  # A name alone is a camera only when it names one
        score = (telescope_id is not None) + (camera_id is not None)
        if score > best_score:
            best, best_score = (telescope_id, telescope or None, camera_id, camera or None), score
    return best


def link_name(conn, kind, name, equipment_id, previous_id=None):
    """Images and sessions given this name (any case) link to equipment_id,
    or to nothing when None - those not linked yet, or linked to previous_id
    (what the name meant before). Ones linked to other equipment are left
    alone. Does not commit; {table: how many changed}."""
    changed = Counter()
    key = _key(name)
    if not key:
        return changed
    for (table, link_kind), (id_column, name_column) in LINKS.items():
        if link_kind != kind:
            continue
        ids = [row_id for row_id, row_name, linked in conn.execute(
            f"SELECT id, {name_column}, {id_column} FROM {table} WHERE {name_column} IS NOT NULL")
            if _key(row_name) == key and linked != equipment_id and (linked is None or linked == previous_id)]
        conn.executemany(f"UPDATE {table} SET {id_column} = ? WHERE id = ?", [(equipment_id, i) for i in ids])
        changed[table] += len(ids)
    return changed


def save_matches(conn, decisions):
    """Save {(kind, name in files): equipment id or None} as the Import Wizard
    does, and link the images and sessions given those names. Commits;
    {table: how many changed}."""
    previous = LibraryScanner.load_equipment_info(conn).aliases
    LibraryScanner.save_equipment_aliases(conn, decisions)
    changed = Counter()
    for (kind, name), equipment_id in decisions.items():
        changed += link_name(conn, kind, name, equipment_id, previous.get((kind, _key(name))))
    conn.commit()
    return changed


def forget_match(conn, kind, name):
    """Remove a saved match: the Import Wizard asks about the name again.
    Images and sessions stay as they are. Commits."""
    LibraryScanner.load_equipment_info(conn)  # The table exists
    conn.execute("DELETE FROM equipmentaliases WHERE kind = ? AND alias_key = ?", (kind, _key(name)))
    conn.commit()


def matches(conn, info=None):
    """The names the Matches tab lists, by kind then name: [{"kind", "name",
    "equipment_id" (None: not matched), "saved" (a saved match), "used"
    ({table: how many were given the name})}]. The saved matches (but not a
    name matched to equipment of that same name), then the names on images
    and sessions that aren't the name of the user's equipment."""
    info = info or LibraryScanner.load_equipment_info(conn)
    used, spelled = defaultdict(Counter), {}
    for (table, kind), (_id_column, name_column) in LINKS.items():
        for (name,) in conn.execute(f"SELECT {name_column} FROM {table} "
                                    f"WHERE TRIM(COALESCE({name_column}, '')) != ''"):
            used[(kind, _key(name))][table] += 1
            spelled.setdefault((kind, _key(name)), name.strip())
    found = {}
    for kind, key, header_value, equipment_id in conn.execute(
            "SELECT kind, alias_key, header_value, equipment_id FROM equipmentaliases"):
        if kind not in COLUMNS:
            continue
        if equipment_id not in info.items(kind):
            equipment_id = None  # Matched to equipment that's since been deleted
        elif _key(info.items(kind)[equipment_id]["name"]) == key:
            continue  # Its own name - nothing to manage (until it's renamed)
        found[(kind, key)] = {"kind": kind, "name": header_value or spelled.get((kind, key)) or key,
                              "equipment_id": equipment_id, "saved": True, "used": used.get((kind, key), Counter())}
    for (kind, key), name in spelled.items():
        if (kind, key) in found or any(_key(item["name"]) == key for item in info.items(kind).values()):
            continue
        found[(kind, key)] = {"kind": kind, "name": name, "equipment_id": None, "saved": False,
                              "used": used[(kind, key)]}
    return sorted(found.values(), key=lambda m: (m["kind"] != TELESCOPE, m["name"].casefold()))


def keep_names_of(conn, kind, equipment_id):
    """The equipment is being deleted: images and sessions given it keep its
    name, no longer linked. Does not commit; {table: how many}."""
    changed = Counter()
    for (table, link_kind), (id_column, name_column) in LINKS.items():
        if link_kind == kind:
            changed[table] += conn.execute(
                f"UPDATE {table} SET {name_column} = COALESCE((SELECT name FROM {TABLES[kind]} WHERE id = ?), "
                f"{name_column}), {id_column} = NULL WHERE {id_column} = ?", (equipment_id, equipment_id)).rowcount
    return changed


def images_using(conn, kind, equipment_id):
    """How many images are linked to the equipment."""
    id_column, _name_column = COLUMNS[kind]
    return conn.execute(f"SELECT COUNT(*) FROM userimages WHERE {id_column} = ?", (equipment_id,)).fetchone()[0]


# ---------------------------------------------------------------------------
# Sessions: what their files say
# ---------------------------------------------------------------------------

def files_gear(conn, session_id, info):
    """What a session's files were taken with, through the user's matches:
    (telescope id or None, camera id or None, camera name or None) - their
    most common TELESCOP and INSTRUME."""
    def most_common(column):
        row = conn.execute(f"SELECT {column} FROM usersessionfiles WHERE session_id = ? "
                           f"AND TRIM(COALESCE({column}, '')) != '' GROUP BY {column} "
                           f"ORDER BY COUNT(*) DESC LIMIT 1", (session_id,)).fetchone()
        return row[0].strip() if row else None
    telescope, camera = most_common("telescope"), most_common("camera")
    return resolve(TELESCOPE, telescope, info), resolve(CAMERA, camera, info), camera


def fill_session_gear(conn, session_id, info=None, telescope=True):
    """Give a session that has no camera (or, with telescope, no telescope)
    the one its files were taken with, through the user's matches. Never
    changes one that's set. Does not commit."""
    info = info or LibraryScanner.load_equipment_info(conn)
    telescope_id, camera_id, camera = files_gear(conn, session_id, info)
    conn.execute("UPDATE usersessions SET "
                 "camera_id = CASE WHEN TRIM(COALESCE(camera, '')) = '' THEN ? ELSE camera_id END, "
                 "camera = CASE WHEN TRIM(COALESCE(camera, '')) = '' THEN ? ELSE camera END, "
                 "telescope_id = COALESCE(telescope_id, ?) WHERE id = ?",
                 (camera_id, camera, telescope_id if telescope else None, session_id))


# ---------------------------------------------------------------------------
# Suggestions - for the Equipment window's review
# ---------------------------------------------------------------------------

def suggestions(conn, info=None):
    """Ways to link more images and sessions to the user's equipment:
    [{"kind", "name" (None for a missing camera), "ids" ({table: [ids]}),
    "equipment_id", "reason"}].

    - A name images or sessions were given that isn't linked, when it's
      matched to or named like the user's equipment (as the Import Wizard
      suggests) - unless the user chose to leave it unmatched.
    - Images with a telescope but no camera, when one camera is all that's
      used with that telescope (linked to it in Equipment, or on its other
      images) - a smart telescope's built-in camera, say."""
    info = info or LibraryScanner.load_equipment_info(conn)
    by_name = {}
    for (table, kind), (id_column, name_column) in LINKS.items():
        for row_id, name in conn.execute(f"SELECT id, {name_column} FROM {table} WHERE {id_column} IS NULL "
                                         f"AND TRIM(COALESCE({name_column}, '')) != ''"):
            entry = by_name.setdefault((kind, _key(name)), {"kind": kind, "name": name.strip(),
                                                           "ids": defaultdict(list)})
            entry["ids"][table].append(row_id)
    found = []
    for entry in by_name.values():
        kind, name = entry["kind"], entry["name"]
        equipment_id = resolve(kind, name, info)
        if equipment_id is not None:
            reason = "Matched in the Import Wizard" if info.match(kind, name)[0] is not None else "Same name"
        else:
            equipment_id, _why = suggest(kind, name, info)
            if equipment_id is None:
                continue  # Nothing like it, or left unmatched
            reason = "Similar name"
        found.append(dict(entry, ids=dict(entry["ids"]), equipment_id=equipment_id, reason=reason))

    missing = defaultdict(list)
    for image_id, telescope_id in conn.execute(
            "SELECT id, telescope_id FROM userimages WHERE telescope_id IS NOT NULL AND camera_id IS NULL "
            "AND TRIM(COALESCE(camera_name, '')) = ''"):
        if telescope_id in info.telescopes:
            missing[telescope_id].append(image_id)
    for telescope_id, image_ids in missing.items():
        cameras = {row[0] for row in conn.execute(
            "SELECT te.equipment_id FROM telescope_equipment te JOIN userequipment e ON e.id = te.equipment_id "
            "WHERE te.telescope_id = ? AND e.equipment_type = 'camera'", (telescope_id,))}
        cameras |= {row[0] for row in conn.execute(
            "SELECT DISTINCT camera_id FROM userimages WHERE telescope_id = ? AND camera_id IS NOT NULL",
            (telescope_id,))}
        cameras &= set(info.cameras)
        # A camera that's only a name on its images (not in Equipment) is another one used with it
        unlisted = conn.execute("SELECT COUNT(*) FROM userimages WHERE telescope_id = ? AND camera_id IS NULL "
                                "AND TRIM(COALESCE(camera_name, '')) != ''", (telescope_id,)).fetchone()[0]
        if len(cameras) == 1 and not unlisted:
            found.append({"kind": CAMERA, "name": None, "ids": {IMAGES: image_ids}, "equipment_id": cameras.pop(),
                          "telescope_id": telescope_id,
                          "reason": f"The only camera used with the {info.telescopes[telescope_id]['name']}"})
    return found


def apply_suggestion(conn, suggestion, equipment_id):
    """Link a suggestion's images and sessions to equipment_id (the suggested
    equipment, or another the user chose). A name is saved as a match too,
    so files given it later link as well. Commits; {table: how many changed}."""
    if suggestion["name"] is None:
        changed = Counter({IMAGES: conn.executemany(
            "UPDATE userimages SET camera_id = ? WHERE id = ? AND camera_id IS NULL",
            [(equipment_id, i) for i in suggestion["ids"].get(IMAGES, [])]).rowcount})
        conn.commit()
        return changed
    info = LibraryScanner.load_equipment_info(conn)
    item = info.items(suggestion["kind"]).get(equipment_id)
    if item is not None and _key(item["name"]) == _key(suggestion["name"]):
        changed = link_name(conn, suggestion["kind"], suggestion["name"], equipment_id)  # Its own name
        conn.commit()
        return changed
    return save_matches(conn, {(suggestion["kind"], suggestion["name"]): equipment_id})


# ---------------------------------------------------------------------------
# Converting the old names
# ---------------------------------------------------------------------------

def migrate(conn, first_run=False, sessions_first_run=False):
    """Convert what's from before the equipment columns. On sessions' first
    run with camera_id, link their camera names, and give sessions with no
    telescope or camera what their files say. Then images still holding the
    old equipment text, taking their session's telescope or camera when the
    text has none - on images' first run, images with no equipment at all
    too, and collages made from an equipment suggestion follow its new key.
    Commits; how many images changed."""
    rows = conn.execute("SELECT id, equipment, session_id FROM userimages "
                        "WHERE TRIM(COALESCE(equipment, '')) != ''").fetchall()
    if first_run:
        rows += conn.execute(
            "SELECT id, '', session_id FROM userimages WHERE session_id IS NOT NULL "
            "AND TRIM(COALESCE(equipment, '')) = '' AND telescope_id IS NULL AND camera_id IS NULL "
            "AND TRIM(COALESCE(telescope_name, '')) = '' AND TRIM(COALESCE(camera_name, '')) = ''").fetchall()
    if not rows and not first_run and not sessions_first_run:
        return 0
    info = LibraryScanner.load_equipment_info(conn)
    if sessions_first_run:
        _migrate_sessions(conn, info)

    sessions = {}
    try:
        for session_id, telescope_id, camera_id, camera in conn.execute(
                "SELECT id, telescope_id, camera_id, camera FROM usersessions"):
            sessions[session_id] = (telescope_id if telescope_id in info.telescopes else None,
                                    camera_id if camera_id in info.cameras else None, _clean(camera))
    except sqlite3.OperationalError:
        pass  # No sessions table yet
    changed = 0
    for image_id, text, session_id in rows:
        telescope_id, telescope_name, camera_id, camera_name = parse_equipment_text(text, info)
        session_telescope, session_camera_id, session_camera = sessions.get(session_id, (None, None, None))
        if telescope_id is None and not telescope_name and session_telescope is not None:
            telescope_id, telescope_name = session_telescope, info.telescopes[session_telescope]["name"]
        if camera_id is None and not camera_name and session_camera:
            camera_id, camera_name = session_camera_id or resolve(CAMERA, session_camera, info), session_camera
        if not (text or telescope_id or telescope_name or camera_id or camera_name):
            continue  # Nothing to record (a session that doesn't know either)
        conn.execute("UPDATE userimages SET telescope_id = ?, telescope_name = ?, camera_id = ?, camera_name = ?, "
                     "equipment = NULL WHERE id = ?",
                     (telescope_id, telescope_name, camera_id, camera_name, image_id))
        changed += 1
    if first_run:
        _migrate_collage_rules(conn, info)
    conn.commit()
    if changed:
        logger.info(f"Linked the equipment of {changed} gallery image(s) to the Equipment list")
    return changed


def _migrate_sessions(conn, info):
    """Sessions' camera names linked to the user's equipment, and sessions
    with no telescope or camera given what their files say."""
    try:
        rows = conn.execute("SELECT id, camera FROM usersessions").fetchall()
    except sqlite3.OperationalError:
        return  # No sessions table yet
    linked = 0
    for session_id, camera in rows:
        camera_id = resolve(CAMERA, camera, info)
        if camera_id is not None:
            conn.execute("UPDATE usersessions SET camera_id = ? WHERE id = ?", (camera_id, session_id))
            linked += 1
        try:
            fill_session_gear(conn, session_id, info)
        except sqlite3.OperationalError:
            pass  # No session files table yet
    if linked:
        logger.info(f"Linked the camera of {linked} session(s) to the Equipment list")


def _migrate_collage_rules(conn, info):
    """Collages made from an equipment suggestion stored its text as the
    rule's key ("equipment:ZenithStar 61, ASI2600MC Pro"): its equipment_key now."""
    try:
        rows = conn.execute("SELECT id, settings FROM usercollages WHERE settings LIKE '%equipment:%'").fetchall()
    except sqlite3.OperationalError:
        return  # No collages table yet
    for collage_id, settings in rows:
        try:
            values = json.loads(settings)
        except (TypeError, ValueError):
            continue
        rule = values.get("rule") if isinstance(values, dict) else None
        if not isinstance(rule, dict) or not str(rule.get("key", "")).startswith("equipment:"):
            continue
        text = rule["key"][len("equipment:"):]
        rule["key"] = "equipment:" + equipment_key(*parse_equipment_text(text, info))
        conn.execute("UPDATE usercollages SET settings = ? WHERE id = ?", (json.dumps(values), collage_id))
