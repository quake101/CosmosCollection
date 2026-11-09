"""Collection commands: dso, images, sessions, targets, stats, sites.

Read the same database as the app; the writes (images add, targets add/remove)
make the same changes the app's dialogs do."""

import os

from cosmoscollection.cli.Output import (EXIT_USAGE, CliError, Column, note, print_fields, print_json,
                                         print_table)
from cosmoscollection.core.Formatting import count_text, format_duration, format_magnitude, known_magnitude

REFRESH_NOTE = "If the app is open, its windows show this change after they're refreshed or reopened."


# ================================================================ registration

def register(subparsers, common):
    """Add the commands; {group name: its actions' subparsers}, for other
    modules to add actions to ('sessions check')."""
    groups = {
        "dso": _register_dso(subparsers, common),
        "images": _register_images(subparsers, common),
        "sessions": _register_sessions(subparsers, common),
        "targets": _register_targets(subparsers, common),
    }

    stats = subparsers.add_parser("stats", parents=[common], help="totals for your collection")
    stats.set_defaults(func=run_stats)

    sites = subparsers.add_parser("sites", parents=[common], help="your saved observing locations")
    sites.set_defaults(func=run_sites)
    return groups


def group(subparsers, common, name, help_text):
    """A command group ("dso") whose actions are subcommands ("dso search");
    the group alone prints its help."""
    parser = subparsers.add_parser(name, parents=[common], help=help_text, description=help_text)
    actions = parser.add_subparsers(dest="action", metavar="<action>")

    def show_help(ctx):
        parser.print_help()
        return EXIT_USAGE

    parser.set_defaults(func=show_help)
    return actions


def _register_dso(subparsers, common):
    actions = group(subparsers, common, "dso", "search and look up catalog objects")

    search = actions.add_parser("search", parents=[common], help="find objects by designation, name or position",
                                description="Find objects the way the main window's search box does: "
                                            "designations ('M31', 'sh2-129'), common names ('Andromeda') "
                                            "or part of an RA/Dec ('00h42m').")
    search.add_argument("text", nargs="+", help="what to search for")
    _add_dso_filters(search)
    search.add_argument("--limit", type=int, default=50, help="most results to show (default 50, 0 = all)")
    search.set_defaults(func=run_dso_search)

    listing = actions.add_parser("list", parents=[common], help="list catalog objects (with filters)")
    _add_dso_filters(listing)
    images = listing.add_mutually_exclusive_group()
    images.add_argument("--imaged", action="store_true", help="only objects with images in your gallery")
    images.add_argument("--not-imaged", action="store_true", help="only objects without images")
    listing.add_argument("--sort", choices=("name", "magnitude", "images", "ra", "dec"), default="name")
    listing.add_argument("--limit", type=int, default=0, help="most objects to show (default: all)")
    listing.set_defaults(func=run_dso_list)

    show = actions.add_parser("show", parents=[common], help="details of one object, with your images and sessions")
    show.add_argument("name", nargs="+", help="designation or common name, e.g. M31, 'NGC 7000', 'Orion Nebula'")
    show.set_defaults(func=run_dso_show)
    return actions


def _add_dso_filters(parser):
    parser.add_argument("--catalog", help="only this catalog's objects (M, NGC, IC, Sh2, ...)")
    parser.add_argument("--type", help="only this type: a code (GALXY) or name ('galaxy', 'open cluster')")
    parser.add_argument("--constellation", help="only this constellation: abbreviation (ORI) or name")


def _register_images(subparsers, common):
    actions = group(subparsers, common, "images", "your gallery images")

    listing = actions.add_parser("list", parents=[common], help="list gallery images")
    listing.add_argument("--dso", metavar="NAME", help="only this object's images")
    listing.add_argument("--session", type=int, metavar="ID", help="only images linked to this session")
    listing.add_argument("--limit", type=int, default=0, help="most images to show (default: all)")
    listing.set_defaults(func=run_images_list)

    add = actions.add_parser("add", parents=[common], help="add an image to an object's gallery")
    add.add_argument("dso", metavar="DSO", help="the object, e.g. M31 or 'NGC 7000'")
    add.add_argument("image", metavar="PATH", help="the image file")
    add.add_argument("--equipment", default="", help='telescope and camera: "Telescope: AT72ED, Camera: ASI2600"')
    add.add_argument("--integration", default="",
                     help="integration time: seconds (3600) or as the app writes it ('5h 6m')")
    add.add_argument("--date", default="", help="date taken, YYYY-MM-DD")
    add.add_argument("--notes", default="", help="notes about the image")
    add.add_argument("--favorite", action="store_true", help="make it the object's favorite image")
    add.add_argument("--force", action="store_true", help="add it even if the object already has this file")
    add.set_defaults(func=run_images_add)
    return actions


def _register_sessions(subparsers, common):
    actions = group(subparsers, common, "sessions", "your imaging sessions")

    listing = actions.add_parser("list", parents=[common], help="list sessions, newest first")
    listing.add_argument("--dso", metavar="NAME", help="only sessions of this object")
    listing.add_argument("--status", help="only sessions with this status (Planned, Completed, ...)")
    listing.add_argument("--since", metavar="YYYY-MM-DD", help="only sessions on or after this date")
    listing.add_argument("--until", metavar="YYYY-MM-DD", help="only sessions on or before this date")
    listing.add_argument("--telescope", metavar="TEXT", help="only sessions whose telescope name contains TEXT")
    listing.add_argument("--camera", metavar="TEXT", help="only sessions whose camera name contains TEXT")
    listing.add_argument("--limit", type=int, default=0, help="most sessions to show (default: all)")
    listing.set_defaults(func=run_sessions_list)

    show = actions.add_parser("show", parents=[common], help="one session's nights and per-filter totals")
    show.add_argument("id", type=int, help="the session's id (from 'sessions list')")
    show.set_defaults(func=run_sessions_show)
    return actions


def _register_targets(subparsers, common):
    from cosmoscollection.planning.TargetRows import TARGET_PRIORITIES, TARGET_STATUSES

    actions = group(subparsers, common, "targets", "your Target List")

    listing = actions.add_parser("list", parents=[common], help="list your targets")
    listing.add_argument("--status", choices=TARGET_STATUSES, help="only targets with this status")
    listing.add_argument("--priority", choices=TARGET_PRIORITIES, help="only targets with this priority")
    listing.set_defaults(func=run_targets_list)

    add = actions.add_parser("add", parents=[common], help="add a catalog object to your Target List")
    add.add_argument("name", nargs="+", help="the object, e.g. M31 or 'Orion Nebula'")
    add.add_argument("--priority", choices=TARGET_PRIORITIES, default="Medium")
    add.add_argument("--status", choices=TARGET_STATUSES, default="Not Observed")
    add.add_argument("--notes", default="")
    add.add_argument("--telescope", metavar="NAME", help="the telescope for it (from your Equipment)")
    add.add_argument("--force", action="store_true", help="add it even if it's already on the list")
    add.set_defaults(func=run_targets_add)

    remove = actions.add_parser("remove", parents=[common], help="remove a target (its sessions are kept)")
    remove.add_argument("name", nargs="+", help="the target's name or id")
    remove.set_defaults(func=run_targets_remove)
    return actions


# ================================================================ shared lookups

def _load_dsos(conn):
    """Every catalog object as the main window loads it, and the catalog names."""
    from cosmoscollection.app.DSOData import load_all_dso_data
    return load_all_dso_data(conn)


def _load_dso(conn, dsodetailid):
    from cosmoscollection.app.DSOData import load_dso
    return load_dso(conn, dsodetailid)


def _label(dso):
    """'M 31 (Andromeda Galaxy)'."""
    from cosmoscollection.catalog.DSONames import common_name_from_designations, with_common_name
    return with_common_name(dso["name"], common_name_from_designations(dso["designations"]))


def resolve_dso(conn, name):
    """The one catalog object a name means, as the main window's dict; a
    CliError listing the candidates when there are none or several."""
    from cosmoscollection.app.DSOData import SEARCH_RANK_EXACT, DSOSearch
    from cosmoscollection.catalog.DSONames import catalog_ids

    ids = sorted(catalog_ids(conn, name))
    if len(ids) == 1:
        dso = _load_dso(conn, ids[0])
        if dso:
            return dso
    if len(ids) > 1:
        labels = [_label(d) for d in (_load_dso(conn, i) for i in ids) if d]
        raise CliError(f"'{name}' could be {len(labels)} objects: {', '.join(labels)}.",
                       hint="Use one of those designations.")

    search = DSOSearch()
    matches = search.filter(_load_dsos(conn)[0], name)
    exact = {m["dsodetailid"]: m for m in matches if m.get("search_rank") == SEARCH_RANK_EXACT}
    if len(exact) == 1:
        return next(iter(exact.values()))
    matches.sort(key=lambda m: (m.get("search_rank", 0), search.sort_key()(m)))
    suggestions = ", ".join(_label(m) for m in matches[:6])
    raise CliError(f"No object named '{name}' in the catalog.",
                   hint=f"Did you mean: {suggestions}?" if suggestions else "Try 'dso search' with part of the name.")


def _dso_json(dso):
    from cosmoscollection.core.Formatting import format_dec, format_ra
    from cosmoscollection.catalog.DSONames import (common_name_from_designations, constellation_name,
                                                   type_name)
    return {
        "id": dso["dsodetailid"],
        "name": dso["name"],
        "common_name": common_name_from_designations(dso["designations"]),
        "designations": [d.strip() for d in (dso["designations"] or "").split(",") if d.strip()],
        "type": dso["dso_type"], "type_name": type_name(dso["dso_type"]), "class": dso["dso_class"],
        "constellation": (dso["constellation"] or "").upper() or None,
        "constellation_name": constellation_name(dso["constellation"]),
        "ra_deg": dso["ra_deg"], "dec_deg": dso["dec_deg"],
        "ra": format_ra(dso["ra_deg"]), "dec": format_dec(dso["dec_deg"]),
        "magnitude": known_magnitude(dso["magnitude"]), "surface_brightness": known_magnitude(dso["surface_brightness"]),
        "size_arcmin": [round(dso["size_max"], 2), round(dso["size_min"], 2)],
        "image_count": dso["image_count"],
    }


def _apply_dso_filters(args, dsos, catalogs, text=""):
    """The objects matching the search text and --catalog/--type/--constellation
    (and --imaged/--not-imaged), the DSOSearch that matched them and the catalog."""
    from cosmoscollection.app.DSOData import DSOSearch
    from cosmoscollection.catalog.DSONames import CONSTELLATIONS, type_code

    code = None
    if args.type:
        code = type_code(args.type)
        if code is None:
            from cosmoscollection.catalog.DSONames import DSO_TYPES
            raise CliError(f"Unknown object type '{args.type}'.",
                           EXIT_USAGE, hint="Types: " + ", ".join(f"{c} ({n})" for c, n in DSO_TYPES))
    catalog = None
    if args.catalog:
        # As spelled in the catalog ("Sh2"), whatever case was typed
        catalog = next((c for c in catalogs if c.lower() == args.catalog.strip().lower()), None)
        if catalog is None:
            raise CliError(f"Unknown catalog '{args.catalog}'.", EXIT_USAGE,
                           hint="Catalogs: " + ", ".join(c for c in catalogs if c != "NAME"))
    search = DSOSearch()
    matches = search.filter(dsos, text, catalog, getattr(args, "imaged", False), code,
                            getattr(args, "not_imaged", False))
    if args.constellation:
        wanted = args.constellation.strip().lower()
        abbreviation = next((a for a, n in CONSTELLATIONS.items() if wanted in (a.lower(), n.lower())), None)
        if abbreviation is None:
            raise CliError(f"Unknown constellation '{args.constellation}'.", EXIT_USAGE,
                           hint="Use its 3-letter abbreviation (ORI) or name (Orion).")
        matches = [m for m in matches if (m["constellation"] or "").upper() == abbreviation]
    return matches, search, catalog


def _dso_columns(search, catalog):
    from cosmoscollection.catalog.DSONames import common_name_from_designations, type_name

    def shown_name(dso):
        catalogue, designation = search.display_designation(dso, catalog)
        if catalogue == "NAME":  # Matched by a common name: show the object's designation
            return dso["name"].strip() if not dso["name"].startswith("NAME ") else dso["name"][5:]
        return " ".join(part for part in (catalogue, designation) if part)

    return [
        Column("Name", shown_name),
        Column("Common name", lambda d: common_name_from_designations(d["designations"]) or "", flex=True),
        Column("Type", lambda d: type_name(d["dso_type"])),
        Column("Const", lambda d: (d["constellation"] or "").upper()),
        Column("Mag", lambda d: format_magnitude(d["magnitude"]), ">"),
        Column("Images", lambda d: d["image_count"], ">"),
    ]


def _print_dsos(ctx, matches, search, catalog, limit, what):
    shown = matches[:limit] if limit and limit > 0 else matches
    if ctx.json:
        print_json([dict(_dso_json(d), matched=d.get("matched_designation")) for d in shown])
        return
    if not shown:
        print(f"No {what}.")
        return
    print_table(shown, _dso_columns(search, catalog))
    more = len(matches) - len(shown)
    print(f"\n{count_text(len(matches), 'object')}" + (f" ({more} more not shown - use --limit 0 for all)" if more else ""))


# ================================================================ dso

def run_dso_search(ctx):
    text = " ".join(ctx.args.text)
    with ctx.connection() as conn:
        dsos, catalogs = _load_dsos(conn)
    matches, search, catalog = _apply_dso_filters(ctx.args, dsos, catalogs, text)
    key = search.sort_key(catalog)
    matches.sort(key=lambda m: (m.get("search_rank", 0), key(m)))
    _print_dsos(ctx, matches, search, catalog, ctx.args.limit, f"objects match '{text}'")


def run_dso_list(ctx):
    with ctx.connection() as conn:
        dsos, catalogs = _load_dsos(conn)
    matches, search, catalog = _apply_dso_filters(ctx.args, dsos, catalogs)
    sort = ctx.args.sort
    if sort == "name":
        matches.sort(key=search.sort_key(catalog))
    elif sort == "magnitude":
        matches.sort(key=lambda d: (known_magnitude(d["magnitude"]) is None, known_magnitude(d["magnitude"]) or 0))
    elif sort == "images":
        matches.sort(key=lambda d: -d["image_count"])
    else:
        field = "ra_deg" if sort == "ra" else "dec_deg"
        matches.sort(key=lambda d: d[field] if d[field] is not None else float("-inf"))
    _print_dsos(ctx, matches, search, catalog, ctx.args.limit, "objects match those filters")


def _sessions_for_dso(conn, dso):
    """The object's sessions, as the app finds them (SessionRows.sessions_for_object)."""
    from cosmoscollection.sessions import SessionRows
    ids = {row[0] for row in SessionRows.sessions_for_object(conn, dso["name"], dso["ra_deg"], dso["dec_deg"],
                                                             dso["dsodetailid"])}
    return [s for s in SessionRows.load_sessions(conn) if s["id"] in ids]


def run_dso_show(ctx):
    from cosmoscollection.planning.TargetRows import find_existing_target

    name = " ".join(ctx.args.name)
    with ctx.connection() as conn:
        dso = resolve_dso(conn, name)
        images = _load_images(conn, dsodetailid=dso["dsodetailid"])
        sessions = _sessions_for_dso(conn, dso)
        target = find_existing_target(conn, dso["name"], dso["ra_deg"], dso["dec_deg"])
        if target:
            target = next((t for t in _target_rows(conn) if t["id"] == target["id"]), target)

    data = _dso_json(dso)
    total_seconds = sum(s["integration_seconds"] or 0 for s in sessions)
    if ctx.json:
        data.update(images=images, sessions=sessions, integration_seconds=total_seconds,
                    target=_target_json(target) if target else None)
        print_json(data)
        return

    print(_label(dso))
    print()
    size = f"{data['size_arcmin'][0]:g}' x {data['size_arcmin'][1]:g}'" if any(data["size_arcmin"]) else None
    print_fields([
        ("Designations", ", ".join(d for d in data["designations"] if not d.startswith("NAME "))),
        ("Type", data["type_name"] + (f" ({data['class']})" if data["class"] else "")),
        ("Constellation", data["constellation_name"]),
        ("RA / Dec", f"{data['ra']}  {data['dec']}  ({data['ra_deg']:.4f}, {data['dec_deg']:+.4f})"
         if data["ra_deg"] is not None else None),
        ("Magnitude", format_magnitude(data["magnitude"]) or None),
        ("Surface brightness", format_magnitude(data["surface_brightness"]) or None),
        ("Size", size),
        ("Target List", f"{target['priority']} priority, {target['status']} (as '{target['name']}', "
                        f"id {target['id']})" if target else "not on it"),
    ])
    print()
    if sessions:
        print(f"Sessions: {count_text(len(sessions), 'session')}, {format_duration(total_seconds)} total")
        print_table(sessions, _session_columns(), indent="  ")
    else:
        print("Sessions: none")
    print()
    if images:
        print(f"Images: {len(images)}")
        print_table(images, _image_columns(with_object=False), indent="  ")
    else:
        print("Images: none in your gallery")


# ================================================================ images

def _load_images(conn, dsodetailid=None, session_id=None):
    sql = """SELECT id, dsodetailid, image_path, integration_time, date_taken, notes, is_favorite, session_id,
                    telescope_name, camera_name, created_date FROM userimages"""
    where, params = [], []
    if dsodetailid is not None:
        where.append("dsodetailid = ?")
        params.append(dsodetailid)
    if session_id is not None:
        where.append("session_id = ?")
        params.append(session_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(NULLIF(date_taken, ''), created_date, '') DESC, id DESC"
    keys = ("id", "dsodetailid", "path", "integration", "date_taken", "notes", "favorite", "session_id",
            "telescope", "camera", "added")
    images = [dict(zip(keys, tuple(row))) for row in conn.execute(sql, params).fetchall()]
    from cosmoscollection.catalog.DSONames import object_labels
    labels = object_labels(conn, [i["dsodetailid"] for i in images])
    for image in images:
        image["favorite"] = bool(image["favorite"])
        image["object"] = labels.get(str(image["dsodetailid"]), image["dsodetailid"])
    return images


def _image_columns(with_object=True):
    columns = [Column("ID", lambda i: i["id"], ">")]
    if with_object:
        columns.append(Column("Object", lambda i: i["object"], flex=True))
    columns += [
        Column("Date", lambda i: i["date_taken"] or ""),
        Column("Integration", lambda i: i["integration"] or "", ">"),
        Column("Fav", lambda i: "*" if i["favorite"] else ""),
        Column("Telescope", lambda i: i["telescope"] or "", flex=True),
        Column("File", lambda i: i["path"] or "", flex=True),
    ]
    return columns


def run_images_list(ctx):
    with ctx.connection() as conn:
        dsodetailid = resolve_dso(conn, ctx.args.dso)["dsodetailid"] if ctx.args.dso else None
        images = _load_images(conn, dsodetailid, ctx.args.session)
    shown = images[:ctx.args.limit] if ctx.args.limit > 0 else images
    if ctx.json:
        print_json(shown)
        return
    if not shown:
        print("No images.")
        return
    print_table(shown, _image_columns())
    print(f"\n{count_text(len(images), 'image')}")


def _integration_text(value):
    """The gallery's integration text: seconds (3600) become '1h', text ('5h 6m') is kept."""
    value = (value or "").strip()
    try:
        seconds = float(value)
    except ValueError:
        return value
    if seconds <= 0:
        raise CliError("--integration must be more than 0 seconds.", EXIT_USAGE)
    return format_duration(seconds)


def run_images_add(ctx):
    from datetime import datetime
    from cosmoscollection.core.DatabaseManager import add_user_image_row, set_favorite_image
    from cosmoscollection.equipment import EquipmentMapper
    from cosmoscollection.sessions import LibraryScanner

    args = ctx.args
    # Kept as typed (made absolute): resolve() would turn a mapped drive into a UNC path
    image_path = os.path.abspath(os.path.expanduser(args.image))
    if not os.path.isfile(image_path):
        raise CliError(f"Image file not found: {image_path}")
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            raise CliError(f"--date must be YYYY-MM-DD, not '{args.date}'.", EXIT_USAGE)
    integration = _integration_text(args.integration)

    with ctx.connection() as conn:
        dso = resolve_dso(conn, args.dso)
        existing = conn.execute("SELECT id FROM userimages WHERE dsodetailid = ? AND image_path = ?",
                                (dso["dsodetailid"], image_path)).fetchone()
        if existing and not args.force:
            raise CliError(f"{_label(dso)} already has this image (image id {existing[0]}).",
                           hint="Pass --force to add it again.")
        telescope_id, telescope_name, camera_id, camera_name = EquipmentMapper.parse_equipment_text(
            args.equipment, LibraryScanner.load_equipment_info(conn))
        cursor = conn.cursor()
        try:
            image_id = add_user_image_row(cursor, dso["dsodetailid"], image_path, integration, args.date, args.notes,
                                          equipment={"telescope_id": telescope_id, "telescope_name": telescope_name,
                                                     "camera_id": camera_id, "camera_name": camera_name})
            if args.favorite:
                set_favorite_image(cursor, image_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    result = {"id": image_id, "dsodetailid": dso["dsodetailid"], "object": _label(dso), "path": image_path,
              "integration": integration, "date_taken": args.date, "notes": args.notes, "favorite": args.favorite,
              "telescope": telescope_name, "camera": camera_name}
    if ctx.json:
        print_json(result)
    else:
        print(f"Added image {image_id} to {_label(dso)}" + (" as its favorite" if args.favorite else ""))
        print_fields([("  File", image_path), ("  Integration", integration), ("  Date", args.date),
                      ("  Telescope", telescope_name), ("  Camera", camera_name)])
        note(REFRESH_NOTE)


# ================================================================ sessions

def _load_sessions(conn, session_id=None):
    from cosmoscollection.sessions import SessionRows
    return SessionRows.load_sessions(conn, session_id)


def _session_columns(with_name=True):
    columns = [Column("ID", lambda s: s["id"], ">")]
    if with_name:
        columns.append(Column("Object", lambda s: s["dso_name"], flex=True))
    columns += [
        Column("Date", lambda s: s["session_date"] or ""),
        Column("Status", lambda s: s["status"] or ""),
        Column("Nights", lambda s: s["nights"], ">"),
        Column("Subs", lambda s: s["sub_count"] or 0, ">"),
        Column("Integration", lambda s: format_duration(s["integration_seconds"]) if s["integration_seconds"] else "",
               ">"),
        Column("Filters", lambda s: s["filters_used"] or "", flex=True),
        Column("Telescope", lambda s: s["telescope_name"] or "", flex=True),
    ]
    return columns


def run_sessions_list(ctx):
    args = ctx.args
    with ctx.connection() as conn:
        if args.dso:
            sessions = _sessions_for_dso(conn, resolve_dso(conn, args.dso))
        else:
            sessions = _load_sessions(conn)
    if args.status:
        sessions = [s for s in sessions if (s["status"] or "").lower() == args.status.lower()]
    if args.since:
        sessions = [s for s in sessions if (s["session_date"] or "") >= args.since]
    if args.until:
        sessions = [s for s in sessions if (s["session_date"] or "") <= args.until]
    if args.telescope:
        sessions = [s for s in sessions if args.telescope.lower() in (s["telescope_name"] or "").lower()]
    if args.camera:
        sessions = [s for s in sessions if args.camera.lower() in (s["camera"] or "").lower()]
    shown = sessions[:args.limit] if args.limit > 0 else sessions
    if ctx.json:
        print_json(shown)
        return
    if not shown:
        print("No sessions.")
        return
    print_table(shown, _session_columns())
    total = sum(s["integration_seconds"] or 0 for s in sessions)
    print(f"\n{count_text(len(sessions), 'session')}, {format_duration(total)} total")


def run_sessions_show(ctx):
    from cosmoscollection.sessions import SessionObservations as so

    with ctx.connection() as conn:
        rows = _load_sessions(conn, ctx.args.id)
        if not rows:
            raise CliError(f"No session with id {ctx.args.id}.", hint="See 'sessions list' for the ids.")
        session = rows[0]
        observations = so.load_observations(conn, session["id"])
        breakdowns = so.load_breakdowns(conn, [o["id"] for o in observations])
        images = _load_images(conn, session_id=session["id"])
        loose = [f for f in so.load_session_files(conn, session["id"])
                 if f["observation_id"] is None and f["frame_type"] == "Light"]

    nights = []
    for obs in observations:
        breakdown = breakdowns.get(obs["id"], [])
        subs, seconds, filters = so.summarize_breakdown(breakdown)
        nights.append({"id": obs["id"], "night": obs["night_date"],
                       "span": so.format_night_span(obs["night_date"], obs["start_datetime"], obs["end_datetime"]),
                       "subs": subs, "seconds": seconds, "filters": filters, "files": obs["file_count"],
                       "notes": obs["notes"], "breakdown": breakdown})
    # As the Session Manager counts them: lights not in a night yet too
    filters = so.filter_totals([(o["night_date"], breakdowns.get(o["id"], [])) for o in observations], loose,
                               by_exposure=True)

    if ctx.json:
        print_json(dict(session, nights_detail=nights, filters_detail=filters, images=images))
        return

    print(f"Session {session['id']}: {session['dso_name']}")
    print()
    print_fields([
        ("Status", session["status"]), ("Date", session["session_date"]),
        ("Telescope", session["telescope_name"]), ("Camera", session["camera"]), ("Location", session["location_name"]),
        ("Subs", f"{session['sub_count'] or 0} lights, {format_duration(session['integration_seconds'])}"),
        ("Notes", (session["notes"] or "").strip() or None),
    ])
    print()
    if nights:
        print(f"Nights ({len(nights)}):")
        print_table(nights, [
            Column("Night", lambda n: n["span"]),
            Column("Subs", lambda n: n["subs"], ">"),
            Column("Integration", lambda n: format_duration(n["seconds"]) if n["seconds"] else "", ">"),
            Column("Filters", lambda n: ", ".join(n["filters"]), flex=True),
        ], indent="  ")
        print()
    if filters:
        print("By filter:")
        print_table(filters, [
            Column("Filter", lambda f: f["filter"]),
            Column("Exposure", lambda f: format_duration(f["exposure_seconds"]) if f["exposure_seconds"] else "", ">"),
            Column("Subs", lambda f: f["subs"], ">"),
            Column("Integration", lambda f: format_duration(f["seconds"]), ">"),
        ], indent="  ")
        print()
    if images:
        print(f"Gallery images ({len(images)}):")
        print_table(images, _image_columns(), indent="  ")


# ================================================================ targets

def _target_rows(conn):
    from cosmoscollection.planning.TargetRows import load_target_rows
    return load_target_rows(conn)


def _target_json(target):
    data = {key: target.get(key) for key in ("id", "name", "dso_type", "constellation", "ra_deg", "dec_deg",
                                              "magnitude", "size_info", "priority", "status", "best_months",
                                              "notes", "date_added", "date_observed", "telescope_id",
                                              "session_count")}
    data["magnitude"] = known_magnitude(data["magnitude"])  # None when unknown
    return data


def run_targets_list(ctx):
    from cosmoscollection.catalog.DSONames import type_name
    from cosmoscollection.planning.TargetRows import TARGET_PRIORITIES

    with ctx.connection() as conn:
        targets = _target_rows(conn)
    if ctx.args.status:
        targets = [t for t in targets if t["status"] == ctx.args.status]
    if ctx.args.priority:
        targets = [t for t in targets if t["priority"] == ctx.args.priority]
    # Most important first, then oldest
    rank = {p: i for i, p in enumerate(reversed(TARGET_PRIORITIES))}
    targets.sort(key=lambda t: (rank.get(t["priority"], len(rank)), t["id"]))
    if ctx.json:
        print_json([_target_json(t) for t in targets])
        return
    if not targets:
        print("No targets.")
        return
    print_table(targets, [
        Column("ID", lambda t: t["id"], ">"),
        Column("Name", lambda t: t["name"], flex=True),
        Column("Type", lambda t: type_name(t["dso_type"])),
        Column("Const", lambda t: t["constellation"] or ""),
        Column("Priority", lambda t: t["priority"] or ""),
        Column("Status", lambda t: t["status"] or ""),
        Column("Sessions", lambda t: t["session_count"], ">"),
        Column("Added", lambda t: (t["date_added"] or "")[:10]),
    ])
    print(f"\n{count_text(len(targets), 'target')}")


def _telescope_id(conn, name):
    if not name:
        return None
    rows = conn.execute("SELECT id, name FROM usertelescopes").fetchall()
    exact = [r[0] for r in rows if (r[1] or "").strip().lower() == name.strip().lower()]
    partial = [r[0] for r in rows if name.strip().lower() in (r[1] or "").lower()]
    found = exact or partial
    if len(found) != 1:
        names = ", ".join(r[1] for r in rows) or "none"
        raise CliError(f"{'No' if not found else 'More than one'} telescope matches '{name}'.",
                       hint=f"Your telescopes: {names}")
    return found[0]


def run_targets_add(ctx):
    from cosmoscollection.planning.TargetRows import add_target, find_existing_target

    args = ctx.args
    with ctx.connection() as conn:
        dso = resolve_dso(conn, " ".join(args.name))
        name = _label(dso)  # New entries are named "M 31 (Andromeda Galaxy)", as in the app
        existing = find_existing_target(conn, name, dso["ra_deg"], dso["dec_deg"])
        if existing and not args.force:
            raise CliError(f"{name} is already on your Target List as '{existing['name']}' (id {existing['id']}).",
                           hint="Pass --force to add it again.")
        target_id = add_target(conn, name, dso, args.priority, args.status, _telescope_id(conn, args.telescope),
                               args.notes)
    if ctx.json:
        print_json({"id": target_id, "name": name, "priority": args.priority, "status": args.status})
    else:
        print(f"Added {name} to your Target List (id {target_id}, {args.priority} priority).")
        note(REFRESH_NOTE)


def run_targets_remove(ctx):
    from cosmoscollection.cli.Prompts import confirm
    from cosmoscollection.planning.TargetRows import delete_target, find_existing_target

    text = " ".join(ctx.args.name)
    with ctx.connection() as conn:
        targets = {t["id"]: t for t in _target_rows(conn)}
        target = targets.get(int(text)) if text.isdigit() else None
        if target is None:
            match = find_existing_target(conn, text)
            target = targets.get(match["id"]) if match else None
        if target is None:
            raise CliError(f"'{text}' isn't on your Target List.", hint="See 'targets list' for names and ids.")
        sessions = target["session_count"]
        question = f"Remove '{target['name']}' (id {target['id']}) from your Target List?"
        if sessions:
            question += f" Its {count_text(sessions, 'session')} will be kept, no longer linked to it."
        if not confirm(ctx, question):
            print("Nothing removed.")
            return EXIT_USAGE
        delete_target(conn, target["id"])
    if ctx.json:
        print_json({"removed": target["id"], "name": target["name"], "sessions_unlinked": sessions})
    else:
        print(f"Removed '{target['name']}' from your Target List.")
        note(REFRESH_NOTE)


# ================================================================ stats, sites

def run_stats(ctx):
    from cosmoscollection.sessions import SessionObservations as so

    with ctx.connection() as conn:
        images, objects_imaged = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT dsodetailid) FROM userimages").fetchone()
        sessions = conn.execute("SELECT COUNT(*) FROM usersessions").fetchone()[0]
        statuses = dict(conn.execute("SELECT status, COUNT(*) FROM usertargetlist GROUP BY status").fetchall())
        observations = conn.execute("SELECT id, night_date FROM usersessionobservations").fetchall()
        breakdowns = so.load_breakdowns(conn, [row[0] for row in observations])

    by_year, total_seconds, total_subs, nights = {}, 0.0, 0, 0
    for obs_id, night in observations:
        subs, seconds, _ = so.summarize_breakdown(breakdowns.get(obs_id, []))
        if subs:
            nights += 1
        total_subs += subs
        total_seconds += seconds
        year = str(night or "")[:4] or "?"
        by_year[year] = by_year.get(year, 0.0) + seconds
    by_filter = so.filter_totals([(night, breakdowns.get(obs_id, [])) for obs_id, night in observations])

    data = {"objects_imaged": objects_imaged, "images": images, "sessions": sessions, "nights": nights,
            "subs": total_subs, "integration_seconds": total_seconds,
            "integration_by_year": dict(sorted(by_year.items())),
            "integration_by_filter": {f["filter"]: f["seconds"] for f in
                                      sorted(by_filter, key=lambda f: -f["seconds"])},
            "targets_by_status": statuses}
    if ctx.json:
        print_json(data)
        return
    print_fields([
        ("Objects imaged", objects_imaged), ("Gallery images", images), ("Sessions", sessions),
        ("Nights with subs", nights), ("Light subs", total_subs),
        ("Total integration", format_duration(total_seconds)),
        ("Targets", ", ".join(f"{count} {status.lower()}" for status, count in statuses.items()) or "none"),
    ])
    if by_year:
        print("\nIntegration by year:")
        print_fields([(year, format_duration(seconds)) for year, seconds in sorted(by_year.items()) if seconds],
                     indent="  ")
    if by_filter:
        print("\nIntegration by filter:")
        print_fields([(name, format_duration(seconds)) for name, seconds in data["integration_by_filter"].items()],
                     indent="  ")


def run_sites(ctx):
    from cosmoscollection.planning import Observer

    with ctx.connection() as conn:
        sites = Observer.load_sites(conn)
    if ctx.json:
        print_json([s.to_dict() for s in sites])
        return
    if not sites:
        print("No saved locations. Add one in the app (Settings > Location).")
        return
    print_table(sites, [
        Column("", lambda s: "*" if s.is_active else ""),
        Column("Name", lambda s: s.name, flex=True),
        Column("Latitude", lambda s: f"{s.lat:.4f}", ">"),
        Column("Longitude", lambda s: f"{s.lon:.4f}", ">"),
        Column("Time zone", lambda s: s.timezone or ""),
        Column("Horizon", lambda s: (s.horizon_name or "custom") if s.has_horizon else "", flex=True),
    ])
    print("\n* = active location")
