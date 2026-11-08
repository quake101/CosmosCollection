"""Library commands: 'library scan' (the Import Wizard from the command line),
'library folders', and 'sessions check' (the Session File Check).

They use the same code as the app's windows (LibraryScanner, FileCheck), so a
scan here finds, groups and imports exactly what the wizard would with its
default choices."""

import os
import sys
import threading

from cosmoscollection.cli.Output import (EXIT_USAGE, CliError, Column, note, print_fields, print_json,
                                         print_table)
from cosmoscollection.core.Formatting import count_text, format_duration

REFRESH_NOTE = "If the app is open, its windows show this change after they're refreshed or reopened."

EQUIPMENT_CHOICES = ("ask", "suggested", "add", "keep")


# ================================================================ registration

def register(subparsers, common, groups):
    from cosmoscollection.cli.CollectionCommands import group

    actions = group(subparsers, common, "library", "scan your image folders and import what's new")

    scan = actions.add_parser(
        "scan", parents=[common], help="find sessions and finished images in folders (and import them)",
        description="Scan folders the way the Import Wizard does: subs are grouped into sessions per object and "
                    "equipment, finished images are paired with them. Lists what's new; with --import, adds it "
                    "with the wizard's default choices. Files are never moved or changed.")
    scan.add_argument("folders", nargs="*", metavar="FOLDER",
                      help="folders to scan (default: the Import Wizard's saved folders)")
    scan.add_argument("--import", dest="do_import", action="store_true",
                      help="import the new sessions and images (asks first unless --yes)")
    only = scan.add_mutually_exclusive_group()
    only.add_argument("--sessions-only", action="store_true", help="import sessions, not finished images")
    only.add_argument("--images-only", action="store_true", help="import finished images, not sessions")
    scan.add_argument("--equipment", choices=EQUIPMENT_CHOICES, default="ask",
                      help="telescope/camera names in the files not matched to your Equipment yet: ask (default), "
                           "use the suggested match (else keep the name), add new equipment when there's no "
                           "match, or keep the names unmatched")
    scan.add_argument("--all", action="store_true", help="list everything found, not just what's new")
    scan.add_argument("--full", action="store_true", help="read every file's header again (ignore what earlier "
                                                          "scans found)")
    scan.add_argument("--no-mounts", action="store_true", help="don't measure each night's mount (faster)")
    scan.add_argument("--gap-days", type=int, metavar="DAYS",
                      help="days without imaging that start a new session (default: the wizard's setting)")
    scan.add_argument("--report", metavar="FILE", help="also write the detailed scan report to FILE")
    scan.set_defaults(func=run_library_scan)

    folders = actions.add_parser("folders", parents=[common], help="the Import Wizard's saved folders")
    folders.set_defaults(func=run_library_folders)

    check = groups["sessions"].add_parser(
        "check", parents=[common], help="find sessions' files that are missing on disk or were moved",
        description="Check sessions' attached files on disk, as the Session File Check does: missing files are "
                    "looked for near where they were (a renamed or moved folder). Lists what's wrong; the "
                    "options below fix it. Files on disk are never changed.")
    check.add_argument("ids", nargs="*", type=int, metavar="ID", help="sessions to check (default: all)")
    check.add_argument("--find-in", metavar="FOLDER", help="also look for the missing files anywhere in FOLDER")
    check.add_argument("--relink", action="store_true", help="point moved files at where they were found")
    check.add_argument("--remove-missing", action="store_true",
                       help="remove files that aren't on disk from their sessions (asks first unless --yes)")
    check.add_argument("--remove-no-night", action="store_true",
                       help="remove calibration frames that aren't on any night (asks first unless --yes)")
    check.set_defaults(func=run_sessions_check)


# ================================================================ progress

class Progress:
    """One updating status line on stderr, when it's a terminal (and not --quiet)."""

    def __init__(self, ctx):
        self.enabled = not ctx.args.quiet and sys.stderr is not None and sys.stderr.isatty()
        self._width = 0

    def show(self, text):
        if not self.enabled:
            return
        text = text[:110]
        sys.stderr.write("\r" + text.ljust(self._width))
        sys.stderr.flush()
        self._width = len(text)

    def done(self):
        if self.enabled and self._width:
            sys.stderr.write("\r" + " " * self._width + "\r")
            sys.stderr.flush()
            self._width = 0


def _run_stoppable(work, cancel, progress):
    """Run work() on a thread so Ctrl+C stops it cleanly (cancel is set and the
    work finishes up - a scan saves what it read). Its result, or re-raises."""
    result, error = {}, {}

    def target():
        try:
            result["value"] = work()
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            error["value"] = e

    thread = threading.Thread(target=target, name="cli-work", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(0.2)
    except KeyboardInterrupt:
        progress.done()
        note("Stopping (saving what was read so far)...")
        cancel.set()
        thread.join()
        raise
    if "value" in error:
        raise error["value"]
    return result.get("value")


# ================================================================ library scan

def _session_json(session, group):
    from cosmoscollection.sessions import LibraryScanner
    nights = session.nights
    folders = session.folders
    try:
        folder = os.path.commonpath(folders) if len(folders) > 1 else (folders[0] if folders else "")
    except ValueError:  # Different drives
        folder = folders[0]
    return {
        "object": group.name, "dsodetailid": group.dsodetailid,
        "first_night": nights[0].isoformat() if nights else None,
        "last_night": nights[-1].isoformat() if nights else None, "nights": len(nights),
        "subs": len(session.lights), "new_subs": session.new_lights, "already_imported": session.imported,
        "integration_seconds": session.integration_seconds, "calibration_frames": len(session.calibration),
        "camera": LibraryScanner._clean(session.camera) or None,
        "telescope": LibraryScanner._clean(session.telescope) or None,
        "filters": dict(session.filters.most_common()), "status": session.status, "to_import": session.checked,
        "folder": folder,
    }


def _final_json(final, group):
    return {"object": group.name, "dsodetailid": group.dsodetailid, "path": final.record.path,
            "date": final.date.strftime("%Y-%m-%d") if final.date else None, "new": final.is_new,
            "in_gallery": final.imported, "to_import": final.checked}


def _night_range(first, last):
    if not first:
        return ""
    return first if first == last else f"{first} → {last[5:] if last[:4] == first[:4] else last}"


def _equipment_rows(library):
    """The telescope/camera names in the files nobody has matched to Equipment
    yet, with what each would be matched to: [{"kind", "value", "subs",
    "sessions", "header", "suggested" (id or None), "suggested_name", "why"}]."""
    from cosmoscollection.sessions import LibraryScanner
    info = library.equipment or LibraryScanner.EquipmentInfo()
    rows = []
    for row in LibraryScanner.equipment_in_files(library):
        if info.decided(row["kind"], row["value"]):
            continue
        suggested, why = LibraryScanner.suggest_equipment(row["kind"], row["value"], row["header"], info)
        item = info.items(row["kind"]).get(suggested) if suggested is not None else None
        rows.append(dict(row, suggested=suggested, suggested_name=item["name"] if item else None, why=why))
    return rows


def _decide_equipment(ctx, rows, mode):
    """(decisions {(kind, name): id or None}, new [(kind, name, header)]) for the
    unmatched names, by --equipment (asking each one with 'ask')."""
    from cosmoscollection.cli.Prompts import can_ask, choose
    from cosmoscollection.sessions import LibraryScanner

    if mode == "ask" and rows and not can_ask(ctx):
        names = ", ".join(f"'{r['value']}'" for r in rows)
        raise CliError(f"Your files name equipment that isn't matched to your Equipment yet: {names}.", EXIT_USAGE,
                       hint="Pass --equipment suggested, add or keep to choose without asking.")
    decisions, new = {}, []
    info = None
    for row in rows:
        kind, value = row["kind"], row["value"]
        if mode == "keep":
            choice = "keep"
        elif mode == "suggested":
            choice = row["suggested"] if row["suggested"] is not None else "keep"
        elif mode == "add":
            choice = row["suggested"] if row["suggested"] is not None else "add"
        else:
            if info is None:
                info = _current_equipment(ctx)
            options = [("keep", LibraryScanner.KEEP_NAME_CHOICE)]
            options += LibraryScanner.equipment_choices(info, kind)
            options.append(("add", LibraryScanner.add_equipment_choice(kind, row["header"])))
            why = f" - suggested: {row['suggested_name']} ({row['why']})" if row["suggested"] is not None else ""
            print(f"\n{kind.capitalize()} '{value}' in {count_text(row['subs'], 'sub')}{why}")
            choice = choose(ctx, "Match it to:", options,
                            default=row["suggested"] if row["suggested"] is not None else "keep")
        if choice == "add":
            new.append((kind, value, row["header"]))
        else:
            decisions[(kind, value)] = None if choice == "keep" else choice
    return decisions, new


def _current_equipment(ctx):
    from cosmoscollection.sessions import LibraryScanner
    with ctx.connection() as conn:
        return LibraryScanner.load_equipment_info(conn)


def _print_scan(ctx, scan, library, sessions, finals, unknown, equipment_rows, show_all):
    from cosmoscollection.sessions import LibraryScanner
    seconds = scan.seconds
    print(f"Scanned {count_text(len(scan.roots), 'folder')} in {seconds:.0f}s: {scan.folders_listed:,} folders, "
          f"{scan.files_seen:,} files, {scan.new_files:,} new since the last scan.")
    if library.relinked:
        print(f"{count_text(library.relinked, 'sub')} in renamed or moved folders relinked to where they are now.")
    if scan.cancelled:
        print("The scan was stopped part way; this lists what was found before that.")

    shown_sessions = sessions if show_all else [s for s in sessions if s["to_import"]]
    shown_finals = finals if show_all else [f for f in finals if f["to_import"]]
    print()
    if shown_sessions:
        print("Sessions" + (" found:" if show_all else " to import (new subs):"))
        print_table(shown_sessions, [
            Column("Object", lambda s: s["object"], flex=True),
            Column("Nights", lambda s: _night_range(s["first_night"], s["last_night"])),
            Column("#", lambda s: s["nights"], ">"),
            Column("Subs", lambda s: f"{s['subs']:,}", ">"),
            Column("New", lambda s: f"{s['new_subs']:,}" if s["new_subs"] else "", ">"),
            Column("Integration", lambda s: format_duration(s["integration_seconds"]), ">"),
            Column("Camera", lambda s: s["camera"] or "", flex=True),
            Column("Telescope", lambda s: s["telescope"] or "", flex=True),
            Column("Status", lambda s: s["status"]),
        ], indent="  ")
    else:
        print("Sessions: nothing new to import.")
    print()
    if shown_finals:
        print("Finished images" + (" found:" if show_all else " for the gallery:"))
        print_table(shown_finals, [
            Column("Object", lambda f: f["object"], flex=True),
            Column("Date", lambda f: f["date"] or ""),
            Column("Status", lambda f: "in gallery" if f["in_gallery"] else ("to add" if f["to_import"] else "")),
            Column("File", lambda f: os.path.relpath(f["path"], LibraryScanner._root_of(f["path"], scan.roots)),
                   flex=True),
        ], indent="  ")
    else:
        print("Finished images: nothing new for the gallery.")
    if unknown:
        print()
        print("Not imported - which object these are isn't known (pick it in the Import Wizard):")
        print_table(unknown, [
            Column("Name", lambda u: u["name"], flex=True),
            Column("Sessions", lambda u: u["sessions"], ">"),
            Column("Images", lambda u: u["images"], ">"),
            Column("Folder", lambda u: u["folder"], flex=True),
        ], indent="  ")
    if equipment_rows:
        print()
        print("Equipment names in your files not matched to your Equipment yet (decided when importing):")
        print_table(equipment_rows, [
            Column("Kind", lambda r: r["kind"]),
            Column("Name in files", lambda r: r["value"], flex=True),
            Column("Subs", lambda r: f"{r['subs']:,}", ">"),
            Column("Suggested match", lambda r: f"{r['suggested_name']} ({r['why']})" if r["suggested_name"]
                   else "none - would be kept as named", flex=True),
        ], indent="  ")


def run_library_scan(ctx):
    from cosmoscollection.sessions import LibraryScanner
    from cosmoscollection.sessions.ImportSettings import common_names_for_new_sessions, load_rules, saved_folders

    args = ctx.args
    folders = [os.path.abspath(os.path.expanduser(f)) for f in args.folders] or saved_folders()
    if not folders:
        raise CliError("No folders to scan.", EXIT_USAGE,
                       hint="Name them (library scan FOLDER ...), or add them in the app's Import Wizard.")
    missing = [f for f in folders if not os.path.isdir(f)]
    if missing:
        raise CliError(f"Folder not found: {', '.join(missing)}")
    rules = load_rules()
    if args.gap_days:
        rules.session_gap_days = args.gap_days
    with ctx.connection():
        pass  # The database's tables are set up (a new collection's too) before the scan opens its own connection

    progress = Progress(ctx)
    cancel = threading.Event()

    def show_scan(scan):
        done = scan.headers_read + scan.headers_cached
        progress.show(f"Scanning: {scan.folders_listed:,} folders, {scan.files_seen:,} files, "
                      f"{done:,} headers ({scan.seconds:.0f}s)")

    def work():
        return LibraryScanner.scan_and_build(folders, rules, progress=show_scan, cancel=cancel, reread=args.full,
                                             scan_index=LibraryScanner.ScanIndex(), status=progress.show,
                                             mounts=not args.no_mounts)

    scan, library = _run_stoppable(work, cancel, progress)
    progress.done()

    # Sessions already in the Session Manager get the mounts just measured (as the wizard does)
    if not args.no_mounts:
        try:
            with ctx.connection() as conn:
                LibraryScanner.save_known_mounts(conn, library)
        except Exception as e:  # Only the mounts are missing
            note(f"Couldn't save the measured mounts: {e}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as out:
            LibraryScanner.print_library(library, scan, out, new_only=not args.all)

    equipment_rows = _equipment_rows(library)
    imported = None
    if args.do_import and not scan.cancelled:
        if equipment_rows:
            decisions, new = _decide_equipment(ctx, equipment_rows, args.equipment)
            from cosmoscollection.equipment import EquipmentMapper
            with ctx.connection() as conn:
                equipment = EquipmentMapper.save_wizard_matches(conn, decisions, new)
            LibraryScanner.regroup_sessions(library, equipment)
            equipment_rows = []
        for group in library.groups.values():
            if args.images_only:
                for session in group.sessions:
                    session.checked = False
            if args.sessions_only:
                for final in group.finals:
                    final.checked = False

    groups = sorted(library.groups.values(), key=lambda g: g.name.lower())
    sessions = [_session_json(s, g) for g in groups if g.dsodetailid for s in g.sessions]
    finals = [_final_json(f, g) for g in groups if g.dsodetailid for f in g.finals]
    unknown = []
    for g in groups:
        if g.dsodetailid:
            continue
        paths = [r.folder for s in g.sessions for r in s.lights[:1]] + [f.record.folder for f in g.finals]
        unknown.append({"name": g.name, "sessions": len(g.sessions), "images": len(g.finals),
                        "folder": paths[0] if paths else ""})

    if args.do_import and not scan.cancelled:
        imported = _import(ctx, library, common_names_for_new_sessions(), progress)

    if ctx.json:
        print_json({
            "folders": scan.roots, "seconds": round(scan.seconds, 1), "folders_listed": scan.folders_listed,
            "files": scan.files_seen, "new_files": scan.new_files, "relinked": library.relinked,
            "cancelled": scan.cancelled, "sessions": sessions, "images": finals, "unknown_objects": unknown,
            "unmatched_equipment": [{k: r[k] for k in ("kind", "value", "subs", "sessions", "suggested",
                                                       "suggested_name", "why")} for r in equipment_rows],
            "imported": imported,
        })
        return
    if imported is None:
        _print_scan(ctx, scan, library, sessions, finals, unknown, equipment_rows, args.all)
        if any(s["to_import"] for s in sessions) or any(f["to_import"] for f in finals):
            print("\nNothing has been imported. Run it again with --import to add these.")
    else:
        _print_import(imported)


def _import(ctx, library, common_names, progress):
    """Import what's checked, after asking; the ImportReport as a dict (None if declined)."""
    from cosmoscollection.cli.Prompts import confirm
    from cosmoscollection.sessions import LibraryScanner

    sessions, finals = LibraryScanner.plan(library)
    if not sessions and not finals:
        return {"sessions_created": 0, "sessions_extended": 0, "subs_added": 0, "images_added": 0,
                "nothing_to_import": True}
    lights = sum(len(s.lights) for s in sessions)
    calibration = sum(len(s.calibration) for s in sessions)
    parts = []
    if sessions:
        part = f"{count_text(len(sessions), 'session')} with {count_text(lights, 'sub')}"
        if calibration:
            part += f" and {count_text(calibration, 'calibration frame')}"
        parts.append(part + " into the Session Manager")
    if finals:
        parts.append(f"{count_text(len(finals), 'image')} into the gallery")
    if not confirm(ctx, "Import " + " and ".join(parts) + "? Your files stay where they are."):
        raise CliError("Nothing imported.", EXIT_USAGE)

    cancel = threading.Event()

    def work():
        # Its own connection, as the wizard's import worker opens one: this runs on another thread
        from cosmoscollection.core.DatabaseManager import open_connection
        conn = open_connection(rows=True)
        try:
            return LibraryScanner.import_library(
                conn, library, progress=lambda done, total, text: progress.show(f"Importing {done}/{total}: {text}"),
                cancelled=cancel.is_set, common_names=common_names)
        finally:
            conn.close()

    report = _run_stoppable(work, cancel, progress)
    progress.done()
    return {"sessions_created": report.sessions_created, "sessions_extended": report.sessions_extended,
            "completed": report.completed, "in_progress": report.in_progress, "subs_added": report.subs_added,
            "calibration_added": report.calibration_added, "subs_skipped": report.subs_skipped,
            "images_added": report.images_added, "images_skipped": report.images_skipped,
            "images_linked": report.images_linked, "cancelled": report.cancelled, "errors": report.errors}


def _print_import(report):
    if report.get("nothing_to_import"):
        print("Nothing new to import.")
        return
    lines = []
    if report["sessions_created"]:
        lines.append(f"{count_text(report['sessions_created'], 'new session')} "
                     f"({report['completed']} Completed, {report['in_progress']} In Progress)")
    if report["sessions_extended"]:
        lines.append(f"{count_text(report['sessions_extended'], 'existing session')} got more subs")
    if report["subs_added"] or report["calibration_added"]:
        lines.append(f"{count_text(report['subs_added'], 'sub')} and "
                     f"{count_text(report['calibration_added'], 'calibration frame')}")
    if report["images_added"]:
        lines.append(f"{count_text(report['images_added'], 'image')} added to the gallery")
    if report["subs_skipped"]:
        lines.append(f"{count_text(report['subs_skipped'], 'sub')} skipped - already in another session")
    if report["images_skipped"]:
        lines.append(f"{count_text(report['images_skipped'], 'image')} skipped - already in the gallery")
    if report["images_linked"]:
        lines.append(f"{count_text(report['images_linked'], 'gallery image')} linked to their sessions")
    if report["cancelled"]:
        print("The import was stopped part way. What was imported is kept:")
    else:
        print("Imported:")
    for line in lines or ["nothing new"]:
        print(f"  {line}")
    for error in report["errors"][:10]:
        print(f"  Couldn't import {error}")
    note(REFRESH_NOTE)


def run_library_folders(ctx):
    from cosmoscollection.sessions.ImportSettings import saved_folders
    from cosmoscollection.sessions.LibraryScanner import ScanIndex

    index = ScanIndex()
    folders = []
    for folder in saved_folders():
        try:
            last = index.last_scan(folder)
        except Exception:
            last = None
        folders.append({"folder": folder, "exists": os.path.isdir(folder), "last_scanned": last})
    if ctx.json:
        print_json(folders)
        return
    if not folders:
        print("The Import Wizard has no saved folders. Add them in the app, or name them: library scan FOLDER")
        return
    print_table(folders, [
        Column("Folder", lambda f: f["folder"] + ("" if f["exists"] else "  (not found)"), flex=True),
        Column("Last scanned", lambda f: str(f["last_scanned"] or "never")[:16]),
    ])


# ================================================================ sessions check

def run_sessions_check(ctx):
    from collections import Counter
    from cosmoscollection.cli.Prompts import confirm
    from cosmoscollection.sessions import FileCheck
    from cosmoscollection.sessions import SessionObservations as so

    args = ctx.args
    if args.find_in and not os.path.isdir(args.find_in):
        raise CliError(f"Folder not found: {args.find_in}")
    with ctx.connection() as conn:
        rows = conn.execute("SELECT id, dso_name, session_date FROM usersessions ORDER BY session_date DESC, id DESC"
                            ).fetchall()
        sessions = {row[0]: {"id": row[0], "name": row[1], "date": row[2]} for row in rows}
        wanted = list(dict.fromkeys(args.ids)) or list(sessions)
        unknown_ids = [i for i in wanted if i not in sessions]
        if unknown_ids:
            raise CliError(f"No session with id {', '.join(map(str, unknown_ids))}.",
                           hint="See 'sessions list' for the ids.")
        files_of = {sid: so.load_session_files(conn, sid) for sid in wanted}

    progress = Progress(ctx)
    cancel = threading.Event()
    all_files = [f for sid in wanted for f in files_of[sid]]

    def work():
        result = FileCheck.check_files(all_files, find=True, status=progress.show, cancelled=cancel.is_set)
        if args.find_in:
            still = [f for f in all_files if f["id"] in result.missing and f["id"] not in result.moved
                     and f["id"] not in result.unreachable]
            if still:
                result.moved.update(FileCheck.find_moved(still, args.find_in, progress.show, cancel.is_set))
        return result

    result = _run_stoppable(work, cancel, progress)
    progress.done()
    with ctx.connection() as conn:
        result.moved = FileCheck.unattached(conn, result.moved)

    report = []
    for sid in wanted:
        files = files_of[sid]
        ids = {f["id"] for f in files}
        session_result = FileCheck.CheckResult(result.missing & ids, result.unreachable & ids,
                                               {i: p for i, p in result.moved.items() if i in ids})
        groups = FileCheck.group_problems(files, session_result)
        if groups:
            report.append((sessions[sid], files, groups))

    kinds = {FileCheck.MOVED: "moved", FileCheck.MISSING: "not on disk",
             FileCheck.UNREACHABLE: "drive not available", FileCheck.NO_NIGHT: "on no night"}
    if ctx.json and not (args.relink or args.remove_missing or args.remove_no_night):
        print_json({"checked_sessions": len(wanted), "checked_files": len(all_files), "sessions": [{
            "id": session["id"], "name": session["name"], "problems": [{
                "kind": kind, "folder": folder, "new_folder": new_folder, "files": len(files_in),
                "frames": dict(Counter(f["frame_type"] for f in files_in))}
                for (kind, folder, new_folder), files_in in groups]}
            for session, _files, groups in report]})
        return

    if not ctx.json:
        if not report:
            print(f"All {len(all_files):,} files of {count_text(len(wanted), 'session')} are on disk and on a night.")
            return
        for session, files, groups in report:
            print(f"Session {session['id']}: {session['name']} ({session['date']})")
            for (kind, folder, new_folder), files_in in groups:
                counts = so.describe_frame_counts(Counter(f["frame_type"] for f in files_in))
                where = folder if new_folder is None else f"{folder}  →  {new_folder}"
                print(f"  {counts} {kinds[kind]}: {where}")
            print()

    moves, removals = [], {}
    for session, _files, groups in report:
        for (kind, _folder, _new), files_in in groups:
            if kind == FileCheck.MOVED and args.relink:
                moves += [(f["id"], f["file_path"], result.moved[f["id"]]) for f in files_in]
            elif (kind == FileCheck.MISSING and args.remove_missing) or \
                    (kind == FileCheck.NO_NIGHT and args.remove_no_night):
                removals.setdefault(session["id"], []).extend(f["id"] for f in files_in)
    totals = {
        "moved": sum(len(fs) for _s, _f, gs in report for (k, _a, _b), fs in gs if k == FileCheck.MOVED),
        "missing": sum(len(fs) for _s, _f, gs in report for (k, _a, _b), fs in gs if k == FileCheck.MISSING),
        "unreachable": sum(len(fs) for _s, _f, gs in report for (k, _a, _b), fs in gs if k == FileCheck.UNREACHABLE),
        "no_night": sum(len(fs) for _s, _f, gs in report for (k, _a, _b), fs in gs if k == FileCheck.NO_NIGHT),
    }
    removing = sum(len(ids) for ids in removals.values())
    if removing and not confirm(ctx, f"Remove {count_text(removing, 'file')} from "
                                     f"{count_text(len(removals), 'session')}? The files on disk aren't touched."):
        raise CliError("Nothing changed.", EXIT_USAGE)
    if moves or removals:
        with ctx.connection() as conn:
            try:
                so.relink_files(conn, moves)
                for sid, file_ids in removals.items():
                    so.remove_files(conn, sid, file_ids)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    if ctx.json:
        print_json(dict(totals, relinked=len(moves), removed=removing))
        return
    print_fields([(label, totals[key] or None) for label, key in (
        ("Moved (found elsewhere)", "moved"), ("Not on disk", "missing"), ("Drive not available", "unreachable"),
        ("Calibration on no night", "no_night"))])
    if moves:
        print(f"\nRelinked {count_text(len(moves), 'file')} to the folder{'s' if len(moves) > 1 else ''} "
              f"{'they were' if len(moves) > 1 else 'it was'} found in.")
    if removing:
        print(f"Removed {count_text(removing, 'file')} from their sessions.")
    if moves or removing:
        note(REFRESH_NOTE)
    hints = []
    if totals["moved"] and not args.relink:
        hints.append("--relink to point moved files at their new folder")
    if totals["missing"] and not args.remove_missing:
        hints.append("--find-in FOLDER to look somewhere else, or --remove-missing")
    if totals["unreachable"]:
        hints.append("connect the drive and check again for files on an unavailable drive")
    if hints:
        print("\nNext: " + "; ".join(hints) + ".")
