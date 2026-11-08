"""Imaging commands: 'solve' (plate solving, as the image viewer does) and
'quality' (grading light frames, as the Light Quality review does).

They use the same code as the app's windows (PlateSolver.solve_image,
LightQuality), the same settings (solver order, ASTAP path, astrometry.net key,
Maximum Threads, the review's thresholds) and the same saved measurements, so
the results match the app's."""

import os
import threading
from collections import Counter

from cosmoscollection.cli.LibraryCommands import REFRESH_NOTE, Progress, _run_stoppable
from cosmoscollection.core.Paths import path_key
from cosmoscollection.cli.Output import (EXIT_ERROR, EXIT_USAGE, CliError, Column, note, print_json,
                                         print_table)
from cosmoscollection.core.Formatting import count_text, format_duration

SOLVER_CHOICES = ("builtin", "astap", "astrometry")
PICTURE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
SOLVABLE_EXTENSIONS = (".fits", ".fit", ".fts", ".xisf") + PICTURE_EXTENSIONS
HINT_RADIUS = 15  # degrees ASTAP searches around a position hint (as the image viewer does)


# ================================================================ registration

def register(subparsers, common):
    solve = subparsers.add_parser(
        "solve", parents=[common], help="plate solve images (find where they are in the sky)",
        description="Plate solve images with the solvers set up in the app (Settings > Plate Solving), in its "
                    "order: the built-in solver, ASTAP and astrometry.net. Each solution is saved next to its "
                    "image (.wcs), where every window of the app finds it.")
    solve.add_argument("paths", nargs="+", metavar="PATH",
                       help="FITS, XISF, JPG, PNG or TIFF images, or folders of them (not their subfolders)")
    where = solve.add_mutually_exclusive_group()
    where.add_argument("--object", metavar="NAME",
                       help="the object the images show, to search near (default: the gallery image's object, "
                            "else the RA/Dec in the file's header)")
    where.add_argument("--ra", type=float, metavar="DEG", help="RA to search near, degrees (with --dec)")
    solve.add_argument("--dec", type=float, metavar="DEG", help="Dec to search near, degrees (with --ra)")
    solve.add_argument("--solver", action="append", choices=SOLVER_CHOICES,
                       help="use this solver (repeat for several, tried in the order given; "
                            "default: the app's setting)")
    solve.add_argument("--fresh", action="store_true",
                       help="solve again even if an image has a saved solution, and replace it")
    solve.set_defaults(func=run_solve)

    quality = subparsers.add_parser(
        "quality", parents=[common], help="grade light frames (clouds, seeing, guiding, trails)",
        description="Measure and grade light frames as the Light Quality review does: each frame against the "
                    "others with its filter, exposure and binning on its night, and each night against a "
                    "typical one, with the review's thresholds. Measurements are saved, so only new or "
                    "changed frames are analyzed again. Give a session id (from 'sessions list'), or files "
                    "and folders of subs.",
        epilog="The options that change things act on the Reject frames (with --include-marginal, Marginal "
               "too; with --keep-top, every frame outside the top PCT%), limited by --night and --filter.")
    quality.add_argument("what", nargs="+", metavar="SESSION_ID|PATH",
                         help="a session id, or FITS/XISF light frames or folders of them (with their subfolders)")
    quality.add_argument("--night", metavar="YYYY-MM-DD", help="only this night's frames")
    quality.add_argument("--filter", metavar="NAME", help="only this filter's frames")
    quality.add_argument("--all", action="store_true", help="list every frame, not just those that aren't Good")
    quality.add_argument("--reanalyze", action="store_true", help="measure every frame again (ignore saved "
                                                                  "measurements)")
    quality.add_argument("--workers", type=int, metavar="N",
                         help="frames analyzed at once (default: Settings > Maximum Threads)")
    which = quality.add_mutually_exclusive_group()
    which.add_argument("--include-marginal", action="store_true", help="act on Marginal frames too")
    which.add_argument("--keep-top", type=int, nargs="?", const=0, metavar="PCT",
                       help="act on every frame outside the best PCT%% of each filter (default: the percentage "
                            "last used in the app)")
    action = quality.add_mutually_exclusive_group()
    action.add_argument("--remove", action="store_true",
                        help="remove the frames from the session; the files stay where they are (asks first)")
    action.add_argument("--move", nargs="?", const="", metavar="FOLDER",
                        help="move the frames into FOLDER, or a 'Rejected' folder beside each one, and remove "
                             "them from the session (asks first)")
    action.add_argument("--delete", action="store_true",
                        help="delete the files permanently - not to the Recycle Bin - and remove them from "
                             "the session (asks first)")
    quality.set_defaults(func=run_quality)


# ================================================================ solve

def _image_paths(paths, extensions):
    """The files given, with each folder's own files (not subfolders') that
    have one of extensions; CliError if something given doesn't exist."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            found += sorted(os.path.join(path, name) for name in os.listdir(path)
                            if os.path.splitext(name)[1].lower() in extensions
                            and os.path.isfile(os.path.join(path, name)))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise CliError(f"Not found: {path}")
    return list(dict.fromkeys(os.path.abspath(p) for p in found))


def _file_header(path):
    """A FITS/XISF file's header keywords ({} for other images or unreadable)."""
    from cosmoscollection.sessions import SessionFileScanner as sfs
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in sfs.FITS_EXTENSIONS:
            return sfs.read_fits_header_fast(path)
        if ext in sfs.XISF_EXTENSIONS:
            return sfs.extract_xisf_header(path, keywords=None)
    except Exception:
        pass
    return {}


def _gallery_positions(conn):
    """{normalized image path: (ra, dec, object id)} for the gallery's images."""
    rows = conn.execute("SELECT ui.image_path, d.ra, d.dec, d.id FROM userimages ui "
                        "JOIN dsodetail d ON d.id = ui.dsodetailid "
                        "WHERE ui.image_path IS NOT NULL AND ui.image_path <> ''").fetchall()
    return {path_key(path): (ra, dec, dso) for path, ra, dec, dso in rows
            if ra is not None and dec is not None}


def _solution_json(path, result, source, seconds, saved_before):
    from cosmoscollection.solving.PlateSolver import PlateSolver
    scale = result.pixel_scale
    if scale is None and result.wcs_header:
        scale = PlateSolver._pixel_scale_from_wcs(result.wcs_header)
    return {"path": path, "solved": bool(result.success), "already_solved": saved_before,
            "solver": result.solver_used, "ra": result.ra_center, "dec": result.dec_center,
            "pixel_scale": scale, "hint": source, "seconds": round(seconds, 1),
            "error": None if result.success else result.error_message}


def run_solve(ctx):
    import time
    from cosmoscollection.imaging.WCS import read_wcs_header
    from cosmoscollection.solving.PlateSolver import PlateSolveResult, solve_image

    args = ctx.args
    if (args.ra is None) != (args.dec is None):
        raise CliError("--ra and --dec go together.", EXIT_USAGE)
    paths = _image_paths(args.paths, SOLVABLE_EXTENSIONS)
    if not paths:
        raise CliError("No images to solve.", hint="Give FITS, XISF, JPG, PNG or TIFF files, or folders of them.")

    object_hint = None
    with ctx.connection() as conn:
        if args.object:
            from cosmoscollection.cli.CollectionCommands import _label, resolve_dso
            dso = resolve_dso(conn, args.object)
            if dso.get("ra_deg") is None or dso.get("dec_deg") is None:
                raise CliError(f"{_label(dso)} has no position in the catalog.")
            object_hint = (dso["ra_deg"], dso["dec_deg"], _label(dso))
        gallery = _gallery_positions(conn)

    def hints_for(path):
        """(hints, where they came from) - {} for a blind solve."""
        if args.ra is not None:
            return {"ra": args.ra, "dec": args.dec, "radius": HINT_RADIUS}, "--ra/--dec"
        if object_hint:
            return {"ra": object_hint[0], "dec": object_hint[1], "radius": HINT_RADIUS}, object_hint[2]
        known = gallery.get(path_key(path))
        if known:
            return {"ra": known[0], "dec": known[1], "radius": HINT_RADIUS}, "gallery object"
        from cosmoscollection.sessions.SessionFileScanner import _file_coordinates
        coords = _file_coordinates(_file_header(path))
        if coords:
            return {"ra": coords[0], "dec": coords[1], "radius": HINT_RADIUS}, "file header"
        return {}, None

    progress = Progress(ctx)
    cancel = threading.Event()

    def work():
        results = []
        for number, path in enumerate(paths, 1):
            if cancel.is_set():
                break
            name = os.path.basename(path)
            started = time.perf_counter()
            saved = None if args.fresh else read_wcs_header(path)
            if saved:
                result = PlateSolveResult()
                result.success = True
                result.wcs_header = saved
                result.ra_center, result.dec_center = saved.get("CRVAL1"), saved.get("CRVAL2")
                results.append(_solution_json(path, result, None, 0.0, True))
                continue
            hints, source = hints_for(path)
            progress.show(f"{number}/{len(paths)} {name}: starting...")
            result = solve_image(path, hints, fresh=args.fresh, order=args.solver,
                                 progress=lambda message: progress.show(f"{number}/{len(paths)} {name}: {message}"),
                                 cancelled=cancel.is_set)
            results.append(_solution_json(path, result, source, time.perf_counter() - started, False))
        return results

    results = _run_stoppable(work, cancel, progress)
    progress.done()
    failed = [r for r in results if not r["solved"]]

    if ctx.json:
        print_json(results)
        return EXIT_ERROR if failed else None

    from cosmoscollection.core.Formatting import format_dec, format_ra

    def status(r):
        if r["already_solved"]:
            return "Saved before"
        return f"Solved ({r['solver']})" if r["solved"] else "Failed"

    print_table(results, [
        Column("File", lambda r: os.path.basename(r["path"]), flex=True),
        Column("Result", status),
        Column("RA", lambda r: format_ra(r["ra"]) if r["solved"] and r["ra"] is not None else ""),
        Column("Dec", lambda r: format_dec(r["dec"]) if r["solved"] and r["dec"] is not None else ""),
        Column("Scale", lambda r: f'{r["pixel_scale"]:.2f}"/px' if r["pixel_scale"] else "", ">"),
        Column("Time", lambda r: f"{r['seconds']:.0f}s" if r["seconds"] and not r["already_solved"] else "", ">"),
    ])
    for r in failed:
        print(f"\n{os.path.basename(r['path'])}" + (f" (searched near: {r['hint']})" if r["hint"] else
                                                    " (blind - no position to search near)") + ":")
        for line in (r["error"] or "No solution").split("\n\n"):
            print(f"  {line}")
    solved_now = sum(1 for r in results if r["solved"] and not r["already_solved"])
    before = sum(1 for r in results if r["already_solved"])
    summary = [f"{count_text(solved_now, 'image')} solved"]
    if before:
        summary.append(f"{before} already had a solution (--fresh solves again)")
    if failed:
        summary.append(f"{len(failed)} failed")
    print("\n" + ", ".join(summary) + ".")
    if failed and not args.object and args.ra is None and any(r["hint"] is None for r in failed):
        print("Next: --object NAME gives the solvers a position to search near.")
    return EXIT_ERROR if failed else None


# ================================================================ quality

def _session_of(ctx, what):
    """The session id when what is one number that isn't a file or folder, else None."""
    if len(what) == 1 and what[0].isdigit() and not os.path.exists(what[0]):
        return int(what[0])
    return None


def _not_a_light(header):
    kind = str(header.get("IMAGETYP") or header.get("FRAME") or "").lower()
    return any(word in kind for word in ("flat", "dark", "bias", "offset"))


def _folder_lights(ctx, paths, progress):
    """{path: file dict} for the light frames among paths and in the folders
    (and their subfolders, except 'Rejected' ones), as session_light_files()
    gives them - their filter, exposure, binning, gain and night from the
    headers. Night dates use the observing location's time zone."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timezone
    from cosmoscollection.planning import Observer
    from cosmoscollection.sessions import SessionFileScanner as sfs
    from cosmoscollection.sessions.SessionObservations import night_date_for, parse_obs_datetime

    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d != sfs.REJECTED_FOLDER)
                files += [os.path.join(root, n) for n in sorted(names)
                          if os.path.splitext(n)[1].lower() in sfs.SUPPORTED_EXTENSIONS]
        elif os.path.isfile(path):
            if os.path.splitext(path)[1].lower() not in sfs.SUPPORTED_EXTENSIONS:
                raise CliError(f"Not a FITS or XISF file: {path}")
            files.append(path)
        else:
            raise CliError(f"Not found: {path}")
    files = list(dict.fromkeys(os.path.abspath(f) for f in files))

    try:
        site = ctx.site()
        zone = Observer.local_zone(site.timezone)
    except CliError:
        zone = Observer.local_zone(None)  # No location set up: this computer's time zone

    from cosmoscollection.sessions.SessionFileScanner import to_float as number

    progress.show(f"Reading {count_text(len(files), 'header')}...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        headers = list(pool.map(_file_header, files))
    progress.done()

    lights = {}
    for path, header in zip(files, headers):
        if _not_a_light(header):
            continue
        night = None
        when = parse_obs_datetime(header.get("DATE-OBS"))
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)  # DATE-OBS is UTC
            night = night_date_for(when.astimezone(zone).replace(tzinfo=None)).isoformat()
        exposure = number(header.get("EXPTIME", header.get("EXPOSURE")))
        binning = number(header.get("XBINNING"))
        lights[path] = {"id": None, "file_path": path, "frame_type": "Light",
                        "filter_name": str(header["FILTER"]).strip() if header.get("FILTER") else None,
                        "exptime_seconds": exposure, "xbinning": int(binning) if binning else 1,
                        "gain": number(header.get("GAIN")), "night_date": night,
                        "date_obs": header.get("DATE-OBS")}
    return lights


def _frame_json(path, light, metrics, grade):
    ok = metrics is not None and metrics.ok
    return {
        "path": path, "grade": grade.grade if grade else None,
        "score": round(grade.score, 1) if grade and grade.grade in _ranked() else None,
        "night": light["night_date"], "filter": light["filter_name"] or "No filter",
        "exposure_seconds": light["exptime_seconds"], "binning": light["xbinning"] or 1,
        "stars": metrics.stars if ok else None,
        "fwhm_px": round(metrics.fwhm_px, 3) if ok and metrics.fwhm_px is not None else None,
        "fwhm_arcsec": round(metrics.fwhm_arcsec, 2) if ok and metrics.fwhm_arcsec else None,
        "eccentricity": round(metrics.eccentricity, 3) if ok and metrics.eccentricity is not None else None,
        "background": round(metrics.background, 1) if ok and metrics.background is not None else None,
        "trails": len(metrics.trails) if ok and metrics.trails is not None else None,
        "relative": {k: round(v, 1) for k, v in grade.relative.items()} if grade else {},
        "issues": list(grade.flags) if grade else [],
        "error": metrics.error if metrics is not None and not metrics.ok else None,
    }


def _ranked():
    from cosmoscollection.imaging import FrameQuality as fq
    return (fq.GRADE_GOOD, fq.GRADE_MARGINAL, fq.GRADE_REJECT)


def _night_label(night):
    from datetime import datetime
    try:
        return datetime.strptime(night, "%Y-%m-%d").strftime("%b %d")
    except (TypeError, ValueError):
        return "No date"


def run_quality(ctx):
    import time
    from cosmoscollection.imaging import FrameQuality as fq
    from cosmoscollection.imaging import LightQuality as lq

    args = ctx.args
    session_id = _session_of(ctx, args.what)
    acting = args.remove or args.move is not None or args.delete
    if args.remove and session_id is None:
        raise CliError("--remove takes frames out of a session: give a session id.", EXIT_USAGE)
    if args.move and not os.path.isdir(args.move):
        raise CliError(f"Folder not found: {args.move}", hint="Make the folder first, or leave it out to move "
                                                               "each frame into a 'Rejected' folder beside it.")
    if args.workers is not None and args.workers < 1:
        raise CliError("--workers must be at least 1.", EXIT_USAGE)
    progress = Progress(ctx)

    session = None
    with ctx.connection() as conn:
        lq.ensure_cache_table(conn)
        conn.commit()
        if session_id is not None:
            row = conn.execute("SELECT id, dso_name, session_date FROM usersessions WHERE id = ?",
                               (session_id,)).fetchone()
            if row is None:
                raise CliError(f"No session with id {session_id}.", hint="See 'sessions list' for the ids.")
            session = {"id": row[0], "name": row[1], "date": row[2]}
            lights = lq.session_light_files(conn, session_id)
        else:
            lights = None
    if lights is None:
        lights = _folder_lights(ctx, args.what, progress)
    if not lights:
        raise CliError("This session has no light frames attached." if session else "No light frames found.",
                       hint=None if session else "Give FITS or XISF light frames, or folders of them.")

    with ctx.connection() as conn:
        cached = {} if args.reanalyze else lq.load_cached(conn, lights)
    metrics = {path: m for path, (_size, _mtime, version, m) in cached.items()
               if version == fq.ANALYSIS_VERSION}

    # Analyze what's new or changed, saving measurements as they come in
    cancel = threading.Event()
    counts = {"to_analyze": 0, "analyzed": 0}
    started = time.perf_counter()
    workers = args.workers or lq.analysis_workers()

    def work():
        pending = []

        def flush():
            # Its own connection: the command's is tied to the main thread
            if pending:
                import sqlite3
                from cosmoscollection.core.ResourceManager import ResourceManager
                conn = sqlite3.connect(str(ResourceManager.get_database_path()), timeout=30)
                try:
                    lq.save_cached(conn, pending)
                    conn.commit()
                finally:
                    conn.close()
                pending.clear()

        def on_checked(count):
            counts["to_analyze"] = count
            if count:
                progress.show(f"Analyzing {count_text(count, 'frame')} with {count_text(workers, 'thread')}...")

        def on_frame(m, stamp):
            metrics[m.path] = m
            if stamp is None:
                return
            pending.append((m, stamp[0], stamp[1]))
            counts["analyzed"] += 1
            done, total = counts["analyzed"], counts["to_analyze"]
            elapsed = time.perf_counter() - started
            left = elapsed / done * (total - done)
            progress.show(f"Analyzed {done} of {total}" + (f" · about {format_duration(left)} left" if left >= 60
                                                           else " · under a minute left" if total > done else ""))
            if len(pending) >= 200:
                flush()

        try:
            progress.show(f"Checking {count_text(len(lights), 'light')} against saved measurements...")
            lq.analyze(list(lights), cached, force=args.reanalyze, workers=workers, on_checked=on_checked,
                       on_frame=on_frame, cancelled=cancel.is_set)
        finally:
            flush()

    _run_stoppable(work, cancel, progress)
    progress.done()
    seconds = time.perf_counter() - started

    grades = lq.grade_lights(lights, metrics, lq.saved_grade_settings())

    # What's shown (and acted on): --night and --filter narrow it, as the review's filter does
    shown = [p for p in lights
             if (not args.night or lights[p]["night_date"] == args.night)
             and (not args.filter or (lights[p]["filter_name"] or "No filter").lower() == args.filter.lower())]
    if not shown:
        what = " and ".join(x for x in (f"night {args.night}" if args.night else "",
                                         f"filter {args.filter}" if args.filter else "") if x)
        nights = sorted({l["night_date"] or "" for l in lights.values()})
        filters = sorted({l["filter_name"] or "No filter" for l in lights.values()})
        raise CliError(f"No frames for {what}.", hint=f"Nights: {', '.join(n for n in nights if n) or 'none'}. "
                                                      f"Filters: {', '.join(filters)}.")

    # The frames an action takes
    if args.keep_top is not None:
        percent = args.keep_top or lq.saved_top_percent()
        if not 1 <= percent <= 100:
            raise CliError("--keep-top takes a percentage from 1 to 100.", EXIT_USAGE)
        keep = lq.top_percent_lights(lights, grades, percent)
        selected = [p for p in shown if p in grades and grades[p].grade in _ranked() and p not in keep]
        selection = f"outside the top {percent}%"
        describe = lambda n: f"{count_text(n, 'frame')} outside the top {percent}%"  # noqa: E731
    else:
        wanted = {fq.GRADE_REJECT} | ({fq.GRADE_MARGINAL} if args.include_marginal else set())
        selected = [p for p in shown if p in grades and grades[p].grade in wanted]
        selection = "Reject and Marginal" if args.include_marginal else "Reject"
        describe = lambda n: f"{n} {selection} {'frame' if n == 1 else 'frames'}"  # noqa: E731

    integration = lambda paths: sum(lights[p]["exptime_seconds"] or 0 for p in paths)  # noqa: E731
    by_grade = Counter(grades[p].grade for p in shown if p in grades)
    not_analyzed = [p for p in shown if p not in grades]
    nights_down = lq.bad_nights({p: lights[p] for p in shown}, grades)
    stopped = cancel.is_set()

    result = {
        "session": session, "lights": len(lights), "shown": len(shown),
        "analyzed": counts["analyzed"], "from_saved": len(lights) - counts["to_analyze"],
        "counts": {grade: by_grade.get(grade, 0) for grade in (fq.GRADE_GOOD, fq.GRADE_MARGINAL, fq.GRADE_REJECT,
                                                               fq.GRADE_UNGRADED, fq.GRADE_ERROR)},
        "not_analyzed": len(not_analyzed),
        "nights_graded_down": [{"night": night, "grade": grade, "issue": issue, "frames": count,
                                "frame_grades": dict(frame_grades)}
                               for night, (grade, issue, count, frame_grades) in sorted(nights_down.items())],
        "selected": {"which": selection, "frames": len(selected), "integration_seconds": integration(selected)},
    }

    if not ctx.json:
        _print_quality(ctx, session, lights, shown, metrics, grades, by_grade, nights_down, counts, seconds,
                       stopped, integration)

    if acting and selected:
        result["action"] = _act(ctx, session, lights, selected, describe, integration)
    elif acting:
        result["action"] = {"done": 0, "failed": {}}
        if not ctx.json:
            print(f"\nNo frames to {'remove' if args.remove else 'move' if args.move is not None else 'delete'}: "
                  f"none are {selection}.")
    elif not ctx.json:
        _print_next(session, selected, describe, by_grade, not_analyzed, lights, grades, integration)

    if ctx.json:
        result["frames"] = [_frame_json(p, lights[p], metrics.get(p), grades.get(p)) for p in shown]
        print_json(result)
    return EXIT_ERROR if result.get("action", {}).get("failed") else None


def _print_quality(ctx, session, lights, shown, metrics, grades, by_grade, nights_down, counts, seconds, stopped,
                   integration):
    from cosmoscollection.imaging import FrameQuality as fq
    from cosmoscollection.imaging import LightQuality as lq

    args = ctx.args
    if session:
        print(f"Session {session['id']}: {session['name']} ({session['date']}) - {count_text(len(lights), 'light')}")
    else:
        print(f"{count_text(len(lights), 'light')} found")
    if stopped:
        print(f"Stopped - {counts['analyzed']} of {counts['to_analyze']} frame(s) analyzed.")
    elif counts["to_analyze"]:
        print(f"Analyzed {count_text(counts['to_analyze'], 'frame')} in {format_duration(seconds)}; "
              f"{len(lights) - counts['to_analyze']} from saved measurements.")
    else:
        print("All frames from saved measurements.")
    print()

    # Per filter group
    groups = {}
    for path in shown:
        groups.setdefault(lq.group_key(lights[path]), []).append(path)
    rows = []
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1] or 0, k[2])):
        paths = groups[key]
        tally = Counter(grades[p].grade for p in paths if p in grades)
        rows.append({"filter": key[0], "exposure": key[1], "binning": key[2], "frames": len(paths),
                     "integration": integration(paths), **tally})
    print_table(rows, [
        Column("Filter", lambda r: r["filter"], flex=True),
        Column("Exposure", lambda r: format_duration(r["exposure"]) if r["exposure"] else "?", ">"),
        Column("Bin", lambda r: f"{r['binning']}x{r['binning']}"),
        Column("Frames", lambda r: r["frames"], ">"),
        Column("Integration", lambda r: format_duration(r["integration"]), ">"),
        Column("Good", lambda r: r.get(fq.GRADE_GOOD, 0), ">"),
        Column("Marginal", lambda r: r.get(fq.GRADE_MARGINAL, 0), ">"),
        Column("Reject", lambda r: r.get(fq.GRADE_REJECT, 0), ">"),
        Column("Other", lambda r: r.get(fq.GRADE_UNGRADED, 0) + r.get(fq.GRADE_ERROR, 0) or "", ">"),
    ])

    for night, (grade, issue, count, frame_grades) in sorted(nights_down.items()):
        why = issue.replace("?; ", "? ")
        if grade == fq.GRADE_REJECT:
            kept = count - frame_grades[fq.GRADE_REJECT]
            print(f"\n{_night_label(night)} doesn't match your other nights. {why} "
                  + (f"All {count:,} of its frames are Rejects." if not kept else
                     f"{count - kept:,} of its {count:,} frames are Rejects; the {kept:,} that hold up on their "
                     f"own are kept."))
        else:
            print(f"\n{_night_label(night)} is a little behind your other nights. {why} Its frames are Marginal "
                  f"at best - still worth stacking with Good + Marginal.")

    listed = shown if args.all else [p for p in shown if not (p in grades and grades[p].grade == fq.GRADE_GOOD)]
    if listed:
        order = {fq.GRADE_REJECT: 0, fq.GRADE_ERROR: 1, fq.GRADE_MARGINAL: 2, fq.GRADE_UNGRADED: 3,
                 fq.GRADE_GOOD: 4}
        listed = sorted(listed, key=lambda p: (order.get(grades[p].grade, 5) if p in grades else 6,
                                               grades[p].score if p in grades else 0))
        print(f"\n{'Frames' if args.all else 'Frames that are not Good'} ({len(listed):,}):")

        def fwhm(p):
            m = metrics.get(p)
            if not (m and m.ok and m.fwhm_px is not None):
                return ""
            return f"{m.fwhm_px:.2f}px" + (f' ({m.fwhm_arcsec:.1f}")' if m.fwhm_arcsec else "")

        def issues(p):
            g, m = grades.get(p), metrics.get(p)
            if g and g.flags:
                return "; ".join(g.flags)
            return m.error if m is not None and not m.ok else ""

        print_table(listed, [
            Column("Grade", lambda p: grades[p].grade if p in grades else "Not analyzed"),
            Column("Score", lambda p: f"{grades[p].score:.0f}" if p in grades and grades[p].grade in _ranked()
                   else "", ">"),
            Column("File", lambda p: os.path.basename(p), flex=True),
            Column("Night", lambda p: lights[p]["night_date"] or ""),
            Column("Filter", lambda p: lights[p]["filter_name"] or "No filter"),
            Column("Stars", lambda p: f"{metrics[p].stars:,}" if p in metrics and metrics[p].ok else "", ">"),
            Column("FWHM", fwhm, ">"),
            Column("Ecc", lambda p: f"{metrics[p].eccentricity:.2f}" if p in metrics and metrics[p].ok
                   and metrics[p].eccentricity is not None else "", ">"),
            Column("Issues", issues, flex=True),
        ])

    parts = [f"{by_grade[g]:,} {g} ({format_duration(integration([p for p in shown if p in grades and grades[p].grade == g]))})"
             for g in (fq.GRADE_GOOD, fq.GRADE_MARGINAL, fq.GRADE_REJECT, fq.GRADE_UNGRADED, fq.GRADE_ERROR)
             if by_grade.get(g)]
    print(f"\n{len(shown):,} lights: " + " · ".join(parts))


def _print_next(session, selected, describe, by_grade, not_analyzed, lights, grades, integration):
    from cosmoscollection.imaging import FrameQuality as fq
    from cosmoscollection.imaging import LightQuality as lq
    from cosmoscollection.sessions.SessionFileScanner import REJECTED_FOLDER

    if not_analyzed:
        print(f"{count_text(len(not_analyzed), 'frame')} {'is' if len(not_analyzed) == 1 else 'are'}n't analyzed "
              f"yet - run this again to finish.")
    keep = lq.KEEP_GRADES[lq.LIGHTS_GOOD_MARGINAL]
    suggested = [p for p in lights if p in grades and grades[p].grade in keep]
    if suggested:
        print(f"Suggested: stack {lq.LIGHT_CHOICE_LABELS[lq.LIGHTS_GOOD_MARGINAL]} "
              f"({count_text(len(suggested), 'frame')}, {format_duration(integration(suggested))}).")
    if selected:
        them = "it" if len(selected) == 1 else "them"
        if session:
            print(f"Next: --move puts the {describe(len(selected))} in '{REJECTED_FOLDER}' folders and takes "
                  f"{them} out of the session; --remove only takes {them} out of the session.")
        else:
            print(f"Next: --move puts the {describe(len(selected))} in '{REJECTED_FOLDER}' folders.")
    elif by_grade.get(fq.GRADE_GOOD) or by_grade.get(fq.GRADE_MARGINAL):
        print("No problem frames - ready to stack.")


def _act(ctx, session, lights, selected, describe, integration):
    """Remove, move or delete the selected frames (asking first), and update the
    sessions and saved measurements to match; {"done", "failed", ...}."""
    from cosmoscollection.cli.Prompts import confirm
    from cosmoscollection.imaging import LightQuality as lq
    from cosmoscollection.sessions import SessionObservations as so
    from cosmoscollection.sessions.SessionFileScanner import REJECTED_FOLDER

    args = ctx.args
    described = f"the {describe(len(selected))} ({format_duration(integration(selected))} of integration)"
    with ctx.connection() as conn:
        others = lq.other_sessions(conn, session["id"] if session else -1, selected)
    shared = sum(len(ids) for _label, ids in others.values())

    if args.remove:
        mode = None
        question = f"Remove {described} from the session? The files stay where they are on disk."
    elif args.move is not None:
        mode = lq.MOVE
        where = args.move or f"a '{REJECTED_FOLDER}' folder beside each one"
        question = f"Move {described} into {where}" + (" and remove them from the session?" if session else "?")
        if shared:
            question += f" {count_text(shared, 'file')} also in other sessions stay attached there, at the new place."
    else:
        mode = lq.DELETE
        size = 0
        for path in selected:
            try:
                size += os.path.getsize(path)
            except OSError:
                pass
        question = (f"Permanently delete {described}, {size / 1e9:.1f} GB? They don't go to the Recycle Bin and "
                    f"can't be recovered" + (", and are removed from the session" if session else "") + ".")
        if shared:
            question += f" {count_text(shared, 'file')} also in other sessions are removed from there too."
    if not ctx.json:
        print()
    if not confirm(ctx, question):
        raise CliError("Nothing changed.", EXIT_USAGE)

    # A night left with no lights: its flats/darks/bias go too, if wanted (as the review asks)
    extra = []
    if session:
        with ctx.connection() as conn:
            stranded = so.calibration_left_without_night(conn, session["id"], [lights[p]["id"] for p in selected])
        if stranded and confirm(ctx, f"That leaves {so.stranded_calibration_text(stranded)} with no lights. Also "
                                     f"remove its calibration frames from the session? The files on disk aren't "
                                     f"touched.", default=True):
            extra = [file_id for files in stranded.values() for file_id, _type in files]

    progress = Progress(ctx)
    if mode is None:
        finished, failed = {p: p for p in selected}, {}
    else:
        verb = "Moving" if mode == lq.MOVE else "Deleting"
        cancel = threading.Event()
        finished, failed = _run_stoppable(
            lambda: lq.move_or_delete(mode, selected, args.move or None,
                                      progress=lambda done, total: progress.show(f"{verb} {done} of {total}..."),
                                      cancelled=cancel.is_set),
            cancel, progress)
        progress.done()

    if finished:
        with ctx.connection() as conn:
            try:
                if session:
                    lq.record_removed(conn, session["id"], lights, finished, mode, extra, others)
                else:
                    _record_loose(conn, finished, mode, others)
                conn.commit()
            except Exception:
                conn.rollback()
                done = {None: "", lq.MOVE: " The files were already moved.", lq.DELETE: " The files were "
                        "already deleted."}[mode]
                note(f"Updating the database failed.{done}")
                raise

    if not ctx.json:
        if finished:
            if mode is None:
                print(f"Removed {count_text(len(finished), 'frame')} from the session.")
            elif mode == lq.MOVE:
                where = args.move or f"'{REJECTED_FOLDER}' folders"
                print(f"Moved {count_text(len(finished), 'frame')} into {where}"
                      + (" and removed them from the session." if session else "."))
            else:
                print(f"Deleted {count_text(len(finished), 'frame')}"
                      + (" and removed them from the session." if session else "."))
            if extra:
                print(f"Also removed {count_text(len(extra), 'calibration frame')} from the session.")
            note(REFRESH_NOTE)
        for path, error in list(failed.items())[:10]:
            note(f"Not changed: {os.path.basename(path)}: {error}")
        if len(failed) > 10:
            note(f"...and {len(failed) - 10} more not changed.")
    return {"done": len(finished), "calibration_removed": len(extra),
            "moved_to": {p: new for p, new in finished.items()} if mode == lq.MOVE else None,
            "failed": failed}


def _record_loose(conn, changed, mode, others):
    """Files graded from folders, moved or deleted: sessions they're attached to
    follow a move, and lose a deleted file; the saved measurements follow."""
    from cosmoscollection.sessions import SessionObservations as so
    if mode == "move":
        for old, new in changed.items():
            conn.execute("UPDATE usersessionfiles SET file_path = ? WHERE file_path = ?", (new, old))
            conn.execute("UPDATE OR REPLACE framequality SET file_path = ? WHERE file_path = ?", (new, old))
        return
    for session_id, (_label, ids) in others.items():
        so.remove_files(conn, session_id, ids)
    paths = list(changed)
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        conn.execute(f"DELETE FROM framequality WHERE file_path IN ({','.join('?' * len(chunk))})", chunk)
