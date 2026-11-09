"""Night planning commands: 'tonight' (Best DSO Tonight), 'visibility' (the DSO
Visibility Calculator) and 'weather' (the Weather Forecast).

They use the same code as the app's windows (BestTonightEngine, Visibility,
WeatherFetch), with the same location, custom horizon, Maximum Threads,
weather sources and units."""

import re
import threading
from datetime import date, datetime, timedelta

from cosmoscollection.cli.Output import (EXIT_USAGE, CliError, Column, note, print_fields, print_json,
                                         print_table)
from cosmoscollection.core.Formatting import count_text, format_magnitude

# Best DSO Tonight's defaults (BestTonightEngine's DEFAULT_*), here so --help doesn't load astropy
DEFAULT_MIN_ALTITUDE = 30
DEFAULT_MAX_MAGNITUDE = 12.0
DEFAULT_DSO_LIMIT = 200

TIME_HELP = ("when the objects should be up: night (dusk to dawn, the default), dusk, midnight, dawn, now, "
             "a time (23:30) or a time frame (01:00-03:00)")
SORTS = ("hours", "altitude", "magnitude", "time", "moon", "rank")


# ================================================================ registration

def register(subparsers, common):
    tonight = subparsers.add_parser(
        "tonight", parents=[common], help="the best objects to image tonight (Best DSO Tonight)",
        description="The objects up in a dark sky tonight, as Best DSO Tonight finds them: above the minimum "
                    "altitude and your location's custom horizon while the sun is below -12 degrees. Checks the "
                    "brightest catalog objects (or your Target List), sorted by how long each is up.")
    tonight.add_argument("--time", default="night", metavar="WHEN", help=TIME_HELP)
    tonight.add_argument("--date", metavar="YYYY-MM-DD", help="the night starting on this date (default: tonight)")
    tonight.add_argument("--min-alt", type=float, default=DEFAULT_MIN_ALTITUDE, metavar="DEG",
                         help=f"minimum altitude in degrees (default {DEFAULT_MIN_ALTITUDE})")
    tonight.add_argument("--max-mag", type=float, default=DEFAULT_MAX_MAGNITUDE, metavar="MAG",
                         help=f"faintest magnitude to check (default {DEFAULT_MAX_MAGNITUDE:g}; not with --targets)")
    tonight.add_argument("--objects", type=int, default=DEFAULT_DSO_LIMIT, metavar="N",
                         help=f"how many of the brightest objects to check (default {DEFAULT_DSO_LIMIT}, "
                              f"the window's DSO Limit)")
    tonight.add_argument("--catalog", help="only this catalog's objects (M, NGC, IC, Sh2, ...)")
    tonight.add_argument("--type", help="only this type: a code (GALXY) or name ('galaxy', 'open cluster')")
    tonight.add_argument("--targets", action="store_true", help="check your Target List instead of the catalog")
    tonight.add_argument("--hide-completed", action="store_true", help="with --targets: leave out Completed targets")
    tonight.add_argument("--no-horizon", action="store_true", help="ignore the location's custom horizon")
    tonight.add_argument("--sort", choices=SORTS, default="hours",
                         help="hours up (default, as the window sorts), altitude, magnitude, best time, moon "
                              "distance, or rank (altitude and magnitude together)")
    tonight.add_argument("--limit", type=int, default=30, help="most objects to show (default 30, 0 = all)")
    tonight.set_defaults(func=run_tonight)

    visibility = subparsers.add_parser(
        "visibility", parents=[common], help="when an object is up tonight (DSO Visibility Calculator)",
        description="One object over a night: rise, transit (meridian flip) and set, the sun's times, the times "
                    "it's above the minimum altitude and your custom horizon in a dark sky, and the Moon.")
    visibility.add_argument("name", nargs="*", help="the object: a designation or name (M31, 'Orion Nebula'), or a "
                                                    "Target List entry")
    visibility.add_argument("--ra", type=float, metavar="DEG", help="a position instead of a name: RA in degrees")
    visibility.add_argument("--dec", type=float, metavar="DEG", help="and Dec in degrees")
    visibility.add_argument("--date", metavar="YYYY-MM-DD", help="the night starting on this date (default: tonight)")
    visibility.add_argument("--min-alt", type=float, default=DEFAULT_MIN_ALTITUDE, metavar="DEG",
                            help=f"minimum altitude in degrees (default {DEFAULT_MIN_ALTITUDE})")
    visibility.add_argument("--no-horizon", action="store_true", help="ignore the location's custom horizon")
    visibility.add_argument("--hourly", action="store_true", help="also list its altitude on each hour of the night")
    visibility.set_defaults(func=run_visibility)

    weather = subparsers.add_parser(
        "weather", parents=[common], help="the 7-night forecast with astro scores (Weather Forecast)",
        description="The forecast for your location, scored for imaging per night (evening to the next morning's "
                    "dark hours), from Open-Meteo plus the weather sources turned on in Settings.")
    weather.add_argument("--days", type=int, default=7, metavar="N", help="nights to show (default 7)")
    weather.add_argument("--hourly", metavar="YYYY-MM-DD", nargs="?", const="",
                         help="each hour of one night (default: tonight)")
    weather.set_defaults(func=run_weather)


# ================================================================ shared

def _parse_date(text, option="--date"):
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError):
        raise CliError(f"{option} needs a date as YYYY-MM-DD, not '{text}'.", EXIT_USAGE)


def _zone(site):
    from cosmoscollection.planning.Observer import local_zone
    return local_zone(site.timezone)


def _night_noon(site, location, night=None):
    """Local noon starting the night: of the given date, else tonight - the
    current night until its dawn, as the app's tools count it."""
    from cosmoscollection.planning.NightTimes import resolve_direction_times
    tz = _zone(site)
    if night is not None:
        return datetime(night.year, night.month, night.day, 12, tzinfo=tz)
    times, _note = resolve_direction_times("midnight", "", "", "", location, tz)
    return times[0][1] - timedelta(hours=12)


def _location(site):
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    return EarthLocation(lat=site.lat * u.deg, lon=site.lon * u.deg)


def _horizon(site, ignore):
    if ignore or not site.has_horizon:
        return None
    from cosmoscollection.planning.HorizonProfile import load_horizon_for_site
    return load_horizon_for_site(site)


def _clock(when):
    """A time as the app shows it (Settings > 12/24-hour)."""
    from cosmoscollection.ui.TimeFormatHelper import format_time
    return format_time(when, strip_zero=True) if when else ""  # "7:30 PM"


def _iso(when):
    return when.isoformat(timespec="minutes") if when else None


def _night_title(noon):
    return f"night of {noon:%a %b} {noon.day}, {noon.year}"


def _direction(az):
    from cosmoscollection.planning.NightTimes import compass_direction
    return compass_direction(az)


# ================================================================ tonight

def _time_choice(text):
    """--time as Best DSO Tonight's (mode, at, from, to)."""
    from cosmoscollection.planning.BestTonightEngine import TIME_DEFAULT
    word = (text or "night").strip().lower()
    _mode, at, start, end = TIME_DEFAULT
    if word in ("night", "tonight", "dusk", "midnight", "dawn", "now"):
        return ("night" if word == "tonight" else word, at, start, end)

    def clock(value):
        match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", value.strip())
        if not match or int(match.group(1)) > 23 or int(match.group(2) or 0) > 59:
            raise CliError(f"'{value}' isn't a time - use HH:MM, e.g. 23:30.", EXIT_USAGE)
        return f"{int(match.group(1)):02d}:{int(match.group(2) or 0):02d}"

    if "-" in word:
        first, last = word.split("-", 1)
        return ("frame", at, clock(first), clock(last))
    return ("time", clock(word), start, end)


def _catalog_choice(name):
    if not name:
        return []
    from cosmoscollection.planning.BestTonightEngine import _get_available_catalogs
    catalogs = [c for c in _get_available_catalogs() if c != "NAME"]
    found = next((c for c in catalogs if c.lower() == name.strip().lower()), None)
    if found is None:
        raise CliError(f"Unknown catalog '{name}'.", EXIT_USAGE, hint="Catalogs: " + ", ".join(catalogs))
    return [found]


def _type_choice(text):
    if not text:
        return []
    from cosmoscollection.catalog.DSONames import DSO_TYPES, type_code
    code = type_code(text)
    if code is None:
        raise CliError(f"Unknown object type '{text}'.", EXIT_USAGE,
                       hint="Types: " + ", ".join(f"{c} ({n})" for c, n in DSO_TYPES))
    return [code]


def _tonight_json(result, targets):
    from cosmoscollection.catalog.DSONames import common_name_from_designations, type_name
    info = result["dso_info"]
    magnitude = info["magnitude"]
    data = {
        "name": info["name"],
        "common_name": common_name_from_designations(info.get("designations")),
        "designations": [d.strip() for d in (info.get("designations") or "").split(",") if d.strip()],
        "id": info.get("id"),
        "type": info["type"], "type_name": type_name(info["type"]),
        "constellation": (info["constellation"] or "").upper() or None,
        "magnitude": magnitude,  # None when unknown
        "ra_deg": info["ra_deg"], "dec_deg": info["dec_deg"],
        "max_altitude": round(float(result["max_altitude"]), 1),
        "best_time": _iso(result["optimal_time"]),
        "best_altitude": round(float(result["optimal_altitude"]), 1),
        "best_azimuth": round(float(result["optimal_azimuth"]), 1),
        "direction": result.get("direction"),
        "visible_hours": None if result["visible_hours"] is None else round(result["visible_hours"], 2),
        "moon_separation": None if result.get("moon_separation") is None else round(result["moon_separation"], 1),
    }
    if targets:
        data["telescope"] = info.get("telescope_name")
    return data


def _sort_tonight(results, sort):
    """As the window sorts (visible hours, most first), or by --sort; the
    engine's rank breaks ties."""
    if sort == "rank":
        return results
    keys = {
        "hours": lambda r: -(r["visible_hours"] or 0),
        "altitude": lambda r: -r["max_altitude"],
        "magnitude": lambda r: (r["dso_info"]["magnitude"] is None, r["dso_info"]["magnitude"] or 0),
        "time": lambda r: r["optimal_time"],
        "moon": lambda r: -(r.get("moon_separation") if r.get("moon_separation") is not None else -1),
    }
    return sorted(results, key=keys[sort])


def run_tonight(ctx):
    from cosmoscollection.cli.LibraryCommands import Progress, _run_stoppable
    from cosmoscollection.planning.BestTonightEngine import (BestTonightCalculation, TonightError,
                                                              describe_time_window)

    args = ctx.args
    site = ctx.site()
    choice = _time_choice(args.time)
    now = None
    if args.date:
        if choice[0] == "now":
            raise CliError("--time now can't be used with --date.", EXIT_USAGE)
        night = _parse_date(args.date)
        now = datetime(night.year, night.month, night.day, 12, tzinfo=_zone(site))  # That night, as seen at noon
    if args.hide_completed and not args.targets:
        raise CliError("--hide-completed goes with --targets.", EXIT_USAGE)
    if args.targets and (args.catalog or args.type):
        raise CliError("--catalog and --type don't apply to --targets.", EXIT_USAGE)
    if args.objects < 1:
        raise CliError("--objects needs a number above 0.", EXIT_USAGE)

    calculation = BestTonightCalculation(
        min_altitude=args.min_alt, max_magnitude=args.max_mag, selected_catalogs=_catalog_choice(args.catalog),
        dso_limit=args.objects, selected_dso_types=_type_choice(args.type), use_target_list=args.targets,
        time_choice=choice, use_horizon=not args.no_horizon, hide_completed=args.hide_completed, site=site, now=now)

    progress = Progress(ctx)
    cancel = threading.Event()
    what = "your Target List" if args.targets else count_text(args.objects, "object")
    progress.show(f"Checking {what}...")
    try:
        results = _run_stoppable(
            lambda: calculation.calculate(progress=lambda pct: progress.show(f"Checking {what}... {pct}%"),
                                          cancelled=cancel.is_set),
            cancel, progress)
    except TonightError as e:
        raise CliError(str(e))
    finally:
        progress.done()

    results = _sort_tonight(results, args.sort)
    shown = results[:args.limit] if args.limit and args.limit > 0 else results
    when = describe_time_window(calculation.time_mode, calculation.window_start, calculation.window_end)
    if calculation.whole_window:
        when = f"for all of {when}"
    if now is not None:
        when += f" on the {_night_title(now)}"
    moon = results[0].get("moon_illumination") if results else None
    horizon_name = calculation.horizon.name if calculation.horizon is not None else None

    if ctx.json:
        print_json({
            "site": site.to_dict(),
            "time": {"mode": calculation.time_mode, "start": _iso(calculation.window_start),
                     "end": _iso(calculation.window_end), "whole_window": calculation.whole_window,
                     "note": calculation.window_note or None},
            "min_altitude": args.min_alt, "custom_horizon": horizon_name,
            "checked": "targets" if args.targets else args.objects,
            "found": len(results), "blocked_by_horizon": calculation.blocked_by_horizon,
            "moon_illumination": None if moon is None else round(moon, 3),
            "objects": [_tonight_json(r, args.targets) for r in shown],
        })
        return

    blocked = calculation.blocked_by_horizon
    print(f"Best DSOs {when} from {site.name}")
    details = [f"above {args.min_alt:g}°"]
    if horizon_name is not None:
        details.append(f"custom horizon '{horizon_name}'" if horizon_name else "custom horizon")
    if moon is not None:
        details.append(f"Moon {moon:.0%} lit")
    print(", ".join(details))
    if calculation.window_note:
        note(f"Note: {calculation.window_note}.")
    print()
    if not results:
        print(f"No objects meet the visibility criteria {when}" +
              (f" ({blocked} blocked by your custom horizon)." if blocked else "."))
        return

    from cosmoscollection.catalog.DSONames import common_name_from_designations, type_name
    columns = [
        Column("Object", lambda r: r["dso_info"]["name"]),
        Column("Common name", lambda r: common_name_from_designations(r["dso_info"].get("designations")) or "",
               flex=True),
        Column("Type", lambda r: type_name(r["dso_info"]["type"]), flex=True),
        Column("Const", lambda r: (r["dso_info"]["constellation"] or "").upper()),
        Column("Mag", lambda r: format_magnitude(r["dso_info"]["magnitude"]), ">"),
        Column("Max alt", lambda r: f"{r['max_altitude']:.0f}°", ">"),
        Column("Best time", lambda r: _clock(r["optimal_time"]), ">"),
        Column("Direction", lambda r: r.get("direction") or ""),
        Column("Hours", lambda r: "" if r["visible_hours"] is None else f"{r['visible_hours']:.1f}", ">"),
        Column("Moon", lambda r: "" if r.get("moon_separation") is None else f"{r['moon_separation']:.0f}°", ">"),
    ]
    if args.targets:
        columns.insert(-1, Column("Telescope", lambda r: r["dso_info"].get("telescope_name") or ""))
    print_table(shown, columns)
    more = len(results) - len(shown)
    print(f"\n{count_text(len(results), 'object')} up" + (f", {blocked} more blocked by your custom horizon" if blocked
                                                     else "") +
          (f" ({more} not shown - use --limit 0 for all)" if more else ""))


# ================================================================ visibility

def _visibility_target(conn, args):
    """(label, ra_deg, dec_deg, catalog object dict or None)."""
    from cosmoscollection.cli.CollectionCommands import _label, resolve_dso
    name = " ".join(args.name).strip()
    if (args.ra is None) != (args.dec is None):
        raise CliError("--ra and --dec go together.", EXIT_USAGE)
    if args.ra is not None:
        if not (0 <= args.ra < 360 and -90 <= args.dec <= 90):
            raise CliError("--ra is 0 to 360 degrees and --dec -90 to 90.", EXIT_USAGE)
        return name or f"RA {args.ra:.4f}°, Dec {args.dec:+.4f}°", args.ra, args.dec, None
    if not name:
        raise CliError("Name an object, or give --ra and --dec.", EXIT_USAGE)
    try:
        dso = resolve_dso(conn, name)
    except CliError as missing:
        # A Target List entry of its own (e.g. a custom object or mosaic panel)
        from cosmoscollection.planning.TargetRows import find_existing_target
        target = find_existing_target(conn, name)
        if target and target.get("ra_deg") is not None and (target["ra_deg"] or target["dec_deg"]):
            return target["name"], target["ra_deg"], target["dec_deg"], None
        raise missing
    if dso.get("ra_deg") is None or dso.get("dec_deg") is None:
        raise CliError(f"{_label(dso)} has no position in the catalog.", hint="Give --ra and --dec instead.")
    return _label(dso), dso["ra_deg"], dso["dec_deg"], dso


def _window_json(window):
    return {"start": _iso(window["start"]), "end": _iso(window["end"]), "hours": round(window["hours"], 2),
            "start_azimuth": round(window["start_azimuth"]), "end_azimuth": round(window["end_azimuth"]),
            "max_altitude": round(window["max_altitude"], 1)}


def run_visibility(ctx):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from cosmoscollection.core.Formatting import format_dec, format_ra
    from cosmoscollection.planning.Visibility import DSOVisibilityCalculator, night_summary

    args = ctx.args
    site = ctx.site()
    with ctx.connection() as conn:
        label, ra, dec, dso = _visibility_target(conn, args)
    location = _location(site)
    noon = _night_noon(site, location, _parse_date(args.date) if args.date else None)
    horizon = _horizon(site, args.no_horizon)
    calculator = DSOVisibilityCalculator(site.lat, site.lon, site.timezone or "UTC")
    summary = night_summary(calculator, SkyCoord(ra=ra * u.deg, dec=dec * u.deg), noon, args.min_alt, horizon)

    if ctx.json:
        transit = summary["transit"]
        highest = summary["highest_in_dark"]
        data = {
            "object": label, "id": dso["dsodetailid"] if dso else None, "ra_deg": ra, "dec_deg": dec,
            "site": site.to_dict(), "night": noon.date().isoformat(), "min_altitude": args.min_alt,
            "custom_horizon": (horizon.name or "custom") if horizon is not None else None,
            "sun": {key: _iso(summary[key]) for key in ("sunset", "dark_sky_start", "dark_start", "dark_end",
                                                        "dark_sky_end", "sunrise")},
            "rise": _iso(summary["rise"]), "set": _iso(summary["set"]),
            "always_up": summary["always_up"], "never_up": summary["never_up"],
            "transit": {"time": _iso(transit["time"]), "altitude": round(transit["altitude"], 1)} if transit else None,
            "highest_in_dark": {"time": _iso(highest["time"]), "altitude": round(highest["altitude"], 1),
                                "azimuth": round(highest["azimuth"])} if highest else None,
            "hours_visible": round(summary["hours_visible"], 2),
            "windows": [_window_json(w) for w in summary["windows"]],
            "moon": {"illumination": round(summary["moon_illumination"], 3),
                     "separation": None if summary["moon_separation"] is None else round(summary["moon_separation"], 1),
                     "separation_at": _iso(summary["moon_separation_at"]),
                     "rise": _iso(summary["moon_rise"]), "set": _iso(summary["moon_set"])},
        }
        if args.hourly:
            data["hourly"] = [dict(hour, time=_iso(hour["time"]), altitude=round(hour["altitude"], 1),
                                   azimuth=round(hour["azimuth"], 1), sun_altitude=round(hour["sun_altitude"], 1),
                                   moon_altitude=round(hour["moon_altitude"], 1)) for hour in summary["hourly"]]
        print_json(data)
        return

    print(f"{label} from {site.name}, {_night_title(noon)}")
    print(f"RA {format_ra(ra)}  Dec {format_dec(dec)}")
    print()

    def sun_line():
        parts = []
        if summary["sunset"]:
            parts.append(f"sets {_clock(summary['sunset'])}")
        if summary["dark_start"] and summary["dark_end"]:
            parts.append(f"dark {_clock(summary['dark_start'])} - {_clock(summary['dark_end'])}")
        else:
            parts.append("no astronomical darkness")
        if summary["sunrise"]:
            parts.append(f"rises {_clock(summary['sunrise'])}")
        return ", ".join(parts)

    def object_line():
        if summary["always_up"]:
            parts = ["above the horizon all night"]
        elif summary["never_up"]:
            parts = ["below the horizon all night"]
        else:
            parts = []
            if summary["rise"]:
                parts.append(f"rises {_clock(summary['rise'])}")
            if summary["set"]:
                parts.append(f"sets {_clock(summary['set'])}")
        return ", ".join(parts)

    transit = summary["transit"]
    highest = summary["highest_in_dark"]
    moon_sep = summary["moon_separation"]
    print_fields([
        ("Sun", sun_line()),
        ("Object", object_line()),
        ("Meridian", f"{_clock(transit['time'])} at {transit['altitude']:.0f}° (meridian flip)" if transit else None),
        ("Highest (dark)", f"{highest['altitude']:.0f}° {_direction(highest['azimuth'])} at {_clock(highest['time'])}"
         if highest else "the sky doesn't get dark (sun below -12°)"),
        ("Moon", f"{summary['moon_illumination']:.0%} lit" +
         (f", {moon_sep:.0f}° away at {_clock(summary['moon_separation_at'])}" if moon_sep is not None else "")),
        ("Custom horizon", (horizon.name or "yes") if horizon is not None else None),
    ])
    print()
    windows = summary["windows"]
    limit = f"above {args.min_alt:g}°" + (" and your custom horizon" if horizon is not None else "")
    if windows:
        print(f"Up {limit} in a dark sky: {summary['hours_visible']:.1f} hours")
        print_table(windows, [
            Column("From", lambda w: _clock(w["start"]), ">"),
            Column("To", lambda w: _clock(w["end"]), ">"),
            Column("Hours", lambda w: f"{w['hours']:.1f}", ">"),
            Column("Direction", lambda w: f"{_direction(w['start_azimuth'])} → {_direction(w['end_azimuth'])}"),
            Column("Max alt", lambda w: f"{w['max_altitude']:.0f}°", ">"),
        ], indent="  ")
    else:
        print(f"Not up {limit} in a dark sky this night.")
        print("Try a lower --min-alt or another --date.")

    if args.hourly:
        hours = [h for h in summary["hourly"] if h["sun_altitude"] < 0]
        print()
        print("Each hour of the night")
        print_table(hours, [
            Column("Time", lambda h: _clock(h["time"]), ">"),
            Column("Alt", lambda h: f"{h['altitude']:.0f}°", ">"),
            Column("Az", lambda h: f"{h['azimuth']:.0f}°", ">"),
            Column("Dir", lambda h: _direction(h["azimuth"])),
            Column("Sun", lambda h: f"{h['sun_altitude']:.0f}°", ">"),
            Column("Moon", lambda h: f"{h['moon_altitude']:.0f}°", ">"),
            Column("Up", lambda h: "yes" if h["visible"] else ("behind horizon" if h["below_horizon"] and
                                                                h["altitude"] > 0 else "")),
        ], indent="  ")


# ================================================================ weather

def _temperature(celsius, unit):
    from cosmoscollection.planning.WeatherFetch import convert_temperature
    return f"{convert_temperature(celsius, unit):.0f}°{'F' if unit == 'Fahrenheit' else 'C'}"


def _wind(kmh, unit):
    from cosmoscollection.planning.WeatherFetch import convert_wind
    return f"{convert_wind(kmh, unit):.0f} {unit}"


SOURCE_NAMES = {"openmeteo": "Open-Meteo", "openweather": "OpenWeather", "visualcrossing": "VisualCrossing",
                "weatherapi": "WeatherAPI"}


def _sources(fetcher, summaries):
    sources = [fetcher.baseline_source]
    for name in ("openweather", "visualcrossing", "weatherapi"):
        if name != fetcher.baseline_source and any(getattr(s, f"{name}_blended") for s in summaries):
            sources.append(name)
    return sources


def _hour_score(hour):
    from cosmoscollection.planning.WeatherFetch import calculate_astro_score
    return calculate_astro_score(hour.cloud_cover, hour.humidity, hour.wind_speed, hour.precipitation_probability,
                                 hour.visibility, hour.wind_gusts)


def run_weather(ctx):
    from cosmoscollection.cli.LibraryCommands import Progress
    from cosmoscollection.planning.WeatherFetch import (WeatherFetch, WeatherUnavailable, calculate_sun_altitudes,
                                                        get_rating_label, night_hours, weather_units)
    from cosmoscollection.planning.NightTimes import DARK_SKY_ALTITUDE

    args = ctx.args
    if args.days < 1:
        raise CliError("--days needs a number above 0.", EXIT_USAGE)
    site = ctx.site()
    ctx.require_online("The weather forecast")
    progress = Progress(ctx)
    fetcher = WeatherFetch(site.lat, site.lon, site.timezone, progress=progress.show)
    try:
        summaries = fetcher.fetch()
    except WeatherUnavailable as e:
        raise CliError(str(e))
    finally:
        progress.done()

    errors = {name: getattr(fetcher, f"{name}_error") for name in SOURCE_NAMES if getattr(fetcher, f"{name}_error")}
    sources = _sources(fetcher, summaries)
    temp_unit, wind_unit = weather_units()

    hourly_index = None
    if args.hourly is not None:
        wanted = _parse_date(args.hourly, "--hourly") if args.hourly else _night_noon(site, _location(site)).date()
        hourly_index = next((i for i, s in enumerate(summaries) if s.date.date() == wanted), None)
        if hourly_index is None:
            first, last = summaries[0].date.date(), summaries[-1].date.date()
            raise CliError(f"No forecast for {wanted.isoformat()}.", hint=f"The forecast covers {first} to {last}.")

    nights = summaries[:args.days]

    def night_json(summary):
        moon = summary.moon_phase
        return {
            "date": summary.date.date().isoformat(), "astro_score": summary.astro_score,
            "rating": get_rating_label(summary.astro_score), "seeing": summary.seeing_estimate,
            "dark_cloud_cover": round(summary.tonight_avg_cloud_cover), "cloud_cover": round(summary.avg_cloud_cover),
            "precipitation_probability": round(summary.avg_precipitation_prob),
            "wind_kmh": round(summary.avg_wind_speed, 1), "max_wind_kmh": round(summary.max_wind_speed, 1),
            "temperature_c": {"min": round(summary.min_temperature, 1), "max": round(summary.max_temperature, 1)},
            "humidity": round(summary.avg_humidity),
            "dark_hours": {"start": _iso(summary.dark_hours_start), "end": _iso(summary.dark_hours_end)}
            if summary.dark_hours_start else None,
            "moon": {"phase": moon.phase_name, "illumination": round(moon.illumination)} if moon else None,
        }

    hours = night_hours(summaries[hourly_index].hourly_data,
                        summaries[hourly_index + 1].hourly_data if hourly_index + 1 < len(summaries) else None) \
        if hourly_index is not None else []
    sun = calculate_sun_altitudes(site.lat, site.lon, [h.time for h in hours], site.timezone) if hours else []

    if ctx.json:
        data = {"site": site.to_dict(), "sources": [SOURCE_NAMES[s] for s in sources],
                "unavailable": {SOURCE_NAMES[k]: v for k, v in errors.items()},
                "nights": [night_json(s) for s in nights]}
        if hourly_index is not None:
            data["hourly"] = {"date": summaries[hourly_index].date.date().isoformat(), "hours": [
                {"time": h.time.isoformat(timespec="minutes"), "astro_score": _hour_score(h),
                 "dark": altitude < DARK_SKY_ALTITUDE, "cloud_cover": round(h.cloud_cover), "cloud_low": round(h.cloud_cover_low),
                 "cloud_mid": round(h.cloud_cover_mid), "cloud_high": round(h.cloud_cover_high),
                 "humidity": round(h.humidity), "wind_kmh": round(h.wind_speed, 1),
                 "gusts_kmh": round(h.wind_gusts, 1), "precipitation_probability": round(h.precipitation_probability),
                 "temperature_c": round(h.temperature, 1), "dew_point_c": round(h.dew_point, 1)}
                for h, altitude in zip(hours, sun)]}
        print_json(data)
        return

    print(f"Weather for {site.name} - {', '.join(SOURCE_NAMES[s] for s in sources)}")
    for name, problem in errors.items():
        note(f"Note: {SOURCE_NAMES[name]} unavailable ({problem}).")
    print()
    if hourly_index is None:
        print_table(nights, [
            Column("Night of", lambda s: f"{s.date:%a %b} {s.date.day}"),
            Column("Score", lambda s: s.astro_score, ">"),
            Column("Rating", lambda s: get_rating_label(s.astro_score)),
            Column("Clouds", lambda s: f"{s.tonight_avg_cloud_cover:.0f}%", ">"),
            Column("Seeing", lambda s: s.seeing_estimate),
            Column("Moon", lambda s: f"{s.moon_phase.phase_name} {s.moon_phase.illumination:.0f}%"
                   if s.moon_phase else "", flex=True),
            Column("Dark", lambda s: f"{_clock(s.dark_hours_start)} - {_clock(s.dark_hours_end)}"
                   if s.dark_hours_start else "none", ">"),
            Column("Rain", lambda s: f"{s.avg_precipitation_prob:.0f}%", ">"),
            Column("Wind", lambda s: _wind(s.avg_wind_speed, wind_unit), ">"),
            Column("Temp", lambda s: f"{_temperature(s.min_temperature, temp_unit)} to "
                                     f"{_temperature(s.max_temperature, temp_unit)}", ">"),
        ])
        print("\nScore and clouds are for the night's dark hours (sun below -12°). "
              "See one night's hours with --hourly DATE.")
        return

    summary = summaries[hourly_index]
    title = _night_title(summary.date)
    print(f"{title[0].upper()}{title[1:]}: score {summary.astro_score} ({get_rating_label(summary.astro_score)}), "
          f"seeing {summary.seeing_estimate}")
    print()
    rows = list(zip(hours, sun))
    print_table(rows, [
        Column("Time", lambda r: _clock(r[0].time), ">"),
        Column("Dark", lambda r: "dark" if r[1] < DARK_SKY_ALTITUDE else ""),
        Column("Score", lambda r: _hour_score(r[0]), ">"),
        Column("Clouds", lambda r: f"{r[0].cloud_cover:.0f}%", ">"),
        Column("Low/mid/high", lambda r: f"{r[0].cloud_cover_low:.0f}/{r[0].cloud_cover_mid:.0f}/"
                                         f"{r[0].cloud_cover_high:.0f}", ">"),
        Column("Humidity", lambda r: f"{r[0].humidity:.0f}%", ">"),
        Column("Wind", lambda r: _wind(r[0].wind_speed, wind_unit), ">"),
        Column("Gusts", lambda r: _wind(r[0].wind_gusts, wind_unit), ">"),
        Column("Rain", lambda r: f"{r[0].precipitation_probability:.0f}%", ">"),
        Column("Temp", lambda r: _temperature(r[0].temperature, temp_unit), ">"),
        Column("Dew pt", lambda r: _temperature(r[0].dew_point, temp_unit), ">"),
    ])
    if hourly_index + 1 >= len(summaries):
        print("\nThe forecast ends at midnight, so this night's morning hours aren't in it.")
