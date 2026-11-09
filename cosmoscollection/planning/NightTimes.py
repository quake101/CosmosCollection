"""
Night times for the planning tools: astronomical dusk and dawn, the times a
"Direction at:" / "Time:" choice means (Now, Dusk, Midnight, Dawn, Tonight,
a custom time or time frame), and where objects are in the sky then.

Shared by the Target List's Direction column, Best DSO Tonight and the
command line. No UI imports.
"""

from datetime import datetime

from cosmoscollection.planning.Observer import local_zone


# (key, label) for the "Direction at:" / "Time:" dropdowns
DIRECTION_MODES = (
    ("now", "Now"),
    ("dusk", "Dusk"),
    ("midnight", "Midnight"),
    ("dawn", "Dawn"),
    ("night", "Tonight (dusk → dawn)"),
    ("time", "Custom time"),
    ("frame", "Custom time frame"),
)


def load_night_choice(key, default):
    """(mode, at, from, to) of a "Direction at:" / "Time:" choice saved under a
    settings key as "mode|at|from|to" (times HH:MM), or default."""
    from cosmoscollection.core.Settings import app_settings
    parts = (app_settings().value(key, "", type=str) or "").split("|")
    if len(parts) != 4 or parts[0] not in dict(DIRECTION_MODES):
        return tuple(default)
    return tuple(parts)


def save_night_choice(key, choice):
    """Save a (mode, at, from, to) choice for load_night_choice."""
    from cosmoscollection.core.Settings import app_settings
    app_settings().setValue(key, "|".join(choice))

# The sun's altitude (degrees) at each stage of the night
SUNSET_ALTITUDE = -0.833        # The sun's top at the horizon, with refraction
CIVIL_TWILIGHT_ALTITUDE = -6.0
DARK_SKY_ALTITUDE = -12.0       # Sun below this: a dark sky (viewing windows, dark hours - everywhere in the app)
ASTRONOMICAL_DARK_ALTITUDE = -18.0
# (key, name, sun above) from day to night; the keys are Theme.sky_shade()'s
SKY_STATES = (
    ("day", "Daylight", SUNSET_ALTITUDE),
    ("civil", "Civil twilight", CIVIL_TWILIGHT_ALTITUDE),
    ("nautical", "Nautical twilight", DARK_SKY_ALTITUDE),
    ("astronomical", "Astronomical twilight", ASTRONOMICAL_DARK_ALTITUDE),
    ("night", "Night", -90.0),
)

_COMPASS_POINTS = ('N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW')


def sky_state(sun_alt):
    """(key, name) of SKY_STATES for the sun's altitude: ('civil', 'Civil twilight')."""
    for key, name, above in SKY_STATES:
        if sun_alt > above:
            return key, name
    return SKY_STATES[-1][:2]


def compass_direction(az):
    """'NE' etc. for an azimuth in degrees."""
    return _COMPASS_POINTS[int((az + 11.25) / 22.5) % 16]


def _twilight(location, noon_local, tz):
    """(dusk, dawn) local datetimes of astronomical twilight in the night
    following noon_local, and whether it gets that dark at all (if not, both
    are the darkest moment)."""
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import AltAz, get_sun
    from astropy.time import Time

    times = Time(noon_local) + np.arange(0, 24 * 12 + 1) * 5 * u.min  # every 5 minutes
    sun_alt = get_sun(times).transform_to(AltAz(obstime=times, location=location)).alt.deg
    dark = sun_alt < ASTRONOMICAL_DARK_ALTITUDE
    if dark.any():
        first = int(np.argmax(dark))
        last = len(dark) - 1 - int(np.argmax(dark[::-1]))
        reaches_dark = True
    else:
        first = last = int(np.argmin(sun_alt))
        reaches_dark = False
    to_local = lambda t: t.to_datetime(timezone=tz)
    return to_local(times[first]), to_local(times[last]), reaches_dark


def resolve_direction_times(mode, at, start, end, location, tz, now=None):
    """[(label, local datetime)] the Direction column is for - one time, or a
    frame's start and end - and a note (e.g. no astronomical darkness).

    "Tonight" is the current night until its dawn, then the coming one, so
    during the day the presets plan ahead and after midnight they still mean
    the night you're in.
    """
    from datetime import timedelta
    now = now or datetime.now(tz)
    if mode == "now":
        return [("Now", now)], ""

    today_noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    noon = today_noon
    if now < today_noon:
        previous = today_noon - timedelta(days=1)
        _dusk, dawn, _dark = _twilight(location, previous, tz)
        if now < dawn:
            noon = previous  # Still in last night (before its dawn)

    def on_night(hhmm):
        hours, minutes = (int(x) for x in hhmm.split(":"))
        when = noon.replace(hour=hours, minute=minutes)
        return when if when >= noon else when + timedelta(days=1)

    note = ""
    if mode in ("dusk", "dawn", "night"):
        dusk, dawn, dark = _twilight(location, noon, tz)
        if not dark:
            note = "the sky doesn't get astronomically dark tonight - using the darkest moment"
        if mode == "dusk":
            return [("Dusk", dusk)], note
        if mode == "dawn":
            return [("Dawn", dawn)], note
        return [("Dusk", dusk), ("Dawn", dawn)], note
    if mode == "midnight":
        return [("Midnight", noon + timedelta(hours=12))], ""
    if mode == "time":
        return [("At", on_night(at))], ""
    first, last = on_night(start), on_night(end)
    if last <= first:
        last += timedelta(days=1)
    return [("From", first), ("To", last)], ""


def direction_cells(coords, times, location, horizon):
    """Direction column (text, tooltip, below_throughout) for each of coords
    (a SkyCoord array) at times [(label, local datetime)] - one time, or a
    frame's start and end ("E → SE"). Shared with Best DSO Tonight.
    """
    from astropy.coordinates import AltAz
    from astropy.time import Time

    positions = []  # per time: (az array, alt array)
    for _label, when in times:
        altaz = coords.transform_to(AltAz(obstime=Time(when), location=location))
        positions.append((altaz.az.deg, altaz.alt.deg))

    horizon_word = "your horizon" if horizon is not None else "the horizon"
    cells = []
    for i in range(len(coords)):
        parts, tips, below_flags = [], [], []
        for (label, when), (az, alt) in zip(times, positions):
            limit = float(horizon.altitude_at(az[i])) if horizon is not None else 0.0
            below = alt[i] < limit
            below_flags.append(below)
            direction = compass_direction(az[i])
            parts.append(f"{direction} (below)" if below else direction)
            tip = f"{label} ({when:%H:%M}): {direction}, azimuth {az[i]:.0f}°, altitude {alt[i]:.0f}°"
            tips.append(tip + (f" - below {horizon_word}" if below else ""))
        if len(parts) > 1 and all(below_flags):
            # Below throughout: say so once ("E → ESE (below)")
            parts = [compass_direction(az[i]) for az, _alt in positions]
            parts[-1] += " (below)"
        cells.append((" → ".join(parts), "\n".join(tips), all(below_flags)))
    return cells


def compute_directions(targets, lat, lon, tz_name, horizon, mode, at, start, end):
    """Direction column contents for every target.

    targets: [(id, ra_deg, dec_deg)] with real coordinates.
    Returns {'times': [(label, local datetime)], 'note': str,
             'cells': {id: (text, tooltip, below_throughout)}}.
    """
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord

    tz = local_zone(tz_name)
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    times, note = resolve_direction_times(mode, at, start, end, location, tz)
    if not targets:
        return {"times": times, "note": note, "cells": {}}

    coords = SkyCoord(ra=np.array([t[1] for t in targets]) * u.deg,
                      dec=np.array([t[2] for t in targets]) * u.deg)
    cells = direction_cells(coords, times, location, horizon)
    return {"times": times, "note": note, "cells": {t[0]: cell for t, cell in zip(targets, cells)}}
