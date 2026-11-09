"""
Visibility calculations for the planning tools: where an object is over a
night, when it's above a minimum altitude (and the custom horizon) in a dark
sky, meridian crossings, the Moon's phase and distance.

Used by the DSO Visibility Calculator, Best DSO Tonight, the Target List,
Session Manager and the command line. No UI imports.
"""

import logging
import warnings

import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

from cosmoscollection.core.DatabaseManager import DatabaseManager, open_connection
from cosmoscollection.planning.NightTimes import ASTRONOMICAL_DARK_ALTITUDE, DARK_SKY_ALTITUDE, SUNSET_ALTITUDE
from cosmoscollection.planning.Observer import active_site

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DSOVisibilityCalculator:
    """
    Centralized class for all DSO visibility calculations.

    This class provides a single interface for calculating DSO visibility, altitude,
    azimuth, optimal viewing times, and seasonal visibility across the application.
    """

    def __init__(self, location_lat=None, location_lon=None, timezone=None, height=250):
        """
        Initialize the visibility calculator.

        Args:
            location_lat (float): Observer latitude in degrees (+ for North, - for South)
            location_lon (float): Observer longitude in degrees (+ for East, - for West)
            timezone (str): Timezone string (e.g., 'America/New_York')
            height (float): Observer height above sea level in meters (default: 250)
        """
        self.location = None
        self.timezone = pytz.UTC  # Default to UTC
        # Custom horizon of the active location (HorizonProfile or None). Only
        # loaded when the location comes from the database; callers opt in by
        # passing it to find_optimal_viewing_times(horizon=...)
        self.horizon = None

        site = None
        if location_lat is not None and location_lon is not None:
            self.set_location(location_lat, location_lon, height)
        else:
            site = self._saved_site()
            if site is not None:
                self.set_location(site.lat, site.lon, height)
                from cosmoscollection.planning.HorizonProfile import load_active_horizon
                self.horizon = load_active_horizon()

        if timezone:
            self.set_timezone(timezone)
        else:
            site = site or self._saved_site()
            if site is not None and site.timezone:
                self.set_timezone(site.timezone)

    def set_location(self, lat, lon, height=250):
        """Set observer location."""
        self.location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)

    def set_timezone(self, timezone_str):
        """Set timezone for local time calculations."""
        try:
            self.timezone = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            self.timezone = pytz.UTC

    @staticmethod
    def _saved_site():
        """The active saved location (Observer.active_site), or None."""
        try:
            with DatabaseManager().get_connection() as conn:
                return active_site(conn)
        except Exception as e:
            logger.debug(f"No saved location: {e}")
            return None

    def get_dso_coordinates(self, dso_name):
        """
        Get coordinates for a DSO by name.

        Args:
            dso_name (str): Name of the DSO (e.g., 'M31', 'NGC 7000')

        Returns:
            tuple: (SkyCoord object, error_message) - error_message is None if successful
        """
        try:
            coord = SkyCoord.from_name(dso_name)
            return coord, None
        except Exception as e:
            return None, str(e)

    def get_dso_coordinates_enhanced(self, dso_name):
        """
        Get coordinates for a DSO by name with enhanced name resolution.

        Looks in the app's own catalog first, then online by name and several
        variations of it.

        Args:
            dso_name (str): Name of the DSO (e.g., 'M31', 'NGC 7000', 'sh2 142')

        Returns:
            tuple: (SkyCoord object, error_message) - error_message is None if successful
        """
        # The app's own catalog: offline, and knows designations written any way
        # ("Sh2 129 (Squid Nebula)", "sh2-129") and common names
        try:
            from cosmoscollection.catalog.DSONames import catalog_ids
            # A connection of its own: this runs in worker threads, and the
            # DatabaseManager singleton's connection only works in the thread that made it
            conn = open_connection()
            try:
                ids = catalog_ids(conn, dso_name)
                if len(ids) == 1:
                    row = conn.execute("SELECT ra, dec FROM dsodetail WHERE id = ?", (next(iter(ids)),)).fetchone()
                    if row and row[0] is not None and row[1] is not None:
                        return SkyCoord(ra=row[0] * u.deg, dec=row[1] * u.deg), None
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Catalog lookup failed for {dso_name!r}: {e}")

        # Then online, as written
        coord, error = self.get_dso_coordinates(dso_name)
        if coord is not None:
            return coord, None

        # Try various name formatting variations
        name_variations = [dso_name.strip()]
        original_name = dso_name.strip().upper()

        # Common variations for different naming patterns
        variations_to_try = []

        # Handle spaces vs hyphens (e.g., "sh2 142" vs "sh2-142")
        if ' ' in original_name:
            variations_to_try.append(original_name.replace(' ', '-'))
            variations_to_try.append(original_name.replace(' ', ''))
        if '-' in original_name:
            variations_to_try.append(original_name.replace('-', ' '))
            variations_to_try.append(original_name.replace('-', ''))

        # Handle common catalog prefixes
        catalog_mappings = {
            'SH2': 'SHARPLESS',
            'SHARPLESS': 'SH2',
            'SH': 'SHARPLESS',
            'IC': 'IC',
            'NGC': 'NGC',
            'M': 'MESSIER',
            'MESSIER': 'M',
            'LDN': 'LDN',
            'BARNARD': 'B',
            'B': 'BARNARD'
        }

        # Extract catalog prefix and number
        import re
        match = re.match(r'^([A-Z]+)[\s-]?(\d+)', original_name)
        if match:
            prefix, number = match.groups()

            # Try different catalog name formats
            for alt_prefix in catalog_mappings.get(prefix, [prefix]):
                if alt_prefix != prefix:
                    variations_to_try.extend([
                        f"{alt_prefix} {number}",
                        f"{alt_prefix}-{number}",
                        f"{alt_prefix}{number}"
                    ])

        # Try all variations
        for variation in variations_to_try:
            if variation not in name_variations:  # Avoid duplicates
                name_variations.append(variation)
                coord, _ = self.get_dso_coordinates(variation)
                if coord is not None:
                    return coord, None

        # If all attempts fail, return the original error with helpful message
        error_msg = f"Could not resolve coordinates for '{dso_name}'. Tried variations: {', '.join(name_variations[:5])}"
        if len(name_variations) > 5:
            error_msg += f" (and {len(name_variations)-5} more)"

        return None, error_msg

    def calculate_altaz_over_time(self, dso_coord, start_time, duration_hours, time_resolution=4):
        """
        Calculate altitude and azimuth for a DSO over a time period.

        Args:
            dso_coord (SkyCoord): DSO coordinates
            start_time (str or Time): Start time (ISO format string or astropy Time)
            duration_hours (float): Duration in hours
            time_resolution (int): Time points per hour (default: 4, i.e., every 15 minutes)

        Returns:
            tuple: (time_range, dso_altaz, sun_altaz)
        """
        if self.location is None:
            raise ValueError("Observer location not set")

        if isinstance(start_time, str):
            # Parse date string and interpret as midnight in LOCAL timezone, not UTC
            from datetime import datetime, timedelta
            date_parts = start_time.split('-')
            if len(date_parts) == 3:
                # Create datetime at midnight in local timezone
                year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                local_midnight = self.timezone.localize(datetime(year, month, day, 0, 0, 0))

                # Smart midnight selection: choose the midnight that keeps current time in range
                current_local = datetime.now(self.timezone)

                # Calculate where current time would be relative to this midnight
                hours_from_this_midnight = (current_local - local_midnight).total_seconds() / 3600

                # Choose the midnight that puts current time closest to the center of the range
                # For a 24-hour plot centered on midnight, we want current time between -12 and +12 hours
                half_duration = duration_hours / 2

                # If using this midnight would put current time way outside the range, use next midnight
                if hours_from_this_midnight > half_duration:
                    local_midnight += timedelta(days=1)

                # Shift back by half the duration to center on midnight
                local_start = local_midnight - timedelta(hours=duration_hours / 2)
                # Convert to UTC for astropy
                utc_start = local_start.astimezone(pytz.UTC)
                start_time = Time(utc_start)
            else:
                # Fallback to original behavior for non-date strings
                start_time = Time(start_time)

        # Create time range
        time_range = start_time + np.linspace(0, duration_hours, int(duration_hours * time_resolution)) * u.hour

        # Calculate DSO altitude/azimuth
        altaz_frame = AltAz(obstime=time_range, location=self.location)
        dso_altaz = dso_coord.transform_to(altaz_frame)

        # Calculate sun altitude/azimuth
        sun = get_sun(time_range)
        sun_altaz = sun.transform_to(altaz_frame)

        return time_range, dso_altaz, sun_altaz

    def find_optimal_viewing_times(self, dso_altaz, sun_altaz, min_altitude=30, max_sun_altitude=DARK_SKY_ALTITUDE,
                                   horizon=None):
        """
        Find optimal viewing times based on altitude and darkness criteria.

        Args:
            dso_altaz: DSO altitude/azimuth data
            sun_altaz: Sun altitude/azimuth data
            min_altitude (float): Minimum DSO altitude in degrees (default: 30)
            max_sun_altitude (float): Maximum sun altitude for dark sky (default: -12)
            horizon (HorizonProfile): Optional custom horizon. The DSO must be
                above both min_altitude and the horizon at its azimuth.

        Returns:
            numpy array: Boolean array indicating optimal viewing times
        """
        threshold = min_altitude
        if horizon is not None:
            threshold = np.maximum(min_altitude, horizon.altitude_at(dso_altaz.az.deg))
        dso_visible = dso_altaz.alt.deg > threshold
        dark_sky = sun_altaz.alt.deg < max_sun_altitude
        return dso_visible & dark_sky

    def find_meridian_crossings(self, time_range, dso_coord):
        """Times the object crosses the meridian heading west (hour angle 0, where a
        German equatorial mount flips), interpolated between the time steps.

        Returns:
            list: astropy Times, empty if it never crosses in the range
        """
        from astropy.coordinates import TETE
        try:
            # Apparent RA against apparent sidereal time, so precession since J2000
            # doesn't shift the crossing by a minute or two
            ra = dso_coord.transform_to(TETE(obstime=time_range[len(time_range) // 2])).ra.deg
            lst = time_range.sidereal_time('apparent', longitude=self.location.lon).deg
            hour_angle = (lst - ra + 180) % 360 - 180  # -180..180, 0 on the meridian
            # Negative to positive is the crossing; the +180 -> -180 wrap goes the other way
            steps = np.where((hour_angle[:-1] < 0) & (hour_angle[1:] >= 0))[0]
            return [time_range[i] + (time_range[i + 1] - time_range[i]) *
                    (-hour_angle[i] / (hour_angle[i + 1] - hour_angle[i])) for i in steps]
        except Exception as e:
            logger.debug(f"Could not find meridian crossings: {e}")
            return []

    def calculate_visibility_for_date(self, dso_name, date, duration_hours=24, min_altitude=30, horizon=None,
                                      time_resolution=4):
        """
        Calculate complete visibility information for a DSO on a specific date.

        Args:
            dso_name (str): Name of the DSO
            date (str): Date in ISO format (YYYY-MM-DD)
            duration_hours (float): Duration to calculate (default: 24 hours)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)
            horizon (HorizonProfile): Optional custom horizon (see find_optimal_viewing_times)
            time_resolution (int): Time points per hour (default: 4, every 15 minutes)

        Returns:
            dict: Complete visibility results or None if error
        """
        # Get DSO coordinates with enhanced name resolution
        dso_coord, error = self.get_dso_coordinates_enhanced(dso_name)
        if dso_coord is None:
            return {"error": f"Could not find coordinates for {dso_name}: {error}"}

        return self.calculate_visibility_for_coordinates(dso_coord, date, duration_hours, min_altitude, dso_name,
                                                         horizon=horizon, time_resolution=time_resolution)

    def calculate_visibility_for_coordinates(self, dso_coord, date, duration_hours=24, min_altitude=30, dso_name=None,
                                             horizon=None, time_resolution=4):
        """
        Calculate complete visibility information for DSO coordinates on a specific date.

        Args:
            dso_coord (SkyCoord): Coordinates of the DSO
            date (str): Date in ISO format (YYYY-MM-DD)
            duration_hours (float): Duration to calculate (default: 24 hours)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)
            dso_name (str, optional): Name of the DSO for display purposes
            horizon (HorizonProfile): Optional custom horizon (see find_optimal_viewing_times)
            time_resolution (int): Time points per hour (default: 4, every 15 minutes)

        Returns:
            dict: Complete visibility results or None if error
        """
        try:
            # Calculate altitude/azimuth over time
            time_range, dso_altaz, sun_altaz = self.calculate_altaz_over_time(
                dso_coord, date, duration_hours, time_resolution)

            # Find optimal viewing times
            optimal_times = self.find_optimal_viewing_times(dso_altaz, sun_altaz, min_altitude, horizon=horizon)

            # Calculate summary statistics
            max_altitude = np.max(dso_altaz.alt.deg)
            max_alt_idx = np.argmax(dso_altaz.alt.deg)
            max_alt_time = time_range[max_alt_idx]
            max_alt_azimuth = dso_altaz.az.deg[max_alt_idx]

            # Find viewing windows
            viewing_windows = self._find_viewing_windows(time_range, optimal_times, dso_altaz)

            return {
                "dso_name": dso_name or f"RA {dso_coord.ra.deg:.4f}° DEC {dso_coord.dec.deg:.4f}°",
                "dso_coord": dso_coord,
                "time_range": time_range,
                "dso_altaz": dso_altaz,
                "sun_altaz": sun_altaz,
                "optimal_times": optimal_times,
                "max_altitude": max_altitude,
                "max_alt_time": max_alt_time,
                "max_alt_azimuth": max_alt_azimuth,
                "viewing_windows": viewing_windows,
                "timezone": self.timezone
            }

        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}

    def calculate_visibility_hours_for_day(self, dso_coord, date, min_altitude=30, horizon=None):
        """
        Calculate total visibility hours for a DSO on a specific day.

        Args:
            dso_coord (SkyCoord): Coordinates of the DSO
            date (str): Date in ISO format (YYYY-MM-DD)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)
            horizon (HorizonProfile): Optional custom horizon (see find_optimal_viewing_times)

        Returns:
            float: Total hours the DSO is optimally visible on this day
        """
        try:
            # Calculate for full 24 hours
            time_range, dso_altaz, sun_altaz = self.calculate_altaz_over_time(
                dso_coord, date, 24, time_resolution=4)

            # Find optimal viewing times
            optimal_times = self.find_optimal_viewing_times(dso_altaz, sun_altaz, min_altitude, horizon=horizon)

            # Calculate total hours (each time point represents 15 minutes = 0.25 hours)
            total_hours = np.sum(optimal_times) * 0.25

            return total_hours

        except Exception:
            return 0.0

    def _find_viewing_windows(self, time_range, optimal_times, dso_altaz):
        """Find continuous viewing windows from optimal times array."""
        if not np.any(optimal_times):
            return []

        # Find continuous windows
        diff = np.diff(np.concatenate(([False], optimal_times, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        windows = []
        for start_idx, end_idx in zip(starts, ends):
            start_time = time_range[start_idx]
            end_time = time_range[end_idx - 1]
            duration = (end_time - start_time).to(u.hour).value

            # Calculate mid-window statistics
            mid_idx = (start_idx + end_idx) // 2
            mid_altitude = dso_altaz.alt.deg[mid_idx]
            mid_azimuth = dso_altaz.az.deg[mid_idx]

            windows.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": duration,
                "mid_altitude": mid_altitude,
                "mid_azimuth": mid_azimuth
            })

        return windows

    @staticmethod
    def get_moon_illumination(obs_time, sun=None, moon=None):
        """Return moon illumination fraction 0.0–1.0 at the given astropy Time
        (from the Moon's angle from the Sun; pass their positions if known)."""
        from astropy.coordinates import get_body
        try:
            sun = get_sun(obs_time) if sun is None else sun
            moon = get_body('moon', obs_time) if moon is None else moon
            elongation = sun.separation(moon)
            illumination = (1.0 - np.cos(elongation.rad)) / 2.0
            return float(np.clip(illumination, 0.0, 1.0))
        except Exception:
            return 0.5  # conservative fallback

    @staticmethod
    def get_moon_separation(dso_coord, obs_time):
        """Return angular separation in degrees between dso_coord and the moon, or None on failure."""
        from astropy.coordinates import get_body
        try:
            moon = get_body('moon', obs_time)
            return float(dso_coord.separation(moon).deg)
        except Exception:
            return None


# ---- Through the year -------------------------------------------------------

def seasonal_visibility(calculator, coord, min_altitude=30, horizon=None, year=None, step_days=15):
    """The parts of a year coord is well placed from calculator's location:
    [(first, last)] dates of runs of nights - sampled every step_days from
    Jan 1 - on which it's above min_altitude (and the custom horizon) in a
    dark sky at some point in the 12 hours around local midnight."""
    from datetime import date, timedelta
    year = year or date.today().year
    sampled = []
    for offset in range(0, 365, step_days):
        night = date(year, 1, 1) + timedelta(days=offset)
        try:
            _times, dso_altaz, sun_altaz = calculator.calculate_altaz_over_time(coord, night.isoformat(), 12)
            up = calculator.find_optimal_viewing_times(dso_altaz, sun_altaz, min_altitude, horizon=horizon)
        except Exception as e:
            logger.debug(f"Seasonal visibility skipped {night}: {e}")
            continue
        sampled.append((night, bool(np.any(up))))
    seasons, start = [], None
    for i, (night, up) in enumerate(sampled):
        if up and start is None:
            start = night
        if start is not None and (not up or i == len(sampled) - 1):
            seasons.append((start, night if up else sampled[i - 1][0]))
            start = None
    return seasons


def season_months(seasons):
    """'Nov-Dec, Jan-Mar' - each of seasonal_visibility()'s seasons as its
    first and last month (one month alone when it starts and ends in it)."""
    names = []
    for first, last in seasons:
        start, end = first.strftime("%b"), last.strftime("%b")
        names.append(start if start == end else f"{start}-{end}")
    return ", ".join(names)


# ---- One night, summed up ----------------------------------------------------


def _crossings(times, values, level):
    """[(Time, rising)] where values cross level, interpolated between steps."""
    above = np.asarray(values) >= level
    found = []
    for i in np.where(above[:-1] != above[1:])[0]:
        a, b = values[i] - level, values[i + 1] - level
        found.append((times[i] + (times[i + 1] - times[i]) * float(a / (a - b)), bool(b > a)))
    return found


def _first(crossings, rising):
    return next((t for t, up in crossings if up == rising), None)


def night_summary(calculator, coord, noon_local, min_altitude=30, horizon=None, step_minutes=5):
    """Where coord is over the night from noon_local (an aware datetime) to the
    next noon, from calculator's location: the sun's times, the object's rise,
    transit and set, its viewing windows (above min_altitude and the custom
    horizon, sun below -12°), the Moon, and its position on each hour.
    Times are aware datetimes in noon_local's zone."""
    from astropy.coordinates import get_body

    if calculator.location is None:
        raise ValueError("Observer location not set")
    tz = noon_local.tzinfo
    count = int(round(24 * 60 / step_minutes))
    times = Time(noon_local) + np.arange(count + 1) * step_minutes * u.min
    frame = AltAz(obstime=times, location=calculator.location)
    dso = coord.transform_to(frame)
    sun = get_sun(times).transform_to(frame)
    sun_alt = sun.alt.deg
    moon_altaz = get_body('moon', times).transform_to(frame)
    alt, az = dso.alt.deg, dso.az.deg
    visible = calculator.find_optimal_viewing_times(dso, sun, min_altitude, horizon=horizon)

    def local(t):
        return None if t is None else t.to_datetime(timezone=tz)

    sun_set = _crossings(times, sun_alt, SUNSET_ALTITUDE)
    astro = _crossings(times, sun_alt, ASTRONOMICAL_DARK_ALTITUDE)
    dark_sky = _crossings(times, sun_alt, DARK_SKY_ALTITUDE)
    up = _crossings(times, alt, 0.0)
    moon_up = _crossings(times, moon_altaz.alt.deg, 0.0)

    transit = None
    for crossing in calculator.find_meridian_crossings(times, coord):
        transit_alt = float(coord.transform_to(AltAz(obstime=crossing, location=calculator.location)).alt.deg)
        transit = {"time": local(crossing), "altitude": transit_alt}
        break

    dark = sun_alt < DARK_SKY_ALTITUDE
    highest = None
    if dark.any():
        i = int(np.argmax(np.where(dark, alt, -np.inf)))
        highest = {"time": local(times[i]), "altitude": float(alt[i]), "azimuth": float(az[i])}

    windows = []
    for window in calculator._find_viewing_windows(times, visible, dso):
        first = int(np.argmin(np.abs((times - window["start_time"]).to_value(u.min))))
        last = int(np.argmin(np.abs((times - window["end_time"]).to_value(u.min))))
        windows.append({
            "start": local(window["start_time"]), "end": local(window["end_time"]),
            "hours": (last - first + 1) * step_minutes / 60,
            "start_azimuth": float(az[first]), "end_azimuth": float(az[last]),
            "max_altitude": float(alt[first:last + 1].max()),
        })

    midnight = times[count // 2]
    moon_at = transit["time"] if transit else local(midnight)
    hourly = []
    per_hour = int(round(60 / step_minutes))
    for i in range(0, count + 1, per_hour):
        limit = float(horizon.altitude_at(az[i])) if horizon is not None else 0.0
        hourly.append({"time": local(times[i]), "altitude": float(alt[i]), "azimuth": float(az[i]),
                       "sun_altitude": float(sun_alt[i]), "moon_altitude": float(moon_altaz.alt.deg[i]),
                       "below_horizon": bool(alt[i] < limit), "visible": bool(visible[i])})

    return {
        "sunset": local(_first(sun_set, False)), "sunrise": local(_first(sun_set, True)),
        "dark_start": local(_first(astro, False)), "dark_end": local(_first(astro, True)),
        "dark_sky_start": local(_first(dark_sky, False)), "dark_sky_end": local(_first(dark_sky, True)),
        "rise": local(_first(up, True)), "set": local(_first(up, False)),
        "always_up": bool((alt > 0).all()), "never_up": bool((alt <= 0).all()),
        "transit": transit, "highest_in_dark": highest,
        "windows": windows, "hours_visible": float(visible.sum()) * step_minutes / 60,
        "moon_illumination": DSOVisibilityCalculator.get_moon_illumination(midnight),
        "moon_separation": DSOVisibilityCalculator.get_moon_separation(
            coord, Time(moon_at)) if moon_at else None,
        "moon_separation_at": moon_at,
        "moon_rise": local(_first(moon_up, True)), "moon_set": local(_first(moon_up, False)),
        "hourly": hourly,
    }
