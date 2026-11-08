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

from cosmoscollection.core.DatabaseManager import DatabaseManager

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

        if location_lat is not None and location_lon is not None:
            self.set_location(location_lat, location_lon, height)
        else:
            self._load_location_from_database()
            if self.location is not None:
                from cosmoscollection.planning.HorizonProfile import load_active_horizon
                self.horizon = load_active_horizon()

        if timezone:
            self.set_timezone(timezone)
        else:
            self._load_timezone_from_database()

    def set_location(self, lat, lon, height=250):
        """Set observer location."""
        self.location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)

    def set_timezone(self, timezone_str):
        """Set timezone for local time calculations."""
        try:
            self.timezone = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            self.timezone = pytz.UTC

    def _load_location_from_database(self):
        """Load observer location from database."""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                if row and row[0] is not None and row[1] is not None:
                    self.set_location(row[0], row[1])
        except Exception:
            pass

    def _load_timezone_from_database(self):
        """Load timezone from database."""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT timezone FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                if row and row[0]:
                    self.set_timezone(row[0])
        except Exception:
            pass

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
            import sqlite3
            from cosmoscollection.catalog.DSONames import catalog_ids
            from cosmoscollection.core.ResourceManager import ResourceManager, attach_update_catalogs
            # A connection of its own: this runs in worker threads, and the
            # DatabaseManager singleton's connection only works in the thread that made it
            conn = sqlite3.connect(str(ResourceManager.get_database_path()), timeout=30)
            try:
                attach_update_catalogs(conn)
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

    def find_optimal_viewing_times(self, dso_altaz, sun_altaz, min_altitude=30, max_sun_altitude=-12, horizon=None):
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

    def calculate_seasonal_visibility(self, dso_coord, year=None, min_altitude=30):
        """
        Calculate when a DSO is optimally visible throughout a year.

        Args:
            dso_coord (SkyCoord): DSO coordinates
            year (int): Year to calculate (default: current year)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)

        Returns:
            list: List of date ranges when DSO is optimally visible
        """
        if self.location is None:
            return []

        if year is None:
            from datetime import datetime
            year = datetime.now().year

        try:
            # Sample dates throughout the year (every 10 days)
            dates = []
            visibility_data = []

            for day_of_year in range(1, 366, 10):  # Every 10 days
                try:
                    date = Time(f"{year}-01-01") + (day_of_year - 1) * u.day

                    # Calculate for midnight (when most DSOs are best visible)
                    midnight = date + 12 * u.hour  # Approximate local midnight

                    altaz_frame = AltAz(obstime=midnight, location=self.location)
                    dso_altaz = dso_coord.transform_to(altaz_frame)
                    sun = get_sun(midnight)
                    sun_altaz = sun.transform_to(altaz_frame)

                    # Check if object is well-visible (above min altitude and sun is down)
                    is_visible = (dso_altaz.alt.deg > min_altitude and
                                sun_altaz.alt.deg < -12)

                    dates.append(date)
                    visibility_data.append(is_visible)

                except Exception:
                    continue

            # Find continuous visibility periods
            return self._group_visibility_seasons(dates, visibility_data)

        except Exception:
            return []

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

    def _group_visibility_seasons(self, dates, visibility_data):
        """Group contiguous visibility dates into seasons."""
        if not dates or not any(visibility_data):
            return []

        seasons = []
        current_season_start = None

        for i, (date, is_visible) in enumerate(zip(dates, visibility_data)):
            if is_visible and current_season_start is None:
                current_season_start = date
            elif not is_visible and current_season_start is not None:
                # End of a visibility season
                seasons.append({
                    "start_date": current_season_start,
                    "end_date": dates[i-1] if i > 0 else current_season_start,
                })
                current_season_start = None

        # Handle case where season extends to end of year
        if current_season_start is not None:
            seasons.append({
                "start_date": current_season_start,
                "end_date": dates[-1],
            })

        return seasons

    @staticmethod
    def get_twilight_condition(sun_altitude):
        """
        Get twilight condition based on sun altitude.

        Args:
            sun_altitude (float): Sun altitude in degrees

        Returns:
            str: Twilight condition
        """
        if sun_altitude > 0:
            return "Daylight"
        elif sun_altitude > -6:
            return "Civil Twilight"
        elif sun_altitude > -12:
            return "Nautical Twilight"
        elif sun_altitude > -18:
            return "Astronomical Twilight"
        else:
            return "Night"

    @staticmethod
    def get_moon_illumination(obs_time):
        """Return moon illumination fraction 0.0–1.0 at the given astropy Time."""
        from astropy.coordinates import get_body, get_sun
        import numpy as np
        try:
            sun = get_sun(obs_time)
            moon = get_body('moon', obs_time)
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


# ---- One night, summed up ----------------------------------------------------

SUNSET_ALTITUDE = -0.833        # The sun's top at the horizon, with refraction
DARK_SKY_ALTITUDE = -12.0       # Sun below this: the viewing windows' "dark sky" (as everywhere in the app)
ASTRONOMICAL_DARK_ALTITUDE = -18.0


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
