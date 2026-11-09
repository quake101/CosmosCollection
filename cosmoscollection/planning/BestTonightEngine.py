"""
Best DSO Tonight's calculation: which catalog objects (or Target List
entries) are up in a dark sky at the chosen time, ranked, with their best
time, direction and distance from the Moon.

Used by the Best DSO Tonight window (through DSOCalculationThread) and the
command line ('tonight'). No UI imports - QSettings only, for the saved
Time choice and Maximum Threads.
"""

import logging
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from cosmoscollection.core.DatabaseManager import open_connection
from cosmoscollection.app.DSOData import DESIGNATIONS_SQL
from cosmoscollection.core.Formatting import UNKNOWN_MAGNITUDE_FROM, known_magnitude

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# The "Time:" choice - same options as the Target List's "Direction at:"
TIME_SETTING = "BestDSOTonight/time"  # "mode|at|from|to", times as HH:MM
TIME_DEFAULT = ("night", "22:00", "01:00", "02:00")

DEFAULT_MIN_ALTITUDE = 30
DEFAULT_MAX_MAGNITUDE = 12.0
DEFAULT_DSO_LIMIT = 200


class TonightError(Exception):
    """The calculation couldn't run (no location, nothing to check)."""


class TonightCancelled(Exception):
    """Stopped before it finished."""


def resolve_time_window(time_choice, location, timezone_name, now=None):
    """(start, end, note, times) of the chosen time: local datetimes - start ==
    end for a single moment - and the labeled times the Direction column is for
    ([("Dusk", ...), ("Dawn", ...)] etc.). The coming night (or the current one
    before dawn) - or, with now, the night as seen at that moment."""
    from cosmoscollection.planning.NightTimes import local_zone, resolve_direction_times
    mode, at, start, end = time_choice
    times, note = resolve_direction_times(mode, at, start, end, location, local_zone(timezone_name), now=now)
    return times[0][1], times[-1][1], note, times


def describe_time_window(mode, start, end):
    """'tonight (20:40 → 06:10)', '01:00 → 02:00', 'at midnight'... for status text."""
    if start == end:
        names = {"now": f"right now ({start:%H:%M})", "dusk": f"at dusk ({start:%H:%M})",
                 "midnight": "at midnight", "dawn": f"at dawn ({start:%H:%M})"}
        return names.get(mode, f"at {start:%H:%M}")
    span = f"{start:%H:%M} → {end:%H:%M}"
    return f"tonight ({span})" if mode == "night" else span


def _get_available_catalogs():
    """Get list of available catalogs from database.

    Standalone helper (no location/timezone lookup) so callers that only
    need the catalog list - e.g. populating the catalog dropdown - don't
    have to pay for building a full DSOVisibilityCalculator.
    """
    try:
        conn = open_connection()
        try:
            return [row[0] for row in conn.execute("SELECT DISTINCT catalogue FROM cataloguenr ORDER BY catalogue")]
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error getting catalogs: {e}")
        return []


def saved_max_workers():
    """Settings > Maximum Threads."""
    from cosmoscollection.core.Settings import app_settings
    settings = app_settings()
    return max(1, settings.value("max_threads", max(1, (os.cpu_count() or 4) - 2), type=int))


class BestTonightCalculation:
    """
    Calculates the best DSOs for tonight.

    Uses coordinate-based visibility calculations for reliability and consistency
    with other tools (avoids object name resolution issues like 'sh2 142' vs 'sh2-142').

        calculation = BestTonightCalculation(min_altitude=30, ...)
        visible = calculation.calculate(progress=print)

    After calculate(): window_start/window_end/window_note/window_times (the
    chosen time, resolved) and blocked_by_horizon.
    """

    # Returned by calculate_tonight_visibility for a DSO that clears Min
    # Altitude during dark sky but never rises above the custom horizon
    BLOCKED_BY_HORIZON = object()

    def __init__(self, min_altitude=DEFAULT_MIN_ALTITUDE, max_magnitude=DEFAULT_MAX_MAGNITUDE, selected_catalogs=None,
                 dso_limit=DEFAULT_DSO_LIMIT, selected_dso_types=None, use_target_list=False, time_choice=TIME_DEFAULT,
                 use_horizon=False, hide_completed=False, site=None, max_workers=None, now=None):
        """site: an Observer.Site to calculate for (default: the active
        location). max_workers: threads (default: Settings > Maximum Threads).
        now: the moment "tonight" is seen from (default: the current time)."""
        from cosmoscollection.planning.Visibility import DSOVisibilityCalculator

        self.min_altitude = min_altitude
        self.max_magnitude = max_magnitude
        self.selected_catalogs = selected_catalogs or []
        self.selected_dso_types = selected_dso_types or []
        self.dso_limit = dso_limit
        self.use_target_list = use_target_list
        # Skip target list entries whose status is "Completed"
        self.hide_completed = hide_completed
        # (mode, at, from, to) - resolved to window_start/window_end in calculate()
        self.time_choice = time_choice
        self.time_mode = time_choice[0]
        # A custom time frame lists only objects up for the whole frame ("best
        # viewed from 1am-2am"); Tonight lists anything up at some point
        self.whole_window = self.time_mode == "frame"
        self.window_start = self.window_end = None
        self.window_note = ""
        self.window_times = []  # [(label, local datetime)] for the Direction column
        # Number of DSOs that would be visible but are entirely below the
        # custom horizon - set by calculate()
        self.blocked_by_horizon = 0
        self.max_workers = max_workers
        self.now = now

        if site is not None:
            from cosmoscollection.planning.HorizonProfile import load_horizon_for_site
            self.calculator = DSOVisibilityCalculator(site.lat, site.lon, site.timezone or "UTC")
            self.horizon = load_horizon_for_site(site) if use_horizon else None
        else:
            # The active location, from the database
            self.calculator = DSOVisibilityCalculator()
            self.horizon = self.calculator.horizon if use_horizon else None
        self.location = self.calculator.location
        self.local_tz = self.calculator.timezone

    def _window_sampling(self):
        """(start Time, duration hours, samples per hour) covering the chosen time.

        A single moment is one sample; frames up to 3 hours are sampled every
        5 minutes (so "up for the whole of 1am-2am" is checked closely), longer
        ones every 15 minutes as before.
        """
        hours = (self.window_end - self.window_start).total_seconds() / 3600
        start = Time(self.window_start.astimezone(pytz.UTC).replace(tzinfo=None))
        if hours <= 0:
            return start, 0.25, 4  # linspace(0, 0.25, 1) -> just the start
        return start, hours, 12 if hours <= 3 else 4

    def _qualifies(self, visible):
        """Whether a per-sample visibility mask passes the time rule."""
        if not np.any(visible):
            return False
        return bool(np.all(visible)) if self.whole_window else True

    def _visibility_result(self, dso_info, dso_coord, time_range, dso_altaz, visible, samples_per_hour):
        """Result dict for a qualifying DSO: its altitude, time visible, and best time."""
        altitudes = dso_altaz.alt.deg
        indices = np.where(visible)[0]
        if self.time_mode == "night":
            best_idx = indices[len(indices) // 2]  # Middle of its time up tonight
        else:
            best_idx = indices[int(np.argmax(altitudes[indices]))]  # Highest within the chosen time
        single_moment = len(altitudes) == 1
        best_time_utc = time_range[best_idx].datetime.replace(tzinfo=pytz.UTC)
        return {
            "dso_info": dso_info,
            "max_altitude": np.max(altitudes[visible]),
            "visible_hours": None if single_moment else float(np.sum(visible)) / samples_per_hour,
            "optimal_time": best_time_utc.astimezone(self.local_tz),
            "optimal_altitude": altitudes[best_idx],
            "optimal_azimuth": dso_altaz.az.deg[best_idx],
            "coordinates": dso_coord
        }

    def calculate_tonight_visibility(self, dso_info):
        """Calculate visibility for a DSO in the chosen time using the centralized calculator"""
        try:
            # Coordinate-based calculation for reliability (avoids name resolution issues)
            dso_coord = SkyCoord(ra=dso_info["ra_deg"] * u.deg, dec=dso_info["dec_deg"] * u.deg)
            start_time, duration, samples_per_hour = self._window_sampling()
            time_range, dso_altaz, sun_altaz = self.calculator.calculate_altaz_over_time(
                dso_coord, start_time, duration, samples_per_hour)

            visible = self.calculator.find_optimal_viewing_times(
                dso_altaz, sun_altaz, self.min_altitude, horizon=self.horizon)
            if not self._qualifies(visible):
                # Note DSOs hidden only by the custom horizon, for the status line
                if self.horizon is not None and self._qualifies(
                        self.calculator.find_optimal_viewing_times(dso_altaz, sun_altaz, self.min_altitude)):
                    return self.BLOCKED_BY_HORIZON
                return None
            return self._visibility_result(dso_info, dso_coord, time_range, dso_altaz, visible, samples_per_hour)

        except Exception:
            return None

    def get_available_catalogs(self):
        """Get list of available catalogs from database"""
        return _get_available_catalogs()

    def load_dsos_from_target_list(self):
        """Load DSOs from the user's target list"""
        try:
            conn = open_connection()
            cursor = conn.cursor()

            # Query target list entries with telescope information
            # No magnitude filter - show all target list items since user intentionally added them
            query = """
                SELECT t.name, t.dso_type, t.constellation, t.magnitude, t.ra_deg, t.dec_deg,
                       t.size_info, t.telescope_id, tel.name as telescope_name
                FROM usertargetlist t
                LEFT JOIN usertelescopes tel ON t.telescope_id = tel.id
                WHERE t.ra_deg IS NOT NULL
                    AND t.dec_deg IS NOT NULL
            """
            if self.hide_completed:
                query += " AND COALESCE(t.status, '') != 'Completed'"
            query += " ORDER BY t.name"

            cursor.execute(query)

            rows = cursor.fetchall()
            dsos = []
            for row in rows:
                name, dso_type, constellation, magnitude, ra_deg, dec_deg, size_info, telescope_id, telescope_name = row
                if name and ra_deg is not None and dec_deg is not None:
                    # Query designations from main database by matching coordinates
                    cursor.execute(f"""
                        SELECT {DESIGNATIONS_SQL} as designations
                        FROM dsodetail d
                        WHERE ABS(d.ra - ?) < 0.001
                          AND ABS(d.dec - ?) < 0.001
                        ORDER BY ABS(d.ra - ?) + ABS(d.dec - ?) ASC
                        LIMIT 1
                    """, (ra_deg, dec_deg, ra_deg, dec_deg))

                    desig_row = cursor.fetchone()
                    designations = desig_row[0] if desig_row and desig_row[0] else name

                    dsos.append({
                        "name": name,
                        "type": dso_type or "Unknown",
                        "constellation": constellation or "Unknown",
                        "magnitude": known_magnitude(magnitude),  # None when unknown
                        "surface_brightness": 0.0,
                        "size_min": 0.0,
                        "size_max": 0.0,
                        "dso_class": "Unknown",
                        "ra_deg": float(ra_deg),
                        "dec_deg": float(dec_deg),
                        "designations": designations,
                        "telescope_name": telescope_name or "Any"
                    })

            conn.close()
            return dsos

        except Exception as e:
            logger.error(f"Error loading DSOs from target list: {e}")
            return []

    def load_dsos_from_database(self):
        """Load DSOs from the database with thread-safe connection"""
        try:
            conn = open_connection()
            cursor = conn.cursor()

            # Both searches take the catalog and type filters
            filters = """
                    AND d.constellation IS NOT NULL
                    AND d.dsotype IS NOT NULL
                    AND d.ra IS NOT NULL
                    AND d.dec IS NOT NULL
            """
            filter_params = []
            if self.selected_catalogs:
                catalog_placeholders = ','.join('?' * len(self.selected_catalogs))
                filters += f"""
                    AND d.id IN (
                        SELECT dsodetailid FROM cataloguenr
                        WHERE catalogue IN ({catalog_placeholders})
                    )
                """
                filter_params.extend(self.selected_catalogs)
            if self.selected_dso_types:
                type_placeholders = ','.join('?' * len(self.selected_dso_types))
                filters += f" AND d.dsotype IN ({type_placeholders})"
                filter_params.extend(self.selected_dso_types)

            columns = f"""
                SELECT
                    d.id,
                    {DESIGNATIONS_SQL} as designations,
                    d.dsotype as type,
                    d.constellation,
                    d.magnitude,
                    d.surfacebrightness,
                    CAST(d.sizemin/60.0 AS REAL) as sizemin,
                    CAST(d.sizemax/60.0 AS REAL) as sizemax,
                    d.dsoclass,
                    d.ra as ra_deg,
                    d.dec as dec_deg
                FROM dsodetail d
            """
            limit = int(self.dso_limit)
            # The brightest, up to the DSO Limit, within Max Magnitude
            cursor.execute(columns + """
                WHERE d.magnitude > 0 AND d.magnitude < ? AND d.magnitude <= ?
            """ + filters + f" ORDER BY d.magnitude ASC LIMIT {limit}",
                           [UNKNOWN_MAGNITUDE_FROM, self.max_magnitude] + filter_params)
            rows = cursor.fetchall()
            # And as many again without a known magnitude (mostly nebulae and dark
            # nebulae, which are good targets too), largest first
            cursor.execute(columns + """
                WHERE (d.magnitude IS NULL OR d.magnitude <= 0 OR d.magnitude >= ?)
            """ + filters + f" ORDER BY COALESCE(d.sizemax, d.sizemin, 0) DESC LIMIT {limit}",
                           [UNKNOWN_MAGNITUDE_FROM] + filter_params)
            rows += cursor.fetchall()

            dsos = []
            for row in rows:
                dso_id, designations, dso_type, constellation, magnitude, surface_brightness, size_min, size_max, dso_class, ra_deg, dec_deg = row
                if designations and ra_deg is not None and dec_deg is not None:
                    # Take the first designation as the primary name
                    primary_name = designations.split(',')[0].strip()
                    dsos.append({
                        "id": dso_id,
                        "name": primary_name,
                        "type": dso_type or "Unknown",
                        "constellation": constellation or "Unknown",
                        "magnitude": known_magnitude(magnitude),  # None when unknown
                        "surface_brightness": float(surface_brightness) if surface_brightness is not None else 0.0,
                        "size_min": float(size_min) if size_min is not None else 0.0,
                        "size_max": float(size_max) if size_max is not None else 0.0,
                        "dso_class": dso_class or "Unknown",
                        "ra_deg": float(ra_deg),
                        "dec_deg": float(dec_deg),
                        "designations": designations
                    })

            conn.close()
            return dsos

        except Exception as e:
            logger.error(f"Error loading DSOs from database: {e}")
            return []

    def calculate_dso_batch(self, dso_batch):
        """Calculate visibility for a batch of DSOs

        Returns:
            tuple: (visible DSO results, number blocked by the custom horizon)
        """
        from cosmoscollection.planning.NightTimes import compass_direction
        batch_results = []
        blocked = 0
        for dso_info in dso_batch:
            result = self.calculate_tonight_visibility(dso_info)
            if result is self.BLOCKED_BY_HORIZON:
                blocked += 1
            elif result:
                # Replaced by _add_direction_cells() once every batch is done
                result["direction"] = compass_direction(result["optimal_azimuth"])
                batch_results.append(result)
        return batch_results, blocked

    def _add_direction_cells(self, visible_dsos):
        """The Direction column as the Target List shows it: the direction at
        the chosen time, or at its start and end ("E → SE", dusk and dawn for
        Tonight), marked "(below)" where under the horizon. The tooltip gives
        each time's azimuth and altitude, plus the direction at the Best Time."""
        if not visible_dsos:
            return
        try:
            from cosmoscollection.planning.NightTimes import direction_cells
            from astropy.coordinates import concatenate
            coords = concatenate([dso["coordinates"] for dso in visible_dsos])
            cells = direction_cells(coords, self.window_times, self.location, self.horizon)
        except Exception as e:
            logger.error(f"Direction column failed, showing the Best Time direction: {e}")
            return
        for dso, (text, tooltip, _below) in zip(visible_dsos, cells):
            best = (f"Best Time ({dso['optimal_time']:%H:%M}): {dso['direction']}, "
                    f"azimuth {dso['optimal_azimuth']:.0f}°, altitude {dso['optimal_altitude']:.0f}°")
            if len(self.window_times) > 1:
                tooltip += "\n" + best  # A single time is already the Best Time
            # Not grayed out like the Target List's "below throughout": anything
            # listed here is up at some point in between
            dso.update(direction=text, direction_tooltip=tooltip)

    def _add_moon(self, visible_dsos):
        """Each result's moon_illumination (in the middle of the chosen time)
        and moon_separation (degrees at its best time, or None)."""
        from cosmoscollection.planning.Visibility import DSOVisibilityCalculator as Calc
        try:
            # Moon phase in the middle of the chosen time (it changes ~12% a day)
            middle = self.window_start + (self.window_end - self.window_start) / 2
            moon_illumination = Calc.get_moon_illumination(Time(middle.astimezone(pytz.UTC).replace(tzinfo=None)))

            for dso_data in visible_dsos:
                dso_data["moon_illumination"] = moon_illumination
                moon_sep = None
                try:
                    coord = dso_data.get("coordinates")
                    optimal_time = dso_data.get("optimal_time")
                    if coord is not None and optimal_time is not None:
                        opt_utc = optimal_time.astimezone(pytz.UTC).replace(tzinfo=None)
                        moon_sep = Calc.get_moon_separation(coord, Time(opt_utc))
                except Exception:
                    pass
                dso_data["moon_separation"] = moon_sep  # float degrees or None
        except Exception as e:
            logger.error(f"Moon annotation failed: {e}")
            for dso_data in visible_dsos:
                dso_data.setdefault("moon_illumination", 0.5)
                dso_data.setdefault("moon_separation", None)

    def calculate(self, progress=None, cancelled=None):
        """The visible DSOs, ranked by altitude and magnitude. progress(percent)
        is called as batches finish; cancelled() is checked between them.
        Raises TonightError, or TonightCancelled."""
        if self.location is None:
            raise TonightError("Observer location not configured")

        # The chosen time, on the coming night (or the current one before dawn)
        self.window_start, self.window_end, self.window_note, self.window_times = resolve_time_window(
            self.time_choice, self.location, getattr(self.local_tz, "zone", None), now=self.now)

        # Load DSOs from database or target list
        if self.use_target_list:
            dso_catalog = self.load_dsos_from_target_list()
        else:
            dso_catalog = self.load_dsos_from_database()

        if not dso_catalog:
            if self.use_target_list and self.hide_completed:
                raise TonightError("No targets in your target list that aren't Completed")
            if self.use_target_list:
                raise TonightError("No targets with a position in your target list")
            raise TonightError("No DSOs found in database")

        visible_dsos = []
        self.blocked_by_horizon = 0
        total_dsos = len(dso_catalog)
        max_workers = self.max_workers or saved_max_workers()

        # A few batches per thread, so progress moves and a stop doesn't wait long
        batch_size = max(1, math.ceil(total_dsos / (max_workers * 4)))
        dso_batches = [dso_catalog[i:i + batch_size] for i in range(0, total_dsos, batch_size)]

        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self.calculate_dso_batch, batch): batch for batch in dso_batches}
            for future in as_completed(future_to_batch):
                if cancelled and cancelled():
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise TonightCancelled()
                try:
                    batch_results, blocked = future.result()
                    visible_dsos.extend(batch_results)
                    self.blocked_by_horizon += blocked
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                completed_count += len(future_to_batch[future])
                if progress:
                    progress(min(int(completed_count * 100 / total_dsos), 100))

        # Higher and brighter first. An unknown magnitude (mostly nebulae) counts as
        # the Max Magnitude, so it isn't pushed below everything else.
        def rank(dso):
            magnitude = dso["dso_info"]["magnitude"]
            return -dso["max_altitude"] + (magnitude if magnitude is not None else self.max_magnitude)

        visible_dsos.sort(key=rank)

        self._add_direction_cells(visible_dsos)
        self._add_moon(visible_dsos)
        return visible_dsos
