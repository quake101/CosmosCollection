#!/usr/bin/env python3
"""
Deep Sky Object Visibility Calculator
Uses astropy and PySide6 to determine when DSOs are optimally visible

Contains the centralized DSOVisibilityCalculator class for all visibility calculations
"""

import sys
import os
import calendar
import matplotlib
import numpy as np
from datetime import datetime, date as date_cls, timedelta
from PySide6.QtCore import Qt, QDate, QThread, Signal, QTimer, QSettings, QEvent
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLineEdit, QLabel, QTextEdit,
                               QDateEdit, QSpinBox, QGroupBox, QMessageBox, QCalendarWidget, QSizePolicy,
                               QComboBox, QTableView)
from PySide6.QtGui import QTextCharFormat, QColor

matplotlib.use('Qt5Agg')

# Suppress matplotlib font_manager debug messages
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Set dark theme for matplotlib
plt.style.use('dark_background')
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import pytz
import warnings

warnings.filterwarnings('ignore')

from Theme import COLORS

# Get the application directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Import DatabaseManager from separate file
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from TimeFormatHelper import format_time


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

        if location_lat is not None and location_lon is not None:
            self.set_location(location_lat, location_lon, height)
        else:
            self._load_location_from_database()

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

        Tries multiple name variations and also attempts database lookup for coordinates.

        Args:
            dso_name (str): Name of the DSO (e.g., 'M31', 'NGC 7000', 'sh2 142')

        Returns:
            tuple: (SkyCoord object, error_message) - error_message is None if successful
        """
        # First try the original name
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

        # If name resolution fails, try database lookup
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Try exact name match first
                cursor.execute("""
                    SELECT ra_deg, dec_deg FROM dso
                    WHERE UPPER(TRIM(name)) = ?
                    OR UPPER(TRIM(alternate_names)) LIKE ?
                    LIMIT 1
                """, (original_name, f"%{original_name}%"))

                row = cursor.fetchone()
                if row and row[0] is not None and row[1] is not None:
                    coord = SkyCoord(ra=row[0] * u.deg, dec=row[1] * u.deg)
                    return coord, None

                # Try partial matches with variations
                for variation in name_variations:
                    cursor.execute("""
                        SELECT ra_deg, dec_deg FROM dso
                        WHERE UPPER(TRIM(name)) LIKE ?
                        OR UPPER(TRIM(alternate_names)) LIKE ?
                        LIMIT 1
                    """, (f"%{variation}%", f"%{variation}%"))

                    row = cursor.fetchone()
                    if row and row[0] is not None and row[1] is not None:
                        coord = SkyCoord(ra=row[0] * u.deg, dec=row[1] * u.deg)
                        return coord, None

        except Exception:
            pass  # Database lookup failed, continue with original error

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

    def find_optimal_viewing_times(self, dso_altaz, sun_altaz, min_altitude=30, max_sun_altitude=-12):
        """
        Find optimal viewing times based on altitude and darkness criteria.

        Args:
            dso_altaz: DSO altitude/azimuth data
            sun_altaz: Sun altitude/azimuth data
            min_altitude (float): Minimum DSO altitude in degrees (default: 30)
            max_sun_altitude (float): Maximum sun altitude for dark sky (default: -12)

        Returns:
            numpy array: Boolean array indicating optimal viewing times
        """
        dso_visible = dso_altaz.alt.deg > min_altitude
        dark_sky = sun_altaz.alt.deg < max_sun_altitude
        return dso_visible & dark_sky

    def calculate_visibility_for_date(self, dso_name, date, duration_hours=24, min_altitude=30):
        """
        Calculate complete visibility information for a DSO on a specific date.

        Args:
            dso_name (str): Name of the DSO
            date (str): Date in ISO format (YYYY-MM-DD)
            duration_hours (float): Duration to calculate (default: 24 hours)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)

        Returns:
            dict: Complete visibility results or None if error
        """
        # Get DSO coordinates with enhanced name resolution
        dso_coord, error = self.get_dso_coordinates_enhanced(dso_name)
        if dso_coord is None:
            return {"error": f"Could not find coordinates for {dso_name}: {error}"}

        return self.calculate_visibility_for_coordinates(dso_coord, date, duration_hours, min_altitude, dso_name)

    def calculate_visibility_for_coordinates(self, dso_coord, date, duration_hours=24, min_altitude=30, dso_name=None):
        """
        Calculate complete visibility information for DSO coordinates on a specific date.

        Args:
            dso_coord (SkyCoord): Coordinates of the DSO
            date (str): Date in ISO format (YYYY-MM-DD)
            duration_hours (float): Duration to calculate (default: 24 hours)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)
            dso_name (str, optional): Name of the DSO for display purposes

        Returns:
            dict: Complete visibility results or None if error
        """
        try:
            # Calculate altitude/azimuth over time
            time_range, dso_altaz, sun_altaz = self.calculate_altaz_over_time(
                dso_coord, date, duration_hours)

            # Find optimal viewing times
            optimal_times = self.find_optimal_viewing_times(dso_altaz, sun_altaz, min_altitude)

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

    def calculate_visibility_hours_for_day(self, dso_coord, date, min_altitude=30):
        """
        Calculate total visibility hours for a DSO on a specific day.

        Args:
            dso_coord (SkyCoord): Coordinates of the DSO
            date (str): Date in ISO format (YYYY-MM-DD)
            min_altitude (float): Minimum altitude threshold (default: 30 degrees)

        Returns:
            float: Total hours the DSO is optimally visible on this day
        """
        try:
            # Calculate for full 24 hours
            time_range, dso_altaz, sun_altaz = self.calculate_altaz_over_time(
                dso_coord, date, 24, time_resolution=4)

            # Find optimal viewing times
            optimal_times = self.find_optimal_viewing_times(dso_altaz, sun_altaz, min_altitude)

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
    def azimuth_to_direction(azimuth):
        """
        Convert azimuth angle to cardinal direction.

        Args:
            azimuth (float): Azimuth in degrees (0-360)

        Returns:
            str: Cardinal direction (e.g., 'N', 'NE', 'SSW')
        """
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((azimuth + 11.25) / 22.5) % 16
        return directions[idx]

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


def visibility_hours_to_color(hours):
    """Map a day's visibility hours to (background, foreground) QColor pair.
    Shared by VisibilityCalendar and SessionManager's calendar view so both use
    the same 7-tier green->red scale rather than duplicating the thresholds."""
    if hours >= 8:
        return QColor(0, 150, 0), QColor(255, 255, 255)
    elif hours >= 6:
        return QColor(0, 120, 0), QColor(255, 255, 255)
    elif hours >= 4:
        return QColor(100, 120, 0), QColor(255, 255, 255)
    elif hours >= 2:
        return QColor(150, 100, 0), QColor(255, 255, 255)
    elif hours >= 1:
        return QColor(150, 60, 0), QColor(255, 255, 255)
    elif hours > 0:
        return QColor(120, 40, 0), QColor(255, 255, 255)
    else:
        return QColor(80, 0, 0), QColor(180, 180, 180)


class MonthlyVisibilityThread(QThread):
    """Thread for calculating visibility hours (and moon illumination) for every
    day in a date range, which may span 1-3 displayed calendar months. Modeled
    on SessionManager.SessionMonthlyVisibilityThread, minus its location-override
    params - this always uses the DB's default active location, same as
    DSOVisibilityCalculator() with no args already does."""
    progress = Signal(int, int)  # days completed, total days
    # object, not two dicts: Signal(dict) marshals through QVariantMap across
    # threads, which requires string keys and silently drops non-string keys
    # (our keys are datetime.date objects) - same fix as SessionManager's
    # equivalent thread.
    finished = Signal(object)  # (visibility_hours_by_date, moon_illumination_by_date)
    error = Signal(str)

    def __init__(self, dso_coord, dso_name, start_date, end_date, min_altitude, ra_deg=None, dec_deg=None):
        super().__init__()
        self.dso_coord = dso_coord
        self.dso_name = dso_name
        self.start_date = start_date
        self.end_date = end_date
        self.min_altitude = min_altitude
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.calculator = DSOVisibilityCalculator()

    def run(self):
        """Calculate visibility and moon illumination for each day in the range"""
        try:
            if self.calculator.location is None:
                self.error.emit("Observer location not configured.")
                return

            # Get coordinates if not already provided
            if self.dso_coord is None:
                if self.ra_deg is not None and self.dec_deg is not None:
                    self.dso_coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
                else:
                    self.dso_coord, error = self.calculator.get_dso_coordinates_enhanced(self.dso_name)
                    if self.dso_coord is None:
                        self.error.emit(f"Could not find coordinates: {error}")
                        return

            visibility_hours = {}
            moon_illumination = {}
            total_days = (self.end_date - self.start_date).days + 1
            current = self.start_date
            completed = 0

            while current <= self.end_date:
                date_str = current.strftime("%Y-%m-%d")
                hours = self.calculator.calculate_visibility_hours_for_day(
                    self.dso_coord, date_str, self.min_altitude)
                visibility_hours[current] = hours
                moon_illumination[current] = DSOVisibilityCalculator.get_moon_illumination(
                    Time(f"{date_str}T12:00:00"))
                completed += 1
                self.progress.emit(completed, total_days)
                current += timedelta(days=1)

            self.finished.emit((visibility_hours, moon_illumination))

        except Exception as e:
            self.error.emit(f"Calculation error: {str(e)}")


class VisibilityCalendar(QCalendarWidget):
    """One month of the DSO Visibility Calculator's calendar view. Combines two
    independent, date-keyed layers on the same cells: DSO visibility hours
    (background fill) and weather astro score (a stripe across the top, only
    for the ~7 dates with a live forecast) - plus moon illumination, shown only
    in the hover tooltip. Mirrors SessionManager.SessionMonthCalendar, minus the
    session-marker dots that don't apply here."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.visibility_hours = {}    # date -> hours
        self.weather_scores = {}      # date -> astro_score (0-100)
        self.weather_details = {}     # date -> {"cloud_cover": float, "seeing": str}
        self.moon_illumination = {}   # date -> fraction (0.0-1.0)
        self.setGridVisible(True)
        self.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self.setSelectedDate(QDate.currentDate())
        self.setMouseTracking(True)

        self.tooltip_label = QLabel(self)
        self.tooltip_label.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.tooltip_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['background_lighter']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10pt;
            }}
        """)
        self.tooltip_label.hide()

        # QCalendarWidget renders its day grid via a private internal QTableView
        # subclass - overriding mouseMoveEvent/leaveEvent on the QCalendarWidget
        # itself never fires while hovering over day cells, since those mouse
        # events are consumed by that internal view and don't propagate up to us.
        # Install an event filter on the view's viewport instead, which is where
        # the events actually land. (This replaces a previous implementation that
        # called the non-existent QCalendarWidget.dateAt() and was silently dead
        # code - see SessionManager.SessionMonthCalendar for the same fix.)
        self._day_view = self.findChild(QTableView)
        if self._day_view is not None:
            self._day_view.setMouseTracking(True)
            self._day_view.viewport().setMouseTracking(True)
            self._day_view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if self._day_view is not None and obj is self._day_view.viewport():
            if event.type() == QEvent.Type.MouseMove:
                py_date = self._date_from_view_pos(event.pos())
                self._show_tooltip_for_date(py_date, self._day_view.viewport().mapToGlobal(event.pos()))
            elif event.type() == QEvent.Type.Leave:
                self.tooltip_label.hide()
        return super().eventFilter(obj, event)

    def _date_from_view_pos(self, pos_in_view):
        """QCalendarWidget has no public dateAt()/similar for this - its day grid
        is rendered by an internal QTableView whose model is private/undocumented.
        Resolve the date ourselves from the cell's (row, col) instead: the model's
        row 0 is a non-clickable weekday-name header, so the real day grid starts
        at row 1, which is always firstDayOfWeek()-aligned - walk backward from
        the 1st of the shown month by however many leading days that needs."""
        index = self._day_view.indexAt(pos_in_view)
        if not index.isValid() or index.row() < 1:
            return None
        first_of_month = date_cls(self.yearShown(), self.monthShown(), 1)
        qt_first_day = self.firstDayOfWeek().value  # Qt.Monday=1 .. Qt.Sunday=7
        first_weekday = first_of_month.isoweekday()  # Python: Monday=1 .. Sunday=7
        # Qt always shows at least one full leading week from the previous month,
        # even when the 1st already falls exactly on the configured first day of
        # the week - so this must land in [1, 7], never 0.
        leading_days = ((first_weekday - qt_first_day - 1) % 7) + 1
        first_visible_date = first_of_month - timedelta(days=leading_days)
        cell_offset = (index.row() - 1) * 7 + index.column()
        return first_visible_date + timedelta(days=cell_offset)

    def set_data(self, visibility_hours=None, weather_scores=None, weather_details=None, moon_illumination=None):
        """Update one or more data layers and repaint. Any layer left as None
        (the default) keeps its current values."""
        if visibility_hours is not None:
            self.visibility_hours = visibility_hours
        if weather_scores is not None:
            self.weather_scores = weather_scores
        if weather_details is not None:
            self.weather_details = weather_details
        if moon_illumination is not None:
            self.moon_illumination = moon_illumination
        self.updateCells()
        self.update()

    def clear_visibility(self):
        """Clear the visibility-hours and moon-illumination layers (both computed
        together by MonthlyVisibilityThread). Weather clears independently on its
        own refresh cycle."""
        self.visibility_hours = {}
        self.moon_illumination = {}
        self.tooltip_label.hide()
        self.updateCells()
        self.update()

    def get_color_for_hours(self, hours):
        """Get background and foreground colors for given visibility hours"""
        return visibility_hours_to_color(hours)

    def paintCell(self, painter, rect, date):
        """Override to paint custom cell backgrounds"""
        from PySide6.QtGui import QPen, QBrush
        from WeatherForecast import get_rating_color

        # Check if this date is in the current month
        if date.month() != self.monthShown() or date.year() != self.yearShown():
            # Default painting for dates outside current month (grayed out)
            super().paintCell(painter, rect, date)
            return

        py_date = date.toPython()

        # Determine colors based on visibility data
        if py_date in self.visibility_hours:
            bg_color, fg_color = self.get_color_for_hours(self.visibility_hours[py_date])
        else:
            # Default gray for dates without visibility data
            bg_color = QColor(64, 64, 64)
            fg_color = QColor(200, 200, 200)

        # Fill background
        painter.fillRect(rect, QBrush(bg_color))

        # Weather stripe across the top, only present for the ~7 forecast days
        weather_score = self.weather_scores.get(py_date)
        if weather_score is not None:
            stripe_rect = rect.adjusted(0, 0, 0, -(rect.height() - 6))
            painter.fillRect(stripe_rect, QBrush(QColor(get_rating_color(weather_score))))

        # Draw text
        painter.setPen(QPen(fg_color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(date.day()))

        # Draw selection/today highlight
        if date == self.selectedDate():
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        elif py_date == date_cls.today():
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

    def leaveEvent(self, event):
        """Hide tooltip when mouse leaves the calendar"""
        super().leaveEvent(event)
        self.tooltip_label.hide()

    def _show_tooltip_for_date(self, py_date, global_pos):
        """py_date is a plain datetime.date (or None); global_pos is the screen
        position to anchor the tooltip near."""
        from WeatherForecast import get_rating_label

        if py_date is None or py_date.month != self.monthShown() or py_date.year != self.yearShown():
            self.tooltip_label.hide()
            return

        lines = []
        if py_date in self.visibility_hours:
            lines.append(f"{self.visibility_hours[py_date]:.1f}h visible")
        if py_date in self.moon_illumination:
            lines.append(f"Moon: {self.moon_illumination[py_date] * 100:.0f}% illuminated")
        if py_date in self.weather_scores:
            score = self.weather_scores[py_date]
            weather_line = f"Weather: {score}/100 ({get_rating_label(score)})"
            detail = self.weather_details.get(py_date)
            if detail:
                if detail.get("cloud_cover") is not None:
                    weather_line += f", {detail['cloud_cover']:.0f}% clouds"
                if detail.get("seeing"):
                    weather_line += f", seeing {detail['seeing']}"
            lines.append(weather_line)

        if not lines:
            self.tooltip_label.hide()
            return

        self.tooltip_label.setText("\n".join(lines))
        self.tooltip_label.adjustSize()
        self.tooltip_label.move(global_pos.x() + 15, global_pos.y() + 15)
        self.tooltip_label.show()
        self.tooltip_label.raise_()


class MultiMonthVisibilityCalendar(QWidget):
    """1-3 side-by-side VisibilityCalendar instances kept chronologically
    chained, with a DSO-visibility layer for whichever DSO was last calculated
    and a weather layer for the app's default active location. Modeled on
    SessionManager.SessionCalendarWidget, minus session markers and the
    per-session location override (there's no "session" concept here - just
    whatever DSO the user last calculated, always against the default location,
    same as the rest of this file)."""

    MONTH_COUNT_SETTING = "dso_visibility_calendar_month_count"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calendars = []
        self._dso_coord = None
        self._dso_name = None
        self._ra_deg = None
        self._dec_deg = None
        self._min_altitude = 30
        self.on_date_clicked = None  # callable(date), set by the owning window
        self._weather_worker = None
        self._visibility_thread = None

        today = QDate.currentDate()
        self.anchor_year, self.anchor_month = today.year(), today.month()
        self.month_count = self._load_month_count()

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_controls())
        layout.addLayout(self._build_hours_legend())

        self.calendars_layout = QHBoxLayout()
        layout.addLayout(self.calendars_layout, 1)

        self.status_label = QLabel("Select a DSO and calculate to see monthly visibility")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-style: italic;")
        layout.addWidget(self.status_label)

        self._rebuild_calendars()

    def _build_controls(self):
        row = QHBoxLayout()

        row.addWidget(QLabel("Months:"))
        self.month_count_combo = QComboBox()
        self.month_count_combo.addItems(["1 Month", "2 Months", "3 Months"])
        self.month_count_combo.setCurrentIndex(self.month_count - 1)
        self.month_count_combo.currentIndexChanged.connect(self._on_month_count_changed)
        row.addWidget(self.month_count_combo)

        prev_btn = QPushButton("◀ Previous")
        prev_btn.clicked.connect(self._go_previous)
        row.addWidget(prev_btn)

        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._go_next)
        row.addWidget(next_btn)

        row.addStretch()

        row.addWidget(QLabel("Weather (top stripe):"))
        for label, color in (
            ("Excellent", COLORS['success']), ("Good", COLORS['info']),
            ("Moderate", COLORS['warning']), ("Poor", COLORS['error']),
        ):
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid {COLORS['border']};")
            row.addWidget(swatch)
            row.addWidget(QLabel(label))

        return row

    def _build_hours_legend(self):
        row = QHBoxLayout()
        legend_labels = [
            ("8+ hrs", QColor(0, 150, 0)),
            ("6-8 hrs", QColor(0, 120, 0)),
            ("4-6 hrs", QColor(100, 120, 0)),
            ("2-4 hrs", QColor(150, 100, 0)),
            ("1-2 hrs", QColor(150, 60, 0)),
            ("<1 hr", QColor(120, 40, 0)),
            ("None", QColor(80, 0, 0)),
        ]
        for text, color in legend_labels:
            legend_label = QLabel(text)
            legend_label.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                "padding: 3px; border-radius: 2px;")
            row.addWidget(legend_label)
        row.addStretch()
        return row

    def _load_month_count(self):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        return max(1, min(3, settings.value(self.MONTH_COUNT_SETTING, 1, type=int)))

    def _save_month_count(self, count):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue(self.MONTH_COUNT_SETTING, count)

    def _on_month_count_changed(self, index):
        self.month_count = index + 1
        self._save_month_count(self.month_count)
        self._rebuild_calendars()

    @staticmethod
    def _next_month(year, month):
        return (year + 1, 1) if month == 12 else (year, month + 1)

    @staticmethod
    def _prev_month(year, month):
        return (year - 1, 12) if month == 1 else (year, month - 1)

    def _display_range(self):
        start = date_cls(self.anchor_year, self.anchor_month, 1)
        last_year, last_month = self.anchor_year, self.anchor_month
        for _ in range(self.month_count - 1):
            last_year, last_month = self._next_month(last_year, last_month)
        last_day = calendar.monthrange(last_year, last_month)[1]
        end = date_cls(last_year, last_month, last_day)
        return start, end

    def _rebuild_calendars(self):
        for cal in self.calendars:
            cal.setParent(None)
            cal.deleteLater()
        self.calendars = []

        while self.calendars_layout.count():
            self.calendars_layout.takeAt(0)

        year, month = self.anchor_year, self.anchor_month
        for i in range(self.month_count):
            cal = VisibilityCalendar()
            cal.setMaximumHeight(250)
            cal.setCurrentPage(year, month)
            if i == 0:
                cal.currentPageChanged.connect(self._on_calendar0_page_changed)
            else:
                cal.setNavigationBarVisible(False)
            cal.clicked.connect(self._on_date_clicked)
            self.calendars_layout.addWidget(cal)
            self.calendars.append(cal)
            year, month = self._next_month(year, month)

        self._trigger_recalculation()

    def _set_anchor(self, year, month):
        self.anchor_year, self.anchor_month = year, month
        y, m = year, month
        for cal in self.calendars:
            cal.blockSignals(True)
            cal.setCurrentPage(y, m)
            cal.blockSignals(False)
            y, m = self._next_month(y, m)
        self._trigger_recalculation()

    def _on_calendar0_page_changed(self, year, month):
        self._set_anchor(year, month)

    def _go_previous(self):
        year, month = self._prev_month(self.anchor_year, self.anchor_month)
        self._set_anchor(year, month)

    def _go_next(self):
        year, month = self._next_month(self.anchor_year, self.anchor_month)
        self._set_anchor(year, month)

    def _on_date_clicked(self, qdate):
        if self.on_date_clicked:
            self.on_date_clicked(qdate)

    def _trigger_recalculation(self):
        self._refresh_weather_layer()
        self._refresh_visibility_layer()

    def set_dso(self, dso_coord, dso_name, ra_deg=None, dec_deg=None, min_altitude=30):
        """Set the DSO to show visibility for and (re)start the calculation."""
        self._dso_coord = dso_coord
        self._dso_name = dso_name
        self._ra_deg = ra_deg
        self._dec_deg = dec_deg
        self._min_altitude = min_altitude
        self._refresh_visibility_layer()

    def _refresh_visibility_layer(self):
        if self._dso_coord is None and self._dso_name is None:
            for cal in self.calendars:
                cal.clear_visibility()
            return

        # Never replace a still-running thread's reference - PySide6 can hard-crash
        # (native, uncatchable) if a QThread object is garbage-collected while its
        # underlying OS thread is still executing.
        if self._visibility_thread and self._visibility_thread.isRunning():
            self._visibility_thread.quit()
            self._visibility_thread.wait()

        start_date, end_date = self._display_range()

        for cal in self.calendars:
            cal.clear_visibility()
        dso_label = self._dso_name or "DSO"
        self.status_label.setText(f"Calculating monthly visibility for {dso_label}...")

        self._visibility_thread = MonthlyVisibilityThread(
            self._dso_coord, self._dso_name, start_date, end_date, self._min_altitude,
            self._ra_deg, self._dec_deg)
        self._visibility_thread.progress.connect(self._on_visibility_progress)
        self._visibility_thread.finished.connect(self._on_visibility_finished)
        self._visibility_thread.error.connect(self._on_visibility_error)
        self._visibility_thread.start()

    def _on_visibility_progress(self, completed, total):
        dso_label = self._dso_name or "DSO"
        self.status_label.setText(f"Calculating {dso_label}: {completed}/{total} days...")

    def _on_visibility_finished(self, payload):
        hours_by_date, moon_by_date = payload
        for cal in self.calendars:
            cal.set_data(visibility_hours=hours_by_date, moon_illumination=moon_by_date)
        dso_label = self._dso_name or "DSO"
        self.status_label.setText(f"Monthly visibility for {dso_label} (hover a day for details)")

    def _on_visibility_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")

    def _refresh_weather_layer(self):
        calc = DSOVisibilityCalculator()
        if calc.location is None:
            self._apply_weather_scores({}, {})
            return

        lat = calc.location.lat.deg
        lon = calc.location.lon.deg
        tz = getattr(calc.timezone, 'zone', 'UTC')

        from WeatherForecast import WeatherCache, WeatherWorker

        cache = WeatherCache()
        cached = cache.get(lat, lon)
        if cached:
            self._apply_weather_list(cached)
            return

        if self._weather_worker and self._weather_worker.isRunning():
            self._weather_worker.quit()
            self._weather_worker.wait()

        self._weather_worker = WeatherWorker(lat, lon, tz)
        self._weather_worker.weather_loaded.connect(lambda data: self._on_weather_loaded(data, lat, lon))
        self._weather_worker.error_occurred.connect(
            lambda msg: logger.debug(f"Calendar weather fetch failed: {msg}")
        )
        self._weather_worker.start()

    def _on_weather_loaded(self, data, lat, lon):
        from WeatherForecast import WeatherCache
        WeatherCache().set(lat, lon, data)
        self._apply_weather_list(data)

    def _apply_weather_list(self, daily_summaries):
        scores = {}
        details = {}
        for summary in daily_summaries:
            d = summary.date.date()
            scores[d] = summary.astro_score
            details[d] = {
                "cloud_cover": summary.tonight_avg_cloud_cover,
                "seeing": summary.seeing_estimate,
            }
        self._apply_weather_scores(scores, details)

    def _apply_weather_scores(self, scores, details=None):
        details = details if details is not None else {}
        for cal in self.calendars:
            cal.set_data(weather_scores=scores, weather_details=details)

    def shutdown(self):
        """Stop any running background threads - call from the owning window's closeEvent."""
        if self._visibility_thread and self._visibility_thread.isRunning():
            self._visibility_thread.quit()
            self._visibility_thread.wait()
        if self._weather_worker and self._weather_worker.isRunning():
            self._weather_worker.quit()
            self._weather_worker.wait()
        for cal in self.calendars:
            cal.tooltip_label.hide()
            cal.tooltip_label.destroy()


class CalculationThread(QThread):
    """Thread for performing visibility calculations using centralized calculator"""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dso_name, date, hours, min_altitude, ra_deg=None, dec_deg=None):
        super().__init__()
        self.dso_name = dso_name
        self.date = date
        self.hours = hours
        self.min_altitude = min_altitude
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg

        # Use centralized calculator
        self.calculator = DSOVisibilityCalculator()
        self.location = self.calculator.location
        self.local_tz = self.calculator.timezone

    def run(self):
        """Main calculation thread using centralized calculator"""
        try:
            if self.location is None:
                self.error.emit("Observer location not configured. Please set your location in settings.")
                return

            # If coordinates are provided, use them directly to avoid name resolution
            if self.ra_deg is not None and self.dec_deg is not None:
                from astropy.coordinates import SkyCoord
                from astropy import units as u

                dso_coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
                results = self.calculator.calculate_visibility_for_coordinates(
                    dso_coord, self.date, self.hours, self.min_altitude, self.dso_name)
            else:
                # Use name-based calculation (original behavior)
                results = self.calculator.calculate_visibility_for_date(
                    self.dso_name, self.date, self.hours, self.min_altitude)

            if "error" in results:
                self.error.emit(results["error"])
                return

            # Add local timezone for compatibility with existing UI code
            results['local_tz'] = self.local_tz

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Calculation error: {str(e)}")


class VisibilityPlot(FigureCanvas):
    """Custom matplotlib canvas for PySide6"""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 8), facecolor='#2e2e2e')
        super().__init__(self.figure)
        self.setParent(parent)
        # Set dark background for the canvas
        self.setStyleSheet(f"background-color: {COLORS['background']};")

        # Initialize hover data storage
        self.hover_data = None
        self.annotation = None
        self.cursor_lines = []
        self.last_idx = None  # Cache last index for performance
        self.last_tooltip_text = None  # Cache last tooltip content

        # Initialize current time line storage
        self.current_time_lines = []

        # Set up timer for live current time updates (every 30 seconds)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_current_time_line)
        self.update_timer.setInterval(30000)  # 30 seconds in milliseconds

        # Create custom floating tooltip widget (Qt-based, not matplotlib)
        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import Qt as QtCore
        self.qt_tooltip = QLabel(parent if parent else self)
        self.qt_tooltip.setWindowFlags(QtCore.ToolTip | QtCore.FramelessWindowHint | QtCore.WindowStaysOnTopHint)
        self.qt_tooltip.setAttribute(QtCore.WA_TranslucentBackground, False)  # Reduce flicker
        self.qt_tooltip.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['background_lighter']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 3px;
                padding: 6px 10px;
                font-size: 9pt;
                font-family: monospace;
            }}
        """)
        self.qt_tooltip.hide()

        # Connect mouse motion event
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        # Connect mouse leave event
        self.mpl_connect('axes_leave_event', self.on_mouse_leave)

    def _add_darkness_shading(self, ax, hours_from_start, sun_altitudes):
        """Add background shading to show darkness levels based on sun altitude.

        Colors represent:
        - Daylight (sun > 0°): No shading (default background)
        - Civil twilight (0° to -6°): Light shade
        - Nautical twilight (-6° to -12°): Medium shade
        - Astronomical twilight (-12° to -18°): Dark shade
        - Night (< -18°): Darkest shade
        """
        # Define darkness thresholds and colors (from lightest to darkest)
        # Format: (sun_max, sun_min, color, alpha)
        darkness_levels = [
            (0, -6, '#2a3a4a', 0.6),      # Civil twilight - light blue-gray
            (-6, -12, '#1a2535', 0.7),    # Nautical twilight - medium blue
            (-12, -18, '#101520', 0.8),   # Astronomical twilight - dark blue
            (-18, -90, '#080a10', 0.9),   # Night - very dark blue/black
        ]

        # For each darkness level, find and shade the regions
        for sun_max, sun_min, color, alpha in darkness_levels:
            # Find indices where sun is in this range
            in_range = (sun_altitudes <= sun_max) & (sun_altitudes > sun_min)

            if not np.any(in_range):
                continue

            # Find contiguous regions
            diff = np.diff(np.concatenate(([False], in_range, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # Shade each contiguous region
            for start_idx, end_idx in zip(starts, ends):
                x_start = hours_from_start[start_idx]
                x_end = hours_from_start[min(end_idx, len(hours_from_start) - 1)]
                ax.axvspan(x_start, x_end, facecolor=color, alpha=alpha, zorder=0)

    def plot_visibility(self, results):
        """Create visibility plot with altitude and azimuth"""
        self.figure.clear()
        # Reset subplot parameters to defaults so tight_layout() doesn't
        # progressively shrink the plots on repeated calculations
        self.figure.subplots_adjust(
            left=0.125, bottom=0.11, right=0.9, top=0.88,
            wspace=0.2, hspace=0.2)

        time_range = results['time_range']
        dso_altaz = results['dso_altaz']
        sun_altaz = results['sun_altaz']
        optimal_times = results['optimal_times']
        dso_name = results['dso_name']
        local_tz = results['local_tz']

        # Store data for hover functionality
        self.hover_data = {
            'time_range': time_range,
            'dso_altaz': dso_altaz,
            'sun_altaz': sun_altaz,
            'optimal_times': optimal_times,
            'dso_name': dso_name,
            'local_tz': local_tz
        }

        # Convert times to local timezone for display
        local_times = []
        for t in time_range:
            utc_dt = t.datetime.replace(tzinfo=pytz.UTC)
            local_dt = utc_dt.astimezone(local_tz)
            local_times.append(local_dt)

        # Convert to hours from start for plotting, but use local time labels
        start_local = local_times[0]
        hours_from_start = [(lt - start_local).total_seconds() / 3600 for lt in local_times]

        # Get sun altitudes as numpy array for shading calculations
        sun_altitudes = sun_altaz.alt.deg

        # Create three subplots: altitude, azimuth, and sun
        ax1 = self.figure.add_subplot(3, 1, 1, facecolor='#2e2e2e')

        # Add darkness shading first (so it appears behind the data)
        self._add_darkness_shading(ax1, hours_from_start, sun_altitudes)

        ax1.plot(hours_from_start, dso_altaz.alt.deg, '#00aaff', linewidth=2, label=f'{dso_name} Altitude')
        ax1.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.8, label='Min Altitude (30°)')
        ax1.axhline(y=0, color='#888888', linestyle='-', alpha=0.6, label='Horizon')

        # Highlight optimal viewing times
        optimal_alt = np.where(optimal_times, dso_altaz.alt.deg, np.nan)
        ax1.plot(hours_from_start, optimal_alt, '#ff4444', linewidth=4, alpha=0.8, label='Optimal Viewing')

        ax1.set_ylabel('Altitude (°)', color='white')
        # Get timezone abbreviation for display
        sample_time = local_times[0] if local_times else None
        tz_abbrev = sample_time.strftime('%Z') if sample_time else 'Local Time'
        ax1.set_title(f'{dso_name} Visibility ({tz_abbrev})', color='white', fontsize=14)
        ax1.legend(facecolor='#404040', edgecolor='#666666', loc='upper right')
        ax1.grid(True, alpha=0.3, color='#666666')
        ax1.set_ylim(-20, 90)
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('#666666')

        # Add time labels at key points
        time_ticks = []
        time_labels = []
        for i in range(0, len(hours_from_start), max(1, len(hours_from_start) // 6)):
            time_ticks.append(hours_from_start[i])
            time_labels.append(format_time(local_times[i]))
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(time_labels)

        # DSO azimuth subplot
        ax2 = self.figure.add_subplot(3, 1, 2, facecolor='#2e2e2e')

        # Add darkness shading
        self._add_darkness_shading(ax2, hours_from_start, sun_altitudes)

        ax2.plot(hours_from_start, dso_altaz.az.deg, '#ff8800', linewidth=2, label=f'{dso_name} Azimuth')

        # Add cardinal direction lines
        ax2.axhline(y=0, color='#ff4444', linestyle=':', alpha=0.7, label='N')
        ax2.axhline(y=90, color='#44ff44', linestyle=':', alpha=0.7, label='E')
        ax2.axhline(y=180, color='#ffff44', linestyle=':', alpha=0.7, label='S')
        ax2.axhline(y=270, color='#4444ff', linestyle=':', alpha=0.7, label='W')

        # Highlight optimal viewing times for azimuth too
        optimal_az = np.where(optimal_times, dso_altaz.az.deg, np.nan)
        ax2.plot(hours_from_start, optimal_az, '#ff4444', linewidth=4, alpha=0.8, label='Optimal Viewing')

        ax2.set_ylabel('Azimuth (°)', color='white')
        ax2.legend(facecolor='#404040', edgecolor='#666666', loc='upper right')
        ax2.grid(True, alpha=0.3, color='#666666')
        ax2.set_ylim(0, 360)
        ax2.set_yticks([0, 90, 180, 270, 360])
        ax2.set_yticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('#666666')
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)

        # Sun altitude subplot
        ax3 = self.figure.add_subplot(3, 1, 3, facecolor='#2e2e2e')

        # Add darkness shading
        self._add_darkness_shading(ax3, hours_from_start, sun_altitudes)

        ax3.plot(hours_from_start, sun_altaz.alt.deg, '#ffaa00', linewidth=2, label='Sun Altitude')
        ax3.axhline(y=0, color='#888888', linestyle='-', alpha=0.6, label='Horizon')
        ax3.axhline(y=-12, color='#4488ff', linestyle='--', alpha=0.8, label='Astronomical Twilight')
        ax3.axhline(y=-18, color='#aa44ff', linestyle='--', alpha=0.8, label='Night')

        # Use the same timezone abbreviation as in title
        sample_time = local_times[0] if local_times else None
        tz_abbrev = sample_time.strftime('%Z') if sample_time else 'Local Time'
        ax3.set_xlabel(f'Time ({tz_abbrev})', color='white')
        ax3.set_ylabel('Sun Alt. (°)', color='white')
        ax3.legend(facecolor='#404040', edgecolor='#666666', loc='upper right')
        ax3.grid(True, alpha=0.3, color='#666666')
        ax3.set_ylim(-25, 50)
        ax3.tick_params(colors='white')
        for spine in ax3.spines.values():
            spine.set_color('#666666')
        ax3.set_xticks(time_ticks)
        ax3.set_xticklabels(time_labels)

        self.figure.tight_layout()

        # Store additional data needed for hover and current time updates
        self.hover_data['local_times'] = local_times
        self.hover_data['hours_from_start'] = hours_from_start
        self.hover_data['axes'] = [ax1, ax2, ax3]
        self.hover_data['start_local'] = start_local

        # Reset cursor lines and cached index
        self.cursor_lines = []
        self.last_idx = None
        self.last_tooltip_text = None

        # Clear old current time lines
        for line in self.current_time_lines:
            try:
                line.remove()
            except:
                pass
        self.current_time_lines = []

        # Add initial current time line
        self.update_current_time_line()

        # Start the timer for live updates
        self.update_timer.start()

        self.draw()

    def update_current_time_line(self):
        """Update the current time line to reflect the actual current time"""
        if not self.hover_data or 'start_local' not in self.hover_data:
            return

        from astropy.time import Time as AstropyTime

        # Get current time using astropy (same as plotted data)
        current_astropy_time = AstropyTime.now()

        # Convert to local time using same process as plotted times
        local_tz = self.hover_data.get('local_tz')
        if not local_tz:
            return

        current_utc_dt = current_astropy_time.datetime.replace(tzinfo=pytz.UTC)
        current_local_dt = current_utc_dt.astimezone(local_tz)

        # Calculate current time position in hours from start
        start_local = self.hover_data['start_local']
        current_hours_from_start = (current_local_dt - start_local).total_seconds() / 3600

        hours_from_start = self.hover_data.get('hours_from_start', [])
        if not hours_from_start:
            return

        # Remove old current time lines
        for line in self.current_time_lines:
            try:
                line.remove()
            except:
                pass
        self.current_time_lines = []

        # Draw the line if within a reasonable range of the plot
        if -1 <= current_hours_from_start <= hours_from_start[-1] + 1:
            axes = self.hover_data.get('axes', [])
            # Draw vertical line on all three subplots
            for ax in axes:
                line = ax.axvline(x=current_hours_from_start, color='#00ff00', linestyle='--',
                          linewidth=2.5, alpha=0.9, label='Current Time', zorder=100)
                self.current_time_lines.append(line)

            # Update legend on first subplot to include current time (only if lines were drawn)
            if axes and self.current_time_lines:
                axes[0].legend(facecolor='#404040', edgecolor='#666666', loc='upper right')

            # Redraw the canvas
            self.draw_idle()

    def azimuth_to_direction(self, az):
        """Convert azimuth to cardinal direction using centralized method"""
        return DSOVisibilityCalculator.azimuth_to_direction(az)

    def get_twilight_condition(self, sun_alt):
        """Get twilight condition using centralized method"""
        return DSOVisibilityCalculator.get_twilight_condition(sun_alt)

    def find_nearest_data_point(self, x_pos):
        """Find the nearest data point to the mouse position"""
        if not self.hover_data or 'hours_from_start' not in self.hover_data:
            return None

        hours_from_start = self.hover_data['hours_from_start']
        idx = np.argmin(np.abs(np.array(hours_from_start) - x_pos))
        return idx

    def clear_cursor_elements(self):
        """Clear existing cursor lines and annotation"""
        # Remove cursor lines
        for line in self.cursor_lines:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass  # Already removed or cannot be removed
        self.cursor_lines = []

        # Remove annotation - set visibility to False instead of removing
        if self.annotation:
            try:
                self.annotation.set_visible(False)
            except (ValueError, AttributeError):
                pass  # Already removed or invalid
            self.annotation = None

        # Hide Qt tooltip and reset cache
        if hasattr(self, 'qt_tooltip'):
            self.qt_tooltip.hide()
        self.last_tooltip_text = None

    def on_mouse_move(self, event):
        """Handle mouse movement for hover tooltips with vertical cursor line"""
        if not self.hover_data or event.inaxes is None or event.xdata is None:
            self.clear_cursor_elements()
            self.last_idx = None
            self.last_tooltip_text = None
            self.draw_idle()
            return

        # Find nearest data point
        idx = self.find_nearest_data_point(event.xdata)
        if idx is None:
            return

        # Performance optimization: skip if same data point as last time
        if idx == self.last_idx:
            return
        self.last_idx = idx

        # Clear previous cursor lines only (keep tooltip visible)
        for line in self.cursor_lines:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.cursor_lines = []

        # Get the x-position from our data (for precise alignment)
        hours_from_start = self.hover_data['hours_from_start']
        x_pos = hours_from_start[idx]

        # Add vertical cursor line to all subplots
        axes = self.hover_data.get('axes', [])
        for ax in axes:
            line = ax.axvline(x=x_pos, color='#ffcc00', linestyle='-', alpha=0.7, linewidth=1.5)
            self.cursor_lines.append(line)

        # Get data for this point
        local_time = self.hover_data['local_times'][idx]
        dso_alt = self.hover_data['dso_altaz'].alt.deg[idx]
        dso_az = self.hover_data['dso_altaz'].az.deg[idx]
        sun_alt = self.hover_data['sun_altaz'].alt.deg[idx]
        optimal = self.hover_data['optimal_times'][idx]
        dso_name = self.hover_data['dso_name']

        # Pre-calculate values (avoiding repeated function calls)
        direction = self.azimuth_to_direction(dso_az)
        twilight = self.get_twilight_condition(sun_alt)
        tz_name = local_time.strftime('%Z')

        # Create hover text
        hover_text = (
            f"Time: {format_time(local_time, seconds=True)} {tz_name}\n"
            f"{dso_name} Alt: {dso_alt:.1f}°\n"
            f"{dso_name} Az: {dso_az:.0f}° ({direction})\n"
            f"Sun Alt: {sun_alt:.1f}° ({twilight})\n"
            f"Optimal: {'Yes' if optimal else 'No'}"
        )

        # Only update tooltip if content changed (reduces flicker)
        if hover_text != self.last_tooltip_text:
            self.last_tooltip_text = hover_text

            # Batch updates to reduce flicker
            self.qt_tooltip.setUpdatesEnabled(False)

            # Use Qt tooltip instead of matplotlib annotation (ensures it appears on top of calendar)
            self.qt_tooltip.setText(hover_text)
            self.qt_tooltip.adjustSize()

            # Get mouse position in global screen coordinates
            from PySide6.QtCore import QPoint
            cursor_pos = self.mapToGlobal(QPoint(int(event.x), int(self.height() - event.y)))

            # Smart positioning: offset tooltip to avoid covering data and calendar
            # Get the axis limits to determine position
            x_min, x_max = event.inaxes.get_xlim()
            y_min, y_max = event.inaxes.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Horizontal positioning: left when near right edge, right otherwise
            if x_pos > (x_min + 0.75 * x_range):
                x_offset = -self.qt_tooltip.width() - 15  # Place to the left of cursor
            else:
                x_offset = 15  # Place to the right of cursor

            # Vertical positioning: below when near top (to avoid calendar), above otherwise
            if event.ydata > (y_min + 0.6 * y_range):
                y_offset = 15  # Place below cursor
            else:
                y_offset = -self.qt_tooltip.height() - 15  # Place above cursor

            # Position and show the Qt tooltip
            tooltip_pos = cursor_pos + QPoint(x_offset, y_offset)
            self.qt_tooltip.move(tooltip_pos)

            # Only call show and raise if not already visible
            if not self.qt_tooltip.isVisible():
                self.qt_tooltip.show()
                self.qt_tooltip.raise_()

            # Re-enable updates after all changes are made
            self.qt_tooltip.setUpdatesEnabled(True)

        # Use draw_idle for better performance (for cursor lines)
        self.draw_idle()

    def on_mouse_leave(self, event):
        """Handle mouse leaving the plot area"""
        self.clear_cursor_elements()
        self.draw_idle()

    def leaveEvent(self, event):
        """Hide tooltip when mouse leaves the canvas widget entirely"""
        super().leaveEvent(event)
        if hasattr(self, 'qt_tooltip'):
            self.qt_tooltip.hide()
        self.last_tooltip_text = None
        self.last_idx = None

    def __del__(self):
        """Destructor - clean up the tooltip widget"""
        try:
            if hasattr(self, 'qt_tooltip'):
                self.qt_tooltip.hide()
                self.qt_tooltip.deleteLater()
        except:
            pass  # Ignore errors during cleanup


class DSOVisibilityApp(WindowPositionMixin, QMainWindow):
    """Main application window"""
    WINDOW_POSITION_KEY = "DSOVisibilityApp"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSO Visibility Calculator - Cosmos Collection")
        self.resize(1650, 900)
        self.setup_window_position()

        self.calc_thread = None
        # Store optional coordinates for direct calculation (bypassing name resolution)
        self.dso_ra_deg = None
        self.dso_dec_deg = None
        self.current_dso_coord = None  # Store current DSO coordinates for calendar
        self.current_dso_name = None  # Store current DSO name for window title
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # Location info
        self.location_group = QGroupBox("Observer Location")
        location_layout = QVBoxLayout(self.location_group)
        self.location_name_label = QLabel("Loading...")
        self.location_coords_label = QLabel("Loading...")
        location_layout.addWidget(self.location_name_label)
        location_layout.addWidget(self.location_coords_label)
        left_layout.addWidget(self.location_group)

        # Load location from database
        self._load_location_from_database()

        # Input controls
        input_group = QGroupBox("Observation Parameters")
        input_layout = QVBoxLayout(input_group)

        # DSO name
        input_layout.addWidget(QLabel("Deep Sky Object:"))
        self.dso_input = QLineEdit("M100")
        input_layout.addWidget(self.dso_input)

        # Date
        input_layout.addWidget(QLabel("Date:"))
        self.date_input = QDateEdit(QDate.currentDate())
        self.date_input.setCalendarPopup(True)
        input_layout.addWidget(self.date_input)

        # Hours to calculate
        input_layout.addWidget(QLabel("Hours to calculate:"))
        self.hours_input = QSpinBox()
        self.hours_input.setRange(6, 72)
        self.hours_input.setValue(24)
        input_layout.addWidget(self.hours_input)

        # Minimum altitude
        input_layout.addWidget(QLabel("Minimum altitude (degrees):"))
        self.min_alt_input = QSpinBox()
        self.min_alt_input.setRange(0, 90)
        self.min_alt_input.setValue(30)
        input_layout.addWidget(self.min_alt_input)

        left_layout.addWidget(input_group)

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Visibility")
        self.calculate_btn.clicked.connect(self.calculate_visibility)
        left_layout.addWidget(self.calculate_btn)

        # Results text area
        results_group = QGroupBox("Viewing Windows")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setMinimumHeight(200)
        self.results_text.setReadOnly(True)

        # Set size policy to expand vertically
        self.results_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        results_layout.addWidget(self.results_text)
        left_layout.addWidget(results_group, stretch=1)  # Give it a stretch factor to expand

        # Right panel with calendar and plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Calendar widget
        calendar_group = QGroupBox("Visibility Calendar")
        calendar_group.setMaximumHeight(320)
        calendar_layout = QVBoxLayout(calendar_group)

        self.calendar_widget = MultiMonthVisibilityCalendar()
        self.calendar_widget.on_date_clicked = self.on_calendar_date_selected
        calendar_layout.addWidget(self.calendar_widget)

        right_layout.addWidget(calendar_group)

        # Plot widget
        self.plot_widget = VisibilityPlot()
        right_layout.addWidget(self.plot_widget, stretch=1)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    def _load_location_from_database(self):
        """Load observer location from the database"""
        # Check if observer location should be shown
        from PySide6.QtCore import QSettings
        settings = QSettings("CosmosCollection", "CosmosCollection")
        show_location = settings.value("show_observer_location", True, type=bool)

        # Hide the location group if setting is disabled
        if not show_location:
            self.location_group.setVisible(False)
            return

        # Make sure it's visible if setting is enabled
        self.location_group.setVisible(True)

        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon, location_name, timezone FROM usersettings WHERE is_active = 1 LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT location_lat, location_lon, location_name, timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                if row:
                    lat, lon, location_name, timezone = row
                    self.user_timezone = timezone
                else:
                    lat, lon, location_name, timezone = None, None, None, None
                    self.user_timezone = None

                if lat is not None and lon is not None:
                    # Format coordinates nicely
                    lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
                    lon_str = f"{abs(lon):.2f}°{'W' if lon < 0 else 'E'}"

                    # Use the location name if available, otherwise fall back to "User Location"
                    display_name = location_name if location_name else "User Location"
                    self.location_name_label.setText(display_name)
                    self.location_coords_label.setText(f"Lat: {lat_str}, Lon: {lon_str}")
                else:
                    # No location configured - prompt user to set location
                    self.location_name_label.setText("Location not set")
                    self.location_coords_label.setText("Click 'Set Location' to configure")
                    self.user_timezone = None
                    self._show_location_required()
        except Exception as e:
            # Error accessing database - prompt user to set location
            self.location_name_label.setText("Location not set")
            self.location_coords_label.setText("Click 'Set Location' to configure")
            self.user_timezone = None
            self._show_location_required()

    def on_calculation_finished(self, results):
        """Handle completed calculation"""
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Visibility")

        # Store DSO coordinates for calendar calculations
        self.current_dso_coord = results.get('dso_coord')

        # Update window title with DSO name (use stored name from input, not results)
        if self.current_dso_name:
            self._update_window_title_with_dso(self.current_dso_name)

        # Update plot
        self.plot_widget.plot_visibility(results)

        # Generate text results
        self.update_results_text(results)

        # Start monthly visibility calculation for calendar
        self.calendar_widget.set_dso(
            self.current_dso_coord, self.current_dso_name,
            self.dso_ra_deg, self.dso_dec_deg, self.min_alt_input.value())

    def _update_window_title_with_dso(self, dso_name):
        """Update window title to show DSO name and common name if available"""
        try:
            from DatabaseManager import DatabaseManager

            # Try to get common name from database
            common_name = None

            # Parse the catalog and designation from the DSO name
            import re
            match = re.match(r'^([A-Z]+)\s*(\d+)', dso_name.upper())
            if match:
                catalog = match.group(1)
                designation = match.group(2)

                # Query database for common name
                db_manager = DatabaseManager()
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT d.commonnames
                        FROM dsodetail d
                        JOIN cataloguenr c ON d.id = c.dsodetailid
                        WHERE c.catalogue = ? AND c.designation = ?
                    """, (catalog, designation))

                    result = cursor.fetchone()
                    if result and result[0]:
                        common_name = result[0]

            # Build window title
            if common_name:
                title = f"DSO Visibility Calculator - {dso_name} ({common_name}) - Cosmos Collection"
            else:
                title = f"DSO Visibility Calculator - {dso_name} - Cosmos Collection"

            self.setWindowTitle(title)

        except Exception as e:
            # Fallback to simple title if any error
            self.setWindowTitle(f"DSO Visibility Calculator - {dso_name} - Cosmos Collection")

    def on_calculation_error(self, error_msg):
        """Handle calculation error"""
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Visibility")

        self.results_text.setText(f"Error: {error_msg}")
        QMessageBox.warning(self, "Calculation Error", error_msg)

    def update_results_text(self, results):
        """Update the results text area"""
        time_range = results['time_range']
        optimal_times = results['optimal_times']
        dso_name = results['dso_name']
        dso_altaz = results['dso_altaz']
        dso_coord = results['dso_coord']
        local_tz = results['local_tz']

        text = f"Results for {dso_name}:\n"
        text += f"RA: {dso_coord.ra.to_string(unit=u.hour, precision=1)}\n"
        text += f"Dec: {dso_coord.dec.to_string(unit=u.deg, precision=1)}\n\n"

        # Find maximum altitude and its direction
        max_alt_idx = np.argmax(dso_altaz.alt.deg)
        max_alt_time_utc = time_range[max_alt_idx].datetime.replace(tzinfo=pytz.UTC)
        max_alt_time_local = max_alt_time_utc.astimezone(local_tz)
        max_altitude = dso_altaz.alt.deg[max_alt_idx]
        max_azimuth = dso_altaz.az.deg[max_alt_idx]

        # Convert azimuth to cardinal direction using centralized method
        max_direction = DSOVisibilityCalculator.azimuth_to_direction(max_azimuth)

        # Determine if we're in EST or EDT
        tz_name = max_alt_time_local.strftime('%Z')

        text += f"Maximum altitude: {max_altitude:.1f}° at {format_time(max_alt_time_local)} {tz_name}\n"
        text += f"Direction at max altitude: {max_direction} ({max_azimuth:.0f}°)\n\n"

        # Find viewing windows
        if not np.any(optimal_times):
            text += "No optimal viewing windows found in this time period.\n"
            text += "Try adjusting the minimum altitude or date range."
        else:
            text += f"Optimal viewing windows ({tz_name}):\n"
            text += "=" * 30 + "\n"

            # Find continuous viewing windows
            diff = np.diff(np.concatenate(([False], optimal_times, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for start_idx, end_idx in zip(starts, ends):
                start_time_utc = time_range[start_idx].datetime.replace(tzinfo=pytz.UTC)
                end_time_utc = time_range[end_idx - 1].datetime.replace(tzinfo=pytz.UTC)
                start_time_local = start_time_utc.astimezone(local_tz)
                end_time_local = end_time_utc.astimezone(local_tz)
                duration = (end_time_utc - start_time_utc).total_seconds() / 3600

                # Get azimuth range during this window
                start_az = dso_altaz.az.deg[start_idx]
                end_az = dso_altaz.az.deg[end_idx - 1]
                mid_idx = (start_idx + end_idx) // 2
                mid_az = dso_altaz.az.deg[mid_idx]
                mid_direction = DSOVisibilityCalculator.azimuth_to_direction(mid_az)

                text += f"From: {format_time(start_time_local)} {tz_name}\n"
                text += f"To:   {format_time(end_time_local)} {tz_name}\n"
                text += f"Duration: {duration:.1f} hours\n"
                text += f"Mid-window direction: {mid_direction} ({mid_az:.0f}°)\n"
                text += f"Azimuth range: {start_az:.0f}° → {end_az:.0f}°\n"
                text += "-" * 20 + "\n"

        self.results_text.setText(text)

    def _show_location_required(self):
        """Show message that location is required for calculations"""
        # Disable calculate button when no location is set
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Location Required")
        self.results_text.setText("Please set your observer location to calculate DSO visibility.")

    def calculate_visibility(self):
        """Start visibility calculation"""
        if self.calc_thread and self.calc_thread.isRunning():
            return

        # Check if location is configured
        calc_thread_test = CalculationThread("M1", "2024-01-01", 24, 30)
        if calc_thread_test.location is None:
            QMessageBox.warning(self, "Location Required",
                              "Please set your observer location from the main window's Settings menu.")
            return

        # Disable button during calculation
        self.calculate_btn.setEnabled(True)  # Re-enable if we got here
        self.calculate_btn.setText("Calculating...")
        self.results_text.setText("Calculating visibility...")

        # Get parameters
        dso_name = self.dso_input.text().strip() or "M100"
        date = self.date_input.date().toString("yyyy-MM-dd")
        hours = self.hours_input.value()
        min_altitude = self.min_alt_input.value()

        # Store DSO name for window title
        self.current_dso_name = dso_name

        # Start calculation thread with optional coordinates
        self.calc_thread = CalculationThread(dso_name, date, hours, min_altitude,
                                            self.dso_ra_deg, self.dso_dec_deg)
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.start()

    def on_calendar_date_selected(self, date):
        """Handle calendar date selection"""
        # Update the date input
        self.date_input.setDate(date)

        # If we have DSO data, recalculate for this date
        if self.current_dso_coord:
            self.calculate_visibility()

    def set_dso_coordinates(self, ra_deg, dec_deg):
        """
        Set DSO coordinates for direct calculation (bypassing name resolution).

        Args:
            ra_deg (float): Right Ascension in degrees
            dec_deg (float): Declination in degrees
        """
        self.dso_ra_deg = ra_deg
        self.dso_dec_deg = dec_deg

    def closeEvent(self, event):
        """Handle window close event - clean up tooltips and threads"""
        # Immediately destroy tooltip native windows (destroy() is synchronous,
        # unlike deleteLater() which only schedules deletion for the next event loop
        # tick — which may not run, leaving floating tooltips visible after close)
        if hasattr(self, 'plot_widget') and hasattr(self.plot_widget, 'qt_tooltip'):
            self.plot_widget.qt_tooltip.hide()
            self.plot_widget.qt_tooltip.destroy()

        # Stop any running calculation threads
        if self.calc_thread and self.calc_thread.isRunning():
            self.calc_thread.quit()
            self.calc_thread.wait()

        if hasattr(self, 'calendar_widget'):
            self.calendar_widget.shutdown()

        # Call parent closeEvent
        super().closeEvent(event)
