#!/usr/bin/env python3
"""
Deep Sky Object Visibility Calculator
Uses astropy and PySide6 to determine when DSOs are optimally visible

Contains the centralized DSOVisibilityCalculator class for all visibility calculations
"""

import sys
import os
import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QDate, QThread, Signal
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLineEdit, QLabel, QTextEdit,
                               QDateEdit, QSpinBox, QGroupBox, QMessageBox, QDialog,
                               QComboBox)

matplotlib.use('Qt5Agg')
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

# Get the application directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Import DatabaseManager from separate file
from DatabaseManager import DatabaseManager


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


class CalculationThread(QThread):
    """Thread for performing visibility calculations using centralized calculator"""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dso_name, date, hours, min_altitude):
        super().__init__()
        self.dso_name = dso_name
        self.date = date
        self.hours = hours
        self.min_altitude = min_altitude
        
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
            
            # Use centralized calculator for all calculations
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
        self.setStyleSheet("background-color: #2e2e2e;")
        
        # Initialize hover data storage
        self.hover_data = None
        self.annotation = None
        self.cursor_lines = []
        self.last_idx = None  # Cache last index for performance
        
        # Connect mouse motion event
        self.mpl_connect('motion_notify_event', self.on_mouse_move)

    def plot_visibility(self, results):
        """Create visibility plot with altitude and azimuth"""
        self.figure.clear()

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

        # Create three subplots: altitude, azimuth, and sun
        ax1 = self.figure.add_subplot(3, 1, 1, facecolor='#2e2e2e')
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
            time_labels.append(local_times[i].strftime('%H:%M'))
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(time_labels)

        # DSO azimuth subplot
        ax2 = self.figure.add_subplot(3, 1, 2, facecolor='#2e2e2e')
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
        
        # Store additional data needed for hover
        self.hover_data['local_times'] = local_times
        self.hover_data['hours_from_start'] = hours_from_start
        self.hover_data['axes'] = [ax1, ax2, ax3]
        
        # Reset cursor lines and cached index
        self.cursor_lines = []
        self.last_idx = None
        
        self.draw()

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

    def on_mouse_move(self, event):
        """Handle mouse movement for hover tooltips with vertical cursor line"""
        if not self.hover_data or event.inaxes is None or event.xdata is None:
            self.clear_cursor_elements()
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

        # Clear previous elements
        self.clear_cursor_elements()

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
            f"Time: {local_time.strftime('%H:%M:%S')} {tz_name}\n"
            f"{dso_name} Alt: {dso_alt:.1f}°\n"
            f"{dso_name} Az: {dso_az:.0f}° ({direction})\n"
            f"Sun Alt: {sun_alt:.1f}° ({twilight})\n"
            f"Optimal: {'Yes' if optimal else 'No'}"
        )

        # Create new annotation on the current axis
        self.annotation = event.inaxes.annotate(
            hover_text,
            xy=(x_pos, event.ydata),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='#404040', alpha=0.9, edgecolor='#666666'),
            fontsize=9,
            color='white',
            fontfamily='monospace',
            zorder=1000
        )
        
        # Use draw_idle for better performance
        self.draw_idle()


class LocationSettingsDialog(QDialog):
    """Dialog for setting observer location when none is configured"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Observer Location - DSO Visibility Calculator")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(500, 350)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the location settings dialog UI"""
        layout = QVBoxLayout()
        
        # Info message
        info_label = QLabel("""<b>Location Required</b><br>
        To calculate DSO visibility, please set your observer location.""")
        info_label.setStyleSheet("QLabel { margin-bottom: 10px; }")
        layout.addWidget(info_label)
        
        # Location group
        location_group = QGroupBox("Observer Location")
        location_group_layout = QVBoxLayout(location_group)
        
        # Latitude input
        lat_layout = QHBoxLayout()
        lat_label = QLabel("Latitude (degrees):")
        lat_label.setMinimumWidth(120)
        self.latitude_input = QLineEdit()
        self.latitude_input.setPlaceholderText("e.g., 40.7128 (positive for North, negative for South)")
        lat_layout.addWidget(lat_label)
        lat_layout.addWidget(self.latitude_input)
        location_group_layout.addLayout(lat_layout)
        
        # Longitude input
        lon_layout = QHBoxLayout()
        lon_label = QLabel("Longitude (degrees):")
        lon_label.setMinimumWidth(120)
        self.longitude_input = QLineEdit()
        self.longitude_input.setPlaceholderText("e.g., -74.0060 (positive for East, negative for West)")
        lon_layout.addWidget(lon_label)
        lon_layout.addWidget(self.longitude_input)
        location_group_layout.addLayout(lon_layout)
        
        # Location name (optional)
        name_layout = QHBoxLayout()
        name_label = QLabel("Location Name:")
        name_label.setMinimumWidth(120)
        self.location_name_input = QLineEdit()
        self.location_name_input.setPlaceholderText("e.g., New York City (optional)")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.location_name_input)
        location_group_layout.addLayout(name_layout)
        
        layout.addWidget(location_group)
        
        # Timezone settings group
        timezone_group = QGroupBox("Time Zone Settings")
        timezone_group_layout = QVBoxLayout(timezone_group)
        
        tz_layout = QHBoxLayout()
        tz_label = QLabel("Time Zone:")
        tz_label.setMinimumWidth(120)
        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
        
        # Add common timezones
        common_timezones = [
            "America/New_York",
            "America/Chicago", 
            "America/Denver",
            "America/Los_Angeles",
            "America/Phoenix",
            "America/Anchorage",
            "Pacific/Honolulu",
            "UTC",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Australia/Sydney"
        ]
        self.timezone_combo.addItems(common_timezones)
        self.timezone_combo.setCurrentText("America/New_York")  # Default selection
        tz_layout.addWidget(tz_label)
        tz_layout.addWidget(self.timezone_combo)
        timezone_group_layout.addLayout(tz_layout)
        
        layout.addWidget(timezone_group)
        
        # Help text
        help_text = QLabel("""
<b>Tips:</b>
• Latitude: Positive values for Northern Hemisphere, negative for Southern
• Longitude: Positive values for Eastern Hemisphere, negative for Western  
• You can find coordinates using online tools like Google Maps
        """)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { color: #888888; font-size: 9pt; margin: 10px 0; }")
        layout.addWidget(help_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Location")
        self.save_button.clicked.connect(self._save_settings)
        self.save_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def _save_settings(self):
        """Save the location settings to database"""
        try:
            lat_text = self.latitude_input.text().strip()
            lon_text = self.longitude_input.text().strip()
            
            if not lat_text or not lon_text:
                QMessageBox.warning(self, "Invalid Input", "Please enter both latitude and longitude.")
                return
                
            try:
                lat = float(lat_text)
                lon = float(lon_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for latitude and longitude.")
                return
                
            if not (-90 <= lat <= 90):
                QMessageBox.warning(self, "Invalid Input", "Latitude must be between -90 and 90 degrees.")
                return
                
            if not (-180 <= lon <= 180):
                QMessageBox.warning(self, "Invalid Input", "Longitude must be between -180 and 180 degrees.")
                return
            
            location_name = self.location_name_input.text().strip() or None
            timezone = self.timezone_combo.currentText().strip() or None
            
            # Validate timezone
            if timezone:
                try:
                    pytz.timezone(timezone)
                except pytz.UnknownTimeZoneError:
                    QMessageBox.warning(self, "Invalid Timezone", f"Unknown timezone: {timezone}")
                    return
            
            # Save to database
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if settings exist
                cursor.execute("SELECT id FROM usersettings LIMIT 1")
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing settings
                    cursor.execute("""
                        UPDATE usersettings 
                        SET location_lat = ?, location_lon = ?, location_name = ?, timezone = ?
                        WHERE id = (SELECT id FROM usersettings ORDER BY id DESC LIMIT 1)
                    """, (lat, lon, location_name, timezone))
                else:
                    # Insert new settings
                    cursor.execute("""
                        INSERT INTO usersettings (location_lat, location_lon, location_name, timezone) 
                        VALUES (?, ?, ?, ?)
                    """, (lat, lon, location_name, timezone))
                
                conn.commit()
                
            QMessageBox.information(self, "Success", "Location settings saved successfully!")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")


class DSOVisibilityApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSO Visibility Calculator - Cosmos Collection")
        self.setGeometry(100, 100, 1200, 800)

        # Set dark theme for the application
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QLineEdit, QSpinBox, QDateEdit {
                background-color: #404040;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
                color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDateEdit:focus {
                border: 2px solid #0078d4;
            }
            QTextEdit {
                background-color: #404040;
                border: 1px solid #666666;
                color: #ffffff;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QCalendarWidget {
                background-color: #404040;
                color: #ffffff;
            }
            QCalendarWidget QToolButton {
                background-color: #555555;
                color: #ffffff;
            }
            QCalendarWidget QMenu {
                background-color: #404040;
                color: #ffffff;
            }
            QCalendarWidget QSpinBox {
                background-color: #555555;
                color: #ffffff;
            }
        """)

        self.calc_thread = None
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
        
        # Add Set Location button
        self.set_location_btn = QPushButton("Set Location")
        self.set_location_btn.clicked.connect(self._show_location_dialog)
        left_layout.addWidget(self.set_location_btn)
        
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
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        left_layout.addWidget(results_group)

        left_layout.addStretch()

        # Right panel for plot
        self.plot_widget = VisibilityPlot()

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.plot_widget, stretch=1)

        # No initial calculation - wait for location to be set

    def _load_location_from_database(self):
        """Load observer location from the database"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Try to get location_name and timezone, but fall back if columns don't exist
                try:
                    cursor.execute("SELECT location_lat, location_lon, location_name, timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                    if row:
                        lat, lon, location_name, timezone = row
                        self.user_timezone = timezone  # Store for later use
                    else:
                        lat, lon, location_name, timezone = None, None, None, None
                        self.user_timezone = None
                except Exception:
                    # Fallback to old query if new columns don't exist
                    try:
                        cursor.execute("SELECT location_lat, location_lon, location_name FROM usersettings ORDER BY id DESC LIMIT 1")
                        row = cursor.fetchone()
                        if row:
                            lat, lon, location_name = row
                            self.user_timezone = None
                        else:
                            lat, lon, location_name = None, None, None
                            self.user_timezone = None
                    except Exception:
                        # Fallback to basic query
                        cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                        row = cursor.fetchone()
                        if row:
                            lat, lon, location_name = row[0], row[1], None
                            self.user_timezone = None
                        else:
                            lat, lon, location_name = None, None, None
                            self.user_timezone = None
                
                if lat is not None and lon is not None:
                    # Format coordinates nicely
                    lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
                    lon_str = f"{abs(lon):.2f}°{'W' if lon < 0 else 'E'}"
                    
                    # Use the location name if available, otherwise fall back to "User Location"
                    display_name = location_name if location_name else "User Location"
                    self.location_name_label.setText(display_name)
                    self.location_coords_label.setText(f"Lat: {lat_str}, Lon: {lon_str}")
                    
                    # Update window title with the location name and timezone
                    if self.user_timezone:
                        try:
                            tz_obj = pytz.timezone(self.user_timezone)
                            # Get current timezone abbreviation
                            from datetime import datetime
                            now = datetime.now(tz_obj)
                            tz_abbrev = now.strftime('%Z')
                            self.setWindowTitle(f"DSO Visibility Calculator - {display_name} ({tz_abbrev}) - Cosmos Collection")
                        except Exception:
                            self.setWindowTitle(f"DSO Visibility Calculator - {display_name} - Cosmos Collection")
                    else:
                        self.setWindowTitle(f"DSO Visibility Calculator - {display_name} - Cosmos Collection")
                else:
                    # No location configured - prompt user to set location
                    self.location_name_label.setText("Location not set")
                    self.location_coords_label.setText("Click 'Set Location' to configure")
                    self.setWindowTitle("DSO Visibility Calculator - Location Required")
                    self.user_timezone = None
                    self._show_location_required()
        except Exception as e:
            # Error accessing database - prompt user to set location
            self.location_name_label.setText("Location not set")
            self.location_coords_label.setText("Click 'Set Location' to configure")
            self.setWindowTitle("DSO Visibility Calculator - Location Required")
            self.user_timezone = None
            self._show_location_required()

    def on_calculation_finished(self, results):
        """Handle completed calculation"""
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Visibility")

        # Update plot
        self.plot_widget.plot_visibility(results)

        # Generate text results
        self.update_results_text(results)

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

        text += f"Maximum altitude: {max_altitude:.1f}° at {max_alt_time_local.strftime('%H:%M')} {tz_name}\n"
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

                text += f"From: {start_time_local.strftime('%H:%M')} {tz_name}\n"
                text += f"To:   {end_time_local.strftime('%H:%M')} {tz_name}\n"
                text += f"Duration: {duration:.1f} hours\n"
                text += f"Mid-window direction: {mid_direction} ({mid_az:.0f}°)\n"
                text += f"Azimuth range: {start_az:.0f}° → {end_az:.0f}°\n"
                text += "-" * 20 + "\n"

        self.results_text.setText(text)

    def _show_location_dialog(self):
        """Show the location settings dialog"""
        dialog = LocationSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # Reload location data after successful save
            self._load_location_from_database()
            
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
                              "Please set your observer location before calculating visibility.")
            self._show_location_dialog()
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

        # Start calculation thread
        self.calc_thread = CalculationThread(dso_name, date, hours, min_altitude)
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.start()