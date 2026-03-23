#!/usr/bin/env python3
"""
Weather Forecast Window for Cosmos Collection
Displays astrophotography-relevant weather data from Open-Meteo API
"""

import sys
import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import matplotlib
matplotlib.use('QtAgg')

# Suppress matplotlib font_manager debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Set dark theme for matplotlib
plt.style.use('dark_background')

import requests
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QTimer, QUrl
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QGroupBox, QMessageBox, QProgressBar, QScrollArea,
    QFrame, QGridLayout, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QApplication, QSplitter, QCheckBox, QComboBox
)
from PySide6.QtGui import QColor

from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from TimeFormatHelper import format_time, format_datetime, get_time_format_24h
from UrlOpener import open_url

# Set up logging
logger = logging.getLogger(__name__)

# Cache for weather data (persists across window instances)
CACHE_MAX_AGE_MINUTES = 15


class WeatherCache:
    """Simple cache for weather data to reduce API calls"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WeatherCache, cls).__new__(cls)
            cls._instance._data = None
            cls._instance._timestamp = None
            cls._instance._location = None
            cls._instance._update_callbacks = []
        return cls._instance

    def get(self, lat: float, lon: float) -> Optional[List]:
        """Get cached data if valid and for the same location"""
        if self._data is None or self._timestamp is None:
            return None

        # Check if location matches (within small tolerance for float comparison)
        if self._location is None:
            return None
        cached_lat, cached_lon = self._location
        if abs(cached_lat - lat) > 0.01 or abs(cached_lon - lon) > 0.01:
            logger.debug("Weather cache miss: location changed")
            return None

        # Check if cache is still valid
        age = datetime.now() - self._timestamp
        if age > timedelta(minutes=CACHE_MAX_AGE_MINUTES):
            logger.debug(f"Weather cache miss: data is {age.seconds // 60} minutes old")
            return None

        logger.debug(f"Weather cache hit: data is {age.seconds // 60} minutes old")
        return self._data

    def set(self, lat: float, lon: float, data: List):
        """Store data in cache and notify callbacks"""
        self._data = data
        self._timestamp = datetime.now()
        self._location = (lat, lon)
        logger.debug("Weather data cached")

        # Notify all registered callbacks
        for callback in self._update_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.debug(f"Weather cache callback error: {e}")

    def get_age_str(self) -> Optional[str]:
        """Get a human-readable string of cache age"""
        if self._timestamp is None:
            return None
        age = datetime.now() - self._timestamp
        minutes = age.seconds // 60
        if minutes < 1:
            return "just now"
        elif minutes == 1:
            return "1 minute ago"
        else:
            return f"{minutes} minutes ago"

    def clear(self):
        """Clear the cache"""
        self._data = None
        self._timestamp = None
        self._location = None

    def add_update_callback(self, callback):
        """Register a callback to be called when weather data is updated.

        Args:
            callback: A callable that accepts a list of DailyWeatherSummary objects
        """
        if callback not in self._update_callbacks:
            self._update_callbacks.append(callback)

    def remove_update_callback(self, callback):
        """Remove a previously registered callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)

    def get_cached_data(self) -> Optional[List]:
        """Get the cached data without location/age checks (for tray updates)."""
        return self._data


@dataclass
class MoonPhaseData:
    """Moon phase information for a date"""
    phase_angle: float      # 0-180 degrees (elongation from sun)
    illumination: float     # 0-100 percentage
    phase_name: str         # "New Moon", "Waxing Crescent", etc.
    phase_emoji: str        # Moon phase emoji


@dataclass
class HourlyWeatherData:
    """Dataclass for hourly weather data"""
    time: datetime
    cloud_cover: float  # Total cloud cover %
    cloud_cover_low: float
    cloud_cover_mid: float
    cloud_cover_high: float
    temperature: float  # Celsius
    dew_point: float  # Celsius
    humidity: float  # %
    wind_speed: float  # km/h
    precipitation_probability: float  # %
    wind_gusts: float = 0.0  # km/h — peak gust in the hour
    visibility: Optional[float] = None  # meters
    surface_pressure: Optional[float] = None  # hPa


@dataclass
class DailyWeatherSummary:
    """Dataclass for daily aggregated weather data"""
    date: datetime
    hourly_data: List[HourlyWeatherData]
    avg_cloud_cover: float
    min_cloud_cover: float
    max_cloud_cover: float
    avg_temperature: float
    min_temperature: float
    max_temperature: float
    avg_humidity: float
    avg_wind_speed: float
    max_wind_speed: float
    avg_precipitation_prob: float
    tonight_avg_cloud_cover: float  # avg cloud cover for dark hours only (sun_alt < -12°)
    astro_score: int  # 0-100
    seeing_estimate: str  # Excellent/Good/Moderate/Poor
    moon_phase: Optional[MoonPhaseData] = None
    dark_hours_start: Optional[datetime] = None  # First dark hour (sun_alt < -12°)
    dark_hours_end: Optional[datetime] = None  # Last dark hour (sun_alt < -12°)


class WeatherWorker(QThread):
    """QThread worker for fetching weather data from Open-Meteo API"""
    weather_loaded = Signal(list)  # List of DailyWeatherSummary
    error_occurred = Signal(str)
    progress = Signal(str)

    def __init__(self, lat: float, lon: float, timezone: str = None):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.timezone = timezone

    def run(self):
        """Fetch weather data from Open-Meteo API"""
        try:
            self.progress.emit("Connecting to Open-Meteo API...")

            # Build API URL
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={self.lat}&longitude={self.lon}&"
                f"hourly=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
                f"temperature_2m,dew_point_2m,relative_humidity_2m,"
                f"wind_speed_10m,wind_gusts_10m,precipitation_probability,visibility,surface_pressure&"
                f"forecast_days=7&timezone=auto"
            )

            # Handle SSL for PyInstaller frozen builds
            verify = not getattr(sys, 'frozen', False)

            self.progress.emit("Downloading weather forecast data...")
            response = requests.get(url, timeout=30, verify=verify)
            response.raise_for_status()

            data = response.json()

            self.progress.emit("Processing weather data...")
            daily_summaries = self._process_weather_data(data)

            self.weather_loaded.emit(daily_summaries)

        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Error: {str(e)}")

    def _process_weather_data(self, data: Dict[str, Any]) -> List[DailyWeatherSummary]:
        """Process raw API data into daily summaries"""
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            return []

        # Parse hourly data
        hourly_records: List[HourlyWeatherData] = []
        for i, time_str in enumerate(times):
            try:
                dt = datetime.fromisoformat(time_str)
                hourly_records.append(HourlyWeatherData(
                    time=dt,
                    cloud_cover=hourly.get("cloud_cover", [0] * len(times))[i] or 0,
                    cloud_cover_low=hourly.get("cloud_cover_low", [0] * len(times))[i] or 0,
                    cloud_cover_mid=hourly.get("cloud_cover_mid", [0] * len(times))[i] or 0,
                    cloud_cover_high=hourly.get("cloud_cover_high", [0] * len(times))[i] or 0,
                    temperature=hourly.get("temperature_2m", [0] * len(times))[i] or 0,
                    dew_point=hourly.get("dew_point_2m", [0] * len(times))[i] or 0,
                    humidity=hourly.get("relative_humidity_2m", [0] * len(times))[i] or 0,
                    wind_speed=hourly.get("wind_speed_10m", [0] * len(times))[i] or 0,
                    precipitation_probability=hourly.get("precipitation_probability", [0] * len(times))[i] or 0,
                    wind_gusts=hourly.get("wind_gusts_10m", [0] * len(times))[i] or 0,
                    visibility=hourly.get("visibility", [None] * len(times))[i],
                    surface_pressure=hourly.get("surface_pressure", [None] * len(times))[i]
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing hourly data at index {i}: {e}")
                continue

        # Group by date
        daily_data: Dict[datetime.date, List[HourlyWeatherData]] = {}
        for record in hourly_records:
            date = record.time.date()
            if date not in daily_data:
                daily_data[date] = []
            daily_data[date].append(record)

        # Calculate sun altitudes for all hourly records (vectorized for efficiency)
        all_times = [record.time for record in hourly_records]
        sun_altitudes = calculate_sun_altitudes(self.lat, self.lon, all_times, self.timezone)
        sun_alt_map = dict(zip(all_times, sun_altitudes))

        # Create daily summaries
        sorted_dates = sorted(daily_data.keys())
        daily_summaries: List[DailyWeatherSummary] = []
        for date_idx, date in enumerate(sorted_dates):
            hours = daily_data[date]
            if not hours:
                continue

            # Calculate aggregates for all hours (used for display)
            cloud_covers = [h.cloud_cover for h in hours]
            temps = [h.temperature for h in hours]
            humidities = [h.humidity for h in hours]
            winds = [h.wind_speed for h in hours]
            precip_probs = [h.precipitation_probability for h in hours]

            avg_cloud = sum(cloud_covers) / len(cloud_covers)
            avg_humidity = sum(humidities) / len(humidities)
            avg_wind = sum(winds) / len(winds)
            avg_precip = sum(precip_probs) / len(precip_probs)

            # Build "tonight" hours: evening dark hours of this day + morning dark hours of next day
            evening_dark = [h for h in hours if h.time.hour >= 12 and sun_alt_map.get(h.time, 0) < -12]
            morning_dark = []
            if date_idx + 1 < len(sorted_dates):
                next_date = sorted_dates[date_idx + 1]
                next_hours = daily_data[next_date]
                morning_dark = [h for h in next_hours if h.time.hour < 12 and sun_alt_map.get(h.time, 0) < -12]
            tonight_hours = evening_dark + morning_dark

            # Determine dark hours start and end times
            dark_start = tonight_hours[0].time if tonight_hours else None
            dark_end = tonight_hours[-1].time if tonight_hours else None

            # Calculate astro score as the average of individual hourly scores
            if tonight_hours:
                hourly_scores = [
                    calculate_astro_score(h.cloud_cover, h.humidity, h.wind_speed,
                                          h.precipitation_probability, h.visibility,
                                          h.wind_gusts)
                    for h in tonight_hours
                ]
                astro_score = int(sum(hourly_scores) / len(hourly_scores))
            else:
                # No dark hours (e.g., polar summer) - score is 0
                astro_score = 0

            # Calculate tonight's average cloud cover (dark hours only)
            if tonight_hours:
                tonight_avg_cloud = sum(h.cloud_cover for h in tonight_hours) / len(tonight_hours)
            else:
                tonight_avg_cloud = avg_cloud  # Fallback to full day average

            # Estimate seeing based on tonight's hours
            if tonight_hours:
                night_humidity = sum(h.humidity for h in tonight_hours) / len(tonight_hours)
                night_wind = sum(h.wind_speed for h in tonight_hours) / len(tonight_hours)
                night_gusts = max(h.wind_gusts for h in tonight_hours)
                night_temp_dew_spread = sum(h.temperature - h.dew_point for h in tonight_hours) / len(tonight_hours)
                # Average surface pressure for tonight (filter out None values)
                pressure_values = [h.surface_pressure for h in tonight_hours if h.surface_pressure is not None]
                night_pressure = sum(pressure_values) / len(pressure_values) if pressure_values else None
            else:
                night_humidity = avg_humidity
                night_wind = avg_wind
                night_gusts = max(h.wind_gusts for h in hours)
                night_temp_dew_spread = sum(h.temperature - h.dew_point for h in hours) / len(hours)
                pressure_values = [h.surface_pressure for h in hours if h.surface_pressure is not None]
                night_pressure = sum(pressure_values) / len(pressure_values) if pressure_values else None

            seeing = estimate_seeing(night_humidity, night_wind, night_temp_dew_spread, night_pressure, night_gusts)

            # Calculate moon phase for this date
            moon_phase = calculate_moon_phase(datetime.combine(date, datetime.min.time()), self.timezone)

            summary = DailyWeatherSummary(
                date=datetime.combine(date, datetime.min.time()),
                hourly_data=hours,
                avg_cloud_cover=avg_cloud,
                min_cloud_cover=min(cloud_covers),
                max_cloud_cover=max(cloud_covers),
                avg_temperature=sum(temps) / len(temps),
                min_temperature=min(temps),
                max_temperature=max(temps),
                avg_humidity=avg_humidity,
                avg_wind_speed=avg_wind,
                max_wind_speed=max(winds),
                avg_precipitation_prob=avg_precip,
                tonight_avg_cloud_cover=tonight_avg_cloud,
                astro_score=astro_score,
                seeing_estimate=seeing,
                moon_phase=moon_phase,
                dark_hours_start=dark_start,
                dark_hours_end=dark_end
            )
            daily_summaries.append(summary)

        return daily_summaries


def calculate_astro_score(cloud_cover: float, humidity: float, wind_speed: float,
                          precip_prob: float, visibility: Optional[float] = None,
                          wind_gusts: float = 0.0) -> int:
    """
    Calculate an astrophotography suitability score (0-100).

    Cloud cover is the dominant factor - high cloud cover caps the maximum possible score
    since you cannot do astrophotography through clouds regardless of other conditions.

    Score caps based on cloud cover:
    - >80% clouds: max score 30 (Poor)
    - >60% clouds: max score 50 (Moderate)
    - >40% clouds: max score 70 (Good)

    Score caps based on wind gusts (applied after cloud caps):
    - >40 km/h gusts: max score 35 (severe — tracking essentially impossible)
    - >25 km/h gusts: max score 60 (moderate — long exposures degraded)

    Base scoring weights:
    - Cloud cover (40%): lower is better
    - Transparency/Visibility (15%): higher is better (affects deep sky objects)
    - Humidity (15%): ideal 30-50%, affects transparency and dew risk
    - Wind speed (15%): under 15 km/h is good for tracking/guiding
    - Precipitation (15%): 0% is ideal

    Args:
        cloud_cover: Cloud cover percentage (0-100)
        humidity: Relative humidity percentage (0-100)
        wind_speed: Wind speed in km/h
        precip_prob: Precipitation probability percentage (0-100)
        visibility: Visibility in meters (None if unavailable)
        wind_gusts: Peak wind gust speed in km/h (gusts weighted 75% — intermittent)
    """
    # Cloud cover score (0-100, lower clouds = higher score)
    cloud_score = max(0, 100 - cloud_cover)

    # Visibility/Transparency score (0-100, higher visibility = better transparency)
    # Open-Meteo returns visibility in meters
    # Excellent: > 40km, Good: 20-40km, Moderate: 10-20km, Poor: < 10km
    if visibility is not None:
        visibility_km = visibility / 1000.0
        if visibility_km >= 40:
            visibility_score = 100
        elif visibility_km >= 20:
            # Linear interpolation from 70 to 100 between 20-40km
            visibility_score = 70 + (visibility_km - 20) * 1.5
        elif visibility_km >= 10:
            # Linear interpolation from 40 to 70 between 10-20km
            visibility_score = 40 + (visibility_km - 10) * 3
        else:
            # Below 10km, score drops more steeply
            visibility_score = max(0, visibility_km * 4)
    else:
        # If visibility data unavailable, use a neutral score
        visibility_score = 70

    # Humidity score (0-100, ideal around 40%)
    # High humidity reduces transparency and increases dew risk
    if humidity < 30:
        humidity_score = 70 + humidity  # Slightly penalize very dry
    elif humidity <= 50:
        humidity_score = 100  # Ideal range
    elif humidity <= 70:
        humidity_score = 100 - (humidity - 50) * 2  # 50-100 as humidity goes from 50-70
    else:
        humidity_score = max(0, 60 - (humidity - 70))  # Penalize high humidity

    # Wind score (0-100, under 15 km/h is good)
    # Gusts are intermittent so weighted at 75%; effective_wind >= sustained speed
    effective_wind = max(wind_speed, wind_gusts * 0.75)
    if effective_wind <= 10:
        wind_score = 100
    elif effective_wind <= 15:
        wind_score = 100 - (effective_wind - 10) * 4  # 80-100 range
    elif effective_wind <= 25:
        wind_score = 80 - (effective_wind - 15) * 4  # 40-80 range
    else:
        wind_score = max(0, 40 - (effective_wind - 25) * 2)

    # Precipitation score (0-100, 0% is ideal)
    precip_score = max(0, 100 - precip_prob * 2)

    # Weighted average
    total_score = (
        cloud_score * 0.40 +
        visibility_score * 0.15 +
        humidity_score * 0.15 +
        wind_score * 0.15 +
        precip_score * 0.15
    )

    # Apply cloud cover caps - high clouds should hard-limit the score
    # since astrophotography is impossible through heavy cloud cover
    if cloud_cover > 80:
        total_score = min(total_score, 30)  # Cap at Poor
    elif cloud_cover > 60:
        total_score = min(total_score, 50)  # Cap at low Moderate
    elif cloud_cover > 40:
        total_score = min(total_score, 70)  # Cap at Good

    # Apply wind gust caps — severe gusts ruin tracking regardless of sky clarity
    if wind_gusts > 40:
        total_score = min(total_score, 35)  # Severe gusts — tracking essentially impossible
    elif wind_gusts > 25:
        total_score = min(total_score, 60)  # Moderate gusts — long exposures degraded

    return int(min(100, max(0, total_score)))


def estimate_seeing(humidity: float, wind_speed: float, temp_dew_spread: float,
                    surface_pressure: Optional[float] = None,
                    wind_gusts: float = 0.0) -> str:
    """
    Estimate seeing quality based on atmospheric conditions.

    Seeing (atmospheric turbulence) is affected by:
    - Humidity: High humidity can indicate atmospheric instability
    - Wind: High winds cause turbulence, but moderate wind can indicate stable laminar flow
    - Temperature-dew spread: Indicates moisture content and potential for ground-level turbulence
    - Surface pressure: High, stable pressure generally means better seeing;
      low pressure systems bring unstable air masses

    Args:
        humidity: Relative humidity %
        wind_speed: Wind speed in km/h
        temp_dew_spread: Temperature minus dew point (larger = less moisture)
        surface_pressure: Surface pressure in hPa (None if unavailable)

    Returns:
        Seeing quality string: Excellent/Good/Moderate/Poor
    """
    score = 100

    # Humidity factor (lower is better for seeing)
    if humidity > 80:
        score -= 30
    elif humidity > 65:
        score -= 15
    elif humidity > 50:
        score -= 5

    # Wind factor (calm to light is best, strong winds cause turbulence)
    # However, very calm conditions can lead to ground-layer turbulence
    # Use the worse of sustained speed or gusts for seeing penalty
    gust_wind = max(wind_speed, wind_gusts)
    if gust_wind > 30:
        score -= 40  # Very high winds - severe turbulence
    elif gust_wind > 25:
        score -= 30
    elif gust_wind > 15:
        score -= 15
    elif gust_wind > 10:
        score -= 5
    elif wind_speed < 3:
        score -= 5  # Very calm - potential ground layer issues

    # Dew risk factor (larger spread = better, also indicates drier air column)
    if temp_dew_spread < 2:
        score -= 25  # High dew risk and moist air
    elif temp_dew_spread < 5:
        score -= 10
    elif temp_dew_spread > 15:
        score += 10  # Very dry air column - excellent
    elif temp_dew_spread > 10:
        score += 5  # Very safe from dew

    # Surface pressure factor
    # High pressure (>1020 hPa) typically indicates stable air mass = better seeing
    # Low pressure (<1000 hPa) indicates unstable weather systems = worse seeing
    # Normal range is roughly 980-1040 hPa
    if surface_pressure is not None:
        if surface_pressure >= 1025:
            score += 15  # Strong high pressure - excellent stability
        elif surface_pressure >= 1015:
            score += 10  # High pressure - good stability
        elif surface_pressure >= 1005:
            score += 0   # Normal pressure - neutral
        elif surface_pressure >= 995:
            score -= 10  # Lowish pressure - some instability
        else:
            score -= 20  # Low pressure system - unstable air

    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Moderate"
    else:
        return "Poor"


def get_rating_color(score: int) -> str:
    """Get color based on astro score"""
    if score >= 80:
        return COLORS['success']  # Green
    elif score >= 60:
        return COLORS['info']  # Blue
    elif score >= 40:
        return COLORS['warning']  # Yellow
    else:
        return COLORS['error']  # Red


def get_rating_label(score: int) -> str:
    """Get label based on astro score"""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Moderate"
    else:
        return "Poor"


def calculate_sun_altitudes(lat: float, lon: float, times: List[datetime], timezone: str = None) -> List[float]:
    """
    Calculate sun altitude for each time point.

    Args:
        lat: Observer latitude in degrees
        lon: Observer longitude in degrees
        times: List of datetime objects for each hour (in local time)
        timezone: Optional timezone string (e.g., 'America/New_York') for converting local times to UTC

    Returns:
        List of sun altitudes in degrees for each time point
    """
    if not times:
        return []

    # Convert naive local times to UTC if timezone is provided
    if timezone:
        try:
            import pytz
            local_tz = pytz.timezone(timezone)
            utc_times = []
            for t in times:
                if t.tzinfo is None:
                    local_dt = local_tz.localize(t)
                    utc_dt = local_dt.astimezone(pytz.UTC)
                    utc_times.append(utc_dt.replace(tzinfo=None))  # astropy works with naive UTC
                else:
                    utc_times.append(t)
            times = utc_times
        except Exception as e:
            logger.warning(f"Could not convert times to UTC: {e}")

    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    astropy_times = Time([t.isoformat() for t in times])
    altaz_frame = AltAz(obstime=astropy_times, location=location)
    sun_altaz = get_sun(astropy_times).transform_to(altaz_frame)

    return sun_altaz.alt.deg.tolist()


def _get_moon_phase_name(phase_angle: float, is_waxing: bool) -> tuple:
    """
    Map phase angle (elongation) to moon phase name and emoji.

    Args:
        phase_angle: Elongation from sun in degrees (0-180)
        is_waxing: True if moon is waxing (getting brighter), False if waning

    Returns:
        Tuple of (phase_name, phase_emoji)
    """
    if phase_angle < 22.5:
        return ("New Moon", "🌑")
    elif phase_angle < 67.5:
        if is_waxing:
            return ("Waxing Crescent", "🌒")
        else:
            return ("Waning Crescent", "🌘")
    elif phase_angle < 112.5:
        if is_waxing:
            return ("First Quarter", "🌓")
        else:
            return ("Last Quarter", "🌗")
    elif phase_angle < 157.5:
        if is_waxing:
            return ("Waxing Gibbous", "🌔")
        else:
            return ("Waning Gibbous", "🌖")
    else:
        return ("Full Moon", "🌕")


def calculate_moon_phase(date: datetime, timezone: str = None) -> MoonPhaseData:
    """
    Calculate moon phase information for a given date.

    Args:
        date: The date to calculate moon phase for (naive datetime in local time)
        timezone: Timezone string (e.g., 'America/New_York') for converting to UTC

    Returns:
        MoonPhaseData with phase angle, illumination, name, and emoji
    """
    import numpy as np
    from astropy.coordinates import GeocentricTrueEcliptic

    # Convert local midnight to UTC if timezone is provided
    if timezone:
        try:
            import pytz
            local_tz = pytz.timezone(timezone)
            if date.tzinfo is None:
                local_dt = local_tz.localize(date)
                utc_dt = local_dt.astimezone(pytz.UTC)
                date = utc_dt.replace(tzinfo=None)  # astropy works with naive UTC
        except Exception as e:
            logger.warning(f"Could not convert moon phase time to UTC: {e}")

    # Use midnight of the date for calculation
    obs_time = Time(date.isoformat())

    # Get sun and moon positions
    sun = get_sun(obs_time)
    moon = get_body('moon', obs_time)

    # Calculate elongation (angular separation between sun and moon)
    elongation = sun.separation(moon)
    phase_angle = elongation.deg

    # Calculate illumination fraction
    # Illumination = (1 - cos(elongation)) / 2
    # This gives 0% at new moon (0°) and 100% at full moon (180°)
    illumination = (1 - np.cos(elongation.rad)) / 2 * 100

    # Determine if waxing or waning by comparing ecliptic longitudes
    # Moon is waxing when its ecliptic longitude is ahead of (greater than) the sun's
    sun_ecliptic = sun.transform_to(GeocentricTrueEcliptic(equinox=obs_time))
    moon_ecliptic = moon.transform_to(GeocentricTrueEcliptic(equinox=obs_time))

    sun_lon = sun_ecliptic.lon.deg
    moon_lon = moon_ecliptic.lon.deg

    # Calculate the difference (moon - sun), normalized to 0-360
    lon_diff = (moon_lon - sun_lon) % 360

    # If difference is 0-180, moon is ahead of sun = waxing
    # If difference is 180-360, moon is behind sun = waning
    is_waxing = lon_diff < 180

    # Get phase name and emoji
    phase_name, phase_emoji = _get_moon_phase_name(phase_angle, is_waxing)

    return MoonPhaseData(
        phase_angle=phase_angle,
        illumination=illumination,
        phase_name=phase_name,
        phase_emoji=phase_emoji
    )


class DayWeatherCard(QFrame):
    """Clickable card widget for displaying daily weather summary"""
    clicked = Signal(object)  # Emits DailyWeatherSummary

    def __init__(self, summary: DailyWeatherSummary, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.setFrameShape(QFrame.Box)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(100)
        self.setMaximumWidth(130)

        # Set background and border
        rating_color = get_rating_color(summary.astro_score)
        self.setStyleSheet(f"""
            DayWeatherCard {{
                background-color: {COLORS['background_light']};
                border: 2px solid {rating_color};
                border-radius: 8px;
                padding: 5px;
            }}
            DayWeatherCard:hover {{
                background-color: {COLORS['background_hover']};
                border: 2px solid {rating_color};
            }}
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        # Day name
        day_name = self.summary.date.strftime("%a")
        day_label = QLabel(day_name)
        day_label.setAlignment(Qt.AlignCenter)
        day_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(day_label)

        # Date
        date_str = self.summary.date.strftime("%m/%d")
        date_label = QLabel(date_str)
        date_label.setAlignment(Qt.AlignCenter)
        date_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10pt;")
        layout.addWidget(date_label)

        layout.addSpacing(5)

        # Rating
        rating_color = get_rating_color(self.summary.astro_score)
        rating_label = get_rating_label(self.summary.astro_score)
        rating_text = QLabel(rating_label)
        rating_text.setAlignment(Qt.AlignCenter)
        rating_text.setStyleSheet(f"color: {rating_color}; font-weight: bold; font-size: 12pt;")
        layout.addWidget(rating_text)

        # Score
        score_label = QLabel(f"({self.summary.astro_score})")
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10pt;")
        score_label.setToolTip("Astro Score")
        layout.addWidget(score_label)

        layout.addSpacing(5)

        cloud_text = QLabel("Clouds")
        cloud_text.setAlignment(Qt.AlignCenter)
        cloud_text.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12pt;")
        cloud_text.setToolTip("Average cloud cover for dark hours (sun altitude < -12°)")
        layout.addWidget(cloud_text)

        # Cloud cover (tonight's average - dark hours only)
        cloud_label = QLabel(f"{self.summary.tonight_avg_cloud_cover:.0f}%")
        cloud_label.setAlignment(Qt.AlignCenter)
        cloud_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        layout.addWidget(cloud_label)

        layout.addSpacing(5)

        # Moon phase (compact display)
        if self.summary.moon_phase:
            moon_label = QLabel(f"{self.summary.moon_phase.phase_emoji} {self.summary.moon_phase.illumination:.0f}%")
            moon_label.setAlignment(Qt.AlignCenter)
            moon_label.setToolTip(f"{self.summary.moon_phase.phase_name}")
            layout.addWidget(moon_label)

        layout.addStretch()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to show details"""
        self.clicked.emit(self.summary)
        super().mouseDoubleClickEvent(event)


class HourlyAstroChart(FigureCanvas):
    """Matplotlib chart showing hourly astro scores for a day"""
    hour_hovered = Signal(int)  # Emits the row index when hovering over a bar

    def __init__(self, hourly_data: List[HourlyWeatherData], sun_altitudes: List[float] = None, parent=None):
        self.figure = Figure(figsize=(10, 3), facecolor='#2b2b2b')
        super().__init__(self.figure)
        self.setParent(parent)

        self.hourly_data = hourly_data
        self.sun_altitudes = sun_altitudes
        self.bars = None
        self.ax = None
        self._last_hovered_index = -1
        self._create_chart()

        # Connect mouse motion event
        self.mpl_connect('motion_notify_event', self._on_mouse_move)

    def refresh(self, hourly_data: List[HourlyWeatherData], sun_altitudes: List[float] = None):
        """Refresh the chart with new hourly data"""
        self.hourly_data = hourly_data
        self.sun_altitudes = sun_altitudes
        self._last_hovered_index = -1
        self._create_chart()

    def _create_chart(self):
        """Create the hourly astro score chart"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        # Detect if this is a midnight-centered view (hours not sequential 0-23)
        # by checking if hours wrap around midnight
        is_midnight_view = False
        if len(self.hourly_data) > 1:
            first_hour = self.hourly_data[0].time.hour
            last_hour = self.hourly_data[-1].time.hour
            # Midnight view: starts at 18+ and ends at 5 or less
            if first_hour >= 18 and last_hour <= 5:
                is_midnight_view = True

        # Calculate hourly astro scores
        x_positions = []
        hour_labels = []
        scores = []
        colors = []

        for idx, hour_data in enumerate(self.hourly_data):
            # Use sequential index for x position (works for both views)
            x_positions.append(idx)
            hour_labels.append(format_time(hour_data.time))
            score = calculate_astro_score(
                hour_data.cloud_cover,
                hour_data.humidity,
                hour_data.wind_speed,
                hour_data.precipitation_probability,
                hour_data.visibility,
                hour_data.wind_gusts
            )
            scores.append(score)
            colors.append(get_rating_color(score))

        num_hours = len(self.hourly_data)

        # Add darkness shading based on sun altitude (before bars so it's behind them)
        self._add_darkness_shading(num_hours)

        # Create bar chart using sequential indices
        self.bars = self.ax.bar(x_positions, scores, color=colors, edgecolor='none', width=0.8)

        # Add horizontal lines for rating thresholds
        self.ax.axhline(y=80, color=COLORS['success'], linestyle='--', alpha=0.5, linewidth=1)
        self.ax.axhline(y=60, color=COLORS['info'], linestyle='--', alpha=0.5, linewidth=1)
        self.ax.axhline(y=40, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)

        if is_midnight_view:
            self.ax.set_xlabel('Time (Evening → Morning)', color=COLORS['text'], fontsize=9)
            self.ax.set_title('Tonight\'s Astrophotography Conditions', color=COLORS['text'], fontsize=10, fontweight='bold')
        else:
            self.ax.set_xlabel('Hour of Day', color=COLORS['text'], fontsize=9)
            self.ax.set_title('Hourly Astrophotography Conditions', color=COLORS['text'], fontsize=10, fontweight='bold')

        # Style the chart
        self.ax.set_xlim(-0.5, num_hours - 0.5)
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Astro Score', color=COLORS['text'], fontsize=9)

        # Set x-axis ticks - show every other label to avoid crowding
        tick_step = 2 if num_hours > 12 else 1
        tick_positions = list(range(0, num_hours, tick_step))
        tick_labels = [hour_labels[i] for i in tick_positions]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels, fontsize=8)

        # Style axes
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.spines['top'].set_color(COLORS['border'])
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['right'].set_color(COLORS['border'])

        # Add grid
        self.ax.yaxis.grid(True, linestyle=':', alpha=0.3, color=COLORS['border'])

        # Add legend for rating levels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['success'], label='Excellent (80+)'),
            Patch(facecolor=COLORS['info'], label='Good (60-79)'),
            Patch(facecolor=COLORS['warning'], label='Moderate (40-59)'),
            Patch(facecolor=COLORS['error'], label='Poor (<40)'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
                  facecolor='#353535', edgecolor=COLORS['border'], labelcolor=COLORS['text'])

        self.figure.tight_layout()
        self.draw()

    def _add_darkness_shading(self, num_hours: int):
        """Add background shading to show darkness levels based on sun altitude.

        Colors represent (matching DSO Visibility Calculator):
        - Daylight (sun > 0°): No shading (default background)
        - Civil twilight (0° to -6°): Light shade
        - Nautical twilight (-6° to -12°): Medium shade
        - Astronomical twilight (-12° to -18°): Dark shade
        - Night (< -18°): Darkest shade
        """
        if not self.sun_altitudes or len(self.sun_altitudes) != num_hours:
            return

        # Define sun altitude thresholds and colors
        # Format: (sun_max, sun_min, color, alpha)
        darkness_levels = [
            (90, 0, '#4a4a3a', 0.5),      # Daylight - warm tint
            (0, -6, '#2a3a4a', 0.6),      # Civil twilight - light blue-gray
            (-6, -12, '#1a2535', 0.7),    # Nautical twilight - medium blue
            (-12, -18, '#101520', 0.8),   # Astronomical twilight - dark blue
            (-18, -90, '#080a10', 0.9),   # Night - very dark blue/black
        ]

        # For each hour, shade based on its sun altitude
        for i, sun_alt in enumerate(self.sun_altitudes):
            for sun_max, sun_min, color, alpha in darkness_levels:
                if sun_alt <= sun_max and sun_alt > sun_min:
                    # Shade this bar's column
                    self.ax.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=alpha, zorder=0)
                    break  # Only apply one darkness level per hour

    def _on_mouse_move(self, event):
        """Handle mouse movement to detect bar hover"""
        if event.inaxes != self.ax or self.bars is None:
            if self._last_hovered_index != -1:
                self._last_hovered_index = -1
                self.hour_hovered.emit(-1)
            return

        # Find which bar the mouse is over
        for i, bar in enumerate(self.bars):
            if bar.contains(event)[0]:
                if self._last_hovered_index != i:
                    self._last_hovered_index = i
                    self.hour_hovered.emit(i)
                return

        # Mouse not over any bar
        if self._last_hovered_index != -1:
            self._last_hovered_index = -1
            self.hour_hovered.emit(-1)

    def highlight_bar(self, index: int):
        """Highlight the bar at the given index, unhighlight others"""
        if not hasattr(self, 'bars') or self.bars is None:
            return

        for i, bar in enumerate(self.bars):
            if i == index:
                bar.set_edgecolor('white')
                bar.set_linewidth(2)
            else:
                bar.set_edgecolor('none')
                bar.set_linewidth(0)

        self.draw_idle()

    def clear_bar_highlight(self):
        """Remove highlight from all bars"""
        if not hasattr(self, 'bars') or self.bars is None:
            return

        for bar in self.bars:
            bar.set_edgecolor('none')
            bar.set_linewidth(0)

        self.draw_idle()


class DayDetailDialog(WindowPositionMixin, QDialog):
    """Dialog showing detailed hourly weather data for a day"""

    WINDOW_POSITION_KEY = "DayDetailDialog"

    def __init__(self, summary: DailyWeatherSummary, next_day_summary: Optional[DailyWeatherSummary] = None,
                 lat: float = None, lon: float = None, timezone: str = None, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.next_day_summary = next_day_summary
        self.lat = lat
        self.lon = lon
        self.timezone = timezone
        self.setWindowTitle(f"Weather Details - {summary.date.strftime('%A, %B %d, %Y')} - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self.resize(1000, 750)
        self.setup_window_position()

        # Get unit preferences
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.temp_unit = settings.value("temperature_unit", "Celsius", type=str)
        self.wind_unit = settings.value("wind_speed_unit", "km/h", type=str)
        self.precip_unit = settings.value("precip_visibility_unit", "Metric (mm, km)", type=str)

        self._setup_ui()

    def _get_sun_altitudes(self, hours: List[HourlyWeatherData]) -> Optional[List[float]]:
        """Calculate sun altitudes for the given hours"""
        if self.lat is None or self.lon is None:
            return None

        times = [h.time for h in hours]
        return calculate_sun_altitudes(self.lat, self.lon, times, self.timezone)

    def _convert_temp(self, celsius: float) -> float:
        """Convert temperature based on user preference"""
        if self.temp_unit == "Fahrenheit":
            return celsius * 9 / 5 + 32
        return celsius

    def _temp_suffix(self) -> str:
        """Get temperature unit suffix"""
        return "F" if self.temp_unit == "Fahrenheit" else "C"

    def _convert_wind(self, kmh: float) -> float:
        """Convert wind speed from km/h based on user preference"""
        if self.wind_unit == "mph":
            return kmh * 0.621371
        elif self.wind_unit == "m/s":
            return kmh / 3.6
        return kmh

    def _wind_suffix(self) -> str:
        """Get wind speed unit suffix"""
        return self.wind_unit

    def _setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)

        # Summary header
        header_group = QGroupBox("Daily Summary")
        header_layout = QGridLayout(header_group)

        # Rating
        rating_color = get_rating_color(self.summary.astro_score)
        rating_label = QLabel(f"Astro Rating: {get_rating_label(self.summary.astro_score)} ({self.summary.astro_score})")
        rating_label.setStyleSheet(f"color: {rating_color}; font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(rating_label, 0, 0)

        # Seeing estimate
        seeing_label = QLabel(f"Seeing Estimate: {self.summary.seeing_estimate}")
        header_layout.addWidget(seeing_label, 0, 1)

        # Moon phase (detailed)
        if self.summary.moon_phase:
            mp = self.summary.moon_phase
            moon_label = QLabel(f"Moon: {mp.phase_emoji} {mp.phase_name} ({mp.illumination:.0f}%)")
            moon_label.setToolTip("Lower illumination is better for astrophotography")
            header_layout.addWidget(moon_label, 0, 2)

        # Cloud cover (dynamic - updates based on view mode)
        self.cloud_label = QLabel()
        header_layout.addWidget(self.cloud_label, 1, 0)

        # Temperature (dynamic - updates based on view mode)
        self.temp_label = QLabel()
        header_layout.addWidget(self.temp_label, 1, 1)

        # Wind (dynamic - updates based on view mode)
        self.wind_label = QLabel()
        header_layout.addWidget(self.wind_label, 2, 0)

        # Humidity (dynamic - updates based on view mode)
        self.humidity_label = QLabel()
        header_layout.addWidget(self.humidity_label, 2, 1)

        layout.addWidget(header_group)

        # View mode toggle - show actual dark hours if available
        if self.summary.dark_hours_start and self.summary.dark_hours_end:
            if get_time_format_24h():
                start_str = self.summary.dark_hours_start.strftime("%H:%M")
                end_str = self.summary.dark_hours_end.strftime("%H:%M")
            else:
                start_str = self.summary.dark_hours_start.strftime("%I:%M %p").lstrip("0")
                end_str = self.summary.dark_hours_end.strftime("%I:%M %p").lstrip("0")
            midnight_label = f"Center on Midnight (tonight's dark hours: {start_str}-{end_str})"
            midnight_tip = "Show hours centered around midnight when sun altitude is below -12° (astronomical darkness)"
        else:
            # Fallback if no dark hours computed
            if get_time_format_24h():
                midnight_label = "Center on Midnight (show tonight's hours: 18:00-05:00)"
                midnight_tip = "Show evening hours (18:00-23:00) of this day plus morning hours (00:00-05:00) of the next day"
            else:
                midnight_label = "Center on Midnight (show tonight's hours: 6:00 PM-5:00 AM)"
                midnight_tip = "Show evening hours (6:00 PM-11:00 PM) of this day plus morning hours (12:00 AM-5:00 AM) of the next day"
        self.midnight_view_checkbox = QCheckBox(midnight_label)
        self.midnight_view_checkbox.setEnabled(self.next_day_summary is not None)
        self.midnight_view_checkbox.setToolTip(
            midnight_tip
            if self.next_day_summary is not None
            else "Not available - no forecast data for the next day"
        )
        self.midnight_view_checkbox.toggled.connect(self._on_midnight_view_toggled)
        layout.addWidget(self.midnight_view_checkbox)

        # Hourly astro score chart
        chart_group = QGroupBox("Hourly Astro Score")
        chart_layout = QVBoxLayout(chart_group)
        initial_sun_alts = self._get_sun_altitudes(self.summary.hourly_data)
        self.chart = HourlyAstroChart(self.summary.hourly_data, initial_sun_alts, self)
        self.chart.setMinimumHeight(200)
        self.chart.setMaximumHeight(250)
        self.chart.hour_hovered.connect(self._on_chart_hover)
        chart_layout.addWidget(self.chart)
        layout.addWidget(chart_group)

        # Hourly data table
        table_group = QGroupBox("Hourly Forecast (Nighttime hours highlighted)")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "Time", "Cloud %", "Low", "Mid", "High",
            f"Temp ({self._temp_suffix()})", f"Dew ({self._temp_suffix()})", "Humidity %",
            f"Wind ({self._wind_suffix()})", f"Gusts ({self._wind_suffix()})", "Precip %", "Vis (km)", "Press"
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMouseTracking(True)
        self.table.cellEntered.connect(self._on_table_cell_hover)

        # Configure header
        header = self.table.horizontalHeader()
        # Use Fixed mode for Time column to prevent flickering during row selection
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.resizeSection(0, 60)  # Fixed width for time column
        for i in range(1, 13):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        # Populate table with initial data
        self._populate_table(self.summary.hourly_data)

        table_layout.addWidget(self.table)
        layout.addWidget(table_group)

        # Initialize header stats for default view
        self._update_header_stats()

        # Restore saved midnight view state (after chart and table are created)
        settings = QSettings("CosmosCollection", "CosmosCollection")
        saved_midnight_view = settings.value("weather_center_on_midnight", False, type=bool)
        if saved_midnight_view and self.next_day_summary is not None:
            self.midnight_view_checkbox.setChecked(True)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

    def _on_midnight_view_toggled(self, checked: bool):
        """Save midnight view preference and update the view"""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("weather_center_on_midnight", checked)
        self._update_view()

    def _on_chart_hover(self, row_index: int):
        """Handle hover events from the chart to select table row"""
        if row_index >= 0 and row_index < self.table.rowCount():
            self.table.selectRow(row_index)
            self.table.scrollTo(self.table.model().index(row_index, 0))
        else:
            self.table.clearSelection()

    def _on_table_cell_hover(self, row: int, column: int):
        """Highlight bar and table row when hovering over table"""
        self.table.selectRow(row)
        self.chart.highlight_bar(row)

    def leaveEvent(self, event):
        """Clear bar highlight when mouse leaves the dialog"""
        self.chart.clear_bar_highlight()
        super().leaveEvent(event)

    def _get_display_hours(self) -> List[HourlyWeatherData]:
        """Get the hourly data to display based on current view mode"""
        if self.midnight_view_checkbox.isChecked() and self.next_day_summary:
            # Get evening hours from current day and morning hours from next day
            evening_hours = [h for h in self.summary.hourly_data if h.time.hour >= 12]
            morning_hours = [h for h in self.next_day_summary.hourly_data if h.time.hour < 12]
            all_night_hours = evening_hours + morning_hours

            # Filter to only include actual dark hours (sun altitude < -12°)
            if self.lat is not None and self.lon is not None:
                sun_alts = self._get_sun_altitudes(all_night_hours)
                if sun_alts:
                    dark_hours = [h for h, alt in zip(all_night_hours, sun_alts) if alt < -12]
                    if dark_hours:
                        return dark_hours

            # Fallback to fixed range if sun altitudes unavailable
            evening_hours = [h for h in self.summary.hourly_data if h.time.hour >= 18]
            morning_hours = [h for h in self.next_day_summary.hourly_data if h.time.hour <= 5]
            return evening_hours + morning_hours
        else:
            return self.summary.hourly_data

    def _update_header_stats(self):
        """Update header to reflect current view mode"""
        display_hours = self._get_display_hours()
        if not display_hours:
            return

        # Cloud cover stats
        clouds = [h.cloud_cover for h in display_hours]
        min_cloud, max_cloud, avg_cloud = min(clouds), max(clouds), sum(clouds) / len(clouds)
        self.cloud_label.setText(f"Cloud Cover: {min_cloud:.0f}% - {max_cloud:.0f}% (avg: {avg_cloud:.0f}%)")

        # Temperature stats
        temps = [h.temperature for h in display_hours]
        min_temp = self._convert_temp(min(temps))
        max_temp = self._convert_temp(max(temps))
        self.temp_label.setText(f"Temperature: {min_temp:.1f} - {max_temp:.1f}{self._temp_suffix()}")

        # Wind stats
        winds = [h.wind_speed for h in display_hours]
        avg_wind = self._convert_wind(sum(winds) / len(winds))
        max_wind = self._convert_wind(max(winds))
        self.wind_label.setText(f"Wind: avg {avg_wind:.1f} {self._wind_suffix()}, max {max_wind:.1f} {self._wind_suffix()}")

        # Humidity stats
        humidities = [h.humidity for h in display_hours]
        avg_humidity = sum(humidities) / len(humidities)
        self.humidity_label.setText(f"Humidity: {avg_humidity:.0f}%")

    def _update_view(self):
        """Refresh chart and table when view mode changes"""
        display_hours = self._get_display_hours()
        sun_altitudes = self._get_sun_altitudes(display_hours)
        self.chart.refresh(display_hours, sun_altitudes)
        self._populate_table(display_hours)
        self._update_header_stats()

    def _populate_table(self, hourly_data: List[HourlyWeatherData]):
        """Populate the table with hourly weather data"""
        self.table.setRowCount(len(hourly_data))
        for row, hour_data in enumerate(hourly_data):
            # Check if nighttime (6pm - 6am)
            is_night = hour_data.time.hour >= 18 or hour_data.time.hour < 6

            # Time
            time_item = QTableWidgetItem(format_time(hour_data.time))
            time_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                time_item.setBackground(QColor(COLORS['background_lighter']))
            self.table.setItem(row, 0, time_item)

            # Cloud cover columns
            for col, value in enumerate([
                hour_data.cloud_cover,
                hour_data.cloud_cover_low,
                hour_data.cloud_cover_mid,
                hour_data.cloud_cover_high
            ], start=1):
                item = QTableWidgetItem(f"{value:.0f}")
                item.setTextAlignment(Qt.AlignCenter)
                if is_night:
                    item.setBackground(QColor(COLORS['background_lighter']))
                # Color code cloud cover
                if value <= 20:
                    item.setForeground(QColor(COLORS['success']))
                elif value <= 50:
                    item.setForeground(QColor(COLORS['info']))
                elif value <= 70:
                    item.setForeground(QColor(COLORS['warning']))
                else:
                    item.setForeground(QColor(COLORS['error']))
                self.table.setItem(row, col, item)

            # Temperature
            temp_item = QTableWidgetItem(f"{self._convert_temp(hour_data.temperature):.1f}")
            temp_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                temp_item.setBackground(QColor(COLORS['background_lighter']))
            self.table.setItem(row, 5, temp_item)

            # Dew point
            dew_item = QTableWidgetItem(f"{self._convert_temp(hour_data.dew_point):.1f}")
            dew_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                dew_item.setBackground(QColor(COLORS['background_lighter']))
            # Warn if close to dew point
            temp_dew_spread = hour_data.temperature - hour_data.dew_point
            if temp_dew_spread < 2:
                dew_item.setForeground(QColor(COLORS['error']))
            elif temp_dew_spread < 5:
                dew_item.setForeground(QColor(COLORS['warning']))
            self.table.setItem(row, 6, dew_item)

            # Humidity
            humid_item = QTableWidgetItem(f"{hour_data.humidity:.0f}")
            humid_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                humid_item.setBackground(QColor(COLORS['background_lighter']))
            if hour_data.humidity > 80:
                humid_item.setForeground(QColor(COLORS['error']))
            elif hour_data.humidity > 65:
                humid_item.setForeground(QColor(COLORS['warning']))
            self.table.setItem(row, 7, humid_item)

            # Wind speed (convert for display, but use original km/h for color thresholds)
            wind_converted = self._convert_wind(hour_data.wind_speed)
            wind_item = QTableWidgetItem(f"{wind_converted:.1f}")
            wind_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                wind_item.setBackground(QColor(COLORS['background_lighter']))
            if hour_data.wind_speed > 25:  # Thresholds in km/h
                wind_item.setForeground(QColor(COLORS['error']))
            elif hour_data.wind_speed > 15:
                wind_item.setForeground(QColor(COLORS['warning']))
            self.table.setItem(row, 8, wind_item)

            # Wind gusts
            gusts_converted = self._convert_wind(hour_data.wind_gusts)
            gusts_item = QTableWidgetItem(f"{gusts_converted:.1f}")
            gusts_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                gusts_item.setBackground(QColor(COLORS['background_lighter']))
            if hour_data.wind_gusts > 25:  # Thresholds in km/h
                gusts_item.setForeground(QColor(COLORS['error']))
            elif hour_data.wind_gusts > 15:
                gusts_item.setForeground(QColor(COLORS['warning']))
            self.table.setItem(row, 9, gusts_item)

            # Precipitation probability
            precip_item = QTableWidgetItem(f"{hour_data.precipitation_probability:.0f}")
            precip_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                precip_item.setBackground(QColor(COLORS['background_lighter']))
            if hour_data.precipitation_probability > 50:
                precip_item.setForeground(QColor(COLORS['error']))
            elif hour_data.precipitation_probability > 20:
                precip_item.setForeground(QColor(COLORS['warning']))
            self.table.setItem(row, 10, precip_item)

            # Visibility (convert from meters to km)
            if hour_data.visibility is not None:
                vis_km = hour_data.visibility / 1000.0
                vis_item = QTableWidgetItem(f"{vis_km:.0f}")
            else:
                vis_item = QTableWidgetItem("-")
            vis_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                vis_item.setBackground(QColor(COLORS['background_lighter']))
            # Color code visibility (transparency indicator)
            if hour_data.visibility is not None:
                if vis_km >= 40:
                    vis_item.setForeground(QColor(COLORS['success']))  # Excellent
                elif vis_km >= 20:
                    vis_item.setForeground(QColor(COLORS['info']))  # Good
                elif vis_km >= 10:
                    vis_item.setForeground(QColor(COLORS['warning']))  # Moderate
                else:
                    vis_item.setForeground(QColor(COLORS['error']))  # Poor
            self.table.setItem(row, 11, vis_item)

            # Surface pressure
            if hour_data.surface_pressure is not None:
                press_item = QTableWidgetItem(f"{hour_data.surface_pressure:.0f}")
            else:
                press_item = QTableWidgetItem("-")
            press_item.setTextAlignment(Qt.AlignCenter)
            if is_night:
                press_item.setBackground(QColor(COLORS['background_lighter']))
            # Color code pressure (seeing indicator)
            if hour_data.surface_pressure is not None:
                if hour_data.surface_pressure >= 1020:
                    press_item.setForeground(QColor(COLORS['success']))  # High pressure - stable
                elif hour_data.surface_pressure >= 1010:
                    press_item.setForeground(QColor(COLORS['info']))  # Normal-high
                elif hour_data.surface_pressure >= 1000:
                    pass  # Normal - default color
                else:
                    press_item.setForeground(QColor(COLORS['warning']))  # Low pressure
            self.table.setItem(row, 12, press_item)


class WeatherForecastWindow(WindowPositionMixin, QMainWindow):
    """Main window for weather forecast display"""
    WINDOW_POSITION_KEY = "WeatherForecast"

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("Weather Forecast - Cosmos Collection")
        self.resize(815, 500)
        self.setup_window_position()

        self.worker = None
        self.daily_summaries: List[DailyWeatherSummary] = []
        self.day_cards: List[DayWeatherCard] = []

        # Get temperature unit preference
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.temp_unit = settings.value("temperature_unit", "Celsius", type=str)

        # Auto-refresh timer
        self.auto_refresh_timer = QTimer(self)
        self.auto_refresh_timer.timeout.connect(self._on_auto_refresh_triggered)
        self.next_refresh_time: Optional[datetime] = None

        self._setup_ui()
        self._setup_menu_bar()
        self._restore_auto_refresh_settings()
        self._load_location()

    def _setup_ui(self):
        """Set up the main window UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Header with location and refresh button
        header_layout = QHBoxLayout()
        self.location_label = QLabel("Location: Loading...")
        self.location_label.setStyleSheet("font-size: 11pt;")
        header_layout.addWidget(self.location_label)
        header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh Forecast")
        self.refresh_btn.setToolTip("Fetch fresh weather data (bypasses cache)")
        self.refresh_btn.clicked.connect(lambda: self._refresh_forecast(force=True))
        header_layout.addWidget(self.refresh_btn)

        # Auto-refresh controls
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
        self.auto_refresh_checkbox.setToolTip("Automatically refresh weather data at the selected interval")
        self.auto_refresh_checkbox.setChecked(False)
        self.auto_refresh_checkbox.stateChanged.connect(self._on_auto_refresh_changed)
        header_layout.addWidget(self.auto_refresh_checkbox)

        self.refresh_interval_combo = QComboBox()
        self.refresh_interval_combo.setToolTip("Select auto-refresh interval")
        self.refresh_interval_combo.addItem("15 min", 15)
        self.refresh_interval_combo.addItem("30 min", 30)
        self.refresh_interval_combo.addItem("1 hour", 60)
        self.refresh_interval_combo.addItem("2 hours", 120)
        self.refresh_interval_combo.addItem("4 hours", 240)
        self.refresh_interval_combo.setCurrentIndex(2)  # Default to 1 hour
        self.refresh_interval_combo.currentIndexChanged.connect(self._on_refresh_interval_changed)
        self.refresh_interval_combo.setEnabled(False)  # Disabled until auto-refresh is enabled
        header_layout.addWidget(self.refresh_interval_combo)

        main_layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Weekly overview group
        overview_group = QGroupBox("Weekly Overview")
        overview_layout = QVBoxLayout(overview_group)

        # Scroll area for day cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumHeight(220)  # Minimum to show cards properly

        self.cards_widget = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(10)
        self.cards_layout.setContentsMargins(5, 5, 5, 5)
        self.cards_layout.addStretch()

        scroll_area.setWidget(self.cards_widget)
        overview_layout.addWidget(scroll_area)

        # Help text with attribution
        help_text = QLabel(
            'Double-click a day card for detailed hourly forecast. '
            'Weather data provided by <a href="https://open-meteo.com/" style="color: #6ea8fe;">Open-Meteo</a>.'
        )
        help_text.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 9pt;")
        help_text.setAlignment(Qt.AlignCenter)
        help_text.setOpenExternalLinks(False)
        help_text.linkActivated.connect(lambda url: open_url(url))
        overview_layout.addWidget(help_text)

        main_layout.addWidget(overview_group, 1)  # Give stretch factor to expand

        # Legend
        legend_group = QGroupBox("Rating Legend")
        legend_layout = QHBoxLayout(legend_group)
        legend_layout.addStretch()

        for label, color in [
            ("Excellent (80-100)", COLORS['success']),
            ("Good (60-79)", COLORS['info']),
            ("Moderate (40-59)", COLORS['warning']),
            ("Poor (0-39)", COLORS['error'])
        ]:
            legend_item = QLabel(f"  {label}  ")
            legend_item.setStyleSheet(f"color: {color}; font-weight: bold;")
            legend_layout.addWidget(legend_item)

        legend_layout.addStretch()
        main_layout.addWidget(legend_group)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        main_layout.addWidget(self.status_label)

    def _setup_menu_bar(self):
        """Set up the menu bar with external weather links"""
        menu_bar = self.menuBar()

        # Add actions directly to menu bar (no dropdown needed)
        clear_outside_action = menu_bar.addAction("Clear Outside")
        clear_outside_action.setToolTip("Open Clear Outside astronomy forecast in browser")
        clear_outside_action.triggered.connect(self._open_clear_outside)

        astrospheric_action = menu_bar.addAction("Astrospheric")
        astrospheric_action.setToolTip("Open Astrospheric astronomy forecast in browser")
        astrospheric_action.triggered.connect(self._open_astrospheric)

        noaa_action = menu_bar.addAction("NOAA")
        noaa_action.setToolTip("Open NOAA 7-day weather forecast in browser")
        noaa_action.triggered.connect(self._open_noaa)

    def _open_clear_outside(self):
        """Open Clear Outside forecast in browser"""
        if self.lat is not None and self.lon is not None:
            url = f"https://clearoutside.com/forecast/{self.lat}/{self.lon}"
            open_url(url)

    def _open_astrospheric(self):
        """Open Astrospheric forecast in browser"""
        if self.lat is not None and self.lon is not None:
            url = f"https://www.astrospheric.com/?Latitude={self.lat}&Longitude={self.lon}"
            open_url(url)

    def _open_noaa(self):
        """Open NOAA 7-day forecast in browser"""
        if self.lat is not None and self.lon is not None:
            url = f"https://forecast.weather.gov/MapClick.php?lat={self.lat}&lon={self.lon}"
            open_url(url)

    def _load_location(self):
        """Load location from database"""
        # Check if observer location should be shown
        settings = QSettings("CosmosCollection", "CosmosCollection")
        show_location = settings.value("show_observer_location", True, type=bool)

        # Hide the location label if setting is disabled
        self.location_label.setVisible(show_location)

        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT location_lat, location_lon, location_name, timezone "
                    "FROM usersettings WHERE is_active = 1 LIMIT 1"
                )
                row = cursor.fetchone()
                if not row:
                    cursor.execute(
                        "SELECT location_lat, location_lon, location_name, timezone "
                        "FROM usersettings ORDER BY id DESC LIMIT 1"
                    )
                    row = cursor.fetchone()

                if row:
                    self.lat, self.lon, location_name, self.timezone = row
                    if self.lat is not None and self.lon is not None:
                        lat_str = f"{abs(self.lat):.2f}{'N' if self.lat >= 0 else 'S'}"
                        lon_str = f"{abs(self.lon):.2f}{'W' if self.lon < 0 else 'E'}"
                        display_name = location_name if location_name else "User Location"
                        self.location_label.setText(f"Location: {display_name} ({lat_str}, {lon_str})")

                        # Auto-fetch weather
                        self._refresh_forecast()
                        return

            # No location configured
            self.lat = None
            self.lon = None
            self.timezone = None
            self.location_label.setText("Location: Not configured")
            self.status_label.setText("Please configure your location in Settings to view weather forecast.")
            self.status_label.setStyleSheet(f"color: {COLORS['warning']};")
            self.refresh_btn.setEnabled(False)

        except Exception as e:
            logger.error(f"Error loading location: {str(e)}")
            self.status_label.setText(f"Error loading location: {str(e)}")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")

    def _refresh_forecast(self, force: bool = False):
        """Fetch weather forecast data, using cache if available"""
        if self.lat is None or self.lon is None:
            QMessageBox.warning(self, "No Location",
                "Please configure your observer location in Settings before viewing weather forecast.")
            return

        if self.worker and self.worker.isRunning():
            return

        # Check cache first (unless force refresh)
        if not force:
            cache = WeatherCache()
            cached_data = cache.get(self.lat, self.lon)
            if cached_data is not None:
                self._on_weather_loaded(cached_data, from_cache=True)
                return

        # Show progress
        self.progress_bar.setVisible(True)
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Fetching weather data...")
        self.status_label.setStyleSheet("")

        # Start worker thread
        self.worker = WeatherWorker(self.lat, self.lon, self.timezone)
        self.worker.weather_loaded.connect(self._on_weather_loaded)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.start()

    def _on_progress(self, message: str):
        """Handle progress updates"""
        self.status_label.setText(message)

    def _on_weather_loaded(self, summaries: List[DailyWeatherSummary], from_cache: bool = False):
        """Handle successful weather data load"""
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.daily_summaries = summaries

        # Store in cache if this is fresh data
        if not from_cache and self.lat is not None and self.lon is not None:
            cache = WeatherCache()
            cache.set(self.lat, self.lon, summaries)

        # Clear existing cards
        for card in self.day_cards:
            card.deleteLater()
        self.day_cards.clear()

        # Remove stretch
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create new cards
        for summary in summaries:
            card = DayWeatherCard(summary)
            card.clicked.connect(self._show_day_detail)
            self.cards_layout.addWidget(card)
            self.day_cards.append(card)

        self.cards_layout.addStretch()

        # Update status
        if from_cache:
            cache = WeatherCache()
            age_str = cache.get_age_str()
            status_text = f"Using cached data (fetched {age_str})"
        else:
            now = datetime.now()
            status_text = f"Last updated: {format_datetime(now)}"

        # Append next refresh time if auto-refresh is enabled
        if self.next_refresh_time is not None:
            status_text += f" | Next refresh: {format_time(self.next_refresh_time)}"

        self.status_label.setText(status_text)
        self.status_label.setStyleSheet("")

    def _on_error(self, error_message: str):
        """Handle errors from worker"""
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet(f"color: {COLORS['error']};")
        logger.error(f"Weather Fetch Error: {str(error_message)}")

    def _show_day_detail(self, summary: DailyWeatherSummary):
        """Show detailed dialog for a day"""
        # Find next day's summary if available
        next_day_summary = None
        try:
            current_index = self.daily_summaries.index(summary)
            if current_index + 1 < len(self.daily_summaries):
                next_day_summary = self.daily_summaries[current_index + 1]
        except ValueError:
            pass  # Summary not found in list

        dialog = DayDetailDialog(summary, next_day_summary, self.lat, self.lon, self.timezone, self)
        dialog.show()

    def _on_auto_refresh_changed(self, state: int):
        """Handle auto-refresh checkbox state change"""
        enabled = state == Qt.Checked.value
        self.refresh_interval_combo.setEnabled(enabled)

        # Save setting
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("weather_auto_refresh_enabled", enabled)

        if enabled:
            # Start the timer with the selected interval
            self._start_auto_refresh_timer()
        else:
            self.auto_refresh_timer.stop()
            self.next_refresh_time = None
            self._update_status_with_next_refresh()

    def _on_refresh_interval_changed(self, index: int):
        """Handle refresh interval combo box change"""
        # Save setting
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("weather_auto_refresh_interval", index)

        if self.auto_refresh_checkbox.isChecked():
            # Restart timer with new interval
            self._start_auto_refresh_timer()

    def _restore_auto_refresh_settings(self):
        """Restore auto-refresh settings from saved state"""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Restore interval first (before enabling auto-refresh)
        interval_index = settings.value("weather_auto_refresh_interval", 2, type=int)  # Default to 1 hour
        if 0 <= interval_index < self.refresh_interval_combo.count():
            self.refresh_interval_combo.setCurrentIndex(interval_index)

        # Restore auto-refresh enabled state
        auto_refresh_enabled = settings.value("weather_auto_refresh_enabled", False, type=bool)
        self.auto_refresh_checkbox.setChecked(auto_refresh_enabled)

    def _start_auto_refresh_timer(self):
        """Start the auto-refresh timer and update next refresh time"""
        interval_minutes = self.refresh_interval_combo.currentData()
        self.next_refresh_time = datetime.now() + timedelta(minutes=interval_minutes)
        self.auto_refresh_timer.start(interval_minutes * 60 * 1000)
        self._update_status_with_next_refresh()

    def _on_auto_refresh_triggered(self):
        """Handle auto-refresh timer timeout"""
        self._refresh_forecast(force=True)
        # Schedule next refresh
        if self.auto_refresh_checkbox.isChecked():
            interval_minutes = self.refresh_interval_combo.currentData()
            self.next_refresh_time = datetime.now() + timedelta(minutes=interval_minutes)

    def _update_status_with_next_refresh(self):
        """Update the status label to include next refresh time"""
        current_text = self.status_label.text()
        # Remove any existing next refresh info
        if " | Next refresh:" in current_text:
            current_text = current_text.split(" | Next refresh:")[0]

        if self.next_refresh_time is not None:
            current_text += f" | Next refresh: {format_time(self.next_refresh_time)}"

        self.status_label.setText(current_text)


def main():
    """Main entry point for standalone testing"""
    app = QApplication(sys.argv)
    window = WeatherForecastWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
