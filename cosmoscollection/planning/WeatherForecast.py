#!/usr/bin/env python3
"""
Weather Forecast Window for Cosmos Collection
Displays astrophotography-relevant weather data from Open-Meteo API
"""

import sys
import time
import gc
import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import matplotlib
matplotlib.use('QtAgg')

# Suppress matplotlib font_manager debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import requests
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QGroupBox, QMessageBox, QProgressBar, QScrollArea,
    QFrame, QGridLayout, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QApplication, QSplitter, QCheckBox, QComboBox, QSizePolicy
)
from PySide6.QtGui import QColor

from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.planning.Observer import active_site, coordinates_text
from cosmoscollection.core.Settings import app_settings
from cosmoscollection.ui.WindowPositionManager import WindowPositionMixin
from cosmoscollection.ui.WindowLifetime import release_on_close
from cosmoscollection.ui.Theme import COLORS, chart_background, chart_font_size, font_size, sky_shade, theme_manager, themed_style
from cosmoscollection.ui.TimeFormatHelper import format_time, format_datetime, get_time_format_24h
from cosmoscollection.core.UrlOpener import open_url
from cosmoscollection.core.OnlineLimits import OFFLINE_NOTE, is_offline

# Set up logging
logger = logging.getLogger(__name__)

# Cache for weather data (persists across window instances)
CACHE_MAX_AGE_MINUTES = 15


from cosmoscollection.planning.WeatherFetch import (  # noqa: E402,F401
    DailyWeatherSummary, HourlyWeatherData, MoonPhaseData, WeatherFetch, WeatherUnavailable,
    _get_moon_phase_name, _get_openweather_signature, _get_visualcrossing_signature, _get_weatherapi_signature,
    calculate_astro_score, calculate_moon_phase, calculate_sun_altitudes, estimate_seeing, get_rating_label,
    test_openweather_key, test_visualcrossing_key, test_weatherapi_key)


class WeatherCache:
    """Simple cache for weather data to reduce API calls"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WeatherCache, cls).__new__(cls)
            cls._instance._data = None
            cls._instance._timestamp = None
            cls._instance._location = None
            cls._instance._openweather_signature = None
            cls._instance._visualcrossing_signature = None
            cls._instance._weatherapi_signature = None
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

        # Check if any optional integration's settings changed since this data was
        # fetched (e.g. user just enabled one / added an API key) - if so, the
        # cached data doesn't reflect the current configuration, so treat it as stale.
        if self._openweather_signature != _get_openweather_signature():
            logger.debug("Weather cache miss: OpenWeather integration settings changed")
            return None
        if self._visualcrossing_signature != _get_visualcrossing_signature():
            logger.debug("Weather cache miss: VisualCrossing integration settings changed")
            return None
        if self._weatherapi_signature != _get_weatherapi_signature():
            logger.debug("Weather cache miss: WeatherAPI integration settings changed")
            return None

        logger.debug(f"Weather cache hit: data is {age.seconds // 60} minutes old")
        return self._data

    def set(self, lat: float, lon: float, data: List):
        """Store data in cache and notify callbacks"""
        self._data = data
        self._timestamp = datetime.now()
        self._location = (lat, lon)
        self._openweather_signature = _get_openweather_signature()
        self._visualcrossing_signature = _get_visualcrossing_signature()
        self._weatherapi_signature = _get_weatherapi_signature()
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
        self._openweather_signature = None
        self._visualcrossing_signature = None
        self._weatherapi_signature = None

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


class WeatherWorker(QThread):
    """Runs a WeatherFetch (see WeatherFetch.py) off the UI thread."""
    weather_loaded = Signal(list)  # List of DailyWeatherSummary
    error_occurred = Signal(str)
    progress = Signal(str)

    def __init__(self, lat: float, lon: float, timezone: str = None):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.timezone = timezone
        self.fetcher = WeatherFetch(lat, lon, timezone, progress=self.progress.emit)

    # Which sources failed, and the one the hourly grid came from (see WeatherFetch)
    openmeteo_error = property(lambda self: self.fetcher.openmeteo_error)
    openweather_error = property(lambda self: self.fetcher.openweather_error)
    visualcrossing_error = property(lambda self: self.fetcher.visualcrossing_error)
    weatherapi_error = property(lambda self: self.fetcher.weatherapi_error)
    baseline_source = property(lambda self: self.fetcher.baseline_source)

    def run(self):
        try:
            sources = self.fetcher.fetch_sources()
            self.progress.emit("Processing weather data...")
            # Disable the cyclic GC from here through the weather_loaded emit below. A
            # native crash (Fatal Python error: Illegal instruction) has recurred at
            # several unrelated first-call sites inside _process_weather_data (astropy
            # units.cds import, Time.replicate, an ecliptic frame transform, numpy
            # arrayprint/__str__, pytz.timezone's os.environ lookup) -- the one constant
            # every time is "Garbage-collecting" at the top of the trace, in this thread.
            # That points at a GC pass landing while some C extension has left transient
            # state a GC traversal isn't safe to observe, not at any one of those
            # libraries specifically. Re-enabling gc immediately after
            # _process_weather_data returned (right before this comment used to sit)
            # wasn't enough -- the crash recurred one line later, right after
            # gc.enable(), inside weather_loaded.emit() itself (cross-thread signal
            # delivery allocates too). So the disabled window now extends through the
            # emit call as well. gc.disable()/enable() is process-wide, not thread-local,
            # but this whole stretch is quick and runs once per weather refresh, so
            # briefly deferring collection elsewhere is a small price for not crashing.
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                daily_summaries = self.fetcher.process(*sources)
                self.weather_loaded.emit(daily_summaries)
            finally:
                if gc_was_enabled:
                    gc.enable()

        except WeatherUnavailable as e:
            self.error_occurred.emit(str(e))
        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Error: {str(e)}")


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

        self._setup_ui()
        self._apply_styles()

    def _apply_styles(self):
        """Theme-colored styles (rebuilt automatically when the theme changes)"""
        themed_style(self, lambda self=self: f"""
            DayWeatherCard {{
                background-color: {COLORS['background_light']};
                border: 2px solid {get_rating_color(self.summary.astro_score)};
                border-radius: 8px;
                padding: 5px;
            }}
            DayWeatherCard:hover {{
                background-color: {COLORS['background_hover']};
                border: 2px solid {get_rating_color(self.summary.astro_score)};
            }}
        """)
        themed_style(self.date_label, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(10)};")
        themed_style(self.rating_text, lambda self=self: f"color: {get_rating_color(self.summary.astro_score)}; font-weight: bold; font-size: {font_size(12)};")
        themed_style(self.score_label, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(10)};")
        themed_style(self.cloud_text, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(12)};")

    def _setup_ui(self):
        """Set up the card UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        # Day name
        day_name = self.summary.date.strftime("%a")
        day_label = QLabel(day_name)
        day_label.setAlignment(Qt.AlignCenter)
        themed_style(day_label, lambda: f"font-weight: bold; font-size: {font_size(12)};")
        layout.addWidget(day_label)

        # Date
        date_str = self.summary.date.strftime("%m/%d")
        self.date_label = QLabel(date_str)
        self.date_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.date_label)

        layout.addSpacing(5)

        # Rating
        rating_label = get_rating_label(self.summary.astro_score)
        self.rating_text = QLabel(rating_label)
        self.rating_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.rating_text)

        # Score
        self.score_label = QLabel(f"({self.summary.astro_score})")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setToolTip("Astro Score")
        layout.addWidget(self.score_label)

        layout.addSpacing(5)

        self.cloud_text = QLabel("Clouds")
        self.cloud_text.setAlignment(Qt.AlignCenter)
        self.cloud_text.setToolTip("Average cloud cover for dark hours (sun altitude < -12°)")
        layout.addWidget(self.cloud_text)

        # Cloud cover (tonight's average - dark hours only)
        cloud_label = QLabel(f"{self.summary.tonight_avg_cloud_cover:.0f}%")
        cloud_label.setAlignment(Qt.AlignCenter)
        themed_style(cloud_label, lambda: f"font-size: {font_size(10)}; font-weight: bold;")
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
        self.figure = Figure(figsize=(10, 3))
        super().__init__(self.figure)
        self.setParent(parent)

        self.hourly_data = hourly_data
        self.sun_altitudes = sun_altitudes
        self.bars = None
        self.ax = None
        self._last_hovered_index = -1
        self._create_chart()
        theme_manager().theme_changed.connect(self._create_chart)

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
        self.figure.set_facecolor(chart_background())
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
            # "Tonight" starts on the evening date, even though the view stretches into the next morning
            night_date_str = self.hourly_data[0].time.strftime('%b %d')
            self.ax.set_xlabel('Time (Evening → Morning)', color=COLORS['text'], fontsize=chart_font_size(9))
            self.ax.set_title(f'Astrophotography Conditions - Night of {night_date_str}', color=COLORS['text'], fontsize=chart_font_size(10), fontweight='bold')
        else:
            day_date_str = self.hourly_data[0].time.strftime('%b %d') if self.hourly_data else ''
            self.ax.set_xlabel('Hour of Day', color=COLORS['text'], fontsize=chart_font_size(9))
            self.ax.set_title(f'Hourly Astrophotography Conditions - {day_date_str}', color=COLORS['text'], fontsize=chart_font_size(10), fontweight='bold')

        # Style the chart
        self.ax.set_xlim(-0.5, num_hours - 0.5)
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Astro Score', color=COLORS['text'], fontsize=chart_font_size(9))

        # Set x-axis ticks - show every other label to avoid crowding
        tick_step = 2 if num_hours > 12 else 1
        tick_positions = list(range(0, num_hours, tick_step))
        tick_labels = [hour_labels[i] for i in tick_positions]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels, fontsize=chart_font_size(8))

        # Style axes
        self.ax.set_facecolor(COLORS['background_light'])
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=chart_font_size(8))
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
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=chart_font_size(7),
                  facecolor=COLORS['background_lighter'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

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

        # Sun altitude thresholds and theme shades
        # Format: (sun_max, sun_min, (color, alpha))
        darkness_levels = [
            (90, 0, sky_shade('day')),
            (0, -6, sky_shade('civil')),
            (-6, -12, sky_shade('nautical')),
            (-12, -18, sky_shade('astronomical')),
            (-18, -90, sky_shade('night')),
        ]

        # For each hour, shade based on its sun altitude
        for i, sun_alt in enumerate(self.sun_altitudes):
            for sun_max, sun_min, (color, alpha) in darkness_levels:
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
                bar.set_edgecolor(COLORS['text'])
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
        settings = app_settings()
        self.temp_unit = settings.value("temperature_unit", "Celsius", type=str)
        self.wind_unit = settings.value("wind_speed_unit", "km/h", type=str)
        self.precip_unit = settings.value("precip_visibility_unit", "Metric (mm, km)", type=str)

        self._setup_ui()
        # The table's cell colors are item data, so refill it when the theme
        # changes (the chart and widget styles follow on their own)
        theme_manager().theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        self._populate_table(self._get_display_hours())

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
        rating_label = QLabel(f"Astro Rating: {get_rating_label(self.summary.astro_score)} ({self.summary.astro_score})")
        themed_style(rating_label, lambda self=self: f"color: {get_rating_color(self.summary.astro_score)}; font-weight: bold; font-size: {font_size(12)};")
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
        settings = app_settings()
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
        settings = app_settings()
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
        release_on_close(self)  # Freed once closed (see WindowLifetime)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("Weather Forecast - Cosmos Collection")
        self.resize(815, 500)
        self.setup_window_position()

        self.worker = None
        self.daily_summaries: List[DailyWeatherSummary] = []
        self.day_cards: List[DayWeatherCard] = []

        # Get temperature unit preference
        settings = app_settings()
        self.temp_unit = settings.value("temperature_unit", "Celsius", type=str)

        # Auto-refresh timer
        self.auto_refresh_timer = QTimer(self)
        self.auto_refresh_timer.timeout.connect(self._on_auto_refresh_triggered)
        self.next_refresh_time: Optional[datetime] = None

        self._setup_ui()
        self._setup_menu_bar()
        self._restore_auto_refresh_settings()
        self._load_location()

        # Styles follow theme changes on their own (Theme.themed_style), but the
        # attribution's link colors are part of its text, so rebuild that
        theme_manager().theme_changed.connect(self._update_attribution)

    def _set_status_color(self, color_key):
        """Color the status line with a COLORS key (None for the default text color)"""
        themed_style(self.status_label, lambda color_key=color_key: f"color: {COLORS[color_key]};" if color_key else "")

    def _update_attribution(self):
        """Show which weather sources the current forecast came from"""
        source_links = {
            "openmeteo": ("https://open-meteo.com/", "Open-Meteo"),
            "openweather": ("https://openweathermap.org/", "OpenWeather"),
            "visualcrossing": ("https://www.visualcrossing.com/", "VisualCrossing"),
            "weatherapi": ("https://www.weatherapi.com/", "WeatherAPI"),
        }
        sources = [f'<a href="{source_links[key][0]}" style="color: {COLORS["link"]};">{source_links[key][1]}</a>'
                   for key in self._attribution_sources]
        if len(sources) == 1:
            self.attribution_label.setText(
                'Double-click a day card for detailed hourly forecast. '
                f'Weather data provided by {sources[0]}.'
            )
        else:
            # "A and B" for two sources, "A, B, and C" (and so on) for more
            source_list = (
                f"{sources[0]} and {sources[1]}" if len(sources) == 2
                else f"{', '.join(sources[:-1])}, and {sources[-1]}"
            )
            self.attribution_label.setText(
                'Double-click a day card for detailed hourly forecast. '
                f'Weather data blended from {source_list}.'
            )

    def _setup_ui(self):
        """Set up the main window UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Header with location and refresh button
        header_layout = QHBoxLayout()
        self.location_label = QLabel("Location: Loading...")
        themed_style(self.location_label, lambda: f"font-size: {font_size(11)};")
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

        # Help text with attribution (text updated in _on_weather_loaded to
        # reflect whether OpenWeather, VisualCrossing, and/or WeatherAPI data was blended in)
        self.attribution_label = QLabel()
        self._attribution_sources = ["openmeteo"]
        self._update_attribution()
        themed_style(self.attribution_label, lambda: f"color: {COLORS['text_disabled']}; font-size: {font_size(9)};")
        self.attribution_label.setAlignment(Qt.AlignCenter)
        self.attribution_label.setOpenExternalLinks(False)
        self.attribution_label.linkActivated.connect(lambda url: open_url(url))
        overview_layout.addWidget(self.attribution_label)

        main_layout.addWidget(overview_group, 1)  # Give stretch factor to expand

        # Legend
        legend_group = QGroupBox("Rating Legend")
        legend_layout = QHBoxLayout(legend_group)
        legend_layout.addStretch()

        for label, color_key in [
            ("Excellent (80-100)", 'success'),
            ("Good (60-79)", 'info'),
            ("Moderate (40-59)", 'warning'),
            ("Poor (0-39)", 'error')
        ]:
            legend_item = QLabel(f"  {label}  ")
            themed_style(legend_item, lambda color_key=color_key: f"color: {COLORS[color_key]}; font-weight: bold;")
            legend_layout.addWidget(legend_item)

        legend_layout.addStretch()
        main_layout.addWidget(legend_group)

        # Status bar
        self.status_label = QLabel("Ready")
        self._set_status_color('text_secondary')
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
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
        settings = app_settings()
        show_location = settings.value("show_observer_location", True, type=bool)

        # Hide the location label if setting is disabled
        self.location_label.setVisible(show_location)

        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                site = active_site(conn)
                if site is not None:
                    self.lat, self.lon, self.timezone = site.lat, site.lon, site.timezone
                    self.location_label.setText(f"Location: {site.name} ({coordinates_text(site.lat, site.lon)})")
                    self._refresh_forecast()  # Auto-fetch weather
                    return

            # No location configured
            self.lat = None
            self.lon = None
            self.timezone = None
            self.location_label.setText("Location: Not configured")
            self.status_label.setText("Please configure your location in Settings to view weather forecast.")
            self._set_status_color('warning')
            self.refresh_btn.setEnabled(False)

        except Exception as e:
            logger.error(f"Error loading location: {str(e)}")
            self.status_label.setText(f"Error loading location: {str(e)}")
            self._set_status_color('error')

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

        if is_offline():
            self.status_label.setText(f"The weather forecast needs the internet. {OFFLINE_NOTE}")
            self._set_status_color('warning')
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Fetching weather data...")
        self._set_status_color(None)

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

        # Reflect the actual baseline source (normally Open-Meteo, but a
        # supplemental source if Open-Meteo itself failed on this fetch - see
        # WeatherWorker.baseline_source) plus which other source(s), if any, were
        # blended into it.
        baseline_source = "openmeteo"
        if not from_cache and self.worker is not None:
            baseline_source = getattr(self.worker, "baseline_source", "openmeteo")
        sources = [baseline_source]
        if baseline_source != "openweather" and any(s.openweather_blended for s in summaries):
            sources.append("openweather")
        if baseline_source != "visualcrossing" and any(s.visualcrossing_blended for s in summaries):
            sources.append("visualcrossing")
        if baseline_source != "weatherapi" and any(s.weatherapi_blended for s in summaries):
            sources.append("weatherapi")
        self._attribution_sources = sources
        self._update_attribution()

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

        # Note when an enabled optional integration failed on this fetch attempt, so
        # a broken integration (bad key, network issue, etc.) is never silent. Only
        # meaningful right after a fresh fetch - a cache hit didn't attempt new
        # OpenWeather/VisualCrossing/WeatherAPI calls, so there's nothing new to report.
        integration_notes = []
        if not from_cache and self.worker is not None:
            if getattr(self.worker, "openmeteo_error", None):
                integration_notes.append(f"Open-Meteo unavailable ({self.worker.openmeteo_error})")
            if getattr(self.worker, "openweather_error", None):
                integration_notes.append(f"OpenWeather unavailable ({self.worker.openweather_error})")
            if getattr(self.worker, "visualcrossing_error", None):
                integration_notes.append(f"VisualCrossing unavailable ({self.worker.visualcrossing_error})")
            if getattr(self.worker, "weatherapi_error", None):
                integration_notes.append(f"WeatherAPI unavailable ({self.worker.weatherapi_error})")
        integration_note = f" | {'; '.join(integration_notes)} - using remaining source(s) only" if integration_notes else ""

        # Update status
        if from_cache:
            cache = WeatherCache()
            age_str = cache.get_age_str()
            status_text = f"Using cached data (fetched {age_str})"
        else:
            now = datetime.now()
            status_text = f"Last updated: {format_datetime(now)}"

        status_text += integration_note

        # Append next refresh time if auto-refresh is enabled
        if self.next_refresh_time is not None:
            status_text += f" | Next refresh: {format_time(self.next_refresh_time)}"

        self.status_label.setText(status_text)
        self._set_status_color('warning' if integration_note else None)

    def _on_error(self, error_message: str):
        """Handle errors from worker"""
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        self._set_status_color('error')
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
        settings = app_settings()
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
        settings = app_settings()
        settings.setValue("weather_auto_refresh_interval", index)

        if self.auto_refresh_checkbox.isChecked():
            # Restart timer with new interval
            self._start_auto_refresh_timer()

    def _restore_auto_refresh_settings(self):
        """Restore auto-refresh settings from saved state"""
        settings = app_settings()

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
