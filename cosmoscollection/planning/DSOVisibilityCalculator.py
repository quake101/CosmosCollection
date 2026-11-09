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
from datetime import datetime, timedelta
from PySide6.QtCore import Qt, QDate, QThread, Signal, QTimer
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLineEdit, QLabel, QTextEdit,
                               QDateEdit, QSpinBox, QGroupBox, QMessageBox, QSizePolicy,
                               QComboBox, QSplitter)

matplotlib.use('Qt5Agg')

# Suppress matplotlib font_manager debug messages
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
import pytz
import warnings

warnings.filterwarnings('ignore')

from cosmoscollection.ui.Theme import (
    COLORS, chart_background, chart_color, chart_font_size, font_size, sky_shade, theme_manager, themed_style,
)
from cosmoscollection.planning.Observer import active_site, coordinates_text
from cosmoscollection.planning.MonthCalendars import MultiMonthCalendar, hours_legend_row
from cosmoscollection.planning.NightTimes import compass_direction
from cosmoscollection.core.Settings import app_settings
from cosmoscollection.core.Formatting import format_dec, format_duration, format_ra


# Import DatabaseManager from separate file
from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.ui.WindowPositionManager import WindowPositionMixin
from cosmoscollection.ui.WindowLifetime import release_on_close
from cosmoscollection.ui.TimeFormatHelper import format_time


from cosmoscollection.planning.Visibility import DSOVisibilityCalculator  # noqa: E402,F401
from cosmoscollection.planning.NightTimes import (  # noqa: E402
    ASTRONOMICAL_DARK_ALTITUDE, DARK_SKY_ALTITUDE, SKY_STATES, sky_state)


class MultiMonthVisibilityCalendar(MultiMonthCalendar):
    """The calculator's calendar: the visibility of the object last calculated
    (with the moon's illumination) at the active location, and its weather."""

    MONTH_COUNT_SETTING = "dso_visibility_calendar_month_count"
    CALENDAR_MAX_HEIGHT = 250

    def __init__(self, parent=None):
        self._request = None  # MonthlyVisibilityThread arguments for the object
        super().__init__(parent)

    def _add_above_calendars(self, layout):
        layout.addLayout(hours_legend_row())

    def _add_below_calendars(self, layout):
        self.status_label = QLabel("Select a DSO and calculate to see monthly visibility")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']}; font-style: italic;")
        layout.addWidget(self.status_label)

    def _visibility_status(self, text):
        self.status_label.setText(text)

    def _weather_location(self):
        site = DSOVisibilityCalculator.saved_site()
        return (site.lat, site.lon, site.timezone or 'UTC') if site else None

    def _visibility_request(self):
        return self._request

    def set_dso(self, dso_coord, dso_name, ra_deg=None, dec_deg=None, min_altitude=30):
        """Show the visibility of this object, (re)starting the calculation."""
        self._request = dict(dso_coord=dso_coord, dso_name=dso_name, ra_deg=ra_deg, dec_deg=dec_deg,
                             min_altitude=min_altitude, with_moon=True)
        self._refresh_visibility_layer()


GRAPH_POINTS_PER_HOUR = 12  # The calculator's graph: a point every 5 minutes


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
                    dso_coord, self.date, self.hours, self.min_altitude, self.dso_name,
                    horizon=self.calculator.horizon, time_resolution=GRAPH_POINTS_PER_HOUR)
            else:
                # Use name-based calculation (original behavior)
                results = self.calculator.calculate_visibility_for_date(
                    self.dso_name, self.date, self.hours, self.min_altitude,
                    horizon=self.calculator.horizon, time_resolution=GRAPH_POINTS_PER_HOUR)

            if "error" in results:
                self.error.emit(results["error"])
                return

            # Add local timezone for compatibility with existing UI code
            results['local_tz'] = self.local_tz
            # Active location's custom horizon (or None), drawn on the altitude chart
            results['horizon'] = self.calculator.horizon
            results['min_altitude'] = self.min_altitude
            results['meridian_times'] = self.calculator.find_meridian_crossings(
                results['time_range'], results['dso_coord'])

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Calculation error: {str(e)}")


SUNSET_ALT = -0.833         # The sun's top at the horizon, with refraction
NIGHT_PADDING_HOURS = 0.5   # Night view: this much daylight either side of sunset/sunrise
MIN_ZOOM_HOURS = 1.0        # Narrowest the time axis zooms to
ZOOM_STEP = 0.8             # Each wheel notch shows this fraction of the span
MERIDIAN_COLOR = '#4fc3f7'  # Matches the FOV Simulator's meridian line
NOW_COLOR = '#00ff00'
GRAPH_VIEW_SETTING = "DSOVisibilityApp/graph_view"          # 'night' or 'full'
GRAPH_SPLITTER_SETTING = "DSOVisibilityApp/graph_splitter"  # calendar / graph divider


class ClockTimeLocator(ticker.Locator):
    """Time axis ticks on round local clock times (every 15 minutes up to every 12
    hours, whichever fits the axis width), however far it's zoomed or panned.
    The axis is in hours from start_local."""

    STEPS = (0.25, 0.5, 1, 2, 3, 4, 6, 12)
    PIXELS_PER_TICK = 70

    def __init__(self, start_local):
        self.start_hour = start_local.hour + start_local.minute / 60 + start_local.second / 3600

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin, vmax = min(vmin, vmax), max(vmin, vmax)
        max_ticks = max(3, int(self.axis.axes.bbox.width / self.PIXELS_PER_TICK))
        step = next((s for s in self.STEPS if (vmax - vmin) / s <= max_ticks), self.STEPS[-1])
        first = np.ceil((vmin + self.start_hour) / step) * step - self.start_hour
        return np.arange(first, vmax + 1e-9, step)


class VisibilityPlot(FigureCanvas):
    """Custom matplotlib canvas for PySide6.

    The three charts share one time axis: the wheel zooms it around the mouse,
    dragging pans it, and double-click goes back to the view mode's default
    (the night, or the full calculated range)."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 8))
        super().__init__(self.figure)
        self.setParent(parent)
        self._last_results = None  # redrawn with new colors when the theme changes

        # Time axis view: 'night' (sunset to sunrise) or 'full' (the whole range)
        self.view_mode = 'night'
        self._data_xlim = None     # the whole calculated range, hours from start
        self._night_xlim = None    # sunset to sunrise (padded), None without a night
        self._view_xlim = None     # what's shown now, kept across theme redraws
        self._pan = None           # (mouse x in pixels, xlim) while dragging

        # Hover: a cursor line and a dot on each curve, drawn ("blitted") over a
        # saved copy of the charts so following the mouse doesn't redraw them all
        self.hover_data = None
        self._cursor_artists = []  # [(line, dot)] per chart, animated
        self._background = None    # the charts without the cursor, saved after each full draw
        self._hover_x = None       # hours from start under the mouse, None when not hovering
        self._tooltip_text = None

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
        self.qt_tooltip.hide()
        self._apply_widget_theme()
        theme_manager().theme_changed.connect(self._on_theme_changed)

        # Connect mouse motion event
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        # Connect mouse leave event
        self.mpl_connect('axes_leave_event', self.on_mouse_leave)
        # Zoom and pan the time axis
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_connect('button_release_event', self.on_button_release)
        self.mpl_connect('draw_event', self._on_draw)

    def _apply_widget_theme(self):
        """Theme colors for the canvas and its Qt tooltip (rebuilt on theme changes)"""
        themed_style(self, lambda: f"background-color: {chart_background()};")
        themed_style(self.qt_tooltip, lambda: f"""
            QLabel {{
                background-color: {COLORS['background_lighter']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 3px;
                padding: 6px 10px;
                font-size: {font_size(9)};
                font-family: monospace;
            }}
        """)

    def _on_theme_changed(self):
        """Redraw the chart with the new theme colors (the widget styles
        follow the theme on their own)"""
        if self._last_results is not None:
            self.plot_visibility(self._last_results, keep_view=True)

    def _add_darkness_shading(self, ax, hours_from_start, sun_altitudes):
        """Shade the background by how dark the sky is (NightTimes.SKY_STATES):
        civil, nautical and astronomical twilight, then night, darker each;
        daylight isn't shaded."""
        states = np.array([sky_state(alt)[0] for alt in sun_altitudes])
        for key, _name, _above in SKY_STATES[1:]:
            in_range = states == key
            if not np.any(in_range):
                continue
            color, alpha = sky_shade(key)

            # Find contiguous regions
            diff = np.diff(np.concatenate(([False], in_range, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # Shade each contiguous region
            for start_idx, end_idx in zip(starts, ends):
                x_start = hours_from_start[start_idx]
                x_end = hours_from_start[min(end_idx, len(hours_from_start) - 1)]
                ax.axvspan(x_start, x_end, facecolor=color, alpha=alpha, zorder=0)

    @staticmethod
    def _find_night(hours_from_start, sun_altitudes, now_hours):
        """(start, end) hours of sunset to sunrise, padded a little and kept inside
        the range: tonight (or the night in progress) when now is in the range,
        otherwise the night the range is centered on. None when the sun never
        sets or never rises in the range."""
        hours = np.asarray(hours_from_start)
        down = sun_altitudes < SUNSET_ALT
        if not down.any() or down.all():
            return None
        diff = np.diff(np.concatenate(([False], down, [False])).astype(int))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
        upcoming = [i for i, e in enumerate(ends) if hours[e] >= now_hours]
        if hours[0] <= now_hours <= hours[-1] and upcoming:
            best = upcoming[0]
        else:
            # The calculation is centered on the night's midnight; take the
            # dark stretch nearest to it
            middle = hours[-1] / 2
            distance = [0 if hours[s] <= middle <= hours[e] else min(abs(hours[s] - middle), abs(hours[e] - middle))
                        for s, e in zip(starts, ends)]
            best = int(np.argmin(distance))
        return (max(hours[0], hours[starts[best]] - NIGHT_PADDING_HOURS),
                min(hours[-1], hours[ends[best]] + NIGHT_PADDING_HOURS))

    def resizeEvent(self, event):
        """Re-fit the margins to the new size so the legends beside the charts
        don't get cut off"""
        super().resizeEvent(event)
        if self.hover_data and self.width() > 200 and self.height() > 200:
            try:
                self.figure.tight_layout()
            except Exception:
                pass  # Too small to fit; the next resize tries again

    @staticmethod
    def _legend(ax):
        """Legend just right of the chart, so it never covers the curves however
        the time axis is zoomed"""
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize='small')

    def _default_xlim(self):
        """The view mode's time range"""
        if self.view_mode == 'night' and self._night_xlim is not None:
            return self._night_xlim
        return self._data_xlim

    def _set_xlim(self, lo, hi):
        """Show lo..hi hours on all three charts, kept inside the calculated range
        and no narrower than MIN_ZOOM_HOURS"""
        data_lo, data_hi = self._data_xlim
        span = min(max(hi - lo, MIN_ZOOM_HOURS), data_hi - data_lo)
        if span > hi - lo:  # Widened: keep it centered where it was
            lo -= (span - (hi - lo)) / 2
        lo = min(max(lo, data_lo), data_hi - span)
        self._view_xlim = (lo, lo + span)
        self.hover_data['axes'][0].set_xlim(*self._view_xlim)  # The others share it
        self._background = None  # Saved for the old range; the next draw saves it again
        self._update_title()

    def _update_title(self):
        """Title with the dates on screen ("Wed Sep 24 – Thu Sep 25, 2026"), so
        it's always clear which night the graph shows"""
        start_utc = self.hover_data['start_local'].astimezone(pytz.UTC)
        local_tz = self.hover_data['local_tz']
        first, last = ((start_utc + timedelta(hours=x)).astimezone(local_tz) for x in self._view_xlim)

        def day(moment, year=False):
            return f"{moment:%a %b} {moment.day}" + (f", {moment.year}" if year else "")
        if first.date() == last.date():
            dates = day(first, year=True)
        else:
            dates = f"{day(first, year=first.year != last.year)} – {day(last, year=True)}"
        self.hover_data['axes'][0].set_title(f"{self._title_name} · {dates} ({first:%Z})",
                                             fontsize=chart_font_size(14))

    def set_view_mode(self, mode):
        """'night' or 'full' - shows that range now"""
        self.view_mode = mode
        self.reset_view()

    def reset_view(self):
        """Back to the view mode's time range"""
        if not self.hover_data or self._data_xlim is None:
            return
        self._set_xlim(*self._default_xlim())
        self._hide_cursor()
        self.draw_idle()

    def on_scroll(self, event):
        """Wheel: zoom the time axis around the mouse"""
        if not self.hover_data or self._data_xlim is None or event.inaxes is None or event.xdata is None:
            return
        factor = ZOOM_STEP if event.button == 'up' else 1 / ZOOM_STEP
        lo, hi = event.inaxes.get_xlim()
        x = event.xdata
        self._set_xlim(x - (x - lo) * factor, x + (hi - x) * factor)
        # The full redraw puts the cursor back where the mouse is
        self._hover_x = self._clamp_hours(x)
        self._update_tooltip(event, self._hover_x)
        self.draw_idle()

    def on_button_press(self, event):
        """Double-click resets the view; a left-button press starts a pan"""
        if not self.hover_data or self._data_xlim is None or event.inaxes is None:
            return
        if event.dblclick:
            self._pan = None
            self.reset_view()
        elif event.button == 1:
            self._pan = (event.x, event.inaxes.get_xlim())
            self._hide_cursor()
            self.setCursor(Qt.ClosedHandCursor)

    def on_button_release(self, event):
        """End a pan"""
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
            self.on_mouse_move(event)  # The cursor and tooltip back where the mouse is

    def _drag_pan(self, event):
        """Move the time axis with the mouse while the left button is held"""
        start_x, (lo, hi) = self._pan
        width = self.hover_data['axes'][0].bbox.width
        shift = (event.x - start_x) / width * (hi - lo) if width else 0
        self._set_xlim(lo - shift, hi - shift)
        self.draw_idle()

    def plot_visibility(self, results, keep_view=False):
        """Create visibility plot with altitude and azimuth.

        keep_view: keep the zoomed/panned time range (a theme redraw) instead of
        going back to the view mode's default (a new calculation)."""
        self._last_results = results
        previous_view = self._view_xlim if keep_view else None
        self.figure.clear()
        self.figure.set_facecolor(chart_background())
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
            'local_tz': local_tz,
            'horizon_alt': None,  # filled in below when there's a custom horizon
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
        ax1 = self.figure.add_subplot(3, 1, 1)

        # Add darkness shading first (so it appears behind the data)
        self._add_darkness_shading(ax1, hours_from_start, sun_altitudes)

        ax1.plot(hours_from_start, dso_altaz.alt.deg, color=chart_color('#00aaff'), linewidth=2, label=f'{dso_name} Altitude')
        min_altitude = results.get('min_altitude', 30)
        ax1.axhline(y=min_altitude, color=chart_color('#00ff88'), linestyle='--', alpha=0.8,
                    label=f'Min Altitude ({min_altitude:g}°)')
        ax1.axhline(y=0, color=chart_color('#888888'), linestyle='-', alpha=0.6, label='Horizon')

        # Custom horizon along the object's path: the obstruction height at the
        # object's azimuth at each moment, so the object is blocked wherever its
        # altitude line dips into the shaded area
        horizon = results.get('horizon')
        horizon_alt = None
        if horizon is not None:
            horizon_alt = horizon.altitude_at(dso_altaz.az.deg)
            horizon_color = chart_color('#8a6a45')
            ax1.fill_between(hours_from_start, -20, horizon_alt, color=horizon_color, alpha=0.35,
                             linewidth=0, zorder=1, label=f'Custom Horizon ({horizon.name or "custom"})')
            ax1.plot(hours_from_start, horizon_alt, color=horizon_color, linewidth=1.5, zorder=1)

        # Highlight optimal viewing times
        optimal_alt = np.where(optimal_times, dso_altaz.alt.deg, np.nan)
        ax1.plot(hours_from_start, optimal_alt, color=chart_color('#ff4444'), linewidth=4, alpha=0.8, label='Optimal Viewing')

        # Meridian crossings while the object is up - where a German equatorial
        # mount flips. Drawn on all three charts once they exist.
        meridian_hours = []
        for crossing in results.get('meridian_times', []):
            local = crossing.datetime.replace(tzinfo=pytz.UTC).astimezone(local_tz)
            x = (local - start_local).total_seconds() / 3600
            if np.interp(x, hours_from_start, dso_altaz.alt.deg) > 0:
                meridian_hours.append((x, local))
        for i, (x, _local) in enumerate(meridian_hours):
            ax1.axvline(x=x, color=chart_color(MERIDIAN_COLOR), linestyle='-.', linewidth=1.8, alpha=0.9,
                        zorder=90, label='Meridian Flip' if i == 0 else '_nolegend_')

        ax1.set_ylabel('Altitude (°)')
        # Get timezone abbreviation for display
        sample_time = local_times[0] if local_times else None
        tz_abbrev = sample_time.strftime('%Z') if sample_time else 'Local Time'
        self._title_name = f'{dso_name} Visibility'  # _update_title adds the dates shown
        ax1.set_title(f'{dso_name} Visibility ({tz_abbrev})', fontsize=chart_font_size(14))
        self._legend(ax1)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-20, 90)

        # Time labels on round clock times, re-spaced as the axis zooms. The
        # three charts share the axis, so this covers them all.
        def time_label(x, _pos):
            moment = (start_local.astimezone(pytz.UTC) + timedelta(hours=x)).astimezone(local_tz)
            return format_time(moment)
        ax1.xaxis.set_major_locator(ClockTimeLocator(start_local))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(time_label))

        # DSO azimuth subplot
        ax2 = self.figure.add_subplot(3, 1, 2, sharex=ax1)

        # Add darkness shading
        self._add_darkness_shading(ax2, hours_from_start, sun_altitudes)

        ax2.plot(hours_from_start, dso_altaz.az.deg, color=chart_color('#ff8800'), linewidth=2, label=f'{dso_name} Azimuth')

        # Add cardinal direction lines (the axis labels name them, so they stay
        # out of the legend)
        ax2.axhline(y=0, color=chart_color('#ff4444'), linestyle=':', alpha=0.7)
        ax2.axhline(y=90, color=chart_color('#44ff44'), linestyle=':', alpha=0.7)
        ax2.axhline(y=180, color=chart_color('#ffff44'), linestyle=':', alpha=0.7)
        ax2.axhline(y=270, color=chart_color('#4444ff'), linestyle=':', alpha=0.7)

        # Highlight optimal viewing times for azimuth too
        optimal_az = np.where(optimal_times, dso_altaz.az.deg, np.nan)
        ax2.plot(hours_from_start, optimal_az, color=chart_color('#ff4444'), linewidth=4, alpha=0.8, label='Optimal Viewing')

        ax2.set_ylabel('Azimuth (°)')
        self._legend(ax2)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 360)
        ax2.set_yticks([0, 90, 180, 270, 360])
        ax2.set_yticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
        # Sharing the axis hides the upper charts' time labels; keep them
        ax2.tick_params(axis='x', labelbottom=True)

        # Sun altitude subplot
        ax3 = self.figure.add_subplot(3, 1, 3, sharex=ax1)

        # Add darkness shading
        self._add_darkness_shading(ax3, hours_from_start, sun_altitudes)

        ax3.plot(hours_from_start, sun_altaz.alt.deg, color=chart_color('#ffaa00'), linewidth=2, label='Sun Altitude')
        ax3.axhline(y=0, color=chart_color('#888888'), linestyle='-', alpha=0.6, label='Horizon')
        ax3.axhline(y=DARK_SKY_ALTITUDE, color=chart_color('#4488ff'), linestyle='--', alpha=0.8, label='Astronomical Twilight')
        ax3.axhline(y=ASTRONOMICAL_DARK_ALTITUDE, color=chart_color('#aa44ff'), linestyle='--', alpha=0.8, label='Night')

        # Use the same timezone abbreviation as in title
        sample_time = local_times[0] if local_times else None
        tz_abbrev = sample_time.strftime('%Z') if sample_time else 'Local Time'
        ax3.set_xlabel(f'Time ({tz_abbrev})')
        ax3.set_ylabel('Sun Alt. (°)')
        self._legend(ax3)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-25, 50)

        for ax in (ax2, ax3):
            for x, _local in meridian_hours:
                ax.axvline(x=x, color=chart_color(MERIDIAN_COLOR), linestyle='-.', linewidth=1.8, alpha=0.9,
                           zorder=90)

        # Store additional data needed for hover and current time updates
        self.hover_data['horizon_alt'] = horizon_alt
        self.hover_data['local_times'] = local_times
        self.hover_data['hours_from_start'] = hours_from_start
        self.hover_data['axes'] = [ax1, ax2, ax3]
        self.hover_data['start_local'] = start_local
        self.hover_data['meridian'] = [local for _x, local in meridian_hours]
        # As arrays for interpolating between the points under the mouse
        self.hover_data['hours'] = np.asarray(hours_from_start)
        self.hover_data['dso_alt'] = np.asarray(dso_altaz.alt.deg)
        self.hover_data['dso_az'] = np.unwrap(np.asarray(dso_altaz.az.deg), period=360)  # No jump at north
        self.hover_data['sun_alt'] = np.asarray(sun_altitudes)

        # The hover cursor: animated, so full draws leave it out and it's drawn
        # on its own over the saved background
        cursor_color = chart_color('#ffcc00')
        self._cursor_artists = []
        for ax in (ax1, ax2, ax3):
            line = ax.axvline(x=hours_from_start[0], color=cursor_color, alpha=0.8, linewidth=1.5,
                              animated=True, label='_nolegend_')
            dot, = ax.plot([], [], 'o', color=cursor_color, markersize=7, markeredgecolor=chart_background(),
                           markeredgewidth=1.2, animated=True, zorder=102, label='_nolegend_')
            self._cursor_artists.append((line, dot))
        self._background = None

        # Time range: the night (or everything), or where it was zoomed to
        # before a theme redraw
        self._data_xlim = (hours_from_start[0], hours_from_start[-1])
        now_hours = (datetime.now(pytz.UTC) - start_local).total_seconds() / 3600
        self._night_xlim = self._find_night(hours_from_start, sun_altitudes, now_hours)
        self._set_xlim(*(previous_view or self._default_xlim()))

        self.figure.tight_layout()

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
                line = ax.axvline(x=current_hours_from_start, color=chart_color(NOW_COLOR), linestyle='--',
                          linewidth=2.5, alpha=0.9, label='Current Time', zorder=100)
                self.current_time_lines.append(line)

            # A dot where each curve is right now
            if axes and 0 <= current_hours_from_start <= hours_from_start[-1]:
                x = current_hours_from_start
                azimuths = np.unwrap(self.hover_data['dso_altaz'].az.deg, period=360)
                now_values = (np.interp(x, hours_from_start, self.hover_data['dso_altaz'].alt.deg),
                              np.interp(x, hours_from_start, azimuths) % 360,
                              np.interp(x, hours_from_start, self.hover_data['sun_altaz'].alt.deg))
                for ax, y in zip(axes, now_values):
                    dot, = ax.plot([x], [y], 'o', color=chart_color(NOW_COLOR), markersize=9,
                                   markeredgecolor=chart_background(), markeredgewidth=1.5,
                                   zorder=101, label='_nolegend_')
                    self.current_time_lines.append(dot)

            # Update legend on first subplot to include current time (only if lines were drawn)
            if axes and self.current_time_lines:
                self._legend(axes[0])

            # Redraw the canvas
            self.draw_idle()

    def _clamp_hours(self, x):
        """x (hours from start) kept inside the calculated range"""
        hours = self.hover_data['hours']
        return min(max(x, hours[0]), hours[-1])

    def _on_draw(self, _event):
        """After every full draw: save the charts (without the cursor) and put the
        cursor back on them"""
        if not self._cursor_artists or not self.hover_data:
            return
        self._background = self.copy_from_bbox(self.figure.bbox)
        if self._hover_x is not None:
            self._draw_cursor()

    def _draw_cursor(self):
        """Draw the cursor line and the dots at self._hover_x"""
        x = self._hover_x
        hours = self.hover_data['hours']
        values = (np.interp(x, hours, self.hover_data['dso_alt']),
                  np.interp(x, hours, self.hover_data['dso_az']) % 360,
                  np.interp(x, hours, self.hover_data['sun_alt']))
        for ax, (line, dot), y in zip(self.hover_data['axes'], self._cursor_artists, values):
            line.set_xdata([x, x])
            dot.set_data([x], [y])
            ax.draw_artist(line)
            ax.draw_artist(dot)

    def _show_cursor(self, x):
        """Move the cursor to x: the saved charts, the cursor over them, and only
        that copied to the screen"""
        self._hover_x = x
        if self._background is None:
            self.draw_idle()  # Saves the background and draws the cursor
            return
        self.restore_region(self._background)
        self._draw_cursor()
        self.blit(self.figure.bbox)

    def _hide_cursor(self):
        """Take the cursor and tooltip away"""
        self.qt_tooltip.hide()
        self._tooltip_text = None
        if self._hover_x is None:
            return
        self._hover_x = None
        if self._background is not None:
            self.restore_region(self._background)
            self.blit(self.figure.bbox)

    def _hover_text(self, x):
        """Tooltip text for x hours from start, the values interpolated to the minute"""
        data = self.hover_data
        hours = data['hours']
        start_utc = data['start_local'].astimezone(pytz.UTC)
        moment = (start_utc + timedelta(hours=x)).astimezone(data['local_tz'])
        dso_alt = float(np.interp(x, hours, data['dso_alt']))
        dso_az = float(np.interp(x, hours, data['dso_az'])) % 360
        sun_alt = float(np.interp(x, hours, data['sun_alt']))
        nearest = int(np.argmin(np.abs(hours - x)))
        optimal = data['optimal_times'][nearest]
        dso_name = data['dso_name']

        horizon_line = ""
        horizon_alt = data.get('horizon_alt')
        if horizon_alt is not None:
            obstruction = float(np.interp(x, hours, horizon_alt))
            blocked = " - blocked" if dso_alt <= obstruction else ""
            horizon_line = f"Custom Horizon: {obstruction:.1f}°{blocked}\n"
        meridian_line = ""
        meridian = data.get('meridian')
        if meridian:
            # The nearest crossing, and how far it is from the hovered time
            crossing = min(meridian, key=lambda m: abs((m - moment).total_seconds()))
            minutes = round((crossing - moment).total_seconds() / 60)
            gap = format_duration(abs(minutes) * 60)
            when = "now" if minutes == 0 else (f"in {gap}" if minutes > 0 else f"{gap} ago")
            meridian_line = f"Meridian Flip: {format_time(crossing)} ({when})\n"
        return (
            f"Time: {format_time(moment)} {moment.strftime('%Z')}\n"
            f"{dso_name} Alt: {dso_alt:.1f}°\n"
            f"{dso_name} Az: {dso_az:.0f}° ({compass_direction(dso_az)})\n"
            f"{horizon_line}"
            f"{meridian_line}"
            f"Sun Alt: {sun_alt:.1f}° ({sky_state(sun_alt)[1]})\n"
            f"Optimal: {'Yes' if optimal else 'No'}"
        )

    def _update_tooltip(self, event, x):
        """The tooltip's text for x (rebuilt only when the minute changes) beside
        the mouse - left of it near the right edge, below it near the top"""
        from PySide6.QtCore import QPoint
        text = self._hover_text(x)
        if text != self._tooltip_text:
            self._tooltip_text = text
            self.qt_tooltip.setText(text)
            self.qt_tooltip.adjustSize()
        x_min, x_max = event.inaxes.get_xlim()
        y_min, y_max = event.inaxes.get_ylim()
        x_offset = -self.qt_tooltip.width() - 15 if x > x_min + 0.75 * (x_max - x_min) else 15
        y_offset = 15 if event.ydata > y_min + 0.6 * (y_max - y_min) else -self.qt_tooltip.height() - 15
        # Matplotlib counts pixels from the bottom in physical pixels; Qt from the top in logical ones
        ratio = self.devicePixelRatioF() or 1
        cursor_pos = self.mapToGlobal(QPoint(int(event.x / ratio), int(self.height() - event.y / ratio)))
        self.qt_tooltip.move(cursor_pos + QPoint(x_offset, y_offset))
        if not self.qt_tooltip.isVisible():
            self.qt_tooltip.show()
            self.qt_tooltip.raise_()

    def on_mouse_move(self, event):
        """Hover: the cursor follows the mouse along the time axis, with the
        values under it in the tooltip"""
        if self._pan is not None and self.hover_data:
            self._drag_pan(event)
            return
        if not self.hover_data or event.inaxes is None or event.xdata is None:
            self._hide_cursor()
            return
        x = self._clamp_hours(event.xdata)
        if x != self._hover_x:
            self._show_cursor(x)
        self._update_tooltip(event, x)

    def on_mouse_leave(self, event):
        """Handle mouse leaving a chart"""
        self._hide_cursor()

    def leaveEvent(self, event):
        """Hide the cursor and tooltip when the mouse leaves the canvas entirely"""
        super().leaveEvent(event)
        self._hide_cursor()

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
        release_on_close(self)  # Freed once closed (see WindowLifetime)
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

        # Graph, with its time range picker above it
        settings = app_settings()
        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Show:"))
        self.view_combo = QComboBox()
        self.view_combo.addItem("Night", 'night')
        self.view_combo.addItem("Full range", 'full')
        self.view_combo.setToolTip("Night: sunset to sunrise of the selected date.\n"
                                   "Full range: all the hours calculated.")
        view_row.addWidget(self.view_combo)
        view_row.addStretch()
        zoom_hint = QLabel("Scroll to zoom · Drag to pan · Double-click to reset")
        themed_style(zoom_hint, lambda: f"color: {COLORS['text_secondary']};")
        view_row.addWidget(zoom_hint)
        graph_layout.addLayout(view_row)

        self.plot_widget = VisibilityPlot()
        graph_layout.addWidget(self.plot_widget, stretch=1)

        view_mode = settings.value(GRAPH_VIEW_SETTING, 'night', type=str)
        self.view_combo.setCurrentIndex(max(0, self.view_combo.findData(view_mode)))
        self.plot_widget.view_mode = self.view_combo.currentData()
        self.view_combo.currentIndexChanged.connect(self._on_view_mode_changed)

        # Drag the divider to trade calendar room for graph room
        self.graph_splitter = QSplitter(Qt.Vertical)
        self.graph_splitter.addWidget(calendar_group)
        self.graph_splitter.addWidget(graph_panel)
        self.graph_splitter.setCollapsible(1, False)
        self.graph_splitter.setStretchFactor(1, 1)
        if not self.graph_splitter.restoreState(settings.value(GRAPH_SPLITTER_SETTING, b"")):
            self.graph_splitter.setSizes([320, 600])
        # Saved as it moves: the main window's wrapper borrows this layout
        # without ever calling this window's closeEvent
        self.graph_splitter.splitterMoved.connect(
            lambda *_: app_settings().setValue(
                GRAPH_SPLITTER_SETTING, self.graph_splitter.saveState()))
        right_layout.addWidget(self.graph_splitter, stretch=1)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    def _on_view_mode_changed(self):
        """Night / Full range picked: show it and remember it"""
        mode = self.view_combo.currentData()
        app_settings().setValue(GRAPH_VIEW_SETTING, mode)
        self.plot_widget.set_view_mode(mode)

    def _load_location_from_database(self):
        """Load observer location from the database"""
        # Check if observer location should be shown
        settings = app_settings()
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
                site = active_site(conn)
                self.user_timezone = site.timezone if site else None
                if site is not None:
                    self.location_name_label.setText(site.name)
                    self.location_coords_label.setText(coordinates_text(site.lat, site.lon))
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
        """Window title with the object's name and common name ("M 31 (Andromeda Galaxy)")"""
        from cosmoscollection.catalog.DSONames import display_name
        self.setWindowTitle(f"DSO Visibility Calculator - {display_name(dso_name)} - Cosmos Collection")

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
        text += f"RA: {format_ra(dso_coord.ra.deg)}\n"
        text += f"Dec: {format_dec(dso_coord.dec.deg)}\n\n"

        # Find maximum altitude and its direction
        max_alt_idx = np.argmax(dso_altaz.alt.deg)
        max_alt_time_utc = time_range[max_alt_idx].datetime.replace(tzinfo=pytz.UTC)
        max_alt_time_local = max_alt_time_utc.astimezone(local_tz)
        max_altitude = dso_altaz.alt.deg[max_alt_idx]
        max_azimuth = dso_altaz.az.deg[max_alt_idx]

        # Convert azimuth to cardinal direction using centralized method
        max_direction = compass_direction(max_azimuth)

        # Determine if we're in EST or EDT
        tz_name = max_alt_time_local.strftime('%Z')

        text += f"Maximum altitude: {max_altitude:.1f}° at {format_time(max_alt_time_local)} {tz_name}\n"
        text += f"Direction at max altitude: {max_direction} ({max_azimuth:.0f}°)\n"
        # Meridian crossings while it's up - when a German equatorial mount flips
        crossings = results.get('meridian_times', [])
        for crossing in crossings:
            crossing_local = crossing.datetime.replace(tzinfo=pytz.UTC).astimezone(local_tz)
            index = int(np.argmin(np.abs((time_range - crossing).to_value('hour'))))
            if dso_altaz.alt.deg[index] > 0:
                day = crossing_local.strftime('%b %d ') if len(crossings) > 1 else ""
                text += f"Meridian flip: {day}{format_time(crossing_local)} {crossing_local.strftime('%Z')}\n"
        text += "\n"

        horizon = results.get('horizon')
        if horizon is not None:
            text += f"Custom horizon: {horizon.name or 'custom'}\n"
            text += "Times the object is behind it are excluded from viewing windows.\n\n"

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
                mid_direction = compass_direction(mid_az)

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
        self.calendar_widget.show_date(self.date_input.date())
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
        # 0/0 is how the app stores "no position" (a target added without
        # coordinates): look the object up by name instead of plotting that spot
        if not ra_deg and not dec_deg:
            ra_deg = dec_deg = None
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
