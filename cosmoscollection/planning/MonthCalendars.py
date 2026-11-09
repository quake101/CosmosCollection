#!/usr/bin/env python3
"""
Month calendars that color each day by how long an object is up, with a
weather stripe across the top of the forecast days. The DSO Visibility
Calculator and the Session Manager's Calendar tab are built on these:

- LayeredMonthCalendar: one month. Date-keyed layers are drawn on each day
  (visibility hours as the background, the weather score as a stripe) and
  listed in a hover tooltip, with the moon's illumination when it was
  calculated. Subclasses add layers through LAYERS, paint_extras() and
  tooltip_lines() (the Session Manager's session dots).
- MultiMonthCalendar: 1-3 chained months with Months / Previous / Next, the
  weather layer (one shared fetch per location, see SharedWeatherFetch) and the
  visibility layer (MonthlyVisibilityThread). Neither ever waits on a thread, so
  changing months or objects mid-calculation doesn't freeze the window.
  Subclasses say which object and location to show.
"""

import calendar
import logging
from datetime import date as date_cls, timedelta

from PySide6.QtCore import QDate, QEvent, QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QCalendarWidget, QComboBox, QHBoxLayout, QLabel, QPushButton, QTableView,
                               QVBoxLayout, QWidget)

from cosmoscollection.core.Settings import app_settings
from cosmoscollection.ui.Theme import COLORS, adapt_color, contrast_text, font_size, themed_style, tint
from cosmoscollection.ui.WindowLifetime import keep_alive_until_finished, retire_thread

logger = logging.getLogger(__name__)

MAX_MONTHS = 3
WEATHER_STRIPE_HEIGHT = 6


def visibility_hours_to_color(hours):
    """A day's visibility hours as a (background, foreground) QColor pair, on a
    7-tier green-to-red scale."""
    # Tier colors are adapted to the current theme (see Theme.tint); rank keeps
    # the tiers in order in Night Vision, where the green-to-red hue is lost.
    if hours >= 8:
        base, rank = '#009600', 6
    elif hours >= 6:
        base, rank = '#007800', 5
    elif hours >= 4:
        base, rank = '#647800', 4
    elif hours >= 2:
        base, rank = '#966400', 3
    elif hours >= 1:
        base, rank = '#963c00', 2
    elif hours > 0:
        base, rank = '#782800', 1
    else:
        base, rank = '#500000', 0
    background = tint(base, 0.8, rank=rank / 6)
    return QColor(background), QColor(contrast_text(background))


def no_visibility_data_colors():
    """(background, foreground) for calendar days without visibility data."""
    background = COLORS['background_lighter']
    return QColor(background), QColor(adapt_color(COLORS['text_secondary'], background=background))


def hours_legend_row():
    """A row of labels showing the visibility hours scale."""
    row = QHBoxLayout()
    for text, hours in (("8+ hrs", 8), ("6-8 hrs", 6), ("4-6 hrs", 4), ("2-4 hrs", 2),
                        ("1-2 hrs", 1), ("<1 hr", 0.5), ("None", 0)):
        background, foreground = visibility_hours_to_color(hours)
        label = QLabel(text)
        label.setStyleSheet(f"background-color: {background.name()}; color: {foreground.name()}; "
                            "padding: 3px; border-radius: 2px;")
        row.addWidget(label)
    row.addStretch()
    return row


class MonthlyVisibilityThread(QThread):
    """Visibility hours (and optionally the moon's illumination) for every day
    from start_date to end_date. location is (lat, lon, timezone) for a
    session's own site, with that saved location's custom horizon; None uses the
    active location and its horizon. Stops early when interrupted."""
    progress = Signal(int, int)  # days done, total days
    # object, not dicts: Signal(dict) goes through QVariantMap across threads,
    # which drops non-string keys (these are datetime.date)
    result_ready = Signal(object)  # (hours_by_date, moon_illumination_by_date)
    error = Signal(str)

    def __init__(self, start_date, end_date, min_altitude=30, ra_deg=None, dec_deg=None, dso_name=None,
                 dso_coord=None, location=None, with_moon=False):
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.min_altitude = min_altitude
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.dso_name = dso_name
        self.dso_coord = dso_coord
        self.location = location
        self.with_moon = with_moon
        # Made here, on the UI thread: without a location it reads the active
        # one through DatabaseManager's connection, which belongs to this thread
        from cosmoscollection.planning.Visibility import DSOVisibilityCalculator
        if location is None:
            self.calculator = DSOVisibilityCalculator()
        else:
            lat, lon, timezone = location
            self.calculator = DSOVisibilityCalculator(location_lat=lat, location_lon=lon, timezone=timezone)

    def run(self):
        try:
            from astropy import units as u
            from astropy.coordinates import SkyCoord
            from astropy.time import Time
            from cosmoscollection.planning.Visibility import DSOVisibilityCalculator

            calc = self.calculator
            if self.location is None:
                horizon = calc.horizon
            else:
                from cosmoscollection.planning.HorizonProfile import load_horizon_for_coordinates
                horizon = load_horizon_for_coordinates(*self.location[:2])  # Opens its own connection
            if calc.location is None:
                self.error.emit("Observer location not configured.")
                return

            coord = self.dso_coord
            if coord is None:
                if self.ra_deg is not None and self.dec_deg is not None:
                    coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
                elif self.dso_name:
                    coord, error = calc.get_dso_coordinates_enhanced(self.dso_name)
                    if coord is None:
                        self.error.emit(f"Could not find coordinates: {error}")
                        return
                else:
                    self.error.emit("No coordinates or name available.")
                    return

            hours_by_date, moon_by_date = {}, {}
            total = (self.end_date - self.start_date).days + 1
            day = self.start_date
            for done in range(1, total + 1):
                if self.isInterruptionRequested():
                    return
                date_text = day.strftime("%Y-%m-%d")
                hours_by_date[day] = calc.calculate_visibility_hours_for_day(
                    coord, date_text, self.min_altitude, horizon=horizon)
                if self.with_moon:
                    moon_by_date[day] = DSOVisibilityCalculator.get_moon_illumination(Time(f"{date_text}T12:00:00"))
                self.progress.emit(done, total)
                day += timedelta(days=1)

            self.result_ready.emit((hours_by_date, moon_by_date))
        except Exception as e:
            logger.error(f"Error calculating monthly visibility: {e}")
            self.error.emit(f"Calculation error: {e}")


class LayeredMonthCalendar(QCalendarWidget):
    """One month with date-keyed layers: visibility hours (background fill),
    the weather astro score (a stripe across the top, only on the ~7 forecast
    days) and the moon's illumination (tooltip only)."""

    LAYERS = ("visibility_hours", "moon_illumination", "weather_scores", "weather_details")
    # Calculated together for one object, so cleared together when it changes
    VISIBILITY_LAYERS = ("visibility_hours", "moon_illumination")

    def __init__(self, parent=None):
        super().__init__(parent)
        # date -> hours / fraction 0-1 / astro score 0-100 / {"cloud_cover", "seeing"}
        for name in self.LAYERS:
            setattr(self, name, {})
        self.setGridVisible(True)
        self.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self.setMouseTracking(True)

        self.tooltip_label = QLabel(self)
        self.tooltip_label.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        themed_style(self.tooltip_label, lambda: f"""
            QLabel {{
                background-color: {COLORS['background_lighter']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: {font_size(10)};
            }}
        """)
        self.tooltip_label.hide()

        # The day grid is a private QTableView inside QCalendarWidget, which
        # takes the mouse events over the days - mouseMoveEvent on the calendar
        # itself never fires there. So watch that view's viewport instead.
        self._day_view = self.findChild(QTableView)
        if self._day_view is not None:
            self._day_view.setMouseTracking(True)
            self._day_view.viewport().setMouseTracking(True)
            self._day_view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if self._day_view is not None and obj is self._day_view.viewport():
            if event.type() == QEvent.Type.MouseMove:
                self._show_tooltip_for_date(self._date_from_view_pos(event.pos()),
                                            self._day_view.viewport().mapToGlobal(event.pos()))
            elif event.type() == QEvent.Type.Leave:
                self.tooltip_label.hide()
        return super().eventFilter(obj, event)

    def _date_from_view_pos(self, pos_in_view):
        """The date under a point of the day grid. QCalendarWidget has no public
        dateAt(), so it's worked out from the cell: row 0 is the weekday names,
        and row 1 starts on firstDayOfWeek() before the 1st of the month."""
        index = self._day_view.indexAt(pos_in_view)
        if not index.isValid() or index.row() < 1:
            return None
        first_of_month = date_cls(self.yearShown(), self.monthShown(), 1)
        qt_first_day = self.firstDayOfWeek().value  # Qt.Monday=1 .. Qt.Sunday=7
        first_weekday = first_of_month.isoweekday()  # Monday=1 .. Sunday=7
        # Qt always shows at least one week of the previous month, even when the
        # 1st falls on the first day of the week, so this is 1-7, never 0
        leading_days = ((first_weekday - qt_first_day - 1) % 7) + 1
        first_visible_date = first_of_month - timedelta(days=leading_days)
        return first_visible_date + timedelta(days=(index.row() - 1) * 7 + index.column())

    def set_data(self, **layers):
        """Replace the named layers (see LAYERS) and repaint. None leaves a layer as it is."""
        for name, values in layers.items():
            if name not in self.LAYERS:
                raise TypeError(f"{type(self).__name__} has no layer {name!r}")
            if values is not None:
                setattr(self, name, values)
        self.updateCells()
        self.update()

    def clear_visibility(self):
        self.tooltip_label.hide()
        self.set_data(**{name: {} for name in self.VISIBILITY_LAYERS})

    def paintCell(self, painter, rect, date):
        from cosmoscollection.planning.WeatherForecast import get_rating_color

        if date.month() != self.monthShown() or date.year() != self.yearShown():
            super().paintCell(painter, rect, date)  # Days of the next/previous month, grayed
            return

        py_date = date.toPython()
        if py_date in self.visibility_hours:
            background, foreground = visibility_hours_to_color(self.visibility_hours[py_date])
        else:
            background, foreground = no_visibility_data_colors()
        painter.fillRect(rect, QBrush(background))

        weather_score = self.weather_scores.get(py_date)
        if weather_score is not None:
            stripe = rect.adjusted(0, 0, 0, -(rect.height() - WEATHER_STRIPE_HEIGHT))
            painter.fillRect(stripe, QBrush(QColor(get_rating_color(weather_score))))

        painter.setPen(QPen(foreground))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(date.day()))

        self.paint_extras(painter, rect, py_date)

        if date == self.selectedDate():
            painter.setPen(QPen(QColor(COLORS['warning']), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        elif py_date == date_cls.today():
            painter.setPen(QPen(QColor(COLORS['text']), 1))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

    def paint_extras(self, painter, rect, py_date):
        """Draw a subclass's own layers on a day of this month."""

    def tooltip_lines(self, py_date):
        """The hover tooltip's lines for a day of this month."""
        from cosmoscollection.planning.WeatherFetch import get_rating_label

        lines = []
        if py_date in self.visibility_hours:
            lines.append(f"{self.visibility_hours[py_date]:.1f}h visible")
        if py_date in self.moon_illumination:
            lines.append(f"Moon: {self.moon_illumination[py_date] * 100:.0f}% illuminated")
        if py_date in self.weather_scores:
            score = self.weather_scores[py_date]
            line = f"Weather: {score}/100 ({get_rating_label(score)})"
            detail = self.weather_details.get(py_date) or {}
            if detail.get("cloud_cover") is not None:
                line += f", {detail['cloud_cover']:.0f}% clouds"
            if detail.get("seeing"):
                line += f", seeing {detail['seeing']}"
            lines.append(line)
        return lines

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.tooltip_label.hide()

    def _show_tooltip_for_date(self, py_date, global_pos):
        lines = []
        if py_date is not None and py_date.month == self.monthShown() and py_date.year == self.yearShown():
            lines = self.tooltip_lines(py_date)
        if not lines:
            self.tooltip_label.hide()
            return
        self.tooltip_label.setText("\n".join(lines))
        self.tooltip_label.adjustSize()
        self.tooltip_label.move(global_pos.x() + 15, global_pos.y() + 15)
        self.tooltip_label.show()
        self.tooltip_label.raise_()


def _next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _previous_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)


class MultiMonthCalendar(QWidget):
    """1-3 side-by-side month calendars in order, with a weather layer and a
    visibility layer. Subclasses set MONTH_COUNT_SETTING and override
    _weather_location() and _visibility_request(); optional hooks add controls,
    rows, a status line or more layers. A subclass that needs its own attributes
    in those hooks sets them before calling MultiMonthCalendar.__init__."""

    MONTH_COUNT_SETTING = None  # Settings key of the number of months shown
    CALENDAR_CLASS = LayeredMonthCalendar
    CALENDAR_MAX_HEIGHT = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calendars = []
        self.on_date_clicked = None  # callable(QDate), set by the owner
        self._weather_key = None  # Location of the forecast being waited for
        self._visibility_thread = None
        self._visibility_label = None  # Object name of the running calculation

        today = QDate.currentDate()
        self.anchor_year, self.anchor_month = today.year(), today.month()
        self.month_count = max(1, min(MAX_MONTHS, app_settings().value(self.MONTH_COUNT_SETTING, 1, type=int)))

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_controls())
        self._add_above_calendars(layout)
        self.calendars_layout = QHBoxLayout()
        layout.addLayout(self.calendars_layout, 1)
        self._add_below_calendars(layout)

        self._rebuild_calendars()

    # --- Hooks for subclasses ---

    def _add_controls(self, row):
        """Add controls after Previous / Next."""

    def _legend_text(self):
        return "Weather (top stripe):"

    def _add_above_calendars(self, layout):
        pass

    def _add_below_calendars(self, layout):
        pass

    def _weather_location(self):
        """(lat, lon, timezone) to show the forecast for, or None for none."""
        return None

    def _visibility_request(self):
        """MonthlyVisibilityThread arguments (other than the dates) for the
        object to show, or None for no visibility layer."""
        return None

    def _visibility_status(self, text):
        """Progress of the visibility calculation, for a status line."""

    def _on_months_changed(self):
        """The months shown changed (also called once at the start)."""
        self._refresh_weather_layer()
        self._refresh_visibility_layer()

    # --- Months ---

    def _build_controls(self):
        row = QHBoxLayout()

        row.addWidget(QLabel("Months:"))
        self.month_count_combo = QComboBox()
        self.month_count_combo.addItems(["1 Month"] + [f"{n} Months" for n in range(2, MAX_MONTHS + 1)])
        self.month_count_combo.setCurrentIndex(self.month_count - 1)
        self.month_count_combo.currentIndexChanged.connect(self._on_month_count_changed)
        row.addWidget(self.month_count_combo)

        previous_button = QPushButton("◀ Previous")
        previous_button.clicked.connect(lambda: self._set_anchor(*_previous_month(self.anchor_year,
                                                                                  self.anchor_month)))
        row.addWidget(previous_button)
        next_button = QPushButton("Next ▶")
        next_button.clicked.connect(lambda: self._set_anchor(*_next_month(self.anchor_year, self.anchor_month)))
        row.addWidget(next_button)

        self._add_controls(row)
        row.addStretch()

        from cosmoscollection.planning.WeatherFetch import RATINGS
        from cosmoscollection.planning.WeatherForecast import RATING_COLOR_KEYS
        self.legend_label = QLabel(self._legend_text())
        row.addWidget(self.legend_label)
        for _, label in RATINGS:
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            themed_style(swatch, lambda key=RATING_COLOR_KEYS[label]:
                         f"background-color: {COLORS[key]}; border: 1px solid {COLORS['border']};")
            row.addWidget(swatch)
            row.addWidget(QLabel(label))
        return row

    def _on_month_count_changed(self, index):
        self.month_count = index + 1
        app_settings().setValue(self.MONTH_COUNT_SETTING, self.month_count)
        self._rebuild_calendars()

    def _display_range(self):
        """First and last date of the months shown."""
        last_year, last_month = self.anchor_year, self.anchor_month
        for _ in range(self.month_count - 1):
            last_year, last_month = _next_month(last_year, last_month)
        return (date_cls(self.anchor_year, self.anchor_month, 1),
                date_cls(last_year, last_month, calendar.monthrange(last_year, last_month)[1]))

    def _rebuild_calendars(self):
        for cal in self.calendars:
            cal.tooltip_label.hide()
            cal.setParent(None)
            cal.deleteLater()
        self.calendars = []
        while self.calendars_layout.count():
            self.calendars_layout.takeAt(0)

        year, month = self.anchor_year, self.anchor_month
        for i in range(self.month_count):
            cal = self.CALENDAR_CLASS()
            if self.CALENDAR_MAX_HEIGHT:
                cal.setMaximumHeight(self.CALENDAR_MAX_HEIGHT)
            cal.setCurrentPage(year, month)
            if i == 0:
                cal.currentPageChanged.connect(self._set_anchor)  # Its own arrows and month menu
            else:
                cal.setNavigationBarVisible(False)
            cal.clicked.connect(self._on_date_clicked)
            self.calendars_layout.addWidget(cal)
            self.calendars.append(cal)
            year, month = _next_month(year, month)

        self._on_months_changed()

    def _set_anchor(self, year, month):
        """Show the months starting at year/month."""
        self.anchor_year, self.anchor_month = year, month
        for cal in self.calendars:
            cal.blockSignals(True)
            cal.setCurrentPage(year, month)
            cal.blockSignals(False)
            year, month = _next_month(year, month)
        self._on_months_changed()

    def _on_date_clicked(self, qdate):
        if self.on_date_clicked:
            self.on_date_clicked(qdate)

    def show_date(self, qdate):
        """Box qdate as the selected day, turning the pages to it if it's not in
        the months shown."""
        if not qdate.isValid():
            return
        start, end = self._display_range()
        if not start <= qdate.toPython() <= end:
            self._set_anchor(qdate.year(), qdate.month())
        # Every month gets the date; only the one it falls in draws the box.
        # Selecting turns a calendar's page to the date, so turn it back.
        year, month = self.anchor_year, self.anchor_month
        for cal in self.calendars:
            cal.blockSignals(True)
            cal.setSelectedDate(qdate)
            cal.setCurrentPage(year, month)
            cal.blockSignals(False)
            cal.update()
            year, month = _next_month(year, month)

    def _set_layers(self, **layers):
        for cal in self.calendars:
            cal.set_data(**layers)

    # --- Weather ---

    def _refresh_weather_layer(self):
        from cosmoscollection.planning.WeatherForecast import SharedWeatherFetch, WeatherCache

        location = self._weather_location()
        if location is None:
            self._weather_key = None
            self._apply_weather_list([])
            return
        lat, lon, timezone = location
        self._weather_key = key = SharedWeatherFetch.key(lat, lon)
        cached = WeatherCache().get(lat, lon)
        if cached:
            self._apply_weather_list(cached)
            return
        SharedWeatherFetch.request(lat, lon, timezone, lambda data: self._on_weather_loaded(data, key),
                                   lambda message: logger.debug(f"Calendar weather fetch failed: {message}"))

    def _on_weather_loaded(self, data, key):
        if key != self._weather_key:
            return  # Another location has been chosen since, or the window closed
        try:
            self._apply_weather_list(data)
        except RuntimeError:
            pass  # The window was closed while the fetch ran

    def _apply_weather_list(self, daily_summaries):
        scores, details = {}, {}
        for summary in daily_summaries:
            day = summary.date.date()
            scores[day] = summary.astro_score
            details[day] = {"cloud_cover": summary.tonight_avg_cloud_cover, "seeing": summary.seeing_estimate}
        self._set_layers(weather_scores=scores, weather_details=details)

    # --- Visibility ---

    def _stop_visibility_thread(self):
        """Let a running calculation go without waiting for it (see retire_thread)."""
        if self._visibility_thread is not None and self._visibility_thread.isRunning():
            self._visibility_thread.requestInterruption()
        retire_thread(self._visibility_thread)
        self._visibility_thread = None

    def _refresh_visibility_layer(self):
        self._stop_visibility_thread()
        for cal in self.calendars:
            cal.clear_visibility()
        request = self._visibility_request()
        if request is None:
            return

        start_date, end_date = self._display_range()
        self._visibility_label = request.get("dso_name") or "DSO"
        self._visibility_status(f"Calculating monthly visibility for {self._visibility_label}...")
        thread = MonthlyVisibilityThread(start_date, end_date, **request)
        thread.progress.connect(self._on_visibility_progress)
        thread.result_ready.connect(self._on_visibility_finished)
        thread.error.connect(self._on_visibility_error)
        # Referenced until it ends even if this widget goes first (deleting a
        # running QThread takes the app down)
        keep_alive_until_finished(thread)
        self._visibility_thread = thread
        thread.start()

    def _on_visibility_progress(self, done, total):
        self._visibility_status(f"Calculating {self._visibility_label}: {done}/{total} days...")

    def _on_visibility_finished(self, result):
        hours_by_date, moon_by_date = result
        self._set_layers(visibility_hours=hours_by_date, moon_illumination=moon_by_date)
        self._visibility_status(f"Monthly visibility for {self._visibility_label} (hover a day for details)")

    def _on_visibility_error(self, message):
        logger.debug(f"Calendar visibility calculation failed: {message}")
        self._visibility_status(f"Error: {message}")

    def shutdown(self):
        """Stop the calculations and close the tooltips - for the owner's closeEvent."""
        self._stop_visibility_thread()
        self._weather_key = None
        for cal in self.calendars:
            cal.tooltip_label.hide()
            cal.tooltip_label.destroy()
