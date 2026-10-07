#!/usr/bin/env python3
"""
Light Quality Review Window
Grades a session's light frames with FrameQuality - star brightness (clouds),
star count, FWHM, elongation and sky background, each against the frames that
share its filter, exposure and binning - and lists them best first, so only
the best subs get stacked. Measurements are cached per file (keyed by size and
modification time), so reopening the review only analyzes new or changed subs.
"""

import os
import json
import math
import time
import shutil
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import (Qt, QItemSelection, QItemSelectionModel, QPointF, QSettings, QThread, QTimer, QUrl,
                            Signal)
from PySide6.QtGui import QColor, QCursor, QDesktopServices, QGuiApplication, QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QLabel,
                               QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QSplitter, QGroupBox, QDoubleSpinBox, QProgressBar,
                               QMenu, QMessageBox, QSizePolicy, QGraphicsView, QGraphicsScene, QToolButton,
                               QWidgetAction, QScrollArea, QCheckBox, QFileDialog, QProgressDialog, QSpinBox,
                               QToolTip)

import FrameQuality
import SessionFileScanner
import SessionObservations
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import (COLORS, chart_background, chart_color, chart_font_size, font_size, themed_style,
                   theme_manager)
from SessionManager import format_duration, format_exposure, _retire_thread, _rollback

logger = logging.getLogger(__name__)

def analysis_workers():
    """Parallel file reads + analyses: Settings → Maximum Threads, the same
    setting used for parallel work elsewhere in the app."""
    settings = QSettings("CosmosCollection", "CosmosCollection")
    default_threads = max(1, (os.cpu_count() or 4) - 2)
    return max(1, min(settings.value("max_threads", default_threads, type=int), 128))

SETTINGS_PREFIX = "light_quality/"

GRADE_COLOR_KEYS = {
    FrameQuality.GRADE_GOOD: 'success',
    FrameQuality.GRADE_MARGINAL: 'warning',
    FrameQuality.GRADE_REJECT: 'error',
    FrameQuality.GRADE_UNGRADED: 'text_secondary',
    FrameQuality.GRADE_ERROR: 'error',
}
GRADE_SORT = {FrameQuality.GRADE_GOOD: 0, FrameQuality.GRADE_MARGINAL: 1, FrameQuality.GRADE_REJECT: 2,
              FrameQuality.GRADE_UNGRADED: 3, FrameQuality.GRADE_ERROR: 4}
WAITING = "Waiting"

# GradeSettings field, label, spin box prefix/suffix, range, step, decimals
THRESHOLDS = [
    ("flux_drop_pct", "Dimmer stars (clouds, haze)", "-", "%", 5, 95, 1, 0),
    ("star_drop_pct", "Fewer stars (obstruction)", "-", "%", 10, 95, 1, 0),
    ("fwhm_rise_pct", "Bigger stars (seeing, focus)", "+", "%", 5, 300, 1, 0),
    ("eccentricity_rise", "More elongated (trailing, guiding)", "+", "", 0.02, 0.5, 0.01, 2),
    ("outside_rise_pct", "Light outside star cores (guiding jumps, halos)", "+", "%", 10, 300, 5, 0),
    ("background_rise_pct", "Brighter sky (moon, dawn)", "+", "%", 10, 500, 5, 0),
    ("signal_drop_pct", "Less signal (moon, dawn, haze; 0 = off)", "-", "%", 0, 95, 1, 0),
    ("trail_count", "Satellite trails in one frame (Marginal at, 0 = off)", "", "", 0, 50, 1, 0),
]


# ---- Cache ----------------------------------------------------------------------

def ensure_cache_table(conn):
    """Measurements are keyed by file path, so a sub attached to several sessions
    is only analyzed once."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS framequality (
            file_path TEXT PRIMARY KEY,
            file_size INTEGER,
            file_mtime_ns INTEGER,
            version INTEGER,
            metrics_json TEXT,
            analyzed_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)


def load_cached(conn, paths):
    """{path: (size, mtime_ns, version, FrameMetrics)} for the paths with stored measurements."""
    cached = {}
    paths = list(paths)
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        rows = conn.execute(
            f"SELECT file_path, file_size, file_mtime_ns, version, metrics_json FROM framequality "
            f"WHERE file_path IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        for path, size, mtime_ns, version, metrics_json in rows:
            try:
                cached[path] = (size, mtime_ns, version, FrameQuality.FrameMetrics.from_dict(json.loads(metrics_json)))
            except (TypeError, ValueError):
                continue
    return cached


def save_cached(conn, entries):
    """entries: [(FrameMetrics, size, mtime_ns)]. Does not commit."""
    conn.executemany(
        "INSERT OR REPLACE INTO framequality (file_path, file_size, file_mtime_ns, version, metrics_json) "
        "VALUES (?, ?, ?, ?, ?)",
        [(m.path, size, mtime_ns, m.version, json.dumps(m.to_dict())) for m, size, mtime_ns in entries])


# ---- Grading a session from saved measurements -------------------------------------

# Which lights a stacking handoff takes (SessionCompletionDialog's Lights choice)
LIGHTS_ALL, LIGHTS_GOOD, LIGHTS_GOOD_MARGINAL, LIGHTS_TOP, LIGHTS_PICKED = (
    "all", "good", "good_marginal", "top", "picked")
LIGHT_CHOICE_LABELS = {
    LIGHTS_ALL: "All lights",
    LIGHTS_GOOD: "Good only",
    LIGHTS_GOOD_MARGINAL: "Good + Marginal",
    LIGHTS_TOP: "Top {percent}% of each filter",
    LIGHTS_PICKED: "Picked in the quality review",
}
# Frames in a group too small to compare can't be judged, so they're kept
KEEP_GRADES = {
    LIGHTS_GOOD: {FrameQuality.GRADE_GOOD, FrameQuality.GRADE_UNGRADED},
    LIGHTS_GOOD_MARGINAL: {FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_UNGRADED},
}
TOP_PERCENT_SETTING = SETTINGS_PREFIX + "top_percent"
DEFAULT_TOP_PERCENT = 75
_RANKED_GRADES = (FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_REJECT)


def light_choice_label(choice, percent):
    return LIGHT_CHOICE_LABELS[choice].format(percent=percent)


def saved_top_percent():
    value = QSettings("CosmosCollection", "CosmosCollection").value(TOP_PERCENT_SETTING, DEFAULT_TOP_PERCENT, type=int)
    return min(max(value, 1), 100)


def save_top_percent(percent):
    QSettings("CosmosCollection", "CosmosCollection").setValue(TOP_PERCENT_SETTING, int(percent))


def top_percent_lights(lights, grades, percent):
    """The best `percent` of each group's graded lights by score - always at
    least one. Per group, as scores only compare frames of the same filter,
    exposure and binning (which also keeps the stack's filter balance). Frames
    in groups too small to grade are kept; missing or unanalyzed ones aren't.
    lights: {path: file dict}, grades: {path: FrameGrade}."""
    keep, ranked = set(), {}
    for path, g in grades.items():
        if path not in lights:
            continue
        if g.grade == FrameQuality.GRADE_UNGRADED:
            keep.add(path)
        elif g.grade in _RANKED_GRADES:
            ranked.setdefault(group_key(lights[path]), []).append((g.score, path))
    for members in ranked.values():
        members.sort(reverse=True)
        keep.update(path for _score, path in members[:max(1, math.ceil(len(members) * percent / 100))])
    return keep


def session_light_files(conn, session_id):
    """{path: file dict} for a session's lights, oldest first. Same rule as the
    stacking handoff: untyped files are lights only when the session has no typed lights."""
    files = SessionObservations.load_session_files(conn, session_id)
    has_typed = any(f["frame_type"] == "Light" for f in files)
    lights = {}
    for f in files:
        path = f["file_path"]
        if path and path not in lights and (f["frame_type"] == "Light"
                                            or (f["frame_type"] == "Unknown" and not has_typed)):
            lights[path] = f
    return lights


def group_key(light):
    """Frames are graded against others with the same filter, exposure and binning."""
    return (light["filter_name"] or "No filter", light["exptime_seconds"], light["xbinning"] or 1)


def saved_grade_settings():
    """The thresholds last set in the review dialog."""
    settings = QSettings("CosmosCollection", "CosmosCollection")
    defaults = FrameQuality.GradeSettings()
    return FrameQuality.GradeSettings(**{
        name: settings.value(SETTINGS_PREFIX + name, getattr(defaults, name), type=float)
        for name, *_rest in THRESHOLDS})


def grade_session_lights(conn, session_id):
    """(lights, {path: FrameGrade}) for a session from its saved measurements and
    the saved thresholds. Lights never analyzed have no grade; the grades are
    empty when none has been analyzed."""
    lights = session_light_files(conn, session_id)
    ensure_cache_table(conn)
    metrics = {path: m for path, (_size, _mtime, version, m) in load_cached(conn, lights).items()
               if version == FrameQuality.ANALYSIS_VERSION}
    if not metrics:
        return lights, {}
    grades = FrameQuality.grade_frames(((group_key(lights[p]), m) for p, m in metrics.items()),
                                       saved_grade_settings())
    return lights, grades


# ---- Workers --------------------------------------------------------------------

class AnalysisWorker(QThread):
    """Checks each light against its cached measurements and analyzes the new or
    changed ones, several at a time."""
    checked = Signal(int)                 # how many frames need analyzing
    frame_ready = Signal(object, object)  # FrameMetrics, (size, mtime_ns) to cache it under - or None

    def __init__(self, paths, cached, force=False, workers=1):
        super().__init__()
        self.paths = paths
        self.cached = cached
        self.force = force
        self.workers = workers
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled

    def run(self):
        todo = []
        for path in self.paths:
            if self._cancelled:
                return
            try:
                stat = os.stat(path)
            except OSError:
                self.frame_ready.emit(FrameQuality.FrameMetrics(path=path, error="File not found"), None)
                continue
            hit = None if self.force else self.cached.get(path)
            if (hit and hit[0] == stat.st_size and hit[1] == stat.st_mtime_ns
                    and hit[2] == FrameQuality.ANALYSIS_VERSION):
                continue  # already showing the cached measurements
            todo.append((path, stat.st_size, stat.st_mtime_ns))

        self.checked.emit(len(todo))
        if not todo or self._cancelled:
            return
        pool = ThreadPoolExecutor(max_workers=self.workers)
        futures, emitted = {}, set()
        try:
            futures = {pool.submit(FrameQuality.analyze_file, path): (size, mtime_ns)
                       for path, size, mtime_ns in todo}
            for future in as_completed(futures):
                if self._cancelled:
                    break
                self._emit(future, futures[future])
                emitted.add(future)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        # Frames already being read when stopped still finish - keep them too
        for future, stamp in futures.items():
            if future not in emitted and future.done() and not future.cancelled():
                self._emit(future, stamp)

    def _emit(self, future, stamp):
        metrics = future.result()
        self.frame_ready.emit(metrics, stamp if metrics.ok else None)


def _gray_image(values):
    """A 0-1 float array as an 8-bit grayscale QImage that owns its pixels."""
    pixels = np.ascontiguousarray((np.clip(values, 0, 1) * 255).astype(np.uint8))
    height, width = pixels.shape
    return QImage(pixels.data, width, height, width, QImage.Format_Grayscale8).copy()


def _unused_path(path):
    """path, or 'name (2).fits' etc. if something's already there."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    number = 2
    while os.path.exists(f"{stem} ({number}){ext}"):
        number += 1
    return f"{stem} ({number}){ext}"


class FileOperationWorker(QThread):
    """Moves or permanently deletes files one at a time, off the UI thread (a
    move to another drive copies). Only files that succeed are reported done."""
    MOVE, DELETE = "move", "delete"
    progress = Signal(int, int)
    done = Signal(object, object)  # {path: new path, or None when deleted}, {path: error text}

    def __init__(self, mode, paths, destination=None):
        super().__init__()
        self.mode = mode
        self.paths = paths
        self.destination = destination  # None: a REJECTED_FOLDER beside each file
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        finished, failed = {}, {}
        for i, path in enumerate(self.paths, 1):
            if self._cancelled:
                break
            try:
                if self.mode == self.DELETE:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass  # already gone - still detach it
                    finished[path] = None
                else:
                    folder = self.destination or os.path.join(os.path.dirname(path),
                                                              SessionFileScanner.REJECTED_FOLDER)
                    if os.path.normcase(os.path.abspath(folder)) == os.path.normcase(os.path.dirname(os.path.abspath(path))):
                        failed[path] = "Already in that folder"
                    else:
                        os.makedirs(folder, exist_ok=True)
                        target = _unused_path(os.path.join(folder, os.path.basename(path)))
                        shutil.move(path, target)
                        finished[path] = target
            except OSError as e:
                failed[path] = e.strerror or str(e)
            self.progress.emit(i, len(self.paths))
        self.done.emit(finished, failed)


class InspectionLoader(QThread):
    """Reads a frame and renders its overview and corner/edge/center crops."""
    loaded = Signal(str, object)  # path, {"overview": QImage, "crops": [(QImage, QPointF)]} or None

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            overview, _scale, crops = FrameQuality.inspection_images(self.path)
            result = {"overview": _gray_image(overview),
                      "crops": [(_gray_image(image), QPointF(*center)) for image, center in crops]}
        except Exception as e:
            logger.debug(f"Could not render {self.path} for review: {e}")
            result = None
        self.loaded.emit(self.path, result)


# ---- Widgets --------------------------------------------------------------------

class SortItem(QTableWidgetItem):
    """A table cell that sorts by a key rather than its text."""
    SORT_ROLE = Qt.UserRole + 1

    def __init__(self, text, sort_key, align=Qt.AlignCenter):
        super().__init__(text)
        self.setData(self.SORT_ROLE, sort_key)
        self.setTextAlignment(align)

    def __lt__(self, other):
        try:
            return self.data(self.SORT_ROLE) < other.data(self.SORT_ROLE)
        except TypeError:
            return super().__lt__(other)


class CropTile(QLabel):
    """One crop, scaled up to fill the tile with crisp (unsmoothed) pixels so
    star shapes show as they are. Clicking it zooms the preview there."""
    clicked = Signal(object)  # the crop's center in overview pixels (QPointF)

    def __init__(self):
        super().__init__()
        self._image = None
        self.center = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(60, 60)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to zoom the preview here")

    def set_image(self, image, center=None):
        self._image = image
        self.center = center
        if image is None:
            self.setPixmap(QPixmap())
        else:
            self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._image is not None:
            self.setPixmap(QPixmap.fromImage(self._image).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def mousePressEvent(self, event):
        if self._image is not None and self.center is not None:
            self.clicked.emit(self.center)
        super().mousePressEvent(event)


class ZoomableImageView(QGraphicsView):
    """An image that zooms with the mouse wheel (around the cursor), pans by
    dragging, and toggles fit / 100% on double-click. A zoomed-in view carries
    over to the next image of the same size, so the same stars can be compared
    frame to frame."""
    zoom_changed = Signal()
    ZOOM_STEP = 1.25
    MAX_ZOOM = 16.0

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMinimumSize(320, 200)
        self._item = None
        self._fit = True
        self._view_state = None  # (image size, transform, center) of the last image shown
        self._loading = QLabel("Loading...", self.viewport())
        themed_style(self._loading, lambda: f"background: {COLORS['background']}; color: {COLORS['text']}; "
                                            f"padding: 3px 8px;")
        self._loading.move(8, 8)
        self._loading.hide()

    def set_image(self, image, message=""):
        if self._item is not None:
            self._view_state = (self._item.pixmap().size(), self.transform(),
                                self.mapToScene(self.viewport().rect().center()))
        self.scene().clear()
        self._item = None
        self.show_loading(False)
        if image is None:
            text = self.scene().addText(message)
            text.setDefaultTextColor(QColor(COLORS['text_secondary']))
            self.setSceneRect(text.boundingRect())
            self.setTransform(QTransform())
            self.zoom_changed.emit()
            return
        self._item = self.scene().addPixmap(QPixmap.fromImage(image))
        self.setSceneRect(self._item.boundingRect())
        state = self._view_state
        if not self._fit and state and state[0] == image.size():
            self.setTransform(state[1])
            self.centerOn(state[2])
            self._zoom_applied()
        else:
            self.fit()

    def show_loading(self, loading):
        self._loading.adjustSize()
        self._loading.setVisible(loading)

    def has_image(self):
        return self._item is not None

    def zoom(self):
        return self.transform().m11()

    def _fit_zoom(self):
        size = self._item.pixmap().size()
        viewport = self.viewport().size()
        return min(viewport.width() / max(size.width(), 1), viewport.height() / max(size.height(), 1))

    def fit(self):
        self._fit = True
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)
        self._zoom_applied()

    def set_zoom(self, factor, center=None):
        if self._item is None:
            return
        center = center if center is not None else self.mapToScene(self.viewport().rect().center())
        factor = min(max(factor, self._fit_zoom() / 2), self.MAX_ZOOM)
        self.setTransform(QTransform.fromScale(factor, factor))
        self.centerOn(center)
        self._fit = False
        self._zoom_applied()

    def zoom_by(self, step):
        self.set_zoom(self.zoom() * step)

    def _zoom_applied(self):
        # Smoothing hides the pixels a close look is for
        self.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom() < 2)
        self.zoom_changed.emit()

    def wheelEvent(self, event):
        if self._item is None:
            return
        step = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1 / self.ZOOM_STEP
        target = self.zoom() * step
        if self._fit_zoom() / 2 <= target <= self.MAX_ZOOM:
            self.scale(step, step)  # scale() keeps the point under the cursor in place
            self._fit = False
            self._zoom_applied()

    def mouseDoubleClickEvent(self, event):
        if self._item is None:
            return
        if self._fit or abs(self.zoom() - 1) > 1e-3:
            self.set_zoom(1.0, self.mapToScene(event.position().toPoint()))
        else:
            self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit and self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._zoom_applied()


# ---- Quality chart --------------------------------------------------------------
# The frames as points in capture order, each night a shaded band - so trends
# through a night (seeing, a passing cloud, dawn) and from night to night stand
# out at a glance. Each point is colored and shaped by its grade.

@dataclass
class ChartPoint:
    """One frame on the chart, in capture order."""
    path: str
    night: str            # its night's label - consecutive equal labels form a band
    value: float = None   # None when not measured (not drawn)
    grade: str = ""
    tooltip: str = ""
    checked: bool = False


# Grade -> (COLORS key, marker, legend label). Shape as well as color, so the
# grade never rests on color alone (red/green is the classic color-blind pair).
CHART_GRADE_STYLES = [
    (FrameQuality.GRADE_GOOD, 'success', 'o', "Good"),
    (FrameQuality.GRADE_MARGINAL, 'warning', '^', "Marginal"),
    (FrameQuality.GRADE_REJECT, 'error', 'X', "Reject"),
    (FrameQuality.GRADE_UNGRADED, 'text_secondary', 'o', "Not graded"),
]
CHART_RANKED = (FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_REJECT)
CHART_MARKER_SIZE = 38     # points^2 - an 8 px marker at typical screen dpi
CHART_HIGHLIGHT_SIZE = 150
CHART_CHECKED_SIZE = 95
CHART_HIT_RADIUS_PX = 9    # how close the cursor must be to a point to hover/click it

# What the graph can plot: (key, picker label, axis label). Stars, star
# brightness and sky differ between filters, so they're plotted as % of each
# filter's median - comparable when filters share the graph.
CHART_METRICS = [
    ("score", "Score", "Score"),
    ("signal", "Signal", "Signal - star brightness vs noise (% of median)"),
    ("fwhm", "FWHM", "FWHM (px)"),
    ("eccentricity", "Eccentricity", "Eccentricity"),
    ("stars", "Stars", "Stars (% of median)"),
    ("flux", "Star Brightness", "Star brightness (% of median)"),
    ("sky", "Sky", "Sky (% of median)"),
    ("outside", "Tails/Halos", "Light outside cores (%)"),
]
GRAPH_VISIBLE_SETTING = SETTINGS_PREFIX + "graph_visible"
GRAPH_SPLITTER_SETTING = SETTINGS_PREFIX + "graph_splitter"
GRAPH_METRIC_SETTING = SETTINGS_PREFIX + "graph_metric"
HIDDEN_COLUMNS_SETTING = SETTINGS_PREFIX + "hidden_columns"  # column names, "|"-separated


class QualityChart(FigureCanvas):
    """frame_clicked(path) when a point is clicked, range_selected(paths) when
    dragged across; set_selected() and set_checked() ring points."""
    frame_clicked = Signal(str)
    range_selected = Signal(list)

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 2.2), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._points = []
        self._value_label = ""
        self._lines = []         # (y, label, kind) reference lines, or None for "pick a filter"
        self._selected = set()
        self._xy = np.empty((0, 2))
        self._screen_xy = None   # _xy in display pixels, recomputed after each draw
        self._hover_index = None
        self._highlight = None
        self._press_xy = None    # where the left button went down - a click is a press without a drag
        self._span = None
        self.ax = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("figure_leave_event", lambda _event: self._clear_hover())
        self.mpl_connect("draw_event", lambda _event: setattr(self, "_screen_xy", None))
        theme_manager().theme_changed.connect(self._redraw)
        self._redraw()

    # ---- Data

    def set_points(self, points, value_label, lines=()):
        """points: ChartPoints in capture order. value_label names the y axis.
        lines: (y, label, kind) reference lines - kind 'median', 'marginal' or
        'reject' - or None when they depend on a filter not picked."""
        self._points = list(points)
        self._value_label = value_label
        self._lines = lines if lines is None else list(lines)
        self._redraw()

    def set_selected(self, paths):
        """Ring these frames' points (the frames selected in the table)."""
        self._selected = set(paths)
        self._update_highlight()
        self.draw_idle()

    def set_checked(self, paths):
        """Square-ring the checked frames' points (and show them in the legend)."""
        paths = set(paths)
        if {p.path for p in self._points if p.checked} != (paths & {p.path for p in self._points}):
            for point in self._points:
                point.checked = point.path in paths
            self._redraw()

    # ---- Drawing

    def _redraw(self):
        self.figure.clear()
        self.figure.set_facecolor(chart_background())
        ax = self.ax = self.figure.add_subplot(111)
        surface = COLORS['background_light']
        ax.set_facecolor(surface)
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
            spine.set_linewidth(0.8)
        ax.tick_params(colors=COLORS['text_secondary'], labelsize=chart_font_size(8), length=0)
        ax.set_ylabel(self._value_label, color=COLORS['text_secondary'], fontsize=chart_font_size(8))
        self._hover_index = None

        values = np.array([p.value if p.value is not None else np.nan for p in self._points], float)
        if not self._points or np.isnan(values).all():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No measurements yet", transform=ax.transAxes, ha="center", va="center",
                    color=COLORS['text_secondary'], fontsize=chart_font_size(9))
            self._xy = np.empty((0, 2))
            self._highlight = None
            self.draw_idle()
            return

        x = np.arange(len(self._points), dtype=float)
        self._draw_nights(ax)
        ax.yaxis.grid(True, color=COLORS['border'], linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

        handles = []
        for grade, color_key, marker, label in CHART_GRADE_STYLES:
            mask = np.array([p.grade == grade for p in self._points]) & ~np.isnan(values)
            if not mask.any():
                continue
            color = chart_color(COLORS[color_key])
            # A ring in the plot's own color keeps overlapping points apart
            ax.scatter(x[mask], values[mask], s=CHART_MARKER_SIZE, marker=marker, color=color,
                       edgecolors=surface, linewidths=1.2, zorder=3)
            handles.append(Line2D([], [], linestyle="", marker=marker, markersize=7, color=color,
                                  markeredgecolor=surface, label=f"{label} ({int(mask.sum())})"))

        checked = np.array([p.checked for p in self._points]) & ~np.isnan(values)
        if checked.any():
            ax.scatter(x[checked], values[checked], s=CHART_CHECKED_SIZE, marker="s", facecolors="none",
                       edgecolors=COLORS['text_secondary'], linewidths=1.1, zorder=4)
            handles.append(Line2D([], [], linestyle="", marker="s", markersize=8, markerfacecolor="none",
                                  markeredgecolor=COLORS['text_secondary'], label=f"Checked ({int(checked.sum())})"))
        self._draw_lines(ax, values)

        if handles:
            # Above the plot, so it never covers points (the low ones matter most)
            legend = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles),
                               fontsize=chart_font_size(8), frameon=False, handletextpad=0.3,
                               columnspacing=1.2, borderaxespad=0.2)
            for text in legend.get_texts():
                text.set_color(COLORS['text_secondary'])

        ax.set_xlim(-0.5, len(self._points) - 0.5)
        # Room for the points and any reference line, with a little air
        ys = list(values[~np.isnan(values)]) + [y for y, _label, _kind in (self._lines or [])]
        low, high = min(ys), max(ys)
        pad = (high - low) * 0.08 or abs(high) * 0.05 or 1
        ax.set_ylim(low - pad, high + pad)
        self._xy = np.column_stack([x, values])
        self._highlight = ax.scatter([], [], s=CHART_HIGHLIGHT_SIZE, facecolors="none", edgecolors=COLORS['text'],
                                     linewidths=1.6, zorder=5)
        self._update_highlight()
        # Drag across to select a range of frames; a press without a drag is a click
        self._span = SpanSelector(ax, self._on_span, "horizontal", useblit=True, minspan=0.6, button=1,
                                  props=dict(facecolor=COLORS['text_secondary'], alpha=0.18))
        self.draw_idle()

    def _draw_lines(self, ax, values):
        """The median and the Marginal / Reject limits, labelled at the right end."""
        if self._lines is None:
            ax.text(0.005, 0.97, "Choose one filter in Show: to see its median and limits", transform=ax.transAxes,
                    ha="left", va="top", color=COLORS['text_secondary'], fontsize=chart_font_size(7.5), zorder=6)
            return
        to_right_edge = blended_transform_factory(ax.transAxes, ax.transData)
        for y, label, kind in self._lines:
            if kind == "median":
                ax.axhline(y, color=COLORS['text_secondary'], linewidth=1, alpha=0.8, zorder=2)
            else:
                color = chart_color(COLORS['warning' if kind == "marginal" else 'error'])
                ax.axhline(y, color=color, linewidth=1, linestyle=(0, (4, 3)), alpha=0.9, zorder=2)
            # Text in a text color - the line beside it carries the meaning
            ax.text(0.998, y, label, transform=to_right_edge, ha="right", va="bottom",
                    color=COLORS['text_secondary'], fontsize=chart_font_size(7), zorder=6)

    def _draw_nights(self, ax):
        """Alternate nights shaded, each labelled under its band."""
        runs = []  # (first index, last index, label)
        for i, point in enumerate(self._points):
            if runs and runs[-1][2] == point.night:
                runs[-1][1] = i
            else:
                runs.append([i, i, point.night])
        for n, (first, last, _label) in enumerate(runs):
            if n % 2:
                ax.axvspan(first - 0.5, last + 0.5, color=COLORS['background_lighter'], alpha=0.55,
                           linewidth=0, zorder=0)
            if n:
                ax.axvline(first - 0.5, color=COLORS['border'], linewidth=0.8, zorder=1)
        ax.set_xticks([(first + last) / 2 for first, last, _label in runs])
        ax.set_xticklabels([label for _first, _last, label in runs])

    def _update_highlight(self):
        if self._highlight is None:
            return
        rows = [i for i, p in enumerate(self._points)
                if p.path in self._selected and not np.isnan(self._xy[i, 1])] if len(self._xy) else []
        self._highlight.set_offsets(self._xy[rows] if rows else np.empty((0, 2)))

    # ---- Mouse

    def _point_at(self, event):
        """Index of the point under the mouse, or None."""
        if self.ax is None or event.inaxes is not self.ax or not len(self._xy):
            return None
        if self._screen_xy is None:
            self._screen_xy = self.ax.transData.transform(np.nan_to_num(self._xy, nan=-1e9))
        distance = np.hypot(self._screen_xy[:, 0] - event.x, self._screen_xy[:, 1] - event.y)
        distance[np.isnan(self._xy[:, 1])] = np.inf
        index = int(np.argmin(distance))
        # Event and transform pixels are both device pixels on a high-DPI screen
        return index if distance[index] <= CHART_HIT_RADIUS_PX * getattr(self, "device_pixel_ratio", 1) else None

    def _on_motion(self, event):
        index = self._point_at(event)
        if index == self._hover_index:
            return
        self._hover_index = index
        if index is None:
            self._clear_hover()
        else:
            self.setCursor(Qt.PointingHandCursor)
            QToolTip.showText(QCursor.pos(), self._points[index].tooltip, self)

    def _clear_hover(self):
        self._hover_index = None
        self.unsetCursor()
        QToolTip.hideText()

    def _on_press(self, event):
        self._press_xy = (event.x, event.y) if event.button == 1 else None

    def _on_release(self, event):
        """A click - the button released where it went down (else it was a drag)."""
        if event.button != 1 or self._press_xy is None:
            return
        moved = np.hypot(event.x - self._press_xy[0], event.y - self._press_xy[1])
        self._press_xy = None
        if moved <= 4 * getattr(self, "device_pixel_ratio", 1):
            index = self._point_at(event)
            if index is not None:
                self.frame_clicked.emit(self._points[index].path)

    def _on_span(self, x_min, x_max):
        # A little slack at the ends: a drag that starts or stops on a point includes it
        first, last = math.ceil(x_min - 0.3), math.floor(x_max + 0.3)
        paths = [p.path for i, p in enumerate(self._points)
                 if first <= i <= last and p.value is not None]
        if paths:
            self.range_selected.emit(paths)


# ---- Dialog ---------------------------------------------------------------------

class LightQualityDialog(WindowPositionMixin, QDialog):
    """Analyze and grade one session's light frames."""

    WINDOW_POSITION_KEY = "LightQualityDialog"
    COLUMNS = ["", "Grade", "Score", "File", "Night", "Filter", "Stars", "FWHM", "Eccentricity",
               "Tails/Halos", "Star Brightness", "Sky", "Trails", "Issues"]
    (COL_CHECK, COL_GRADE, COL_SCORE, COL_FILE, COL_NIGHT, COL_FILTER, COL_STARS, COL_FWHM, COL_ECC,
     COL_OUTSIDE, COL_FLUX, COL_SKY, COL_TRAILS, COL_ISSUES) = range(14)
    FIXED_COLUMNS = (COL_CHECK, COL_GRADE, COL_FILE)  # can't be hidden
    # Hidden until chosen from the header menu - the details panel always shows them
    DEFAULT_HIDDEN_COLUMNS = (COL_STARS, COL_ECC, COL_OUTSIDE, COL_FLUX, COL_SKY, COL_TRAILS)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = dict(session)
        self.db_manager = DatabaseManager()
        self._lights = {}       # path -> file dict
        self._groups = {}       # path -> (filter, exposure, binning)
        self._metrics = {}      # path -> FrameMetrics
        self._grades = {}       # path -> FrameGrade
        self._cached = {}
        self._checked = set()
        self._checks_touched = False
        self._pending_cache = []
        self.files_changed = False  # frames were removed from the session (so it needs reloading)
        self._worker = None
        self._analyzing = False      # set/cleared with the worker's start/finished (isRunning() lags finished)
        self._file_op = None
        self._file_op_running = False
        self._preview = None
        self._inspections = OrderedDict()  # path -> rendered preview, the last few frames viewed
        self._shown_path = None
        self._to_analyze = 0
        self._analyzed = 0
        self._started = 0.0
        self._regrade_timer = QTimer(self)
        self._regrade_timer.setSingleShot(True)
        self._regrade_timer.timeout.connect(self._regrade)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._load_preview)

        self.setWindowTitle(f"Light Quality - {self.session.get('dso_name', '')} {self.session.get('session_date', '')}")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        self.setModal(True)
        self.resize(1400, 820)  # default size the first time this dialog is ever opened

        self._load()
        self._setup_ui()
        self._regrade()
        self._size_columns()
        self.setup_window_position()
        QTimer.singleShot(0, self._start_analysis)

    # ---- Data -----------------------------------------------------------

    def _load(self):
        try:
            with self.db_manager.get_connection() as conn:
                ensure_cache_table(conn)
                conn.commit()
                self._lights = session_light_files(conn, self.session["id"])
                self._groups = {path: group_key(f) for path, f in self._lights.items()}
                self._cached = load_cached(conn, self._lights)
        except Exception as e:
            logger.error(f"Error loading lights for quality review: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load the session's lights: {e}")
        # Show what's cached straight away - the worker re-checks it against the files
        for path, (_size, _mtime, version, metrics) in self._cached.items():
            if version == FrameQuality.ANALYSIS_VERSION:
                self._metrics[path] = metrics

    def _group_label(self, key):
        filter_name, exposure, binning = key
        exposure_text = format_exposure(exposure) if exposure is not None else "?s"
        return f"{filter_name} · {exposure_text} · {binning}x{binning}"

    def _fill_group_combo(self):
        """'All filters' plus one entry per filter group that still has frames.
        Keeps the selection while its group exists; when its last frames were
        removed, moves to the next remaining group (the previous one if it was
        last), so the review carries on. Returns the group left behind, or None."""
        keys = sorted(set(self._groups.values()), key=lambda k: (str(k[0]), k[1] or 0, k[2]))
        old_keys = [self.group_combo.itemData(i) for i in range(1, self.group_combo.count())]
        current = self.group_combo.currentData() if self.group_combo.count() else None
        target, emptied = current, None
        if current is not None and current not in keys:
            emptied = current
            # The groups after the emptied one in the old order, then the ones before it
            position = old_keys.index(current)
            following = [k for k in old_keys[position + 1:] + old_keys[:position][::-1] if k in keys]
            target = following[0] if following else None
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("All filters", None)
        for key in keys:
            self.group_combo.addItem(self._group_label(key), key)
        # Matched in Python - tuples don't round-trip reliably through findData
        index = next((i for i in range(1, self.group_combo.count()) if self.group_combo.itemData(i) == target), 0)
        self.group_combo.setCurrentIndex(index)
        self.group_combo.blockSignals(False)
        return emptied

    def _grade_settings(self):
        # Same values saved_grade_settings() reads - each change is saved as it's made
        return FrameQuality.GradeSettings(**{name: spin.value() for name, spin in self.threshold_spins.items()})

    # ---- UI -------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # What to do next, in plain words - the how-it-works note is its tooltip
        self.guide_label = QLabel()
        self.guide_label.setWordWrap(True)
        self.guide_label.setTextFormat(Qt.RichText)
        self.guide_label.setToolTip("Each frame is compared with the others of the same filter, exposure and "
                                    "binning.\nGrades update instantly when you change the thresholds - "
                                    "no re-analysis needed.")
        self.guide_label.linkActivated.connect(lambda _link: self.stack_btn.showMenu())
        themed_style(self.guide_label, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.guide_label)

        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        self.group_combo = QComboBox()
        self._fill_group_combo()
        self.group_combo.currentIndexChanged.connect(self._apply_group_filter)
        top.addWidget(self.group_combo)
        top.addWidget(self._build_thresholds_button())
        self.graph_btn = QToolButton()
        self.graph_btn.setText("Graph")
        self.graph_btn.setCheckable(True)
        self.graph_btn.setToolTip("Show the quality graph above the frame list")
        top.addWidget(self.graph_btn)
        top.addSpacing(16)
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.RichText)
        top.addWidget(self.summary_label, 1)
        # Beside the grade counts it picks from
        self.stack_btn = QPushButton("Stack Lights")
        self.stack_btn.setToolTip("Hand the chosen lights to Siril or PixInsight (closes this review)")
        stack_menu = QMenu(self.stack_btn)
        stack_menu.aboutToShow.connect(self._fill_stack_menu)
        self.stack_btn.setMenu(stack_menu)
        top.addWidget(self.stack_btn)
        layout.addLayout(top)

        checks = QHBoxLayout()
        checks.addWidget(QLabel("Check:"))
        for text, grades in (("Rejects", {FrameQuality.GRADE_REJECT}),
                             ("Rejects + Marginal", {FrameQuality.GRADE_REJECT, FrameQuality.GRADE_MARGINAL}),
                             ("None", set())):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, g=grades: self._check_grades(g, touched=True))
            checks.addWidget(button)
        self.checked_label = QLabel()
        checks.addSpacing(12)
        checks.addWidget(self.checked_label)
        checks.addStretch()
        checks.addWidget(QLabel("Checked frames:"))
        self.remove_btn = QPushButton("Remove from Session")
        self.remove_btn.setToolTip("Detach them from this session. The files stay where they are.")
        self.remove_btn.clicked.connect(self._remove_checked)
        self.move_btn = QPushButton("Move...")
        self.move_btn.setToolTip(f"Move them into a '{SessionFileScanner.REJECTED_FOLDER}' folder (or one you choose) "
                                 "and detach them from this session.")
        self.move_btn.clicked.connect(self._move_checked)
        self.delete_btn = QPushButton("Delete...")
        self.delete_btn.setToolTip("Permanently delete the files and detach them from this session.")
        self.delete_btn.clicked.connect(self._delete_checked)
        for button in (self.remove_btn, self.move_btn, self.delete_btn):
            checks.addWidget(button)
        layout.addLayout(checks)

        # The graph above the frame list - a time series wants the full width
        self.list_splitter = QSplitter(Qt.Vertical)
        self.list_splitter.addWidget(self._build_chart_strip())
        self.list_splitter.addWidget(self._build_table())
        self.list_splitter.setCollapsible(0, False)  # hidden with the Graph button instead
        self.list_splitter.setStretchFactor(1, 1)
        settings = QSettings("CosmosCollection", "CosmosCollection")
        if not self.list_splitter.restoreState(settings.value(GRAPH_SPLITTER_SETTING, b"")):
            self.list_splitter.setSizes([220, 600])
        self.list_splitter.splitterMoved.connect(self._save_graph_layout)
        graph_visible = settings.value(GRAPH_VISIBLE_SETTING, True, type=bool)
        self.chart_strip.setVisible(graph_visible)
        self.graph_btn.setChecked(graph_visible)
        self.graph_btn.toggled.connect(self._on_graph_toggled)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.list_splitter)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 400])
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.hide()
        bottom.addWidget(self.progress_bar)
        self.status_label = QLabel()
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        bottom.addWidget(self.status_label, 1)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.stop_btn.hide()
        bottom.addWidget(self.stop_btn)
        self.reanalyze_btn = QPushButton("Re-analyze All")
        self.reanalyze_btn.setToolTip("Measure every frame again, ignoring the saved measurements.")
        self.reanalyze_btn.clicked.connect(lambda: self._start_analysis(force=True))
        bottom.addWidget(self.reanalyze_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _build_table(self):
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        # Not ResizeToContents: that re-measures a column on every cell set, which
        # took ~40s to fill a few hundred rows. Columns are sized once per fill instead.
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setSortIndicator(self.COL_SCORE, Qt.DescendingOrder)  # best first
        self.table.setSortingEnabled(True)
        tips = {
            self.COL_CHECK: "Checked frames are the ones Remove, Move and Delete act on",
            self.COL_GRADE: "Good, Marginal or Reject against the frame's group - see Issues for why",
            self.COL_SCORE: "Ranks frames by what they'd add to a stack (Top % uses it).\n"
                            "Sharpness (FWHM) and signal (star brightness vs noise) count most; the grade is decided separately.",
            self.COL_STARS: "Stars detected",
            self.COL_FWHM: "Star size (full width at half maximum), in sensor pixels and arcseconds",
            self.COL_ECC: "0 = round, 1 = a line. Elongation that all points one way is trailing.",
            self.COL_OUTSIDE: "Star light outside the round star cores vs the group's median frame.\n"
                              "A guiding jump leaves round cores with faint tails, which only this shows;\n"
                              "double images and dew or cloud halos raise it too.",
            self.COL_FLUX: "Brightness of the same stars, as % of the group's median frame.\n"
                           "Clouds and haze dim it; moonlight and seeing don't.",
            self.COL_SKY: "Sky background vs the group's median frame",
            self.COL_TRAILS: "Satellite, plane and meteor trails found in the frame.\n"
                             "Stacking's pixel rejection removes a few; many can leave traces.",
        }
        for col in range(len(self.COLUMNS)):
            tip = tips.get(col, "")
            self.table.horizontalHeaderItem(col).setToolTip(
                (tip + "\n\n" if tip else "") + "Right-click the column headers to choose which columns show.")
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_menu)
        self._apply_hidden_columns(self._saved_hidden_columns())
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(lambda: self._preview_timer.start(150))
        self.table.itemSelectionChanged.connect(self._sync_chart_selection)
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)
        return self.table

    # ---- Column choice --------------------------------------------------

    def _saved_hidden_columns(self):
        """Hidden columns, saved by name so new columns don't shift anyone's choice."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        if not settings.contains(HIDDEN_COLUMNS_SETTING):
            return set(self.DEFAULT_HIDDEN_COLUMNS)
        names = [n for n in str(settings.value(HIDDEN_COLUMNS_SETTING, "") or "").split("|") if n]
        return {self.COLUMNS.index(n) for n in names if n in self.COLUMNS} - set(self.FIXED_COLUMNS)

    def _apply_hidden_columns(self, hidden, save=False):
        for col in range(len(self.COLUMNS)):
            was_hidden = self.table.isColumnHidden(col)
            self.table.setColumnHidden(col, col in hidden)
            if was_hidden and col not in hidden and self.table.columnWidth(col) < 20:
                self.table.resizeColumnToContents(col)  # hidden since opening, so never sized
        if save:
            QSettings("CosmosCollection", "CosmosCollection").setValue(
                HIDDEN_COLUMNS_SETTING, "|".join(self.COLUMNS[col] for col in sorted(hidden)))

    def _show_header_menu(self, position):
        menu = QMenu(self)
        hidden = {col for col in range(len(self.COLUMNS)) if self.table.isColumnHidden(col)}
        for col, name in enumerate(self.COLUMNS):
            if col in self.FIXED_COLUMNS:
                continue
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(col not in hidden)
            action.toggled.connect(lambda shown, c=col: self._apply_hidden_columns(
                {x for x in range(len(self.COLUMNS)) if self.table.isColumnHidden(x) and x != c} | ({c} if not shown else set()),
                save=True))
        menu.addSeparator()
        menu.addAction("Show All Columns").triggered.connect(lambda: self._apply_hidden_columns(set(), save=True))
        menu.addAction("Reset to Default Columns").triggered.connect(
            lambda: self._apply_hidden_columns(set(self.DEFAULT_HIDDEN_COLUMNS), save=True))
        menu.exec(self.table.horizontalHeader().viewport().mapToGlobal(position))

    def _build_chart_strip(self):
        self.chart_strip = QWidget()
        layout = QVBoxLayout(self.chart_strip)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Graph:"))
        self.chart_metric_combo = QComboBox()
        for key, label, _axis in CHART_METRICS:
            self.chart_metric_combo.addItem(label, key)
        saved = QSettings("CosmosCollection", "CosmosCollection").value(GRAPH_METRIC_SETTING, "score", type=str)
        self.chart_metric_combo.setCurrentIndex(max(self.chart_metric_combo.findData(saved), 0))
        self.chart_metric_combo.setToolTip("Stars, Star Brightness and Sky are shown as % of their filter's median, "
                                           "so filters can share the graph")
        self.chart_metric_combo.currentIndexChanged.connect(self._on_chart_metric_changed)
        bar.addWidget(self.chart_metric_combo)
        hint = QLabel("Frames in the order taken, a band per night · click a point, or drag across several, "
                      "to select them")
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(8)};")
        bar.addWidget(hint)
        bar.addStretch()
        layout.addLayout(bar)
        self.chart = QualityChart()
        self.chart.frame_clicked.connect(self._on_chart_frame_clicked)
        self.chart.range_selected.connect(self._on_chart_range_selected)
        layout.addWidget(self.chart, 1)
        return self.chart_strip

    def _build_side_panel(self):
        splitter = QSplitter(Qt.Vertical)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_view = ZoomableImageView()
        themed_style(self.preview_view, lambda: f"background: {COLORS['background_light']};")
        self.preview_view.zoom_changed.connect(self._update_zoom_label)
        bar = QHBoxLayout()
        for text, tip, slot in (
                ("−", "Zoom out", lambda: self.preview_view.zoom_by(1 / ZoomableImageView.ZOOM_STEP)),
                ("+", "Zoom in", lambda: self.preview_view.zoom_by(ZoomableImageView.ZOOM_STEP)),
                ("Fit", "Show the whole frame", self.preview_view.fit),
                ("100%", "One screen pixel per image pixel", lambda: self.preview_view.set_zoom(1.0))):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        self.zoom_label = QLabel()
        bar.addWidget(self.zoom_label)
        bar.addStretch()
        hint = QLabel("Wheel: zoom · Drag: pan · Double-click: 100% / fit")
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(8)};")
        bar.addWidget(hint)
        preview_layout.addLayout(bar)
        preview_layout.addWidget(self.preview_view, 1)
        self.preview_view.set_image(None, "Select a frame to preview it")
        splitter.addWidget(preview)

        crops = QWidget()
        crops_layout = QVBoxLayout(crops)
        crops_layout.setContentsMargins(0, 0, 0, 0)
        crops_title = QLabel("Corners, edges and center at full resolution - "
                             "trailing, soft focus or tilt shows here. Click one to zoom there.")
        crops_title.setWordWrap(True)
        themed_style(crops_title, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(8)};")
        crops_layout.addWidget(crops_title)
        grid = QGridLayout()
        grid.setSpacing(3)
        self.crop_tiles = []
        for i in range(9):
            tile = CropTile()
            themed_style(tile, lambda: f"background: {COLORS['background_light']};")
            tile.clicked.connect(self._zoom_to_crop)
            grid.addWidget(tile, i // 3, i % 3)
            self.crop_tiles.append(tile)
        crops_layout.addLayout(grid, 1)
        splitter.addWidget(crops)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        details = QScrollArea()
        details.setWidgetResizable(True)
        details.setWidget(self.details_label)
        splitter.addWidget(details)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([320, 320, 130])
        return splitter

    def _build_thresholds_button(self):
        """The grading thresholds, in a drop-down so the preview gets the room."""
        button = QToolButton()
        button.setText("Thresholds")
        button.setToolTip("How far a frame may fall behind its group before it's Marginal or a Reject")
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        action = QWidgetAction(menu)
        action.setDefaultWidget(self._build_thresholds_box())
        menu.addAction(action)
        button.setMenu(menu)
        return button

    def _build_thresholds_box(self):
        box = QGroupBox("Reject when worse than the group's median by")
        grid = QGridLayout(box)
        note = QLabel("Half of each is Marginal. A brighter sky alone is Marginal at most, though the signal "
                      "it costs can reject a frame. Fewer stars and less signal only count on their own - "
                      "moonlight causes both. The median leaves out "
                      "rejected frames, so removing them doesn't turn up new rejects. Satellite trails are "
                      "a count per frame instead, and make it Marginal at most.")
        note.setWordWrap(True)
        themed_style(note, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(8)};")
        grid.addWidget(note, 0, 0, 1, 2)

        settings = QSettings("CosmosCollection", "CosmosCollection")
        defaults = FrameQuality.GradeSettings()
        self.threshold_spins = {}
        for row, (name, label, prefix, suffix, low, high, step, decimals) in enumerate(THRESHOLDS, 1):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setPrefix(prefix)
            spin.setSuffix(suffix)
            spin.setValue(settings.value(SETTINGS_PREFIX + name, getattr(defaults, name), type=float))
            spin.valueChanged.connect(lambda value, n=name: self._on_threshold_changed(n, value))
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(spin, row, 1)
            self.threshold_spins[name] = spin
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_thresholds)
        grid.addWidget(reset_btn, len(THRESHOLDS) + 1, 1)
        return box

    # ---- Analysis -------------------------------------------------------

    def _analysis_running(self):
        return self._worker is not None and self._worker.isRunning()

    def _start_analysis(self, force=False):
        if self._analysis_running() or not self._lights:
            if not self._lights:
                self.status_label.setText("This session has no light frames attached.")
            return
        if force:
            self._cached = {}
        # Read per run, so a change in Settings applies to the next analysis
        self._worker = AnalysisWorker(list(self._lights), self._cached, force=force, workers=analysis_workers())
        self._worker.checked.connect(self._on_checked)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_analysis_finished)
        self._to_analyze = self._analyzed = 0
        self._started = time.perf_counter()
        self.status_label.setText(f"Checking {len(self._lights)} light(s) against saved measurements...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.stop_btn.show()
        self.reanalyze_btn.setEnabled(False)
        self._analyzing = True
        self._worker.start()
        self._update_action_buttons()
        self._refresh_guide()

    def _on_checked(self, count):
        self._to_analyze = count
        self._started = time.perf_counter()
        if count:
            self.progress_bar.setRange(0, count)
            self.progress_bar.setValue(0)
            self.status_label.setText(f"Analyzing {count} frame(s)...")

    def _on_frame_ready(self, metrics, stamp):
        self._metrics[metrics.path] = metrics
        if stamp is not None:
            self._pending_cache.append((metrics, stamp[0], stamp[1]))
            self._analyzed += 1
            self.progress_bar.setValue(self._analyzed)
            self.status_label.setText(self._progress_text())
        if not self._regrade_timer.isActive():
            self._regrade_timer.start(500)  # regrading 400 rows per result would stall the UI

    def _progress_text(self):
        elapsed = time.perf_counter() - self._started
        remaining = self._to_analyze - self._analyzed
        text = f"Analyzed {self._analyzed} of {self._to_analyze}"
        if self._analyzed and remaining:
            seconds = elapsed / self._analyzed * remaining
            text += f" · about {format_duration(seconds)} left" if seconds >= 60 else " · under a minute left"
        return text

    def _stop_analysis(self):
        if self._analysis_running():
            self._worker.cancel()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping after the frames being read...")

    def _on_analysis_finished(self):
        self._analyzing = False
        self._flush_cache()
        stopped = self._worker is not None and self._worker.cancelled
        self.progress_bar.hide()
        self.stop_btn.hide()
        self.stop_btn.setEnabled(True)
        self.reanalyze_btn.setEnabled(True)
        missing = sum(1 for m in self._metrics.values() if m.error == "File not found")
        if stopped:
            text = f"Stopped - {self._analyzed} of {self._to_analyze} frame(s) analyzed."
        elif self._to_analyze:
            text = (f"Analyzed {self._to_analyze} frame(s) in {format_duration(time.perf_counter() - self._started)}"
                    f"; {len(self._lights) - self._to_analyze} from saved measurements.")
        else:
            text = "All frames from saved measurements."
        if missing:
            text += f"  {missing} file(s) not found on disk."
        self.status_label.setText(text)
        self._regrade()
        self._size_columns()
        if not self._checks_touched:
            self._check_grades({FrameQuality.GRADE_REJECT})

    def _flush_cache(self):
        if not self._pending_cache:
            return
        try:
            with self.db_manager.get_connection() as conn:
                save_cached(conn, self._pending_cache)
                conn.commit()
            self._pending_cache = []
        except Exception as e:
            logger.error(f"Error saving frame quality measurements: {e}")

    # ---- Grading and the table ------------------------------------------

    def _on_threshold_changed(self, name, value):
        QSettings("CosmosCollection", "CosmosCollection").setValue(SETTINGS_PREFIX + name, value)
        self._regrade()

    def _reset_thresholds(self):
        defaults = FrameQuality.GradeSettings()
        for name, spin in self.threshold_spins.items():
            spin.setValue(getattr(defaults, name))

    def _regrade(self):
        self._flush_cache()
        self._grades = FrameQuality.grade_frames(
            ((self._groups[path], m) for path, m in self._metrics.items() if path in self._groups),
            self._grade_settings())
        self._refresh_table()

    def _refresh_table(self):
        selected = self._selected_paths()
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._lights))
        for row, path in enumerate(self._lights):
            self._fill_row(row, path)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)
        self._apply_group_filter()
        if selected:
            self._select_paths(selected)

    def _size_columns(self):
        """On opening and when analysis ends - not per regrade, which would jiggle
        widths while results arrive and undo widths the user dragged."""
        self.table.resizeColumnsToContents()
        # Room for values still to come (rows may all be waiting on analysis)
        for col, sample in ((self.COL_SCORE, "100"), (self.COL_STARS, "88,888"), (self.COL_FWHM, '8.88 px (88.8")'),
                            (self.COL_OUTSIDE, "+888%"), (self.COL_FLUX, "100%"), (self.COL_SKY, "+888%"),
                            (self.COL_TRAILS, "88")):
            width = self.table.fontMetrics().horizontalAdvance(sample) + 24
            self.table.setColumnWidth(col, max(self.table.columnWidth(col), width))
        # NINA-style names are long - a fixed (draggable) width keeps the metrics in view
        self.table.setColumnWidth(self.COL_FILE, min(self.table.columnWidth(self.COL_FILE), 230))

    def _fill_row(self, row, path):
        f = self._lights[path]
        m = self._metrics.get(path)
        g = self._grades.get(path)
        grade = g.grade if g else WAITING

        check = SortItem("", 1 if path in self._checked else 0)
        check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        check.setCheckState(Qt.Checked if path in self._checked else Qt.Unchecked)
        check.setData(Qt.UserRole, path)
        self.table.setItem(row, self.COL_CHECK, check)

        grade_item = SortItem(grade, GRADE_SORT.get(grade, 5))
        grade_item.setForeground(QColor(COLORS[GRADE_COLOR_KEYS.get(grade, 'text_secondary')]))
        self.table.setItem(row, self.COL_GRADE, grade_item)
        graded = g is not None and grade in (FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL,
                                             FrameQuality.GRADE_REJECT)
        self.table.setItem(row, self.COL_SCORE, SortItem(f"{g.score:.0f}" if graded else "", g.score if graded else -1))

        name_item = SortItem(os.path.basename(path), os.path.basename(path).lower(), Qt.AlignLeft | Qt.AlignVCenter)
        name_item.setToolTip(path)
        self.table.setItem(row, self.COL_FILE, name_item)
        night = f["night_date"] or ""
        self.table.setItem(row, self.COL_NIGHT, SortItem(night, night))
        filter_name = self._groups[path][0]
        self.table.setItem(row, self.COL_FILTER, SortItem(filter_name, filter_name))

        ok = m is not None and m.ok
        relative = g.relative if g else {}
        stars_item = SortItem(f"{m.stars:,}" if ok else "", m.stars if ok else -1)
        if "stars" in relative:
            stars_item.setToolTip(f"{relative['stars']:.0f}% of the group's median")
        self.table.setItem(row, self.COL_STARS, stars_item)

        fwhm = m.fwhm_px if ok else None
        fwhm_text = ""
        if fwhm is not None:
            fwhm_text = f"{fwhm:.2f} px" + (f' ({m.fwhm_arcsec:.1f}")' if m.fwhm_arcsec else "")
        self.table.setItem(row, self.COL_FWHM, SortItem(fwhm_text, fwhm if fwhm is not None else -1))

        ecc = m.eccentricity if ok else None
        ecc_item = SortItem(f"{ecc:.2f}" if ecc is not None else "", ecc if ecc is not None else -1)
        if ecc is not None and m.alignment is not None:
            ecc_item.setToolTip(f"Alignment {m.alignment:.2f} (0 = random directions, 1 = all one way)")
        self.table.setItem(row, self.COL_ECC, ecc_item)

        outside = relative.get("outside")
        outside_item = SortItem(f"{outside - 100:+.0f}%" if outside is not None else "",
                                outside if outside is not None else -1)
        if ok and g is not None and g.outside is not None:
            outside_item.setToolTip(f"{g.outside * 100:.1f}% of the star light is outside the cores")
        self.table.setItem(row, self.COL_OUTSIDE, outside_item)

        flux = relative.get("flux")
        self.table.setItem(row, self.COL_FLUX, SortItem(f"{flux:.0f}%" if flux is not None else "",
                                                       flux if flux is not None else -1))
        sky = relative.get("background")
        sky_item = SortItem(f"{sky - 100:+.0f}%" if sky is not None else "", sky if sky is not None else -1)
        if ok and m.background is not None:
            sky_item.setToolTip(f"{m.background:.0f} ADU")
        self.table.setItem(row, self.COL_SKY, sky_item)

        trails = m.trails if ok else None
        trails_item = SortItem(str(len(trails)) if trails else "", len(trails) if trails is not None else -1)
        if trails:
            trails_item.setToolTip("\n".join(f"{math.hypot(x1 - x0, y1 - y0):.0f} px long, {excess:.0f}x the noise"
                                             for x0, y0, x1, y1, excess in trails))
        self.table.setItem(row, self.COL_TRAILS, trails_item)

        issues ="; ".join(g.flags) if g else ("Analyzing..." if not m else "")
        issues_item = SortItem(issues, issues, Qt.AlignLeft | Qt.AlignVCenter)
        issues_item.setToolTip(issues.replace("; ", "\n"))
        self.table.setItem(row, self.COL_ISSUES, issues_item)

    def _row_path(self, row):
        item = self.table.item(row, self.COL_CHECK)
        return item.data(Qt.UserRole) if item else None

    def _apply_group_filter(self):
        key = self.group_combo.currentData()
        for row in range(self.table.rowCount()):
            path = self._row_path(row)
            self.table.setRowHidden(row, key is not None and self._groups.get(path) != key)
        self._refresh_summary()
        self._refresh_chart()

    # ---- Quality graph --------------------------------------------------

    def _refresh_chart(self):
        """Redraw the graph from the frames shown (it follows the filter selector)."""
        if not hasattr(self, "chart") or not self.graph_btn.isChecked():
            return  # hidden - redrawn when shown
        key = self.chart_metric_combo.currentData()
        axis = next(axis for k, _label, axis in CHART_METRICS if k == key)
        paths = sorted(self._visible_paths(), key=lambda p: self._lights[p]["date_obs"] or "~")
        points = []
        for path in paths:
            f, m, g = self._lights[path], self._metrics.get(path), self._grades.get(path)
            value = self._chart_value(key, m, g)
            tooltip = [self._local_time(f["date_obs"]), os.path.basename(path)]
            if g:
                tooltip.insert(1, f"{g.grade}" + (f" · score {g.score:.0f}" if g.grade in CHART_RANKED else ""))
            if value is not None and key != "score":
                tooltip.insert(2, f"{axis}: {value:.2f}" if abs(value) < 10 else f"{axis}: {value:,.0f}")
            points.append(ChartPoint(path=path, night=self._night_label(f["night_date"]), value=value,
                                     grade=g.grade if g else "", tooltip="\n".join(t for t in tooltip if t),
                                     checked=path in self._checked))
        self.chart.set_points(points, axis, self._chart_lines(key, paths))

    def _chart_lines(self, key, paths):
        """Reference lines for the plotted measurement: the median and the
        Marginal / Reject limits from the thresholds. None when the measurement's
        median differs by filter and more than one is shown."""
        s = self._grade_settings()
        if key == "score":
            return []  # the grade colors already say it
        if key == "signal":
            if not s.signal_drop_pct:
                return [(100, "Median", "median")]  # grading by signal is off - it only ranks
            return [(100, "Median", "median"), (100 - s.signal_drop_pct / 2, "Marginal", "marginal"),
                    (100 - s.signal_drop_pct, "Reject", "reject")]
        if key in ("stars", "flux", "sky"):  # plotted as % of each filter's median
            if key == "sky":  # a bright sky alone is Marginal at most
                return [(100, "Median", "median"), (100 + s.background_rise_pct / 2, "Marginal", "marginal")]
            drop = s.star_drop_pct if key == "stars" else s.flux_drop_pct
            return [(100, "Median", "median"), (100 - drop / 2, "Marginal", "marginal"),
                    (100 - drop, "Reject", "reject")]
        medians = [self._grades[p].medians for p in paths if p in self._grades and self._grades[p].medians]
        if len({self._groups[p] for p in paths if p in self._grades and self._grades[p].medians}) != 1:
            return None if medians else []
        med = medians[0]
        if key == "fwhm" and med.get("fwhm_px"):
            m, rise = med["fwhm_px"], s.fwhm_rise_pct / 100
            return [(m, "Median", "median"), (m * (1 + rise / 2), "Marginal", "marginal"),
                    (m * (1 + rise), "Reject", "reject")]
        if key == "eccentricity" and med.get("eccentricity") is not None:
            m, rise = med["eccentricity"], s.eccentricity_rise
            return [(m, "Median", "median"), (m + rise / 2, "Marginal", "marginal"), (m + rise, "Reject", "reject")]
        if key == "outside" and med.get("outside_light"):
            m, rise = med["outside_light"] * 100, s.outside_rise_pct / 100
            return [(m, "Median", "median"), (m * (1 + rise / 2), "Marginal", "marginal"),
                    (m * (1 + rise), "Reject", "reject")]
        return []

    def _on_chart_range_selected(self, paths):
        """Frames dragged across in the graph - select them in the list."""
        self._select_paths(paths)
        for row in range(self.table.rowCount()):
            if self._row_path(row) == paths[0]:
                self.table.scrollToItem(self.table.item(row, self.COL_FILE), QAbstractItemView.PositionAtCenter)
                break
        self.status_label.setText(f"Selected {len(paths)} frame(s) from the graph - right-click the list to check them.")

    @staticmethod
    def _chart_value(key, metrics, grade):
        if metrics is None or not metrics.ok:
            return None
        relative = grade.relative if grade else {}
        if key == "score":
            return grade.score if grade and grade.grade in CHART_RANKED else None
        if key == "fwhm":
            return metrics.fwhm_px
        if key == "eccentricity":
            return metrics.eccentricity
        if key == "outside":
            return grade.outside * 100 if grade and grade.outside is not None else None
        return relative.get({"stars": "stars", "flux": "flux", "sky": "background", "signal": "signal"}[key])

    @staticmethod
    def _night_label(night_date):
        try:
            return datetime.strptime(night_date, "%Y-%m-%d").strftime("%b %d")
        except (TypeError, ValueError):
            return "No date"

    def _local_time(self, date_obs):
        """DATE-OBS (UTC) as the session's local time, for the graph's tooltips."""
        if not date_obs:
            return ""
        try:
            moment = datetime.fromisoformat(str(date_obs)[:19]).replace(tzinfo=timezone.utc)
        except ValueError:
            return str(date_obs)
        zone = self.session.get("location_timezone")
        if zone:
            try:
                import pytz
                return moment.astimezone(pytz.timezone(zone)).strftime("%b %d %H:%M")
            except Exception:
                pass
        return moment.strftime("%b %d %H:%M UTC")

    def _on_chart_metric_changed(self):
        QSettings("CosmosCollection", "CosmosCollection").setValue(GRAPH_METRIC_SETTING,
                                                                   self.chart_metric_combo.currentData())
        self._refresh_chart()

    def _on_graph_toggled(self, visible):
        self.chart_strip.setVisible(visible)
        self._save_graph_layout()
        if visible:
            self._refresh_chart()
            self._sync_chart_selection()

    def _save_graph_layout(self, *_):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue(GRAPH_VISIBLE_SETTING, self.graph_btn.isChecked())
        if self.chart_strip.isVisible():
            settings.setValue(GRAPH_SPLITTER_SETTING, self.list_splitter.saveState())

    def _sync_chart_selection(self):
        self.chart.set_selected(self._selected_paths())

    def _on_chart_frame_clicked(self, path):
        """A point was clicked - select its frame in the list (and so the preview)."""
        for row in range(self.table.rowCount()):
            if self._row_path(row) == path:
                self._select_paths([path])
                self.table.scrollToItem(self.table.item(row, self.COL_FILE), QAbstractItemView.PositionAtCenter)
                break

    def _visible_paths(self):
        key = self.group_combo.currentData()
        return [p for p in self._lights if key is None or self._groups[p] == key]

    def _integration(self, paths):
        return sum(self._lights[p]["exptime_seconds"] or 0 for p in paths)

    def _refresh_summary(self):
        visible = self._visible_paths()
        parts = []
        for grade in (FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_REJECT,
                      FrameQuality.GRADE_UNGRADED, FrameQuality.GRADE_ERROR):
            paths = [p for p in visible if self._grades.get(p) and self._grades[p].grade == grade]
            if paths:
                color = COLORS[GRADE_COLOR_KEYS[grade]]
                parts.append(f"<span style='color:{color};'><b>{len(paths)}</b> {grade}</span> "
                             f"({format_duration(self._integration(paths))})")
        waiting = sum(1 for p in visible if p not in self._grades)
        if waiting:
            parts.append(f"{waiting} waiting")
        self.summary_label.setText(f"{len(visible)} lights: " + " · ".join(parts))

        checked = [p for p in visible if p in self._checked]
        self.checked_label.setText(f"{len(checked)} checked ({format_duration(self._integration(checked))})"
                                   if checked else "None checked")
        self._update_action_buttons()
        self._refresh_guide()
        if hasattr(self, "chart") and self.graph_btn.isChecked():
            self.chart.set_checked(self._checked)  # redraws only if the checks changed

    def _refresh_guide(self):
        """The next-step line: what the results mean and what to do, for the
        whole session - the same frames Stack Lights hands off."""
        if not self._lights:
            self.guide_label.setText("This session has no light frames attached.")
            return
        if self._analyzing:
            self.guide_label.setText(f"Analyzing {len(self._lights)} frames - grades appear as each one finishes.")
            return
        grades = {p: self._grades[p] for p in self._lights if p in self._grades}
        of = lambda grade: [p for p, g in grades.items() if g.grade == grade]
        plural = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"
        rejects = of(FrameQuality.GRADE_REJECT)
        errors = of(FrameQuality.GRADE_ERROR)
        missing = [p for p in errors if self._metrics.get(p) and self._metrics[p].error == "File not found"]
        unreadable = len(errors) - len(missing)
        ungraded = of(FrameQuality.GRADE_UNGRADED)
        trails = sum(1 for p in grades if self._metrics.get(p) and self._metrics[p].trails)

        parts = []
        waiting = len(self._lights) - len(grades)  # analysis was stopped part way
        if waiting:
            parts.append(f"<b>{plural(waiting, 'frame')} {'isn' if waiting == 1 else 'aren'}'t analyzed yet</b> "
                         "- reopen the review to finish them.")
        if rejects:
            lead = f"<b>{plural(len(rejects), 'frame')} {'has' if len(rejects) == 1 else 'have'} problems</b>"
            if all(p in self._checked for p in rejects):
                parts.append(f"{lead} and {'is' if len(rejects) == 1 else 'are'} checked - "
                             "Remove from Session or Move them, then stack.")
            else:
                parts.append(f"{lead} - check them (Check: Rejects), then Remove from Session or Move them.")
        elif grades:
            parts.append("<b>No problem frames</b> - ready to stack.")
        if missing:
            parts.append(f"{plural(len(missing), 'file')} {'is' if len(missing) == 1 else 'are'} missing "
                         "on disk - remove them from the session.")
        if unreadable:
            parts.append(f"{plural(unreadable, 'frame')} couldn't be analyzed - see Issues.")
        if ungraded:
            parts.append(f"{plural(len(ungraded), 'frame')} couldn't be graded - too few in their filter "
                         "group to compare.")
        if trails:
            parts.append(f"Satellite trails in {plural(trails, 'frame')} - stacking removes these.")

        # Good + Marginal: Marginal frames are only slightly worse and still add signal
        suggested = self._stack_paths(LIGHTS_GOOD_MARGINAL)
        if suggested:
            parts.append(f"<b>Suggested:</b> <a href='stack' style='color:{COLORS['link']};'>stack "
                         f"{LIGHT_CHOICE_LABELS[LIGHTS_GOOD_MARGINAL]} ({plural(len(suggested), 'frame')}, "
                         f"{format_duration(self._integration(suggested))})</a>")
        self.guide_label.setText("  ".join(parts))

    def _update_action_buttons(self):
        # Not mid-analysis: the worker could be reading the very file being moved or deleted
        idle = not self._analyzing and not self._file_op_running
        self.stack_btn.setEnabled(idle and bool(self._lights))
        enabled = bool(self._action_paths()) and idle
        for button in (self.remove_btn, self.move_btn, self.delete_btn):
            button.setEnabled(enabled)
            button.setToolTip(button.toolTip().split("\n")[0]
                              + ("\nAvailable once the analysis has finished or been stopped." if self._analyzing else ""))

    # ---- Checking -------------------------------------------------------

    def _check_grades(self, grades, touched=False):
        visible = set(self._visible_paths())
        self._checked = {p for p in self._checked if p not in visible}
        self._checked |= {p for p in visible if self._grades.get(p) and self._grades[p].grade in grades}
        if touched:
            self._checks_touched = True
        self._sync_checkboxes()

    def _set_checked(self, paths, checked):
        self._checked = (self._checked | set(paths)) if checked else (self._checked - set(paths))
        self._checks_touched = True
        self._sync_checkboxes()

    def _sync_checkboxes(self):
        self.table.setSortingEnabled(False)  # re-sorting mid-update would shuffle rows under the loop
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_CHECK)
            if item:
                checked = item.data(Qt.UserRole) in self._checked
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                item.setData(SortItem.SORT_ROLE, 1 if checked else 0)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        self._refresh_summary()

    def _on_item_changed(self, item):
        if item.column() != self.COL_CHECK:
            return
        path = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._checked.add(path)
        else:
            self._checked.discard(path)
        self._checks_touched = True
        self.table.blockSignals(True)
        item.setData(SortItem.SORT_ROLE, 1 if path in self._checked else 0)
        self.table.blockSignals(False)
        self._refresh_summary()

    # ---- Actions on checked frames --------------------------------------

    def _action_paths(self):
        """The checked frames the summary counts - those in the filter shown."""
        return [p for p in self._visible_paths() if p in self._checked]

    def _describe(self, paths):
        return f"{len(paths)} frame(s) ({format_duration(self._integration(paths))} of integration)"

    def _other_sessions(self, paths):
        """{session_id: (label, [file ids])} for other sessions these files are also attached to."""
        others = {}
        paths = list(paths)
        try:
            with self.db_manager.get_connection() as conn:
                for i in range(0, len(paths), 500):
                    chunk = paths[i:i + 500]
                    rows = conn.execute(
                        f"SELECT f.session_id, f.id, s.dso_name, s.session_date FROM usersessionfiles f "
                        f"JOIN usersessions s ON s.id = f.session_id "
                        f"WHERE f.session_id != ? AND f.file_path IN ({','.join('?' * len(chunk))})",
                        [self.session["id"]] + chunk).fetchall()
                    for session_id, file_id, name, date in rows:
                        others.setdefault(session_id, (f"{name} ({date})", []))[1].append(file_id)
        except Exception as e:
            logger.error(f"Error checking other sessions for files: {e}")
        return others

    def _other_sessions_note(self, others, moving):
        if not others:
            return ""
        count = sum(len(ids) for _label, ids in others.values())
        names = ", ".join(label for label, _ids in list(others.values())[:3]) + (", ..." if len(others) > 3 else "")
        which = f"{count} of them {'is' if count == 1 else 'are'} also attached to {names}"
        if moving:
            return f"\n\n{which}, which will keep {'it' if count == 1 else 'them'} at the new location."
        return f"\n\n{which}, and will be removed from there too."

    def _remove_checked(self):
        paths = self._action_paths()
        if not paths:
            return
        if QMessageBox.question(
                self, "Remove from Session",
                f"Remove {self._describe(paths)} from this session?\n\n"
                "The files stay where they are on disk - only the session forgets them.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if self._record_removed({p: p for p in paths}, mode=None):
            self.status_label.setText(f"Removed {len(paths)} frame(s) from the session.")

    def _move_checked(self):
        paths = self._action_paths()
        if not paths:
            return
        folder = SessionFileScanner.REJECTED_FOLDER
        box = QMessageBox(self)
        box.setWindowTitle("Move Frames")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Move {self._describe(paths)} and remove them from this session?")
        box.setInformativeText(
            f"A '{folder}' folder next to each frame keeps them close by, and attaching that night "
            f"again skips it." + self._other_sessions_note(self._other_sessions(paths), moving=True))
        beside_btn = box.addButton(f"Into '{folder}' Folders", QMessageBox.AcceptRole)
        choose_btn = box.addButton("Choose a Folder...", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(beside_btn)
        box.exec()
        if box.clickedButton() is beside_btn:
            destination = None
        elif box.clickedButton() is choose_btn:
            destination = QFileDialog.getExistingDirectory(self, "Move Frames To")
            if not destination:
                return
        else:
            return
        self._run_file_operation(FileOperationWorker.MOVE, paths, destination)

    def _delete_checked(self):
        paths = self._action_paths()
        if not paths:
            return
        size = 0
        for path in paths:
            try:
                size += os.path.getsize(path)
            except OSError:
                pass
        box = QMessageBox(self)
        box.setWindowTitle("Delete Frames Permanently")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"Permanently delete {self._describe(paths)}, {size / 1e9:.1f} GB?")
        box.setInformativeText(
            "The files are deleted from disk - not moved to the Recycle Bin - and can't be recovered. "
            "They're also removed from this session."
            + self._other_sessions_note(self._other_sessions(paths), moving=False))
        confirm = QCheckBox("I understand these files can't be recovered")
        box.setCheckBox(confirm)
        delete_btn = box.addButton(f"Delete {len(paths)} Files", QMessageBox.DestructiveRole)
        delete_btn.setEnabled(False)
        confirm.toggled.connect(delete_btn.setEnabled)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(cancel_btn)
        box.exec()
        if box.clickedButton() is not delete_btn:
            return
        self._run_file_operation(FileOperationWorker.DELETE, paths)

    def _run_file_operation(self, mode, paths, destination=None):
        verb = "Moving" if mode == FileOperationWorker.MOVE else "Deleting"
        self._op_progress = QProgressDialog(f"{verb} {len(paths)} frame(s)...", "Stop", 0, len(paths), self)
        self._op_progress.setWindowTitle(f"{verb} Frames")
        self._op_progress.setWindowModality(Qt.WindowModal)
        self._op_progress.setMinimumDuration(400)
        self._file_op = FileOperationWorker(mode, paths, destination)
        self._file_op.progress.connect(lambda done, _total: self._op_progress.setValue(done))
        self._file_op.done.connect(lambda finished, failed: self._on_file_operation_done(mode, finished, failed))
        self._op_progress.canceled.connect(self._file_op.cancel)
        self._file_op_running = True
        self._update_action_buttons()
        self._file_op.start()

    def _on_file_operation_done(self, mode, finished, failed):
        self._file_op_running = False
        self._op_progress.close()
        recorded = bool(finished) and self._record_removed(finished, mode)
        if recorded:
            if mode == FileOperationWorker.MOVE:
                where = (f"'{SessionFileScanner.REJECTED_FOLDER}' folders" if self._file_op.destination is None
                         else self._file_op.destination)
                self.status_label.setText(f"Moved {len(finished)} frame(s) into {where} and removed them from the session.")
            else:
                self.status_label.setText(f"Deleted {len(finished)} frame(s) and removed them from the session.")
        if failed:
            lines = "\n".join(f"{os.path.basename(p)}: {error}" for p, error in list(failed.items())[:10])
            more = f"\n...and {len(failed) - 10} more" if len(failed) > 10 else ""
            QMessageBox.warning(self, "Some Frames Weren't Changed",
                                f"{len(failed)} frame(s) couldn't be {'moved' if mode == FileOperationWorker.MOVE else 'deleted'} "
                                f"and are still in the session:\n\n{lines}{more}")
        self._update_action_buttons()

    def _record_removed(self, changed, mode):
        """Detach frames from the session - changed maps each path to its new
        path when moved, or None when deleted - and update other sessions and
        the saved measurements to match. Returns False if the database update failed."""
        session_id = self.session["id"]
        # Looked up before the transaction, so it only holds the writes
        others =self._other_sessions(changed) if mode == FileOperationWorker.DELETE else {}
        try:
            with self.db_manager.get_connection() as conn:
                SessionObservations.remove_files(conn, session_id, [self._lights[p]["id"] for p in changed])
                if mode == FileOperationWorker.MOVE:
                    for old, new in changed.items():
                        conn.execute("UPDATE usersessionfiles SET file_path = ? WHERE file_path = ? AND session_id != ?",
                                     (new, old, session_id))
                        conn.execute("UPDATE OR REPLACE framequality SET file_path = ? WHERE file_path = ?", (new, old))
                elif mode == FileOperationWorker.DELETE:
                    for other_id, (_label, ids) in others.items():
                        SessionObservations.remove_files(conn, other_id, ids)
                    paths = list(changed)
                    for i in range(0, len(paths), 500):
                        chunk = paths[i:i + 500]
                        conn.execute(f"DELETE FROM framequality WHERE file_path IN ({','.join('?' * len(chunk))})", chunk)
                conn.commit()
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error removing reviewed frames from session {session_id}: {e}")
            done = {None: "", FileOperationWorker.MOVE: " The files were already moved.",
                    FileOperationWorker.DELETE: " The files were already deleted."}[mode]
            QMessageBox.critical(self, "Error", f"Failed to update the session: {e}{done}")
            return False

        self.files_changed = True
        for path in changed:
            for store in (self._lights, self._groups, self._metrics, self._grades, self._inspections):
                store.pop(path, None)
            self._checked.discard(path)
        if self._shown_path in changed:
            self._shown_path = None
            self.preview_view.set_image(None, "Select a frame to preview it")
            for tile in self.crop_tiles:
                tile.set_image(None)
            self.details_label.setText("")
        emptied = self._fill_group_combo()  # drops filters with no frames left
        self._regrade()
        if emptied is not None:
            # After the caller's own status message, which would otherwise replace this
            QTimer.singleShot(0, lambda: self.status_label.setText(
                f"{self.status_label.text()}  No {emptied[0]} frames left - showing "
                f"{self.group_combo.currentText()}."))
        return True

    # ---- Stacking -------------------------------------------------------

    STACK_CHOICES = (LIGHTS_GOOD, LIGHTS_GOOD_MARGINAL, LIGHTS_TOP, LIGHTS_PICKED)

    def _stack_paths(self, choice):
        """The lights a Stack choice takes - the grade-based picks span every
        filter, not just the one shown."""
        usable = [p for p in self._lights
                  if not (p in self._grades and self._grades[p].grade == FrameQuality.GRADE_ERROR)]
        if choice == LIGHTS_TOP:
            keep = top_percent_lights(self._lights, self._grades, saved_top_percent())
            return [p for p in usable if p in keep]
        if choice == LIGHTS_PICKED:
            return [p for p in usable if p not in self._checked]
        return [p for p in usable if p in self._grades and self._grades[p].grade in KEEP_GRADES[choice]]

    def _stack_text(self, choice):
        label = "All except checked" if choice == LIGHTS_PICKED else light_choice_label(choice, saved_top_percent())
        paths = self._stack_paths(choice)
        return f"{label} - {len(paths)} frames, {format_duration(self._integration(paths))}", bool(paths)

    def _fill_stack_menu(self):
        import ProcessingHandoff as handoff
        menu = self.stack_btn.menu()
        menu.clear()
        settings = QSettings("CosmosCollection", "CosmosCollection")
        apps = [app for app in (handoff.SIRIL, handoff.PIXINSIGHT)
                if settings.value(f"{app}_integration_enabled", False, type=bool) and handoff.integration_status(app)[0]]
        if not apps:
            menu.addAction("Set up Siril or PixInsight under Settings → Integrations first").setEnabled(False)
            return

        # The Top % choice's percent, set right in the menu
        percent_row = QWidget()
        row = QHBoxLayout(percent_row)
        row.setContentsMargins(8, 4, 8, 4)
        row.addWidget(QLabel("Top"))
        percent_spin = QSpinBox()
        percent_spin.setRange(1, 100)
        percent_spin.setSuffix("%")
        percent_spin.setValue(saved_top_percent())
        percent_spin.setToolTip("The best frames of each filter by quality score")
        row.addWidget(percent_spin)
        row.addWidget(QLabel("of each filter, by quality score"))
        row.addStretch()
        percent_action = QWidgetAction(menu)
        percent_action.setDefaultWidget(percent_row)
        menu.addAction(percent_action)
        menu.addSeparator()

        entries = {}  # choice -> the action whose text shows its frame count
        for choice in self.STACK_CHOICES:
            text, enabled = self._stack_text(choice)
            if len(apps) == 1:
                action = menu.addAction(f"{text}  →  {handoff.APP_LABELS[apps[0]]}...")
                action.triggered.connect(lambda _=False, a=apps[0], c=choice: self._stack(a, c))
                entries[choice] = action
            else:
                submenu = menu.addMenu(text)
                for app in apps:
                    submenu.addAction(f"{handoff.APP_LABELS[app]}...").triggered.connect(
                        lambda _=False, a=app, c=choice: self._stack(a, c))
                entries[choice] = submenu.menuAction()
            entries[choice].setEnabled(enabled)

        def on_percent_changed(value):
            save_top_percent(value)
            text, enabled = self._stack_text(LIGHTS_TOP)
            action = entries[LIGHTS_TOP]
            action.setText(f"{text}  →  {handoff.APP_LABELS[apps[0]]}..." if len(apps) == 1 else text)
            action.setEnabled(enabled)
        percent_spin.valueChanged.connect(on_percent_changed)

    def _stack(self, app, choice):
        """Close the review and open the processing handoff with these lights. It
        opens over the main window: a Siril run outlives dialogs it was started from."""
        from SessionCompletion import SessionCompletionDialog
        session = dict(self.session)
        try:
            with self.db_manager.get_connection() as conn:
                row = conn.execute("SELECT * FROM usersessions WHERE id = ?", (session["id"],)).fetchone()
            if row:
                session.update(dict(row))  # current totals - frames may have been removed here
        except Exception as e:
            logger.warning(f"Couldn't reload session {session['id']} for stacking: {e}")
        parent = self.parentWidget()
        while isinstance(parent, QDialog) and parent.parentWidget() is not None:
            parent = parent.parentWidget()
        # Worked out now, so it matches the current percent and checks
        paths = self._stack_paths(choice) if choice == LIGHTS_PICKED else None
        self.accept()
        SessionCompletionDialog(session, parent=parent, app=app, light_choice=choice, light_paths=paths).exec()

    # ---- Selection, preview and menu ------------------------------------

    def _selected_paths(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [p for p in (self._row_path(r) for r in rows) if p]

    def _select_paths(self, paths):
        wanted = set(paths)
        model = self.table.model()
        selection = QItemSelection()
        for row in range(self.table.rowCount()):
            if self._row_path(row) in wanted:
                selection.select(model.index(row, 0), model.index(row, model.columnCount() - 1))
        self.table.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def _load_preview(self):
        paths = self._selected_paths()
        if len(paths) != 1:
            self.details_label.setText("")
            return
        path = paths[0]
        self.details_label.setText(self._details_text(path))
        if path == self._shown_path:
            return
        if path in self._inspections:
            self._inspections.move_to_end(path)
            self._show_inspection(path, self._inspections[path])
            return
        if self._preview is not None and self._preview.isRunning() and self._preview.path == path:
            return
        _retire_thread(self._preview)
        # The previous frame stays up (keeping the zoom) until this one is ready
        self.preview_view.show_loading(True)
        self._preview = InspectionLoader(path)
        self._preview.loaded.connect(self._on_preview_loaded)
        self._preview.start()

    def _on_preview_loaded(self, path, result):
        if result is not None:
            self._inspections[path] = result
            while len(self._inspections) > 8:  # ~7 MB each for a 26 MP color camera
                self._inspections.popitem(last=False)
        if self._selected_paths() != [path]:
            return
        if result is None:
            self._shown_path = None
            self.preview_view.set_image(None, "Could not load this frame")
            for tile in self.crop_tiles:
                tile.set_image(None)
        else:
            self._show_inspection(path, result)

    def _show_inspection(self, path, result):
        self._shown_path = path
        self.preview_view.set_image(result["overview"])
        for tile, (image, center) in zip(self.crop_tiles, result["crops"]):
            tile.set_image(image, center)

    def _zoom_to_crop(self, center):
        self.preview_view.set_zoom(max(2.0, self.preview_view.zoom()), center)

    def _update_zoom_label(self):
        self.zoom_label.setText(f"{self.preview_view.zoom() * 100:.0f}%" if self.preview_view.has_image() else "")

    def _details_text(self, path):
        m = self._metrics.get(path)
        g = self._grades.get(path)
        lines = [f"<b>{os.path.basename(path)}</b>"]
        if m is None:
            lines.append("Not analyzed yet.")
        elif not m.ok:
            lines.append(f"Could not analyze: {m.error}")
        else:
            rel = g.relative if g else {}

            def pct(name):
                return f" ({rel[name]:.0f}% of median)" if name in rel else ""
            fwhm = (f"{m.fwhm_px:.2f} px" + (f' / {m.fwhm_arcsec:.2f}"' if m.fwhm_arcsec else "")
                    if m.fwhm_px is not None else "-")
            lines += [
                f"Stars: {m.stars:,}{pct('stars')} · {m.measured:,} measured · {m.saturated} saturated",
                f"FWHM: {fwhm}{pct('fwhm')} · HFR: {m.hfr_px:.2f} px" if m.hfr_px is not None else f"FWHM: {fwhm}",
                f"Eccentricity: {m.eccentricity:.2f} · alignment {m.alignment:.2f}" if m.eccentricity is not None else "",
                f"Light outside the star cores: {g.outside * 100:.1f}%{pct('outside')}"
                if g and g.outside is not None else "",
                f"Star brightness: {rel['flux']:.0f}% of median" if "flux" in rel else "",
                f"Signal (star brightness vs noise): {rel['signal']:.0f}% of median" if "signal" in rel else "",
                f"Sky: {m.background:.0f} ADU{pct('background')} · noise {m.noise:.1f} ADU · gradient {m.gradient_pct:.1f}%"
                if m.background is not None else "",
                f"Satellite/plane trails: {len(m.trails)}" if m.trails else "",
            ]
            if g:
                color = COLORS[GRADE_COLOR_KEYS.get(g.grade, 'text_secondary')]
                lines.append(f"<span style='color:{color};'><b>{g.grade}</b></span>"
                             + (f" - {g.score:.0f}/100" if g.grade in GRADE_SORT and GRADE_SORT[g.grade] < 3 else ""))
                lines += g.flags
        return "<br>".join(line for line in lines if line)

    def _show_table_menu(self, position):
        item = self.table.itemAt(position)
        if not item:
            return
        if not self.table.item(item.row(), self.COL_CHECK).isSelected():
            self.table.selectRow(item.row())
        paths = self._selected_paths()
        menu = QMenu(self)
        menu.addAction(f"Check {len(paths)} Selected").triggered.connect(lambda: self._set_checked(paths, True))
        menu.addAction(f"Uncheck {len(paths)} Selected").triggered.connect(lambda: self._set_checked(paths, False))
        menu.addSeparator()
        menu.addAction("Open in Image Viewer").triggered.connect(lambda: self._open_viewer(paths[0]))
        menu.addAction("Open Containing Folder").triggered.connect(lambda: self._open_folder(paths[0]))
        menu.addAction("Copy File Path" if len(paths) == 1 else f"Copy {len(paths)} File Paths").triggered.connect(
            lambda: self._copy_paths(paths))
        menu.exec(self.table.viewport().mapToGlobal(position))

    def _copy_paths(self, paths):
        """The paths on the clipboard, one per line, with the platform's separators."""
        QGuiApplication.clipboard().setText("\n".join(os.path.normpath(p) for p in paths))
        self.status_label.setText(f"Copied {len(paths)} file path(s) to the clipboard.")

    def _on_item_double_clicked(self, item):
        # A double-click on the checkbox just toggles it twice - not an "open"
        if item.column() != self.COL_CHECK:
            self._open_viewer(self._row_path(item.row()))

    def _open_viewer(self, path):
        """The frame at full resolution, screen-stretched, in the image viewer."""
        if not path:
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "File Not Found", f"The file no longer exists:\n{path}")
            return
        from ImageViewer import ImageViewerWindow
        # Opens straight away and loads in the background; a child of this dialog,
        # so it stays usable while the (modal) review is open
        viewer = ImageViewerWindow(None, os.path.basename(path), path, self,
                                   dso_ra=self.session.get("ra_deg"), dso_dec=self.session.get("dec_deg"),
                                   screen_stretch=True)
        self._viewers = [v for v in getattr(self, "_viewers", []) if v.isVisible()] + [viewer]
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()

    def _open_folder(self, path):
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Folder Not Found", f"The folder no longer exists:\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ---- Closing --------------------------------------------------------

    def done(self, result):
        self._save_graph_layout()
        # Keep the measurements made so far; let running threads finish unseen
        if self._worker is not None:
            self._worker.cancel()
        self._flush_cache()
        _retire_thread(self._worker)
        _retire_thread(self._preview)
        super().done(result)
