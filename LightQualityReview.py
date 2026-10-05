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
import time
import shutil
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PySide6.QtCore import (Qt, QItemSelection, QItemSelectionModel, QPointF, QSettings, QThread, QTimer, QUrl,
                            Signal)
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QLabel,
                               QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QSplitter, QGroupBox, QDoubleSpinBox, QProgressBar,
                               QMenu, QMessageBox, QSizePolicy, QGraphicsView, QGraphicsScene, QToolButton,
                               QWidgetAction, QScrollArea, QCheckBox, QFileDialog, QProgressDialog)

import FrameQuality
import SessionFileScanner
import SessionObservations
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS, font_size, themed_style
from SessionManager import format_duration, format_exposure, _retire_thread, _rollback

logger = logging.getLogger(__name__)

# Parallel file reads + analyses.
ANALYSIS_WORKERS = 4

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
LIGHTS_ALL, LIGHTS_GOOD, LIGHTS_GOOD_MARGINAL, LIGHTS_PICKED = "all", "good", "good_marginal", "picked"
LIGHT_CHOICE_LABELS = {
    LIGHTS_ALL: "All lights",
    LIGHTS_GOOD: "Good only",
    LIGHTS_GOOD_MARGINAL: "Good + Marginal",
    LIGHTS_PICKED: "Picked in the quality review",
}
# Frames in a group too small to compare can't be judged, so they're kept
KEEP_GRADES = {
    LIGHTS_GOOD: {FrameQuality.GRADE_GOOD, FrameQuality.GRADE_UNGRADED},
    LIGHTS_GOOD_MARGINAL: {FrameQuality.GRADE_GOOD, FrameQuality.GRADE_MARGINAL, FrameQuality.GRADE_UNGRADED},
}


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
    """(lights, {path: grade}) for a session from its saved measurements and the
    saved thresholds. Lights never analyzed have no grade; the grades are empty
    when none has been analyzed."""
    lights = session_light_files(conn, session_id)
    ensure_cache_table(conn)
    metrics = {path: m for path, (_size, _mtime, version, m) in load_cached(conn, lights).items()
               if version == FrameQuality.ANALYSIS_VERSION}
    if not metrics:
        return lights, {}
    grades = FrameQuality.grade_frames(((group_key(lights[p]), m) for p, m in metrics.items()),
                                       saved_grade_settings())
    return lights, {path: g.grade for path, g in grades.items()}


# ---- Workers --------------------------------------------------------------------

class AnalysisWorker(QThread):
    """Checks each light against its cached measurements and analyzes the new or
    changed ones, several at a time."""
    checked = Signal(int)                 # how many frames need analyzing
    frame_ready = Signal(object, object)  # FrameMetrics, (size, mtime_ns) to cache it under - or None

    def __init__(self, paths, cached, force=False):
        super().__init__()
        self.paths = paths
        self.cached = cached
        self.force = force
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
        pool = ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS)
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


# ---- Dialog ---------------------------------------------------------------------

class LightQualityDialog(WindowPositionMixin, QDialog):
    """Analyze and grade one session's light frames."""

    WINDOW_POSITION_KEY = "LightQualityDialog"
    COLUMNS = ["", "Grade", "Score", "File", "Night", "Filter", "Stars", "FWHM", "Eccentricity",
               "Tails/Halos", "Star Brightness", "Sky", "Issues"]
    (COL_CHECK, COL_GRADE, COL_SCORE, COL_FILE, COL_NIGHT, COL_FILTER, COL_STARS, COL_FWHM, COL_ECC,
     COL_OUTSIDE, COL_FLUX, COL_SKY, COL_ISSUES) = range(13)

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

    def _grade_settings(self):
        # Same values saved_grade_settings() reads - each change is saved as it's made
        return FrameQuality.GradeSettings(**{name: spin.value() for name, spin in self.threshold_spins.items()})

    # ---- UI -------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel("Each frame is compared with the others of the same filter, exposure and binning. "
                       "Grades update instantly when you change the thresholds - no re-analysis needed.")
        intro.setWordWrap(True)
        themed_style(intro, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(intro)

        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        self.group_combo = QComboBox()
        self.group_combo.addItem("All filters", None)
        for key in sorted(set(self._groups.values()), key=lambda k: (str(k[0]), k[1] or 0, k[2])):
            self.group_combo.addItem(self._group_label(key), key)
        self.group_combo.currentIndexChanged.connect(self._apply_group_filter)
        top.addWidget(self.group_combo)
        top.addWidget(self._build_thresholds_button())
        top.addSpacing(16)
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.RichText)
        top.addWidget(self.summary_label, 1)
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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_table())
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
        self.stack_btn = QToolButton()
        self.stack_btn.setText("Stack")
        self.stack_btn.setToolTip("Hand the chosen lights to Siril or PixInsight (closes this review)")
        self.stack_btn.setPopupMode(QToolButton.InstantPopup)
        self.stack_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        stack_menu = QMenu(self.stack_btn)
        stack_menu.aboutToShow.connect(self._fill_stack_menu)
        self.stack_btn.setMenu(stack_menu)
        bottom.addWidget(self.stack_btn)
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
        for col, tip in ((self.COL_STARS, "Stars detected"),
                         (self.COL_FWHM, "Star size (full width at half maximum), in sensor pixels and arcseconds"),
                         (self.COL_ECC, "0 = round, 1 = a line. Elongation that all points one way is trailing."),
                         (self.COL_OUTSIDE, "Star light outside the round star cores vs the group's median frame.\n"
                                            "A guiding jump leaves round cores with faint tails, which only this shows;\n"
                                            "double images and dew or cloud halos raise it too."),
                         (self.COL_FLUX, "Brightness of the same stars, as % of the group's median frame.\n"
                                         "Clouds and haze dim it; moonlight and seeing don't."),
                         (self.COL_SKY, "Sky background vs the group's median frame")):
            self.table.horizontalHeaderItem(col).setToolTip(tip)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(lambda: self._preview_timer.start(150))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)
        return self.table

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
        note = QLabel("Half of each is Marginal. A brighter sky alone is Marginal at most, and fewer stars "
                      "only count on their own - moonlight hides faint stars too. The median leaves out "
                      "rejected frames, so removing them doesn't turn up new rejects.")
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
        self._worker = AnalysisWorker(list(self._lights), self._cached, force=force)
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
                            (self.COL_OUTSIDE, "+888%"), (self.COL_FLUX, "100%"), (self.COL_SKY, "+888%")):
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
        if ok and m.outside_light is not None:
            outside_item.setToolTip(f"{m.outside_light * 100:.1f}% of the star light is outside the cores")
        self.table.setItem(row, self.COL_OUTSIDE, outside_item)

        flux = relative.get("flux")
        self.table.setItem(row, self.COL_FLUX, SortItem(f"{flux:.0f}%" if flux is not None else "",
                                                       flux if flux is not None else -1))
        sky = relative.get("background")
        sky_item = SortItem(f"{sky - 100:+.0f}%" if sky is not None else "", sky if sky is not None else -1)
        if ok and m.background is not None:
            sky_item.setToolTip(f"{m.background:.0f} ADU")
        self.table.setItem(row, self.COL_SKY, sky_item)

        issues = "; ".join(g.flags) if g else ("Analyzing..." if not m else "")
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
        self._regrade()
        return True

    # ---- Stacking -------------------------------------------------------

    def _stack_choices(self):
        """[(choice, label, paths)] - the grade-based picks span every filter, not just the one shown."""
        graded = {p: g.grade for p, g in self._grades.items()}
        usable = [p for p in self._lights if graded.get(p) != FrameQuality.GRADE_ERROR]
        return [
            (LIGHTS_GOOD, LIGHT_CHOICE_LABELS[LIGHTS_GOOD],
             [p for p in usable if graded.get(p) in KEEP_GRADES[LIGHTS_GOOD]]),
            (LIGHTS_GOOD_MARGINAL, LIGHT_CHOICE_LABELS[LIGHTS_GOOD_MARGINAL],
             [p for p in usable if graded.get(p) in KEEP_GRADES[LIGHTS_GOOD_MARGINAL]]),
            (LIGHTS_PICKED, "All except checked", [p for p in usable if p not in self._checked]),
        ]

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
        for choice, label, paths in self._stack_choices():
            text = f"{label} - {len(paths)} frames, {format_duration(self._integration(paths))}"
            if len(apps) == 1:
                action = menu.addAction(f"{text}  →  {handoff.APP_LABELS[apps[0]]}...")
                action.triggered.connect(lambda _=False, a=apps[0], c=choice, p=paths: self._stack(a, c, p))
                action.setEnabled(bool(paths))
            else:
                submenu = menu.addMenu(text)
                submenu.setEnabled(bool(paths))
                for app in apps:
                    submenu.addAction(f"{handoff.APP_LABELS[app]}...").triggered.connect(
                        lambda _=False, a=app, c=choice, p=paths: self._stack(a, c, p))

    def _stack(self, app, choice, paths):
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
        self.accept()
        SessionCompletionDialog(session, parent=parent, app=app, light_choice=choice,
                                light_paths=paths if choice == LIGHTS_PICKED else None).exec()

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
                f"Light outside the star cores: {m.outside_light * 100:.1f}%{pct('outside')}"
                if m.outside_light is not None else "",
                f"Star brightness: {rel['flux']:.0f}% of median" if "flux" in rel else "",
                f"Sky: {m.background:.0f} ADU{pct('background')} · noise {m.noise:.1f} ADU · gradient {m.gradient_pct:.1f}%"
                if m.background is not None else "",
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
        menu.addAction("Open Containing Folder").triggered.connect(lambda: self._open_folder(paths[0]))
        menu.exec(self.table.viewport().mapToGlobal(position))

    def _open_folder(self, path):
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Folder Not Found", f"The folder no longer exists:\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ---- Closing --------------------------------------------------------

    def done(self, result):
        # Keep the measurements made so far; let running threads finish unseen
        if self._worker is not None:
            self._worker.cancel()
        self._flush_cache()
        _retire_thread(self._worker)
        _retire_thread(self._preview)
        super().done(result)
