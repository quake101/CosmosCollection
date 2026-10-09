#!/usr/bin/env python3
"""
NINA Dashboard Window for Cosmos Collection
Displays real-time NINA status, current imaging, live stack images, and guiding graphs.
"""

import hashlib
import logging
import math
import re
import sys
import time
import warnings
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use('QtAgg')

# Suppress matplotlib font_manager debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
# Harmless: raised when a graph in a hidden dock tab is drawn at zero size
warnings.filterwarnings('ignore', message='constrained_layout not applied')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings, QByteArray, QPointF, QRectF, QSize
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QGroupBox, QProgressBar, QComboBox, QFrame,
    QGridLayout, QSizePolicy, QDockWidget, QCheckBox, QSpinBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLineEdit,
    QTabWidget, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QTreeWidget, QTreeWidgetItem, QSplitter, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QIcon, QColor

from NINAIntegration import NINAIntegration
from WindowPositionManager import WindowPositionMixin
from Theme import (
    COLORS, adapt_color, chart_background, chart_color, chart_font_size, font_px, theme_manager,
    themed_style,
)
from TimeFormatHelper import format_time

# Set up logging
logger = logging.getLogger(__name__)

# Threads that have to outlive a stop request or their window. Dropping the last
# reference to a running QThread destroys it mid-run, which aborts the whole app.
_running_threads = set()


def keep_alive_until_finished(thread):
    """Hold a reference to a running QThread until it finishes."""
    _running_threads.add(thread)
    thread.finished.connect(lambda: _running_threads.discard(thread))


def skip_zero_quality(spin):
    """Step a -1..100 quality spin box straight between -1 (PNG) and 1: NINA
    doesn't accept a quality of 0."""
    spin.setProperty("last_quality", spin.value())

    def on_change(value):
        if value == 0:
            # Coming down from 1 -> PNG; coming up from PNG (or typed 0) -> 1
            spin.setValue(-1 if spin.property("last_quality") == 1 else 1)
            return
        spin.setProperty("last_quality", value)
    # Connected first, so the fix happens before other handlers see 0
    spin.valueChanged.connect(on_change)


def valid_size(text):
    """Whether text is an image size NINA accepts ('WxH', e.g. '1920x1080')."""
    return re.fullmatch(r'\d+x\d+', str(text).strip()) is not None


def finite_number(value):
    """value if it's a real, finite number, else None.

    NINA sends missing readings (e.g. the temperature of a focuser without a
    probe) as the string "NaN".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def format_image_stat(value, decimals=None, allow_negative=True):
    """Format an image-history statistic, returning "--" when it wasn't measured.

    Non-LIGHT frames (e.g. flats) report Stars as -1 and HFR as NaN, which NINA
    serializes as the string "NaN".
    """
    if not isinstance(value, (int, float)) or math.isnan(value) or (not allow_negative and value < 0):
        return "--"
    return f"{value:.{decimals}f}" if decimals is not None else str(int(value))


# Sequence entry statuses (NINA SequenceEntityStatus) and the COLORS key each is drawn in.
# CREATED (not run yet) uses the normal text color and shows no status text.
SEQUENCE_STATUS_COLORS = {
    'RUNNING': 'info',
    'FINISHED': 'success',
    'FAILED': 'error',
    'SKIPPED': 'text_disabled',
    'DISABLED': 'text_disabled',
}


def sequence_display_name(entry, kind):
    """Readable name for a sequence entry; kind is 'item', 'condition' or 'trigger'.

    NINA suffixes containers, triggers and conditions ('Bubble Nebula_Container'),
    and some instructions (e.g. inside the flat wizard) serialize with no name at all.
    """
    name = entry.get('Name') or ""
    for suffix in ('_Container', '_Trigger', '_Condition'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    if name:
        return name
    # Infer unnamed entries from their fields
    if kind == 'condition':
        return "Loop For Iterations" if 'Iterations' in entry else "Condition"
    if kind == 'trigger':
        return "Trigger"
    if 'Items' in entry:
        return "Container"
    if 'ExposureTime' in entry:
        return "Take Exposure"
    if 'Filter' in entry:
        return "Switch Filter"
    return "Instruction"


def sequence_children(entry):
    """(kind, child) pairs under a sequence container: conditions, triggers, then items."""
    children = [('condition', c) for c in entry.get('Conditions') or [] if isinstance(c, dict)]
    children += [('trigger', t) for t in entry.get('Triggers') or [] if isinstance(t, dict)]
    children += [('item', i) for i in entry.get('Items') or [] if isinstance(i, dict)]
    return children


def _format_timespan(text):
    """Turn a .NET TimeSpan string ('1.02:03:04.567') into 'h:mm:ss'."""
    match = re.match(r'^(-)?(?:(\d+)\.)?(\d+):(\d\d):(\d\d)', str(text))
    if not match:
        return str(text)
    sign, days, hours, minutes, seconds = match.groups()
    hours = int(hours) + int(days or 0) * 24
    return f"{sign or ''}{hours}:{minutes}:{seconds}"


def _format_hours(hours):
    """Turn fractional hours into '2h 35m'."""
    total_minutes = max(0, round(hours * 60))
    h, m = divmod(total_minutes, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _format_number(value):
    """Drop a trailing .0 so 300.0 reads as 300."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def sequence_entry_details(entry, kind):
    """One-line summary of a sequence entry's settings and progress."""
    parts = []
    if kind == 'condition':
        target_time = entry.get('TargetTime')
        if target_time:
            try:
                parts.append(f"Until {format_time(datetime.fromisoformat(target_time).astimezone())}")
            except (ValueError, TypeError):
                parts.append(f"Until {target_time}")
        if entry.get('RemainingTime'):
            parts.append(f"{_format_timespan(entry['RemainingTime'])} left")
        if 'Iterations' in entry:
            parts.append(f"{entry.get('CompletedIterations', 0)}/{entry['Iterations']} done")
    elif kind == 'trigger':
        time_to_flip = entry.get('TimeToFlip')
        if isinstance(time_to_flip, (int, float)):
            parts.append(f"Flip in {_format_hours(time_to_flip)}" if time_to_flip > 0 else "Flip due")
        if isinstance(entry.get('TargetDrift'), (int, float)):
            drift = entry.get('Drift')
            drift_text = f"{drift:.2f}" if isinstance(drift, (int, float)) else "--"
            parts.append(f"Drift {drift_text}′ of {_format_number(entry['TargetDrift'])}′")
        if isinstance(entry.get('HFRTrendPercentage'), (int, float)):
            parts.append(f"HFR trend {entry['HFRTrendPercentage']:+.1f}% "
                         f"of {_format_number(entry.get('DeltaHFR', '?'))}%")
    else:
        if isinstance(entry.get('Filter'), str):
            parts.append(entry['Filter'])
        if isinstance(entry.get('ExposureTime'), (int, float)):
            exposure = f"{_format_number(entry['ExposureTime'])}s"
            if entry.get('Type'):
                exposure += f" {entry['Type']}"
            parts.append(exposure)
        binning = entry.get('Binning')
        if isinstance(binning, dict) and binning.get('Name') and binning.get('Name') != '1x1':
            parts.append(f"Bin {binning['Name']}")
        # -1 means the camera default
        for key in ('Gain', 'Offset'):
            if isinstance(entry.get(key), (int, float)) and entry[key] >= 0:
                parts.append(f"{key} {_format_number(entry[key])}")
        if 'Iterations' in entry and 'CompletedIterations' in entry:
            parts.append(f"{entry['CompletedIterations']}/{entry['Iterations']} done")
        if entry.get('ExposureCount'):
            parts.append(f"{entry['ExposureCount']} taken")
        coords = entry.get('Coordinates')
        if isinstance(coords, dict):
            coords = coords.get('Coordinates', coords)  # Center After Drift nests them
            if coords.get('RAString') and coords.get('DecString'):
                parts.append(f"RA {coords['RAString']}  Dec {coords['DecString']}")
    if not parts:
        # Unknown (e.g. plugin) entries: show their simple settings
        parts = [f"{key}: {_format_number(value)}" for key, value in entry.items()
                 if key not in ('Name', 'Status') and isinstance(value, (str, int, float))
                 and not isinstance(value, bool)][:4]
    return " · ".join(parts)


def sequence_activity(entries):
    """Summarize what the sequence is doing from the /sequence/json entries.

    Returns a dict with 'state' ('running', 'finished' or 'idle'), 'path' (the
    chain of RUNNING entries from a top-level container down to the deepest one),
    'trigger' (a RUNNING trigger, which runs between instructions, or None),
    'loop' (the condition of the nearest running container that has one, or None)
    and 'next' (the next not-yet-run entry after the current one, or None).
    """
    containers = [e for e in entries if isinstance(e, dict) and 'Items' in e]
    path = []
    level = containers
    while True:
        running = next((e for e in level if isinstance(e, dict) and e.get('Status') == 'RUNNING'), None)
        if running is None:
            break
        path.append(running)
        level = running.get('Items') or []

    trigger = None
    global_triggers = next((e.get('GlobalTriggers') for e in entries
                            if isinstance(e, dict) and 'GlobalTriggers' in e), None) or []
    for owner_triggers in [global_triggers] + [e.get('Triggers') or [] for e in path]:
        for t in owner_triggers:
            if isinstance(t, dict) and t.get('Status') == 'RUNNING':
                trigger = t

    loop = None
    for container in reversed(path):
        conditions = [c for c in container.get('Conditions') or [] if isinstance(c, dict)]
        if conditions:
            loop = conditions[0]
            break

    # Next entry: the first not-yet-run sibling after the current entry, walking up the path
    next_entry = None
    for depth in range(len(path) - 1, -1, -1):
        siblings = containers if depth == 0 else path[depth - 1].get('Items') or []
        index = next((i for i, s in enumerate(siblings) if s is path[depth]), None)
        if index is None:
            continue
        next_entry = next((s for s in siblings[index + 1:]
                           if isinstance(s, dict) and s.get('Status') == 'CREATED'), None)
        if next_entry is not None:
            break

    if path:
        state = 'running'
    elif containers and all(c.get('Status') == 'FINISHED' for c in containers):
        state = 'finished'
    else:
        state = 'idle'
    return {'state': state, 'path': path, 'trigger': trigger, 'loop': loop, 'next': next_entry}


def nina_solution_to_wcs_header(solution, width, height):
    """Build a TAN WCS header for an image of width x height display pixels from
    a NINA plate-solve result, for AnnotationOverlay.

    NINA reports the solve relative to the image as it displays it: the field
    center, PositionAngle, Flipped and the field Radius (half the diagonal, in
    degrees). The live stack is shown the same way up, just resized, so the
    display pixel scale follows from the radius and this image's diagonal.

    This inverts NINA's ASTAPSolver/WorldCoordinateSystem conversion. NINA
    solves a FITS copy whose first row is the top of the displayed image, so
    its CD matrix has y increasing downward; from it NINA derives
        PositionAngle = 360 - (Rotation - 180), Flipped = not wcs_flipped
    (PositionAngle 0, not flipped = North up, East left). AnnotationOverlay
    uses standard FITS y (increasing upward from the bottom row), so the y
    column of the CD matrix is negated at the end.

    Returns the header dict, or None if the solution lacks what's needed.
    """
    coords = solution.get('Coordinates') or {}
    # The API docs say DECDegrees, but NINA sends Dec (already in degrees)
    ra = coords.get('RADegrees')
    dec = coords.get('DECDegrees', coords.get('Dec'))
    position_angle = solution.get('PositionAngle')
    if not all(isinstance(v, (int, float)) for v in (ra, dec, position_angle)) or width <= 0 or height <= 0:
        return None

    radius = solution.get('Radius')
    if isinstance(radius, (int, float)) and radius > 0:
        scale = 2 * radius / math.hypot(width, height)  # degrees per display pixel
    else:
        return None

    rotation = math.radians((540 - position_angle) % 360)  # NINA's wcs.Rotation
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    if not solution.get('Flipped'):
        # NINA's wcs was flipped (positive determinant)
        cd1_1, cd1_2, cd2_1, cd2_2 = scale * cos_r, -scale * sin_r, scale * sin_r, scale * cos_r
    else:
        cd1_1, cd1_2, cd2_1, cd2_2 = -scale * cos_r, -scale * sin_r, -scale * sin_r, scale * cos_r

    return {
        'CRVAL1': ra, 'CRVAL2': dec,
        'CRPIX1': (width + 1) / 2, 'CRPIX2': (height + 1) / 2,
        # y flipped from NINA's downward rows to FITS' upward rows
        'CD1_1': cd1_1, 'CD1_2': -cd1_2, 'CD2_1': cd2_1, 'CD2_2': -cd2_2,
    }


# A failed/cancelled autofocus never emits AUTOFOCUS-FINISHED; treat a run with no
# AF events for this long as ended. Points normally arrive every ~15-30 seconds.
AUTOFOCUS_STALE_SECONDS = 180


class ZoomableImageWidget(QWidget):
    """Widget that displays an image with zoom (mouse wheel) and pan (drag) support."""

    def __init__(self, parent=None, placeholder_text="No image available"):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._pan_offset = QPointF(0, 0)
        self._last_mouse_pos = None
        self._placeholder_text = placeholder_text
        self._overlay = None  # AnnotationRenderer drawn over the image, in image pixels

        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

    def setPixmap(self, pixmap):
        """Set the image to display, preserving zoom/pan if already viewing an image."""
        had_image = self._pixmap is not None and not self._pixmap.isNull()
        self._pixmap = pixmap
        if not had_image:
            self._reset_view()
        self.update()

    def setPlaceholderText(self, text):
        """Set the placeholder text shown when no image is loaded."""
        self._placeholder_text = text
        self.update()

    def pixmapSize(self):
        """Size of the displayed image (QSize; empty when there's none)."""
        return self._pixmap.size() if self._pixmap and not self._pixmap.isNull() else QSize()

    def setOverlay(self, renderer):
        """Draw an AnnotationRenderer over the image (None to remove it)."""
        self._overlay = renderer
        self.update()

    def _reset_view(self):
        """Reset zoom and pan to fit the image in the widget."""
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)

    def _get_fit_zoom(self):
        """Calculate the zoom level that fits the image in the widget."""
        if not self._pixmap or self._pixmap.isNull():
            return 1.0
        widget_size = self.size()
        pixmap_size = self._pixmap.size()
        scale_x = (widget_size.width() - 10) / pixmap_size.width()
        scale_y = (widget_size.height() - 10) / pixmap_size.height()
        return min(scale_x, scale_y, 1.0)  # Don't upscale beyond 100%

    def paintEvent(self, event):
        """Paint the image with current zoom and pan."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fill background
        painter.fillRect(self.rect(), Qt.black)

        if not self._pixmap or self._pixmap.isNull():
            # Draw placeholder text
            painter.setPen(QColor(adapt_color('#a0a0a0', on_dark=True)))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder_text)
            return

        # Calculate the effective zoom (fit zoom * user zoom)
        fit_zoom = self._get_fit_zoom()
        effective_zoom = fit_zoom * self._zoom

        # Calculate scaled image size
        scaled_width = self._pixmap.width() * effective_zoom
        scaled_height = self._pixmap.height() * effective_zoom

        # Center the image, then apply pan offset
        x = (self.width() - scaled_width) / 2 + self._pan_offset.x()
        y = (self.height() - scaled_height) / 2 + self._pan_offset.y()

        # Draw the image
        target_rect = QRectF(x, y, scaled_width, scaled_height)
        source_rect = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(target_rect, self._pixmap, source_rect)

        # Annotations follow the image's zoom and pan
        if self._overlay is not None and self._overlay.wcs is not None:
            painter.save()
            painter.setClipRect(target_rect)  # No labels for objects just outside the frame
            self._overlay.render(painter, effective_zoom, x, y)
            painter.restore()

        # Draw zoom indicator if zoomed
        if self._zoom != 1.0:
            zoom_text = f"{self._zoom * 100:.0f}%"
            painter.setPen(QColor(adapt_color('#ffffff', on_dark=True)))
            painter.drawText(10, 20, zoom_text)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if not self._pixmap or self._pixmap.isNull():
            return

        # Get zoom delta from wheel
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1 / 1.15

        # Calculate new zoom, clamped to limits
        new_zoom = self._zoom * zoom_factor
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))

        if new_zoom != self._zoom:
            # Zoom toward mouse position
            mouse_pos = event.position()
            old_zoom = self._zoom
            self._zoom = new_zoom

            # Adjust pan to zoom toward mouse position
            zoom_change = new_zoom / old_zoom
            center = QPointF(self.width() / 2, self.height() / 2)
            mouse_offset = mouse_pos - center - self._pan_offset
            self._pan_offset = self._pan_offset - mouse_offset * (zoom_change - 1)

            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on mouse press."""
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan the image on mouse drag."""
        if self._last_mouse_pos is not None and self._pixmap and not self._pixmap.isNull():
            delta = event.position() - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End panning on mouse release."""
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Reset view on double-click."""
        self._reset_view()
        self.update()

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        self.update()


class SequencePanel(QWidget):
    """Sequence dock contents: the activity summary and the sequence tree in a
    splitter, side by side when the dock is wide (e.g. in the bottom area) and
    stacked when it's tall. The activity's width (side by side) and height
    (stacked) are each remembered, so dragging one doesn't change the other."""

    DEFAULT_ACTIVITY_WIDTH = 340
    DEFAULT_ACTIVITY_HEIGHT = 190

    def __init__(self, activity, tree, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(activity)
        self._splitter.addWidget(tree)
        # The activity keeps its size as the dock resizes; the tree takes the rest
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        layout.addWidget(self._splitter)
        self._activity_size = {Qt.Horizontal: self.DEFAULT_ACTIVITY_WIDTH,
                               Qt.Vertical: self.DEFAULT_ACTIVITY_HEIGHT}
        self._sized = False

    def activity_sizes(self):
        """(width when side by side, height when stacked)."""
        return self._activity_size[Qt.Horizontal], self._activity_size[Qt.Vertical]

    def set_activity_sizes(self, width, height):
        self._activity_size[Qt.Horizontal] = width
        self._activity_size[Qt.Vertical] = height
        self._apply_activity_size()

    def _apply_activity_size(self):
        orientation = self._splitter.orientation()
        length = self._splitter.width() if orientation == Qt.Horizontal else self._splitter.height()
        length -= self._splitter.handleWidth()
        if length <= 0:
            return
        # Leave the tree some room if the dock is now smaller than the saved size
        activity = max(0, min(self._activity_size[orientation], length - 100))
        self._splitter.setSizes([activity, length - activity])
        self._sized = True

    def _on_splitter_moved(self, pos, index):
        self._activity_size[self._splitter.orientation()] = self._splitter.sizes()[0]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        orientation = Qt.Horizontal if self.width() > 1.6 * self.height() else Qt.Vertical
        if self._splitter.orientation() != orientation:
            self._splitter.setOrientation(orientation)
            self._apply_activity_size()
        elif not self._sized:
            self._apply_activity_size()


class _NINAUnreachable(Exception):
    """NINA's API isn't answering (NINA closed, or its PC is off)."""


class NINAStatusWorker(QThread):
    """Background thread for polling NINA API endpoints."""

    status_updated = Signal(dict)  # Emits combined status data
    image_updated = Signal(bytes, dict)  # Emits image data and metadata
    image_fetching = Signal(int, int)  # Emits (bytes_received, total_bytes); total_bytes=-1 if unknown
    livestack_updated = Signal(bytes, dict, list)  # Emits livestack image, status, and available stacks
    livestack_fetching = Signal(int, int)  # Emits (bytes_received, total_bytes); total_bytes=-1 if unknown
    liveview_updated = Signal(bytes)  # Emits prepared image JPEG frame data
    guiding_updated = Signal(dict)  # Emits NINA's guide graph (steps + its RMS)
    event_occurred = Signal(dict)  # Emits new NINA event data
    events_loaded = Signal(list)  # Emits recent past events once on connect (for the event log)
    autofocus_report = Signal(dict)  # Emits the last-af report when an autofocus run completes
    error_occurred = Signal(str)  # Emits error message
    connection_changed = Signal(bool, str, str, int)  # Emits connected state, version, host, port
    history_thumbnail = Signal(int, bytes, dict)  # Emits (index, small_thumbnail_data, image_stats)
    history_reset = Signal()  # NINA's image history started over (e.g. NINA restarted)
    sequence_updated = Signal(object, str)  # Emits (sequence entries, [] if none loaded, None on failure; error)

    # Adaptive polling rates
    POLL_RATE_ACTIVE = 0.5  # when exposing/guiding
    POLL_RATE_IDLE = 2    # when idle

    INITIAL_EVENT_BACKLOG = 200  # Past events shown in the event log on connect
    HISTORY_THUMBNAILS = 20  # Most thumbnails sent at once (on connect, or after catching up)
    HISTORY_CHECK_SECONDS = 30  # How often to confirm NINA still has our latest image
    SEQUENCE_POLL_SECONDS = 2  # The sequence tree is large; refresh it less often than equipment
    LIVESTACK_RETRY_SECONDS = 10  # Pause before retrying a failed live stack image download

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self._running = False
        self._poll_interval = self.POLL_RATE_IDLE  # Start with idle rate
        self._fetch_images = True
        self._initial_image_check_done = False  # Have we found NINA's latest image yet?
        self._last_image_index = -1  # Track the last known image index (-1 = no images yet)
        self._last_history_check = 0.0  # time.monotonic() of the last "latest image still there" check
        self._history_check_failures = 0  # Consecutive checks that didn't find our latest image
        self._last_livestack_hash = None  # Track livestack image hash
        self._last_livestack_count = None  # Track livestack stack count
        self._last_livestack_running = False  # Track if livestack was running
        self._livestack_retry_at = 0.0  # time.monotonic() before which a failed image fetch isn't retried
        self._livestack_target = None  # User-selected livestack target
        self._livestack_filter = None  # User-selected livestack filter
        self._consecutive_failures = 0  # Track consecutive API failures to detect disconnect
        self._version = ""  # Store NINA API version
        self._last_event_time = None  # Aware datetime of the newest processed event
        self._seen_at_last_time = set()  # (Time, Event) keys already emitted at _last_event_time
        # Image quality/size settings
        self._image_quality = -1  # -1 = PNG (lossless)
        self._image_size = "1920x1080"
        self._livestack_quality = 100
        self._livestack_size = "1920x1080"
        # Live view settings
        self._liveview_active = False
        self._liveview_quality = 80
        self._liveview_size = "800x600"
        # Active image tab (0=Live View, 1=Latest Image, 2=Live Stack)
        self._active_image_tab = 1
        self._pending_image_fetch = False  # New image detected while not on Latest Image tab
        # Sequence dock polling (only while the dock is visible)
        self._sequence_active = False
        self._last_sequence_poll = 0.0  # time.monotonic() of the last sequence fetch

    def set_image_quality_settings(self, image_quality, image_size, livestack_quality, livestack_size):
        """Update image quality/size settings (thread-safe for primitive types)."""
        self._image_quality = image_quality
        self._image_size = image_size
        self._livestack_quality = livestack_quality
        self._livestack_size = livestack_size

    def run(self):
        """Main polling loop."""
        self._running = True

        # Test connection first
        success, message, version = NINAIntegration.test_connection(self.host, self.port)
        if success:
            self._version = version or "Unknown"
            self.connection_changed.emit(True, self._version, self.host, self.port)
            self._load_initial_events()
        else:
            self.connection_changed.emit(False, "", self.host, self.port)
            self.error_occurred.emit(message)
            return

        while self._running:
            try:
                # While NINA isn't answering, only check whether it's back (one small
                # request) rather than a full poll whose every request would time out
                if self._consecutive_failures and not NINAIntegration.is_reachable(self.host, self.port):
                    raise _NINAUnreachable()

                # Fetch equipment status
                status_data = {}

                camera_info = NINAIntegration.get_camera_info(self.host, self.port)
                if camera_info and isinstance(camera_info, dict):
                    status_data['camera'] = camera_info
                elif not NINAIntegration.is_reachable(self.host, self.port):
                    # No answer from NINA itself: skip the rest of this poll. A closed port
                    # on a PC whose firewall drops connections makes each request wait for
                    # its whole timeout, so a full poll would take most of a minute.
                    raise _NINAUnreachable()

                mount_info = NINAIntegration.get_mount_info(self.host, self.port)
                if mount_info and isinstance(mount_info, dict):
                    status_data['mount'] = mount_info

                guider_info = NINAIntegration.get_guider_info(self.host, self.port)
                if guider_info and isinstance(guider_info, dict):
                    status_data['guider'] = guider_info

                filterwheel_info = NINAIntegration.get_filterwheel_info(self.host, self.port)
                if filterwheel_info and isinstance(filterwheel_info, dict):
                    status_data['filterwheel'] = filterwheel_info

                focuser_info = NINAIntegration.get_focuser_info(self.host, self.port)
                if focuser_info and isinstance(focuser_info, dict):
                    status_data['focuser'] = focuser_info

                # Use image-history endpoint for statistics (more reliable than capture/statistics)
                if self._last_image_index >= 0:
                    statistics = NINAIntegration.get_image_statistics(self.host, self.port, self._last_image_index)
                    if statistics and isinstance(statistics, dict):
                        status_data['statistics'] = statistics

                # Always emit status update so UI can show current state
                self.status_updated.emit(status_data)

                # Adaptive polling: faster when active, slower when idle
                camera = status_data.get('camera', {})
                guider = status_data.get('guider', {})
                is_exposing = camera.get('IsExposing', False) if isinstance(camera, dict) else False
                guider_state = guider.get('State', '') if isinstance(guider, dict) else ''
                is_guiding = guider.get('Connected', False) and guider_state == 'Guiding' if isinstance(guider, dict) else False

                if is_exposing or is_guiding:
                    self._poll_interval = self.POLL_RATE_ACTIVE
                else:
                    self._poll_interval = self.POLL_RATE_IDLE

                # Check if we got any data - if not, NINA might be disconnected
                if status_data:
                    if self._consecutive_failures >= 2:
                        # Reconnected after being disconnected
                        logger.debug("NINA connection restored")
                        self.connection_changed.emit(True, self._version, self.host, self.port)
                    self._consecutive_failures = 0  # Reset on success
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        logger.debug(f"NINA connection lost ({self._consecutive_failures} consecutive failures)")
                        self.connection_changed.emit(False, "", self.host, self.port)
                        # Keep counting but don't reset - we want to stay disconnected

                # --- New images (on every tab: the history dock is always visible) ---
                if self._fetch_images and status_data:
                    self._check_for_new_images(status_data)

                # --- Live Stack fetching (only when tab 2 is active) ---
                if self._active_image_tab == 2:
                    livestack_status = NINAIntegration.get_livestack_status(self.host, self.port)
                    is_livestacking = (livestack_status and
                                       isinstance(livestack_status, dict) and
                                       livestack_status.get('running', False))
                    if is_livestacking:
                        # Get available stacks for the dropdown
                        available_stacks = NINAIntegration.get_livestack_available(self.host, self.port)

                        # Resolve which target/filter to display
                        actual_target, actual_filter = NINAIntegration.resolve_livestack_selection(
                            available_stacks, self._livestack_target, self._livestack_filter
                        )
                        if actual_target and actual_filter:
                            self._last_livestack_running = True
                            livestack_status['selected_target'] = actual_target
                            livestack_status['selected_filter'] = actual_filter

                            # Fetch lightweight info to check stack count
                            livestack_info = NINAIntegration.get_livestack_info(
                                self.host, self.port, actual_target, actual_filter
                            )
                            if livestack_info:
                                livestack_status.update(livestack_info)

                            # Determine current stack count from info
                            current_count = None
                            if livestack_info:
                                current_count = (livestack_info.get('StackCount')
                                                 or livestack_info.get('RedStackCount')
                                                 or livestack_info.get('GreenStackCount')
                                                 or livestack_info.get('BlueStackCount'))

                            # Only fetch the image when stack count changes (or first time).
                            # The count is recorded once the image arrives, so a failed
                            # download is retried (after a pause) rather than waiting
                            # for the next frame to be stacked.
                            if (current_count != self._last_livestack_count
                                    and time.monotonic() >= self._livestack_retry_at):
                                # Recalculate integration time from full image history
                                # so mixed-exposure stacks are always accurate
                                if current_count and actual_target:
                                    history = NINAIntegration.get_all_image_history(self.host, self.port)
                                    integration = NINAIntegration.calculate_integration_from_history(
                                        history, actual_target, int(current_count), actual_filter)
                                    if integration is not None:
                                        livestack_status['calculated_integration'] = integration
                                self.livestack_fetching.emit(0, -1)
                                livestack_image = NINAIntegration.fetch_livestack_image_data(
                                    self.host, self.port, actual_target, actual_filter,
                                    quality=self._livestack_quality, size=self._livestack_size,
                                    progress_callback=lambda recv, total: self.livestack_fetching.emit(recv, total)
                                )
                                if livestack_image:
                                    self._last_livestack_count = current_count
                                    self.livestack_updated.emit(livestack_image, livestack_status, available_stacks)
                                else:
                                    self._livestack_retry_at = time.monotonic() + self.LIVESTACK_RETRY_SECONDS
                                    self.livestack_updated.emit(b'', livestack_status, available_stacks)
                            else:
                                # Stack count unchanged - emit status only (no image data)
                                self.livestack_updated.emit(b'', livestack_status, available_stacks)
                        elif not self._last_livestack_running:
                            # Livestacking is running but no stacks available yet
                            self._last_livestack_running = True
                            self.livestack_updated.emit(b'', livestack_status, available_stacks)
                    else:
                        # Emit empty to reset tab (only if state changed)
                        if self._last_livestack_running or self._last_livestack_count is not None:
                            self._last_livestack_hash = None
                            self._last_livestack_count = None
                            self._last_livestack_running = False
                            self.livestack_updated.emit(b'', {'running': False}, [])

                # Fetch guiding graph data only if guider is connected and guiding
                guider = status_data.get('guider', {})
                guider_state = guider.get('State', '') if isinstance(guider, dict) else ''
                is_guiding = guider.get('Connected', False) and guider_state == 'Guiding'
                if is_guiding:
                    guiding_data = NINAIntegration.get_guiding_graph_data(self.host, self.port)
                    if guiding_data:
                        self.guiding_updated.emit(guiding_data)

                # Fetch event history and emit new events
                events = NINAIntegration.get_event_history(self.host, self.port)
                for event in self._new_events(events):
                    self.event_occurred.emit(event)
                    if event.get('Event') == 'AUTOFOCUS-FINISHED':
                        report = NINAIntegration.get_last_autofocus(self.host, self.port)
                        if report:
                            self.autofocus_report.emit(report)

                # Fetch the sequence — only while the Sequence dock is visible, or the
                # Live Stack tab is open (it follows the sequence's current target)
                if ((self._sequence_active or self._active_image_tab == 2) and
                        time.monotonic() - self._last_sequence_poll >= self.SEQUENCE_POLL_SECONDS):
                    self._last_sequence_poll = time.monotonic()
                    sequence, sequence_error = NINAIntegration.get_sequence(self.host, self.port)
                    self.sequence_updated.emit(sequence, sequence_error)

                # Fetch live view (prepared image) — only when tab 0 is active
                if self._active_image_tab == 0 and self._liveview_active and not is_exposing:
                    image_data = NINAIntegration.get_prepared_image(
                        self.host, self.port,
                        quality=self._liveview_quality,
                        size_wh=self._liveview_size
                    )
                    if image_data:
                        self.liveview_updated.emit(image_data)

            except _NINAUnreachable:
                self._note_unreachable()
            except Exception as e:
                logger.error(f"Error in NINA status worker: {e}")
                self.error_occurred.emit(str(e))

            # Sleep for polling interval
            # Use faster rate (~5 FPS) when live view is active
            if self._running:
                poll = 0.2 if (self._liveview_active and self._active_image_tab == 0) else self._poll_interval
                sleep_iterations = int(poll * 20)  # 50ms per iteration
                for _ in range(max(1, sleep_iterations)):
                    if not self._running:
                        break
                    self.msleep(50)

    def _note_unreachable(self):
        """Count a poll NINA didn't answer; two in a row means it's gone."""
        self._consecutive_failures += 1
        self._poll_interval = self.POLL_RATE_IDLE
        if self._consecutive_failures == 2:
            logger.debug("NINA connection lost (not answering)")
            self.connection_changed.emit(False, "", self.host, self.port)

    def _check_for_new_images(self, status_data):
        """Send images NINA saved since the last poll to the UI.

        Looks for the next image history index on every poll rather than waiting
        to see the camera stop exposing: in a sequence NINA starts the next
        exposure as soon as the last one downloads, so that gap is usually too
        short for a poll to catch, and the history fell behind.
        """
        if not self._initial_image_check_done:
            self._initial_image_check_done = True
            self._last_history_check = time.monotonic()
            image_count = NINAIntegration.get_image_count(self.host, self.port)
            self._last_image_index = image_count - 1
            if image_count == 0:
                logger.debug("No images available yet, waiting for first exposure")
                return
            logger.debug(f"Latest image on connect: index {self._last_image_index}")
            self._pending_image_fetch = True
            # Newest first; the list adds each older thumbnail below
            for index in range(self._last_image_index,
                               max(self._last_image_index - self.HISTORY_THUMBNAILS, -1), -1):
                self._emit_history_thumbnail(index)
        else:
            # Catch up on every image saved since the last poll
            new_indexes = []
            next_index = self._last_image_index + 1
            while self._running and NINAIntegration._image_exists(self.host, self.port, next_index):
                new_indexes.append(next_index)
                next_index += 1
            if new_indexes:
                logger.debug(f"New image(s) at index {new_indexes[0]}-{new_indexes[-1]}")
                self._last_image_index = new_indexes[-1]
                self._pending_image_fetch = True
                self._history_check_failures = 0
                # Oldest first, so each lands on top of the list
                for index in new_indexes[-self.HISTORY_THUMBNAILS:]:
                    self._emit_history_thumbnail(index)
            elif self._history_was_reset():
                logger.debug("NINA's image history was reset; reloading it")
                self._initial_image_check_done = False
                self._last_image_index = -1
                self._pending_image_fetch = False
                self.history_reset.emit()
                return

        # The full-size image is only needed on the Latest Image tab; on other
        # tabs it's fetched when that tab is next shown
        if self._pending_image_fetch and self._active_image_tab == 1 and self._last_image_index >= 0:
            self._pending_image_fetch = False
            self.image_fetching.emit(0, -1)
            image_data, _ = NINAIntegration.get_image(
                self.host, self.port, self._last_image_index,
                quality=self._image_quality, size_wh=self._image_size,
                progress_callback=lambda recv, total: self.image_fetching.emit(recv, total)
            )
            self.image_fetching.emit(-1, -1)
            if image_data:
                # status_data's statistics were fetched at the start of this poll,
                # before the new image was found, so they belong to the previous one
                image_meta = dict(status_data)
                stats = NINAIntegration.get_image_statistics(self.host, self.port, self._last_image_index)
                if stats:
                    image_meta['statistics'] = stats
                self.image_updated.emit(image_data, image_meta)

    def _history_was_reset(self):
        """Whether NINA no longer has our latest image (its history started over).

        Checked every HISTORY_CHECK_SECONDS; takes two misses in a row so one
        failed request doesn't reload the history.
        """
        if self._last_image_index < 0:
            return False
        interval = self.HISTORY_CHECK_SECONDS if not self._history_check_failures else 5
        if time.monotonic() - self._last_history_check < interval:
            return False
        self._last_history_check = time.monotonic()
        if NINAIntegration._image_exists(self.host, self.port, self._last_image_index):
            self._history_check_failures = 0
            return False
        self._history_check_failures += 1
        if self._history_check_failures < 2:
            return False
        self._history_check_failures = 0
        return True

    def _emit_history_thumbnail(self, index):
        """Fetch a history thumbnail and its statistics and send them to the UI."""
        thumb_data, _ = NINAIntegration.get_image_thumbnail(self.host, self.port, index, 200)
        if thumb_data:
            stats = NINAIntegration.get_image_statistics(self.host, self.port, index) or {}
            self.history_thumbnail.emit(index, thumb_data, stats)

    @staticmethod
    def _parse_event_time(time_str):
        """Parse a NINA event timestamp into an aware datetime (None if unparseable).

        Most NINA timestamps carry a UTC offset, but some (e.g. ERROR-PLATESOLVE)
        do not; those are assumed to be local time.
        """
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except (ValueError, TypeError, AttributeError):
            return None
        return dt if dt.tzinfo else dt.astimezone()

    def _new_events(self, events):
        """Return events not seen before, in order, and advance the seen marker."""
        new_events = []
        for event in events or []:
            event_time = self._parse_event_time(event.get('Time'))
            if event_time is None:
                continue
            key = (event.get('Time'), event.get('Event'))
            if self._last_event_time is not None:
                if event_time < self._last_event_time:
                    continue
                # Events can share a timestamp; dedupe those at the boundary
                if event_time == self._last_event_time and key in self._seen_at_last_time:
                    continue
            if self._last_event_time is None or event_time > self._last_event_time:
                self._last_event_time = event_time
                self._seen_at_last_time = set()
            self._seen_at_last_time.add(key)
            new_events.append(event)
        return new_events

    def _load_initial_events(self):
        """Mark existing events as seen and send the last AF report and recent events to the UI."""
        events = NINAIntegration.get_event_history(self.host, self.port)
        self._new_events(events)  # Advance the marker so old events aren't re-emitted

        # Show the last completed AF run so the graph isn't empty on open
        # (emitted first so an in-progress run from the backlog replaces it)
        report = NINAIntegration.get_last_autofocus(self.host, self.port)
        if report:
            self.autofocus_report.emit(report)

        if events:
            self.events_loaded.emit(events[-self.INITIAL_EVENT_BACKLOG:])

    def stop(self):
        """Stop the polling loop."""
        self._running = False

    def set_fetch_images(self, enabled):
        """Enable or disable image fetching."""
        self._fetch_images = enabled

    def set_livestack_selection(self, target, filter_name):
        """Set the livestack target and filter to fetch."""
        self._livestack_target = target
        self._livestack_filter = filter_name
        # Reset cached count so the next poll fetches the image for the new selection
        self._last_livestack_count = None
        self._livestack_retry_at = 0.0

    def set_liveview_active(self, active):
        """Start or stop live view polling."""
        self._liveview_active = active

    def set_liveview_settings(self, quality, size):
        """Update live view quality/size settings (thread-safe for primitive types)."""
        self._liveview_quality = quality
        self._liveview_size = size

    def set_active_image_tab(self, index):
        """Set which image tab is active (0=Live View, 1=Latest Image, 2=Live Stack)."""
        self._active_image_tab = index

    def refresh_sequence(self):
        """Fetch the sequence on the next poll (e.g. right after starting or stopping it)."""
        self._last_sequence_poll = 0.0

    def set_sequence_active(self, active):
        """Start or stop sequence polling; becoming active fetches on the next poll."""
        if active and not self._sequence_active:
            self._last_sequence_poll = 0.0
        self._sequence_active = active


class GuidingGraph(FigureCanvas):
    """Matplotlib canvas for RA/Dec guiding deviation plot."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 2.5))
        super().__init__(self.figure)
        self.setParent(parent)
        theme_manager().theme_changed.connect(self._on_theme_changed)

        # NINA's current guide graph window (replaced on every update)
        self.max_points = 100  # Empty chart width; NINA's default history size
        self.ra_data = []
        self.dec_data = []
        self.time_data = []
        self.dither_positions = []
        self.rms = None  # NINA's RMS for the window, in pixels
        self._factor, self._unit = 1.0, '"'  # Pixels -> display units

        self.ax = None
        self._create_empty_chart()

    def _on_theme_changed(self):
        """Redraw with the new theme colors."""
        if self.ra_data:
            self._redraw_chart()
        else:
            self._create_empty_chart()

    def _create_empty_chart(self):
        """Create an empty chart placeholder."""
        self.figure.clear()
        self.figure.set_facecolor(chart_background())
        self.ax = self.figure.add_subplot(111)

        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(-3, 3)
        self.ax.set_ylabel('Deviation (arcsec)', color=COLORS['text'], fontsize=chart_font_size(9))
        self.ax.set_xlabel('Time', color=COLORS['text'], fontsize=chart_font_size(9))
        self.ax.set_title('Guiding Performance', color=COLORS['text'], fontsize=chart_font_size(10), fontweight='bold')

        # Add threshold lines
        self.ax.axhline(y=1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1, label='+1"')
        self.ax.axhline(y=-1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1, label='-1"')
        self.ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.3, linewidth=1)

        # Style
        self.ax.set_facecolor(COLORS['background_light'])
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=chart_font_size(8))
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.spines['top'].set_color(COLORS['border'])
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['right'].set_color(COLORS['border'])
        self.ax.yaxis.grid(True, linestyle=':', alpha=0.3, color=COLORS['border'])

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=chart_color('#4488ff'), linewidth=2, label='RA'),
            Line2D([0], [0], color=chart_color('#ff8844'), linewidth=2, label='Dec'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=chart_font_size(8),
                       facecolor=COLORS['background_lighter'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

        self.figure.tight_layout()
        self.draw()

    def update_data(self, graph):
        """Show NINA's guide graph: its last HistorySize steps and its own RMS.

        Each poll returns NINA's whole window, so it replaces the data rather
        than adding to it. Distances are plotted in arcseconds from NINA's raw
        pixel values and pixel scale (pixels if the scale isn't known).
        """
        steps = graph.get('GuideSteps') or []
        pixel_scale = graph.get('PixelScale') or (graph.get('RMS') or {}).get('Scale')
        if isinstance(pixel_scale, (int, float)) and pixel_scale > 0:
            self._factor, self._unit = pixel_scale, '"'
        else:
            self._factor, self._unit = 1.0, ' px'

        self.ra_data.clear()
        self.dec_data.clear()
        self.time_data.clear()
        self.dither_positions = []
        for step in steps:
            dither = step.get('Dither')
            if isinstance(dither, (int, float)) and not math.isnan(dither):
                # NINA's dither marker (zero distances), not a guide step
                self.dither_positions.append(len(self.ra_data) - 0.5)
                continue
            ra = step.get('RADistanceRaw')
            dec = step.get('DECDistanceRaw')
            self.ra_data.append((ra if isinstance(ra, (int, float)) else 0) * self._factor)
            self.dec_data.append((dec if isinstance(dec, (int, float)) else 0) * self._factor)
            self.time_data.append(len(self.time_data))
        self.rms = graph.get('RMS') if isinstance(graph.get('RMS'), dict) else None

        self._redraw_chart()

    def _rms_values(self):
        """(RA, Dec, total) RMS in guide camera pixels - NINA's figures when it sent them.

        NINA's RMS is the standard deviation (the mean offset removed) of the
        guide steps in the window, excluding dither markers.
        """
        rms = self.rms or {}
        if all(isinstance(rms.get(key), (int, float)) for key in ('RA', 'Dec', 'Total')):
            return rms['RA'], rms['Dec'], rms['Total']

        def std(values):  # Plotted values are in display units; back to pixels
            pixels = [v / self._factor for v in values]
            mean = sum(pixels) / len(pixels)
            return math.sqrt(sum((v - mean) ** 2 for v in pixels) / len(pixels))
        ra_rms, dec_rms = std(self.ra_data), std(self.dec_data)
        return ra_rms, dec_rms, math.hypot(ra_rms, dec_rms)

    def _rms_text(self):
        """The RMS the way NINA shows it: 'RA: 0.11 (0.48")  Dec: 0.12 (0.50")  Tot: 0.16 (0.69")'
        - pixels, then arcseconds when the pixel scale is known."""
        parts = []
        for label, pixels in zip(('RA', 'Dec', 'Tot'), self._rms_values()):
            arcsec = f' ({pixels * self._factor:.2f}")' if self._unit == '"' else ''
            parts.append(f'{label}: {pixels:.2f}{arcsec}')
        return '  '.join(parts)

    def _redraw_chart(self):
        """Redraw the chart with current data."""
        if not self.ra_data:
            return

        self.figure.clear()
        self.figure.set_facecolor(chart_background())
        self.ax = self.figure.add_subplot(111)

        x_data = list(range(len(self.ra_data)))

        # Dithers, as NINA marks them
        for position in self.dither_positions:
            self.ax.axvline(x=position, color=COLORS['text_secondary'], linestyle=':', alpha=0.6, linewidth=1)

        # Plot RA and Dec
        self.ax.plot(x_data, list(self.ra_data), color=chart_color('#4488ff'), linewidth=1.5, label='RA')
        self.ax.plot(x_data, list(self.dec_data), color=chart_color('#ff8844'), linewidth=1.5, label='Dec')

        # Add threshold lines
        self.ax.axhline(y=1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
        self.ax.axhline(y=-1, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
        self.ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.3, linewidth=1)

        self.ax.set_title(f'Guiding Performance  |  {self._rms_text()}',
                          color=COLORS['text'], fontsize=chart_font_size(10), fontweight='bold')

        # Axis limits
        self.ax.set_xlim(0, max(len(self.ra_data), 60))
        y_max = max(3, max(abs(min(self.ra_data)), abs(max(self.ra_data)),
                          abs(min(self.dec_data)), abs(max(self.dec_data))) * 1.2)
        self.ax.set_ylim(-y_max, y_max)

        self.ax.set_ylabel('Deviation (arcsec)' if self._unit == '"' else 'Deviation (pixels)',
                           color=COLORS['text'], fontsize=chart_font_size(9))
        self.ax.set_xlabel('Guide steps', color=COLORS['text'], fontsize=chart_font_size(9))

        # Style
        self.ax.set_facecolor(COLORS['background_light'])
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=chart_font_size(8))
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.spines['top'].set_color(COLORS['border'])
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['right'].set_color(COLORS['border'])
        self.ax.yaxis.grid(True, linestyle=':', alpha=0.3, color=COLORS['border'])

        # Legend
        self.ax.legend(loc='upper right', fontsize=chart_font_size(8),
                       facecolor=COLORS['background_lighter'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

        self.figure.tight_layout()
        self.draw()

    def clear_data(self):
        """Clear all guiding data."""
        self.ra_data.clear()
        self.dec_data.clear()
        self.time_data.clear()
        self.dither_positions = []
        self.rms = None
        self._create_empty_chart()


class AutofocusGraph(FigureCanvas):
    """Matplotlib canvas for the autofocus V-curve (focuser position vs HFR)."""

    _NUM = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    _HYPERBOLIC_RE = re.compile(rf'y = {_NUM} \* cosh\(asinh\(\({_NUM} - x\) / {_NUM}\)\)')
    _QUADRATIC_RE = re.compile(rf'y = {_NUM} \* x\^2 \+ {_NUM} \* x \+ {_NUM}')
    _LINE_RE = re.compile(rf'y = {_NUM} \* x \+ {_NUM}')

    POINT_COLOR = '#4488ff'
    FIT_COLOR = '#ff8844'
    TREND_COLOR = '#aaaaaa'

    def __init__(self, parent=None):
        # Constrained layout re-fits margins on every draw, including dock resizes
        self.figure = Figure(figsize=(6, 3.5), layout='constrained')
        self.figure.get_layout_engine().set(h_pad=0.04, w_pad=0.04)
        super().__init__(self.figure)
        self.setParent(parent)
        theme_manager().theme_changed.connect(self._redraw_chart)

        self.points = []  # [(position, hfr, error)]
        self.report = None  # last-af report for the completed run, None while running
        self.title = 'AutoFocus'
        self._redraw_chart()

    def start_run(self):
        """Clear the graph for a new autofocus run."""
        self.points = []
        self.report = None
        self.title = 'AutoFocus running...'
        self._redraw_chart()

    def add_point(self, position, hfr):
        """Add a measured point from an AUTOFOCUS-POINT-ADDED event."""
        self.points.append((position, hfr, 0))
        self._redraw_chart()

    def end_run(self, title):
        """Mark the current run as ended without a report (failed or cancelled)."""
        self.title = title
        self._redraw_chart()

    def set_report(self, report):
        """Show a completed run from the last-af report, including the fitted curve."""
        self.report = report
        self.points = [
            (p.get('Position'), p.get('Value'), p.get('Error') or 0)
            for p in report.get('MeasurePoints', [])
            if isinstance(p.get('Position'), (int, float)) and isinstance(p.get('Value'), (int, float))
        ]
        focus = report.get('CalculatedFocusPoint') or {}
        position, hfr = focus.get('Position'), focus.get('Value')
        if isinstance(position, (int, float)) and isinstance(hfr, (int, float)):
            self.title = f'AutoFocus: position {position}  |  HFR {hfr:.2f}'
        else:
            self.title = 'AutoFocus complete'
        self._redraw_chart()

    def _fit_curves(self, x_min, x_max):
        """Build (label, xs, ys, color, style) curves from the report's fitting formulas."""
        report = self.report or {}
        fitting = str(report.get('Fitting', '')).upper()
        fittings = report.get('Fittings') or {}
        steps = 200
        xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]
        curves = []

        if 'HYPERBOLIC' in fitting:
            m = self._HYPERBOLIC_RE.search(fittings.get('Hyperbolic', ''))
            if m:
                a, p, b = (float(g) for g in m.groups())
                if b:
                    ys = [a * math.cosh(math.asinh((p - x) / b)) for x in xs]
                    curves.append(('Hyperbolic', xs, ys, self.FIT_COLOR, '-'))

        if 'PARABOLIC' in fitting:
            m = self._QUADRATIC_RE.search(fittings.get('Quadratic', ''))
            if m:
                a, b, c = (float(g) for g in m.groups())
                ys = [a * x * x + b * x + c for x in xs]
                curves.append(('Parabolic', xs, ys, self.FIT_COLOR, '-'))

        if 'TREND' in fitting:
            trend_x = ((report.get('Intersections') or {}).get('TrendLineIntersection') or {}).get('Position')
            if isinstance(trend_x, (int, float)) and x_min < trend_x < x_max:
                for key, lo, hi in (('LeftTrend', x_min, trend_x), ('RightTrend', trend_x, x_max)):
                    m = self._LINE_RE.search(fittings.get(key, ''))
                    if m:
                        slope, intercept = (float(g) for g in m.groups())
                        curves.append(('Trend' if key == 'LeftTrend' else None, [lo, hi],
                                       [slope * lo + intercept, slope * hi + intercept],
                                       self.TREND_COLOR, '--'))
        return curves

    def _redraw_chart(self):
        """Redraw the chart with current data."""
        self.figure.clear()
        self.figure.set_facecolor(chart_background())
        self.ax = self.figure.add_subplot(111)
        ax = self.ax

        if self.points:
            positions = [p[0] for p in self.points]
            hfrs = [p[1] for p in self.points]
            x_min, x_max = min(positions), max(positions)
            pad = max((x_max - x_min) * 0.05, 5)

            if self.report:
                for label, xs, ys, color, style in self._fit_curves(x_min - pad, x_max + pad):
                    ax.plot(xs, ys, color=chart_color(color), linestyle=style, linewidth=1.2, label=label, alpha=0.9)
                ax.errorbar(positions, hfrs, yerr=[p[2] for p in self.points], fmt='o',
                            color=chart_color(self.POINT_COLOR), ecolor=chart_color(self.POINT_COLOR), elinewidth=1,
                            capsize=2, markersize=5, label='Measured', alpha=0.9)
                focus = self.report.get('CalculatedFocusPoint') or {}
                if isinstance(focus.get('Position'), (int, float)) and isinstance(focus.get('Value'), (int, float)):
                    ax.axvline(focus['Position'], color=COLORS['success'], linestyle=':', linewidth=1, alpha=0.7)
                    ax.plot([focus['Position']], [focus['Value']], marker='*', markersize=12,
                            color=COLORS['success'], linestyle='none', label='Focus')
            else:
                ax.plot(positions, hfrs, 'o', color=chart_color(self.POINT_COLOR), markersize=6, label='Measured')
                best = min(self.points, key=lambda p: p[1])
                ax.plot([best[0]], [best[1]], 'o', markersize=10, markerfacecolor='none',
                        markeredgecolor=COLORS['success'], linestyle='none', label='Best so far')

            ax.set_xlim(x_min - pad, x_max + pad)
            y_max = max(h + e for _, h, e in self.points)
            ax.set_ylim(0, y_max * 1.3)  # Headroom for the legend
            ax.legend(loc='upper center', fontsize=chart_font_size(8), ncol=4,
                      facecolor=COLORS['background_lighter'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])
        else:
            ax.text(0.5, 0.5, 'No autofocus data', transform=ax.transAxes, ha='center', va='center',
                    color=COLORS['text_secondary'], fontsize=chart_font_size(10))
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(self.title, color=COLORS['text'], fontsize=chart_font_size(10), fontweight='bold')
        ax.set_xlabel('Focuser Position', color=COLORS['text'], fontsize=chart_font_size(9))
        ax.set_ylabel('HFR', color=COLORS['text'], fontsize=chart_font_size(9))

        # Style
        ax.set_facecolor(COLORS['background_light'])
        ax.tick_params(colors=COLORS['text_secondary'], labelsize=chart_font_size(8))
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
        ax.grid(True, linestyle=':', alpha=0.3, color=COLORS['border'])

        self.draw()


class CaptureSettingsDialog(QDialog):
    """Dialog for configuring capture settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Form layout for settings
        form_layout = QFormLayout()

        # Duration
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(0.001, 3600)  # 1ms to 1 hour
        self.duration_spinbox.setDecimals(3)
        self.duration_spinbox.setSuffix(" s")
        form_layout.addRow("Duration:", self.duration_spinbox)

        # Gain
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setRange(-1, 1000)  # -1 means use camera default
        self.gain_spinbox.setSpecialValueText("Default")
        form_layout.addRow("Gain:", self.gain_spinbox)

        # Image type
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["SNAPSHOT", "LIGHT", "DARK", "BIAS", "FLAT"])
        form_layout.addRow("Image Type:", self.image_type_combo)

        # Save to disk
        self.save_checkbox = QCheckBox()
        form_layout.addRow("Save to Disk:", self.save_checkbox)

        layout.addLayout(form_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load saved settings
        self._load_settings()

    def _load_settings(self):
        """Load saved capture settings."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.duration_spinbox.setValue(settings.value("capture_duration", 1.0, type=float))
        self.gain_spinbox.setValue(settings.value("capture_gain", -1, type=int))
        self.image_type_combo.setCurrentText(settings.value("capture_image_type", "SNAPSHOT", type=str))
        self.save_checkbox.setChecked(settings.value("capture_save", True, type=bool))

    def _save_settings(self):
        """Save capture settings."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("capture_duration", self.duration_spinbox.value())
        settings.setValue("capture_gain", self.gain_spinbox.value())
        settings.setValue("capture_image_type", self.image_type_combo.currentText())
        settings.setValue("capture_save", self.save_checkbox.isChecked())

    def _on_accepted(self):
        """Handle dialog accepted - save settings and close."""
        self._save_settings()
        self.accept()

    def get_settings(self):
        """Return the capture settings as a dict."""
        gain = self.gain_spinbox.value()
        return {
            'duration': self.duration_spinbox.value(),
            'gain': gain if gain >= 0 else None,  # None means use camera default
            'image_type': self.image_type_combo.currentText(),
            'save': self.save_checkbox.isChecked()
        }


class SlewDialog(QDialog):
    """Dialog for entering slew coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slew to Coordinates")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Search group
        search_group = QGroupBox("Search Object")
        search_layout = QHBoxLayout(search_group)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("e.g., M31, NGC 7000, Vega...")
        self.search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._on_search)
        search_layout.addWidget(self.search_btn)

        layout.addWidget(search_group)

        self.search_status_label = QLabel("")
        themed_style(self.search_status_label, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.search_status_label)

        # Form layout for coordinates
        form_layout = QFormLayout()

        # RA input (hours)
        ra_widget = QWidget()
        ra_layout = QHBoxLayout(ra_widget)
        ra_layout.setContentsMargins(0, 0, 0, 0)
        ra_layout.setSpacing(2)

        self.ra_h_spinbox = QSpinBox()
        self.ra_h_spinbox.setRange(0, 23)
        self.ra_h_spinbox.setSuffix("h")
        ra_layout.addWidget(self.ra_h_spinbox)

        self.ra_m_spinbox = QSpinBox()
        self.ra_m_spinbox.setRange(0, 59)
        self.ra_m_spinbox.setSuffix("m")
        ra_layout.addWidget(self.ra_m_spinbox)

        self.ra_s_spinbox = QDoubleSpinBox()
        self.ra_s_spinbox.setRange(0, 59.99)
        self.ra_s_spinbox.setDecimals(2)
        self.ra_s_spinbox.setSuffix("s")
        ra_layout.addWidget(self.ra_s_spinbox)

        form_layout.addRow("RA:", ra_widget)

        # Dec input (degrees)
        dec_widget = QWidget()
        dec_layout = QHBoxLayout(dec_widget)
        dec_layout.setContentsMargins(0, 0, 0, 0)
        dec_layout.setSpacing(2)

        self.dec_sign_combo = QComboBox()
        self.dec_sign_combo.addItems(["+", "-"])
        dec_layout.addWidget(self.dec_sign_combo)

        self.dec_d_spinbox = QSpinBox()
        self.dec_d_spinbox.setRange(0, 90)
        self.dec_d_spinbox.setSuffix("°")
        dec_layout.addWidget(self.dec_d_spinbox)

        self.dec_m_spinbox = QSpinBox()
        self.dec_m_spinbox.setRange(0, 59)
        self.dec_m_spinbox.setSuffix("'")
        dec_layout.addWidget(self.dec_m_spinbox)

        self.dec_s_spinbox = QDoubleSpinBox()
        self.dec_s_spinbox.setRange(0, 59.99)
        self.dec_s_spinbox.setDecimals(2)
        self.dec_s_spinbox.setSuffix('"')
        dec_layout.addWidget(self.dec_s_spinbox)

        form_layout.addRow("Dec:", dec_widget)

        layout.addLayout(form_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load saved coordinates
        self._load_settings()

    def _load_settings(self):
        """Load saved slew coordinates."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        self.ra_h_spinbox.setValue(settings.value("slew_ra_h", 0, type=int))
        self.ra_m_spinbox.setValue(settings.value("slew_ra_m", 0, type=int))
        self.ra_s_spinbox.setValue(settings.value("slew_ra_s", 0.0, type=float))
        self.dec_sign_combo.setCurrentText(settings.value("slew_dec_sign", "+", type=str))
        self.dec_d_spinbox.setValue(settings.value("slew_dec_d", 0, type=int))
        self.dec_m_spinbox.setValue(settings.value("slew_dec_m", 0, type=int))
        self.dec_s_spinbox.setValue(settings.value("slew_dec_s", 0.0, type=float))

    def _save_settings(self):
        """Save slew coordinates."""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("slew_ra_h", self.ra_h_spinbox.value())
        settings.setValue("slew_ra_m", self.ra_m_spinbox.value())
        settings.setValue("slew_ra_s", self.ra_s_spinbox.value())
        settings.setValue("slew_dec_sign", self.dec_sign_combo.currentText())
        settings.setValue("slew_dec_d", self.dec_d_spinbox.value())
        settings.setValue("slew_dec_m", self.dec_m_spinbox.value())
        settings.setValue("slew_dec_s", self.dec_s_spinbox.value())

    def _on_accepted(self):
        """Handle dialog accepted - save settings and close."""
        self._save_settings()
        self.accept()

    def get_coordinates_degrees(self):
        """Return RA and Dec in degrees."""
        # Convert RA from hours to degrees (1h = 15°)
        ra_hours = self.ra_h_spinbox.value() + self.ra_m_spinbox.value() / 60 + self.ra_s_spinbox.value() / 3600
        ra_deg = ra_hours * 15

        # Convert Dec from DMS to degrees
        dec_deg = self.dec_d_spinbox.value() + self.dec_m_spinbox.value() / 60 + self.dec_s_spinbox.value() / 3600
        if self.dec_sign_combo.currentText() == "-":
            dec_deg = -dec_deg

        return ra_deg, dec_deg

    def _on_search(self):
        """Search for an object by name in the local database."""
        import re
        from DatabaseManager import DatabaseManager

        object_name = self.search_input.text().strip()
        if not object_name:
            self.search_status_label.setText("Enter an object name to search")
            themed_style(self.search_status_label, lambda: f"color: {COLORS['warning']};")
            return

        self.search_status_label.setText(f"Searching for '{object_name}'...")
        themed_style(self.search_status_label, lambda: f"color: {COLORS['text_secondary']};")
        self.search_btn.setEnabled(False)

        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            # Normalize search term (e.g., "M31" -> "M 31", "NGC7000" -> "NGC 7000")
            search_upper = object_name.upper().strip()
            match = re.match(r'^([A-Z]+)\s*(\d+)([A-Z]?)$', search_upper)
            if match:
                catalog = match.group(1)
                number = match.group(2)
                suffix = match.group(3)
                search_catalog = catalog
                search_designation = f"{number}{suffix}"
            else:
                search_catalog = None
                search_designation = search_upper

            db = DatabaseManager()

            # Search by catalogue and designation
            if search_catalog:
                # Try exact match first
                rows = db.execute_query("""
                    SELECT d.ra, d.dec, c.catalogue, c.designation
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE UPPER(c.catalogue) = ? AND UPPER(c.designation) = ?
                    LIMIT 1
                """, (search_catalog, search_designation))
            else:
                rows = []

            if not rows:
                # Try partial match on designation
                rows = db.execute_query("""
                    SELECT d.ra, d.dec, c.catalogue, c.designation
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE UPPER(c.catalogue || ' ' || c.designation) LIKE ?
                       OR UPPER(c.catalogue || c.designation) LIKE ?
                    LIMIT 1
                """, (f"%{search_upper}%", f"%{search_upper.replace(' ', '')}%"))

            if not rows:
                self.search_status_label.setText(f"'{object_name}' not found in database")
                themed_style(self.search_status_label, lambda: f"color: {COLORS['error']};")
                return

            row = rows[0]
            ra_deg = float(row[0])
            dec_deg = float(row[1])
            found_name = f"{row[2]} {row[3]}"

            # Convert RA from degrees to hours
            ra_hours = ra_deg / 15.0
            ra_h = int(ra_hours)
            ra_m = int((ra_hours - ra_h) * 60)
            ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60

            # Convert Dec to DMS
            dec_sign = "+" if dec_deg >= 0 else "-"
            dec_abs = abs(dec_deg)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60

            # Update the coordinate fields
            self.ra_h_spinbox.setValue(ra_h)
            self.ra_m_spinbox.setValue(ra_m)
            self.ra_s_spinbox.setValue(round(ra_s, 2))
            self.dec_sign_combo.setCurrentText(dec_sign)
            self.dec_d_spinbox.setValue(dec_d)
            self.dec_m_spinbox.setValue(dec_m)
            self.dec_s_spinbox.setValue(round(dec_s, 2))

            self.search_status_label.setText(f"Found: {found_name}")
            themed_style(self.search_status_label, lambda: f"color: {COLORS['success']};")

        except Exception as e:
            self.search_status_label.setText(f"Search failed: {str(e)}")
            themed_style(self.search_status_label, lambda: f"color: {COLORS['error']};")
        finally:
            self.search_btn.setEnabled(True)


class NINADashboardWindow(WindowPositionMixin, QMainWindow):
    """Main NINA Dashboard window."""
    WINDOW_POSITION_KEY = "NINADashboard"
    DOCK_LAYOUT_VERSION = 4  # Bump when adding docks so older saved layouts get default placement
    _image_fetch_done = Signal(object)

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("NINA Dashboard - Cosmos Collection")
        self.resize(900, 700)
        self.setup_window_position()

        self.worker = None

        self._connected = False
        self._version = ""
        self._last_update = None
        self._exposure_end_time = None
        self._exposure_total_time = None  # Total exposure duration captured on first detection
        self._camera_seen_idle = False  # Saw the camera not exposing, so the next exposure's start is seen
        self._current_image_pixmap = None  # Store original pixmap for rescaling
        self._viewing_history_index = None  # Set when viewing a historical image (not the latest)
        self._current_livestack_pixmap = None  # Store original livestack pixmap
        self._current_liveview_pixmap = None  # Store original liveview pixmap
        self._sub_exposure_by_target = {}  # {target_name: latest exposure_time} from image history
        self._total_integration_by_target = {}  # {(target, filter): integration seconds from history}
        self._restoring_settings = False  # Guard to prevent re-fetch during settings restore

        self._image_fetch_done.connect(self._on_image_fetch_done)

        self._setup_ui()
        # Widget styles follow theme changes on their own (Theme.themed_style);
        # the event log's row colors are item data, so recolor those
        theme_manager().theme_changed.connect(self._recolor_event_log)
        self._auto_connect()
        # Defer settings restore until after window is shown (dock widgets need visible geometry)
        QTimer.singleShot(100, self._restore_settings)

    def _setup_ui(self):
        """Set up the main window UI with dockable panels."""
        # Central widget - contains header, image panel, and status bar
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Header bar
        header_layout = QHBoxLayout()

        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.setToolTip("Reconnect to NINA")
        self.reconnect_btn.clicked.connect(self._reconnect)
        header_layout.addWidget(self.reconnect_btn)

        self.connection_label = QLabel("Connection: Disconnected")
        themed_style(self.connection_label, lambda: f"color: {COLORS['warning']};")
        header_layout.addWidget(self.connection_label)

        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Image panel in central widget (main content area)
        self._create_image_panel(main_layout)

        # Status bar at bottom of central widget
        status_layout_h = QHBoxLayout()
        self.status_label = QLabel("Ready")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        status_layout_h.addWidget(self.status_label)
        status_layout_h.addStretch()
        self.countdown_label = QLabel("")
        themed_style(self.countdown_label, lambda: f"color: {COLORS['text_secondary']};")
        status_layout_h.addWidget(self.countdown_label)
        main_layout.addLayout(status_layout_h)

        # Create dock widgets (added to window in _restore_settings)
        self._create_equipment_docks()
        self._create_actions_docks()
        self._create_guiding_dock()
        self._create_image_history_dock()
        self._create_autofocus_dock()
        self._create_event_log_dock()
        self._create_sequence_dock()

        # Set up View menu
        self._setup_view_menu()

    def _create_equipment_docks(self):
        """Create individual dock widgets for each equipment type."""
        dock_features = (
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock_areas = (
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )

        # Camera dock
        self.camera_dock = QDockWidget("Camera", self)
        self.camera_dock.setObjectName("CameraDock")
        self.camera_dock.setAllowedAreas(dock_areas)
        self.camera_dock.setFeatures(dock_features)

        camera_widget = QWidget()
        camera_layout = QGridLayout(camera_widget)
        camera_layout.setContentsMargins(8, 8, 8, 8)

        camera_layout.addWidget(QLabel("Name:"), 0, 0)
        self.camera_name_label = QLabel("--")
        camera_layout.addWidget(self.camera_name_label, 0, 1)
        camera_layout.addWidget(QLabel("Status:"), 1, 0)
        self.camera_status_label = QLabel("--")
        camera_layout.addWidget(self.camera_status_label, 1, 1)
        camera_layout.addWidget(QLabel("Progress:"), 2, 0)
        self.camera_progress = QProgressBar()
        self.camera_progress.setMaximum(100)
        self.camera_progress.setValue(0)
        camera_layout.addWidget(self.camera_progress, 2, 1)
        camera_layout.addWidget(QLabel("Temp:"), 3, 0)
        self.camera_temp_label = QLabel("--")
        camera_layout.addWidget(self.camera_temp_label, 3, 1)

        # Cooling controls
        camera_layout.addWidget(QLabel("Cooling:"), 4, 0)
        cooling_widget = QWidget()
        cooling_layout = QHBoxLayout(cooling_widget)
        cooling_layout.setContentsMargins(0, 0, 0, 0)
        cooling_layout.setSpacing(5)
        self.camera_cooling_checkbox = QCheckBox("On")
        self.camera_cooling_checkbox.setChecked(False)  # Explicitly start unchecked
        self.camera_cooling_checkbox.setToolTip("Enable/disable camera cooling")
        self.camera_cooling_checkbox.stateChanged.connect(self._on_cooling_changed)
        logger.debug(f"[Cooling] Checkbox created, initial checked state: {self.camera_cooling_checkbox.isChecked()}")
        cooling_layout.addWidget(self.camera_cooling_checkbox)
        self.camera_target_temp_spinbox = QSpinBox()
        self.camera_target_temp_spinbox.setRange(-40, 20)
        self.camera_target_temp_spinbox.setValue(-10)
        self.camera_target_temp_spinbox.setSuffix("°C")
        self.camera_target_temp_spinbox.setToolTip("Target cooling temperature")
        # Sent once the value settles, not on every arrow click
        self._target_temp_timer = QTimer(self)
        self._target_temp_timer.setSingleShot(True)
        self._target_temp_timer.setInterval(800)
        self._target_temp_timer.timeout.connect(self._apply_target_temp)
        # (a lambda: valueChanged passes the value, which start() would take as the interval)
        self.camera_target_temp_spinbox.valueChanged.connect(lambda _value: self._target_temp_timer.start())
        cooling_layout.addWidget(self.camera_target_temp_spinbox)
        cooling_layout.addStretch()
        camera_layout.addWidget(cooling_widget, 4, 1)

        # Dew heater control
        camera_layout.addWidget(QLabel("Dew Heater:"), 5, 0)
        self.camera_dewheater_checkbox = QCheckBox("On")
        self.camera_dewheater_checkbox.setToolTip("Enable/disable dew heater")
        self.camera_dewheater_checkbox.stateChanged.connect(self._on_dewheater_changed)
        camera_layout.addWidget(self.camera_dewheater_checkbox, 5, 1)

        camera_layout.setRowStretch(6, 1)

        # Track if we're updating from API to avoid triggering callbacks
        self._updating_camera_controls = False
        # Track if user recently changed settings (prevents sync from overriding)
        self._user_changing_cooling = False
        self._user_changing_dewheater = False
        # A cooling / dew heater request is on its way to NINA (its control stays disabled)
        self._cooling_request_pending = False
        self._dewheater_request_pending = False
        # Track the last cooling state we intentionally set (to avoid duplicate API calls)
        self._last_cooling_enabled = None
        self._last_cooling_temp = None
        # Timers to clear the user-changing flags
        self._cooling_change_timer = QTimer(self)
        self._cooling_change_timer.setSingleShot(True)
        self._cooling_change_timer.timeout.connect(self._clear_cooling_change_flag)
        self._dewheater_change_timer = QTimer(self)
        self._dewheater_change_timer.setSingleShot(True)
        self._dewheater_change_timer.timeout.connect(self._clear_dewheater_change_flag)

        self.camera_dock.setWidget(camera_widget)

        # Mount dock
        self.mount_dock = QDockWidget("Mount", self)
        self.mount_dock.setObjectName("MountDock")
        self.mount_dock.setAllowedAreas(dock_areas)
        self.mount_dock.setFeatures(dock_features)

        mount_widget = QWidget()
        mount_layout = QGridLayout(mount_widget)
        mount_layout.setContentsMargins(8, 8, 8, 8)

        mount_layout.addWidget(QLabel("Name:"), 0, 0)
        self.mount_name_label = QLabel("--")
        mount_layout.addWidget(self.mount_name_label, 0, 1)
        mount_layout.addWidget(QLabel("Status:"), 1, 0)
        self.mount_status_label = QLabel("--")
        mount_layout.addWidget(self.mount_status_label, 1, 1)
        mount_layout.addWidget(QLabel("RA/Dec:"), 2, 0)
        self.mount_coords_label = QLabel("--")
        mount_layout.addWidget(self.mount_coords_label, 2, 1)
        mount_layout.setRowStretch(3, 1)

        self.mount_dock.setWidget(mount_widget)

        # Guider dock
        self.guider_dock = QDockWidget("Guider", self)
        self.guider_dock.setObjectName("GuiderDock")
        self.guider_dock.setAllowedAreas(dock_areas)
        self.guider_dock.setFeatures(dock_features)

        guider_widget = QWidget()
        guider_layout = QGridLayout(guider_widget)
        guider_layout.setContentsMargins(8, 8, 8, 8)

        guider_layout.addWidget(QLabel("Name:"), 0, 0)
        self.guider_name_label = QLabel("--")
        guider_layout.addWidget(self.guider_name_label, 0, 1)
        guider_layout.addWidget(QLabel("Status:"), 1, 0)
        self.guider_status_label = QLabel("--")
        guider_layout.addWidget(self.guider_status_label, 1, 1)
        guider_layout.addWidget(QLabel("RMS:"), 2, 0)
        self.guider_rms_label = QLabel("--")
        guider_layout.addWidget(self.guider_rms_label, 2, 1)
        guider_layout.setRowStretch(3, 1)

        self.guider_dock.setWidget(guider_widget)

        # Filter Wheel dock
        self.filterwheel_dock = QDockWidget("Filter Wheel", self)
        self.filterwheel_dock.setObjectName("FilterWheelDock")
        self.filterwheel_dock.setAllowedAreas(dock_areas)
        self.filterwheel_dock.setFeatures(dock_features)

        filterwheel_widget = QWidget()
        filterwheel_layout = QGridLayout(filterwheel_widget)
        filterwheel_layout.setContentsMargins(8, 8, 8, 8)

        filterwheel_layout.addWidget(QLabel("Name:"), 0, 0)
        self.filterwheel_name_label = QLabel("--")
        filterwheel_layout.addWidget(self.filterwheel_name_label, 0, 1)
        filterwheel_layout.addWidget(QLabel("Status:"), 1, 0)
        self.filterwheel_status_label = QLabel("--")
        filterwheel_layout.addWidget(self.filterwheel_status_label, 1, 1)
        filterwheel_layout.addWidget(QLabel("Filter:"), 2, 0)
        self.filterwheel_combo = QComboBox()
        self.filterwheel_combo.setToolTip("Select filter to change to")
        self.filterwheel_combo.setEnabled(False)
        self.filterwheel_combo.currentIndexChanged.connect(self._on_filter_changed)
        filterwheel_layout.addWidget(self.filterwheel_combo, 2, 1)
        filterwheel_layout.setRowStretch(3, 1)

        # Track filter wheel state to prevent feedback loops
        self._updating_filterwheel = False
        self._user_changing_filter = False
        self._last_filter_id = None
        self._available_filters = []  # List of {'Id': int, 'Name': str}

        self.filterwheel_dock.setWidget(filterwheel_widget)

        # Focuser dock
        self.focuser_dock = QDockWidget("Focuser", self)
        self.focuser_dock.setObjectName("FocuserDock")
        self.focuser_dock.setAllowedAreas(dock_areas)
        self.focuser_dock.setFeatures(dock_features)

        focuser_widget = QWidget()
        focuser_layout = QGridLayout(focuser_widget)
        focuser_layout.setContentsMargins(8, 8, 8, 8)

        focuser_layout.addWidget(QLabel("Name:"), 0, 0)
        self.focuser_name_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_name_label, 0, 1)
        focuser_layout.addWidget(QLabel("Status:"), 1, 0)
        self.focuser_status_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_status_label, 1, 1)
        focuser_layout.addWidget(QLabel("Position:"), 2, 0)
        self.focuser_position_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_position_label, 2, 1)
        focuser_layout.addWidget(QLabel("Temp:"), 3, 0)
        self.focuser_temp_label = QLabel("--")
        focuser_layout.addWidget(self.focuser_temp_label, 3, 1)
        focuser_layout.setRowStretch(4, 1)

        self.focuser_dock.setWidget(focuser_widget)

        # Statistics dock
        self.statistics_dock = QDockWidget("Statistics", self)
        self.statistics_dock.setObjectName("StatisticsDock")
        self.statistics_dock.setAllowedAreas(dock_areas)
        self.statistics_dock.setFeatures(dock_features)

        statistics_widget = QWidget()
        statistics_layout = QGridLayout(statistics_widget)
        statistics_layout.setContentsMargins(8, 8, 8, 8)

        statistics_layout.addWidget(QLabel("Stars:"), 0, 0)
        self.stats_stars_label = QLabel("--")
        statistics_layout.addWidget(self.stats_stars_label, 0, 1)
        statistics_layout.addWidget(QLabel("HFR:"), 1, 0)
        self.stats_hfr_label = QLabel("--")
        statistics_layout.addWidget(self.stats_hfr_label, 1, 1)
        statistics_layout.addWidget(QLabel("Median:"), 2, 0)
        self.stats_median_label = QLabel("--")
        statistics_layout.addWidget(self.stats_median_label, 2, 1)
        statistics_layout.addWidget(QLabel("HFR StDev:"), 3, 0)
        self.stats_hfrstdev_label = QLabel("--")
        statistics_layout.addWidget(self.stats_hfrstdev_label, 3, 1)
        statistics_layout.addWidget(QLabel("Mean:"), 4, 0)
        self.stats_mean_label = QLabel("--")
        statistics_layout.addWidget(self.stats_mean_label, 4, 1)
        statistics_layout.addWidget(QLabel("StDev:"), 5, 0)
        self.stats_stdev_label = QLabel("--")
        statistics_layout.addWidget(self.stats_stdev_label, 5, 1)
        statistics_layout.addWidget(QLabel("Min:"), 6, 0)
        self.stats_min_label = QLabel("--")
        statistics_layout.addWidget(self.stats_min_label, 6, 1)
        statistics_layout.addWidget(QLabel("Max:"), 7, 0)
        self.stats_max_label = QLabel("--")
        statistics_layout.addWidget(self.stats_max_label, 7, 1)
        statistics_layout.setRowStretch(8, 1)

        self.statistics_dock.setWidget(statistics_widget)

    def _create_actions_docks(self):
        """Create action dock widgets (Imaging, etc.)."""
        dock_features = (
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        dock_areas = (
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )

        # Imaging dock
        self.imaging_dock = QDockWidget("Imaging", self)
        self.imaging_dock.setObjectName("ImagingDock")
        self.imaging_dock.setAllowedAreas(dock_areas)
        self.imaging_dock.setFeatures(dock_features)

        imaging_widget = QWidget()
        imaging_layout = QHBoxLayout(imaging_widget)
        imaging_layout.setContentsMargins(8, 8, 8, 8)

        # Capture group
        capture_group = QGroupBox("Capture")
        capture_layout = QHBoxLayout(capture_group)
        capture_layout.setContentsMargins(8, 4, 8, 4)

        self.imaging_start_btn = QPushButton("Start")
        self.imaging_start_btn.clicked.connect(self._on_imaging_start)
        self.imaging_start_btn.setEnabled(False)  # Disabled until connected
        self.imaging_stop_btn = QPushButton("Stop")
        self.imaging_stop_btn.clicked.connect(self._on_imaging_stop)
        self.imaging_stop_btn.setEnabled(False)  # Disabled until exposing

        capture_layout.addWidget(self.imaging_start_btn)
        capture_layout.addWidget(self.imaging_stop_btn)

        # AutoFocus group
        autofocus_group = QGroupBox("AutoFocus")
        autofocus_layout = QHBoxLayout(autofocus_group)
        autofocus_layout.setContentsMargins(8, 4, 8, 4)

        self.autofocus_start_btn = QPushButton("Start")
        self.autofocus_start_btn.clicked.connect(self._on_autofocus_start)
        self.autofocus_start_btn.setEnabled(False)  # Disabled until connected
        self.autofocus_cancel_btn = QPushButton("Cancel")
        self.autofocus_cancel_btn.clicked.connect(self._on_autofocus_cancel)
        self.autofocus_cancel_btn.setEnabled(False)  # Disabled until autofocus running

        autofocus_layout.addWidget(self.autofocus_start_btn)
        autofocus_layout.addWidget(self.autofocus_cancel_btn)

        imaging_layout.addWidget(capture_group)
        imaging_layout.addWidget(autofocus_group)

        imaging_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.imaging_dock.setWidget(imaging_widget)

        # Guider dock
        self.guider_action_dock = QDockWidget("Guider", self)
        self.guider_action_dock.setObjectName("GuiderActionDock")
        self.guider_action_dock.setAllowedAreas(dock_areas)
        self.guider_action_dock.setFeatures(dock_features)

        guider_action_widget = QWidget()
        guider_action_layout = QHBoxLayout(guider_action_widget)
        guider_action_layout.setContentsMargins(8, 8, 8, 8)

        # Guiding group
        guiding_group = QGroupBox("Guiding")
        guiding_layout = QHBoxLayout(guiding_group)
        guiding_layout.setContentsMargins(8, 4, 8, 4)

        self.guiding_start_btn = QPushButton("Start")
        self.guiding_start_btn.clicked.connect(self._on_guiding_start)
        self.guiding_start_btn.setEnabled(False)  # Disabled until connected
        self.guiding_stop_btn = QPushButton("Stop")
        self.guiding_stop_btn.clicked.connect(self._on_guiding_stop)
        self.guiding_stop_btn.setEnabled(False)  # Disabled until guiding

        guiding_layout.addWidget(self.guiding_start_btn)
        guiding_layout.addWidget(self.guiding_stop_btn)

        guider_action_layout.addWidget(guiding_group)

        guider_action_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.guider_action_dock.setWidget(guider_action_widget)

        # Mount dock
        self.mount_action_dock = QDockWidget("Mount", self)
        self.mount_action_dock.setObjectName("MountActionDock")
        self.mount_action_dock.setAllowedAreas(dock_areas)
        self.mount_action_dock.setFeatures(dock_features)

        mount_action_widget = QWidget()
        mount_action_layout = QHBoxLayout(mount_action_widget)
        mount_action_layout.setContentsMargins(8, 8, 8, 8)

        # Parking group
        parking_group = QGroupBox("Parking")
        parking_layout = QHBoxLayout(parking_group)
        parking_layout.setContentsMargins(8, 4, 8, 4)

        self.mount_home_btn = QPushButton("Home")
        self.mount_home_btn.clicked.connect(self._on_mount_home)
        self.mount_home_btn.setEnabled(False)  # Disabled until connected
        self.mount_park_btn = QPushButton("Park")
        self.mount_park_btn.clicked.connect(self._on_mount_park)
        self.mount_park_btn.setEnabled(False)  # Disabled until connected
        self.mount_unpark_btn = QPushButton("Unpark")
        self.mount_unpark_btn.clicked.connect(self._on_mount_unpark)
        self.mount_unpark_btn.setEnabled(False)  # Disabled until connected

        parking_layout.addWidget(self.mount_home_btn)
        parking_layout.addWidget(self.mount_park_btn)
        parking_layout.addWidget(self.mount_unpark_btn)

        # Slew group
        slew_group = QGroupBox("Slew")
        slew_layout = QHBoxLayout(slew_group)
        slew_layout.setContentsMargins(8, 4, 8, 4)

        self.mount_slew_btn = QPushButton("Slew...")
        self.mount_slew_btn.clicked.connect(self._on_mount_slew)
        self.mount_slew_btn.setEnabled(False)  # Disabled until connected

        slew_layout.addWidget(self.mount_slew_btn)

        mount_action_layout.addWidget(parking_group)
        mount_action_layout.addWidget(slew_group)

        mount_action_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mount_action_dock.setWidget(mount_action_widget)

        # Spacer dock - absorbs extra space so action docks stay compact
        self.actions_spacer_dock = QDockWidget("", self)
        self.actions_spacer_dock.setObjectName("ActionsSpacerDock")
        self.actions_spacer_dock.setAllowedAreas(dock_areas)
        self.actions_spacer_dock.setFeatures(QDockWidget.DockWidgetMovable)  # No close button
        self.actions_spacer_dock.setTitleBarWidget(QWidget())  # Hide title bar

        spacer_widget = QWidget()
        spacer_widget.setMinimumWidth(0)
        spacer_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.actions_spacer_dock.setWidget(spacer_widget)

    def _create_image_panel(self, parent_layout):
        """Create the image display panel with tabs for Live View, Latest Image, and Live Stack."""
        # The size boxes are editable: apply a size once typing pauses, not on every keystroke
        self._size_timer = QTimer(self)
        self._size_timer.setSingleShot(True)
        self._size_timer.setInterval(600)
        self._size_timer.timeout.connect(self._on_size_changed)
        # Create tab widget instead of group box
        self.image_tabs = QTabWidget()
        self.image_tabs.setDocumentMode(False)
        themed_style(self.image_tabs, lambda: f"""
            QTabBar::tab {{
                border: 1px solid {COLORS['border']};
                border-bottom: none;
                padding: 4px 12px;
            }}
            QTabBar::tab:selected {{
                border: 2px solid {COLORS['accent']};
                border-bottom: none;
            }}
        """)

        # Tab 0: Live View (prepared image from NINA's imaging tab)
        liveview_widget = QWidget()
        liveview_layout = QVBoxLayout(liveview_widget)
        liveview_layout.setContentsMargins(5, 5, 5, 5)
        liveview_layout.setSpacing(2)

        # Live view settings row
        liveview_settings_layout = QHBoxLayout()
        liveview_settings_layout.setSpacing(8)

        liveview_settings_layout.addWidget(QLabel("Quality:"))
        self.liveview_quality_spin = QSpinBox()
        self.liveview_quality_spin.setRange(1, 100)
        self.liveview_quality_spin.setValue(80)
        self.liveview_quality_spin.setToolTip("JPEG quality 1-100 (lower = faster transfer)")
        self.liveview_quality_spin.valueChanged.connect(self._on_liveview_quality_changed)
        liveview_settings_layout.addWidget(self.liveview_quality_spin)

        liveview_settings_layout.addWidget(QLabel("Size:"))
        self.liveview_size_combo = QComboBox()
        self.liveview_size_combo.setEditable(True)
        self.liveview_size_combo.addItems(["400x300", "800x600", "1280x960", "1920x1080"])
        self.liveview_size_combo.setCurrentText("800x600")
        self.liveview_size_combo.setToolTip("Image size (WxH). Type a custom value or select a preset.")
        self.liveview_size_combo.setFixedWidth(110)
        self.liveview_size_combo.currentTextChanged.connect(lambda _text: self._size_timer.start())
        liveview_settings_layout.addWidget(self.liveview_size_combo)

        self.liveview_toggle_btn = QPushButton("Start")
        self.liveview_toggle_btn.setFixedWidth(70)
        self.liveview_toggle_btn.setToolTip("Start/stop live view from NINA's camera")
        self.liveview_toggle_btn.clicked.connect(self._on_liveview_toggle)
        liveview_settings_layout.addWidget(self.liveview_toggle_btn)

        liveview_settings_layout.addStretch()
        liveview_layout.addLayout(liveview_settings_layout, 0)

        self.liveview_label = ZoomableImageWidget(placeholder_text="Live view not active")
        liveview_layout.addWidget(self.liveview_label, 1)

        self.liveview_info_label = QLabel("")
        self.liveview_info_label.setAlignment(Qt.AlignCenter)
        self.liveview_info_label.setFixedHeight(20)
        themed_style(self.liveview_info_label, lambda: f"color: {COLORS['text_secondary']};")
        liveview_layout.addWidget(self.liveview_info_label, 0)

        self.image_tabs.addTab(liveview_widget, "Live View")

        # Tab 1: Latest Image
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)
        image_layout.setSpacing(2)

        # Image quality/size settings row
        image_settings_layout = QHBoxLayout()
        image_settings_layout.setSpacing(8)

        image_settings_layout.addWidget(QLabel("Quality:"))
        self.image_quality_spin = QSpinBox()
        self.image_quality_spin.setRange(-1, 100)
        self.image_quality_spin.setValue(-1)
        self.image_quality_spin.setToolTip("PNG (lossless), or 1-100 = JPEG quality")
        self.image_quality_spin.setSpecialValueText("PNG")  # -1
        skip_zero_quality(self.image_quality_spin)
        self.image_quality_spin.valueChanged.connect(self._on_image_quality_changed)
        image_settings_layout.addWidget(self.image_quality_spin)

        image_settings_layout.addWidget(QLabel("Size:"))
        self.image_size_combo = QComboBox()
        self.image_size_combo.setEditable(True)
        self.image_size_combo.addItems(["400x300", "800x600", "1280x960", "1920x1080"])
        self.image_size_combo.setCurrentText("800x600")
        self.image_size_combo.setToolTip("Image size (WxH). Type a custom value or select a preset.")
        self.image_size_combo.setFixedWidth(110)
        self.image_size_combo.currentTextChanged.connect(lambda _text: self._size_timer.start())
        image_settings_layout.addWidget(self.image_size_combo)

        image_settings_layout.addStretch()
        image_layout.addLayout(image_settings_layout, 0)

        self.image_label = ZoomableImageWidget(placeholder_text="No image available")
        image_layout.addWidget(self.image_label, 1)

        self.image_info_label = QLabel("Target: -- | Exp: --")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setFixedHeight(20)
        themed_style(self.image_info_label, lambda: f"color: {COLORS['text_secondary']};")
        image_layout.addWidget(self.image_info_label, 0)

        self.image_tabs.addTab(image_widget, "Latest Image")

        # Tab 2: Live Stack
        livestack_widget = QWidget()
        livestack_layout = QVBoxLayout(livestack_widget)
        livestack_layout.setContentsMargins(5, 5, 5, 5)
        livestack_layout.setSpacing(2)

        # Selection row for target and filter
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(8)

        selection_layout.addWidget(QLabel("Target:"))
        self.livestack_target_combo = QComboBox()
        self.livestack_target_combo.setToolTip("Select livestack target")
        self.livestack_target_combo.setMinimumWidth(120)
        # Grow with the names (the list is filled after the tab is first shown)
        self.livestack_target_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.livestack_target_combo.currentIndexChanged.connect(self._on_livestack_selection_changed)
        selection_layout.addWidget(self.livestack_target_combo)

        selection_layout.addWidget(QLabel("Filter:"))
        self.livestack_filter_combo = QComboBox()
        self.livestack_filter_combo.setToolTip("Select livestack filter")
        self.livestack_filter_combo.setMinimumWidth(80)
        self.livestack_filter_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.livestack_filter_combo.currentIndexChanged.connect(self._on_livestack_selection_changed)
        selection_layout.addWidget(self.livestack_filter_combo)

        self.livestack_annotations_check = QCheckBox("Annotations")
        self.livestack_annotations_check.setToolTip(
            "Label stars, deep-sky objects and the coordinate grid on the live stack.\n"
            "NINA plate-solves the stack's first frame once per stack.\n"
            "Which layers show is set in the image viewer's Annotations dialog.")
        self.livestack_annotations_check.toggled.connect(self._on_livestack_annotations_toggled)
        selection_layout.addWidget(self.livestack_annotations_check)
        self.livestack_annotations_status = QLabel("")
        themed_style(self.livestack_annotations_status, lambda: f"color: {COLORS['text_secondary']};")
        selection_layout.addWidget(self.livestack_annotations_status)

        selection_layout.addStretch()
        livestack_layout.addLayout(selection_layout, 0)

        # Livestack quality/size settings row
        livestack_settings_layout = QHBoxLayout()
        livestack_settings_layout.setSpacing(8)

        livestack_settings_layout.addWidget(QLabel("Quality:"))
        self.livestack_quality_spin = QSpinBox()
        self.livestack_quality_spin.setRange(-1, 100)
        self.livestack_quality_spin.setValue(100)
        self.livestack_quality_spin.setToolTip("PNG (lossless), or 1-100 = JPEG quality")
        self.livestack_quality_spin.setSpecialValueText("PNG")  # -1
        skip_zero_quality(self.livestack_quality_spin)
        self.livestack_quality_spin.valueChanged.connect(self._on_livestack_quality_changed)
        livestack_settings_layout.addWidget(self.livestack_quality_spin)

        livestack_settings_layout.addWidget(QLabel("Size:"))
        self.livestack_size_combo = QComboBox()
        self.livestack_size_combo.setEditable(True)
        self.livestack_size_combo.addItems(["400x300", "800x600", "1280x960", "1920x1080"])
        self.livestack_size_combo.setCurrentText("800x600")
        self.livestack_size_combo.setToolTip("Image size (WxH). Type a custom value or select a preset.")
        self.livestack_size_combo.setFixedWidth(110)
        self.livestack_size_combo.currentTextChanged.connect(lambda _text: self._size_timer.start())
        livestack_settings_layout.addWidget(self.livestack_size_combo)

        livestack_settings_layout.addStretch()
        livestack_layout.addLayout(livestack_settings_layout, 0)

        # Track available stacks to avoid unnecessary updates
        self._livestack_available_stacks = []
        self._updating_livestack_combos = False
        # Following the sequence's target: names of the containers the sequence is
        # running in, names still waiting for their first stack, and the last target selected
        self._sequence_container_names = None
        self._livestack_follow_pending = None
        self._livestack_followed_target = None
        # Live stack annotations: the (target, filter) stack they're for, its frame
        # count (a drop means the stack restarted), and NINA's plate solve of it
        self._ls_annot_key = None
        self._ls_annot_count = None
        self._ls_annot_solution = None
        self._ls_annot_solving = False
        self._ls_annot_renderer = None

        self.livestack_label = ZoomableImageWidget(placeholder_text="Live stack not active")
        livestack_layout.addWidget(self.livestack_label, 1)

        self.livestack_info_label = QLabel("")
        self.livestack_info_label.setAlignment(Qt.AlignCenter)
        self.livestack_info_label.setFixedHeight(28)
        themed_style(self.livestack_info_label, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_px(16)};")
        livestack_layout.addWidget(self.livestack_info_label, 0)

        self.image_tabs.addTab(livestack_widget, "Live Stack")

        # Default to Latest Image tab (Live View is tab 0 but off by default)
        self.image_tabs.setCurrentIndex(1)
        self.image_tabs.currentChanged.connect(self._on_image_tab_changed)

        parent_layout.addWidget(self.image_tabs, 1)  # stretch factor 1 to expand

    def _create_guiding_dock(self):
        """Create the Guiding Graph dock widget."""
        self.guiding_dock = QDockWidget("Guiding Graph", self)
        self.guiding_dock.setObjectName("GuidingGraphDock")
        self.guiding_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.guiding_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        # Guiding graph content
        guiding_widget = QWidget()
        guiding_layout = QVBoxLayout(guiding_widget)
        guiding_layout.setContentsMargins(5, 5, 5, 5)

        self.guiding_graph = GuidingGraph(self)
        self.guiding_graph.setMinimumHeight(120)
        self.guiding_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        guiding_layout.addWidget(self.guiding_graph)

        self.guiding_dock.setWidget(guiding_widget)

    def _create_image_history_dock(self):
        """Create the Image History dock widget."""
        self.image_history_dock = QDockWidget("Image History", self)
        self.image_history_dock.setObjectName("ImageHistoryDock")
        self.image_history_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.image_history_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(5, 5, 5, 5)

        self.image_history_list = QListWidget()
        self.image_history_list.setViewMode(QListWidget.IconMode)
        self.image_history_list.setIconSize(QSize(80, 60))
        self.image_history_list.setGridSize(QSize(90, 80))
        self.image_history_list.setResizeMode(QListWidget.Adjust)
        self.image_history_list.setMovement(QListWidget.Static)
        self.image_history_list.itemClicked.connect(self._on_history_item_clicked)
        history_layout.addWidget(self.image_history_list)

        self.image_history_dock.setWidget(history_widget)

    def _create_autofocus_dock(self):
        """Create the AutoFocus Graph dock widget (live V-curve from NINA events)."""
        self.autofocus_dock = QDockWidget("AutoFocus Graph", self)
        self.autofocus_dock.setObjectName("AutofocusGraphDock")
        self.autofocus_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.autofocus_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        autofocus_widget = QWidget()
        autofocus_layout = QVBoxLayout(autofocus_widget)
        autofocus_layout.setContentsMargins(5, 5, 5, 5)

        self.autofocus_graph = AutofocusGraph(self)
        self.autofocus_graph.setMinimumHeight(200)
        self.autofocus_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        autofocus_layout.addWidget(self.autofocus_graph)

        self.autofocus_info_label = QLabel("")
        themed_style(self.autofocus_info_label, lambda: f"color: {COLORS['text_secondary']};")
        autofocus_layout.addWidget(self.autofocus_info_label)

        self.autofocus_dock.setWidget(autofocus_widget)

        # Autofocus run state, driven by NINA events
        self._autofocus_running = False
        self._focuser_connected = False
        self._autofocus_stale_timer = QTimer(self)
        self._autofocus_stale_timer.setSingleShot(True)
        self._autofocus_stale_timer.setInterval(AUTOFOCUS_STALE_SECONDS * 1000)
        self._autofocus_stale_timer.timeout.connect(self._on_autofocus_stale)

    def _create_event_log_dock(self):
        """Create the NINA Event Log dock widget."""
        self.event_log_dock = QDockWidget("Event Log", self)
        self.event_log_dock.setObjectName("EventLogDock")
        self.event_log_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.event_log_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(5, 5, 5, 5)

        self.event_log_table = QTableWidget(0, 3)
        self.event_log_table.setHorizontalHeaderLabels(["Time", "Event", "Details"])
        self.event_log_table.verticalHeader().setVisible(False)
        self.event_log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.event_log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_log_table.setWordWrap(False)
        header = self.event_log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        log_layout.addWidget(self.event_log_table)

        self.event_log_dock.setWidget(log_widget)

    def _create_sequence_dock(self):
        """Create the Sequence dock: current activity summary above the sequence tree."""
        self.sequence_dock = QDockWidget("Sequence", self)
        self.sequence_dock.setObjectName("SequenceDock")
        self.sequence_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.sequence_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        # Current activity
        activity_group = QGroupBox("Current Activity")
        activity_layout = QGridLayout(activity_group)
        activity_layout.setColumnStretch(1, 1)

        buttons_layout = QHBoxLayout()
        self.sequence_start_btn = QPushButton("Start")
        self.sequence_start_btn.setToolTip("Start the sequence loaded in NINA")
        self.sequence_start_btn.setEnabled(False)
        self.sequence_start_btn.clicked.connect(self._on_sequence_start)
        buttons_layout.addWidget(self.sequence_start_btn)
        self.sequence_stop_btn = QPushButton("Stop")
        self.sequence_stop_btn.setToolTip("Stop the running sequence")
        self.sequence_stop_btn.setEnabled(False)
        self.sequence_stop_btn.clicked.connect(self._on_sequence_stop)
        buttons_layout.addWidget(self.sequence_stop_btn)
        buttons_layout.addStretch()
        activity_layout.addLayout(buttons_layout, 0, 0, 1, 2)

        self._sequence_activity_rows = {}
        for row, (key, title) in enumerate((
                ('status', "Status:"), ('container', "Container:"), ('now', "Now:"),
                ('loop', "Loop:"), ('next', "Next:"))):
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            value_label = QLabel("--")
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            activity_layout.addWidget(title_label, row + 1, 0)
            activity_layout.addWidget(value_label, row + 1, 1)
            self._sequence_activity_rows[key] = (title_label, value_label)
        activity_layout.setRowStretch(len(self._sequence_activity_rows) + 1, 1)

        # Sequence tree
        self.sequence_tree = QTreeWidget()
        self.sequence_tree.setColumnCount(3)
        self.sequence_tree.setHeaderLabels(["Item", "Status", "Details"])
        self.sequence_tree.setUniformRowHeights(True)
        self.sequence_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.sequence_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.sequence_tree.setColumnWidth(0, 220)

        self.sequence_panel = SequencePanel(activity_group, self.sequence_tree)
        self.sequence_dock.setWidget(self.sequence_panel)

        self._sequence_running_key = None  # Path of the deepest running tree item, to follow it
        self._sequence_state = None  # sequence_activity() state; 'none' when no sequence is loaded
        self._sequence_command_busy = False  # A start/stop request is in progress
        # Only poll the sequence while the dock can be seen
        self.sequence_dock.visibilityChanged.connect(self._on_sequence_dock_visibility)
        theme_manager().theme_changed.connect(self._recolor_sequence_tree)

    def _setup_view_menu(self):
        """Set up the View menu for panel visibility and layout reset."""
        view_menu = self.menuBar().addMenu("View")

        # Actions docks
        actions_menu = view_menu.addMenu("Actions")
        actions_menu.addAction(self.imaging_dock.toggleViewAction())
        actions_menu.addAction(self.guider_action_dock.toggleViewAction())
        actions_menu.addAction(self.mount_action_dock.toggleViewAction())

        # Equipment docks
        equipment_menu = view_menu.addMenu("Equipment")
        equipment_menu.addAction(self.camera_dock.toggleViewAction())
        equipment_menu.addAction(self.mount_dock.toggleViewAction())
        equipment_menu.addAction(self.guider_dock.toggleViewAction())
        equipment_menu.addAction(self.filterwheel_dock.toggleViewAction())
        equipment_menu.addAction(self.focuser_dock.toggleViewAction())
        equipment_menu.addAction(self.statistics_dock.toggleViewAction())

        view_menu.addAction(self.guiding_dock.toggleViewAction())
        view_menu.addAction(self.autofocus_dock.toggleViewAction())
        view_menu.addAction(self.event_log_dock.toggleViewAction())
        view_menu.addAction(self.image_history_dock.toggleViewAction())
        view_menu.addAction(self.sequence_dock.toggleViewAction())
        view_menu.addSeparator()
        reset_action = view_menu.addAction("Reset Layout")
        reset_action.triggered.connect(self._reset_layout)

    def _set_default_layout(self):
        """Set default dock positions."""
        self._apply_dock_corners()

        # Action docks at top, side by side with spacer absorbing extra space
        self.addDockWidget(Qt.TopDockWidgetArea, self.imaging_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.guider_action_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.mount_action_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.actions_spacer_dock)
        # Arrange horizontally: Imaging | Guider | Mount | Spacer
        self.splitDockWidget(self.imaging_dock, self.guider_action_dock, Qt.Horizontal)
        self.splitDockWidget(self.guider_action_dock, self.mount_action_dock, Qt.Horizontal)
        self.splitDockWidget(self.mount_action_dock, self.actions_spacer_dock, Qt.Horizontal)

        # Add equipment docks to left area, tabified together
        self.addDockWidget(Qt.LeftDockWidgetArea, self.camera_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.mount_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.guider_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filterwheel_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.focuser_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.statistics_dock)

        # Stack them vertically in the left area
        self.splitDockWidget(self.camera_dock, self.mount_dock, Qt.Vertical)
        self.splitDockWidget(self.mount_dock, self.guider_dock, Qt.Vertical)
        self.splitDockWidget(self.guider_dock, self.filterwheel_dock, Qt.Vertical)
        self.splitDockWidget(self.filterwheel_dock, self.focuser_dock, Qt.Vertical)
        self._tabify_statistics_dock()

        # Add guiding graph at bottom, with autofocus graph and event log as tabs
        self.addDockWidget(Qt.BottomDockWidgetArea, self.guiding_dock)
        self._tabify_bottom_docks()
        # Sequence beside them
        self._place_sequence_dock()

        # Add image history on right
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_history_dock)

        # Set initial sizes
        self.resizeDocks([self.imaging_dock], [60], Qt.Vertical)  # Compact height for imaging
        self.resizeDocks([self.camera_dock], [250], Qt.Horizontal)
        self.resizeDocks([self.guiding_dock], [200], Qt.Vertical)

    def _tabify_bottom_docks(self):
        """Place the autofocus graph and event log as tabs alongside the guiding graph."""
        # Re-tabifying a dock that's already in the group leaves an empty tab bar
        # layered over the real one, which swallows clicks
        already_tabbed = self.tabifiedDockWidgets(self.guiding_dock)
        for dock in (self.autofocus_dock, self.event_log_dock):
            if dock not in already_tabbed:
                self.addDockWidget(Qt.BottomDockWidgetArea, dock)
                self.tabifyDockWidget(self.guiding_dock, dock)
        self.guiding_dock.raise_()

    def _apply_dock_corners(self):
        """Top action docks span the full width; the left equipment column runs down
        to the window's bottom edge, with the bottom docks beside it.

        With the bottom docks below the column instead, the two stack and the
        window is too tall for a 1080p screen. saveState() includes the corners,
        so this is reapplied after restoring older saved layouts.
        """
        self.setCorner(Qt.TopLeftCorner, Qt.TopDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.TopDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)

    def _tabify_statistics_dock(self):
        """Make Statistics a tab beside Focuser (in front, as it changes with every frame).

        Six stacked equipment docks are too tall for a 1080p screen.
        """
        self.tabifyDockWidget(self.focuser_dock, self.statistics_dock)
        self.statistics_dock.raise_()

    def _place_sequence_dock(self):
        """Put the sequence dock at the right end of the bottom area, beside the graphs."""
        # (splitDockWidget would add it as another tab of the tabbed guiding graph)
        self.removeDockWidget(self.sequence_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.sequence_dock, Qt.Horizontal)
        self.sequence_dock.show()
        width = self.width()
        self.resizeDocks([self.guiding_dock, self.sequence_dock],
                         [int(width * 0.55), int(width * 0.45)], Qt.Horizontal)

    def _reset_layout(self):
        """Reset dock layout to defaults."""
        # Remove all docks first
        self.removeDockWidget(self.imaging_dock)
        self.removeDockWidget(self.guider_action_dock)
        self.removeDockWidget(self.mount_action_dock)
        self.removeDockWidget(self.actions_spacer_dock)
        self.removeDockWidget(self.camera_dock)
        self.removeDockWidget(self.mount_dock)
        self.removeDockWidget(self.guider_dock)
        self.removeDockWidget(self.filterwheel_dock)
        self.removeDockWidget(self.focuser_dock)
        self.removeDockWidget(self.statistics_dock)
        self.removeDockWidget(self.guiding_dock)
        self.removeDockWidget(self.autofocus_dock)
        self.removeDockWidget(self.event_log_dock)
        self.removeDockWidget(self.image_history_dock)
        self.removeDockWidget(self.sequence_dock)

        # Re-add in default positions
        self._set_default_layout()

        # Show all docks
        self.imaging_dock.show()
        self.guider_action_dock.show()
        self.mount_action_dock.show()
        self.actions_spacer_dock.show()
        self.camera_dock.show()
        self.mount_dock.show()
        self.guider_dock.show()
        self.filterwheel_dock.show()
        self.focuser_dock.show()
        self.statistics_dock.show()
        self.guiding_dock.show()
        self.autofocus_dock.show()
        self.event_log_dock.show()
        self.image_history_dock.show()
        self.sequence_dock.show()
        self.guiding_dock.raise_()
        self.statistics_dock.raise_()

    def _restore_settings(self):
        """Restore saved dock layout, refresh rate, and image quality settings."""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Restore image quality/size settings (suppress re-fetch during restore)
        self._restoring_settings = True
        image_quality = settings.value("nina_image_quality", -1, type=int)
        image_size = settings.value("nina_image_size", "800x600", type=str)
        livestack_quality = settings.value("nina_livestack_quality", 100, type=int)
        livestack_size = settings.value("nina_livestack_size", "800x600", type=str)
        liveview_quality = settings.value("nina_liveview_quality", 80, type=int)
        liveview_size = settings.value("nina_liveview_size", "800x600", type=str)

        self.image_quality_spin.setValue(image_quality)
        self.image_size_combo.setCurrentText(image_size)
        self.livestack_quality_spin.setValue(livestack_quality)
        self.livestack_size_combo.setCurrentText(livestack_size)
        self.liveview_quality_spin.setValue(liveview_quality)
        self.liveview_size_combo.setCurrentText(liveview_size)
        self._restoring_settings = False

        self.livestack_annotations_check.setChecked(
            settings.value("nina_livestack_annotations", False, type=bool))

        self.sequence_panel.set_activity_sizes(
            settings.value("nina_sequence_activity_width", SequencePanel.DEFAULT_ACTIVITY_WIDTH, type=int),
            settings.value("nina_sequence_activity_height", SequencePanel.DEFAULT_ACTIVITY_HEIGHT, type=int),
        )

        # Apply to worker if already running
        self._apply_image_settings_to_worker()

        # Check for saved dock state
        dock_state = settings.value("nina_dashboard_dock_state")

        if dock_state is not None:
            # Handle different types that QSettings might return
            if isinstance(dock_state, QByteArray):
                state_bytes = dock_state
            elif isinstance(dock_state, bytes):
                state_bytes = QByteArray(dock_state)
            else:
                logger.debug(f"Unexpected dock_state type: {type(dock_state)}")
                self._set_default_layout()
                return

            if not state_bytes.isEmpty():
                # Add docks first (required for restoreState), then restore positions
                self.addDockWidget(Qt.TopDockWidgetArea, self.imaging_dock)
                self.addDockWidget(Qt.TopDockWidgetArea, self.guider_action_dock)
                self.addDockWidget(Qt.TopDockWidgetArea, self.mount_action_dock)
                self.addDockWidget(Qt.TopDockWidgetArea, self.actions_spacer_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.camera_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.mount_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.guider_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.filterwheel_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.focuser_dock)
                self.addDockWidget(Qt.LeftDockWidgetArea, self.statistics_dock)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.guiding_dock)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.autofocus_dock)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.event_log_dock)
                self.addDockWidget(Qt.RightDockWidgetArea, self.image_history_dock)
                self.addDockWidget(Qt.RightDockWidgetArea, self.sequence_dock)
                self.restoreState(state_bytes)
                self._apply_dock_corners()
                # Layouts saved before a dock existed don't place it
                layout_version = settings.value("nina_dashboard_dock_layout_version", 1, type=int)
                if layout_version < 2:  # Autofocus graph and event log
                    self._tabify_bottom_docks()
                if layout_version < 3:  # Sequence
                    self._place_sequence_dock()
                if layout_version < 4:  # Fit 1080p: Statistics no longer stacked under Focuser
                    stacked_in_left_column = all(
                        self.dockWidgetArea(dock) == Qt.LeftDockWidgetArea and not dock.isFloating()
                        for dock in (self.focuser_dock, self.statistics_dock))
                    # Leave it alone if the user moved it or already tabbed it
                    if stacked_in_left_column and not self.tabifiedDockWidgets(self.statistics_dock):
                        self._tabify_statistics_dock()
                logger.debug(f"Restored dock state, size={state_bytes.size()}")
            else:
                self._set_default_layout()
        else:
            self._set_default_layout()

    def _save_settings(self):
        """Save dock layout and refresh rate."""
        settings = QSettings("CosmosCollection", "CosmosCollection")

        # Only save dock state if docks are properly attached to the window
        camera_area = self.dockWidgetArea(self.camera_dock)
        if camera_area == Qt.NoDockWidgetArea:
            logger.debug("Skipping dock state save - docks not attached")
            settings.sync()
            return

        # Save dock state
        dock_state = self.saveState()
        state_bytes = bytes(dock_state.data())
        settings.setValue("nina_dashboard_dock_state", state_bytes)
        settings.setValue("nina_dashboard_dock_layout_version", self.DOCK_LAYOUT_VERSION)
        logger.debug(f"Saved dock state, size={dock_state.size()}")

        # Save image quality/size settings
        settings.setValue("nina_image_quality", self.image_quality_spin.value())
        settings.setValue("nina_image_size", self.image_size_combo.currentText())
        settings.setValue("nina_livestack_quality", self.livestack_quality_spin.value())
        settings.setValue("nina_livestack_size", self.livestack_size_combo.currentText())
        settings.setValue("nina_liveview_quality", self.liveview_quality_spin.value())
        settings.setValue("nina_liveview_size", self.liveview_size_combo.currentText())
        settings.setValue("nina_livestack_annotations", self.livestack_annotations_check.isChecked())
        activity_width, activity_height = self.sequence_panel.activity_sizes()
        settings.setValue("nina_sequence_activity_width", activity_width)
        settings.setValue("nina_sequence_activity_height", activity_height)

        settings.sync()

    def _auto_connect(self):
        """Automatically attempt to connect when the window opens."""
        if NINAIntegration.is_enabled():
            self._reconnect()
        else:
            self.connection_label.setText("Connection: NINA integration disabled")
            themed_style(self.connection_label, lambda: f"color: {COLORS['warning']};")
            self.status_label.setText("Enable NINA integration in Settings to use this dashboard")

    def _reconnect(self):
        """Reconnect to NINA."""
        # Stop existing worker
        self._stop_worker()
        # AF state is rebuilt from the event backlog on connect
        self._autofocus_running = False
        self._autofocus_stale_timer.stop()
        # Select the sequence's current target again once connected
        self._sequence_container_names = None
        self._livestack_followed_target = None
        # The new worker reloads the recent thumbnails
        self.image_history_list.clear()

        self.connection_label.setText("Connection: Connecting...")
        themed_style(self.connection_label, lambda: f"color: {COLORS['info']};")

        # Get settings
        host, port = NINAIntegration.get_settings()

        # Create and start worker (uses adaptive polling)
        self.worker = NINAStatusWorker(host, port)
        self.worker.connection_changed.connect(self._on_connection_changed)
        self.worker.status_updated.connect(self._on_status_updated)
        self.worker.image_updated.connect(self._on_image_updated)
        self.worker.image_fetching.connect(self._on_image_fetching)
        self.worker.livestack_updated.connect(self._on_livestack_updated)
        self.worker.livestack_fetching.connect(self._on_livestack_fetching)
        self.worker.liveview_updated.connect(self._on_liveview_updated)
        self.worker.guiding_updated.connect(self._on_guiding_updated)
        self.worker.event_occurred.connect(self._on_event_occurred)
        self.worker.events_loaded.connect(self._on_events_loaded)
        self.worker.autofocus_report.connect(self._on_autofocus_report)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.history_thumbnail.connect(self._on_history_thumbnail)
        self.worker.history_reset.connect(self.image_history_list.clear)
        self.worker.sequence_updated.connect(self._on_sequence_updated)
        self.worker.set_sequence_active(self.sequence_dock.isVisible())
        self._apply_image_settings_to_worker()
        self.worker.start()

    def _stop_worker(self):
        """Stop the worker thread."""
        worker, self.worker = self.worker, None
        if worker:
            worker.stop()
            if not worker.wait(2000):
                # Still inside a slow request (e.g. NINA unreachable). Stop listening and
                # let it finish in the background - destroying it mid-run aborts the app.
                for signal in (worker.connection_changed, worker.status_updated, worker.image_updated,
                               worker.image_fetching, worker.livestack_updated, worker.livestack_fetching,
                               worker.liveview_updated, worker.guiding_updated, worker.event_occurred,
                               worker.events_loaded, worker.autofocus_report, worker.error_occurred,
                               worker.history_thumbnail, worker.history_reset, worker.sequence_updated):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)  # Signals with no connections
                        try:
                            signal.disconnect()
                        except (RuntimeError, TypeError):
                            pass
                keep_alive_until_finished(worker)
        # Reset live view UI state
        self.liveview_toggle_btn.setText("Start")
        self.liveview_info_label.setText("")

    def _on_connection_changed(self, connected, version, host, port):
        """Handle connection state change."""
        self._connected = connected
        self._version = version

        if connected:
            self.connection_label.setText(f"Connection: Connected to NINA {version} ({host}:{port})")
            themed_style(self.connection_label, lambda: f"color: {COLORS['success']};")
            self.status_label.setText("Connected - adaptive polling active")
            themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
            # Enable start buttons (stop/cancel remain disabled until active)
            self.imaging_start_btn.setEnabled(True)
            self.autofocus_start_btn.setEnabled(True)
            self.guiding_start_btn.setEnabled(True)
            self.mount_home_btn.setEnabled(True)
            self.mount_park_btn.setEnabled(True)
            self.mount_unpark_btn.setEnabled(True)
            self.mount_slew_btn.setEnabled(True)
        else:
            self.connection_label.setText("Connection: Disconnected")
            themed_style(self.connection_label, lambda: f"color: {COLORS['error']};")
            self.status_label.setText("NINA disconnected - click Reconnect to retry")
            themed_style(self.status_label, lambda: f"color: {COLORS['warning']};")
            # Disable all action buttons when disconnected
            self.imaging_start_btn.setEnabled(False)
            self.imaging_stop_btn.setEnabled(False)
            self.autofocus_start_btn.setEnabled(False)
            self.autofocus_cancel_btn.setEnabled(False)
            self.guiding_start_btn.setEnabled(False)
            self.guiding_stop_btn.setEnabled(False)
            self.mount_home_btn.setEnabled(False)
            self.mount_park_btn.setEnabled(False)
            self.mount_unpark_btn.setEnabled(False)
            self.mount_slew_btn.setEnabled(False)
        self._update_sequence_buttons()

    def _on_status_updated(self, status_data):
        """Handle status update from worker."""
        self._last_update = datetime.now()

        # Update last update display
        update_str = format_time(self._last_update)
        self.countdown_label.setText(f"Last update: {update_str}")

        # Update camera info
        camera = status_data.get('camera', {})
        if camera:
            is_exposing = camera.get('IsExposing', False)
            camera_connected = camera.get('Connected', False)

            # Update imaging button states based on connection and exposure status
            self.imaging_start_btn.setEnabled(self._connected and camera_connected and not is_exposing)
            self.imaging_stop_btn.setEnabled(self._connected and camera_connected and is_exposing)

            # Disable live view start when camera is busy; auto-stop if exposure begins
            if is_exposing and hasattr(self, 'worker') and self.worker and self.worker._liveview_active:
                self.worker.set_liveview_active(False)
                self.liveview_toggle_btn.setText("Start")
                self.liveview_info_label.setText("Stopped (camera busy)")
            can_liveview = self._connected and camera_connected and not is_exposing
            self.liveview_toggle_btn.setEnabled(can_liveview)

            name = camera.get('Name') or camera.get('DeviceName', '--')
            self.camera_name_label.setText(name)

            connected = camera.get('Connected', False)
            if not connected:
                self.camera_status_label.setText("Disconnected")
                themed_style(self.camera_status_label, lambda: f"color: {COLORS['text_disabled']};")
                self.camera_progress.setValue(0)
                self.camera_temp_label.setText("--")
                self._exposure_end_time = None
                self._exposure_total_time = None
                self._camera_seen_idle = False
                self._last_cooling_enabled = None  # Sync again when the camera reconnects
                # Disable controls when disconnected (block signals to prevent callbacks)
                self.camera_cooling_checkbox.blockSignals(True)
                self.camera_cooling_checkbox.setChecked(False)
                self.camera_cooling_checkbox.setEnabled(False)
                self.camera_cooling_checkbox.blockSignals(False)

                self.camera_target_temp_spinbox.blockSignals(True)
                self.camera_target_temp_spinbox.setEnabled(False)
                self.camera_target_temp_spinbox.blockSignals(False)

                self.camera_dewheater_checkbox.blockSignals(True)
                self.camera_dewheater_checkbox.setChecked(False)
                self.camera_dewheater_checkbox.setEnabled(False)
                self.camera_dewheater_checkbox.blockSignals(False)
            else:
                # Determine camera state
                if is_exposing:
                    self.camera_status_label.setText("Exposing")
                    themed_style(self.camera_status_label, lambda: f"color: {COLORS['success']};")
                else:
                    self.camera_status_label.setText("Idle")
                    themed_style(self.camera_status_label, lambda: f"color: {COLORS['text']};")

                # Exposure progress - calculate from ExposureEndTime
                exposure_end_str = camera.get('ExposureEndTime')
                if is_exposing and exposure_end_str:
                    try:
                        # Parse ISO format timestamp
                        from datetime import timezone
                        exposure_end = datetime.fromisoformat(exposure_end_str.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc) if exposure_end.tzinfo else datetime.now()

                        remaining = max(0, (exposure_end - now).total_seconds())

                        # New exposure detected
                        if self._exposure_end_time != exposure_end:
                            saw_start = self._camera_seen_idle or self._exposure_end_time is not None
                            self._exposure_end_time = exposure_end
                            self._exposure_total_time = remaining
                            if not saw_start:
                                # Opened mid-exposure: the remaining time isn't the length.
                                # NINA doesn't report the length, so use the latest
                                # image's (normally the same sub length) if it fits.
                                last_exposure = finite_number(
                                    (status_data.get('statistics') or {}).get('ExposureTime'))
                                if last_exposure and last_exposure >= remaining:
                                    self._exposure_total_time = last_exposure

                        if self._exposure_total_time and self._exposure_total_time > 0:
                            elapsed = self._exposure_total_time - remaining
                            progress_pct = (elapsed / self._exposure_total_time) * 100
                            self.camera_progress.setValue(int(min(100, max(0, progress_pct))))
                        else:
                            self.camera_progress.setValue(0)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing ExposureEndTime: {e}")
                        self.camera_progress.setValue(0)
                elif is_exposing:
                    # Exposing but no end time info
                    self.camera_progress.setValue(0)
                else:
                    self.camera_progress.setValue(0)
                    self._exposure_end_time = None
                    self._exposure_total_time = None
                    self._camera_seen_idle = True

                # Camera temperature
                temp = finite_number(camera.get('Temperature'))
                if temp is not None:
                    self.camera_temp_label.setText(f"{temp:.1f}°C")
                else:
                    self.camera_temp_label.setText("--")

                # Sync cooling controls with camera state (skip if user recently changed)
                self.camera_cooling_checkbox.setEnabled(not self._cooling_request_pending)
                self.camera_target_temp_spinbox.setEnabled(True)
                self.camera_dewheater_checkbox.setEnabled(not self._dewheater_request_pending)

                # Only sync cooling on/off state on initial load (when we haven't set it yet)
                # After user sets it, we respect their choice and don't override
                # Note: We don't sync target temp - that's user-controlled only
                cooling_on_raw = camera.get('CoolerOn', False)
                cooler_power = camera.get('CoolerPower', 0)
                target_temp = camera.get('TargetTemp') or camera.get('TemperatureSetPoint')
                at_target = camera.get('AtTargetTemp', False)

                # Handle potential string values from API
                if isinstance(cooling_on_raw, str):
                    cooling_on = cooling_on_raw.lower() in ('true', '1', 'yes')
                else:
                    cooling_on = bool(cooling_on_raw)
                    logger.debug(f"[Cooling Sync] API: CoolerOn={cooling_on_raw!r}, CoolerPower={cooler_power}, TargetTemp={target_temp}, AtTarget={at_target}")

                # Follow NINA's cooler state, e.g. when a sequence warms the camera, but not
                # while a change made here is still taking effect (or being sent)
                if (not self._user_changing_cooling and not self._cooling_request_pending
                        and (self._last_cooling_enabled is None
                             or self.camera_cooling_checkbox.isChecked() != cooling_on)):
                    logger.debug(f"[Cooling] Sync - setting checkbox to {cooling_on}")
                    # Block signals to prevent triggering callbacks
                    self.camera_cooling_checkbox.blockSignals(True)
                    self.camera_cooling_checkbox.setChecked(cooling_on)
                    self.camera_cooling_checkbox.blockSignals(False)
                    self._last_cooling_enabled = cooling_on

                # Only sync dew heater state if user isn't actively changing it
                dewheater_on = camera.get('DewHeaterOn', False)
                logger.debug(f"[DewHeater Sync] API DewHeaterOn={dewheater_on}, _user_changing_dewheater={self._user_changing_dewheater}")
                if not self._user_changing_dewheater:
                    self.camera_dewheater_checkbox.blockSignals(True)
                    self.camera_dewheater_checkbox.setChecked(dewheater_on)
                    self.camera_dewheater_checkbox.blockSignals(False)
        else:
            # No camera data from API - show as disconnected
            self.camera_name_label.setText("--")
            self.camera_status_label.setText("Disconnected")
            themed_style(self.camera_status_label, lambda: f"color: {COLORS['text_disabled']};")
            self.camera_progress.setValue(0)
            self.camera_temp_label.setText("--")
            self.camera_cooling_checkbox.blockSignals(True)
            self.camera_cooling_checkbox.setChecked(False)
            self.camera_cooling_checkbox.setEnabled(False)
            self.camera_cooling_checkbox.blockSignals(False)
            self.camera_target_temp_spinbox.setEnabled(False)
            self.camera_dewheater_checkbox.blockSignals(True)
            self.camera_dewheater_checkbox.setChecked(False)
            self.camera_dewheater_checkbox.setEnabled(False)
            self.camera_dewheater_checkbox.blockSignals(False)
            self._last_cooling_enabled = None

        # Update mount info
        mount = status_data.get('mount', {})
        if mount:
            name = mount.get('Name') or mount.get('DeviceName', '--')
            self.mount_name_label.setText(name)

            mount_connected = mount.get('Connected', False)
            if not mount_connected:
                self.mount_status_label.setText("Disconnected")
                themed_style(self.mount_status_label, lambda: f"color: {COLORS['text_disabled']};")
                self.mount_coords_label.setText("--")
                # Disable mount buttons when mount disconnected
                self.mount_home_btn.setEnabled(False)
                self.mount_park_btn.setEnabled(False)
                self.mount_unpark_btn.setEnabled(False)
                self.mount_slew_btn.setEnabled(False)
            else:
                tracking = mount.get('TrackingEnabled', False) or mount.get('Tracking', False)
                slewing = mount.get('Slewing', False)
                at_park = mount.get('AtPark', False)

                if slewing:
                    self.mount_status_label.setText("Slewing")
                    themed_style(self.mount_status_label, lambda: f"color: {COLORS['info']};")
                elif at_park:
                    self.mount_status_label.setText("Parked")
                    themed_style(self.mount_status_label, lambda: f"color: {COLORS['text']};")
                elif tracking:
                    self.mount_status_label.setText("Tracking")
                    themed_style(self.mount_status_label, lambda: f"color: {COLORS['success']};")
                else:
                    self.mount_status_label.setText("Idle")
                    themed_style(self.mount_status_label, lambda: f"color: {COLORS['text']};")

                # Update mount button states based on park status
                self.mount_home_btn.setEnabled(self._connected and not at_park and not slewing)
                self.mount_park_btn.setEnabled(self._connected and not at_park and not slewing)
                self.mount_unpark_btn.setEnabled(self._connected and at_park)
                self.mount_slew_btn.setEnabled(self._connected and not at_park and not slewing)

                # Coordinates
                ra = finite_number(mount.get('RightAscension'))
                if ra is None:
                    ra = finite_number(mount.get('RA'))
                dec = finite_number(mount.get('Declination'))
                if dec is None:
                    dec = finite_number(mount.get('Dec'))
                if ra is not None and dec is not None:
                    # Convert RA from hours to HH:MM:SS
                    ra_h = int(ra)
                    ra_m = int((ra - ra_h) * 60)
                    ra_s = ((ra - ra_h) * 60 - ra_m) * 60
                    # Dec in degrees
                    dec_sign = '+' if dec >= 0 else '-'
                    dec_abs = abs(dec)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    self.mount_coords_label.setText(f"{ra_h:02d}h{ra_m:02d}m / {dec_sign}{dec_d}d{dec_m:02d}m")
                else:
                    self.mount_coords_label.setText("--")
        else:
            # No mount data from API - show as disconnected
            self.mount_name_label.setText("--")
            self.mount_status_label.setText("Disconnected")
            themed_style(self.mount_status_label, lambda: f"color: {COLORS['text_disabled']};")
            self.mount_coords_label.setText("--")
            # Disable mount buttons when no mount data
            self.mount_home_btn.setEnabled(False)
            self.mount_park_btn.setEnabled(False)
            self.mount_unpark_btn.setEnabled(False)
            self.mount_slew_btn.setEnabled(False)

        # Update guider info
        guider = status_data.get('guider', {})
        if guider:
            name = guider.get('Name') or guider.get('DeviceName', '--')
            self.guider_name_label.setText(name)

            guider_connected = guider.get('Connected', False)
            if not guider_connected:
                self.guider_status_label.setText("Disconnected")
                themed_style(self.guider_status_label, lambda: f"color: {COLORS['text_disabled']};")
                self.guider_rms_label.setText("--")
                # Disable guiding buttons when guider disconnected
                self.guiding_start_btn.setEnabled(False)
                self.guiding_stop_btn.setEnabled(False)
            else:
                # Check State field for guiding status
                guider_state = guider.get('State', 'Stopped')
                is_guiding = guider_state == 'Guiding'
                if is_guiding:
                    self.guider_status_label.setText("Guiding")
                    themed_style(self.guider_status_label, lambda: f"color: {COLORS['success']};")
                elif guider_state == 'Calibrating':
                    self.guider_status_label.setText("Calibrating")
                    themed_style(self.guider_status_label, lambda: f"color: {COLORS['warning']};")
                elif guider_state == 'Looping':
                    self.guider_status_label.setText("Looping")
                    themed_style(self.guider_status_label, lambda: f"color: {COLORS['text']};")
                elif guider_state == 'LostLock':
                    self.guider_status_label.setText("Lost Lock")
                    themed_style(self.guider_status_label, lambda: f"color: {COLORS['error']};")
                else:
                    self.guider_status_label.setText("Idle")
                    themed_style(self.guider_status_label, lambda: f"color: {COLORS['text']};")

                # Update guiding button states
                self.guiding_start_btn.setEnabled(self._connected and not is_guiding)
                self.guiding_stop_btn.setEnabled(self._connected and is_guiding)

                # RMS - extract from nested RMSError object
                rms_error = guider.get('RMSError', {})
                if isinstance(rms_error, dict):
                    # RMSError contains RA, Dec, Total objects with Pixel and Arcseconds values
                    rms_total = rms_error.get('Total', {})
                    if isinstance(rms_total, dict):
                        total_arcsec = finite_number(rms_total.get('Arcseconds'))
                        if total_arcsec:
                            self.guider_rms_label.setText(f'{total_arcsec:.2f}"')
                        else:
                            self.guider_rms_label.setText("--")
                    else:
                        self.guider_rms_label.setText("--")
                else:
                    self.guider_rms_label.setText("--")
        else:
            # No guider data from API - show as disconnected
            self.guider_name_label.setText("--")
            self.guider_status_label.setText("Disconnected")
            themed_style(self.guider_status_label, lambda: f"color: {COLORS['text_disabled']};")
            self.guider_rms_label.setText("--")
            # Disable guiding buttons when no guider data
            self.guiding_start_btn.setEnabled(False)
            self.guiding_stop_btn.setEnabled(False)

        # Update filter wheel info
        filterwheel = status_data.get('filterwheel', {})
        if filterwheel:
            name = filterwheel.get('Name') or filterwheel.get('DeviceName', '--')
            self.filterwheel_name_label.setText(name)

            connected = filterwheel.get('Connected', False)
            is_moving = filterwheel.get('IsMoving', False)

            if not connected:
                self.filterwheel_status_label.setText("Disconnected")
                themed_style(self.filterwheel_status_label, lambda: f"color: {COLORS['text_disabled']};")
                self.filterwheel_combo.setEnabled(False)
                self._updating_filterwheel = True
                self.filterwheel_combo.clear()
                self._updating_filterwheel = False
                self._available_filters = []
                self._last_filter_id = None
            else:
                if is_moving:
                    self.filterwheel_status_label.setText("Moving...")
                    themed_style(self.filterwheel_status_label, lambda: f"color: {COLORS['warning']};")
                    self.filterwheel_combo.setEnabled(False)
                else:
                    self.filterwheel_status_label.setText("Connected")
                    themed_style(self.filterwheel_status_label, lambda: f"color: {COLORS['success']};")
                    # Only enable if user isn't actively changing
                    if not self._user_changing_filter:
                        self.filterwheel_combo.setEnabled(True)

                # Update available filters list if changed
                available = filterwheel.get('AvailableFilters', [])
                if available != self._available_filters:
                    logger.debug(f"[FilterWheel] Available filters changed: {available}")
                    self._available_filters = available
                    self._updating_filterwheel = True
                    self.filterwheel_combo.clear()
                    for f in available:
                        filter_name = f.get('Name', f'Filter {f.get("Id", "?")}')
                        filter_id = f.get('Id', -1)
                        self.filterwheel_combo.addItem(filter_name, filter_id)
                    self._updating_filterwheel = False

                # Sync current filter selection (only if user isn't actively changing)
                if not self._user_changing_filter:
                    selected = filterwheel.get('SelectedFilter') or filterwheel.get('Filter')
                    if selected and isinstance(selected, dict):
                        current_id = selected.get('Id')
                        if current_id is not None and current_id != self._last_filter_id:
                            logger.debug(f"[FilterWheel Sync] Current filter changed: ID={current_id}, last={self._last_filter_id}")
                            self._last_filter_id = current_id
                            # Find and select the matching filter in combo
                            self._updating_filterwheel = True
                            for i in range(self.filterwheel_combo.count()):
                                if self.filterwheel_combo.itemData(i) == current_id:
                                    self.filterwheel_combo.setCurrentIndex(i)
                                    break
                            self._updating_filterwheel = False
        else:
            # No filter wheel data from API - show as disconnected
            self.filterwheel_name_label.setText("--")
            self.filterwheel_status_label.setText("Disconnected")
            themed_style(self.filterwheel_status_label, lambda: f"color: {COLORS['text_disabled']};")
            self.filterwheel_combo.setEnabled(False)
            self._updating_filterwheel = True
            self.filterwheel_combo.clear()
            self._updating_filterwheel = False
            self._available_filters = []
            self._last_filter_id = None

        # Update focuser info
        focuser = status_data.get('focuser', {})
        if focuser:
            name = focuser.get('Name') or focuser.get('DeviceName', '--')
            self.focuser_name_label.setText(name)

            focuser_connected = focuser.get('Connected', False)
            if not focuser_connected:
                self.focuser_status_label.setText("Disconnected")
                themed_style(self.focuser_status_label, lambda: f"color: {COLORS['text_disabled']};")
                self.focuser_position_label.setText("--")
                self.focuser_temp_label.setText("--")
                # Disable autofocus buttons when focuser disconnected
                self._focuser_connected = False
                self.autofocus_start_btn.setEnabled(False)
                self.autofocus_cancel_btn.setEnabled(False)
            else:
                is_moving = focuser.get('IsMoving', False)
                if is_moving:
                    self.focuser_status_label.setText("Moving")
                    themed_style(self.focuser_status_label, lambda: f"color: {COLORS['info']};")
                else:
                    self.focuser_status_label.setText("Connected")
                    themed_style(self.focuser_status_label, lambda: f"color: {COLORS['success']};")

                # Update autofocus button states. The focuser is idle while each AF frame
                # exposes, so use the event-driven run state rather than focuser movement.
                self._focuser_connected = True
                self.autofocus_start_btn.setEnabled(self._connected and not is_moving and not self._autofocus_running)
                self.autofocus_cancel_btn.setEnabled(self._connected and (is_moving or self._autofocus_running))

                # Position
                position = focuser.get('Position')
                if position is not None:
                    self.focuser_position_label.setText(str(position))
                else:
                    self.focuser_position_label.setText("--")

                # Temperature
                temp = finite_number(focuser.get('Temperature'))
                if temp is not None:
                    self.focuser_temp_label.setText(f"{temp:.1f}°C")
                else:
                    self.focuser_temp_label.setText("--")
        else:
            # No focuser data from API - show as disconnected
            self.focuser_name_label.setText("--")
            self.focuser_status_label.setText("Disconnected")
            themed_style(self.focuser_status_label, lambda: f"color: {COLORS['text_disabled']};")
            self.focuser_position_label.setText("--")
            self.focuser_temp_label.setText("--")
            # Disable autofocus buttons when no focuser data
            self._focuser_connected = False
            self.autofocus_start_btn.setEnabled(False)
            self.autofocus_cancel_btn.setEnabled(False)

        # Update image statistics (from image-history endpoint)
        # Skip if viewing a historical image (stats are managed by history click handler)
        if self._viewing_history_index is None:
            self._update_statistics_dock(status_data.get('statistics', {}))

        # Track per-sub exposure time per target for integration time calculation
        stats = status_data.get('statistics', {})
        exp = stats.get('ExposureTime')
        target_name = stats.get('TargetName')
        if isinstance(exp, (int, float)) and exp > 0 and target_name:
            self._sub_exposure_by_target[target_name] = exp

    def _update_statistics_dock(self, statistics):
        """Update the statistics dock labels from an image-history stats dict."""
        fmt = format_image_stat
        if statistics:
            self.stats_stars_label.setText(fmt(statistics.get('Stars'), allow_negative=False))
            self.stats_hfr_label.setText(fmt(statistics.get('HFR'), 2, allow_negative=False))
            self.stats_median_label.setText(fmt(statistics.get('Median')))
            self.stats_hfrstdev_label.setText(fmt(statistics.get('HFRStDev'), 2, allow_negative=False))
            self.stats_mean_label.setText(fmt(statistics.get('Mean')))
            self.stats_stdev_label.setText(fmt(statistics.get('StDev')))
            self.stats_min_label.setText(fmt(statistics.get('Min')))
            self.stats_max_label.setText(fmt(statistics.get('Max')))
        else:
            self.stats_stars_label.setText("--")
            self.stats_hfr_label.setText("--")
            self.stats_median_label.setText("--")
            self.stats_hfrstdev_label.setText("--")
            self.stats_mean_label.setText("--")
            self.stats_stdev_label.setText("--")
            self.stats_min_label.setText("--")
            self.stats_max_label.setText("--")

    def _on_image_fetching(self, bytes_received, total_bytes):
        """Show loading indicator while a new image is being fetched."""
        if bytes_received == -1:
            # Fetch complete
            self.status_label.setText("Connected - adaptive polling active")
            themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        elif total_bytes > 0:
            pct = min(int(bytes_received * 100 / total_bytes), 100)
            self.status_label.setText(f"Fetching latest image... {pct}%")
            themed_style(self.status_label, lambda: f"color: {COLORS['info']};")
        else:
            self.status_label.setText("Fetching latest image...")
            themed_style(self.status_label, lambda: f"color: {COLORS['info']};")

    def _on_image_updated(self, image_data, image_meta):
        """Handle image update from worker."""
        self._viewing_history_index = None  # Back to viewing the latest image
        if image_data:
            try:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                if not pixmap.isNull():
                    self._current_image_pixmap = pixmap
                    self.image_label.setPixmap(pixmap)
            except Exception as e:
                logger.error(f"Error loading image: {e}")

        # Update image info from image-history statistics
        if image_meta:
            statistics = image_meta.get('statistics', {})
            camera = image_meta.get('camera', {})
            filterwheel = image_meta.get('filterwheel', {})

            # Exposure time from image-history, fallback to camera
            exp = statistics.get('ExposureTime') or camera.get('LastExposureTime')
            if isinstance(exp, (int, float)) and exp > 0:
                exp_text = f"{exp:.0f}s"
            else:
                exp_text = "--"

            # Filter from image-history, fallback to filterwheel
            filter_name = statistics.get('Filter', '')
            if not filter_name:
                selected_filter = filterwheel.get('SelectedFilter') or filterwheel.get('Filter')
                if isinstance(selected_filter, dict):
                    filter_name = selected_filter.get('Name', '')

            # Stars/HFR from image-history (not measured on non-LIGHT frames)
            stars = format_image_stat(statistics.get('Stars'), allow_negative=False)
            hfr = format_image_stat(statistics.get('HFR'), 2, allow_negative=False)

            info_text = f"Exp: {exp_text}"
            if filter_name:
                info_text += f" | {filter_name}"
            if stars != "--":
                info_text += f" | Stars: {stars}"
            if hfr != "--":
                info_text += f" | HFR: {hfr}"

            self.image_info_label.setText(info_text)
        else:
            self.image_info_label.setText("")

    def _on_history_thumbnail(self, index, thumb_data, stats):
        """Handle a history thumbnail from the worker."""
        pixmap = QPixmap()
        pixmap.loadFromData(thumb_data)
        if pixmap.isNull():
            return

        item = QListWidgetItem(QIcon(pixmap), f"#{index}")
        item.setData(Qt.UserRole, index)

        # Build tooltip from image statistics
        if stats:
            lines = []
            target = stats.get('TargetName')
            if target:
                lines.append(f"Target: {target}")
            exp = finite_number(stats.get('ExposureTime'))
            if exp is not None:
                lines.append(f"Exposure: {exp:.0f}s")
            filt = stats.get('Filter')
            if filt:
                lines.append(f"Filter: {filt}")
            gain = stats.get('Gain')
            if gain is not None:
                lines.append(f"Gain: {gain}")
            for label, key, decimals, allow_negative in (
                ("Stars", 'Stars', None, False), ("HFR", 'HFR', 2, False),
                ("Median", 'Median', None, True), ("Mean", 'Mean', None, True),
            ):
                text = format_image_stat(stats.get(key), decimals, allow_negative)
                if text != "--":
                    lines.append(f"{label}: {text}")
            temp = finite_number(stats.get('Temperature'))
            if temp is not None:
                lines.append(f"Temp: {temp:.1f}°C")
            if lines:
                item.setToolTip("\n".join(lines))

        # If this index is newer than anything in the list, prepend; otherwise append
        if self.image_history_list.count() > 0:
            first_item = self.image_history_list.item(0)
            existing_max = first_item.data(Qt.UserRole) if first_item else -1
            if index > existing_max:
                self.image_history_list.insertItem(0, item)
                return
        self.image_history_list.addItem(item)

    def _on_history_item_clicked(self, item):
        """Handle click on a history thumbnail - load full size in background."""
        index = item.data(Qt.UserRole)
        if index is None:
            return

        self._viewing_history_index = index

        host, port = NINAIntegration.get_settings()
        quality = self.image_quality_spin.value()
        size_wh = self.image_size_combo.currentText()

        def fetch():
            image_data, _ = NINAIntegration.get_image(
                host, port, index, quality=quality, size_wh=size_wh
            )
            stats = NINAIntegration.get_image_statistics(host, port, index)
            return image_data, stats

        self._run_in_background(fetch, lambda result: self._image_fetch_done.emit(result) if result else None)

    def _on_image_fetch_done(self, result):
        """Handle completed background image fetch for history click."""
        image_data, stats = result if isinstance(result, tuple) else (result, None)
        if image_data:
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            if not pixmap.isNull():
                self._current_image_pixmap = pixmap
                self.image_label.setPixmap(pixmap)

        # Show image info with history indicator and update statistics dock
        index = self._viewing_history_index
        if index is not None:
            info_parts = [f"Image #{index}"]
            if stats and isinstance(stats, dict):
                target = stats.get('TargetName')
                if target:
                    info_parts.append(target)
                exp = stats.get('ExposureTime')
                if isinstance(exp, (int, float)) and exp > 0:
                    info_parts.append(f"{exp:.0f}s")
                filt = stats.get('Filter')
                if filt:
                    info_parts.append(filt)
                stars = format_image_stat(stats.get('Stars'), allow_negative=False)
                if stars != "--":
                    info_parts.append(f"Stars: {stars}")
                hfr = format_image_stat(stats.get('HFR'), 2, allow_negative=False)
                if hfr != "--":
                    info_parts.append(f"HFR: {hfr}")
                self._update_statistics_dock(stats)
            self.image_info_label.setText(" | ".join(info_parts))

    def _on_livestack_fetching(self, bytes_received, total_bytes):
        """Show loading indicator while a new livestack image is being fetched."""
        if total_bytes > 0:
            pct = min(int(bytes_received * 100 / total_bytes), 100)
            self.status_label.setText(f"Fetching live stack image... {pct}%")
            themed_style(self.status_label, lambda: f"color: {COLORS['info']};")
        elif bytes_received == 0 and total_bytes == -1:
            self.status_label.setText("Fetching live stack image...")
            themed_style(self.status_label, lambda: f"color: {COLORS['info']};")

    def _on_livestack_updated(self, image_data, status, available_stacks):
        """Handle livestack update from worker."""
        # Clear our own "Fetching live stack image..." message, but not other
        # messages (e.g. a sequence or filter change result)
        if self.status_label.text().startswith("Fetching live stack image"):
            self.status_label.setText("Connected - adaptive polling active")
            themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")

        is_running = status.get('running', False)
        if is_running:
            # Update comboboxes if available stacks changed
            self._update_livestack_combos(available_stacks, status)
            self._follow_sequence_target()

            # Update livestack tab with indicator
            self.image_tabs.setTabText(2, "Live Stack *")
            if image_data:
                # New image data - update the pixmap
                try:
                    pixmap = QPixmap()
                    pixmap.loadFromData(image_data)
                    if not pixmap.isNull():
                        self._current_livestack_pixmap = pixmap
                        self.livestack_label.setPixmap(pixmap)
                except Exception as e:
                    logger.error(f"Error loading livestack image: {e}")
            elif not self._current_livestack_pixmap:
                # Running but no image yet
                self.livestack_label.setPlaceholderText("Waiting for first image...")
                self.livestack_label.setPixmap(None)

            # Always update info label with target/filter/stack count
            target = status.get('selected_target', '')
            filter_name = status.get('selected_filter', '')
            # API returns per-channel counts (RedStackCount, etc.) not a unified StackCount
            stack_count = (status.get('StackCount')
                           or status.get('RedStackCount')
                           or status.get('GreenStackCount')
                           or status.get('BlueStackCount'))
            self._update_livestack_annotations(target, filter_name, stack_count)
            if target and filter_name:
                info_text = f"{target} - {filter_name}"
                if stack_count is not None:
                    info_text += f" | Stacks: {stack_count}"
                    # Accumulate integration time based on the delta since last update,
                    # using the current sub-exposure time for only the new frames.
                    # This handles mid-session exposure changes correctly.
                    # Use integration time calculated from full image history in the worker
                    if 'calculated_integration' in status:
                        self._total_integration_by_target[(target, filter_name)] = status['calculated_integration']
                    total_secs = self._total_integration_by_target.get((target, filter_name))
                    if total_secs:
                        if total_secs >= 3600:
                            h = int(total_secs // 3600)
                            m = int((total_secs % 3600) // 60)
                            info_text += f" | Int: {h}h {m}m"
                        elif total_secs >= 60:
                            m = int(total_secs // 60)
                            s = int(total_secs % 60)
                            info_text += f" | Int: {m}m {s}s"
                        else:
                            info_text += f" | Int: {int(total_secs)}s"
                self.livestack_info_label.setText(info_text)
            elif not self._current_livestack_pixmap:
                self.livestack_info_label.setText("Live stack active")
        else:
            # Reset tab text and show inactive message
            self.image_tabs.setTabText(2, "Live Stack")
            self.livestack_label.setPlaceholderText("Live stack not active")
            self.livestack_label.setPixmap(None)
            self._current_livestack_pixmap = None
            self.livestack_info_label.setText("")
            # Reset integration cache so the next session recalculates from scratch
            self._total_integration_by_target.clear()
            self._reset_livestack_annotations()
            # Clear comboboxes when not running
            self._updating_livestack_combos = True
            self.livestack_target_combo.clear()
            self.livestack_filter_combo.clear()
            self._livestack_available_stacks = []
            self._updating_livestack_combos = False
            # Select the sequence's target again when stacking restarts
            self._livestack_followed_target = None
            if self._sequence_container_names:
                self._livestack_follow_pending = self._sequence_container_names[::-1]

    def _follow_sequence_target(self):
        """Select the sequence's current target in the Live Stack target list.

        Runs after the sequence moves to another container; waits until that
        target has a stack. A target picked by hand stays selected until the
        sequence moves to a different target.
        """
        stacks = self._livestack_available_stacks
        if not self._livestack_follow_pending or not stacks:
            return
        targets = {s.get('Target') for s in stacks}
        # Match the deepest running container that has a stack (a target sits
        # inside 'Targets', and may hold loop containers of its own)
        target = next((n for n in self._livestack_follow_pending if n in targets), None)
        if target is None:
            return  # No stack for it yet; try again when the stacks update
        self._livestack_follow_pending = None
        if target == self._livestack_followed_target:
            return  # Same target (e.g. only an inner loop changed); keep the user's choice
        self._livestack_followed_target = target
        if self.livestack_target_combo.currentText() == target:
            return

        # Keep the selected filter if the new target has it, else prefer RGB
        filters = [s.get('Filter') for s in stacks if s.get('Target') == target and s.get('Filter')]
        filter_name = self.livestack_filter_combo.currentText()
        if filter_name not in filters:
            filter_name = 'RGB' if 'RGB' in filters else (filters[0] if filters else filter_name)

        self._updating_livestack_combos = True
        self.livestack_target_combo.setCurrentText(target)
        self.livestack_filter_combo.setCurrentText(filter_name)
        self._updating_livestack_combos = False
        logger.debug(f"Live stack following sequence target: {target} ({filter_name})")
        self._on_livestack_selection_changed()

    def _on_livestack_annotations_toggled(self, enabled):
        """Show or hide the live stack annotations, solving the stack if needed."""
        if not enabled:
            self.livestack_label.setOverlay(None)
            self.livestack_annotations_status.setText("")
            return
        if self._ls_annot_renderer is not None:
            self._apply_livestack_annotation_layers()
            self._sync_livestack_annotation_wcs()
            self.livestack_label.setOverlay(self._ls_annot_renderer)
            self._show_livestack_annotation_count()
        elif self._ls_annot_key is not None:
            self._start_livestack_solve()
        else:
            self.livestack_annotations_status.setText("Waiting for the live stack")

    def _reset_livestack_annotations(self):
        """Forget the current stack's plate solve (live stacking stopped or restarted)."""
        self._ls_annot_key = None
        self._ls_annot_count = None
        self._ls_annot_solution = None
        self._ls_annot_renderer = None
        self.livestack_label.setOverlay(None)
        if self.livestack_annotations_check.isChecked():
            self.livestack_annotations_status.setText("Waiting for the live stack")

    def _update_livestack_annotations(self, target, filter_name, stack_count):
        """Keep the annotations matched to the live stack being shown.

        Solves once per stack: again only when another target/filter is shown
        or the stack restarts (its frame count drops).
        """
        if not target or not filter_name:
            return
        key = (target, filter_name)
        restarted = (isinstance(stack_count, (int, float)) and isinstance(self._ls_annot_count, (int, float))
                     and stack_count < self._ls_annot_count)
        if key != self._ls_annot_key or restarted:
            self._reset_livestack_annotations()
            self._ls_annot_key = key
        if stack_count:
            self._ls_annot_count = stack_count

        if not self.livestack_annotations_check.isChecked():
            return
        if self._ls_annot_renderer is not None:
            self._sync_livestack_annotation_wcs()  # The stack image may have changed size
        elif not self._ls_annot_solving and self._ls_annot_count:
            self._start_livestack_solve()

    def _start_livestack_solve(self):
        """Have NINA plate-solve the first frame of the shown stack."""
        if self._ls_annot_solving or self._ls_annot_key is None:
            return
        key, stack_count = self._ls_annot_key, self._ls_annot_count or 1
        self._ls_annot_solving = True
        self.livestack_annotations_status.setText("Plate solving...")
        host, port = NINAIntegration.get_settings()

        def solve():
            # LIGHT history is oldest first; the stack's frames are the target's
            # most recent stack_count frames, aligned to the first of them
            history = NINAIntegration.get_all_image_history(host, port)
            target, filter_name = key
            frames = [i for i, image in enumerate(history)
                      if str(image.get('TargetName', '')).casefold() == target.casefold()]
            # Mono stacks are per filter; one-shot color stacks (RGB, R_OSC, ...) aren't
            same_filter = [i for i in frames if history[i].get('Filter') == filter_name]
            frames = same_filter or frames
            if not frames:
                return key, None, f"No {target} frames in NINA's image history"
            index = frames[-int(stack_count)] if len(frames) >= stack_count else frames[0]
            solution, error = NINAIntegration.solve_image(host, port, index, 'LIGHT')
            return key, solution, error

        self._run_in_background(solve, self._on_livestack_solved)

    def _on_livestack_solved(self, result):
        """Set up the annotations from NINA's plate solve and look up the objects in view."""
        key, solution, error = result
        self._ls_annot_solving = False
        if key != self._ls_annot_key:
            # The live stack moved on to another target/filter while solving
            if self.livestack_annotations_check.isChecked() and self._ls_annot_key is not None:
                self._start_livestack_solve()
            return
        if solution is None:
            self.livestack_annotations_status.setText(f"Plate solve failed: {error}")
            return

        from AnnotationOverlay import AnnotationRenderer, CatalogQueryWorker
        self._ls_annot_solution = solution
        renderer = AnnotationRenderer()
        self._ls_annot_renderer = renderer
        self._apply_livestack_annotation_layers()
        if not self._sync_livestack_annotation_wcs():
            self.livestack_annotations_status.setText("Plate solve result incomplete")
            self._ls_annot_renderer = None
            return

        self.livestack_annotations_status.setText("Looking up objects...")
        worker = CatalogQueryWorker(renderer.wcs, magnitude_limit=8.0)
        # (CatalogQueryWorker's finished(stars, dsos) replaces QThread.finished)
        worker.finished.connect(
            lambda stars, dsos: self._on_livestack_catalog_done(worker, renderer, stars, dsos))
        _running_threads.add(worker)  # Outlives the window if it's closed meanwhile
        worker.start()

    def _on_livestack_catalog_done(self, worker, renderer, stars, dsos):
        """Show the annotations once the catalog lookup finishes."""
        # finished is emitted as run() ends; let the thread exit before dropping it
        worker.wait(2000)
        _running_threads.discard(worker)
        if renderer is not self._ls_annot_renderer:
            return  # A newer solve replaced it
        renderer.set_objects(stars, dsos)
        if self.livestack_annotations_check.isChecked():
            self.livestack_label.setOverlay(renderer)
            self._show_livestack_annotation_count()

    def _show_livestack_annotation_count(self):
        renderer = self._ls_annot_renderer
        if renderer is not None:
            self.livestack_annotations_status.setText(
                f"{len(renderer.dsos)} objects, {len(renderer.stars)} stars")

    def _sync_livestack_annotation_wcs(self):
        """Fit the plate solution to the displayed stack image's size."""
        renderer, solution = self._ls_annot_renderer, self._ls_annot_solution
        size = self.livestack_label.pixmapSize()
        if renderer is None or solution is None or size.isEmpty():
            return False
        if renderer.wcs is not None and (renderer.wcs.width, renderer.wcs.height) == (size.width(), size.height()):
            return True
        header = nina_solution_to_wcs_header(solution, size.width(), size.height())
        if header is None:
            return False
        renderer.set_wcs(header, size.width(), size.height())
        self.livestack_label.update()
        return True

    def _apply_livestack_annotation_layers(self):
        """Use the layers chosen in the image viewer's Annotations dialog."""
        renderer = self._ls_annot_renderer
        if renderer is None:
            return
        settings = QSettings("CosmosCollection", "CosmosCollection")
        renderer.show_dsos = settings.value("annotation_show_dsos", True, type=bool)
        renderer.show_stars = settings.value("annotation_show_stars", True, type=bool)
        renderer.show_constellation_lines = settings.value("annotation_show_constellations", True, type=bool)
        renderer.show_grid = settings.value("annotation_show_grid", True, type=bool)

    def _update_livestack_combos(self, available_stacks, status):
        """Update the livestack target and filter comboboxes."""
        if not available_stacks:
            return

        # Check if stacks have changed
        if available_stacks == self._livestack_available_stacks:
            return

        self._livestack_available_stacks = available_stacks
        self._updating_livestack_combos = True

        # Extract unique targets and filters
        targets = sorted(set(s.get('Target', '') for s in available_stacks if s.get('Target')))
        filters = sorted(set(s.get('Filter', '') for s in available_stacks if s.get('Filter')))

        # Update target combo
        current_target = self.livestack_target_combo.currentText()
        self.livestack_target_combo.clear()
        self.livestack_target_combo.addItems(targets)

        # Restore selection or select from status
        if current_target in targets:
            self.livestack_target_combo.setCurrentText(current_target)
        elif status.get('selected_target') in targets:
            self.livestack_target_combo.setCurrentText(status.get('selected_target'))

        # Update filter combo
        current_filter = self.livestack_filter_combo.currentText()
        self.livestack_filter_combo.clear()
        self.livestack_filter_combo.addItems(filters)

        # Restore selection or select from status
        if current_filter in filters:
            self.livestack_filter_combo.setCurrentText(current_filter)
        elif status.get('selected_filter') in filters:
            self.livestack_filter_combo.setCurrentText(status.get('selected_filter'))

        self._updating_livestack_combos = False

    def _on_livestack_selection_changed(self):
        """Handle user changing the livestack target or filter selection."""
        if self._updating_livestack_combos:
            return

        target = self.livestack_target_combo.currentText() or None
        filter_name = self.livestack_filter_combo.currentText() or None

        # Update worker with new selection
        if hasattr(self, 'worker') and self.worker:
            self.worker.set_livestack_selection(target, filter_name)

    def _apply_image_settings_to_worker(self):
        """Push current quality/size settings to the worker thread (sizes still being
        typed keep the last complete one)."""
        if hasattr(self, 'worker') and self.worker:
            image_size = self.image_size_combo.currentText().strip()
            livestack_size = self.livestack_size_combo.currentText().strip()
            self.worker.set_image_quality_settings(
                self.image_quality_spin.value(),
                image_size if valid_size(image_size) else self.worker._image_size,
                self.livestack_quality_spin.value(),
                livestack_size if valid_size(livestack_size) else self.worker._livestack_size,
            )

    def _on_size_changed(self):
        """A size box changed and typing has paused: apply the sizes that are complete."""
        if valid_size(self.image_size_combo.currentText()):
            self._on_image_quality_changed()
        if valid_size(self.livestack_size_combo.currentText()):
            self._on_livestack_quality_changed()
        if valid_size(self.liveview_size_combo.currentText()):
            self._on_liveview_quality_changed()

    def _on_image_quality_changed(self):
        """Handle user changing the latest-image quality or size."""
        if self._restoring_settings:
            return
        self._apply_image_settings_to_worker()
        size_wh = self.image_size_combo.currentText().strip()
        if not valid_size(size_wh):
            return
        # Re-fetch the image being shown (a history image, or the latest) at the new quality/size
        if self._connected and hasattr(self, 'worker') and self.worker:
            host, port = self.worker.host, self.worker.port
            history_index = self._viewing_history_index
            index = history_index if history_index is not None else self.worker._last_image_index
            if index >= 0:
                quality = self.image_quality_spin.value()

                def fetch():
                    data, _ = NINAIntegration.get_image(
                        host, port, index, quality=quality, size_wh=size_wh
                    )
                    # A history image's info line is rebuilt from its statistics
                    stats = NINAIntegration.get_image_statistics(host, port, index) if history_index is not None else None
                    return data, stats, index

                self._run_in_background(fetch, self._on_quality_refetch_done)

    def _on_quality_refetch_done(self, result):
        """Show an image re-fetched at a new quality/size, if it's still the one being viewed."""
        data, stats, index = result
        viewing = self._viewing_history_index
        current = viewing if viewing is not None else (self.worker._last_image_index if self.worker else -1)
        if not data or index != current:
            return  # Another image is being shown by now
        if viewing is not None:
            self._image_fetch_done.emit((data, stats))
        else:
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                self._current_image_pixmap = pixmap
                self.image_label.setPixmap(pixmap)

    def _on_livestack_quality_changed(self):
        """Handle user changing the livestack quality or size."""
        if self._restoring_settings:
            return
        self._apply_image_settings_to_worker()
        # Force re-fetch on next poll by resetting stack count tracker
        if hasattr(self, 'worker') and self.worker:
            self.worker._last_livestack_count = None

    def _on_liveview_updated(self, image_data):
        """Handle live view frame from worker."""
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            if not pixmap.isNull():
                self._current_liveview_pixmap = pixmap
                self.liveview_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"Error loading live view image: {e}")

    def _on_liveview_toggle(self):
        """Handle live view start/stop button click."""
        if not hasattr(self, 'worker') or not self.worker:
            return
        active = not self.worker._liveview_active
        self.worker.set_liveview_active(active)
        if active:
            self.liveview_toggle_btn.setText("Stop")
            self.liveview_info_label.setText("Live view active")
            self.liveview_label.setPlaceholderText("Waiting for image...")
            # Push current settings to worker
            self.worker.set_liveview_settings(
                self.liveview_quality_spin.value(),
                self.liveview_size_combo.currentText()
            )
        else:
            self.liveview_toggle_btn.setText("Start")
            self.liveview_info_label.setText("")

    def _on_liveview_quality_changed(self):
        """Handle user changing the live view quality or size."""
        if self._restoring_settings or not valid_size(self.liveview_size_combo.currentText()):
            return
        if hasattr(self, 'worker') and self.worker and self.worker._liveview_active:
            self.worker.set_liveview_settings(
                self.liveview_quality_spin.value(),
                self.liveview_size_combo.currentText()
            )

    def _on_image_tab_changed(self, index):
        """Handle image tab switch — tell worker which tab is active."""
        if hasattr(self, 'worker') and self.worker:
            self.worker.set_active_image_tab(index)

    def _on_guiding_updated(self, guiding_data):
        """Handle guiding data update from worker."""
        if guiding_data:
            self.guiding_graph.update_data(guiding_data)

    def _on_error(self, error_message):
        """Handle error from worker."""
        self.status_label.setText(f"Error: {error_message}")
        themed_style(self.status_label, lambda: f"color: {COLORS['error']};")

    EVENT_LOG_MAX_ROWS = 500
    EVENT_COLOR_ROLE = Qt.UserRole + 1  # COLORS key of an event log row's text color

    def _on_event_occurred(self, event):
        """Handle NINA event from worker."""
        event_type = event.get('Event', '')

        self._append_event_log(event)
        self._handle_autofocus_event(event)

        # Map events to user-friendly messages
        event_messages = {  # (message, COLORS key)
            'AUTOFOCUS-FINISHED': ("AutoFocus complete", 'success'),
            'SEQUENCE-STARTING': ("Sequence started", 'text_secondary'),
            'SEQUENCE-FINISHED': ("Sequence finished", 'success'),
            'GUIDER-START': ("Guiding started", 'text_secondary'),
            'GUIDER-STOP': ("Guiding stopped", 'text_secondary'),
            'IMAGE-SAVE': ("Image saved", 'text_secondary'),
            'MOUNT-HOMED': ("Mount homed", 'text_secondary'),
        }

        if event_type in event_messages:
            message, color_key = event_messages[event_type]
            self.status_label.setText(message)
            themed_style(self.status_label, lambda: f"color: {COLORS[color_key]};")
        elif event_type == 'AUTOFOCUS-STARTING':
            self.status_label.setText("AutoFocus running...")
            themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        elif event_type == 'AUTOFOCUS-POINT-ADDED':
            count = len(self.autofocus_graph.points)
            hfr = event.get('HFR')
            hfr_text = f", HFR {hfr:.2f}" if isinstance(hfr, (int, float)) else ""
            self.status_label.setText(f"AutoFocus running... point {count} (position {event.get('Position')}{hfr_text})")
            themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        elif event_type.startswith('ERROR-'):
            self.status_label.setText(f"NINA error: {self._event_display_name(event_type)}")
            themed_style(self.status_label, lambda: f"color: {COLORS['error']};")

    def _on_events_loaded(self, events):
        """Backfill the event log on connect and pick up an autofocus run already in progress."""
        self.event_log_table.setRowCount(0)
        for event in events:
            self._append_event_log(event)

        # Find the most recent AF run; replay it if it hasn't finished and is still active.
        # A failed run never emits AUTOFOCUS-FINISHED, so require recent activity too.
        af_events = [e for e in events if str(e.get('Event', '')).startswith(('AUTOFOCUS-', 'ERROR-AF'))]
        start_idx = next((i for i in range(len(af_events) - 1, -1, -1)
                          if af_events[i].get('Event') == 'AUTOFOCUS-STARTING'), None)
        if start_idx is None:
            return
        run = af_events[start_idx:]
        if any(e.get('Event') != 'AUTOFOCUS-POINT-ADDED' for e in run[1:]):
            return  # Finished or failed
        last_time = NINAStatusWorker._parse_event_time(run[-1].get('Time'))
        if last_time and (datetime.now().astimezone() - last_time).total_seconds() < AUTOFOCUS_STALE_SECONDS:
            for event in run:
                self._handle_autofocus_event(event)

    def _handle_autofocus_event(self, event):
        """Update the autofocus graph and running state from an AF-related event."""
        event_type = event.get('Event', '')
        if event_type == 'AUTOFOCUS-STARTING':
            self._autofocus_running = True
            self.autofocus_graph.start_run()
            self.autofocus_info_label.setText("")
            self._autofocus_stale_timer.start()
        elif event_type == 'AUTOFOCUS-POINT-ADDED':
            position, hfr = event.get('Position'), event.get('HFR')
            if not self._autofocus_running:
                # Joined mid-run (or missed the start event)
                self._autofocus_running = True
                self.autofocus_graph.start_run()
            if isinstance(position, (int, float)) and isinstance(hfr, (int, float)):
                self.autofocus_graph.add_point(position, hfr)
            self._autofocus_stale_timer.start()
        elif event_type == 'AUTOFOCUS-FINISHED':
            # The graph is completed by the last-af report the worker fetches next
            self._autofocus_running = False
            self._autofocus_stale_timer.stop()
        elif event_type.startswith('ERROR-AF'):
            self._autofocus_running = False
            self._autofocus_stale_timer.stop()
            self.autofocus_graph.end_run('AutoFocus failed')
        else:
            return
        self._update_autofocus_buttons()

    def _on_autofocus_stale(self):
        """No AF events for a while and no finish event: the run failed or was cancelled."""
        if self._autofocus_running:
            self._autofocus_running = False
            self.autofocus_graph.end_run('AutoFocus ended without a result')
            self._update_autofocus_buttons()

    def _on_autofocus_report(self, report):
        """Show a completed autofocus run's fitted curve and summary."""
        if self._autofocus_running:
            return  # Stale report from before the current run
        self.autofocus_graph.set_report(report)

        parts = []
        if report.get('Filter'):
            parts.append(f"Filter: {report['Filter']}")
        temp = report.get('Temperature')
        if isinstance(temp, (int, float)) and not math.isnan(temp):
            parts.append(f"Temp: {temp:.1f}°C")
        fitting = report.get('Fitting')
        if fitting:
            fitting_names = {
                'TRENDHYPERBOLIC': 'Trend + Hyperbolic', 'TRENDPARABOLIC': 'Trend + Parabolic',
                'TRENDLINES': 'Trend lines', 'HYPERBOLIC': 'Hyperbolic', 'PARABOLIC': 'Parabolic',
            }
            parts.append(f"Fitting: {fitting_names.get(str(fitting).upper(), fitting)}")
        fitting_upper = str(fitting).upper()
        r2_key = ('Hyperbolic' if 'HYPERBOLIC' in fitting_upper
                  else 'Quadratic' if 'PARABOLIC' in fitting_upper else None)
        r2 = (report.get('RSquares') or {}).get(r2_key) if r2_key else None
        if isinstance(r2, (int, float)) and not math.isnan(r2):
            parts.append(f"R²: {r2:.3f}")
        duration = report.get('Duration')
        if isinstance(duration, str) and ':' in duration:
            h, m, s = duration.split('.')[0].split(':')[-3:]
            parts.append(f"Duration: {int(h) * 60 + int(m)}m {int(s)}s")
        time_str = report.get('Timestamp')
        finished = NINAStatusWorker._parse_event_time(time_str)
        if finished:
            parts.append(f"Finished: {format_time(finished.astimezone())}")
        self.autofocus_info_label.setText("  |  ".join(parts))

    def _update_autofocus_buttons(self):
        """Refresh AutoFocus Start/Cancel buttons from the event-driven running state."""
        if not self._connected:
            return
        self.autofocus_start_btn.setEnabled(not self._autofocus_running and self._focuser_connected)
        self.autofocus_cancel_btn.setEnabled(self._autofocus_running)

    @staticmethod
    def _event_display_name(event_type):
        """Turn 'AUTOFOCUS-POINT-ADDED' into 'Autofocus point added'."""
        return event_type.replace('-', ' ').capitalize()

    @staticmethod
    def _format_event_details(event):
        """Summarize an event's payload fields for the event log."""
        event_type = event.get('Event', '')
        if event_type == 'AUTOFOCUS-POINT-ADDED':
            hfr = event.get('HFR')
            hfr_text = f"{hfr:.2f}" if isinstance(hfr, (int, float)) else hfr
            return f"Position {event.get('Position')}, HFR {hfr_text}"
        if event_type == 'STACK-UPDATED':
            return f"{event.get('Target', '')} ({event.get('Filter', '')}) - {event.get('StackCount', '?')} stacked"

        def simplify(value):
            if isinstance(value, dict):
                # e.g. FILTERWHEEL-CHANGED {"Name": ..., "Id": ...}
                value = value.get('Name', value)
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value) or "--"
            return value

        previous, new = event.get('Previous'), event.get('New')
        if previous is not None or new is not None:
            return f"{simplify(previous)} → {simplify(new)}"
        extras = {k: v for k, v in event.items() if k not in ('Event', 'Time')}
        return ", ".join(f"{k}: {simplify(v)}" for k, v in extras.items())

    def _append_event_log(self, event):
        """Insert an event at the top of the event log table."""
        event_type = event.get('Event', '')
        event_time = NINAStatusWorker._parse_event_time(event.get('Time'))
        time_text = format_time(event_time.astimezone(), seconds=True) if event_time else ""

        if event_type.startswith('ERROR-'):
            color_key = 'error'
        elif event_type.endswith('-DISCONNECTED'):
            color_key = 'warning'
        elif event_type.endswith(('-FINISHED', '-CONNECTED')):
            color_key = 'success'
        else:
            color_key = None

        self.event_log_table.insertRow(0)
        for col, text in enumerate((time_text, event_type, self._format_event_details(event))):
            item = QTableWidgetItem(str(text))
            if color_key:
                item.setForeground(QColor(COLORS[color_key]))
                # Remembered so the row can be recolored when the theme changes
                item.setData(self.EVENT_COLOR_ROLE, color_key)
            self.event_log_table.setItem(0, col, item)

        if self.event_log_table.rowCount() > self.EVENT_LOG_MAX_ROWS:
            self.event_log_table.setRowCount(self.EVENT_LOG_MAX_ROWS)

    def _recolor_event_log(self):
        """Apply the current theme's colors to the event log's colored rows."""
        table = self.event_log_table
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                item = table.item(row, col)
                color_key = item.data(self.EVENT_COLOR_ROLE) if item else None
                if color_key:
                    item.setForeground(QColor(COLORS[color_key]))

    SEQUENCE_ITEM_ROLE = Qt.UserRole + 1  # {'key', 'kind', 'status'} of a sequence tree item

    def _on_sequence_dock_visibility(self, visible):
        """Poll the sequence only while its dock is shown (not closed or behind another tab)."""
        if self.worker:
            self.worker.set_sequence_active(visible)

    def _on_sequence_updated(self, entries, error):
        """Refresh the sequence tree and current activity from /sequence/json."""
        if entries is None:
            # Request failed; keep showing the last sequence
            self._set_sequence_activity_row('status', f"Not updating: {error}", 'warning')
            return
        if not entries:
            self._sequence_state = 'none'
            self._update_sequence_buttons()
            self.sequence_tree.clear()
            self._sequence_running_key = None
            self._set_sequence_activity_row('status', error or "No sequence loaded", 'warning')
            for key in ('container', 'now', 'loop', 'next'):
                self._set_sequence_activity_row(key, None)
            return

        # Global triggers get their own branch, like NINA's sequencer
        roots = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if 'GlobalTriggers' in entry:
                if entry['GlobalTriggers']:
                    roots.append(('item', {'Name': "Global Triggers", 'Status': '',
                                           'Triggers': entry['GlobalTriggers']}))
            else:
                roots.append(('item', entry))

        self.sequence_tree.setUpdatesEnabled(False)
        running = []  # (path key, item) of each RUNNING tree item, outermost first
        self._sync_sequence_children(self.sequence_tree.invisibleRootItem(), roots, "", running)
        self.sequence_tree.setUpdatesEnabled(True)

        # Follow the current instruction as the sequence advances
        running_key, running_item = running[-1] if running else (None, None)
        if running_key != self._sequence_running_key:
            self._sequence_running_key = running_key
            if running_item is not None:
                # Scroll vertically only; scrollToItem would also pan to the item's indent
                h_bar = self.sequence_tree.horizontalScrollBar()
                h_value = h_bar.value()
                self.sequence_tree.scrollToItem(running_item, QAbstractItemView.EnsureVisible)
                h_bar.setValue(h_value)

        activity = sequence_activity(entries)
        self._update_sequence_activity(activity)
        self._sequence_state = activity['state']
        self._update_sequence_buttons()

        # When the sequence moves to another container (e.g. the next target),
        # have the Live Stack tab follow it
        names = tuple(sequence_display_name(e, 'item') for e in activity['path'] if 'Items' in e)
        if names and names != self._sequence_container_names:
            self._livestack_follow_pending = names[::-1]  # Deepest container first
        self._sequence_container_names = names
        self._follow_sequence_target()

    def _sync_sequence_children(self, parent, children, parent_key, running):
        """Update parent's child items to match children in place, keeping expansion and scroll."""
        for index, (kind, entry) in enumerate(children):
            name = sequence_display_name(entry, kind)
            # A string, since item data turns tuples into lists
            key = f"{parent_key}{kind}:{name}"
            status = entry.get('Status') or ''

            item = parent.child(index)
            info = item.data(0, self.SEQUENCE_ITEM_ROLE) if item is not None else None
            if not info or info['key'] != key:
                # New entry, or the sequence changed shape here: replace it
                if item is not None:
                    parent.takeChild(index)
                item = QTreeWidgetItem()
                parent.insertChild(index, item)
                info = None
            previous_status = info['status'] if info else None

            details = sequence_entry_details(entry, kind)
            item.setText(0, name)
            item.setText(1, status.capitalize() if status != 'CREATED' else "")
            item.setText(2, details)
            item.setToolTip(0, name)
            item.setToolTip(2, details)
            item.setData(0, self.SEQUENCE_ITEM_ROLE, {'key': key, 'kind': kind, 'status': status})
            self._style_sequence_item(item, kind, status)
            if status == 'RUNNING':
                running.append((key, item))

            grandchildren = sequence_children(entry) if kind == 'item' else []
            self._sync_sequence_children(item, grandchildren, key, running)
            if grandchildren:
                # Open containers as they start and close them as they finish;
                # otherwise leave the user's expand/collapse choices alone
                if status == 'RUNNING' and previous_status != 'RUNNING':
                    item.setExpanded(True)
                elif previous_status == 'RUNNING' and status != 'RUNNING':
                    item.setExpanded(False)
                elif previous_status is None and not parent_key and status != 'FINISHED':
                    item.setExpanded(True)  # Top-level containers start open

        while parent.childCount() > len(children):
            parent.takeChild(parent.childCount() - 1)

    def _update_sequence_buttons(self):
        """Enable Start when a loaded sequence isn't running, Stop while it runs."""
        ready = self._connected and not self._sequence_command_busy
        self.sequence_start_btn.setEnabled(ready and self._sequence_state in ('idle', 'finished'))
        self.sequence_stop_btn.setEnabled(ready and self._sequence_state == 'running')

    def _on_sequence_start(self):
        """Check the sequence for issues, then start it."""
        self._sequence_command_busy = True
        self._update_sequence_buttons()
        self.status_label.setText("Checking sequence...")
        themed_style(self.status_label, lambda: f"color: {COLORS['info']};")
        host, port = NINAIntegration.get_settings()
        self._run_in_background(lambda: NINAIntegration.get_sequence_issues(host, port),
                                self._on_sequence_issues_checked)

    def _on_sequence_issues_checked(self, issues):
        """Confirm with the user if the sequence has issues, then send the start."""
        if issues:
            # The same issue repeats across many items (e.g. "Camera not connected"),
            # so list each issue once with the items it affects
            items_by_issue = {}
            for name, issue in issues:
                items = items_by_issue.setdefault(issue, [])
                display_name = sequence_display_name({'Name': name}, 'item')
                if display_name not in items:
                    items.append(display_name)
            shown = []
            for issue, items in list(items_by_issue.items())[:10]:
                more = f", +{len(items) - 4} more" if len(items) > 4 else ""
                shown.append(f"• {issue} ({', '.join(items[:4])}{more})")
            if len(items_by_issue) > len(shown):
                shown.append(f"...and {len(items_by_issue) - len(shown)} more issues")
            answer = QMessageBox.warning(
                self, "Sequence Issues",
                "NINA reports issues with this sequence:\n\n" + "\n".join(shown) +
                "\n\nStart the sequence anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self._sequence_command_busy = False
                self._update_sequence_buttons()
                self.status_label.setText("Sequence not started")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                return
        # Validated here, so skip NINA's check: it would ask about issues in a dialog
        # on the NINA computer. If the issues couldn't be read, let NINA validate.
        skip_validation = issues is not None
        host, port = NINAIntegration.get_settings()
        self._run_in_background(
            lambda: ("start", *NINAIntegration.start_sequence(host, port, skip_validation)),
            self._on_sequence_command_done)

    def _on_sequence_stop(self):
        """Stop the running sequence after confirming."""
        answer = QMessageBox.question(
            self, "Stop Sequence",
            "Stop the running sequence?\n\nAn exposure in progress will be aborted.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self._sequence_command_busy = True
        self._update_sequence_buttons()
        host, port = NINAIntegration.get_settings()
        self._run_in_background(lambda: ("stop", *NINAIntegration.stop_sequence(host, port)),
                                self._on_sequence_command_done)

    def _on_sequence_command_done(self, result):
        """Report the result of a start/stop request and refresh the sequence."""
        command, success, message = result
        self._sequence_command_busy = False
        if success:
            self.status_label.setText(message or f"Sequence {command} sent")
            themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        else:
            self.status_label.setText(f"Couldn't {command} the sequence: {message}")
            themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
        if self.worker:
            self.worker.refresh_sequence()
        self._update_sequence_buttons()

    def _style_sequence_item(self, item, kind, status):
        """Color a sequence row by status; conditions and triggers are muted and italic."""
        status_key = SEQUENCE_STATUS_COLORS.get(status)
        name_key = 'info' if status == 'RUNNING' else ('text_secondary' if kind != 'item' else status_key)
        for col, color_key in ((0, name_key), (1, status_key), (2, 'text_secondary')):
            item.setForeground(col, QColor(COLORS[color_key or 'text']))
            font = item.font(col)
            font.setBold(status == 'RUNNING')
            font.setItalic(kind != 'item')
            item.setFont(col, font)

    def _recolor_sequence_tree(self):
        """Apply the current theme's colors to the sequence tree."""
        def recolor(parent):
            for index in range(parent.childCount()):
                item = parent.child(index)
                info = item.data(0, self.SEQUENCE_ITEM_ROLE)
                if info:
                    self._style_sequence_item(item, info['kind'], info['status'])
                recolor(item)
        recolor(self.sequence_tree.invisibleRootItem())

    def _set_sequence_activity_row(self, key, text, color_key=None):
        """Show a Current Activity row with text (hidden when text is None)."""
        title_label, value_label = self._sequence_activity_rows[key]
        title_label.setVisible(text is not None)
        value_label.setVisible(text is not None)
        if text is not None:
            value_label.setText(text)
            # Restyle only on change; this runs on every sequence poll
            if value_label.property("sequence_color") != (color_key or ""):
                value_label.setProperty("sequence_color", color_key or "")
                themed_style(value_label, lambda: f"color: {COLORS[color_key]};" if color_key else "")

    def _update_sequence_activity(self, activity):
        """Fill the Current Activity rows from sequence_activity()."""
        def describe(entry, kind):
            details = sequence_entry_details(entry, kind)
            name = sequence_display_name(entry, kind)
            return f"{name} — {details}" if details else name

        state = activity['state']
        path = activity['path']
        if state == 'running':
            self._set_sequence_activity_row('status', "Running", 'info')
        elif state == 'finished':
            self._set_sequence_activity_row('status', "Finished", 'success')
        else:
            self._set_sequence_activity_row('status', "Not running")

        # Containers the current instruction sits in, e.g. "Targets › Bubble Nebula"
        containers = [sequence_display_name(e, 'item') for e in path if 'Items' in e]
        self._set_sequence_activity_row('container', " › ".join(containers) if containers else None)

        if activity['trigger'] is not None:
            now = f"{describe(activity['trigger'], 'trigger')} (trigger)"
        elif path and 'Items' not in path[-1]:
            now = describe(path[-1], 'item')
        else:
            now = None
        self._set_sequence_activity_row('now', now)

        loop = activity['loop']
        self._set_sequence_activity_row('loop', describe(loop, 'condition') if loop else None)

        next_entry = activity['next']
        self._set_sequence_activity_row(
            'next', describe(next_entry, 'item') if next_entry is not None and state == 'running' else None)

    def _on_cooling_changed(self, state):
        """Handle cooling checkbox change."""
        if self._updating_camera_controls:
            logger.debug(f"[Cooling] Ignoring change - _updating_camera_controls is True")
            return

        enabled = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else state == 2
        temp = self.camera_target_temp_spinbox.value() if enabled else None

        logger.debug(f"[Cooling] User changed: enabled={enabled}, temp={temp}, last_enabled={self._last_cooling_enabled}, last_temp={self._last_cooling_temp}")

        # Skip if this is the same state we already sent
        if enabled == self._last_cooling_enabled and (not enabled or temp == self._last_cooling_temp):
            logger.debug(f"[Cooling] Skipping - same state already sent")
            return

        # Set flag and start timer to clear it after 15 seconds
        self._user_changing_cooling = True
        self._cooling_change_timer.start(15000)
        logger.debug(f"[Cooling] Set _user_changing_cooling=True, started 15s timer")

        if enabled:
            self.status_label.setText(f"Setting cooling to {temp}°C...")
        else:
            self.status_label.setText("Disabling cooling...")
        self._cooling_request_pending = True
        self.camera_cooling_checkbox.setEnabled(False)  # One request at a time

        host, port = NINAIntegration.get_settings()

        def done(success):
            self._cooling_request_pending = False
            self.camera_cooling_checkbox.setEnabled(True)
            if success:
                self._last_cooling_enabled = enabled
                self._last_cooling_temp = temp
                logger.debug(f"[Cooling] Success - updated last_enabled={enabled}, last_temp={temp}")
                self.status_label.setText("Cooling " + ("enabled" if enabled else "disabled"))
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
            else:
                logger.debug(f"[Cooling] Failed - reverting checkbox")
                self.status_label.setText("Failed to change cooling - check console")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                # Revert checkbox state
                self.camera_cooling_checkbox.blockSignals(True)
                self.camera_cooling_checkbox.setChecked(not enabled)
                self.camera_cooling_checkbox.blockSignals(False)
                # Clear the flag since we reverted
                self._user_changing_cooling = False
                self._cooling_change_timer.stop()

        self._run_in_background(lambda: NINAIntegration.set_camera_cooling(host, port, enabled, temp), done)

    def _clear_cooling_change_flag(self):
        """Clear the cooling change flag after timeout."""
        logger.debug(f"[Cooling] Timer expired - clearing _user_changing_cooling flag")
        self._user_changing_cooling = False

    def _apply_target_temp(self):
        """Send the target temperature once the spin box has settled."""
        if self._updating_camera_controls:
            logger.debug(f"[Cooling] Target temp change ignored - _updating_camera_controls is True")
            return

        temp = self.camera_target_temp_spinbox.value()

        # Skip if cooling is off - just remember the temp for when it's turned on
        if not self.camera_cooling_checkbox.isChecked():
            logger.debug(f"[Cooling] Target temp change ignored - cooling is off")
            return

        # Skip if this is the same temp we already sent
        if temp == self._last_cooling_temp:
            logger.debug(f"[Cooling] Target temp change ignored - same as last ({temp})")
            return

        logger.debug(f"[Cooling] User changed target temp: {temp}°C (last was {self._last_cooling_temp})")

        # Set flag to prevent sync from overriding
        self._user_changing_cooling = True
        self._cooling_change_timer.start(15000)

        self.status_label.setText(f"Setting target temp to {temp}°C...")

        host, port = NINAIntegration.get_settings()

        def done(success):
            if success:
                self._last_cooling_temp = temp
                logger.debug(f"[Cooling] Target temp success - last_temp={temp}")
                self.status_label.setText(f"Target temp set to {temp}°C")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
            else:
                logger.debug(f"[Cooling] Target temp failed")
                self.status_label.setText("Failed to set target temperature")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")

        self._run_in_background(lambda: NINAIntegration.set_camera_cooling(host, port, True, temp), done)

    def _on_dewheater_changed(self, state):
        """Handle dew heater checkbox change."""
        if self._updating_camera_controls:
            logger.debug(f"[DewHeater] Ignoring change - _updating_camera_controls is True")
            return

        # Set flag and start timer to clear it after 15 seconds
        self._user_changing_dewheater = True
        self._dewheater_change_timer.start(15000)

        enabled = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else state == 2
        logger.debug(f"[DewHeater] User changed: enabled={enabled}")
        self.status_label.setText("Turning dew heater " + ("on" if enabled else "off") + "...")
        self._dewheater_request_pending = True
        self.camera_dewheater_checkbox.setEnabled(False)  # One request at a time

        host, port = NINAIntegration.get_settings()

        def done(success):
            self._dewheater_request_pending = False
            self.camera_dewheater_checkbox.setEnabled(True)
            if success:
                logger.debug(f"[DewHeater] Success")
                self.status_label.setText("Dew heater " + ("enabled" if enabled else "disabled"))
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
            else:
                logger.debug(f"[DewHeater] Failed - reverting checkbox")
                self.status_label.setText("Failed to change dew heater")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                # Revert checkbox state
                self._updating_camera_controls = True
                self.camera_dewheater_checkbox.setChecked(not enabled)
                self._updating_camera_controls = False
                # Clear the flag since we reverted
                self._user_changing_dewheater = False
                self._dewheater_change_timer.stop()

        self._run_in_background(lambda: NINAIntegration.set_camera_dew_heater(host, port, enabled), done)

    def _clear_dewheater_change_flag(self):
        """Clear the dew heater change flag after timeout."""
        logger.debug(f"[DewHeater] Timer expired - clearing _user_changing_dewheater flag")
        self._user_changing_dewheater = False

    def _on_imaging_start(self):
        """Start a camera capture."""
        dialog = CaptureSettingsDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        settings = dialog.get_settings()
        host, port = NINAIntegration.get_settings()
        self.imaging_start_btn.setEnabled(False)

        def done(success):
            if success:
                self.status_label.setText(f"Capture started ({settings['duration']}s)")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                self.imaging_stop_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to start capture")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self.imaging_start_btn.setEnabled(True)

        self._run_in_background(lambda: NINAIntegration.capture_image(
            host, port, duration=settings['duration'], gain=settings['gain'],
            save=settings['save'], image_type=settings['image_type']), done)

    def _on_imaging_stop(self):
        """Abort the current exposure."""
        host, port = NINAIntegration.get_settings()
        self.imaging_stop_btn.setEnabled(False)

        def done(success):
            if success:
                self.status_label.setText("Exposure aborted")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                self.imaging_start_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to abort exposure")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self.imaging_stop_btn.setEnabled(True)

        self._run_in_background(lambda: NINAIntegration.abort_exposure(host, port), done)

    def _on_autofocus_start(self):
        """Start an autofocus run."""
        host, port = NINAIntegration.get_settings()
        self.autofocus_start_btn.setEnabled(False)

        def done(success):
            if success:
                self.status_label.setText("AutoFocus started")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                # Mark running now so the next status poll doesn't flip the buttons back
                # before AUTOFOCUS-STARTING arrives
                self._autofocus_running = True
                self._autofocus_stale_timer.start()
                self.autofocus_cancel_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to start AutoFocus")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self._update_autofocus_buttons()

        self._run_in_background(lambda: NINAIntegration.start_autofocus(host, port), done)

    def _on_autofocus_cancel(self):
        """Cancel the running autofocus."""
        host, port = NINAIntegration.get_settings()
        self.autofocus_cancel_btn.setEnabled(False)

        def done(success):
            if success:
                self.status_label.setText("AutoFocus cancelled")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                if self._autofocus_running:
                    self._autofocus_running = False
                    self._autofocus_stale_timer.stop()
                    self.autofocus_graph.end_run('AutoFocus cancelled')
                self.autofocus_start_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to cancel AutoFocus")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self._update_autofocus_buttons()

        self._run_in_background(lambda: NINAIntegration.cancel_autofocus(host, port), done)

    def _on_guiding_start(self):
        """Start guiding."""
        host, port = NINAIntegration.get_settings()
        self.guiding_start_btn.setEnabled(False)

        def done(success):
            if success:
                self.status_label.setText("Guiding started")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                self.guiding_stop_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to start guiding")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self.guiding_start_btn.setEnabled(True)

        self._run_in_background(lambda: NINAIntegration.start_guiding(host, port), done)

    def _on_guiding_stop(self):
        """Stop guiding."""
        host, port = NINAIntegration.get_settings()
        self.guiding_stop_btn.setEnabled(False)

        def done(success):
            if success:
                self.status_label.setText("Guiding stopped")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
                self.guiding_start_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to stop guiding")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self.guiding_stop_btn.setEnabled(True)

        self._run_in_background(lambda: NINAIntegration.stop_guiding(host, port), done)

    def _on_mount_home(self):
        """Home the mount."""
        host, port = NINAIntegration.get_settings()
        self.mount_home_btn.setEnabled(False)
        self.status_label.setText("Homing mount...")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")

        def do_home():
            return NINAIntegration.home_mount(host, port)

        def on_home_complete(success):
            if success:
                self.status_label.setText("Mount homing...")
            else:
                self.status_label.setText("Failed to home mount")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
            # Button state will be updated by status polling

        self._run_in_background(do_home, on_home_complete)

    def _on_mount_park(self):
        """Park the mount."""
        host, port = NINAIntegration.get_settings()
        self.mount_park_btn.setEnabled(False)
        self.status_label.setText("Parking mount...")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")

        def do_park():
            return NINAIntegration.park_mount(host, port)

        def on_park_complete(success):
            if success:
                self.status_label.setText("Mount parking...")
                self.mount_unpark_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to park mount")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self.mount_park_btn.setEnabled(True)

        self._run_in_background(do_park, on_park_complete)

    def _on_mount_unpark(self):
        """Unpark the mount."""
        host, port = NINAIntegration.get_settings()
        self.mount_unpark_btn.setEnabled(False)
        self.status_label.setText("Unparking mount...")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")

        def do_unpark():
            return NINAIntegration.unpark_mount(host, port)

        def on_unpark_complete(success):
            if success:
                self.status_label.setText("Mount unparking...")
                self.mount_park_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to unpark mount")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                self.mount_unpark_btn.setEnabled(True)

        self._run_in_background(do_unpark, on_unpark_complete)

    def _on_mount_slew(self):
        """Slew the mount to coordinates."""
        dialog = SlewDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        ra_deg, dec_deg = dialog.get_coordinates_degrees()
        host, port = NINAIntegration.get_settings()

        self.status_label.setText(f"Slewing to RA={ra_deg:.4f}° Dec={dec_deg:.4f}°...")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        self.mount_slew_btn.setEnabled(False)

        # Run slew in background thread to avoid blocking UI
        def do_slew():
            return NINAIntegration.slew_mount(host, port, ra_deg, dec_deg)

        def on_slew_complete(success):
            if success:
                self.status_label.setText("Slew started")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
            else:
                self.status_label.setText("Failed to start slew")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
            # Button will be re-enabled by status updates when slew completes

        self._run_in_background(do_slew, on_slew_complete)

    def _on_filter_changed(self, index):
        """Handle filter selection change."""
        if self._updating_filterwheel:
            logger.debug(f"[FilterWheel] Ignoring change - _updating_filterwheel is True")
            return

        if index < 0:
            return

        filter_id = self.filterwheel_combo.itemData(index)
        filter_name = self.filterwheel_combo.itemText(index)

        # Skip if this is the same filter we already have
        if filter_id == self._last_filter_id:
            logger.debug(f"[FilterWheel] Skipping - same filter already selected (ID={filter_id})")
            return

        logger.debug(f"[FilterWheel] User selected: {filter_name} (ID={filter_id}), last={self._last_filter_id}")

        # Set flag to prevent sync from overriding during change (the wheel can take
        # a while to move, so the request runs in the background)
        self._user_changing_filter = True
        self.filterwheel_combo.setEnabled(False)
        self.status_label.setText(f"Changing to {filter_name}...")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")

        host, port = NINAIntegration.get_settings()

        def done(success):
            if success:
                self._last_filter_id = filter_id
                logger.debug(f"[FilterWheel] Success - filter changed to {filter_name}")
                self.status_label.setText(f"Filter changed to {filter_name}")
                themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
            else:
                logger.debug(f"[FilterWheel] Failed - reverting selection")
                self.status_label.setText("Failed to change filter")
                themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
                # Revert combo selection to last known good filter
                if self._last_filter_id is not None:
                    self._updating_filterwheel = True
                    for i in range(self.filterwheel_combo.count()):
                        if self.filterwheel_combo.itemData(i) == self._last_filter_id:
                            self.filterwheel_combo.setCurrentIndex(i)
                            break
                    self._updating_filterwheel = False

            # Clear the flag and re-enable combo
            self._user_changing_filter = False
            self.filterwheel_combo.setEnabled(True)

        self._run_in_background(lambda: NINAIntegration.change_filter(host, port, filter_id), done)

    def _run_in_background(self, func, callback):
        """Run a function in a background thread and call callback with result on completion."""
        class BackgroundWorker(QThread):
            finished_with_result = Signal(object)

            def __init__(self, func):
                super().__init__()
                self._func = func

            def run(self):
                result = self._func()
                self.finished_with_result.emit(result)

        worker = BackgroundWorker(func)
        worker.finished_with_result.connect(callback)
        worker.finished.connect(worker.deleteLater)
        # Referenced at module level so it survives the window being closed and replaced
        keep_alive_until_finished(worker)
        worker.start()

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        # ZoomableImageWidget handles its own resizing

    def closeEvent(self, event):
        """Clean up when window is closed."""
        self._save_settings()
        self._stop_worker()
        super().closeEvent(event)


def main():
    """Main entry point for standalone testing."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = NINADashboardWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
