"""
Collage Builder - arrange your images on a canvas and export them as one picture.

A collage is a canvas of a set size (e.g. 3840 x 2160) holding image frames you
drag in from the library, then move and resize. Changes save automatically and
can be undone. Export draws the same layout from the full-resolution files, so
the result matches the canvas exactly.

Previews (up to PREVIEW_MAX px) are decoded once in the background and cached on
disk, so opening a collage doesn't decode every full-size image again.
"""

import os
import re
import sys
import copy
import json
import math
import hashlib
import logging
import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime

from PySide6.QtCore import (Qt, QObject, QEvent, QThread, QRunnable, QThreadPool, Signal, QTimer,
                            QRectF, QPointF, QLineF, QSize, QSizeF, QMimeData, QByteArray,
                            QSettings, QUrl)
from PySide6.QtGui import (QColor, QPainter, QPainterPath, QPixmap, QImage, QFont, QFontMetricsF,
                           QPen, QIcon, QUndoCommand, QUndoStack, QUndoGroup, QKeySequence,
                           QAction, QDesktopServices, QBrush, QLinearGradient, QRadialGradient, QTransform,
                           QCursor)
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSplitter,
                               QListWidget, QListWidgetItem, QListView, QAbstractItemView,
                               QStyledItemDelegate, QGraphicsView, QGraphicsScene, QGraphicsItem,
                               QPushButton, QToolButton, QLabel, QLineEdit, QSpinBox, QCheckBox,
                               QComboBox, QGroupBox, QMenu, QMessageBox, QInputDialog, QFileDialog,
                               QColorDialog, QProgressDialog, QStackedWidget, QApplication, QFrame,
                               QSlider, QDoubleSpinBox, QPlainTextEdit, QFontComboBox, QRadioButton,
                               QButtonGroup, QScrollArea)

from WindowPositionManager import WindowPositionMixin
from Theme import COLORS, contrast_text, font_px, themed_style, theme_manager
from DatabaseManager import DatabaseManager

logger = logging.getLogger(__name__)

LAYOUT_VERSION = 1                 # usercollages.layout_version of canvas collages
DEFAULT_CANVAS = (3840, 2160)      # 4K, 16:9
DEFAULT_BACKGROUND = "#000000"
MAX_CANVAS = 16000                 # px per side - a 16000 x 16000 export needs ~1 GB
MIN_FRAME = 16                     # smallest frame side, canvas px
NEW_FRAME_FRACTION = 0.3           # a new frame's long side, as a fraction of the canvas's short side
FIT_FILL, FIT_WHOLE = "fill", "fit"
LABEL_POSITIONS = ("Bottom Center", "Top Center", "Bottom Left", "Bottom Right",
                   "Top Left", "Top Right", "Center Overlay")
LABEL_SCALE = 0.045                # label text height, as a fraction of the frame's short side
PREVIEW_MAX = 1024                 # long side of the cached preview images
THUMB_SIZE = QSize(120, 90)        # library thumbnails
SNAP_PX = 7                        # snapping distance, screen px
HANDLE_PX = 9                      # resize handle size, screen px
ROTATE_PX = 26                     # rotate knob distance above a frame, screen px
ROTATE_SNAP = 5                    # degrees within which rotation snaps to 0 / 90 / 180 / 270
GUIDE_COLOR = "#ff3d9a"
LIBRARY_MIME = "application/x-cosmoscollection-userimages"

# An image's DSO by its primary designation ("M 31"), preferring M, NGC, IC
DSO_NAME_SQL = """(SELECT c.catalogue || ' ' || c.designation
                   FROM cataloguenr c
                   WHERE c.dsodetailid = ui.dsodetailid
                   ORDER BY CASE c.catalogue WHEN 'M' THEN 1 WHEN 'NGC' THEN 2 WHEN 'IC' THEN 3 ELSE 4 END,
                            c.designation
                   LIMIT 1)"""

# QThreads still running after their owner is gone - dropping the last
# reference to a running QThread aborts the app
_running_threads = set()


def keep_alive_until_finished(thread):
    _running_threads.add(thread)
    thread.finished.connect(lambda: _running_threads.discard(thread))


def natural_key(text):
    """Sort key putting "M 2" before "M 13"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text or "")]


# --------------------------------------------------------------------
# Model
# --------------------------------------------------------------------

_frame_ids = itertools.count(1)


@dataclass
class Frame:
    """One element on the canvas: an image, a catalog placeholder or a text
    box. x, y, w, h are canvas pixels, before rotation (which turns it about
    its center).

    An image frame with no image (image_id None) is a catalog placeholder: an
    object in a catalog collage that hasn't been imaged yet, shown as its slot
    name ("M 12"). Dropping an image on it fills it.
    """
    kind: str = "image"         # "image" (or placeholder) / "text"
    image_id: int = None
    image_path: str = ""
    dso_name: str = ""          # Automatic label: the image's DSO(s), kept up to date by the window
    x: float = 0.0
    y: float = 0.0
    w: float = 400.0
    h: float = 400.0
    fit: str = FIT_FILL         # fill: the image covers the frame (cropped); fit: the whole image shows
    crop_x: float = 0.5         # center of the visible part when cropped, as a fraction of the image
    crop_y: float = 0.5
    zoom: float = 1.0           # >1 zooms into the image when cropped
    rotation: float = 0.0
    label: str = None           # Custom label text; None: the automatic one
    label_visible: bool = None  # None: the collage's setting
    label_position: str = None  # None: the collage's setting
    label_content: str = None   # "number" / "name" (number and name); None: the collage's setting
    common_name: str = ""       # Automatic: the DSO(s)' common names ("Bode's Galaxy & Cigar Galaxy")
    slot: str = None            # Catalog slot this frame fills ("M 12"), if any
    slot_name: str = None       # The slot object's common name ("Gumball Globular")
    slot_dso: str = None        # The slot object's dsodetailid, to fill it from the library
    # Frames merged into this one (objects that fit in its image, e.g. M 32 in
    # M 31's): [{"names": [...], "dsos": [...], "frame": the merged frame}]
    merged: list = None
    premerge_rect: list = None  # [x, y, w, h] before the first merge, for Unmerge
    # Style (canvas px)
    border_width: float = 0.0
    border_color: str = "#ffffff"
    corner_radius: float = 0.0
    shadow: float = 0.0         # 0 - 1
    # Text boxes
    text: str = ""
    font_family: str = ""       # "": the app font
    font_size: float = 96.0     # canvas px
    text_color: str = "#ffffff"
    bold: bool = True
    italic: bool = False
    align: str = "center"       # left / center / right
    text_box: str = None        # Backing box: None, "dark" or "light"
    uid: int = field(default_factory=lambda: next(_frame_ids))  # identifies the frame on the canvas

    LAYOUT_KEYS = ("kind", "x", "y", "w", "h", "fit", "crop_x", "crop_y", "zoom", "rotation", "label",
                   "label_visible", "label_position", "label_content", "slot", "slot_name", "slot_dso", "merged",
                   "premerge_rect",
                   "border_width", "border_color", "corner_radius", "shadow",
                   "text", "font_family", "font_size", "text_color", "bold", "italic", "align", "text_box")

    @property
    def is_text(self):
        return self.kind == "text"

    @property
    def is_image(self):
        """Has an image (not a placeholder or a text box)."""
        return self.kind == "image" and self.image_id is not None

    def merged_names(self):
        return [name for entry in self.merged or [] for name in entry.get("names", [])]

    def merged_dsos(self):
        return [dso for entry in self.merged or [] for dso in entry.get("dsos", []) if dso]

    @property
    def is_placeholder(self):
        """An empty catalog slot (an image frame with no image)."""
        return self.kind == "image" and self.image_id is None

    def rect(self):
        return QRectF(self.x, self.y, self.w, self.h)

    def set_rect(self, rect):
        self.x, self.y, self.w, self.h = rect.x(), rect.y(), rect.width(), rect.height()

    def label_text(self):
        return self.dso_name if self.label is None else self.label

    def layout(self):
        return {key: getattr(self, key) for key in self.LAYOUT_KEYS}

    def clear_image(self):
        """Back to an empty catalog slot."""
        self.image_id, self.image_path, self.dso_name = None, "", self.slot or ""
        self.fit, self.crop_x, self.crop_y, self.zoom, self.label = FIT_FILL, 0.5, 0.5, 1.0, None


LABEL_CONTENTS = (("number", "Number (“M 81”)"), ("name", "Number and name"))
PLACEHOLDER_STYLES = (("number", "Number (“M 12”)"),
                      ("name", "Number and name"),
                      ("hidden", "Hidden (keep the gap)"))


@dataclass(eq=False)
class Collage:
    name: str
    width: int = DEFAULT_CANVAS[0]
    height: int = DEFAULT_CANVAS[1]
    background: str = DEFAULT_BACKGROUND
    show_labels: bool = True
    label_position: str = "Bottom Center"
    label_content: str = "number"  # "number" ("M 81") / "name" (number, and the common name below)
    placeholder_style: str = "number"
    rule: dict = None           # The suggestion it was made from, e.g. {"key": "catalog:M", "seen": [ids]}
    background2: str = None     # Second gradient color; None: a solid background
    gradient: str = "vertical"  # vertical / horizontal / diagonal / radial
    frames: list = field(default_factory=list)  # paint order, bottom first
    db_id: int = None
    dsodetailid: object = None

    SETTINGS = ("width", "height", "background", "show_labels", "label_position", "label_content",
                "placeholder_style", "rule", "background2", "gradient")

    def canvas_rect(self):
        return QRectF(0, 0, self.width, self.height)

    def render_settings(self):
        """What painting needs besides the frames (also in state()["settings"]).
        A copy - snapshots mustn't share the rule dict."""
        return copy.deepcopy({key: getattr(self, key) for key in self.SETTINGS})

    def state(self):
        """Everything undo restores - settings and frames - as plain data."""
        return {"settings": self.render_settings(), "frames": [asdict(frame) for frame in self.frames]}

    def restore(self, state):
        for key, value in state["settings"].items():
            setattr(self, key, value)
        self.frames = [Frame(**data) for data in state["frames"]]

    def image_ids(self):
        return {frame.image_id for frame in self.frames if frame.image_id is not None}

    def has_placeholders(self):
        return any(frame.slot for frame in self.frames)


def _json_dict(text):
    try:
        value = json.loads(text) if text else {}
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _color_name(value):
    color = QColor(value or DEFAULT_BACKGROUND)
    return color.name() if color.isValid() else DEFAULT_BACKGROUND


def load_library(conn):
    """Every user image, as dicts, with its DSO name."""
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT ui.id, ui.image_path, ui.dsodetailid, ui.date_taken, ui.equipment,
               ui.integration_time, ui.is_favorite, ui.created_date, {DSO_NAME_SQL},
               d.dsotype, d.constellation
        FROM userimages ui
        LEFT JOIN dsodetail d ON d.id = ui.dsodetailid
    """)
    keys = ("id", "image_path", "dsodetailid", "date_taken", "equipment", "integration_time",
            "is_favorite", "created_date", "dso_name", "dsotype", "constellation")
    rows = []
    for row in cursor.fetchall():
        image = dict(zip(keys, tuple(row)))
        image["image_path"] = image["image_path"] or ""
        image["dso_name"] = image["dso_name"] or "Unknown DSO"
        rows.append(image)
    common = load_common_names(conn, [row["dsodetailid"] for row in rows])
    for row in rows:
        row["common_name"] = common.get(str(row["dsodetailid"]))
    return rows


def load_common_names(conn, dsodetailids):
    """{dsodetailid: readable common name} ("Bode's Galaxy") from the NAME catalog."""
    ids = sorted({str(i) for i in dsodetailids if i})
    aliases = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for dso, name in conn.execute(f"SELECT dsodetailid, designation FROM cataloguenr WHERE catalogue = 'NAME' "
                                      f"AND dsodetailid IN ({','.join('?' * len(chunk))})", chunk).fetchall():
            aliases.setdefault(str(dso), []).append(str(name))
    return {dso: name for dso, names in aliases.items() if (name := pick_common_name(names))}


def convert_grid_layout(collage, grid_width, grid_height, cell_size, spacing, images):
    """Frames for a collage saved by the old grid editor: the same cells, each
    showing the whole image, on a canvas the size of the old export."""
    columns = max(1, int(grid_width or 3))
    rows = max(1, int(grid_height or 3), -(-len(images) // columns))  # Room for every image
    cell = max(MIN_FRAME, int(cell_size or 400))
    spacing = max(0, int(spacing or 0))
    collage.width = cell * columns + spacing * (columns + 1)
    collage.height = cell * rows + spacing * (rows + 1)
    collage.frames = []
    for index, (image_id, path, _layout, dso_name) in enumerate(images):
        column, row = index % columns, index // columns
        collage.frames.append(Frame(
            image_id=image_id, image_path=path or "", dso_name=dso_name or "Unknown DSO",
            x=spacing + column * (cell + spacing), y=spacing + row * (cell + spacing),
            w=cell, h=cell, fit=FIT_WHOLE))


def load_collages(conn, dsodetailid=None):
    """Saved collages, most recently changed first. dsodetailid None: all of them."""
    cursor = conn.cursor()
    where, params = ("WHERE dsodetailid = ?", (dsodetailid,)) if dsodetailid is not None else ("", ())
    cursor.execute(f"""
        SELECT id, name, dsodetailid, grid_width, grid_height, cell_size, spacing,
               background_color, canvas_width, canvas_height, settings, layout_version
        FROM usercollages {where}
        ORDER BY modified_date DESC
    """, params)
    collages = []
    for row in cursor.fetchall():
        (collage_id, name, collage_dso, grid_width, grid_height, cell_size, spacing,
         background, canvas_width, canvas_height, settings_json, layout_version) = tuple(row)
        cursor.execute(f"""
            SELECT uci.userimage_id, ui.image_path, uci.layout, {DSO_NAME_SQL}
            FROM usercollageimages uci
            JOIN userimages ui ON ui.id = uci.userimage_id
            WHERE uci.collage_id = ?
            ORDER BY uci.position_index, uci.id
        """, (collage_id,))
        images = [tuple(image) for image in cursor.fetchall()]

        collage = Collage(name=name or "Collage", db_id=collage_id, dsodetailid=collage_dso,
                          background=_color_name(background))
        if layout_version is None:
            convert_grid_layout(collage, grid_width, grid_height, cell_size, spacing, images)
        else:
            collage.width = max(1, int(canvas_width or DEFAULT_CANVAS[0]))
            collage.height = max(1, int(canvas_height or DEFAULT_CANVAS[1]))
            settings = _json_dict(settings_json)
            collage.show_labels = bool(settings.get("show_labels", True))
            if settings.get("label_position") in LABEL_POSITIONS:
                collage.label_position = settings["label_position"]
            if settings.get("label_content") in dict(LABEL_CONTENTS):
                collage.label_content = settings["label_content"]
            if settings.get("placeholder_style") in dict(PLACEHOLDER_STYLES):
                collage.placeholder_style = settings["placeholder_style"]
            if isinstance(settings.get("rule"), dict):
                collage.rule = settings["rule"]
            if settings.get("background2"):
                collage.background2 = _color_name(settings["background2"])
            if settings.get("gradient") in dict(GRADIENTS):
                collage.gradient = settings["gradient"]
            # Images and placeholders, back in their saved stacking order
            ordered = []
            for order, (image_id, path, layout_json, dso_name) in enumerate(images):
                layout = _json_dict(layout_json)
                frame = Frame(image_id=image_id, image_path=path or "", dso_name=dso_name or "Unknown DSO",
                              **{key: layout[key] for key in Frame.LAYOUT_KEYS if key in layout})
                ordered.append((layout.get("z", order), frame))
            for layout in settings.get("placeholders", []):
                if isinstance(layout, dict):
                    frame = Frame(**{key: layout[key] for key in Frame.LAYOUT_KEYS if key in layout})
                    frame.dso_name = frame.slot or ""
                    ordered.append((layout.get("z", len(ordered)), frame))
            collage.frames = [frame for _z, frame in sorted(ordered, key=lambda pair: pair[0])]
        collages.append(collage)
    return collages


def save_collage(conn, collage):
    """Write the collage and its frames (inserting it if it's new). Images go
    in usercollageimages; placeholders, which have no image, in the settings."""
    now = datetime.now().isoformat()
    placeholders = [dict(frame.layout(), z=z) for z, frame in enumerate(collage.frames) if not frame.is_image]
    settings = json.dumps({"show_labels": collage.show_labels, "label_position": collage.label_position,
                           "label_content": collage.label_content,
                           "placeholder_style": collage.placeholder_style, "rule": collage.rule,
                           "background2": collage.background2, "gradient": collage.gradient,
                           "placeholders": placeholders})
    cursor = conn.cursor()
    try:
        if collage.db_id is not None:
            cursor.execute("""
                UPDATE usercollages
                SET name = ?, canvas_width = ?, canvas_height = ?, background_color = ?,
                    settings = ?, layout_version = ?, modified_date = ?
                WHERE id = ?
            """, (collage.name, collage.width, collage.height, collage.background,
                  settings, LAYOUT_VERSION, now, collage.db_id))
            if cursor.rowcount == 0:
                collage.db_id = None  # Deleted elsewhere - save it again
        if collage.db_id is None:
            cursor.execute("""
                INSERT INTO usercollages (dsodetailid, name, canvas_width, canvas_height,
                                          background_color, settings, layout_version,
                                          created_date, modified_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (collage.dsodetailid, collage.name, collage.width, collage.height,
                  collage.background, settings, LAYOUT_VERSION, now, now))
            collage.db_id = cursor.lastrowid
        cursor.execute("DELETE FROM usercollageimages WHERE collage_id = ?", (collage.db_id,))
        cursor.executemany("""
            INSERT INTO usercollageimages (collage_id, userimage_id, position_index, layout)
            VALUES (?, ?, ?, ?)
        """, [(collage.db_id, frame.image_id, z, json.dumps(dict(frame.layout(), z=z)))
              for z, frame in enumerate(collage.frames) if frame.is_image])
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_collage(conn, collage_id):
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM usercollageimages WHERE collage_id = ?", (collage_id,))
        cursor.execute("DELETE FROM usercollages WHERE id = ?", (collage_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------

def default_frame_size(collage, aspect):
    """(w, h) of a new frame for an image of the given aspect (width / height)."""
    long_side = min(collage.width, collage.height) * NEW_FRAME_FRACTION
    aspect = aspect if aspect and aspect > 0 else 1.0
    if aspect >= 1:
        return round(long_side), round(long_side / aspect)
    return round(long_side * aspect), round(long_side)


def find_free_spot(collage, w, h, occupied):
    """Top-left of the first gap on the canvas (reading order) a w x h frame
    fits into without touching the occupied rects, or None."""
    gap = min(collage.width, collage.height) * 0.02
    xs = sorted({gap} | {rect.right() + gap for rect in occupied})
    ys = sorted({gap} | {rect.bottom() + gap for rect in occupied})
    blocked = [rect.adjusted(-gap + 0.5, -gap + 0.5, gap - 0.5, gap - 0.5) for rect in occupied]
    for y in ys:
        for x in xs:
            candidate = QRectF(x, y, w, h)
            if candidate.right() > collage.width - gap + 0.5 or candidate.bottom() > collage.height - gap + 0.5:
                continue
            if not any(candidate.intersects(rect) for rect in blocked):
                return QPointF(round(x), round(y))
    return None


def grid_layout(count, cell=400, gap=16, columns=None):
    """(canvas w, h, [rects]) for count square cells in reading order - a
    catalog collage (Messier: 11 x 10)."""
    columns = columns or max(1, math.ceil(math.sqrt(count * 1.1)))
    rows = max(1, math.ceil(count / columns))
    rects = [QRectF(gap + (i % columns) * (cell + gap), gap + (i // columns) * (cell + gap), cell, cell)
             for i in range(count)]
    return columns * cell + (columns + 1) * gap, rows * cell + (rows + 1) * gap, rects


def _balanced_rows(aspects, row_count):
    """Split images, in order, into row_count rows of about equal total width."""
    total = sum(aspects)
    rows, row, running = [], [], 0.0
    for index, aspect in enumerate(aspects):
        remaining_rows = row_count - len(rows)
        remaining_images = len(aspects) - index
        boundary = total * (len(rows) + 1) / row_count
        # Close the row before this image if that lands nearer the boundary,
        # or if every remaining row still needs an image
        if row and (remaining_images < remaining_rows or
                    (remaining_rows > 1 and abs(running - boundary) < abs(running + aspect - boundary))):
            rows.append(row)
            row = []
        row.append(index)
        running += aspect
    rows.append(row)
    return rows


def justified_layout(aspects, width, height, gap):
    """Rects filling width x height with rows of images at their own shapes (a
    mosaic). Every row count is tried; the one covering the canvas best wins."""
    inner_w, inner_h = width - 2 * gap, height - 2 * gap
    count = len(aspects)
    if not count or inner_w <= 0 or inner_h <= 0:
        return []
    aspects = [a if a and a > 0 else 1.0 for a in aspects]
    best = None
    for row_count in range(1, count + 1):
        rows = _balanced_rows(aspects, row_count)
        heights = [(inner_w - gap * (len(row) - 1)) / sum(aspects[i] for i in row) for row in rows]
        total = sum(heights) + gap * (len(rows) - 1)
        if total > inner_h:  # Too tall: shrink every row to fit (they no longer span the width)
            scale = (inner_h - gap * (len(rows) - 1)) / sum(heights)
            heights = [h * scale for h in heights]
        covered = sum(aspects[i] * h * h for row, h in zip(rows, heights) for i in row)
        if best is None or covered > best[0]:
            best = (covered, rows, heights)
    _covered, rows, heights = best
    total = sum(heights) + gap * (len(rows) - 1)
    rects = [None] * count
    y = gap + (inner_h - total) / 2
    for row, row_height in zip(rows, heights):
        row_width = sum(aspects[i] * row_height for i in row) + gap * (len(row) - 1)
        x = gap + (inner_w - row_width) / 2
        for index in row:
            w = aspects[index] * row_height
            rects[index] = QRectF(round(x), round(y), round(w), round(row_height))
            x += w + gap
        y += row_height + gap
    return rects


def scale_layout(frames, old_size, new_size):
    """Scale and center frames from a canvas of old_size onto new_size."""
    scale = min(new_size[0] / old_size[0], new_size[1] / old_size[1])
    dx = (new_size[0] - old_size[0] * scale) / 2
    dy = (new_size[1] - old_size[1] * scale) / 2
    for frame in frames:
        frame.x, frame.y = round(frame.x * scale + dx), round(frame.y * scale + dy)
        frame.w, frame.h = max(MIN_FRAME, round(frame.w * scale)), max(MIN_FRAME, round(frame.h * scale))
        frame.font_size = max(6, round(frame.font_size * scale))
        frame.border_width = round(frame.border_width * scale, 1)
        frame.corner_radius = round(frame.corner_radius * scale, 1)


ARRANGEMENTS = (("grid", "Grid"), ("mosaic", "Mosaic (keep shapes)"), ("feature", "Feature + tiles"))


def reading_order(frames):
    """Frames in reading order of where they are now: rows top to bottom
    (centers within half a frame of each other), left to right in each."""
    rows = []
    for frame in sorted(frames, key=lambda f: (f.y + f.h / 2, f.x)):
        middle = frame.y + frame.h / 2
        if rows and abs(middle - rows[-1][0]) < min(frame.h, rows[-1][1]) / 2:
            rows[-1][2].append(frame)
        else:
            rows.append([middle, frame.h, [frame]])
    return [frame for _middle, _height, row in rows for frame in sorted(row, key=lambda f: f.x)]


def best_columns(count, width, height, gap):
    """Grid columns giving the biggest, squarest cells."""
    best = (0, 1)
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        cell_w, cell_h = (width - (columns - 1) * gap) / columns, (height - (rows - 1) * gap) / rows
        if cell_w > 0 and cell_h > 0 and min(cell_w, cell_h) > best[0]:
            best = (min(cell_w, cell_h), columns)
    return best[1]


def grid_rects(count, area, gap, columns=0):
    """count equal cells filling area; a short last row is centered."""
    if count <= 0:
        return []
    columns = min(count, columns or best_columns(count, area.width(), area.height(), gap))
    rows = math.ceil(count / columns)
    cell_w = (area.width() - (columns - 1) * gap) / columns
    cell_h = (area.height() - (rows - 1) * gap) / rows
    rects = []
    for index in range(count):
        row, column = divmod(index, columns)
        in_row = columns if row < rows - 1 else count - columns * (rows - 1)
        left = area.left() + (area.width() - (in_row * cell_w + (in_row - 1) * gap)) / 2
        rects.append(QRectF(left + column * (cell_w + gap), area.top() + row * (cell_h + gap), cell_w, cell_h))
    return rects


def mosaic_rects(aspects, area, gap, fill=False):
    """Rows of images at their own shapes (justified_layout) within area. fill:
    stretch the rows to fill the area too, cropping the images a little."""
    rects = [r.translated(area.left() - gap, area.top() - gap)
             for r in justified_layout(aspects, area.width() + 2 * gap, area.height() + 2 * gap, gap)]
    if not fill or not rects:
        return rects
    rows = {}
    for index, rect in enumerate(rects):
        rows.setdefault(round(rect.top()), []).append(index)
    tops = sorted(rows)
    heights = [rects[rows[top][0]].height() for top in tops]
    scale_h = (area.height() - gap * (len(tops) - 1)) / sum(heights)
    y = area.top()
    for top, height in zip(tops, heights):
        row = sorted(rows[top], key=lambda i: rects[i].left())
        row_height = height * scale_h
        widths = [rects[i].width() for i in row]
        scale_w = (area.width() - gap * (len(row) - 1)) / sum(widths)
        x = area.left()
        for index, width in zip(row, widths):
            rects[index] = QRectF(x, y, width * scale_w, row_height)
            x += width * scale_w + gap
        y += row_height + gap
    return rects


def feature_rects(count, area, gap):
    """One large frame (first) beside - or above, on a tall area - a grid of the rest."""
    if count <= 1:
        return [QRectF(area)][:count]
    if area.width() >= area.height():
        width = (area.width() - gap) * 0.62
        feature = QRectF(area.left(), area.top(), width, area.height())
        rest = QRectF(feature.right() + gap, area.top(), area.width() - width - gap, area.height())
    else:
        height = (area.height() - gap) * 0.62
        feature = QRectF(area.left(), area.top(), area.width(), height)
        rest = QRectF(area.left(), feature.bottom() + gap, area.width(), area.height() - height - gap)
    return [feature] + grid_rects(count - 1, rest, gap)


def arrange_frames(frames, mode, area, gap, aspects, columns=0, fill=False, feature_uid=None, order=None):
    """Lay frames out in area (grid / mosaic / feature), in reading order (or
    the given order); aspects: {uid: width / height of the image}."""
    order = list(order) if order is not None else reading_order(frames)
    if mode == "feature" and feature_uid is not None:
        order.sort(key=lambda f: f.uid != feature_uid)  # The feature first, the rest in order
    if mode == "mosaic":
        rects = mosaic_rects([aspects.get(f.uid) or 1.0 for f in order], area, gap, fill)
    elif mode == "feature":
        rects = feature_rects(len(order), area, gap)
    else:
        rects = grid_rects(len(order), area, gap, columns)
    for frame, rect in zip(order, rects):
        frame.set_rect(QRectF(round(rect.x()), round(rect.y()), max(MIN_FRAME, round(rect.width())),
                              max(MIN_FRAME, round(rect.height()))))
        frame.rotation = 0.0
        if frame.is_image:
            frame.fit = FIT_FILL  # Tidy cells; any crop you set is kept


def resize_rect(origin, handle, delta, from_center=False, keep_shape=False, snap=None):
    """(new rect, snapped x, snapped y) for dragging a resize handle ("tl",
    "r"...) of origin by delta: from the center (Alt), keeping the shape
    (Shift), with the dragged edges snapped by snap(value, horizontal) ->
    (shift, target) if given."""
    left, top, right, bottom = origin.left(), origin.top(), origin.right(), origin.bottom()
    center = origin.center()
    horizontal = "l" in handle or "r" in handle
    vertical = "t" in handle or "b" in handle
    if "l" in handle:
        left += delta.x()
    if "r" in handle:
        right += delta.x()
    if "t" in handle:
        top += delta.y()
    if "b" in handle:
        bottom += delta.y()

    snap_x = snap_y = None
    if snap is not None:
        if "l" in handle:
            shift, snap_x = snap(left, True)
            left += shift
        elif "r" in handle:
            shift, snap_x = snap(right, True)
            right += shift
        if "t" in handle:
            shift, snap_y = snap(top, False)
            top += shift
        elif "b" in handle:
            shift, snap_y = snap(bottom, False)
            bottom += shift

    if from_center:
        if horizontal:
            half = abs((left if "l" in handle else right) - center.x())
            left, right = center.x() - half, center.x() + half
        if vertical:
            half = abs((top if "t" in handle else bottom) - center.y())
            top, bottom = center.y() - half, center.y() + half

    width, height = max(MIN_FRAME, right - left), max(MIN_FRAME, bottom - top)
    if keep_shape:
        aspect = origin.width() / max(1.0, origin.height())
        if horizontal and vertical:
            if width / origin.width() >= height / origin.height():
                height = width / aspect
            else:
                width = height * aspect
        elif horizontal:
            height = width / aspect
        else:
            width = height * aspect
        width, height = max(MIN_FRAME, width), max(MIN_FRAME, height)

    # Anchor: the center, else the side opposite the dragged one
    if from_center or not horizontal:
        left = center.x() - width / 2 if (from_center or keep_shape) else origin.left()
    elif "l" in handle:
        left = origin.right() - width
    else:
        left = origin.left()
    if from_center or not vertical:
        top = center.y() - height / 2 if (from_center or keep_shape) else origin.top()
    elif "t" in handle:
        top = origin.bottom() - height
    else:
        top = origin.top()
    return QRectF(left, top, width, height), snap_x, snap_y


def snap_targets(canvas, others):
    """(xs, ys) a moving frame's edges and center snap to: the canvas's and the
    other frames' edges and centers."""
    xs = [canvas.left(), canvas.center().x(), canvas.right()]
    ys = [canvas.top(), canvas.center().y(), canvas.bottom()]
    for rect in others:
        xs += [rect.left(), rect.center().x(), rect.right()]
        ys += [rect.top(), rect.center().y(), rect.bottom()]
    return xs, ys


def snap_axis(values, targets, threshold):
    """(correction, target) bringing the nearest of values onto a target within
    threshold, or (0, None)."""
    best = (0.0, None)
    for value in values:
        for target in targets:
            delta = target - value
            if abs(delta) <= threshold and (best[1] is None or abs(delta) < abs(best[0])):
                best = (delta, target)
    return best


def grid_snap(values, size, threshold):
    """(correction, target) bringing the nearest of values onto a grid line, or (0, None)."""
    best = (0.0, None)
    if size <= 0:
        return best
    for value in values:
        target = round(value / size) * size
        delta = target - value
        if abs(delta) <= threshold and (best[1] is None or abs(delta) < abs(best[0])):
            best = (delta, target)
    return best


def spacing_snap(group, others, threshold, horizontal=True):
    """(correction, gap marks) making the moving group's gaps to its neighbors
    in the same row (horizontal) or column equal: centered between two, or
    repeating the gap next to one of them. Marks: [(start, end, across)] - the
    gaps to show, along the axis, at position `across`."""
    def span(r):  # Along the axis
        return (r.left(), r.right()) if horizontal else (r.top(), r.bottom())

    def overlaps(r):  # Across it
        return (r.top() < group.bottom() and r.bottom() > group.top()) if horizontal else \
               (r.left() < group.right() and r.right() > group.left())

    start, end = span(group)
    size = end - start
    line = [r for r in others if overlaps(r)]
    before = [r for r in line if span(r)[1] <= start + threshold]
    after = [r for r in line if span(r)[0] >= end - threshold]
    previous = max(before, key=lambda r: span(r)[1], default=None)
    following = min(after, key=lambda r: span(r)[0], default=None)
    candidates = []
    if previous is not None and following is not None:  # Centered between the two
        new_start = (span(previous)[1] + span(following)[0] - size) / 2
        if new_start >= span(previous)[1]:
            candidates.append((new_start - start, [(span(previous)[1], new_start), (new_start + size, span(following)[0])]))
    if previous is not None:  # The same gap as previous has to its own neighbor
        further = max((r for r in line if span(r)[1] <= span(previous)[0]), key=lambda r: span(r)[1], default=None)
        if further is not None:
            gap = span(previous)[0] - span(further)[1]
            new_start = span(previous)[1] + gap
            candidates.append((new_start - start, [(span(further)[1], span(previous)[0]), (span(previous)[1], new_start)]))
    if following is not None:
        further = min((r for r in line if span(r)[0] >= span(following)[1]), key=lambda r: span(r)[0], default=None)
        if further is not None:
            gap = span(further)[0] - span(following)[1]
            new_start = span(following)[0] - gap - size
            candidates.append((new_start - start, [(new_start + size, span(following)[0]), (span(following)[1], span(further)[0])]))
    candidates = [c for c in candidates if abs(c[0]) <= threshold]
    if not candidates:
        return 0.0, []
    shift, gaps = min(candidates, key=lambda c: abs(c[0]))
    middle = group.center().y() if horizontal else group.center().x()
    return shift, [(a, b, middle) for a, b in gaps]


ALIGNMENTS = (("left", "Align Left Edges"), ("hcenter", "Align Centers (Horizontally)"), ("right", "Align Right Edges"),
              ("top", "Align Top Edges"), ("vcenter", "Align Middles (Vertically)"), ("bottom", "Align Bottom Edges"))


def align_shift(rect, reference, how):
    """(dx, dy) moving rect to line up with reference: "left", "hcenter", "right", "top", "vcenter", "bottom"."""
    return {"left": (reference.left() - rect.left(), 0), "right": (reference.right() - rect.right(), 0),
            "hcenter": (reference.center().x() - rect.center().x(), 0),
            "top": (0, reference.top() - rect.top()), "bottom": (0, reference.bottom() - rect.bottom()),
            "vcenter": (0, reference.center().y() - rect.center().y())}[how]


def distribute_shifts(rects, horizontal=True):
    """Shifts spacing rects evenly between the outer two (same gap between each)."""
    order = sorted(range(len(rects)), key=lambda i: rects[i].left() if horizontal else rects[i].top())
    first, last = rects[order[0]], rects[order[-1]]
    lengths = [rects[i].width() if horizontal else rects[i].height() for i in order]
    total = (last.right() - first.left()) if horizontal else (last.bottom() - first.top())
    gap = (total - sum(lengths)) / max(1, len(rects) - 1)
    shifts = [0.0] * len(rects)
    position = first.left() if horizontal else first.top()
    for index, length in zip(order, lengths):
        current = rects[index].left() if horizontal else rects[index].top()
        shifts[index] = position - current
        position += length + gap
    return shifts


def guide_lines(snap_x, snap_y, rect, canvas):
    """Guide lines through the snapped x / y, across the canvas and the frame."""
    lines = []
    if snap_x is not None:
        lines.append(QLineF(snap_x, min(canvas.top(), rect.top()), snap_x, max(canvas.bottom(), rect.bottom())))
    if snap_y is not None:
        lines.append(QLineF(min(canvas.left(), rect.left()), snap_y, max(canvas.right(), rect.right()), snap_y))
    return lines


def handle_points(rect):
    """Resize handle positions of a frame rect, by name."""
    center = rect.center()
    return {"tl": rect.topLeft(), "t": QPointF(center.x(), rect.top()), "tr": rect.topRight(),
            "r": QPointF(rect.right(), center.y()), "br": rect.bottomRight(),
            "b": QPointF(center.x(), rect.bottom()), "bl": rect.bottomLeft(),
            "l": QPointF(rect.left(), center.y())}


# Direction each resize handle drags in (degrees, screen y down) on an unrotated frame
HANDLE_ANGLES = {"r": 0, "br": 45, "b": 90, "bl": 135, "l": 180, "tl": 225, "t": 270, "tr": 315}


def handle_cursor(handle, rotation=0.0):
    """The resize pointer for a handle, turned with the frame: the arrow
    nearest the direction it actually drags in."""
    angle = (HANDLE_ANGLES[handle] + rotation) % 180
    return (Qt.SizeHorCursor, Qt.SizeFDiagCursor, Qt.SizeVerCursor, Qt.SizeBDiagCursor)[round(angle / 45) % 4]


_rotate_cursor = None


def rotate_cursor():
    """A circular-arrow pointer for the rotate knob (Qt has none)."""
    global _rotate_cursor
    if _rotate_cursor is None:
        ratio = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0
        size = 24
        pixmap = QPixmap(round(size * ratio), round(size * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        arc = QRectF(5, 5, 14, 14)
        path = QPainterPath()
        path.arcMoveTo(arc, 120)
        path.arcTo(arc, 120, 270)  # Most of a circle, open at the top left
        end = path.currentPosition()
        head = QPainterPath()  # Arrowhead at the arc's end, pointing along it
        head.moveTo(end + QPointF(-4.5, -2.5))
        head.lineTo(end + QPointF(2.5, -2.0))
        head.lineTo(end + QPointF(-1.0, 4.5))
        head.closeSubpath()
        for color, width in ((QColor(0, 0, 0), 4.0), (QColor(255, 255, 255), 2.0)):  # Outlined, for any background
            painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
            painter.setBrush(color)
            painter.drawPath(head)
        painter.end()
        _rotate_cursor = QCursor(pixmap, size // 2, size // 2)
    return _rotate_cursor


# --------------------------------------------------------------------
# Suggested collages
# --------------------------------------------------------------------

MIN_SUGGESTION = 4  # A suggestion needs at least this many images

CONSTELLATIONS = {
    "AND": "Andromeda", "ANT": "Antlia", "APS": "Apus", "AQR": "Aquarius", "AQL": "Aquila", "ARA": "Ara",
    "ARI": "Aries", "AUR": "Auriga", "BOO": "Boötes", "CAE": "Caelum", "CAM": "Camelopardalis",
    "CNC": "Cancer", "CVN": "Canes Venatici", "CMA": "Canis Major", "CMI": "Canis Minor",
    "CAP": "Capricornus", "CAR": "Carina", "CAS": "Cassiopeia", "CEN": "Centaurus", "CEP": "Cepheus",
    "CET": "Cetus", "CHA": "Chamaeleon", "CIR": "Circinus", "COL": "Columba", "COM": "Coma Berenices",
    "CRA": "Corona Australis", "CRB": "Corona Borealis", "CRV": "Corvus", "CRT": "Crater", "CRU": "Crux",
    "CYG": "Cygnus", "DEL": "Delphinus", "DOR": "Dorado", "DRA": "Draco", "EQU": "Equuleus",
    "ERI": "Eridanus", "FOR": "Fornax", "GEM": "Gemini", "GRU": "Grus", "HER": "Hercules",
    "HOR": "Horologium", "HYA": "Hydra", "HYI": "Hydrus", "IND": "Indus", "LAC": "Lacerta", "LEO": "Leo",
    "LMI": "Leo Minor", "LEP": "Lepus", "LIB": "Libra", "LUP": "Lupus", "LYN": "Lynx", "LYR": "Lyra",
    "MEN": "Mensa", "MIC": "Microscopium", "MON": "Monoceros", "MUS": "Musca", "NOR": "Norma",
    "OCT": "Octans", "OPH": "Ophiuchus", "ORI": "Orion", "PAV": "Pavo", "PEG": "Pegasus", "PER": "Perseus",
    "PHE": "Phoenix", "PIC": "Pictor", "PSC": "Pisces", "PSA": "Piscis Austrinus", "PUP": "Puppis",
    "PYX": "Pyxis", "RET": "Reticulum", "SGE": "Sagitta", "SGR": "Sagittarius", "SCO": "Scorpius",
    "SCL": "Sculptor", "SCT": "Scutum", "SER": "Serpens", "SEX": "Sextans", "TAU": "Taurus",
    "TEL": "Telescopium", "TRI": "Triangulum", "TRA": "Triangulum Australe", "TUC": "Tucana",
    "UMA": "Ursa Major", "UMI": "Ursa Minor", "VEL": "Vela", "VIR": "Virgo", "VOL": "Volans",
    "VUL": "Vulpecula",
}

# Object types by dsodetail.dsotype - broad groups first
TYPE_GROUPS = (
    ("galaxies", "Galaxies", {"GALXY", "GALCL", "GX+DN"}),
    ("nebulae", "Nebulae", {"BRTNB", "CL+NB", "DRKNB", "PLNNB", "SNREM", "LMCCN", "SMCCN", "LMCDN", "SMCDN"}),
    ("clusters", "Star Clusters", {"OPNCL", "GLOCL", "LMCOC", "LMCGC", "SMCOC", "SMCGC"}),
    ("bright", "Bright Nebulae", {"BRTNB", "CL+NB", "LMCCN", "SMCCN"}),
    ("planetary", "Planetary Nebulae", {"PLNNB"}),
    ("snr", "Supernova Remnants", {"SNREM"}),
    ("dark", "Dark Nebulae", {"DRKNB", "LMCDN", "SMCDN"}),
    ("globular", "Globular Clusters", {"GLOCL", "LMCGC", "SMCGC"}),
    ("open", "Open Clusters", {"OPNCL", "LMCOC", "SMCOC"}),
)

# Catalogs whose every member gets a tile (placeholders for those not imaged)
PLACEHOLDER_CATALOGS = {"M": ("Messier", 110), "Caldwell": ("Caldwell", 109), "Sh2": ("Sharpless", 313)}
CATALOG_TITLES = {"M": "Messier", "Sh2": "Sharpless", "LBN": "Lynds Bright Nebulae",
                  "LDN": "Lynds Dark Nebulae", "Barnard": "Barnard", "Caldwell": "Caldwell"}
CATALOG_PREFIXES = {"Sh2": "Sh2-"}  # "Sh2-155", not "Sh2 155"
COMMON_NAME_WORDS = {"nebula", "galaxy", "cluster", "remnant", "loop", "veil", "cloud", "pillars", "pinwheel",
                     "whirlpool", "sombrero", "pleiades", "beehive", "rosette", "lagoon", "trifid", "eagle"}


def slot_label(catalogue, designation):
    return f"{CATALOG_PREFIXES[catalogue]}{designation}" if catalogue in CATALOG_PREFIXES \
        else f"{catalogue} {designation}"


def pick_common_name(aliases):
    """The most readable of an object's NAME aliases: "Orion Nebula", not
    "Ori Nebula" or "CRAB NEB"."""
    def score(name):
        words = name.split()
        abbreviation = any(word.upper() in CONSTELLATIONS and len(word) == 3 for word in words)
        descriptive = words[-1].lower() in COMMON_NAME_WORDS
        return (abbreviation, not descriptive, len(words) == 1, len(name))
    readable = [name for name in aliases if name and not name.isupper() and len(name) > 3
                and not CATALOG_NUMBER_NAME.search(name)]
    return min(readable, key=score) if readable else None


# "New General Catalogue 6913", "Messier Object 106", "NGC 2244 Satellite
# Cluster": a catalog number in words, not a name
CATALOG_NUMBER_NAME = re.compile(
    r"(?i)(?=.*\d)(?=.*\b(?:catalog|catalogue|messier|object|ngc|ic|index|melotte|collinder|caldwell|"
    r"sharpless|sh2|lbn|ldn|barnard|ugc|pgc|abell|cr|mel)\b)"
    r"|\b(?:M|NGC|IC|Sh2|C)[\s-]?\d+\b")  # ...or the designation itself ("Spiral Galaxy M106")


def read_capture_date(path):
    """DATE-OBS from a FITS or XISF header, as text ("" if none). FITS headers
    are 80-character text cards, read directly - astropy takes seconds to
    import just for this."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xisf":
        from SessionFileScanner import extract_xisf_header
        return str(extract_xisf_header(path).get("DATE-OBS") or "")
    with open(path, "rb") as file:
        for _block in range(64):  # Headers are a few 2880-byte blocks
            block = file.read(2880)
            if len(block) < 80:
                break
            for start in range(0, len(block) - 79, 80):
                card = block[start:start + 80].decode("ascii", "replace")
                if card.startswith("DATE-OBS="):
                    value = card[10:].strip()
                    if value.startswith("'"):  # A quoted string; "/" starts a comment only outside quotes
                        return value[1:].split("'")[0].strip()
                    return value.split("/")[0].strip()
                if card.startswith("END") and not card[3:].strip():
                    return ""
    return ""


def image_year_and_time(row, header_reader=read_capture_date):
    """(year, sortable timestamp, source) of when an image was taken: its date
    taken, else the capture date in a FITS/XISF header, else the file's date.
    source is "taken", "header" or "file"; year None if unknown."""
    text = str(row.get("date_taken") or "")
    match = re.search(r"((?:19|20)\d{2})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?", text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1)
        try:
            return year, datetime(year, min(max(month, 1), 12), min(max(day, 1), 28)).timestamp(), "taken"
        except ValueError:
            return year, datetime(year, 1, 1).timestamp(), "taken"
    path = row.get("image_path") or ""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts", ".xisf") and os.path.isfile(path):
        try:
            date_obs = header_reader(path)
            when = datetime.fromisoformat(date_obs[:19]) if date_obs else None
            if when is not None:
                return when.year, when.timestamp(), "header"
        except Exception as e:
            logger.debug(f"No capture date in {path}: {e}")
    try:
        stamp = os.path.getmtime(path)
        return datetime.fromtimestamp(stamp).year, stamp, "file"
    except OSError:
        return None, 0.0, "file"


@dataclass
class Suggestion:
    key: str                 # "all", "favorites", "year:2026", "type:galaxies", "catalog:M", "constellation:CYG"...
    title: str               # "Galaxies"
    detail: str              # "13 images" / "30 of 110 imaged"
    image_ids: list          # One image per object (all images for "all"), in collage order
    catalog: str = None      # A placeholder catalog: every member can get a tile
    sort_by_catalog: bool = False


def load_object_info(conn, dsodetailids):
    """{dsodetailid: [(catalogue, designation)]} for the user's objects."""
    ids = sorted({str(i) for i in dsodetailids if i})
    info = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        cursor = conn.execute(f"SELECT dsodetailid, catalogue, designation FROM cataloguenr "
                              f"WHERE dsodetailid IN ({','.join('?' * len(chunk))})", chunk)
        for dso, catalogue, designation in cursor.fetchall():
            info.setdefault(str(dso), []).append((catalogue, str(designation)))
    return info


def load_catalog_members(conn, catalogue):
    """[(number, slot label, dsodetailid, common name)] of a placeholder catalog,
    by number - only plain numbers (Messier also holds Minkowski "M 1-1"...)."""
    limit = PLACEHOLDER_CATALOGS[catalogue][1]
    members = {}
    for dso, designation in conn.execute(
            "SELECT dsodetailid, designation FROM cataloguenr WHERE catalogue = ?", (catalogue,)).fetchall():
        designation = str(designation).strip()
        if designation.isdigit() and 1 <= int(designation) <= limit and int(designation) not in members:
            members[int(designation)] = str(dso)
    names = {}
    ids = list(members.values())
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for dso, name in conn.execute(f"SELECT dsodetailid, designation FROM cataloguenr WHERE catalogue = 'NAME' "
                                      f"AND dsodetailid IN ({','.join('?' * len(chunk))})", chunk).fetchall():
            names.setdefault(str(dso), []).append(str(name))
    return [(number, slot_label(catalogue, number), dso, pick_common_name(names.get(dso, [])))
            for number, dso in sorted(members.items())]


def best_image_per_object(rows, times):
    """One row per object (dsodetailid): the favorite, else the newest."""
    best = {}
    for row in rows:
        key = row["dsodetailid"] or f"image:{row['id']}"
        rank = (bool(row.get("is_favorite")), times.get(row["id"], 0.0), row["id"])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, row)
    return [row for _rank, row in best.values()]


def build_suggestions(library, object_info, years, times, min_images=MIN_SUGGESTION):
    """Suggested collages for the user's images, as Suggestions (only those
    with at least min_images images)."""
    rows = list(library.values())
    by_name = lambda rs: sorted(rs, key=lambda r: (natural_key(r["dso_name"]), r["id"]))
    unique_files = {}
    for row in by_name(rows):
        unique_files.setdefault(os.path.normcase(row["image_path"]), row)  # One frame per file
    suggestions = [Suggestion("all", "All Images", "", [r["id"] for r in unique_files.values()])]

    def add(key, title, chosen, **extra):
        chosen = by_name(best_image_per_object(chosen, times))
        if len(chosen) >= min_images:
            suggestions.append(Suggestion(key, title, "", [r["id"] for r in chosen], **extra))

    add("favorites", "Favorites", [r for r in rows if r.get("is_favorite")])
    this_year = datetime.now().year
    for year in sorted({y for y in years.values() if y}, reverse=True):
        title = f"{year} (This Year)" if year == this_year else str(year)
        add(f"year:{year}", title, [r for r in rows if years.get(r["id"]) == year])

    seen_sets = []
    for key, title, codes in TYPE_GROUPS:
        chosen = [r for r in rows if (r.get("dsotype") or "") in codes]
        objects = {r["dsodetailid"] for r in chosen}
        if objects and objects not in seen_sets:  # Skip a group identical to a broader one
            seen_sets.append(objects)
            add(f"type:{key}", title, chosen)

    catalogs = {}
    for row in rows:
        for catalogue, designation in object_info.get(str(row["dsodetailid"]), []):
            if catalogue == "NAME":
                continue
            if catalogue in PLACEHOLDER_CATALOGS and not (designation.isdigit()
                                                          and 1 <= int(designation) <= PLACEHOLDER_CATALOGS[catalogue][1]):
                continue
            catalogs.setdefault(catalogue, []).append((designation, row))
    for catalogue, entries in sorted(catalogs.items(), key=lambda item: (item[0] not in PLACEHOLDER_CATALOGS,
                                                                         -len({r["dsodetailid"] for _d, r in item[1]}))):
        chosen = best_image_per_object([row for _d, row in entries], times)
        number_of = {row["id"]: designation for designation, row in entries}
        chosen.sort(key=lambda r: natural_key(number_of.get(r["id"], "")))
        if len(chosen) < min_images:
            continue
        title = CATALOG_TITLES.get(catalogue, catalogue)
        placeholder = catalogue if catalogue in PLACEHOLDER_CATALOGS else None
        suggestions.append(Suggestion(f"catalog:{catalogue}", title,
                                      f"{len(chosen)} of {PLACEHOLDER_CATALOGS[catalogue][1]} imaged" if placeholder else "",
                                      [r["id"] for r in chosen], catalog=placeholder, sort_by_catalog=True))

    for code in sorted({(r.get("constellation") or "").upper() for r in rows} - {""}):
        add(f"constellation:{code}", CONSTELLATIONS.get(code, code),
            [r for r in rows if (r.get("constellation") or "").upper() == code])
    for equipment in sorted({(r.get("equipment") or "").strip() for r in rows} - {""}, key=str.lower):
        add(f"equipment:{equipment}", equipment, [r for r in rows if (r.get("equipment") or "").strip() == equipment])

    for suggestion in suggestions:
        if not suggestion.detail:
            count = len(suggestion.image_ids)
            suggestion.detail = f"{count} image{'s' if count != 1 else ''}"
    return suggestions


def build_suggested_collage(collage, suggestion, library, aspects, members=None, placeholders=False):
    """Fill an empty collage from a suggestion: catalogs as a grid in catalog
    order (with a placeholder for every object not imaged, if asked), anything
    else as a mosaic on the collage's canvas."""
    rows = [library[i] for i in suggestion.image_ids if i in library]
    collage.rule = {"key": suggestion.key, "seen": list(suggestion.image_ids)}  # Later images get offered
    if suggestion.catalog and placeholders and members:
        collage.rule["placeholders"] = True
        image_for = {}
        for row in rows:
            image_for.setdefault(str(row["dsodetailid"]), row)
        width, height, rects = grid_layout(len(members))
        collage.width, collage.height = width, height
        for (number, slot, dso, name), rect in zip(members, rects):
            frame = Frame(x=rect.x(), y=rect.y(), w=rect.width(), h=rect.height(),
                          slot=slot, slot_name=name, slot_dso=dso, dso_name=slot)
            row = image_for.get(dso)
            if row is not None:
                frame.image_id, frame.image_path, frame.dso_name = row["id"], row["image_path"], row["dso_name"]
            collage.frames.append(frame)
        return
    if suggestion.sort_by_catalog:
        width, height, rects = grid_layout(len(rows))
        collage.width, collage.height = width, height
    else:
        gap = round(min(collage.width, collage.height) * 0.012)
        rects = justified_layout([aspects.get(r["image_path"]) or 1.5 for r in rows],
                                 collage.width, collage.height, gap)
    for row, rect in zip(rows, rects):
        collage.frames.append(Frame(image_id=row["id"], image_path=row["image_path"], dso_name=row["dso_name"],
                                    x=rect.x(), y=rect.y(), w=rect.width(), h=rect.height()))


# --------------------------------------------------------------------
# Painting - shared by the canvas and export, so they always match
# --------------------------------------------------------------------

def frame_image_rects(frame, rect, image_width, image_height):
    """(source rect in the image, target rect) to draw frame's image into rect."""
    if frame.fit == FIT_WHOLE:
        scale = min(rect.width() / image_width, rect.height() / image_height)
        width, height = image_width * scale, image_height * scale
        center = rect.center()
        return (QRectF(0, 0, image_width, image_height),
                QRectF(center.x() - width / 2, center.y() - height / 2, width, height))
    scale = max(rect.width() / image_width, rect.height() / image_height) * max(1.0, frame.zoom)
    width, height = min(image_width, rect.width() / scale), min(image_height, rect.height() / scale)
    center_x = min(max(frame.crop_x * image_width, width / 2), image_width - width / 2)
    center_y = min(max(frame.crop_y * image_height, height / 2), image_height - height / 2)
    return QRectF(center_x - width / 2, center_y - height / 2, width, height), QRectF(rect)


def paint_label(painter, text, frame_rect, image_rect, position, short_side=None, subtitle=None):
    """A DSO label: white outlined text on a translucent box, at position
    (Bottom Center...) on the visible image - never in a whole-image frame's
    empty margin, which can overlap the frames around it. subtitle: a second,
    smaller line (the common name). Its size comes from the image's short side
    (short_side, if the rects are rotated boxes)."""
    if not text:
        return
    area = image_rect.intersected(frame_rect) if image_rect is not None else QRectF(frame_rect)
    if area.isEmpty():
        area = QRectF(frame_rect)
    short_side = short_side if short_side is not None else min(area.width(), area.height())
    size = max(6, round(short_side * LABEL_SCALE))
    pad = size * 0.25
    margin = pad + size * 0.3
    room = max(1.0, area.width() - 2 * margin)
    lines = []  # (font, metrics, text)
    for line, scale in ((text, 1.0), (subtitle, 0.8)):
        if line:
            font = QFont()
            font.setPixelSize(max(5, round(size * scale)))
            metrics = QFontMetricsF(font)
            lines.append((font, metrics, metrics.elidedText(line, Qt.ElideRight, room)))
    block_width = max(metrics.horizontalAdvance(line) for _font, metrics, line in lines)
    block_height = sum(metrics.height() for _font, metrics, _line in lines)

    if position.endswith("Left"):
        x = area.left() + margin
    elif position.endswith("Right"):
        x = area.right() - block_width - margin
    else:
        x = area.center().x() - block_width / 2
    if position.startswith("Top"):
        y = area.top() + margin
    elif position.startswith("Center"):
        y = area.center().y() - block_height / 2
    else:
        y = area.bottom() - block_height - margin
    x = max(area.left() + margin, min(x, area.right() - block_width - margin))
    y = max(area.top() + margin, min(y, area.bottom() - block_height - margin))

    painter.save()
    painter.setClipRect(area, Qt.IntersectClip)  # On the image only
    painter.fillRect(QRectF(x - pad, y - pad, block_width + 2 * pad, block_height + 2 * pad), QColor(0, 0, 0, 128))
    outline = QPen(QColor(0, 0, 0), max(1.0, size * 0.12))
    outline.setJoinStyle(Qt.RoundJoin)
    top = y
    for font, metrics, line in lines:
        width = metrics.horizontalAdvance(line)
        # Lines line up with the label's side: left, right or centered
        left = x if position.endswith("Left") else (x + block_width - width if position.endswith("Right")
                                                    else x + (block_width - width) / 2)
        path = QPainterPath()
        path.addText(left, top + metrics.ascent(), font, line)
        painter.strokePath(path, outline)
        painter.fillPath(path, QColor(255, 255, 255))
        top += metrics.height()
    painter.restore()


def paint_slot(painter, rect, frame, settings, editing=False):
    """An empty catalog slot: its number (and name) on a faint tile, in the
    collage's placeholder style. Hidden ones show only on the canvas."""
    style = settings.get("placeholder_style", "number")
    ink = QColor(contrast_text(_color_name(settings.get("background"))))
    if style == "hidden":
        if editing:
            pen = QPen(ink, 1, Qt.DashLine)
            pen.setCosmetic(True)
            ink.setAlpha(60)
            pen.setColor(ink)
            painter.save()
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            painter.restore()
        return
    painter.save()
    tile = QColor(ink)
    tile.setAlpha(14)
    painter.fillRect(rect, tile)
    border = QColor(ink)
    border.setAlpha(50)
    pen = QPen(border, max(1.0, min(rect.width(), rect.height()) * 0.004))
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(rect)
    short = min(rect.width(), rect.height())
    number = QFont()
    number.setPixelSize(max(6, round(short * 0.15)))
    number.setBold(True)
    text = QColor(ink)
    text.setAlpha(110)
    painter.setPen(text)
    painter.setFont(number)
    name = frame.slot_name if style == "name" else None
    if name:
        top = QRectF(rect.left(), rect.top(), rect.width(), rect.height() * 0.55)
        painter.drawText(top, Qt.AlignHCenter | Qt.AlignBottom, frame.slot or "")
        small = QFont()
        small.setPixelSize(max(5, round(short * 0.065)))
        painter.setFont(small)
        text.setAlpha(90)
        painter.setPen(text)
        bottom = QRectF(rect.left() + short * 0.08, rect.top() + rect.height() * 0.58,
                        rect.width() - short * 0.16, rect.height() * 0.38)
        painter.drawText(bottom, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, name)
    else:
        painter.drawText(rect, Qt.AlignCenter, frame.slot or "")
    painter.restore()


GRADIENTS = (("vertical", "Top to bottom"), ("horizontal", "Left to right"),
             ("diagonal", "Diagonal"), ("radial", "From the center"))
TEXT_BOXES = ((None, "No box"), ("dark", "Dark box"), ("light", "Light box"))


def paint_background(painter, rect, settings):
    """The canvas: a color, or a gradient to background2."""
    first = QColor(_color_name(settings.get("background")))
    second = settings.get("background2")
    if not second:
        painter.fillRect(rect, first)
        return
    kind = settings.get("gradient", "vertical")
    if kind == "radial":
        gradient = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.75)
    else:
        start, end = {"horizontal": (rect.topLeft(), rect.topRight()),
                      "diagonal": (rect.topLeft(), rect.bottomRight())}.get(
            kind, (QPointF(rect.center().x(), rect.top()), QPointF(rect.center().x(), rect.bottom())))
        gradient = QLinearGradient(start, end)
    gradient.setColorAt(0, first)
    gradient.setColorAt(1, QColor(_color_name(second)))
    painter.fillRect(rect, QBrush(gradient))


def shadow_margin(frame):
    """How far a frame's drop shadow reaches past it (canvas px)."""
    if frame.shadow <= 0:
        return 0.0
    if frame.is_text:
        return frame.font_size * 0.12 * frame.shadow
    return min(frame.w, frame.h) * 0.11 * frame.shadow


def _outline(rect, radius):
    path = QPainterPath()
    radius = min(radius, rect.width() / 2, rect.height() / 2)
    if radius > 0:
        path.addRoundedRect(rect, radius, radius)
    else:
        path.addRect(rect)
    return path


def paint_shadow(painter, rect, frame):
    """A soft drop shadow: layered translucent outlines, down and right -
    painted directly so export (in a worker thread) draws the same."""
    strength = frame.shadow
    short = min(rect.width(), rect.height())
    offset, blur = short * 0.04 * strength, short * 0.07 * strength
    steps = 10
    painter.save()
    painter.setPen(Qt.NoPen)
    base = rect.translated(offset, offset)
    for step in range(steps):
        grow = blur * (1 - step / steps)
        painter.setBrush(QColor(0, 0, 0, max(1, round(150 * strength / steps))))
        painter.drawPath(_outline(base.adjusted(-grow, -grow, grow, grow), frame.corner_radius + grow))
    painter.restore()


def text_font(frame):
    font = QFont(frame.font_family) if frame.font_family else QFont()
    font.setPixelSize(max(4, round(frame.font_size)))
    font.setBold(bool(frame.bold))
    font.setItalic(bool(frame.italic))
    return font


def paint_text(painter, rect, frame, editing=False):
    """A text box: wrapped, aligned text, optionally on a rounded backing box
    and with a drop shadow."""
    painter.save()
    font = text_font(frame)
    painter.setFont(font)
    flags = {"left": Qt.AlignLeft, "right": Qt.AlignRight}.get(frame.align, Qt.AlignHCenter) \
        | Qt.AlignVCenter | Qt.TextWordWrap
    text = frame.text or ""
    if text and frame.text_box in ("dark", "light"):
        bounds = painter.boundingRect(rect, flags, text)
        pad = frame.font_size * 0.35
        box = bounds.adjusted(-pad, -pad * 0.6, pad, pad * 0.6).intersected(rect.adjusted(-pad, -pad, pad, pad))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150) if frame.text_box == "dark" else QColor(255, 255, 255, 170))
        painter.drawPath(_outline(box, frame.font_size * 0.3))
    if text and frame.shadow > 0:
        offset = frame.font_size * 0.06 * frame.shadow
        painter.setPen(QColor(0, 0, 0, round(200 * frame.shadow)))
        painter.drawText(rect.translated(offset, offset), flags, text)
    if text:
        painter.setPen(QColor(_color_name(frame.text_color)))
        painter.drawText(rect, flags, text)
    elif editing:
        hint = QColor(_color_name(frame.text_color))
        hint.setAlpha(110)
        painter.setPen(hint)
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, "Double-click to type")
    if editing and frame.text_box is None:
        outline = QColor(_color_name(frame.text_color))
        outline.setAlpha(60)
        pen = QPen(outline, 1, Qt.DotLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
    painter.restore()


def paint_element(painter, frame, image, settings, placeholder=None, device_scale=1.0, editing=False):
    """Draw a frame at its place on the canvas, rotated about its center."""
    rect = frame.rect()
    if not frame.rotation % 360:
        paint_frame(painter, rect, frame, image, settings, placeholder, device_scale, editing)
        return
    painter.save()
    painter.translate(rect.center())
    painter.rotate(frame.rotation)
    local = QRectF(-rect.width() / 2, -rect.height() / 2, rect.width(), rect.height())
    paint_frame(painter, local, frame, image, settings, placeholder, device_scale, editing)
    painter.restore()


def paint_frame(painter, rect, frame, image, settings, placeholder=None, device_scale=1.0, editing=False):
    """Draw a frame's image (QImage or QPixmap, None if unavailable), style and
    label into rect - or its placeholder tile or text. settings:
    Collage.render_settings(). device_scale is device pixels per rect unit: the
    image is downscaled to exactly that size first, which looks much better
    than letting the painter shrink it."""
    if frame.is_text:
        paint_text(painter, rect, frame, editing)
        return
    if frame.is_placeholder:
        paint_slot(painter, rect, frame, settings, editing)
        return
    image_rect = source = None
    if image is not None and not image.isNull():
        source, image_rect = frame_image_rects(frame, rect, image.width(), image.height())
    # Shadow, corners and border follow what's visible: a whole image's own edges
    visible = image_rect if image_rect is not None and frame.fit == FIT_WHOLE else rect
    if frame.shadow > 0:
        paint_shadow(painter, visible, frame)
    painter.save()
    if frame.corner_radius > 0:
        painter.setClipPath(_outline(visible, frame.corner_radius), Qt.IntersectClip)
    else:
        painter.setClipRect(rect, Qt.IntersectClip)
    if image_rect is not None:
        target = image_rect
        source_rect = source.toAlignedRect()
        piece = image if source_rect == image.rect() else image.copy(source_rect)
        size = QSize(max(1, round(target.width() * device_scale)), max(1, round(target.height() * device_scale)))
        if size.width() < piece.width() or size.height() < piece.height():
            piece = piece.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        if isinstance(piece, QImage):
            painter.drawImage(target, piece, QRectF(piece.rect()))
        else:
            painter.drawPixmap(target, piece, QRectF(piece.rect()))
    elif placeholder:
        painter.fillRect(rect, QColor(128, 128, 128, 70))
        font = QFont()
        font.setPixelSize(max(6, round(min(rect.width(), rect.height()) * 0.07)))
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, placeholder)
    painter.restore()
    if frame.border_width > 0:
        width = frame.border_width
        pen = QPen(QColor(_color_name(frame.border_color)), width)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.save()
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(_outline(visible.adjusted(width / 2, width / 2, -width / 2, -width / 2),
                                  max(0.0, frame.corner_radius - width / 2)))
        painter.restore()
    show = settings["show_labels"] if frame.label_visible is None else frame.label_visible
    if not show:
        return
    position = frame.label_position or settings["label_position"]
    content = frame.label_content or settings.get("label_content", "number")
    # The common name goes under the number - not under your own label text
    subtitle = frame.common_name if content == "name" and frame.label is None else None
    if not frame.rotation % 360:
        paint_label(painter, frame.label_text(), rect, image_rect, position, subtitle=subtitle)
        return
    # Rotated: keep the label upright, placed on the frame and image as they
    # appear on the canvas - the largest upright rects inside them
    center = rect.center()
    turn = QTransform().translate(center.x(), center.y()).rotate(frame.rotation).translate(-center.x(), -center.y())
    painter.save()
    painter.translate(center)
    painter.rotate(-frame.rotation)
    painter.translate(-center)
    visible = image_rect if image_rect is not None else rect
    paint_label(painter, frame.label_text(), upright_inner_rect(rect, frame.rotation, turn),
                upright_inner_rect(image_rect, frame.rotation, turn) if image_rect is not None else None,
                position, short_side=min(visible.width(), visible.height()), subtitle=subtitle)
    painter.restore()


def upright_inner_rect(rect, rotation, turn):
    """The largest upright (unrotated) rectangle inside rect turned by rotation
    about turn - where an upright label fits. At multiples of 90 degrees it's
    the turned rect itself."""
    width, height = rect.width(), rect.height()
    angle = math.radians(rotation)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    long_side, short_side = max(width, height), min(width, height)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        half = short_side / 2
        inner_w, inner_h = (half / sin_a, half / cos_a) if width >= height else (half / cos_a, half / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        inner_w, inner_h = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a
    center = turn.map(rect.center())
    return QRectF(center.x() - inner_w / 2, center.y() - inner_h / 2, inner_w, inner_h)


# --------------------------------------------------------------------
# Preview images - decoded in the background, cached on disk
# --------------------------------------------------------------------

MISSING_FILE = "File not found"  # PreviewStore.error() for an image file that isn't there


def _path_parts(path):
    return [part for part in re.split(r"[\\/]+", path or "") if part]


def find_moved_images(old_path, new_path, missing):
    """Missing images that moved the same way as old_path -> new_path.

    The parts the two paths end with in common (e.g. "M 42/APP/x.jpg") show
    which folder moved where ("adfed:/Our Documents" -> "H:/Our Documents");
    each missing path under the old folder is looked for under the new one.
    missing: [(image id, path)]. Returns {image id: new path} for files that exist.
    """
    old_parts, new_parts = _path_parts(old_path), _path_parts(new_path)
    common = 0
    while (common < min(len(old_parts), len(new_parts)) - 1
           and old_parts[-1 - common].lower() == new_parts[-1 - common].lower()):
        common += 1
    moved = max(common, 1)  # At least the file name: the folder moved even if the file was renamed
    old_folder = [part.lower() for part in old_parts[:len(old_parts) - moved]]
    new_folder = os.path.normpath(new_path)
    for _ in range(moved):
        new_folder = os.path.dirname(new_folder)
    if not old_folder:
        return {}
    found = {}
    for image_id, path in missing:
        parts = _path_parts(path)
        if len(parts) > len(old_folder) and [part.lower() for part in parts[:len(old_folder)]] == old_folder:
            candidate = os.path.join(new_folder, *parts[len(old_folder):])
            if os.path.isfile(candidate):
                found[image_id] = candidate
    return found


def preview_cache_dir():
    """The platform's cache folder - disposable data, kept out of the Windows
    roaming profile and out of backups of ~/Library/Application Support."""
    from pathlib import Path
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "CosmosCollection" / "cache"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "CosmosCollection"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "cosmoscollection"
    path = base / "collage_previews"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_file(path):
    """Cache file for an image - its name changes when the image file does."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = f"{os.path.normcase(os.path.abspath(path))}|{stat.st_mtime_ns}|{stat.st_size}"
    return preview_cache_dir() / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.jpg"


def load_preview_qimage(path):
    """(preview QImage, error) - from the disk cache, or decoded from the image
    and cached. Safe in a worker thread."""
    cache_file = _cache_file(path) if path else None
    if cache_file is None:
        return None, MISSING_FILE
    if cache_file.exists():
        image = QImage(str(cache_file))
        if not image.isNull():
            return image, None
    from ImageLoader import load_image_qimage
    image, error = load_image_qimage(path)
    if image is None:
        return None, error or "Can't read this image"
    if max(image.width(), image.height()) > PREVIEW_MAX:
        image = image.scaled(PREVIEW_MAX, PREVIEW_MAX, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    if not image.save(str(cache_file), "JPG", 90):
        logger.debug(f"Could not cache the preview of {path}")
    return image, None


class _PreviewSignals(QObject):
    loaded = Signal(str, QImage, str)  # path, preview (null on failure), error


class _PreviewTask(QRunnable):
    def __init__(self, path, signals):
        super().__init__()
        self.path = path
        self.signals = signals

    def run(self):
        try:
            image, error = load_preview_qimage(self.path)
        except Exception as e:
            image, error = None, str(e)
        try:
            self.signals.loaded.emit(self.path, image if image is not None else QImage(), error or "")
        except RuntimeError:
            pass  # The store was replaced/deleted while this was loading


class PreviewStore(QObject):
    """Previews and thumbnails by image path, loaded on request. ready(path)
    fires when one becomes available (or fails - see error())."""

    ready = Signal(str)

    def __init__(self):
        super().__init__()
        self._previews = {}   # path -> QPixmap, for images on a canvas
        self._thumbs = {}     # path -> QPixmap, for the library
        self._aspects = {}    # path -> width / height
        self._errors = {}
        self._wanted = set()  # paths whose full preview is kept in memory
        self._pending = set()
        self._signals = _PreviewSignals()
        self._signals.loaded.connect(self._on_loaded)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max(2, min(6, (os.cpu_count() or 4) // 2)))

    def _request(self, path):
        if path and path not in self._pending and path not in self._errors:
            self._pending.add(path)
            self._pool.start(_PreviewTask(path, self._signals))

    def preview(self, path):
        """The preview pixmap, or None while loading / if it can't be read."""
        self._wanted.add(path)
        pixmap = self._previews.get(path)
        if pixmap is None:
            self._request(path)
        return pixmap

    def thumbnail(self, path):
        pixmap = self._thumbs.get(path)
        if pixmap is None:
            self._request(path)
        return pixmap

    def aspect(self, path):
        return self._aspects.get(path)

    def error(self, path):
        return self._errors.get(path)

    def _on_loaded(self, path, image, error):
        self._pending.discard(path)
        if image.isNull():
            self._errors[path] = error or "Can't read this image"
            logger.warning(f"Collage preview unavailable for {path}: {self._errors[path]}")
        else:
            self._aspects[path] = image.width() / max(1, image.height())
            self._thumbs[path] = QPixmap.fromImage(
                image.scaled(THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if path in self._wanted:
                self._previews[path] = QPixmap.fromImage(image)
        self.ready.emit(path)


_preview_store = None


def preview_store():
    """The app-wide preview store (kept for the session, so reopening the
    Collage Builder doesn't reload anything)."""
    global _preview_store
    if _preview_store is None:
        _preview_store = PreviewStore()
    return _preview_store


# --------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------

class FrameItem(QGraphicsItem):
    """A frame on the canvas. Selection outlines and handles are drawn by the
    scene, so the item's cached rendering only changes with its content."""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self._size = (frame.w, frame.h)
        self._signature = None
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setPos(frame.x, frame.y)

    def sync(self, frame, labels, pixmap_key):
        """Follow frame's geometry, repainting only if something visible changed."""
        self.frame = frame
        if (frame.w, frame.h) != self._size:
            self.prepareGeometryChange()
            self._size = (frame.w, frame.h)
        if (self.pos().x(), self.pos().y()) != (frame.x, frame.y):
            self.setPos(frame.x, frame.y)
        self.setTransformOriginPoint(frame.w / 2, frame.h / 2)  # Rotate about the center
        if self.rotation() != frame.rotation:
            self.setRotation(frame.rotation)
        signature = (frame.w, frame.h, frame.fit, frame.crop_x, frame.crop_y, frame.zoom,
                     frame.label_text(), frame.label_visible, frame.label_position, frame.image_id,
                     frame.image_path, frame.slot, frame.slot_name, frame.border_width, frame.border_color,
                     frame.corner_radius, frame.shadow, frame.text, frame.font_family, frame.font_size,
                     frame.text_color, frame.bold, frame.italic, frame.align, frame.text_box, frame.rotation,
                     frame.label_content, frame.common_name,
                     labels, pixmap_key)
        if signature != self._signature:
            self._signature = signature
            self.prepareGeometryChange()  # The clickable shape and shadow may have changed
            self.update()

    def frame_rect(self):
        """The frame itself, in item coordinates (boundingRect adds the shadow)."""
        return QRectF(0, 0, self._size[0], self._size[1])

    def boundingRect(self):
        frame = self.frame
        margin = shadow_margin(frame) + (frame.font_size * 0.45 if frame.is_text else 0)
        return self.frame_rect().adjusted(-margin, -margin, margin, margin)

    def shape(self):
        """Where clicks hit the frame: the visible image. A whole-image frame's
        empty margins let clicks through to frames underneath."""
        rect = self.frame_rect()
        aspect = preview_store().aspect(self.frame.image_path) if self.frame.is_image else None
        if self.frame.fit == FIT_WHOLE and aspect:
            _source, rect = frame_image_rects(self.frame, rect, aspect, 1.0)
        path = QPainterPath()
        path.addRect(rect)
        return path

    def outline(self):
        """The frame's outline on the canvas (rotated, if it is)."""
        return self.mapToScene(self.frame_rect())

    def visual_rect(self):
        """The frame's bounding box on the canvas, rotation included."""
        return self.mapRectToScene(self.frame_rect())

    def paint(self, painter, option, widget=None):
        scene = self.scene()
        if scene is None or scene.collage is None:
            return
        store = preview_store()
        pixmap = store.preview(self.frame.image_path)
        placeholder = None
        if pixmap is None:
            error = store.error(self.frame.image_path)
            placeholder = (f"{MISSING_FILE}\nDouble-click to locate it" if error == MISSING_FILE
                           else error or "Loading…")
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        # Physical pixels per canvas pixel - on high-DPI screens (Retina, Windows
        # display scaling) the paint device has more pixels than logical units
        device = painter.device()
        pixel_ratio = device.devicePixelRatioF() if device is not None else 1.0
        transform = painter.worldTransform()
        device_scale = (math.hypot(transform.m11(), transform.m12()) or 1.0) * pixel_ratio  # Rotation-proof
        paint_frame(painter, self.frame_rect(), self.frame, pixmap, scene.paint_settings,
                    placeholder=placeholder, device_scale=device_scale, editing=True)


class CollageScene(QGraphicsScene):
    """The canvas of the current collage, kept in step with it by sync()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemIndexMethod(QGraphicsScene.NoIndex)  # Items move a lot
        self.collage = None
        self._items = {}  # frame uid -> FrameItem
        self.guides = []
        self.drop_target = None
        self.crop_item = None  # The frame being cropped (CollageView crop mode)
        self.spacing_marks = []  # Equal gaps found while moving
        self.grid_size = 0       # Shown grid spacing (0: no grid)
        self.paint_settings = {}  # The collage's render_settings(), refreshed by sync()
        preview_store().ready.connect(self._on_preview_ready)

    def set_collage(self, collage):
        for item in self._items.values():
            self.removeItem(item)
        self._items = {}
        self.collage = collage
        self.guides = []
        self.drop_target = None
        self.crop_item = None
        if collage is not None:
            self.sync()
        self.update()

    def _labels(self):
        """Collage settings that change how frames look."""
        c = self.collage
        return (c.show_labels, c.label_position, c.label_content, c.placeholder_style, c.background, c.background2)

    def sync(self):
        """Match the items to the collage's frames (by uid)."""
        collage = self.collage
        if collage is None:
            return
        self.paint_settings = collage.render_settings()
        canvas = collage.canvas_rect()
        margin = max(collage.width, collage.height) * 0.5
        area = canvas.adjusted(-margin, -margin, margin, margin)
        live = set()
        for z, frame in enumerate(collage.frames):
            item = self._items.get(frame.uid)
            if item is None:
                item = FrameItem(frame)
                self.addItem(item)
                self._items[frame.uid] = item
            self.refresh_item(item, frame)
            item.setZValue(z)
            live.add(frame.uid)
            area = area.united(frame.rect())
        for uid in [uid for uid in self._items if uid not in live]:
            self.removeItem(self._items.pop(uid))
        if self.drop_target is not None and self.drop_target.frame.uid not in live:
            self.drop_target = None
        self.setSceneRect(area)
        self.update()

    def refresh_item(self, item, frame=None):
        frame = frame or item.frame
        store = preview_store()
        pixmap = store.preview(frame.image_path)
        item.sync(frame, self._labels(), pixmap.cacheKey() if pixmap is not None else store.error(frame.image_path))

    def _on_preview_ready(self, path):
        if self.collage is None:
            return
        for item in self._items.values():
            if item.frame.image_path == path:
                self.refresh_item(item)

    def frame_items(self):
        return list(self._items.values())

    def selected_frame_items(self):
        return sorted((item for item in self.selectedItems() if isinstance(item, FrameItem)),
                      key=lambda item: item.zValue())

    @staticmethod
    def handle_positions(item):
        """Resize handle positions of an item on the canvas, by name (rotated with it)."""
        return {name: item.mapToScene(point) for name, point in handle_points(item.frame_rect()).items()}

    @staticmethod
    def rotate_handle(item, scale):
        """The rotate knob: ROTATE_PX (screen) above the top edge's middle."""
        return item.mapToScene(QPointF(item.frame.w / 2, -ROTATE_PX / scale))

    def set_guides(self, guides, marks=()):
        """Snap guide lines, and equal-spacing marks [(start, end, across, horizontal)]."""
        marks = list(marks)
        if guides or self.guides or marks or self.spacing_marks:
            self.guides = guides
            self.spacing_marks = marks
            self.update()

    def set_drop_target(self, item):
        if item is not self.drop_target:
            self.drop_target = item
            self.update()

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor(COLORS['background_lighter']))
        if self.collage is None:
            return
        canvas = self.collage.canvas_rect()
        offset = 6 / (abs(painter.worldTransform().m11()) or 1.0)
        painter.fillRect(canvas.translated(offset, offset), QColor(0, 0, 0, 90))
        paint_background(painter, canvas, self.paint_settings or self.collage.render_settings())

    def drawForeground(self, painter, rect):
        collage = self.collage
        if collage is None:
            return
        scale = abs(painter.worldTransform().m11()) or 1.0
        canvas = collage.canvas_rect()

        # Off the canvas isn't exported - fade it into the pasteboard
        outside = QPainterPath()
        outside.addRect(rect.united(canvas))
        inside = QPainterPath()
        inside.addRect(canvas)
        fade = QColor(COLORS['background_lighter'])
        fade.setAlpha(170)
        painter.fillPath(outside.subtracted(inside), fade)

        pen = QPen(QColor(COLORS['border']), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(canvas)

        if self.grid_size > 0:  # Faint grid over the canvas (every few lines if they'd crowd)
            step = self.grid_size
            while step * scale < 8:
                step *= 2
            ink = QColor(contrast_text(collage.background))
            ink.setAlpha(45)
            pen = QPen(ink, 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            area = rect.intersected(canvas)
            x = math.ceil(area.left() / step) * step
            while x <= area.right():
                painter.drawLine(QLineF(x, area.top(), x, area.bottom()))
                x += step
            y = math.ceil(area.top() / step) * step
            while y <= area.bottom():
                painter.drawLine(QLineF(area.left(), y, area.right(), y))
                y += step

        if not collage.frames:
            hint = QColor(contrast_text(collage.background))
            hint.setAlpha(150)
            font = QFont()
            font.setPixelSize(max(8, round(min(canvas.width(), canvas.height()) / 22)))
            painter.setFont(font)
            painter.setPen(hint)
            painter.drawText(canvas.adjusted(canvas.width() * 0.1, 0, -canvas.width() * 0.1, 0),
                             Qt.AlignCenter | Qt.TextWordWrap,
                             "Drag images here from the library,\nor double-click one to add it")

        accent = QColor(COLORS['accent'])
        if self.crop_item is not None:
            self._draw_crop(painter, accent)
            return
        selected = self.selected_frame_items()
        for item in selected:
            pen = QPen(accent, 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(item.outline())
        if len(selected) == 1:
            item = selected[0]
            size = HANDLE_PX / scale
            pen = QPen(accent, 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            top, knob = self.handle_positions(item)["t"], self.rotate_handle(item, scale)
            painter.drawLine(top, knob)
            painter.setBrush(QColor(255, 255, 255))
            for point in self.handle_positions(item).values():
                painter.drawRect(QRectF(point.x() - size / 2, point.y() - size / 2, size, size))
            painter.setBrush(accent)
            painter.drawEllipse(knob, size * 0.65, size * 0.65)

        if self.drop_target is not None:
            pen = QPen(accent, 3, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(self.drop_target.outline())

        if self.guides:
            pen = QPen(QColor(GUIDE_COLOR), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for line in self.guides:
                painter.drawLine(line)
        if self.spacing_marks:
            self._draw_spacing(painter, scale)

    def _draw_spacing(self, painter, scale):
        """Equal gaps: a line across each, with end ticks and its size in px."""
        pen = QPen(QColor(GUIDE_COLOR), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        tick = 5 / scale
        font = QFont()
        font.setPixelSize(max(1, round(11 / scale)))
        painter.setFont(font)
        for start, end, across, horizontal in self.spacing_marks:
            if horizontal:
                painter.drawLine(QLineF(start, across, end, across))
                painter.drawLine(QLineF(start, across - tick, start, across + tick))
                painter.drawLine(QLineF(end, across - tick, end, across + tick))
                painter.drawText(QRectF(start, across - 16 / scale, end - start, 14 / scale),
                                 Qt.AlignCenter, f"{end - start:.0f}")
            else:
                painter.drawLine(QLineF(across, start, across, end))
                painter.drawLine(QLineF(across - tick, start, across + tick, start))
                painter.drawLine(QLineF(across - tick, end, across + tick, end))
                painter.drawText(QRectF(across + 4 / scale, start, 60 / scale, end - start),
                                 Qt.AlignLeft | Qt.AlignVCenter, f"{end - start:.0f}")

    def crop_geometry(self, item):
        """(whole image rect on the canvas, visible part's source rect, preview
        pixmap) of a frame being cropped, or None."""
        pixmap = preview_store().preview(item.frame.image_path)
        if pixmap is None:
            return None
        frame_rect = item.visual_rect()
        source, target = frame_image_rects(item.frame, frame_rect, pixmap.width(), pixmap.height())
        scale = target.width() / source.width()
        whole = QRectF(target.left() - source.left() * scale, target.top() - source.top() * scale,
                       pixmap.width() * scale, pixmap.height() * scale)
        return whole, source, pixmap

    def _draw_crop(self, painter, accent):
        """The whole image, faint around the frame, so you see what's cropped."""
        geometry = self.crop_geometry(self.crop_item)
        frame_rect = self.crop_item.visual_rect()
        if geometry is not None:
            whole, _source, pixmap = geometry
            outside = QPainterPath()
            outside.addRect(whole)
            inside = QPainterPath()
            inside.addRect(frame_rect)
            painter.save()
            painter.setClipPath(outside.subtracted(inside))
            painter.setOpacity(0.35)
            painter.drawPixmap(whole, pixmap, QRectF(pixmap.rect()))
            painter.restore()
            pen = QPen(QColor(255, 255, 255, 140), 1, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(whole)
        pen = QPen(accent, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(frame_rect)


def _mime_image_ids(mime):
    try:
        return [int(i) for i in json.loads(bytes(mime.data(LIBRARY_MIME)).decode("utf-8"))]
    except (TypeError, ValueError):
        return []


class CollageView(QGraphicsView):
    """Canvas interaction: select, move, resize (with snapping), zoom, pan and
    dropping images from the library."""

    zoom_changed = Signal(float)

    def __init__(self, editor):
        super().__init__(editor.scene)
        self.editor = editor
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)  # Guides span the view
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFrameShape(QFrame.NoFrame)
        # Pan with Space+drag, the middle button or the wheel instead
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.fit_mode = True     # Refit when the window resizes, until the user zooms
        self._mode = None        # "move", "resize", "rubber" or "pan" while dragging
        self._space = False
        self._moved = False
        self._before = None
        self._press_scene = QPointF()
        self._press_view = None
        self._pan_last = None
        self._origins = {}
        self._targets = ([], [])
        self._resize_item = None
        self._handle = None
        self._resize_origin = QRectF()
        self.crop_item = None
        self._crop_before = None
        self._crop_start = (0.5, 0.5)
        self._start_rects = {}
        self._other_rects = []
        self._rotate_center = QPointF()

    # Zoom

    def zoom(self):
        return abs(self.transform().m11()) or 1.0

    def set_zoom(self, zoom, under_mouse=False):
        zoom = max(0.02, min(8.0, zoom))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse if under_mouse else QGraphicsView.AnchorViewCenter)
        factor = zoom / self.zoom()
        self.scale(factor, factor)
        self.fit_mode = False
        self.zoom_changed.emit(self.zoom())

    def fit(self):
        collage = self.editor.current
        if collage is None:
            return
        canvas = collage.canvas_rect()
        pad = max(canvas.width(), canvas.height()) * 0.03
        self.fitInView(canvas.adjusted(-pad, -pad, pad, pad), Qt.KeepAspectRatio)
        self.fit_mode = True
        self.zoom_changed.emit(self.zoom())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit()

    def viewportEvent(self, event):
        # Trackpad pinch (macOS, Windows precision touchpads): zoom; smart zoom
        # (two-finger double tap on macOS): fit the canvas
        if event.type() == QEvent.NativeGesture:
            if event.gestureType() == Qt.ZoomNativeGesture:
                self.set_zoom(self.zoom() * (1.0 + event.value()), under_mouse=True)
                return True
            if event.gestureType() == Qt.SmartZoomNativeGesture:
                self.fit()
                return True
        return super().viewportEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            steps = event.angleDelta().y() / 120
            if steps:
                self.set_zoom(self.zoom() * (1.15 ** steps), under_mouse=True)
            event.accept()
            return
        if self.crop_item is not None:  # Cropping: the wheel zooms the image in its frame
            steps = event.angleDelta().y() / 120 or event.pixelDelta().y() / 60
            if steps:
                self._crop_zoom(1.1 ** steps)
            event.accept()
            return
        # Pan in both directions at once (trackpads scroll diagonally); a mouse
        # wheel scrolls 60 px a notch, sideways with Shift
        pixels = event.pixelDelta()
        if not pixels.isNull():
            dx, dy = pixels.x(), pixels.y()
        else:
            dx, dy = event.angleDelta().x() / 2, event.angleDelta().y() / 2
            if event.modifiers() & Qt.ShiftModifier and not dx:
                dx, dy = dy, 0
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(dx))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(dy))
        event.accept()

    # Hit testing

    def _frame_at(self, view_pos):
        for item in self.items(view_pos):
            if isinstance(item, FrameItem):
                return item
        return None

    def frames_at(self, view_pos):
        """Every frame whose area includes view_pos, top first - hidden ones too."""
        return [item for item in self.scene().items(self.mapToScene(view_pos), Qt.IntersectsItemBoundingRect,
                                                    Qt.DescendingOrder)
                if isinstance(item, FrameItem)]

    def _handle_at(self, view_pos):
        selected = self.scene().selected_frame_items()
        if len(selected) != 1 or self.crop_item is not None:
            return None, None
        reach = HANDLE_PX / 2 + 3
        scene, item = self.scene(), selected[0]
        knob = self.mapFromScene(scene.rotate_handle(item, self.zoom()))
        if abs(knob.x() - view_pos.x()) <= reach and abs(knob.y() - view_pos.y()) <= reach:
            return "rotate", item
        for name, point in scene.handle_positions(item).items():
            screen = self.mapFromScene(point)
            if abs(screen.x() - view_pos.x()) <= reach and abs(screen.y() - view_pos.y()) <= reach:
                return name, item
        return None, None

    def _update_hover_cursor(self, view_pos):
        handle, item = self._handle_at(view_pos)
        if handle:
            cursor = rotate_cursor() if handle == "rotate" else handle_cursor(handle, item.frame.rotation)
        elif self.crop_item is not None:
            cursor = Qt.OpenHandCursor if self._in_crop_area(self.mapToScene(view_pos)) else Qt.ArrowCursor
        elif self._space:
            cursor = Qt.OpenHandCursor
        else:
            item = self._frame_at(view_pos)
            cursor = Qt.SizeAllCursor if item is not None and item.isSelected() else Qt.ArrowCursor
        self.viewport().setCursor(cursor)

    # Mouse

    def _begin_drag(self, scene_pos, view_pos, moving):
        scene = self.scene()
        self._before = self.editor.current.state()
        self._press_scene = scene_pos
        self._press_view = view_pos
        self._moved = False
        others = [item.visual_rect() for item in scene.frame_items() if item not in moving]
        self._targets = snap_targets(self.editor.current.canvas_rect(), others)
        self._start_rects = {item: item.visual_rect() for item in moving}  # What's seen, rotation included
        self._other_rects = others

    def mousePressEvent(self, event):
        self.setFocus()
        pos = event.position().toPoint()
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space):
            self._mode = "pan"
            self._pan_last = pos
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.LeftButton or self.editor.current is None:
            super().mousePressEvent(event)
            return
        scene = self.scene()
        scene_pos = self.mapToScene(pos)

        if self.crop_item is not None:
            if self._in_crop_area(scene_pos):  # Drag the image around in its frame
                frame = self.crop_item.frame
                self._mode = "crop"
                self._press_scene = scene_pos
                self._crop_start = (frame.crop_x, frame.crop_y)
                geometry = scene.crop_geometry(self.crop_item)
                if geometry is not None:  # Start from where the image actually is
                    whole, source, pixmap = geometry
                    self._crop_start = (source.center().x() / pixmap.width(), source.center().y() / pixmap.height())
                self.viewport().setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            self.end_crop(apply=True)  # Clicking elsewhere finishes cropping

        handle, item = self._handle_at(pos)
        if handle == "rotate":
            self._mode = "rotate"
            self.viewport().setCursor(rotate_cursor())  # Held for the whole drag
            self._resize_item = item
            self._rotate_center = item.mapToScene(item.frame_rect().center())
            self._begin_drag(scene_pos, pos, {item})
            event.accept()
            return
        if handle:
            self._mode = "resize"
            self.viewport().setCursor(handle_cursor(handle, item.frame.rotation))  # Held for the whole drag
            self._handle = handle
            self._resize_item = item
            self._resize_origin = QRectF(item.frame.rect())
            self._begin_drag(scene_pos, pos, {item})
            event.accept()
            return

        item = self._frame_at(pos)
        if item is not None:
            if event.modifiers() & Qt.ControlModifier:
                item.setSelected(not item.isSelected())
                if not item.isSelected():
                    event.accept()
                    return
            elif not item.isSelected():
                scene.clearSelection()
                item.setSelected(True)
            self._mode = "move"
            moving = scene.selected_frame_items()
            self._origins = {moving_item: QPointF(moving_item.pos()) for moving_item in moving}
            self._begin_drag(scene_pos, pos, set(moving))
            event.accept()
            return

        self._mode = "rubber"  # Empty canvas: select by dragging a box
        self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._mode == "pan":
            delta = pos - self._pan_last
            self._pan_last = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._mode == "crop":
            self._crop_pan(self.mapToScene(pos) - self._press_scene)
            return
        if self._mode == "rotate":
            self._moved = True
            self._drag_rotate(self.mapToScene(pos), event.modifiers())
            return
        if self._mode in ("move", "resize"):
            if not self._moved and (pos - self._press_view).manhattanLength() < QApplication.startDragDistance():
                return
            if not self._moved:
                # Show what's being dragged on top; dropping it (sync) restores its layer
                for item in (self._origins or [self._resize_item]):
                    item.setZValue(item.zValue() + 1_000_000)
            self._moved = True
            delta = self.mapToScene(pos) - self._press_scene
            if self._mode == "move":
                self._drag_move(delta, event.modifiers(), self.mapToScene(pos))
            else:
                self._drag_resize(delta, event.modifiers())
            return
        if self._mode is None:
            self._update_hover_cursor(pos)
        super().mouseMoveEvent(event)

    def _swap_target(self, pointer):
        """The frame under the pointer a single dragged frame would swap with (Ctrl held)."""
        if len(self._origins) != 1:
            return None
        moving = next(iter(self._origins))
        for item in self.scene().items(pointer, Qt.IntersectsItemShape, Qt.DescendingOrder):
            if isinstance(item, FrameItem) and item is not moving:
                return item
        return None

    def _drag_move(self, delta, modifiers, pointer):
        scene = self.scene()
        group = None
        for rect in self._start_rects.values():
            group = QRectF(rect) if group is None else group.united(rect)
        group.translate(delta)

        # Ctrl over another frame: drop to swap places with it
        swap = self._swap_target(pointer) if modifiers & Qt.ControlModifier else None
        scene.set_drop_target(swap)
        dx = dy = 0.0
        guides, marks = [], []
        if swap is None and not modifiers & Qt.AltModifier:
            options = self.editor.snap_options
            threshold = SNAP_PX / self.zoom()
            xs, ys = self._targets
            canvas = self.editor.current.canvas_rect()

            def best(horizontal):
                edges = ((group.left(), group.center().x(), group.right()) if horizontal
                         else (group.top(), group.center().y(), group.bottom()))
                choices = []  # (shift, guide position or None, spacing marks)
                if options["objects"]:
                    shift, target = snap_axis(edges, xs if horizontal else ys, threshold)
                    if target is not None:
                        choices.append((shift, target, []))
                if options["grid"]:
                    shift, target = grid_snap((edges[0], edges[2]), options["grid_size"], threshold)
                    if target is not None:
                        choices.append((shift, target, []))
                if options["spacing"]:
                    shift, gaps = spacing_snap(group, self._other_rects, threshold, horizontal)
                    if gaps:
                        choices.append((shift, None, [(a, b, across, horizontal) for a, b, across in gaps]))
                return min(choices, key=lambda c: abs(c[0])) if choices else (0.0, None, [])

            dx, snap_x, marks_x = best(True)
            dy, snap_y, marks_y = best(False)
            group.translate(dx, dy)
            guides = guide_lines(snap_x, snap_y, group, canvas)
            # Gap marks follow the snapped group across the axis
            marks = [(a, b, group.center().y(), True) for a, b, _across, _h in marks_x] + \
                    [(a, b, group.center().x(), False) for a, b, _across, _h in marks_y]
        for item, origin in self._origins.items():
            item.setPos(origin + delta + QPointF(dx, dy))
        scene.set_guides(guides, marks)

    def _drag_resize(self, delta, modifiers):
        origin, handle = self._resize_origin, self._handle
        item = self._resize_item
        from_center = bool(modifiers & Qt.AltModifier)
        keep_shape = bool(modifiers & Qt.ShiftModifier)
        angle = item.frame.rotation % 360
        if angle:
            # Rotated: resize in the frame's own (unrotated) coordinates, then put
            # it back so the side opposite the dragged one stays in place
            local_delta = QTransform().rotate(-angle).map(delta)
            local = QRectF(0, 0, origin.width(), origin.height())
            rect, _snap_x, _snap_y = resize_rect(local, handle, local_delta, from_center, keep_shape)
            center = origin.center() + QTransform().rotate(angle).map(rect.center() - local.center())
            item.frame.set_rect(QRectF(center.x() - rect.width() / 2, center.y() - rect.height() / 2,
                                       rect.width(), rect.height()))
            self.scene().refresh_item(item)
            return

        threshold = SNAP_PX / self.zoom()
        xs, ys = self._targets
        options = self.editor.snap_options

        def snap(value, horizontal):
            choices = []
            if options["objects"]:
                choices.append(snap_axis((value,), xs if horizontal else ys, threshold))
            if options["grid"]:
                choices.append(grid_snap((value,), options["grid_size"], threshold))
            choices = [c for c in choices if c[1] is not None]
            return min(choices, key=lambda c: abs(c[0])) if choices else (0.0, None)

        rect, snap_x, snap_y = resize_rect(origin, handle, delta, from_center, keep_shape, snap=snap)
        item.frame.set_rect(rect)
        self.scene().refresh_item(item)
        # Guides only where the edge still sits on its target
        if snap_x is not None and min(abs(rect.left() - snap_x), abs(rect.right() - snap_x)) > 0.01:
            snap_x = None
        if snap_y is not None and min(abs(rect.top() - snap_y), abs(rect.bottom() - snap_y)) > 0.01:
            snap_y = None
        self.scene().set_guides(guide_lines(snap_x, snap_y, rect, self.editor.current.canvas_rect()))

    def _drag_rotate(self, scene_pos, modifiers):
        """Turn the frame to face the pointer; snaps to 0/90/180/270 (Shift: 15° steps)."""
        item = self._resize_item
        vector = scene_pos - self._rotate_center
        angle = math.degrees(math.atan2(vector.y(), vector.x())) + 90
        if modifiers & Qt.ShiftModifier:
            angle = round(angle / 15) * 15
        else:
            nearest = round(angle / 90) * 90
            if abs(angle - nearest) <= ROTATE_SNAP:
                angle = nearest
        angle = (angle + 180) % 360 - 180  # -180 ... 180
        item.frame.rotation = round(angle, 1)
        self.scene().refresh_item(item)
        self.editor.status_hint(f"Rotation {item.frame.rotation:g}°")

    def mouseReleaseEvent(self, event):
        mode, self._mode = self._mode, None
        if mode in ("pan", "crop"):
            self._update_hover_cursor(event.position().toPoint())
            return
        if mode == "rubber":
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.NoDrag)
            return
        if mode == "rotate":
            if self._moved:
                self.editor.commit("Rotate", self._before)
            self._resize_item = None
            self._update_hover_cursor(event.position().toPoint())
            return
        if mode in ("move", "resize"):
            self.scene().set_guides([])
            swap = self.scene().drop_target if mode == "move" else None
            self.scene().set_drop_target(None)
            if self._moved:
                if swap is not None and len(self._origins) == 1:
                    moving = next(iter(self._origins))
                    self.editor.swap_frames(moving.frame.uid, swap.frame.uid, self._before)
                elif mode == "move":
                    for item in self._origins:
                        item.frame.x, item.frame.y = round(item.pos().x()), round(item.pos().y())
                    self.editor.commit("Move", self._before)
                else:
                    frame = self._resize_item.frame
                    frame.x, frame.y, frame.w, frame.h = (round(frame.x), round(frame.y),
                                                          round(frame.w), round(frame.h))
                    self.editor.commit("Resize", self._before)
            self._origins = {}
            self._resize_item = None
            self._update_hover_cursor(event.position().toPoint())
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.editor.current is not None:
            if self.crop_item is not None:
                self.end_crop(apply=True)
                event.accept()
                return
            item = self._frame_at(event.position().toPoint())
            if item is not None and item.frame.is_text:
                self.editor.edit_text(item.frame.uid)
                event.accept()
                return
            if item is not None and item.frame.is_image:
                if preview_store().error(item.frame.image_path) == MISSING_FILE:
                    self.editor.locate_image(item.frame.image_id)
                else:
                    self.begin_crop(item)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # Cropping: the image moves and zooms inside its frame

    def _in_crop_area(self, scene_pos):
        geometry = self.scene().crop_geometry(self.crop_item)
        area = geometry[0] if geometry is not None else self.crop_item.visual_rect()
        return area.contains(scene_pos) or self.crop_item.visual_rect().contains(scene_pos)

    def begin_crop(self, item):
        """Crop mode for a frame: drag to move the image, wheel to zoom, Enter
        or a click elsewhere to finish, Esc to cancel. A whole-image frame is
        switched to fill first (cropping is what fill does)."""
        frame = item.frame
        if not frame.is_image or preview_store().preview(frame.image_path) is None:
            return False
        if frame.rotation % 360:
            self.editor.status_hint("Set the rotation to 0° to adjust the crop (or use the Zoom slider)")
            return False
        self.end_crop(apply=True)
        self._crop_before = self.editor.current.state()
        if frame.fit != FIT_FILL:
            frame.fit = FIT_FILL
            self.scene().refresh_item(item)
        self.scene().clearSelection()
        item.setSelected(True)
        self.crop_item = item
        self.scene().crop_item = item
        self.scene().update()
        self.editor.on_crop_mode(True)
        self.setFocus()
        return True

    def end_crop(self, apply=True):
        item = self.crop_item
        if item is None:
            return
        self.crop_item = None
        self.scene().crop_item = None
        if self._mode == "crop":
            self._mode = None
        if apply:
            self.editor.commit("Crop", self._crop_before)
        else:
            self.editor.apply_state(self.editor.current, self._crop_before)
        self.editor.on_crop_mode(False)
        self.scene().update()

    def _set_crop_center(self, frame, center_x, center_y, visible_w, visible_h):
        frame.crop_x = min(max(center_x, visible_w / 2), 1 - visible_w / 2)
        frame.crop_y = min(max(center_y, visible_h / 2), 1 - visible_h / 2)

    def _crop_pan(self, delta):
        item = self.crop_item
        geometry = self.scene().crop_geometry(item)
        if geometry is None:
            return
        whole, source, pixmap = geometry
        start_x, start_y = self._crop_start
        self._set_crop_center(item.frame, start_x - delta.x() / whole.width(), start_y - delta.y() / whole.height(),
                              source.width() / pixmap.width(), source.height() / pixmap.height())
        self.scene().refresh_item(item)
        self.scene().update()

    def _crop_zoom(self, factor):
        item = self.crop_item
        frame = item.frame
        frame.zoom = min(max(frame.zoom * factor, 1.0), 10.0)
        geometry = self.scene().crop_geometry(item)
        if geometry is not None:  # Keep the center, within the image
            _whole, source, pixmap = geometry
            self._set_crop_center(frame, frame.crop_x, frame.crop_y,
                                  source.width() / pixmap.width(), source.height() / pixmap.height())
        self.scene().refresh_item(item)
        self.scene().update()
        self.editor.on_crop_mode(True)

    def select_only(self, item):
        self.scene().clearSelection()
        item.setSelected(True)
        self.setFocus()

    def cancel_drag(self):
        """Put frames back where they were before the current drag."""
        if self._mode in ("move", "resize", "rotate") and self._before is not None:
            self._mode = None
            self.scene().set_guides([])
            self.editor.apply_state(self.editor.current, self._before)
            self._origins = {}
            self._resize_item = None

    # Keyboard

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space and not event.isAutoRepeat():
            self._space = True
            if self._mode is None:
                self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if self.crop_item is not None:
            if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
                self.end_crop(apply=key != Qt.Key_Escape)
            event.accept()  # Other keys wait until cropping is done
            return
        if key == Qt.Key_Escape:
            if self._mode in ("move", "resize", "rotate"):
                self.cancel_drag()
            else:
                self.scene().clearSelection()
            event.accept()
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.editor.remove_selected()
            event.accept()
            return
        arrows = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
        if key in arrows:
            step = 10 if event.modifiers() & Qt.ShiftModifier else 1
            dx, dy = arrows[key]
            self.editor.nudge(dx * step, dy * step)
            event.accept()
            return
        if event.matches(QKeySequence.SelectAll):
            for item in self.scene().frame_items():
                item.setSelected(True)
            event.accept()
            return
        if key in (Qt.Key_Tab, Qt.Key_Backtab):
            self.select_next(backwards=key == Qt.Key_Backtab or bool(event.modifiers() & Qt.ShiftModifier))
            event.accept()
            return
        super().keyPressEvent(event)

    def focusNextPrevChild(self, forward):
        return False  # Tab cycles through the images instead of leaving the canvas

    def select_next(self, backwards=False):
        """Select the next image in reading order (Tab) - a way to reach hidden ones."""
        items = {item.frame.uid: item for item in self.scene().frame_items()}
        if not items:
            return
        order = [frame.uid for frame in reading_order([item.frame for item in items.values()])]
        selected = [item.frame.uid for item in self.scene().selected_frame_items()]
        index = order.index(selected[0]) if len(selected) == 1 and selected[0] in order else -1
        index = (index - 1 if backwards else index + 1) % len(order)
        self.select_only(items[order[index]])
        self.ensureVisible(items[order[index]].visual_rect(), 40, 40)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = False
            if self._mode is None:
                self.viewport().setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space = False
        super().focusOutEvent(event)

    def contextMenuEvent(self, event):
        if self.editor.current is None:
            return
        self.end_crop(apply=True)
        scene = self.scene()
        item = self._frame_at(event.pos())
        if item is not None and not item.isSelected():
            scene.clearSelection()
            item.setSelected(True)
        menu = QMenu(self)
        stacked = self.frames_at(event.pos())
        if len(stacked) > 1:
            # Overlapping frames: pick any of them, even one that's hidden
            pick = menu.addMenu("Select Image Here")
            for frame_item in stacked:
                action = pick.addAction(frame_item.frame.label_text() or os.path.basename(frame_item.frame.image_path))
                action.setCheckable(True)
                action.setChecked(frame_item.isSelected())
                action.triggered.connect(lambda checked=False, i=frame_item: self.select_only(i))
            menu.addSeparator()
        selected = scene.selected_frame_items()
        if item is not None and item.frame.is_text:
            menu.addAction("Edit Text", lambda: self.editor.edit_text(item.frame.uid))
            menu.addSeparator()
        if item is not None and item.frame.is_image:
            frame = item.frame
            self.editor._add_view_actions(menu, frame.image_id)
            menu.addSeparator()
            if preview_store().error(frame.image_path) == MISSING_FILE:
                menu.addAction("Locate Image…", lambda: self.editor.locate_image(frame.image_id))
            else:
                menu.addAction("Adjust Crop…", lambda: self.begin_crop(item))
                if frame.fit == FIT_FILL and (frame.zoom != 1.0 or (frame.crop_x, frame.crop_y) != (0.5, 0.5)):
                    menu.addAction("Reset Crop", self.editor.reset_crop)
            others = self.editor.other_images(frame)
            if others:
                swap = menu.addMenu(f"Other Images of {others[0]['dso_name']}")
                for row in others:
                    when = f" ({row['date_taken']})" if row.get("date_taken") else ""
                    swap.addAction(f"{os.path.basename(row['image_path'])}{when}",
                                   lambda checked=False, uid=frame.uid, i=row["id"]: self.editor.replace_image(uid, i))
            menu.addSeparator()
        choices = self.editor.merge_choices()
        if len(choices) == 1:
            menu.addAction(choices[0][0], lambda: self.editor.merge_selected(choices[0][1]))
            menu.addSeparator()
        elif choices:
            merge = menu.addMenu("Merge into One Frame")
            for text, uid in choices:
                merge.addAction(text, lambda checked=False, u=uid: self.editor.merge_selected(u))
            menu.addSeparator()
        if len(selected) == 1 and selected[0].frame.merged:
            frame = selected[0].frame
            menu.addAction(f"Unmerge {join_names(frame.merged_names())}", lambda: self.editor.unmerge(frame.uid))
            menu.addSeparator()
        if selected:
            menu.addAction("Bring to Front\tCtrl+]", lambda: self.editor.restack(to_front=True))
            menu.addAction("Send to Back\tCtrl+[", lambda: self.editor.restack(to_front=False))
            self.editor.fill_align_menu(menu.addMenu("Align"))
            if len(selected) >= 2:
                menu.addAction("Arrange Selected…\tCtrl+L", self.editor.open_arrange)
            menu.addSeparator()
            images = [frame_item for frame_item in selected if frame_item.frame.is_image]
            if images:
                fits = {frame_item.frame.fit for frame_item in images}
                fill = menu.addAction("Fill Frame (Crop Image)", lambda: self.editor.set_fit(FIT_FILL))
                fill.setCheckable(True)
                fill.setChecked(fits == {FIT_FILL})
                whole = menu.addAction("Show Whole Image", lambda: self.editor.set_fit(FIT_WHOLE))
                whole.setCheckable(True)
                whole.setChecked(fits == {FIT_WHOLE})
                menu.addSeparator()
            menu.addAction("Duplicate\tCtrl+D", self.editor.duplicate_selected)
            menu.addAction("Copy Style\tCtrl+Shift+C", self.editor.copy_style)
            paste = menu.addAction("Paste Style\tCtrl+Shift+V", self.editor.paste_style)
            paste.setEnabled(self.editor.copied_style is not None)
            menu.addSeparator()
            menu.addAction("Remove from Collage\tDel", self.editor.remove_selected)
        else:
            menu.addAction("Select All\tCtrl+A", lambda: [i.setSelected(True) for i in scene.frame_items()])
            menu.addAction("Add Text\tT", self.editor.add_text)
            menu.addAction("Arrange All…\tCtrl+L", self.editor.open_arrange)
            menu.addSeparator()
            menu.addAction("Fit Canvas in Window\tCtrl+0", self.fit)
        menu.exec(event.globalPos())

    # Drops from the library

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(LIBRARY_MIME) and self.editor.current is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if not event.mimeData().hasFormat(LIBRARY_MIME) or self.editor.current is None:
            event.ignore()
            return
        single = len(_mime_image_ids(event.mimeData())) == 1
        # One image dropped on a frame replaces that frame's image (not a text box's)
        target = self._frame_at(event.position().toPoint()) if single else None
        self.scene().set_drop_target(target if target is not None and not target.frame.is_text else None)
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.scene().set_drop_target(None)

    def dropEvent(self, event):
        ids = _mime_image_ids(event.mimeData())
        target = self.scene().drop_target
        self.scene().set_drop_target(None)
        if not ids or self.editor.current is None:
            event.ignore()
            return
        if target is not None and len(ids) == 1:
            self.editor.replace_image(target.frame.uid, ids[0])
        else:
            self.editor.add_images(ids, self.mapToScene(event.position().toPoint()))
        event.acceptProposedAction()
        self.setFocus()


class SnapshotCommand(QUndoCommand):
    """Undo step: the collage before and after an edit. Steps with the same
    merge_key (nudges, typing a canvas size) combine into one."""

    MERGE_ID = 7301

    def __init__(self, editor, collage, text, before, after, merge_key=None):
        super().__init__(text)
        self.editor = editor
        self.collage = collage
        self.before = before
        self.after = after
        self.merge_key = merge_key

    def redo(self):
        self.editor.apply_state(self.collage, self.after)

    def undo(self):
        self.editor.apply_state(self.collage, self.before)

    def id(self):
        return self.MERGE_ID if self.merge_key is not None else -1

    def mergeWith(self, other):
        if getattr(other, "merge_key", None) != self.merge_key:
            return False
        self.after = other.after
        return True


# --------------------------------------------------------------------
# Library
# --------------------------------------------------------------------

GROUP_ROLE = Qt.UserRole + 1  # Library item: ids of every entry sharing its file


def combo_index(combo, value):
    """Index of the item whose data is value (None included - findData can't match None)."""
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            return index
    return -1


def join_names(names):
    """"M 31", "M 31 & M 32", "M 8, M 20 & M 21"."""
    names = list(dict.fromkeys(name for name in names if name))
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} & {names[-1]}"


class LibraryList(QListWidget):
    """The user's images as thumbnails, dragged onto the canvas to add them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.IconMode)
        self.setIconSize(THUMB_SIZE)
        self.setGridSize(QSize(THUMB_SIZE.width() + 14, THUMB_SIZE.height() + 30))
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setUniformItemSizes(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)

    def mimeTypes(self):
        return [LIBRARY_MIME]

    def mimeData(self, items):
        mime = QMimeData()
        ids = [item.data(Qt.UserRole) for item in items]
        mime.setData(LIBRARY_MIME, QByteArray(json.dumps(ids).encode("utf-8")))
        return mime


class LibraryDelegate(QStyledItemDelegate):
    """Marks images already in the current collage with a check badge."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        ids = index.data(GROUP_ROLE) or [index.data(Qt.UserRole)]  # Every entry sharing the file
        if not self.editor.used_image_ids.intersection(ids):
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        badge = QRectF(option.rect.right() - 24, option.rect.top() + 4, 18, 18)
        accent = QColor(COLORS['accent'])
        painter.setPen(Qt.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(badge)
        painter.setPen(QColor(contrast_text(accent.name())))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignCenter, "✓")
        painter.restore()


def _placeholder_icon(text):
    pixmap = QPixmap(THUMB_SIZE)
    pixmap.fill(QColor(COLORS['background_lighter']))
    if text:
        painter = QPainter(pixmap)
        painter.setPen(QColor(COLORS['text_disabled']))
        painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.TextWordWrap, text)
        painter.end()
    return QIcon(pixmap)


# --------------------------------------------------------------------
# New Collage: start blank or from a suggestion
# --------------------------------------------------------------------

CANVAS_PRESETS = (("4K UHD (16:9)", 3840, 2160), ("Full HD (16:9)", 1920, 1080), ("Square", 4096, 4096),
                  ("Portrait 4:5", 3456, 4320), ("A4 at 300 dpi", 3508, 2480), ("Letter at 300 dpi", 3300, 2550))


def preset_index(width, height):
    """Index into CANVAS_PRESETS matching a size in either orientation, or -1."""
    for index, (_name, w, h) in enumerate(CANVAS_PRESETS):
        if (width, height) in ((w, h), (h, w)):
            return index
    return -1


class SuggestionWorker(QThread):
    """Works out the suggestions off the GUI thread - image dates may mean
    reading FITS/XISF headers or file dates on slow drives."""

    ready = Signal(object)  # {"suggestions", "members" {catalog: [...]}, "sources" {image id: date source}}

    def __init__(self, library):
        super().__init__()
        self.library = {key: dict(row) for key, row in library.items()}

    def run(self):
        import sqlite3
        try:
            years, times, sources = {}, {}, {}
            for image_id, row in self.library.items():
                years[image_id], times[image_id], sources[image_id] = image_year_and_time(row)
            from ResourceManager import ResourceManager, attach_update_catalogs
            conn = sqlite3.connect(str(ResourceManager.get_database_path()))
            try:
                attach_update_catalogs(conn)  # The same names the rest of the app sees
                info = load_object_info(conn, [row["dsodetailid"] for row in self.library.values()])
                # Every rule, however few images - the dialog shows only the bigger ones,
                # new-match banners need the rest
                suggestions = build_suggestions(self.library, info, years, times, min_images=1)
                members = {s.catalog: load_catalog_members(conn, s.catalog) for s in suggestions if s.catalog}
            finally:
                conn.close()
            self.ready.emit({"suggestions": suggestions, "members": members, "sources": sources})
        except Exception as e:
            logger.error(f"Collage suggestions failed: {e}", exc_info=True)
            self.ready.emit({"suggestions": [], "members": {}, "sources": {}, "error": str(e)})


class StartFromDialog(QDialog):
    """New Collage: pick Blank or a suggestion (All Images, Favorites, a year,
    a type of object, a catalog...), then Create."""

    CARD = QSize(168, 112)

    def __init__(self, editor, default_name):
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("New Collage")
        self.resize(900, 640)
        self.suggestions = []
        self.members = {}
        self._name_edited = False
        self.result_suggestion = None

        layout = QVBoxLayout(self)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(default_name)
        self.name_edit.textEdited.connect(lambda _text: setattr(self, "_name_edited", True))
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        self.status = QLabel("Start from a blank canvas, or from a suggestion based on your images. "
                             "Finding suggestions…")
        self.status.setWordWrap(True)
        themed_style(self.status, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.status)

        self.cards = QListWidget()
        self.cards.setViewMode(QListView.IconMode)
        self.cards.setIconSize(self.CARD)
        self.cards.setGridSize(QSize(self.CARD.width() + 24, self.CARD.height() + 50))
        self.cards.setResizeMode(QListView.Adjust)
        self.cards.setMovement(QListView.Static)
        self.cards.setWordWrap(True)
        self.cards.setUniformItemSizes(True)
        self.cards.currentRowChanged.connect(self._on_card_changed)
        self.cards.itemDoubleClicked.connect(lambda _item: self._accept())
        layout.addWidget(self.cards, 1)

        options = QFormLayout()
        self.options = options
        self.size_combo = QComboBox()
        for name, w, h in CANVAS_PRESETS:
            self.size_combo.addItem(f"{name} — {w} × {h}", (w, h))
        options.addRow("Canvas:", self.size_combo)
        self.size_note = QLabel("The canvas fits the catalog grid.")
        themed_style(self.size_note, lambda: f"color: {COLORS['text_secondary']};")
        options.addRow("", self.size_note)
        self.placeholders_check = QCheckBox()
        self.placeholders_check.toggled.connect(lambda on: self.style_combo.setEnabled(on))
        options.addRow("", self.placeholders_check)
        self.style_combo = QComboBox()
        for key, label in PLACEHOLDER_STYLES:
            self.style_combo.addItem(label, key)
        self.style_label = QLabel("Placeholders:")
        options.addRow(self.style_label, self.style_combo)
        layout.addLayout(options)

        buttons = QHBoxLayout()
        buttons.addStretch()
        create = QPushButton("Create")
        create.setDefault(True)
        create.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(create)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        blank = QListWidgetItem(self._blank_icon(), "Blank\nan empty canvas")
        blank.setData(Qt.UserRole, -1)
        self.cards.addItem(blank)
        self.cards.setCurrentRow(0)

        preview_store().ready.connect(self._refresh_icons)
        self.worker = SuggestionWorker(editor.library)
        self.worker.ready.connect(self._on_ready)
        keep_alive_until_finished(self.worker)
        self.worker.start()

    def _blank_icon(self):
        pixmap = QPixmap(self.CARD)
        pixmap.fill(QColor(COLORS['background_lighter']))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(COLORS['text_secondary']), 3)
        painter.setPen(pen)
        center = QPointF(self.CARD.width() / 2, self.CARD.height() / 2)
        painter.drawLine(center - QPointF(16, 0), center + QPointF(16, 0))
        painter.drawLine(center - QPointF(0, 16), center + QPointF(0, 16))
        painter.end()
        return QIcon(pixmap)

    def _card_icon(self, suggestion):
        """Up to four of the suggestion's images in a 2 x 2 grid."""
        pixmap = QPixmap(self.CARD)
        pixmap.fill(QColor(COLORS['background_lighter']))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        store = preview_store()
        ids = suggestion.image_ids[:4]
        columns = 1 if len(ids) == 1 else 2
        rows = 1 if len(ids) <= 2 else 2
        cell_w, cell_h = (self.CARD.width() - (columns - 1) * 2) / columns, (self.CARD.height() - (rows - 1) * 2) / rows
        for index, image_id in enumerate(ids):
            row = self.editor.library.get(image_id)
            thumb = store.thumbnail(row["image_path"]) if row else None
            cell = QRectF((index % columns) * (cell_w + 2), (index // columns) * (cell_h + 2), cell_w, cell_h)
            if thumb is not None:
                cover = Frame(fit=FIT_FILL)
                source, target = frame_image_rects(cover, cell, thumb.width(), thumb.height())
                painter.drawPixmap(target, thumb, source)
        painter.end()
        return QIcon(pixmap)

    def _on_ready(self, result):
        self.editor.set_suggestion_data(result)
        self.suggestions = [s for s in result["suggestions"]
                            if s.key == "all" or len(s.image_ids) >= MIN_SUGGESTION]
        self.members = result["members"]
        if result.get("error"):
            self.status.setText(f"Suggestions aren't available ({result['error']}) - you can still start blank.")
        elif self.suggestions:
            files = sum(1 for source in result["sources"].values() if source == "file")
            note = (f" Dates for {files} image{'s' if files != 1 else ''} without a date taken come from the "
                    f"file's capture date or modification date." if files else "")
            self.status.setText("Start from a blank canvas, or from a suggestion based on your images." + note)
        else:
            self.status.setText("No suggestions yet - add more images. You can start blank.")
        for index, suggestion in enumerate(self.suggestions):
            item = QListWidgetItem(self._card_icon(suggestion), f"{suggestion.title}\n{suggestion.detail}")
            item.setData(Qt.UserRole, index)
            item.setToolTip(f"{suggestion.title} — {suggestion.detail}")
            self.cards.addItem(item)

    def _refresh_icons(self, _path=None):
        for row in range(1, self.cards.count()):
            item = self.cards.item(row)
            item.setIcon(self._card_icon(self.suggestions[item.data(Qt.UserRole)]))

    def selected_suggestion(self):
        item = self.cards.currentItem()
        index = item.data(Qt.UserRole) if item is not None else -1
        return self.suggestions[index] if index is not None and 0 <= index < len(self.suggestions) else None

    def _on_card_changed(self, _row):
        suggestion = self.selected_suggestion()
        grid = suggestion is not None and suggestion.sort_by_catalog
        self.options.setRowVisible(self.size_combo, not grid)
        self.options.setRowVisible(self.size_note, grid)
        catalog = suggestion.catalog if suggestion is not None else None
        self.options.setRowVisible(self.placeholders_check, bool(catalog))
        self.options.setRowVisible(self.style_combo, bool(catalog))
        if catalog:
            title, size = PLACEHOLDER_CATALOGS[catalog]
            missing = len(self.members.get(catalog, [])) - len(suggestion.image_ids)
            self.placeholders_check.setText(f"Show the {missing} {title} objects not imaged yet as numbered tiles")
            self.placeholders_check.setChecked(size <= 120)  # Messier, Caldwell - not all of Sharpless
            self.style_combo.setEnabled(self.placeholders_check.isChecked())
        if not self._name_edited:
            self.name_edit.setText(self.editor._unique_name(suggestion.title if suggestion else "Collage"))

    def _accept(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "New Collage", "Please enter a name for the collage.")
            return
        if any(c.name.lower() == name.lower() for c in self.editor.collages):
            QMessageBox.warning(self, "New Collage", f"There's already a collage called “{name}”.")
            return
        self.result_suggestion = self.selected_suggestion()
        self.accept()

    def choice(self):
        """(name, suggestion or None, canvas size, placeholders, placeholder style)."""
        suggestion = self.result_suggestion
        return (self.name_edit.text().strip(), suggestion, self.size_combo.currentData(),
                bool(suggestion and suggestion.catalog and self.placeholders_check.isChecked()),
                self.style_combo.currentData())


# --------------------------------------------------------------------
# Arrange
# --------------------------------------------------------------------

class ArrangeDialog(QDialog):
    """Grid / Mosaic / Feature layout for the chosen frames, previewed live on
    the canvas (changed(params) on every change)."""

    changed = Signal(object)
    SETTING = "CollageBuilder/arrange"

    def __init__(self, editor, frames, subset, collage):
        super().__init__(editor)
        self.setWindowTitle("Arrange")
        self.setModal(True)
        layout = QVBoxLayout(self)
        count = len(frames)
        info = QLabel(f"Arranges the {count} selected images within the area they cover." if subset else
                      f"Arranges all {count} images on the canvas. Select some first to arrange just those.")
        info.setWordWrap(True)
        themed_style(info, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(info)

        saved = _json_dict(QSettings("CosmosCollection", "CosmosCollection").value(self.SETTING, "", type=str))
        default_gap = round(min(collage.width, collage.height) * 0.012)
        self.modes = QButtonGroup(self)
        mode_row = QHBoxLayout()
        for index, (key, label) in enumerate(ARRANGEMENTS):
            button = QRadioButton(label)
            button.setProperty("mode", key)
            self.modes.addButton(button, index)
            mode_row.addWidget(button)
            if key == saved.get("mode", "grid"):
                button.setChecked(True)
        if self.modes.checkedButton() is None:
            self.modes.button(0).setChecked(True)
        layout.addLayout(mode_row)

        form = QFormLayout()
        self.form = form
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(0, max(1, count))
        self.columns_spin.setSpecialValueText("Auto")
        self.columns_spin.setValue(min(int(saved.get("columns", 0)), count))
        form.addRow("Columns:", self.columns_spin)
        self.fill_check = QCheckBox("Fill the area (crops the images a little)")
        self.fill_check.setChecked(bool(saved.get("fill", False)))
        form.addRow("", self.fill_check)
        self.feature_combo = QComboBox()
        images = [f for f in frames if f.is_image] or frames
        for frame in sorted(images, key=lambda f: -f.w * f.h):  # The biggest one now first
            self.feature_combo.addItem(frame.label_text() or os.path.basename(frame.image_path), frame.uid)
        form.addRow("Feature:", self.feature_combo)
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 1000)
        self.gap_spin.setSuffix(" px")
        self.gap_spin.setValue(int(saved.get("gap", default_gap)))
        form.addRow("Spacing:", self.gap_spin)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 2000)
        self.margin_spin.setSuffix(" px")
        self.margin_spin.setValue(int(saved.get("margin", default_gap)))
        form.addRow("Margin:", self.margin_spin)
        form.setRowVisible(self.margin_spin, not subset)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(apply)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        self.modes.idToggled.connect(lambda _id, on: on and self._changed())
        for signal in (self.columns_spin.valueChanged, self.gap_spin.valueChanged, self.margin_spin.valueChanged,
                       self.fill_check.toggled, self.feature_combo.currentIndexChanged):
            signal.connect(lambda *_args: self._changed())
        self._show_options()

    def params(self):
        return {"mode": self.modes.checkedButton().property("mode"), "columns": self.columns_spin.value(),
                "fill": self.fill_check.isChecked(), "feature": self.feature_combo.currentData(),
                "gap": self.gap_spin.value(), "margin": self.margin_spin.value()}

    def _show_options(self):
        mode = self.params()["mode"]
        self.form.setRowVisible(self.columns_spin, mode == "grid")
        self.form.setRowVisible(self.fill_check, mode == "mosaic")
        self.form.setRowVisible(self.feature_combo, mode == "feature")

    def _changed(self):
        self._show_options()
        self.changed.emit(self.params())

    def _accept(self):
        params = dict(self.params())
        params.pop("feature")
        QSettings("CosmosCollection", "CosmosCollection").setValue(self.SETTING, json.dumps(params))
        self.accept()


# --------------------------------------------------------------------
# Export
# --------------------------------------------------------------------

EXPORT_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF"}


class ExportWorker(QThread):
    """Draws the collage from the full-resolution images and saves it."""

    progress = Signal(int, int, str)  # frames done, total, name being drawn
    done = Signal(bool, str, int)     # success, error message ("cancelled"), images that couldn't be read

    def __init__(self, state, path, scale=1.0):
        super().__init__()
        self.state = state
        self.path = path
        self.scale = scale
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._render()
        except Exception as e:
            logger.error(f"Collage export failed: {e}", exc_info=True)
            self.done.emit(False, str(e), 0)

    def _render(self):
        from ImageLoader import load_image_qimage
        settings = self.state["settings"]
        frames = [Frame(**data) for data in self.state["frames"]]
        width = max(1, round(settings["width"] * self.scale))
        height = max(1, round(settings["height"] * self.scale))
        image = QImage(width, height, QImage.Format_RGB32)
        if image.isNull():
            self.done.emit(False, f"Not enough memory for a {width} × {height} image", 0)
            return
        image.fill(QColor(settings["background"]))

        missing = 0
        painter = QPainter(image)
        try:
            painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
            painter.scale(self.scale, self.scale)
            paint_background(painter, QRectF(0, 0, settings["width"], settings["height"]), settings)
            loaded_path, loaded = None, None
            for index, frame in enumerate(frames):
                if self._cancelled:
                    self.done.emit(False, "cancelled", 0)
                    return
                self.progress.emit(index, len(frames), (frame.text if frame.is_text else frame.label_text())
                                   or os.path.basename(frame.image_path) or "text")
                if not frame.is_image:  # Placeholder or text box
                    paint_element(painter, frame, None, settings, device_scale=self.scale)
                    continue
                if frame.image_path != loaded_path:
                    loaded = None  # Free the previous full-size image first
                    loaded, error = load_image_qimage(frame.image_path)
                    loaded_path = frame.image_path
                    if loaded is None:
                        logger.warning(f"Collage export: can't read {frame.image_path}: {error}")
                if loaded is None:
                    missing += 1
                paint_element(painter, frame, loaded, settings, device_scale=self.scale)
        finally:
            painter.end()

        self.progress.emit(len(frames), len(frames), "")
        image_format = EXPORT_FORMATS.get(os.path.splitext(self.path)[1].lower(), "JPEG")
        ok = image.save(self.path, image_format, 95 if image_format == "JPEG" else -1)
        self.done.emit(ok, "" if ok else f"Couldn't write the {image_format} file", missing)


# --------------------------------------------------------------------
# Window
# --------------------------------------------------------------------

class CollageBuilderWindow(WindowPositionMixin, QDialog):
    """Create collages: pick a collage on the left, drag images onto its canvas."""

    WINDOW_POSITION_KEY = "CollageBuilder"
    SPLITTER_SETTING = "CollageBuilder/splitter"
    EXPORT_DIR_SETTING = "CollageBuilder/export_dir"

    def __init__(self, user_images=None, dso_name="All DSO Images", dsodetailid=None, parent=None):
        # user_images is no longer needed - the library reads the database itself
        super().__init__(parent)
        self.dsodetailid = dsodetailid
        self.collages = []
        self.current = None
        self.library = {}             # image id -> library row
        self._library_items = {}      # image id -> QListWidgetItem
        self.used_image_ids = set()   # images in the current collage (library badges)
        self._groups = {}             # file -> ids of the library entries using it
        self._common_by_dso = {}      # dsodetailid -> common name ("Bode's Galaxy")
        self.copied_style = None      # Copy Style / Paste Style
        self.snap_options = self._load_snap_options()
        self._suggestion_data = None  # SuggestionWorker results, for new-match offers
        self._suggestions_by_key = {}
        self._group_of = {}           # image id -> its file's ids
        self._stacks = {}             # id(collage) -> QUndoStack
        self._dirty = {}              # id(collage) -> collage, waiting to be saved
        self._updating_panels = False

        self.setWindowTitle("Collage Builder - CosmosCollection")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1400, 900)
        self.setup_window_position()
        self.setMinimumSize(900, 600)

        self.undo_group = QUndoGroup(self)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._save_dirty)

        self.scene = CollageScene(self)
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.view = CollageView(self)

        self._build_ui()
        self.scene.grid_size = self.snap_options["grid_size"] if self.snap_options["grid"] else 0
        self.grid_size_spin.setEnabled(self.snap_options["grid"])
        preview_store().ready.connect(self._on_preview_ready)
        theme_manager().theme_changed.connect(self._on_theme_changed)
        self._load_library()
        self._load_collages()
        self._start_suggestion_data()

    # UI

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(self._build_toolbar())

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self._build_left_panel())

        self.center = QStackedWidget()
        self.center.addWidget(self._build_empty_state())
        canvas_page = QWidget()
        canvas_layout = QVBoxLayout(canvas_page)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self._build_banner())
        canvas_layout.addWidget(self.view, 1)
        self.center.addWidget(canvas_page)
        self.splitter.addWidget(self.center)
        self.splitter.addWidget(self._build_right_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([300, 840, 270])
        saved = QSettings("CosmosCollection", "CosmosCollection").value(self.SPLITTER_SETTING)
        if saved is not None:
            self.splitter.restoreState(saved)
        root.addWidget(self.splitter, 1)
        # Enter (e.g. in the canvas size boxes) mustn't click a dialog button
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _tool_button(self, text, tooltip, slot):
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.clicked.connect(slot)
        return button

    def _build_toolbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(2)

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcuts(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.undo_group.undo)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.redo_action.triggered.connect(self.undo_group.redo)
        for action in (self.undo_action, self.redo_action):
            action.setEnabled(False)
            self.addAction(action)
        self.undo_group.canUndoChanged.connect(self.undo_action.setEnabled)
        self.undo_group.canRedoChanged.connect(self.redo_action.setEnabled)
        self.undo_group.undoTextChanged.connect(
            lambda text: self.undo_action.setToolTip(f"Undo {text} (Ctrl+Z)" if text else "Undo (Ctrl+Z)"))
        self.undo_group.redoTextChanged.connect(
            lambda text: self.redo_action.setToolTip(f"Redo {text} (Ctrl+Y)" if text else "Redo (Ctrl+Y)"))

        for action, label in ((self.undo_action, "↶ Undo"), (self.redo_action, "↷ Redo")):
            button = QToolButton()
            button.setDefaultAction(action)
            button.setAutoRaise(True)
            button.setText(label)
            action.changed.connect(lambda b=button, t=label: b.setText(t))  # Keep the arrow
            bar.addWidget(button)
        bar.addSpacing(12)

        bar.addWidget(self._tool_button("−", "Zoom out (Ctrl+−)", lambda: self.view.set_zoom(self.view.zoom() / 1.25)))
        self.zoom_button = QToolButton()
        self.zoom_button.setAutoRaise(True)
        self.zoom_button.setPopupMode(QToolButton.InstantPopup)
        self.zoom_button.setToolTip("Zoom (Ctrl+wheel)")
        zoom_menu = QMenu(self.zoom_button)
        zoom_menu.addAction("Fit Canvas (Ctrl+0)", self.view.fit)
        for percent in (25, 50, 100, 200):
            zoom_menu.addAction(f"{percent}%", lambda checked=False, p=percent: self.view.set_zoom(p / 100))
        self.zoom_button.setMenu(zoom_menu)
        self.zoom_button.setMinimumWidth(56)
        bar.addWidget(self.zoom_button)
        bar.addWidget(self._tool_button("+", "Zoom in (Ctrl++)", lambda: self.view.set_zoom(self.view.zoom() * 1.25)))
        self.view.zoom_changed.connect(lambda zoom: self.zoom_button.setText(f"{zoom * 100:.0f}%"))
        for keys, slot in ((["Ctrl+=", "Ctrl++"], lambda: self.view.set_zoom(self.view.zoom() * 1.25)),
                           (["Ctrl+-"], lambda: self.view.set_zoom(self.view.zoom() / 1.25)),
                           (["Ctrl+0"], self.view.fit),
                           (["Ctrl+1"], lambda: self.view.set_zoom(1.0))):
            action = QAction(self)
            action.setShortcuts([QKeySequence(key) for key in keys])
            action.triggered.connect(slot)
            self.addAction(action)
        bar.addSpacing(12)

        self.front_button = self._tool_button("Bring to Front", "Put the selected images on top (Ctrl+])",
                                              lambda: self.restack(to_front=True))
        self.back_button = self._tool_button("Send to Back", "Put the selected images underneath (Ctrl+[)",
                                             lambda: self.restack(to_front=False))
        self.remove_button = self._tool_button("Remove", "Remove the selected images from this collage (Delete)",
                                               self.remove_selected)
        for button in (self.front_button, self.back_button, self.remove_button):
            bar.addWidget(button)
        bar.addSpacing(12)
        self.text_button = self._tool_button("Add Text", "Add a title or caption (T)", self.add_text)
        self.arrange_button = self._tool_button(
            "Arrange…", "Lay out the selected images, or all of them, as a grid, mosaic or feature (Ctrl+L)",
            self.open_arrange)
        bar.addWidget(self.text_button)
        bar.addWidget(self.arrange_button)
        self.align_button = QToolButton()
        self.align_button.setText("Align")
        self.align_button.setToolTip("Line up, space out or size the selected images")
        self.align_button.setAutoRaise(True)
        self.align_button.setPopupMode(QToolButton.InstantPopup)
        align_menu = QMenu(self.align_button)
        align_menu.aboutToShow.connect(lambda: self.fill_align_menu(align_menu))
        self.align_button.setMenu(align_menu)
        bar.addWidget(self.align_button)
        for key, slot in (("T", self.add_text), ("Ctrl+L", self.open_arrange), ("Ctrl+D", self.duplicate_selected),
                          ("Ctrl+Shift+C", self.copy_style), ("Ctrl+Shift+V", self.paste_style)):
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(slot)
            self.view.addAction(action)  # Only while the canvas has focus (T is typed in text boxes)
        help_button = self._tool_button("?", "Keyboard and mouse shortcuts (F1)", self.show_shortcuts)
        bar.addWidget(help_button)
        help_action = QAction(self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self.show_shortcuts)
        self.addAction(help_action)
        for key, front in (("Ctrl+]", True), ("Ctrl+[", False)):
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(lambda checked=False, f=front: self.restack(to_front=f))
            self.addAction(action)

        bar.addStretch()
        self.status_label = QLabel()
        bar.addWidget(self.status_label)
        bar.addSpacing(10)

        self.export_button = QToolButton()
        self.export_button.setText("Export…")
        self.export_button.setToolTip("Save the collage as a JPEG, PNG or TIFF image (Ctrl+E)")
        self.export_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.export_button.clicked.connect(lambda: self._export(1.0))
        export_menu = QMenu(self.export_button)
        export_menu.addAction("Export Full Size…", lambda: self._export(1.0))
        export_menu.addAction("Export Half Size (for Sharing)…", lambda: self._export(0.5))
        self.export_button.setMenu(export_menu)
        themed_style(self.export_button, lambda: (
            f"QToolButton {{ background-color: {COLORS['accent']}; color: {COLORS['text_on_accent']};"
            f" font-weight: bold; padding: 4px 12px; border-radius: 3px; }}"))
        bar.addWidget(self.export_button)
        export_action = QAction(self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(lambda: self._export(1.0))
        self.addAction(export_action)
        return bar

    def _build_left_panel(self):
        panel = QSplitter(Qt.Vertical)

        collages_box = QGroupBox("Collages")
        layout = QVBoxLayout(collages_box)
        self.collage_list = QListWidget()
        self.collage_list.currentRowChanged.connect(self._on_collage_row_changed)
        self.collage_list.itemDoubleClicked.connect(lambda _item: self._rename_collage())
        layout.addWidget(self.collage_list)
        buttons = QHBoxLayout()
        for text, tooltip, slot in (("New", "Start a new collage", self._new_collage),
                                    ("Rename", "Rename this collage", self._rename_collage),
                                    ("Duplicate", "Copy this collage", self._duplicate_collage),
                                    ("Delete", "Delete this collage (your images aren't affected)", self._delete_collage)):
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        panel.addWidget(collages_box)

        library_box = QGroupBox("Images")
        layout = QVBoxLayout(library_box)
        self.library_search = QLineEdit()
        self.library_search.setPlaceholderText("Search by name, file or equipment…")
        self.library_search.setClearButtonEnabled(True)
        self.library_search.textChanged.connect(self._filter_library)
        layout.addWidget(self.library_search)
        self.library_list = LibraryList()
        self.library_list.setItemDelegate(LibraryDelegate(self))
        self.library_list.itemDoubleClicked.connect(self._on_library_double_click)
        self.library_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.library_list.customContextMenuRequested.connect(self._library_menu)
        layout.addWidget(self.library_list, 1)
        hint = QLabel("Drag images onto the canvas, or double-click to add. "
                      "Drop one on an image to replace it.")
        hint.setWordWrap(True)
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_px(11)};")
        layout.addWidget(hint)
        panel.addWidget(library_box)
        panel.setSizes([180, 600])
        return panel

    def _build_empty_state(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        title = QLabel("No collages yet")
        title.setAlignment(Qt.AlignCenter)
        themed_style(title, lambda: f"font-size: {font_px(18)}; font-weight: bold; color: {COLORS['text']};")
        layout.addWidget(title)
        text = QLabel("Create a collage, then drag your images onto its canvas.")
        text.setAlignment(Qt.AlignCenter)
        themed_style(text, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(text)
        button = QPushButton("New Collage")
        button.clicked.connect(self._new_collage)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        return page

    def _size_spin(self, minimum, maximum, slot):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(10)
        spin.setKeyboardTracking(False)  # Typing applies on Enter / leaving the box
        spin.setMinimumWidth(spin.fontMetrics().horizontalAdvance("-16000") + 40)
        spin.valueChanged.connect(slot)
        return spin

    def _build_right_panel(self):
        """Canvas settings with nothing selected; the selected images' settings
        (size, crop, label) otherwise."""
        panel = QWidget()
        panel.setMinimumWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.properties = QStackedWidget()
        for page in (self._build_canvas_page(), self._build_selection_page()):
            scroll = QScrollArea()  # Pages scroll on short screens
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.properties.addWidget(scroll)
        layout.addWidget(self.properties, 1)
        tips = QLabel(f"{'Cmd' if sys.platform == 'darwin' else 'Ctrl'}+wheel or pinch: zoom · Space+drag: pan\n"
                      "Double-click an image: crop it\n"
                      "Shift: keep shape while resizing\n"
                      "Alt: resize from the center, or move without snapping\n"
                      "Arrows: nudge (Shift: 10 px) · Tab: next image\n"
                      f"Hold {'Cmd' if sys.platform == 'darwin' else 'Ctrl'} while dropping on an image: swap · "
                      "F1: all shortcuts")
        tips.setWordWrap(True)
        themed_style(tips, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_px(11)};")
        layout.addWidget(tips)
        return panel

    def _build_canvas_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas_box = QGroupBox("Canvas")
        form = QFormLayout(canvas_box)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        for name, w, h in CANVAS_PRESETS:
            self.preset_combo.addItem(name, (w, h))
        self.preset_combo.addItem("Custom size")
        self.preset_combo.setToolTip("Choosing a size scales the layout to fit it")
        self.preset_combo.activated.connect(self._on_preset_chosen)
        preset_row.addWidget(self.preset_combo, 1)
        self.orientation_button = QToolButton()
        self.orientation_button.setText("⇄")
        self.orientation_button.setToolTip("Swap portrait / landscape (scales the layout to fit)")
        self.orientation_button.clicked.connect(self._swap_orientation)
        preset_row.addWidget(self.orientation_button)
        form.addRow("Size:", preset_row)
        size_row = QHBoxLayout()
        self.width_spin = self._size_spin(64, MAX_CANVAS, self._on_canvas_size_changed)
        self.height_spin = self._size_spin(64, MAX_CANVAS, self._on_canvas_size_changed)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        form.addRow("Pixels:", size_row)
        self.background_button = QPushButton()
        self.background_button.setToolTip("Canvas background color")
        self.background_button.clicked.connect(self._choose_background)
        form.addRow("Background:", self.background_button)
        gradient_row = QHBoxLayout()
        self.gradient_check = QCheckBox("Gradient to")
        self.gradient_check.setToolTip("Blend the background into a second color")
        self.gradient_check.toggled.connect(self._on_gradient_changed)
        gradient_row.addWidget(self.gradient_check)
        self.background2_button = QPushButton()
        self.background2_button.setToolTip("The gradient's second color")
        self.background2_button.clicked.connect(lambda: self._choose_background(second=True))
        gradient_row.addWidget(self.background2_button, 1)
        form.addRow("", gradient_row)
        self.gradient_combo = QComboBox()
        for key, label in GRADIENTS:
            self.gradient_combo.addItem(label, key)
        self.gradient_combo.activated.connect(self._on_gradient_changed)
        form.addRow("", self.gradient_combo)
        self.canvas_info = QLabel()
        themed_style(self.canvas_info, lambda: f"color: {COLORS['text_secondary']};")
        form.addRow(self.canvas_info)
        self.missing_label = QLabel()
        self.missing_label.setWordWrap(True)
        self.missing_label.setToolTip("Find the images this collage can't load (moved or renamed files)")
        self.missing_label.linkActivated.connect(lambda _link: self._locate_missing_in_collage())
        themed_style(self.missing_label, lambda: f"color: {COLORS['warning']};")
        self.missing_label.hide()
        form.addRow(self.missing_label)
        layout.addWidget(canvas_box)

        snap_box = QGroupBox("Snapping")
        snap_layout = QVBoxLayout(snap_box)
        self.snap_objects_check = QCheckBox("To images and the canvas")
        self.snap_objects_check.setToolTip("Edges and centers line up with other images and the canvas")
        self.snap_spacing_check = QCheckBox("Equal spacing")
        self.snap_spacing_check.setToolTip("Gaps between images in a row or column snap to the same size")
        grid_row = QHBoxLayout()
        self.snap_grid_check = QCheckBox("Grid every")
        self.snap_grid_check.setToolTip("Edges snap to a grid (shown on the canvas)")
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(4, 4000)
        self.grid_size_spin.setSuffix(" px")
        self.grid_size_spin.setKeyboardTracking(False)
        grid_row.addWidget(self.snap_grid_check)
        grid_row.addWidget(self.grid_size_spin, 1)
        options = self.snap_options
        self.snap_objects_check.setChecked(options["objects"])
        self.snap_spacing_check.setChecked(options["spacing"])
        self.snap_grid_check.setChecked(options["grid"])
        self.grid_size_spin.setValue(options["grid_size"])
        for widget in (self.snap_objects_check, self.snap_spacing_check, self.snap_grid_check):
            widget.toggled.connect(self._on_snap_options_changed)
        self.grid_size_spin.valueChanged.connect(self._on_snap_options_changed)
        snap_layout.addWidget(self.snap_objects_check)
        snap_layout.addWidget(self.snap_spacing_check)
        snap_layout.addLayout(grid_row)
        hint = QLabel("Hold Alt while moving to place freely.")
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_px(11)};")
        snap_layout.addWidget(hint)
        layout.addWidget(snap_box)

        labels_box = QGroupBox("Labels")
        labels_layout = QVBoxLayout(labels_box)
        self.labels_check = QCheckBox("Show DSO names")
        self.labels_check.toggled.connect(self._on_labels_changed)
        labels_layout.addWidget(self.labels_check)
        self.label_position_combo = QComboBox()
        self.label_position_combo.addItems(LABEL_POSITIONS)
        self.label_position_combo.currentTextChanged.connect(self._on_labels_changed)
        labels_layout.addWidget(self.label_position_combo)
        self.label_content_combo = QComboBox()
        for key, label in LABEL_CONTENTS:
            self.label_content_combo.addItem(label, key)
        self.label_content_combo.setToolTip("Number: “M 81”. Number and name: the common name too "
                                            "(“Bode's Galaxy”), on a second line")
        self.label_content_combo.activated.connect(lambda _i: self._on_labels_changed())
        labels_layout.addWidget(self.label_content_combo)
        layout.addWidget(labels_box)

        self.slots_box = QGroupBox("Catalog Placeholders")
        slots_layout = QVBoxLayout(self.slots_box)
        self.slots_info = QLabel()
        self.slots_info.setWordWrap(True)
        slots_layout.addWidget(self.slots_info)
        self.slot_style_combo = QComboBox()
        for key, label in PLACEHOLDER_STYLES:
            self.slot_style_combo.addItem(label, key)
        self.slot_style_combo.setToolTip("How objects not imaged yet show (on the canvas and in exports)")
        self.slot_style_combo.activated.connect(self._on_slot_style_changed)
        slots_layout.addWidget(self.slot_style_combo)
        layout.addWidget(self.slots_box)

        hint = QLabel("Click an image to change its crop and label, or drag a box around several.")
        hint.setWordWrap(True)
        themed_style(hint, lambda: f"color: {COLORS['text_secondary']};")
        layout.addWidget(hint)
        layout.addStretch()
        return page

    def _build_selection_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.selection_label = QLabel()
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        self.geometry_box = QGroupBox("Position and Size")
        form = QFormLayout(self.geometry_box)
        self.x_spin = self._size_spin(-MAX_CANVAS, MAX_CANVAS, self._on_geometry_changed)
        self.y_spin = self._size_spin(-MAX_CANVAS, MAX_CANVAS, self._on_geometry_changed)
        self.w_spin = self._size_spin(MIN_FRAME, MAX_CANVAS, self._on_geometry_changed)
        self.h_spin = self._size_spin(MIN_FRAME, MAX_CANVAS, self._on_geometry_changed)
        for label, first, second in (("X, Y:", self.x_spin, self.y_spin), ("W × H:", self.w_spin, self.h_spin)):
            row = QHBoxLayout()
            row.addWidget(first)
            row.addWidget(second)
            form.addRow(label, row)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setWrapping(True)
        self.rotation_spin.setKeyboardTracking(False)
        self.rotation_spin.setToolTip("Turn about the center (or drag the round handle above the frame)")
        self.rotation_spin.valueChanged.connect(self._on_rotation_changed)
        form.addRow("Rotation:", self.rotation_spin)
        layout.addWidget(self.geometry_box)

        self.text_group = QGroupBox("Text")
        form = QFormLayout(self.text_group)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFixedHeight(70)
        self.text_edit.setPlaceholderText("Type the title or caption")
        self.text_edit.textChanged.connect(self._on_text_changed)
        form.addRow(self.text_edit)
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(lambda _font: self._on_text_style_changed())
        form.addRow("Font:", self.font_combo)
        size_row = QHBoxLayout()
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 4000)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.setKeyboardTracking(False)
        self.font_size_spin.valueChanged.connect(lambda _v: self._on_text_style_changed())
        size_row.addWidget(self.font_size_spin, 1)
        self.bold_button = QToolButton()
        self.bold_button.setText("B")
        self.bold_button.setCheckable(True)
        self.bold_button.setToolTip("Bold")
        self.italic_button = QToolButton()
        self.italic_button.setText("I")
        self.italic_button.setCheckable(True)
        self.italic_button.setToolTip("Italic")
        for button in (self.bold_button, self.italic_button):
            button.toggled.connect(lambda _on: self._on_text_style_changed())
            size_row.addWidget(button)
        form.addRow("Size:", size_row)
        self.text_color_button = QPushButton()
        self.text_color_button.setToolTip("Text color")
        self.text_color_button.clicked.connect(self._choose_text_color)
        form.addRow("Color:", self.text_color_button)
        self.align_combo = QComboBox()
        for key, label in (("left", "Left"), ("center", "Center"), ("right", "Right")):
            self.align_combo.addItem(label, key)
        self.align_combo.activated.connect(lambda _i: self._on_text_style_changed())
        form.addRow("Align:", self.align_combo)
        self.text_box_combo = QComboBox()
        for key, label in TEXT_BOXES:
            self.text_box_combo.addItem(label, key)
        self.text_box_combo.setToolTip("A translucent box behind the text, so it reads over images")
        self.text_box_combo.activated.connect(lambda _i: self._on_text_style_changed())
        form.addRow("Backing:", self.text_box_combo)
        layout.addWidget(self.text_group)

        self.style_group = QGroupBox("Style")
        form = QFormLayout(self.style_group)
        border_row = QHBoxLayout()
        self.border_spin = QSpinBox()
        self.border_spin.setRange(0, 400)
        self.border_spin.setSuffix(" px")
        self.border_spin.setKeyboardTracking(False)
        self.border_spin.setToolTip("Border width (0: none)")
        self.border_spin.valueChanged.connect(lambda _v: self._on_style_changed("border"))
        border_row.addWidget(self.border_spin, 1)
        self.border_color_button = QPushButton()
        self.border_color_button.setToolTip("Border color")
        self.border_color_button.clicked.connect(self._choose_border_color)
        border_row.addWidget(self.border_color_button, 1)
        self.border_label = QLabel("Border:")
        form.addRow(self.border_label, border_row)
        self.corner_spin = QSpinBox()
        self.corner_spin.setRange(0, 4000)
        self.corner_spin.setSuffix(" px")
        self.corner_spin.setKeyboardTracking(False)
        self.corner_spin.setToolTip("Rounded corners (0: square)")
        self.corner_spin.valueChanged.connect(lambda _v: self._on_style_changed("corners"))
        self.corner_label = QLabel("Corners:")
        form.addRow(self.corner_label, self.corner_spin)
        self.shadow_slider = QSlider(Qt.Horizontal)
        self.shadow_slider.setRange(0, 100)
        self.shadow_slider.setToolTip("Drop shadow strength (0: none)")
        self.shadow_slider.valueChanged.connect(lambda _v: self._on_style_changed("shadow"))
        form.addRow("Shadow:", self.shadow_slider)
        layout.addWidget(self.style_group)

        self.image_box = QGroupBox("Image")
        image_layout = QVBoxLayout(self.image_box)
        self.fit_combo = QComboBox()
        self.fit_combo.addItem("Fill frame (crop image)", FIT_FILL)
        self.fit_combo.addItem("Show whole image", FIT_WHOLE)
        self.fit_combo.setToolTip("Fill: the image covers the frame, cropping what doesn't fit.\n"
                                  "Whole image: the entire image shows, with background around it.")
        self.fit_combo.activated.connect(lambda index: self.set_fit(self.fit_combo.itemData(index)))
        image_layout.addWidget(self.fit_combo)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 800)  # 1x - 8x
        self.zoom_slider.setToolTip("Zoom into the image within its frame")
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        zoom_row.addWidget(self.zoom_slider, 1)
        image_layout.addLayout(zoom_row)
        crop_row = QHBoxLayout()
        self.crop_button = QPushButton("Adjust Crop…")
        self.crop_button.setToolTip("Drag the image within its frame (or double-click the image)")
        self.crop_button.clicked.connect(self._adjust_crop)
        self.reset_crop_button = QPushButton("Reset")
        self.reset_crop_button.setToolTip("Center the image and zoom out")
        self.reset_crop_button.clicked.connect(self.reset_crop)
        crop_row.addWidget(self.crop_button, 1)
        crop_row.addWidget(self.reset_crop_button)
        image_layout.addLayout(crop_row)
        layout.addWidget(self.image_box)

        self.label_box = QGroupBox("Label")
        form = QFormLayout(self.label_box)
        self.frame_label_show = QComboBox()
        self.frame_label_show.addItem("Collage setting", None)
        self.frame_label_show.addItem("Show", True)
        self.frame_label_show.addItem("Hide", False)
        self.frame_label_show.activated.connect(self._on_frame_label_changed)
        form.addRow("Show:", self.frame_label_show)
        self.frame_label_text = QLineEdit()
        self.frame_label_text.setClearButtonEnabled(True)
        self.frame_label_text.setToolTip("Your own label text - clear it to use the DSO name again")
        self.frame_label_text.editingFinished.connect(self._on_frame_label_changed)
        form.addRow("Text:", self.frame_label_text)
        self.frame_label_position = QComboBox()
        self.frame_label_position.addItem("Collage setting", None)
        for position in LABEL_POSITIONS:
            self.frame_label_position.addItem(position, position)
        self.frame_label_position.activated.connect(self._on_frame_label_changed)
        form.addRow("Position:", self.frame_label_position)
        self.frame_label_content = QComboBox()
        self.frame_label_content.addItem("Collage setting", None)
        for key, label in LABEL_CONTENTS:
            self.frame_label_content.addItem(label, key)
        self.frame_label_content.activated.connect(self._on_frame_label_changed)
        form.addRow("Shows:", self.frame_label_content)
        layout.addWidget(self.label_box)

        self.objects_box = QGroupBox("Objects")
        objects_layout = QVBoxLayout(self.objects_box)
        self.objects_label = QLabel()
        self.objects_label.setWordWrap(True)
        objects_layout.addWidget(self.objects_label)
        self.merge_button = QPushButton("Merge into One Frame…")
        self.merge_button.setToolTip("Show the selected objects in one frame - e.g. M 31 and M 32 in one image")
        self.merge_button.clicked.connect(self._merge_from_panel)
        objects_layout.addWidget(self.merge_button)
        self.unmerge_button = QPushButton("Unmerge")
        self.unmerge_button.setToolTip("Put the merged objects back in their own frames")
        self.unmerge_button.clicked.connect(
            lambda: self.unmerge(next(iter(self._selected_uids()))) if len(self._selected_uids()) == 1 else None)
        objects_layout.addWidget(self.unmerge_button)
        layout.addWidget(self.objects_box)

        back = QLabel("<a href='canvas'>Canvas settings</a>")
        back.linkActivated.connect(lambda _link: self.scene.clearSelection())
        layout.addWidget(back)
        layout.addStretch()
        return page

    def _set_status(self, text, error=False):
        self.status_label.setText(text)
        themed_style(self.status_label, lambda: f"color: {COLORS['error'] if error else COLORS['text_secondary']};")

    def status_hint(self, text):
        """A passing message in the status area (until the next save)."""
        self._set_status(text)

    @staticmethod
    def _color_button(button, color):
        button.setText(color.upper())
        button.setStyleSheet(f"QPushButton {{ background-color: {color}; color: {contrast_text(color)}; }}")

    # New images for suggested collages

    def _build_banner(self):
        """A bar above the canvas offering images that match a suggested
        collage's rule (e.g. a new galaxy for "Galaxies")."""
        self.banner = QFrame()
        self.banner.setObjectName("matchBanner")
        themed_style(self.banner, lambda: f"#matchBanner {{ background-color: {COLORS['background_light']};"
                                          f" border-bottom: 1px solid {COLORS['border']}; }}")
        row = QHBoxLayout(self.banner)
        row.setContentsMargins(10, 6, 10, 6)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        row.addWidget(self.banner_label, 1)
        self.banner_add = QPushButton()
        self.banner_add.clicked.connect(self.add_new_matches)
        row.addWidget(self.banner_add)
        later = QPushButton("Not Now")
        later.setToolTip("Don't offer these images for this collage again")
        later.clicked.connect(self.dismiss_new_matches)
        row.addWidget(later)
        self.banner.hide()
        return self.banner

    def set_suggestion_data(self, data):
        """SuggestionWorker results: what each suggested collage's rule matches."""
        if data.get("error"):
            return
        self._suggestion_data = data
        self._suggestions_by_key = {s.key: s for s in data["suggestions"]}
        self._update_banner()

    def _start_suggestion_data(self):
        if not self.library:
            return
        worker = SuggestionWorker(self.library)
        worker.ready.connect(self.set_suggestion_data)
        keep_alive_until_finished(worker)
        worker.start()

    def _object_of(self, image_id):
        row = self.library.get(image_id)
        return (str(row["dsodetailid"]) if row["dsodetailid"] else f"image:{image_id}") if row else None

    def new_matches(self, collage):
        """Image ids matching collage's rule that it doesn't show and hasn't
        been offered before (rule["seen"])."""
        rule = collage.rule if collage is not None else None
        suggestion = self._suggestions_by_key.get(rule.get("key")) if rule else None
        if suggestion is None:
            return []
        if "seen" not in rule:  # Made before new-match offers: start from what matches now
            rule["seen"] = list(suggestion.image_ids)
            self._dirty[id(collage)] = collage
            self._save_timer.start()
            return []
        seen = set(rule["seen"])
        files = {self._file_key(f.image_path) for f in collage.frames if f.is_image}
        shown = {self._object_of(f.image_id) for f in collage.frames if f.is_image}
        shown |= {dso for f in collage.frames for dso in f.merged_dsos()}
        new = []
        for image_id in suggestion.image_ids:
            row = self.library.get(image_id)
            if image_id in seen or row is None:
                continue
            if rule["key"] == "all":
                if self._file_key(row["image_path"]) not in files:
                    new.append(image_id)
            elif self._object_of(image_id) not in shown:
                new.append(image_id)
        return new

    def _update_banner(self):
        collage = self.current
        new = self.new_matches(collage)
        self.banner.setVisible(bool(new))
        if not new:
            return
        suggestion = self._suggestions_by_key[collage.rule["key"]]
        names = sorted({self.library[i]["dso_name"] for i in new}, key=natural_key)
        listed = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
        count = len(new)
        self.banner_label.setText(f"<b>{count} new image{'s' if count != 1 else ''}</b> "
                                  f"match{'es' if count == 1 else ''} “{suggestion.title}”: {listed}")
        tiles = collage.rule.get("placeholders")
        self.banner_add.setText("Fill Their Tiles" if tiles else "Add and Re-arrange")
        self.banner_add.setToolTip("Put them in their catalog tiles" if tiles else
                                   "Add them and lay the collage out again as it was suggested (Undo reverts it)")

    def add_new_matches(self):
        collage = self.current
        new = self.new_matches(collage)
        if not new:
            return
        before = collage.state()
        rule = collage.rule
        rule["seen"] = list(rule.get("seen", [])) + new
        slots = {f.slot_dso: f for f in collage.frames if f.is_placeholder and f.slot_dso}
        added = []
        for image_id in new:
            row = self.library[image_id]
            slot = slots.pop(str(row["dsodetailid"]), None)
            if slot is not None:
                slot.image_id, slot.image_path, slot.dso_name = image_id, row["image_path"], row["dso_name"]
            else:
                frame = Frame(image_id=image_id, image_path=row["image_path"], dso_name=row["dso_name"])
                collage.frames.append(frame)
                added.append(frame)
        if added:  # Lay it out again the way the suggestion did
            existing = reading_order([f for f in collage.frames if not f.is_text and f not in added])
            gap = round(min(collage.width, collage.height) * 0.012)
            area = collage.canvas_rect().adjusted(gap, gap, -gap, -gap)
            if rule["key"].startswith("catalog:"):  # Catalog order
                rank = {image_id: index for index, image_id in enumerate(
                    self._suggestions_by_key[rule["key"]].image_ids)}
                order = sorted(existing + added, key=lambda f: rank.get(f.image_id, len(rank)))
                mode = "grid"
            else:  # New images after the rest
                order, mode = existing + added, "mosaic"
            arrange_frames(order, mode, area, gap, self._aspects(order), order=order)
        self.commit("Add New Images", before)
        self._update_banner()

    def dismiss_new_matches(self):
        collage = self.current
        new = self.new_matches(collage)
        if new:
            before = collage.state()
            collage.rule["seen"] = list(collage.rule.get("seen", [])) + new
            self.commit("Not Now", before)
        self.banner.hide()

    # Text boxes

    def add_text(self):
        collage = self.current
        if collage is None:
            return
        self.view.end_crop(apply=True)
        before = collage.state()
        height = round(collage.height * 0.14)
        frame = Frame(kind="text", text="Title", x=round(collage.width * 0.2), y=round(collage.height * 0.04),
                      w=round(collage.width * 0.6), h=height, font_size=round(collage.height * 0.07),
                      text_color=contrast_text(collage.background))
        collage.frames.append(frame)
        self.commit("Add Text", before)
        self.edit_text(frame.uid)

    def edit_text(self, uid):
        """Select a text box and put the cursor in its text, in the panel."""
        self._select_uids({uid})
        self.text_edit.setFocus()
        self.text_edit.selectAll()

    def _on_text_changed(self):
        if self._updating_panels:
            return
        text = self.text_edit.toPlainText()
        uids = self._selected_uids()
        self._change_selected("Edit Text", lambda f: setattr(f, "text", text) if f.is_text else None,
                              merge_key=("text", frozenset(uids)))

    def _on_text_style_changed(self):
        if self._updating_panels:
            return
        family = self.font_combo.currentFont().family()
        size, bold, italic = self.font_size_spin.value(), self.bold_button.isChecked(), self.italic_button.isChecked()
        align, box = self.align_combo.currentData(), self.text_box_combo.currentData()

        def change(frame):
            if frame.is_text:
                frame.font_family, frame.font_size, frame.bold, frame.italic = family, size, bold, italic
                frame.align, frame.text_box = align, box

        self._change_selected("Text Style", change, merge_key=("text style", frozenset(self._selected_uids())))

    def _choose_text_color(self):
        texts = [item.frame for item in self.scene.selected_frame_items() if item.frame.is_text]
        if not texts:
            return
        color = QColorDialog.getColor(QColor(texts[0].text_color), self, "Text Color")
        if color.isValid():
            self._change_selected("Text Color", lambda f: setattr(f, "text_color", color.name()) if f.is_text else None)

    # Style and rotation

    def _on_style_changed(self, which):
        if self._updating_panels:
            return
        value = {"border": self.border_spin.value(), "corners": self.corner_spin.value(),
                 "shadow": self.shadow_slider.value() / 100}[which]
        attribute = {"border": "border_width", "corners": "corner_radius", "shadow": "shadow"}[which]

        def change(frame):
            if frame.is_image or (frame.is_text and which == "shadow"):
                setattr(frame, attribute, value)

        self._change_selected("Style", change, merge_key=(which, frozenset(self._selected_uids())))

    def _choose_border_color(self):
        images = [item.frame for item in self.scene.selected_frame_items() if item.frame.is_image]
        if not images:
            return
        color = QColorDialog.getColor(QColor(images[0].border_color), self, "Border Color")
        if color.isValid():
            def change(frame):
                if frame.is_image:
                    frame.border_color = color.name()
                    if not frame.border_width:  # Picking a color means a border
                        frame.border_width = max(2, round(min(frame.w, frame.h) * 0.01))
            self._change_selected("Border Color", change)

    def _on_rotation_changed(self, value):
        uids = self._selected_uids()
        if len(uids) == 1:
            self._change_selected("Rotate", lambda f: setattr(f, "rotation", round(value, 1)),
                                  merge_key=("rotation", frozenset(uids)))

    def _on_gradient_changed(self, *_args):
        collage = self.current
        if collage is None or self._updating_panels:
            return
        before = collage.state()
        if not self.gradient_check.isChecked():
            collage.background2 = None
        elif not collage.background2:  # Start from a lighter (or, on a light background, darker) shade
            first = QColor(collage.background)
            collage.background2 = (first.lighter(300) if first.lightness() < 128 else first.darker(200)).name()
            if collage.background2 == first.name():  # Pure black doesn't lighten
                collage.background2 = "#1c2a4a"
        collage.gradient = self.gradient_combo.currentData() or "vertical"
        self.commit("Background Gradient", before)

    # Arrange

    def _aspects(self, frames):
        """{uid: width / height} of the frames' images (1 if unknown)."""
        store = preview_store()
        return {f.uid: (store.aspect(f.image_path) or 1.0) if f.is_image else 1.0 for f in frames}

    def open_arrange(self):
        """Arrange the selected images (2 or more) within their area, or all of them on the canvas."""
        collage = self.current
        if collage is None:
            return
        self.view.end_crop(apply=True)
        selected = [item.frame for item in self.scene.selected_frame_items() if not item.frame.is_text]
        subset = len(selected) >= 2
        frames = selected if subset else [f for f in collage.frames if not f.is_text]
        if not frames:
            return
        uids = {f.uid for f in frames}
        area = None
        if subset:
            items = [item for item in self.scene.selected_frame_items() if item.frame.uid in uids]
            area = items[0].visual_rect()
            for item in items[1:]:
                area = area.united(item.visual_rect())
        before = collage.state()
        aspects = self._aspects(frames)
        dialog = ArrangeDialog(self, frames, subset, collage)

        def preview(params):
            collage.restore(before)
            chosen = [f for f in collage.frames if f.uid in uids]
            margin = params["margin"]
            target = area if area is not None else collage.canvas_rect().adjusted(margin, margin, -margin, -margin)
            arrange_frames(chosen, params["mode"], target, params["gap"], aspects, params["columns"],
                           params["fill"], params["feature"])
            self._refresh_auto_labels(collage)
            self.scene.sync()

        dialog.changed.connect(preview)
        preview(dialog.params())
        if dialog.exec() == QDialog.Accepted:
            self.commit("Arrange", before)
            self._select_uids(uids if subset else set())
        else:  # Put everything back as it was
            collage.restore(before)
            self._refresh_auto_labels(collage)
            self.scene.sync()

    def _on_theme_changed(self):
        self.scene.update()
        self._refresh_library_icons()
        self.library_list.viewport().update()

    # Library

    def _load_library(self):
        try:
            with DatabaseManager().get_connection() as conn:
                rows = load_library(conn)
        except Exception as e:
            logger.error(f"Collage Builder: could not load images: {e}", exc_info=True)
            rows = []
        self.library = {row["id"]: row for row in rows}
        self._rebuild_library_items()

    @staticmethod
    def _file_key(path):
        return os.path.normcase(os.path.normpath(path)) if path else ""

    def _rebuild_library_items(self):
        """One library item per image file: entries sharing a file (one image
        of M 31 and M 32) show once, named "M 31 & M 32"."""
        self._common_by_dso = {str(row["dsodetailid"]): row.get("common_name") for row in self.library.values()
                               if row["dsodetailid"] and row.get("common_name")}
        groups = {}
        for row in sorted(self.library.values(), key=lambda r: (natural_key(r["dso_name"]), r["id"])):
            groups.setdefault(self._file_key(row["image_path"]) or f"id:{row['id']}", []).append(row["id"])
        self._groups = groups
        self._group_of = {image_id: ids for ids in groups.values() for image_id in ids}
        self.library_list.clear()
        self._library_items = {}
        for ids in groups.values():
            item = QListWidgetItem(join_names(self.library[i]["dso_name"] for i in ids))
            item.setData(Qt.UserRole, ids[0])
            item.setData(GROUP_ROLE, ids)
            self.library_list.addItem(item)
            self._library_items[ids[0]] = item
        self._refresh_library_icons()
        self._filter_library(self.library_search.text())

    def _library_tooltip(self, image_id):
        ids = self._group_of.get(image_id, [image_id])
        row = self.library[image_id]
        details = [join_names(self.library[i]["dso_name"] for i in ids), row["image_path"]]
        for label, key in (("Date", "date_taken"), ("Equipment", "equipment"), ("Integration", "integration_time")):
            if row.get(key):
                details.append(f"{label}: {row[key]}")
        if preview_store().error(row["image_path"]) == MISSING_FILE:
            details.append("\nFile not found - double-click to locate it")
        return "\n".join(details)

    def _is_missing(self, image_id):
        row = self.library.get(image_id)
        return row is not None and preview_store().error(row["image_path"]) == MISSING_FILE

    def _on_library_double_click(self, item):
        image_id = item.data(Qt.UserRole)
        if self._is_missing(image_id):
            self.locate_image(image_id)
        else:
            self.add_images([image_id])

    def _library_menu(self, pos):
        item = self.library_list.itemAt(pos)
        if item is None:
            return
        image_id = item.data(Qt.UserRole)
        ids = [i.data(Qt.UserRole) for i in self.library_list.selectedItems()] or [image_id]
        menu = QMenu(self)
        add = menu.addAction("Add to Collage" if len(ids) == 1 else f"Add {len(ids)} Images to Collage",
                             lambda: self.add_images(ids))
        add.setEnabled(self.current is not None)
        if self._is_missing(image_id):
            menu.addAction("Locate Image…", lambda: self.locate_image(image_id))
        menu.addSeparator()
        self._add_view_actions(menu, image_id)
        menu.exec(self.library_list.viewport().mapToGlobal(pos))

    def _refresh_library_icons(self):
        for image_id in self._library_items:
            self._update_library_item(image_id)

    def _update_library_item(self, image_id):
        store = preview_store()
        row, item = self.library[image_id], self._library_items[image_id]
        path = row["image_path"]
        thumb = store.thumbnail(path)
        error = store.error(path)
        if thumb is not None:
            item.setIcon(QIcon(thumb))
        elif error:
            item.setIcon(_placeholder_icon(MISSING_FILE if error == MISSING_FILE else "Can't read"))
        else:
            item.setIcon(_placeholder_icon(""))  # Loading
        item.setToolTip(self._library_tooltip(image_id))

    def _on_preview_ready(self, path):
        for image_id in self._library_items:
            if self.library[image_id]["image_path"] == path:
                self._update_library_item(image_id)
        if self.current is not None and any(frame.image_path == path for frame in self.current.frames):
            self._update_missing_label()

    def _missing_in_collage(self):
        """Image ids in the current collage whose files can't be found."""
        if self.current is None:
            return []
        ids = []
        for frame in self.current.frames:
            if (frame.is_image and frame.image_id not in ids
                    and preview_store().error(frame.image_path) == MISSING_FILE):
                ids.append(frame.image_id)
        return ids

    def _update_missing_label(self):
        missing = self._missing_in_collage()
        if missing:
            count = len(missing)
            self.missing_label.setText(f"⚠ {count} image{'s' if count != 1 else ''} can't be found — "
                                       f"<a href='locate'>Locate…</a>")
        self.missing_label.setVisible(bool(missing))

    def _locate_missing_in_collage(self):
        missing = self._missing_in_collage()
        if missing:
            self.locate_image(missing[0])

    def locate_image(self, image_id):
        """Ask where a missing image is now and update it everywhere in the app,
        along with any other missing images that moved the same way."""
        row = self.library.get(image_id)
        if row is None:
            return
        old_path = row["image_path"]
        settings = QSettings("CosmosCollection", "CosmosCollection")
        # Start in the nearest folder of the old location that still exists
        start = os.path.dirname(old_path)
        while start and not os.path.isdir(start) and os.path.dirname(start) != start:
            start = os.path.dirname(start)
        if not start or not os.path.isdir(start):
            start = settings.value("CollageBuilder/locate_dir", "", type=str) or os.path.expanduser("~")
        try:
            from DSOGallery import SUPPORTED_IMAGE_EXTENSIONS
            patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_EXTENSIONS))
            name_filter = f"Image Files ({patterns});;All Files (*)"
        except ImportError:
            name_filter = "All Files (*)"
        new_path, _ = QFileDialog.getOpenFileName(
            self, f"Locate {row['dso_name']} — {os.path.basename(old_path)}", start, name_filter)
        if not new_path:
            return
        new_path = os.path.normpath(new_path)
        settings.setValue("CollageBuilder/locate_dir", os.path.dirname(new_path))

        # Entries sharing the file (e.g. M 31 and M 32 in one image) move with it
        updates = {other["id"]: new_path for other in self.library.values() if other["image_path"] == old_path}
        updates[image_id] = new_path
        missing = [(other["id"], other["image_path"]) for other in self.library.values()
                   if other["id"] not in updates and not os.path.exists(other["image_path"])]
        moved = find_moved_images(old_path, new_path, missing)
        if moved:
            names = sorted((self.library[i]["dso_name"] for i in moved), key=natural_key)
            listed = "\n".join(f"  • {name}" for name in names[:12])
            if len(names) > 12:
                listed += f"\n  …and {len(names) - 12} more"
            count = len(moved)
            reply = QMessageBox.question(
                self, "Locate Images",
                f"{count} other missing image{'s were' if count != 1 else ' was'} found in the same new place:\n\n"
                f"{listed}\n\nUpdate {'them' if count != 1 else 'it'} too?")
            if reply == QMessageBox.Yes:
                updates.update(moved)

        try:
            with DatabaseManager().get_connection() as conn:
                conn.executemany("UPDATE userimages SET image_path = ? WHERE id = ?",
                                 [(path, i) for i, path in updates.items()])
                conn.commit()
        except Exception as e:
            logger.error(f"Could not update image locations: {e}", exc_info=True)
            QMessageBox.critical(self, "Locate Image", f"The new location couldn't be saved:\n{e}")
            return

        for i, path in updates.items():
            self.library[i]["image_path"] = path
        self._rebuild_library_items()  # Files may now be shared differently
        for collage in self.collages:
            for frame in collage.frames:
                if frame.image_id in updates:
                    frame.image_path = updates[frame.image_id]
            self._refresh_auto_labels(collage)
        self.scene.sync()
        self._refresh_panels()
        count = len(updates)
        self._set_status(f"Located {count} image{'s' if count != 1 else ''} ✓")

    def _filter_library(self, text):
        words = text.lower().split()
        for image_id, item in self._library_items.items():
            row = self.library[image_id]
            names = " ".join(self.library[i]["dso_name"] for i in self._group_of.get(image_id, [image_id]))
            haystack = " ".join((names, os.path.basename(row["image_path"]),
                                 str(row.get("equipment") or ""), str(row.get("date_taken") or ""))).lower()
            item.setHidden(not all(word in haystack for word in words))

    # Collages

    def _load_collages(self):
        try:
            with DatabaseManager().get_connection() as conn:
                self.collages = load_collages(conn, self.dsodetailid)
        except Exception as e:
            logger.error(f"Collage Builder: could not load collages: {e}", exc_info=True)
            QMessageBox.warning(self, "Collage Builder", f"Could not load your collages:\n{e}")
            self.collages = []
        for collage in self.collages:
            self._refresh_auto_labels(collage)
        self.collage_list.blockSignals(True)
        self.collage_list.clear()
        for collage in self.collages:
            self.collage_list.addItem(collage.name)
        self.collage_list.blockSignals(False)
        if self.collages:
            self.collage_list.setCurrentRow(0)
        else:
            self._switch_to(None)

    def _stack_for(self, collage):
        stack = self._stacks.get(id(collage))
        if stack is None:
            stack = QUndoStack(self)
            stack.setUndoLimit(200)
            self.undo_group.addStack(stack)
            self._stacks[id(collage)] = stack
        return stack

    def _on_collage_row_changed(self, row):
        self._switch_to(self.collages[row] if 0 <= row < len(self.collages) else None)

    def _switch_to(self, collage):
        self.view.end_crop(apply=True)
        if self.view._mode in ("move", "resize", "rotate"):
            self.view.cancel_drag()
        self._save_dirty()
        self.current = collage
        self.scene.set_collage(collage)
        if collage is None:
            self.center.setCurrentIndex(0)
            self.undo_group.setActiveStack(None)
        else:
            self.center.setCurrentIndex(1)
            self.undo_group.setActiveStack(self._stack_for(collage))
            QTimer.singleShot(0, self.view.fit)  # Once the view has its size
        self._refresh_panels()
        self._update_banner()

    def _ask_name(self, title, default, renaming=None):
        name = default
        while True:
            name, ok = QInputDialog.getText(self, title, "Collage name:", text=name)
            name = name.strip()
            if not ok or not name:
                return None
            if any(c is not renaming and c.name.lower() == name.lower() for c in self.collages):
                QMessageBox.warning(self, title, f"There's already a collage called “{name}”.")
                continue
            return name

    def _unique_name(self, base):
        names = {c.name.lower() for c in self.collages}
        if base.lower() not in names:
            return base
        for n in itertools.count(2):
            candidate = f"{base} {n}"
            if candidate.lower() not in names:
                return candidate

    def _add_collage(self, collage):
        self._dirty[id(collage)] = collage
        if not self._save_dirty():
            return
        self.collages.insert(0, collage)
        self.collage_list.insertItem(0, collage.name)
        self.collage_list.setCurrentRow(0)

    def _new_collage(self):
        """Start blank or from a suggestion (StartFromDialog)."""
        dialog = StartFromDialog(self, self._unique_name("Collage"))
        if dialog.exec() != QDialog.Accepted:
            return
        name, suggestion, size, placeholders, style = dialog.choice()
        members = dialog.members.get(suggestion.catalog) if suggestion is not None and suggestion.catalog else None
        self.create_collage(name, suggestion, size, members, placeholders, style)

    def create_collage(self, name, suggestion=None, size=DEFAULT_CANVAS, members=None, placeholders=False,
                       style="number"):
        """Add a collage, filled from a suggestion if there is one."""
        collage = Collage(name=name, dsodetailid=self.dsodetailid, width=size[0], height=size[1],
                          placeholder_style=style)
        if suggestion is not None:
            store = preview_store()
            aspects = {row["image_path"]: store.aspect(row["image_path"]) for row in self.library.values()}
            build_suggested_collage(collage, suggestion, self.library, aspects, members, placeholders)
        self._refresh_auto_labels(collage)
        self._add_collage(collage)
        return collage

    def _duplicate_collage(self):
        if self.current is None:
            return
        name = self._ask_name("Duplicate Collage", self._unique_name(f"{self.current.name} copy"))
        if not name:
            return
        duplicate = Collage(name=name, dsodetailid=self.current.dsodetailid)
        duplicate.restore(self.current.state())
        for frame in duplicate.frames:
            frame.uid = next(_frame_ids)
        self._add_collage(duplicate)

    def _rename_collage(self):
        collage = self.current
        if collage is None:
            return
        name = self._ask_name("Rename Collage", collage.name, renaming=collage)
        if not name or name == collage.name:
            return
        collage.name = name
        self._dirty[id(collage)] = collage
        self._save_dirty()
        self.collage_list.item(self.collages.index(collage)).setText(name)

    def _delete_collage(self):
        collage = self.current
        if collage is None:
            return
        reply = QMessageBox.question(
            self, "Delete Collage",
            f"Delete the collage “{collage.name}”?\n\nYour images aren't affected. This can't be undone.")
        if reply != QMessageBox.Yes:
            return
        self._dirty.pop(id(collage), None)
        if collage.db_id is not None:
            try:
                with DatabaseManager().get_connection() as conn:
                    delete_collage(conn, collage.db_id)
            except Exception as e:
                logger.error(f"Could not delete collage: {e}", exc_info=True)
                QMessageBox.critical(self, "Delete Collage", f"Could not delete the collage:\n{e}")
                return
        stack = self._stacks.pop(id(collage), None)
        if stack is not None:
            self.undo_group.removeStack(stack)
            stack.deleteLater()
        row = self.collages.index(collage)
        self.collages.pop(row)
        self.collage_list.blockSignals(True)
        self.collage_list.takeItem(row)
        self.collage_list.blockSignals(False)
        self.current = None  # Already saved/deleted - nothing to flush
        if self.collages:
            self.collage_list.setCurrentRow(min(row, len(self.collages) - 1))
            self._on_collage_row_changed(self.collage_list.currentRow())
        else:
            self._switch_to(None)

    # Editing - every change goes through commit() so it can be undone and is saved

    def commit(self, text, before, merge_key=None):
        collage = self.current
        if collage is None or before is None:
            return
        after = collage.state()
        if after == before:
            self.scene.sync()  # Nothing changed - put any dragged items back in step
            return
        self._stack_for(collage).push(SnapshotCommand(self, collage, text, before, after, merge_key))

    def apply_state(self, collage, state):
        """Show a collage state (an edit, undo or redo) and schedule saving it."""
        old_size = (collage.width, collage.height)
        collage.restore(state)
        for frame in collage.frames:  # Keep images located since this state was saved
            row = self.library.get(frame.image_id)
            if row is not None:
                frame.image_path = row["image_path"]
        self._refresh_auto_labels(collage)
        if collage is self.current:
            self.scene.sync()
            if (collage.width, collage.height) != old_size and self.view.fit_mode:
                self.view.fit()
            self._refresh_panels()
        self._dirty[id(collage)] = collage
        self._set_status("Saving…")
        self._save_timer.start()

    def _frames_by_uid(self):
        return {frame.uid: frame for frame in self.current.frames}

    def _selected_uids(self):
        return {item.frame.uid for item in self.scene.selected_frame_items()}

    def _select_uids(self, uids):
        self.scene.clearSelection()
        for item in self.scene.frame_items():
            if item.frame.uid in uids:
                item.setSelected(True)

    def add_images(self, image_ids, at=None):
        """Add library images as new frames: centered on `at` (canvas coordinates,
        side by side for several), else in the first free spots."""
        collage = self.current
        if collage is None:
            return
        rows = [self.library[i] for i in image_ids if i in self.library]
        if not rows:
            return
        before = collage.state()
        store = preview_store()
        gap = min(collage.width, collage.height) * 0.02
        occupied = [frame.rect() for frame in collage.frames]
        added = []
        cursor = None
        row_height = 0
        empty_slots = {frame.slot_dso: frame for frame in collage.frames if frame.is_placeholder and frame.slot_dso}
        for row in rows:
            slot = empty_slots.pop(str(row["dsodetailid"]), None) if at is None else None
            if slot is not None:  # A catalog collage's tile for this object: fill it
                slot.image_id, slot.image_path, slot.dso_name = row["id"], row["image_path"], row["dso_name"]
                added.append(slot.uid)
                continue
            w, h = default_frame_size(collage, store.aspect(row["image_path"]))
            if at is not None:
                if cursor is None:
                    cursor = QPointF(at.x() - w / 2, at.y() - h / 2)
                elif cursor.x() + w > collage.width and cursor.x() > at.x() - w / 2:
                    cursor = QPointF(at.x() - w / 2, cursor.y() + row_height + gap)  # Next row
                    row_height = 0
                spot = QPointF(round(cursor.x()), round(cursor.y()))
                cursor = QPointF(cursor.x() + w + gap, cursor.y())
                row_height = max(row_height, h)
            else:
                spot = find_free_spot(collage, w, h, occupied)
                if spot is None:  # Canvas full - stagger from the center
                    step = gap * 2 * (len(added) % 10)
                    spot = QPointF(round(collage.width / 2 - w / 2 + step), round(collage.height / 2 - h / 2 + step))
            frame = Frame(image_id=row["id"], image_path=row["image_path"], dso_name=row["dso_name"],
                          x=spot.x(), y=spot.y(), w=w, h=h)
            collage.frames.append(frame)
            occupied.append(frame.rect())
            added.append(frame.uid)
        self.commit("Add Image" if len(added) == 1 else f"Add {len(added)} Images", before)
        self._select_uids(set(added))
        self.view.setFocus()

    def replace_image(self, uid, image_id):
        collage = self.current
        row = self.library.get(image_id)
        frame = self._frames_by_uid().get(uid) if collage else None
        if frame is None or row is None or frame.image_id == image_id:
            return
        before = collage.state()
        if frame.is_text:
            return
        filling = frame.is_placeholder
        frame.image_id, frame.image_path, frame.dso_name = row["id"], row["image_path"], row["dso_name"]
        frame.crop_x = frame.crop_y = 0.5
        frame.zoom = 1.0
        frame.label = None  # A custom label was about the old image
        self.commit("Fill Placeholder" if filling else "Replace Image", before)
        self._select_uids({uid})

    def remove_selected(self):
        """Remove the selected images. In a catalog collage an image's tile
        stays, as a placeholder again; removing a placeholder removes the tile."""
        uids = self._selected_uids()
        if self.current is None or not uids:
            return
        before = self.current.state()
        kept = []
        for frame in self.current.frames:
            if frame.uid not in uids:
                kept.append(frame)
            elif frame.slot and not frame.is_placeholder:
                frame.clear_image()
                kept.append(frame)
        self.current.frames = kept
        self.commit("Remove Image" if len(uids) == 1 else f"Remove {len(uids)} Images", before)

    def nudge(self, dx, dy):
        uids = self._selected_uids()
        if self.current is None or not uids:
            return
        before = self.current.state()
        for frame in self.current.frames:
            if frame.uid in uids:
                frame.x += dx
                frame.y += dy
        self.commit("Nudge", before, merge_key=("nudge", frozenset(uids)))

    def restack(self, to_front=True):
        uids = self._selected_uids()
        if self.current is None or not uids:
            return
        before = self.current.state()
        chosen = [frame for frame in self.current.frames if frame.uid in uids]
        others = [frame for frame in self.current.frames if frame.uid not in uids]
        self.current.frames = others + chosen if to_front else chosen + others
        self.commit("Bring to Front" if to_front else "Send to Back", before)

    def set_fit(self, mode):
        uids = self._selected_uids()
        if self.current is None or not uids:
            return
        before = self.current.state()
        for frame in self.current.frames:
            if frame.uid in uids:
                frame.fit = mode
        self.commit("Fill Frame" if mode == FIT_FILL else "Show Whole Image", before)

    def _on_canvas_size_changed(self):
        collage = self.current
        if collage is None or self._updating_panels:
            return
        before = collage.state()
        collage.width, collage.height = self.width_spin.value(), self.height_spin.value()
        self.commit("Canvas Size", before, merge_key="canvas_size")

    def _choose_background(self, second=False):
        collage = self.current
        if collage is None:
            return
        current = collage.background2 if second and collage.background2 else collage.background
        color = QColorDialog.getColor(QColor(current), self,
                                      "Gradient Color" if second else "Canvas Background")
        if color.isValid():
            before = collage.state()
            if second:
                collage.background2 = color.name()
            else:
                collage.background = color.name()
            self.commit("Background Color", before)

    def _on_labels_changed(self):
        collage = self.current
        if collage is None or self._updating_panels:
            return
        before = collage.state()
        collage.show_labels = self.labels_check.isChecked()
        collage.label_position = self.label_position_combo.currentText()
        collage.label_content = self.label_content_combo.currentData() or "number"
        self.commit("Labels", before)

    # Panels

    def _refresh_panels(self):
        collage = self.current
        self._updating_panels = True
        try:
            has_collage = collage is not None
            for widget in (self.preset_combo, self.orientation_button, self.width_spin, self.height_spin,
                           self.background_button, self.labels_check, self.label_position_combo, self.export_button,
                           self.gradient_check, self.text_button, self.arrange_button, self.align_button):
                widget.setEnabled(has_collage)
            if has_collage:
                index = preset_index(collage.width, collage.height)
                self.preset_combo.setCurrentIndex(index if index >= 0 else self.preset_combo.count() - 1)
                self.width_spin.setValue(collage.width)
                self.height_spin.setValue(collage.height)
                self._color_button(self.background_button, collage.background)
                gradient = bool(collage.background2)
                self.gradient_check.setChecked(gradient)
                self.background2_button.setEnabled(gradient)
                self.gradient_combo.setEnabled(gradient)
                if gradient:
                    self._color_button(self.background2_button, collage.background2)
                else:
                    self.background2_button.setText("")
                    self.background2_button.setStyleSheet("")
                self.gradient_combo.setCurrentIndex(max(0, self.gradient_combo.findData(collage.gradient)))
                self.labels_check.setChecked(collage.show_labels)
                self.label_position_combo.setCurrentText(collage.label_position)
                self.label_position_combo.setEnabled(collage.show_labels)
                self.label_content_combo.setCurrentIndex(max(0, self.label_content_combo.findData(collage.label_content)))
                self.label_content_combo.setEnabled(collage.show_labels)
                frames = sum(1 for frame in collage.frames if frame.is_image)
                self.canvas_info.setText(f"{frames} image{'s' if frames != 1 else ''}")
                self.export_button.setEnabled(bool(collage.frames))
                slots = [frame for frame in collage.frames if frame.slot]
                self.slots_box.setVisible(bool(slots))
                if slots:
                    filled = sum(1 for frame in slots if frame.is_image)
                    self.slots_info.setText(f"{filled} of {len(slots)} imaged. Drop an image on a numbered "
                                            f"tile to fill it.")
                    self.slot_style_combo.setCurrentIndex(self.slot_style_combo.findData(collage.placeholder_style))
            else:
                self.canvas_info.setText("")
                self.background_button.setText("")
                self.background_button.setStyleSheet("")
                self.slots_box.hide()
        finally:
            self._updating_panels = False
        self.used_image_ids = collage.image_ids() if collage else set()
        self.library_list.viewport().update()
        self._update_missing_label()
        self._on_selection_changed()

    def _on_selection_changed(self):
        """Show the canvas settings, or the selected images' settings."""
        selected = self.scene.selected_frame_items() if self.current else []
        for widget in (self.front_button, self.back_button, self.remove_button):
            widget.setEnabled(bool(selected))
        self.properties.setCurrentIndex(1 if selected else 0)
        self.view.viewport().update()
        if not selected:
            return
        frames = [item.frame for item in selected]
        images = [frame for frame in frames if frame.is_image]
        single = frames[0] if len(frames) == 1 else None
        self._updating_panels = True
        try:
            if single is not None and single.is_placeholder:
                name = f"<br>{single.slot_name}" if single.slot_name else ""
                self.selection_label.setText(f"<b>{single.slot}</b>{name}<br>Not imaged yet - drop an image "
                                             f"from the library on it.")
            elif single is not None and single.is_text:
                self.selection_label.setText("<b>Text box</b>")
            elif single is not None:
                self.selection_label.setText(f"<b>{single.label_text()}</b><br>{os.path.basename(single.image_path)}")
            else:
                self.selection_label.setText(f"<b>{len(frames)} items selected</b>")

            self.geometry_box.setVisible(single is not None)
            if single is not None:
                for spin, value in ((self.x_spin, single.x), (self.y_spin, single.y),
                                    (self.w_spin, single.w), (self.h_spin, single.h)):
                    spin.setValue(round(value))
                self.rotation_spin.setValue(single.rotation)

            texts = [frame for frame in frames if frame.is_text]
            self.text_group.setVisible(bool(texts))
            if texts:
                first = texts[0]
                self.text_edit.setEnabled(len(texts) == 1)
                if self.text_edit.toPlainText() != first.text:  # Not while typing - keeps the cursor
                    self.text_edit.setPlainText(first.text)
                self.font_combo.setCurrentFont(text_font(first))
                self.font_size_spin.setValue(round(first.font_size))
                self.bold_button.setChecked(bool(first.bold))
                self.italic_button.setChecked(bool(first.italic))
                self._color_button(self.text_color_button, first.text_color)
                self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(first.align)))
                self.text_box_combo.setCurrentIndex(max(0, self.text_box_combo.findData(first.text_box)))

            styled = images + texts
            self.style_group.setVisible(bool(styled))
            if styled:
                first = images[0] if images else texts[0]
                for widget in (self.border_label, self.border_spin, self.border_color_button,
                               self.corner_label, self.corner_spin):
                    widget.setVisible(bool(images))  # Text boxes have a shadow only
                self.border_spin.setValue(round(first.border_width))
                self._color_button(self.border_color_button, first.border_color)
                self.corner_spin.setValue(round(first.corner_radius))
                self.shadow_slider.setValue(round(first.shadow * 100))

            self.image_box.setVisible(bool(images))
            self.label_box.setVisible(bool(images))
            if images:
                fits = {frame.fit for frame in images}
                self.fit_combo.setCurrentIndex(self.fit_combo.findData(fits.pop()) if len(fits) == 1 else -1)
                fills = [frame for frame in images if frame.fit == FIT_FILL]
                self.zoom_slider.setEnabled(bool(fills))
                self.zoom_slider.setValue(round((fills[0].zoom if fills else 1.0) * 100))
                one_image = images[0] if len(images) == 1 else None
                self.crop_button.setEnabled(one_image is not None and preview_store().preview(one_image.image_path) is not None)
                self.reset_crop_button.setEnabled(any(f.zoom != 1.0 or (f.crop_x, f.crop_y) != (0.5, 0.5) for f in fills))

                visible = {frame.label_visible for frame in images}
                self.frame_label_show.setCurrentIndex(combo_index(self.frame_label_show, visible.pop()) if len(visible) == 1 else -1)
                positions = {frame.label_position for frame in images}
                self.frame_label_position.setCurrentIndex(
                    combo_index(self.frame_label_position, positions.pop()) if len(positions) == 1 else -1)
                contents = {frame.label_content for frame in images}
                self.frame_label_content.setCurrentIndex(
                    combo_index(self.frame_label_content, contents.pop()) if len(contents) == 1 else -1)
                names = {frame.common_name for frame in images}
                self.frame_label_content.setToolTip(
                    f"Common name: {names.pop() or 'none in the catalog'}" if len(names) == 1 else "")
                self.frame_label_text.setEnabled(one_image is not None)
                self.frame_label_text.setText(one_image.label or "" if one_image is not None else "")
                self.frame_label_text.setPlaceholderText(one_image.dso_name if one_image is not None else "")

            choices = self.merge_choices()
            merged = single.merged_names() if single is not None else []
            self.objects_box.setVisible(bool(choices or merged))
            self.merge_button.setVisible(bool(choices))
            self.merge_button.setText("Merge into One Frame" if len(choices) == 1 else "Merge into One Frame…")
            self.unmerge_button.setVisible(bool(merged))
            if merged:
                self.objects_label.setText(f"Also shows {join_names(merged)}.")
            else:
                self.objects_label.setText("Objects that fit in one image (like M 31 and M 32) can share "
                                           "a frame, labeled with both names.")
        finally:
            self._updating_panels = False

    # Selection properties

    def _change_selected(self, text, change, merge_key=None):
        """Apply change(frame) to the selected frames as one undo step."""
        uids = self._selected_uids()
        if self.current is None or not uids or self._updating_panels:
            return
        before = self.current.state()
        for frame in self.current.frames:
            if frame.uid in uids:
                change(frame)
        self.commit(text, before, merge_key=merge_key)

    def _on_geometry_changed(self):
        uids = self._selected_uids()
        if len(uids) != 1:
            return
        values = (self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value())
        self._change_selected("Position and Size", lambda f: f.set_rect(QRectF(*values)),
                              merge_key=("geometry", frozenset(uids)))

    def _on_zoom_slider(self, value):
        uids = self._selected_uids()

        def zoom(frame):
            if frame.fit == FIT_FILL and frame.is_image:
                frame.zoom = value / 100

        self._change_selected("Zoom", zoom, merge_key=("zoom", frozenset(uids)))

    def reset_crop(self):
        def reset(frame):
            frame.crop_x = frame.crop_y = 0.5
            frame.zoom = 1.0

        self._change_selected("Reset Crop", reset)

    def _adjust_crop(self):
        selected = [item for item in self.scene.selected_frame_items() if item.frame.is_image]
        if len(selected) == 1:
            self.view.begin_crop(selected[0])

    def on_crop_mode(self, cropping):
        """Crop mode started (or the zoom changed) / ended."""
        if cropping:
            item = self.view.crop_item
            self.status_label.setText(f"Cropping — drag to move the image, scroll to zoom "
                                      f"({item.frame.zoom:.1f}×). Enter: done, Esc: cancel")
        else:
            self._set_status("All changes saved ✓" if not self._dirty else "Saving…")

    def _on_frame_label_changed(self):
        if self._updating_panels:
            return
        show_index, position_index = self.frame_label_show.currentIndex(), self.frame_label_position.currentIndex()
        content_index = self.frame_label_content.currentIndex()
        text = self.frame_label_text.text().strip()
        single = len(self._selected_uids()) == 1

        def change(frame):
            if not frame.is_image:
                return
            if show_index >= 0:
                frame.label_visible = self.frame_label_show.itemData(show_index)
            if position_index >= 0:
                frame.label_position = self.frame_label_position.itemData(position_index)
            if content_index >= 0:
                frame.label_content = self.frame_label_content.itemData(content_index)
            if single:
                frame.label = text if text and text != frame.dso_name else None

        self._change_selected("Label", change)

    def other_images(self, frame):
        """The user's other images of a frame's object (for swapping)."""
        row = self.library.get(frame.image_id)
        if row is None or not row["dsodetailid"]:
            return []
        key = self._file_key(row["image_path"])
        return [other for other in self.library.values()
                if other["dsodetailid"] == row["dsodetailid"] and self._file_key(other["image_path"]) != key]

    # Snapping (your preference, for every collage)

    SNAP_SETTING = "CollageBuilder/snapping"

    def _load_snap_options(self):
        options = {"objects": True, "spacing": True, "grid": False, "grid_size": 100}
        saved = _json_dict(QSettings("CosmosCollection", "CosmosCollection").value(self.SNAP_SETTING, "", type=str))
        options.update({key: value for key, value in saved.items() if key in options})
        return options

    def _on_snap_options_changed(self, *_args):
        self.snap_options = {"objects": self.snap_objects_check.isChecked(),
                             "spacing": self.snap_spacing_check.isChecked(),
                             "grid": self.snap_grid_check.isChecked(), "grid_size": self.grid_size_spin.value()}
        QSettings("CosmosCollection", "CosmosCollection").setValue(self.SNAP_SETTING, json.dumps(self.snap_options))
        self.grid_size_spin.setEnabled(self.snap_options["grid"])
        self.scene.grid_size = self.snap_options["grid_size"] if self.snap_options["grid"] else 0
        self.scene.update()

    # Align, distribute, size

    def fill_align_menu(self, menu):
        """Align / Distribute / Same Size entries for the selection (also the
        toolbar's Align menu). One image lines up with the canvas; several with
        each other."""
        menu.clear()
        count = len(self.scene.selected_frame_items()) if self.current else 0
        for key, label in ALIGNMENTS:
            action = menu.addAction(label + (" (to Canvas)" if count == 1 else ""),
                                    lambda checked=False, k=key: self.align_selected(k))
            action.setEnabled(count >= 1)
            if key == "right":
                menu.addSeparator()
        menu.addSeparator()
        center = menu.addAction("Center on the Canvas", self.center_selected)
        center.setEnabled(count >= 1)
        menu.addSeparator()
        for horizontal, label in ((True, "Distribute Horizontally"), (False, "Distribute Vertically")):
            action = menu.addAction(label, lambda checked=False, h=horizontal: self.distribute_selected(h))
            action.setEnabled(count >= 3)
            action.setToolTip("Equal gaps between three or more images")
        menu.addSeparator()
        for which, label in (("both", "Same Size"), ("width", "Same Width"), ("height", "Same Height")):
            action = menu.addAction(f"{label} (as the Largest)", lambda checked=False, w=which: self.same_size(w))
            action.setEnabled(count >= 2)
        menu.setToolTipsVisible(True)

    def _selection_rects(self):
        """[(frame uid, rect as seen on the canvas)] of the selected items."""
        return [(item.frame.uid, item.visual_rect()) for item in self.scene.selected_frame_items()]

    def _move_by(self, text, shifts):
        """Move frames {uid: (dx, dy)} as one undo step."""
        collage = self.current
        if collage is None or not any(dx or dy for dx, dy in shifts.values()):
            return
        before = collage.state()
        for frame in collage.frames:
            if frame.uid in shifts:
                dx, dy = shifts[frame.uid]
                frame.x, frame.y = round(frame.x + dx), round(frame.y + dy)
        self.commit(text, before)

    def align_selected(self, how):
        rects = self._selection_rects()
        if not rects:
            return
        if len(rects) == 1:
            reference = self.current.canvas_rect()
        else:
            reference = rects[0][1]
            for _uid, rect in rects[1:]:
                reference = reference.united(rect)
        self._move_by(dict(ALIGNMENTS)[how], {uid: align_shift(rect, reference, how) for uid, rect in rects})

    def center_selected(self):
        """Center the selection, as a group, on the canvas."""
        rects = self._selection_rects()
        if not rects:
            return
        group = rects[0][1]
        for _uid, rect in rects[1:]:
            group = group.united(rect)
        center = self.current.canvas_rect().center()
        dx, dy = center.x() - group.center().x(), center.y() - group.center().y()
        self._move_by("Center on Canvas", {uid: (dx, dy) for uid, _rect in rects})

    def distribute_selected(self, horizontal=True):
        rects = self._selection_rects()
        if len(rects) < 3:
            return
        shifts = distribute_shifts([rect for _uid, rect in rects], horizontal)
        self._move_by("Distribute", {uid: ((shift, 0) if horizontal else (0, shift))
                                     for (uid, _rect), shift in zip(rects, shifts)})

    def same_size(self, which="both"):
        """Give the selected frames the largest one's width and/or height (keeping their centers)."""
        collage = self.current
        uids = self._selected_uids()
        frames = [f for f in collage.frames if f.uid in uids] if collage else []
        if len(frames) < 2:
            return
        largest = max(frames, key=lambda f: f.w * f.h)
        before = collage.state()
        for frame in frames:
            center = frame.rect().center()
            if which in ("both", "width"):
                frame.w = largest.w
            if which in ("both", "height"):
                frame.h = largest.h
            frame.x, frame.y = round(center.x() - frame.w / 2), round(center.y() - frame.h / 2)
        self.commit("Same Size", before)

    def swap_frames(self, uid_a, uid_b, before):
        """Swap two frames' places (position, size, rotation) - Ctrl+drag."""
        collage = self.current
        frames = self._frames_by_uid()
        a, b = frames.get(uid_a), frames.get(uid_b)
        if a is None or b is None:
            return
        place = lambda f: (f.x, f.y, f.w, f.h, f.rotation)
        (a.x, a.y, a.w, a.h, a.rotation), (b.x, b.y, b.w, b.h, b.rotation) = place(b), place(a)
        self.commit("Swap", before)
        self._select_uids({uid_a})

    # Duplicate, copy / paste style

    def duplicate_selected(self):
        """Copies of the selected items, a little down and right, selected."""
        collage = self.current
        uids = self._selected_uids()
        if collage is None or not uids:
            return
        before = collage.state()
        offset = round(min(collage.width, collage.height) * 0.03)
        copies = []
        for frame in [f for f in collage.frames if f.uid in uids]:
            data = asdict(frame)
            data.pop("uid")
            copy_frame = Frame(**data)
            copy_frame.x, copy_frame.y = frame.x + offset, frame.y + offset
            # A copy is a free frame: not a catalog tile, nothing merged into it
            copy_frame.slot = copy_frame.slot_name = copy_frame.slot_dso = None
            copy_frame.merged = copy_frame.premerge_rect = None
            if copy_frame.is_placeholder:
                continue  # An empty tile has nothing to copy
            copies.append(copy_frame)
        if not copies:
            return
        collage.frames.extend(copies)
        self.commit("Duplicate", before)
        self._select_uids({f.uid for f in copies})

    STYLE_KEYS = ("border_width", "border_color", "corner_radius", "shadow", "fit", "label_visible",
                  "label_position", "label_content")
    TEXT_STYLE_KEYS = ("font_family", "font_size", "text_color", "bold", "italic", "align", "text_box", "shadow")

    def copy_style(self):
        selected = self.scene.selected_frame_items()
        if not selected:
            return
        frame = selected[-1].frame
        keys = self.TEXT_STYLE_KEYS if frame.is_text else self.STYLE_KEYS
        self.copied_style = {"text": frame.is_text, "values": {key: getattr(frame, key) for key in keys}}
        self.status_hint("Style copied - select images and use Paste Style (Ctrl+Shift+V)")

    def paste_style(self):
        style = self.copied_style
        if style is None:
            return

        def apply(frame):
            if frame.is_text == style["text"] and not frame.is_placeholder:
                for key, value in style["values"].items():
                    setattr(frame, key, value)

        self._change_selected("Paste Style", apply)

    def show_shortcuts(self):
        mod = "Cmd" if sys.platform == "darwin" else "Ctrl"
        rows = (("Select", "Click; " + mod + "+click to add; drag a box around several; Tab / Shift+Tab: next / previous"),
                ("Move", "Drag; arrow keys nudge 1 px (Shift: 10 px); Alt: no snapping"),
                ("Swap two images", f"Drag one onto the other, then hold {mod} while dropping"),
                ("Resize", "Drag a handle; Shift keeps the shape, Alt resizes from the center"),
                ("Rotate", "Drag the round knob above; Shift: 15° steps"),
                ("Crop", "Double-click an image; drag / scroll; Enter: done, Esc: cancel"),
                ("Zoom / pan", f"{mod}+wheel or pinch; {mod}+0 fit, {mod}+1 100%; Space+drag or middle button: pan"),
                ("Undo / redo", f"{mod}+Z / {mod}+Y"),
                ("Duplicate", f"{mod}+D"), ("Copy / paste style", f"{mod}+Shift+C / {mod}+Shift+V"),
                ("Front / back", f"{mod}+] / {mod}+["), ("Remove", "Delete"),
                ("Add text", "T"), ("Arrange", f"{mod}+L"), ("Export", f"{mod}+E"),
                ("Add images", "Drag from the library, or double-click one; drop on an image to replace it"))
        html = "<table cellspacing='6'>" + "".join(
            f"<tr><td><b>{name}</b></td><td>{keys}</td></tr>" for name, keys in rows) + "</table>"
        box = QMessageBox(QMessageBox.NoIcon, "Collage Builder Shortcuts", html, QMessageBox.Close, self)
        box.setTextFormat(Qt.RichText)
        box.exec()

    # Merging: several objects shown by one frame ("M 31 & M 32")

    def merge_choices(self):
        """[(menu text, uid of the frame to keep)] for merging the selection:
        one choice if the frames show one image file (or only one has an
        image), else one per image to keep. Empty if fewer than 2 are selected."""
        frames = [item.frame for item in self.scene.selected_frame_items()]
        if len(frames) < 2 or any(frame.is_text for frame in frames):
            return []
        by_file = {}
        for frame in sorted(frames, key=lambda f: -f.w * f.h):  # Largest frame of each file first
            if frame.is_image:
                by_file.setdefault(self._file_key(frame.image_path), frame)
        if len(by_file) <= 1:
            keep = next(iter(by_file.values()), frames[0])
            return [("Merge into One Frame", keep.uid)]
        choices = []
        for frame in by_file.values():
            row = self.library.get(frame.image_id)
            name = row["dso_name"] if row else frame.label_text()
            choices.append((f"Keep {name}'s Image ({os.path.basename(frame.image_path)})", frame.uid))
        return choices

    def _object_entry(self, frame):
        """What a frame merged into another one contributes: its names and
        objects (with anything already merged into it), and itself for Unmerge."""
        if frame.is_placeholder:
            names, dsos = [frame.slot or frame.dso_name], [frame.slot_dso]
        else:
            row = self.library.get(frame.image_id)
            names = [row["dso_name"] if row else frame.dso_name]
            dsos = [str(row["dsodetailid"]) if row and row["dsodetailid"] else None]
        return {"names": names + frame.merged_names(), "dsos": dsos + frame.merged_dsos(), "frame": asdict(frame)}

    def merge_selected(self, keep_uid=None):
        """Merge the selected frames into one (keep_uid's, or the first choice):
        the others are removed and their objects added to its label. If the
        frames sit side by side with nothing between, it grows to cover them."""
        collage = self.current
        choices = self.merge_choices()
        if collage is None or not choices:
            return
        keep_uid = keep_uid if keep_uid is not None else choices[0][1]
        uids = self._selected_uids()
        before = collage.state()
        frames = {frame.uid: frame for frame in collage.frames}
        keep = frames[keep_uid]
        others = [frame for frame in collage.frames if frame.uid in uids and frame is not keep]
        area = keep.rect()
        for frame in others:
            area = area.united(frame.rect())
        blocked = any(frame.rect().intersects(area.adjusted(1, 1, -1, -1))
                      for frame in collage.frames if frame.uid not in uids)
        if keep.premerge_rect is None:
            keep.premerge_rect = [keep.x, keep.y, keep.w, keep.h]
        keep.merged = list(keep.merged or []) + [self._object_entry(frame) for frame in others]
        if not blocked:
            keep.set_rect(area)
        collage.frames = [frame for frame in collage.frames if frame.uid not in uids or frame is keep]
        self.commit("Merge", before)
        self._select_uids({keep.uid})

    def unmerge(self, uid):
        """Put a merged frame's objects back in their own frames, where they were."""
        collage = self.current
        frame = self._frames_by_uid().get(uid) if collage else None
        if frame is None or not frame.merged:
            return
        before = collage.state()
        restored = []
        for entry in frame.merged:
            data = dict(entry["frame"])
            data.pop("uid", None)
            restored.append(Frame(**data))
        if frame.premerge_rect:
            frame.set_rect(QRectF(*frame.premerge_rect))
        frame.merged = None
        frame.premerge_rect = None
        index = collage.frames.index(frame)
        collage.frames[index + 1:index + 1] = restored
        self.commit("Unmerge", before)
        self._select_uids({frame.uid} | {f.uid for f in restored})

    def _merge_from_panel(self):
        choices = self.merge_choices()
        if len(choices) == 1:
            self.merge_selected(choices[0][1])
        elif choices:  # Which image to keep
            menu = QMenu(self)
            for text, uid in choices:
                menu.addAction(text, lambda checked=False, u=uid: self.merge_selected(u))
            menu.exec(self.merge_button.mapToGlobal(self.merge_button.rect().bottomLeft()))

    # Canvas properties

    def _resize_canvas(self, size, text):
        """New canvas size, with the layout scaled to fit it."""
        collage = self.current
        if collage is None or (collage.width, collage.height) == tuple(size):
            return
        before = collage.state()
        scale_layout(collage.frames, (collage.width, collage.height), size)
        collage.width, collage.height = size
        self.commit(text, before)

    def _on_preset_chosen(self, index):
        size = self.preset_combo.itemData(index)
        collage = self.current
        if collage is None or size is None:
            self._refresh_panels()  # "Custom size": type it in the boxes below
            return
        width, height = size
        if collage.height > collage.width and width > height:
            width, height = height, width  # Keep the orientation
        self._resize_canvas((width, height), "Canvas Size")

    def _swap_orientation(self):
        if self.current is not None:
            self._resize_canvas((self.current.height, self.current.width), "Swap Orientation")

    def _on_slot_style_changed(self, index):
        collage = self.current
        if collage is None:
            return
        before = collage.state()
        collage.placeholder_style = self.slot_style_combo.itemData(index)
        self.commit("Placeholder Style", before)

    # Labels for images shared by several objects

    def _refresh_auto_labels(self, collage):
        """Each frame's automatic label: its DSO, or every DSO sharing the image
        file ("M 31 & M 32") - except DSOs with a frame of their own in the
        collage, with any image of them (a catalog tile, another photo)."""
        def object_of(image_id):
            row = self.library.get(image_id)
            return (row["dsodetailid"] or f"image:{image_id}") if row else None

        def merged_commons(frame):
            return [self._common_by_dso.get(str(dso)) for dso in frame.merged_dsos()]

        for frame in collage.frames:
            if frame.is_text:
                continue
            if frame.is_placeholder:
                frame.dso_name = join_names([frame.slot or ""] + frame.merged_names())
                frame.common_name = join_names([frame.slot_name] + merged_commons(frame))
                continue
            mine = object_of(frame.image_id)
            elsewhere = set()
            for other in collage.frames:
                if other is frame:
                    continue
                if other.is_image:
                    elsewhere.add(object_of(other.image_id))
                if other.slot_dso:
                    elsewhere.add(other.slot_dso)
                elsewhere.update(other.merged_dsos())  # Shown merged into another frame
            elsewhere.discard(mine)
            group = self._group_of.get(frame.image_id, [frame.image_id])
            shown = [i for i in group if i in self.library and (i == frame.image_id or object_of(i) not in elsewhere)]
            names = [self.library[i]["dso_name"] for i in shown]
            frame.dso_name = join_names(names + frame.merged_names()) or frame.dso_name
            frame.common_name = join_names([self.library[i].get("common_name") for i in shown] + merged_commons(frame))

    # Saving

    def _save_dirty(self):
        """Save collages with unsaved changes now. False if that failed."""
        self._save_timer.stop()
        if not self._dirty:
            return True
        pending, self._dirty = self._dirty, {}
        try:
            with DatabaseManager().get_connection() as conn:
                for collage in pending.values():
                    save_collage(conn, collage)
        except Exception as e:
            logger.error(f"Collage Builder: save failed: {e}", exc_info=True)
            self._dirty.update(pending)  # Retried with the next change
            self._set_status(f"Not saved: {e}", error=True)
            return False
        self._set_status("All changes saved ✓")
        return True

    # Export

    def _export(self, scale=1.0):
        collage = self.current
        if collage is None or not collage.frames:
            return
        self._save_dirty()
        settings = QSettings("CosmosCollection", "CosmosCollection")
        folder = settings.value(self.EXPORT_DIR_SETTING, "", type=str) or os.path.expanduser("~")
        base = re.sub(r'[<>:"/\\|?*]+', "_", collage.name).strip() or "Collage"
        if scale != 1.0:
            base += f" ({scale * 100:.0f}%)"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export Collage", os.path.join(folder, f"{base}.jpg"),
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff)")
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in EXPORT_FORMATS:
            path += {"PNG": ".png", "TIFF": ".tif"}.get(chosen.split(" ")[0], ".jpg")
        settings.setValue(self.EXPORT_DIR_SETTING, os.path.dirname(path))

        progress = QProgressDialog("Preparing…", "Cancel", 0, len(collage.frames), self)
        progress.setWindowTitle("Exporting Collage")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        worker = ExportWorker(collage.state(), path, scale)
        self._export_job = (worker, progress, path, f"{round(collage.width * scale)} × {round(collage.height * scale)}")
        worker.progress.connect(self._on_export_progress)
        worker.done.connect(self._on_export_done)
        progress.canceled.connect(worker.cancel)
        keep_alive_until_finished(worker)
        worker.start()
        progress.show()

    def _on_export_progress(self, done, total, name):
        progress = self._export_job[1]
        progress.setMaximum(total)
        progress.setValue(done)
        progress.setLabelText(f"Drawing {name} ({done + 1} of {total})…" if name and done < total
                              else "Saving the file…")

    def _on_export_done(self, ok, message, missing):
        _worker, progress, path, size = self._export_job
        progress.close()
        if message == "cancelled":
            return
        if not ok:
            QMessageBox.critical(self, "Export Failed", f"The collage couldn't be exported:\n{message}")
            return
        text = f"Saved the collage ({size} px) to:\n{path}"
        if missing:
            text += (f"\n\n{missing} image{'s' if missing != 1 else ''} couldn't be read "
                     f"(moved or deleted?) and {'were' if missing != 1 else 'was'} left out.")
        box = QMessageBox(QMessageBox.Warning if missing else QMessageBox.Information, "Collage Exported", text,
                          parent=self)
        open_button = box.addButton("Open Image", QMessageBox.ActionRole)
        folder_button = box.addButton("Show in Folder", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        elif box.clickedButton() is folder_button:
            self._show_in_folder(path)

    @staticmethod
    def _show_in_folder(path):
        import subprocess
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
        except Exception as e:
            logger.error(f"Could not show {path}: {e}")

    # Other windows

    def show_collage(self, collage_id, image_ids=None):
        """Switch to a collage (database id), selecting the frames that show image_ids."""
        row = next((i for i, c in enumerate(self.collages) if c.db_id == collage_id), None)
        if row is None:
            return False
        if self.collage_list.currentRow() != row:
            self.collage_list.setCurrentRow(row)
        if image_ids:
            wanted = set(image_ids)
            uids = {f.uid for f in self.current.frames
                    if f.image_id in wanted or any((e.get("frame") or {}).get("image_id") in wanted
                                                   for e in f.merged or [])}
            self._select_uids(uids)
            QTimer.singleShot(50, lambda: self._reveal(uids))  # After the view fits the new canvas
        return True

    def _reveal(self, uids):
        items = [item for item in self.scene.frame_items() if item.frame.uid in uids]
        if items:
            area = items[0].visual_rect()
            for item in items[1:]:
                area = area.united(item.visual_rect())
            self.view.ensureVisible(area, 40, 40)

    def add_to_collage(self, collage_id, image_ids):
        """Add images from elsewhere (Gallery, DSO Detail) to a collage - None:
        a new one - filling their catalog tiles if it has them."""
        if any(i not in self.library for i in image_ids):
            self._load_library()  # Added since this window opened
        rows = [self.library[i] for i in image_ids if i in self.library]
        if not rows:
            return
        if collage_id is None:
            name = self._ask_name("New Collage", self._unique_name(join_names(r["dso_name"] for r in rows) or "Collage"))
            if not name:
                return
            collage = self.create_collage(name)
        elif self.show_collage(collage_id):
            collage = self.current
        else:
            return
        new = [r["id"] for r in rows if r["id"] not in collage.image_ids()]
        if new:
            self.add_images(new)  # Fills catalog tiles, else free spots; selects them
            names = join_names(self.library[i]["dso_name"] for i in new)
            self.status_hint(f"Added {names} to “{collage.name}”")
        else:
            self.show_collage(collage.db_id, image_ids)

    def _image_position(self, image_id):
        """(ra, dec) in degrees of an image's object, or (None, None)."""
        row = self.library.get(image_id)
        try:
            with DatabaseManager().get_connection() as conn:
                found = conn.execute("SELECT ra, dec FROM dsodetail WHERE id = ?", (row["dsodetailid"],)).fetchone()
            return (found[0], found[1]) if found else (None, None)
        except Exception:
            return None, None

    def view_full_image(self, image_id):
        row = self.library.get(image_id)
        if row is not None:
            from DSOGallery import open_image_viewer
            ra, dec = self._image_position(image_id)
            self._viewer = open_image_viewer(self, join_names(self.library[i]["dso_name"]
                                                              for i in self._group_of.get(image_id, [image_id])),
                                             row["image_path"], ra, dec)

    def view_dso_details(self, image_id):
        row = self.library.get(image_id)
        if row is not None and row["dsodetailid"]:
            from DSOGallery import open_dso_details
            open_dso_details(self, row["dsodetailid"], row["dso_name"])

    def _add_view_actions(self, menu, image_id):
        """View Full Image / View DSO Details for an image (canvas and library menus)."""
        row = self.library.get(image_id)
        if row is None:
            return
        full = menu.addAction("View Full Image", lambda: self.view_full_image(image_id))
        full.setEnabled(not self._is_missing(image_id))
        details = menu.addAction(f"View {row['dso_name']} Details", lambda: self.view_dso_details(image_id))
        details.setEnabled(bool(row["dsodetailid"]))

    # Window

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.accept()  # Esc deselects on the canvas; don't close the window
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.view.end_crop(apply=True)
        if self.view._mode in ("move", "resize", "rotate"):
            self.view.cancel_drag()
        self._save_dirty()
        QSettings("CosmosCollection", "CosmosCollection").setValue(self.SPLITTER_SETTING, self.splitter.saveState())
        super().closeEvent(event)


# --------------------------------------------------------------------
# The rest of the app: open the Collage Builder, add images to collages
# --------------------------------------------------------------------

_builder = None  # The Collage Builder window - one, shared by everything that opens it


def _main_window():
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, "_show_collage_builder"):
            return widget
    return None


def open_collage_builder(parent=None, collage_id=None, image_ids=None):
    """Show the Collage Builder (the open one, or a fresh one), optionally on
    a collage (database id) with the frames of image_ids selected."""
    global _builder
    window = _builder
    try:
        alive = window is not None and window.isVisible()
    except RuntimeError:  # Deleted
        alive = False
    if not alive:
        if window is not None:
            try:
                window.deleteLater()  # Closed earlier - start fresh, with current data
            except RuntimeError:
                pass
        window = CollageBuilderWindow([], "All DSO Images", None, _main_window() or parent)
        _builder = window
    if window.isMinimized():
        window.showNormal()
    window.show()
    window.raise_()
    window.activateWindow()
    if collage_id is not None:
        window.show_collage(collage_id, image_ids)
    return window


def list_collages(conn):
    """[(collage id, name)], most recently changed first."""
    return [tuple(row) for row in conn.execute("SELECT id, name FROM usercollages ORDER BY modified_date DESC")]


def collages_using(conn, image_ids):
    """[(collage id, name)] of the collages showing any of image_ids - in a
    frame of their own, or merged into another one."""
    wanted = set(image_ids)
    found = {}
    for collage_id, name, image_id, layout in conn.execute("""
            SELECT uci.collage_id, c.name, uci.userimage_id, uci.layout
            FROM usercollageimages uci JOIN usercollages c ON c.id = uci.collage_id"""):
        if image_id in wanted or any((entry.get("frame") or {}).get("image_id") in wanted
                                     for entry in _json_dict(layout).get("merged") or []):
            found[collage_id] = name
    return sorted(found.items(), key=lambda item: natural_key(item[1]))


def collage_names_by_image(conn):
    """{image id: [collage names]} for every image in a collage (merged ones too)."""
    names = {}
    for name, image_id, layout in conn.execute("""
            SELECT c.name, uci.userimage_id, uci.layout
            FROM usercollageimages uci JOIN usercollages c ON c.id = uci.collage_id"""):
        ids = [image_id] + [(entry.get("frame") or {}).get("image_id")
                            for entry in _json_dict(layout).get("merged") or []]
        for found in ids:
            if found is not None and name not in names.setdefault(found, []):
                names[found].append(name)
    return {image_id: sorted(found, key=natural_key) for image_id, found in names.items()}


def add_collage_actions(menu, parent, image_ids):
    """Add "Add to Collage" and "Show in Collage" submenus for images to a
    context menu (the Gallery's, DSO Detail's). Returns the collages already
    showing them."""
    try:
        with DatabaseManager().get_connection() as conn:
            collages, using = list_collages(conn), collages_using(conn, image_ids)
    except Exception as e:
        logger.error(f"Could not list collages: {e}", exc_info=True)
        return []
    showing = dict(using)
    add = menu.addMenu("Add to Collage")
    for collage_id, name in collages:
        if collage_id in showing:  # Already in it: go and see it instead
            add.addAction(f"{name} (already in it)",
                          lambda checked=False, c=collage_id: open_collage_builder(parent, c, image_ids))
        else:
            add.addAction(name, lambda checked=False, c=collage_id: add_to_collage(parent, c, image_ids))
    if collages:
        add.addSeparator()
    add.addAction("New Collage\u2026", lambda: add_to_collage(parent, None, image_ids))
    if using:
        show = menu.addMenu("Show in Collage")
        for collage_id, name in using:
            show.addAction(name, lambda checked=False, c=collage_id: open_collage_builder(parent, c, image_ids))
    return using


def add_to_collage(parent, collage_id, image_ids):
    """Add images to a collage (None: a new one, named in the Collage
    Builder), shown in the Collage Builder - where it can be undone."""
    window = open_collage_builder(parent)
    window.add_to_collage(collage_id, image_ids)
    return window
