#!/usr/bin/env python3
"""
DSO Image Gallery
Displays all DSO objects with images in a responsive grid gallery format
"""

import sys
import os
import re
import logging
import threading
from datetime import datetime
from PySide6.QtCore import Qt, Signal, QTimer, QThreadPool, QRunnable, QObject, QRectF
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QGroupBox,
                               QMessageBox, QScrollArea, QComboBox, QLineEdit,
                               QFrame, QGridLayout, QMenu, QApplication,
                               QDialog, QFileDialog, QFormLayout, QDialogButtonBox,
                               QCompleter, QSlider, QProgressDialog, QPlainTextEdit,
                               QSizePolicy, QCheckBox, QToolButton, QProgressBar,
                               QListWidget, QListWidgetItem)
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QFontMetricsF

import EquipmentMapper
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS, font_px, font_size, themed_style
from ImageLoader import load_astro_pixmap, load_astro_qimage, read_image_file, write_image_file
from PlateSolver import describe_solving
from AnnotationOverlay import (ANNOTATION_LAYERS, DEFAULT_ANNOTATION_LAYERS, annotation_store, paint_annotations,
                               plate_solve_queue)

logger = logging.getLogger(__name__)

# Image extensions supported for DSO images throughout the gallery
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit', '.fts', '.xisf'}

# Thumbnail badges for images without a plate solve (while annotations are shown)
SOLVE_BADGES = {"unsolved": "Not solved", "queued": "Waiting to solve", "solving": "Solving…",
                "failed": "Solve failed"}

# The gallery's Show filter: a card per object (its other images behind it), or per image file
SHOW_BY_OBJECT, SHOW_EVERY_IMAGE = "One per object", "Every image"


def _file_key(path):
    """Paths of the same file compare equal ('/' or '\\', any case on Windows)."""
    return os.path.normcase(os.path.normpath(path)) if path else ""


def _main_name(name):
    """An object's main designation: "M 31" of "M 31, NGC 224"."""
    return (name or "").split(", ")[0]


def _newest(row):
    """Sort key for an object's images, newest last: when taken (or saved, for
    processed files), then when added."""
    return row.get('date_taken') or '', row.get('created_date') or '', row['imageid']


def insert_user_image(db_manager, image_data, others=None):
    """Insert an AddImageDialog.get_image_data() result into userimages and commit:
    a row for its DSO, plus one (with the same details) for each of the other
    objects chosen in it ('also_dsodetailids') that doesn't have this image yet.
    others: rows for just these other objects instead. Returns the new rows'
    ids, its DSO's first."""
    if others is None:
        targets = [(image_data['dsodetailid'], False)] + \
                  [(dsodetailid, True) for dsodetailid in image_data.get('also_dsodetailids', [])]
    else:
        targets = [(dsodetailid, True) for dsodetailid in others]
    from DatabaseManager import add_user_image_row
    details = (image_data['image_path'], image_data['integration_time'],
               image_data['date_taken'], image_data['notes'], image_data.get('session_id'))
    ids = []
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        for dsodetailid, other in targets:
            if other:
                cursor.execute("SELECT 1 FROM userimages WHERE dsodetailid = ? AND image_path = ?",
                               (dsodetailid, image_data['image_path']))
                if cursor.fetchone():
                    continue
            ids.append(add_user_image_row(cursor, dsodetailid, *details, equipment=image_data.get('equipment')))
        conn.commit()
    return ids


def refresh_open_image_views():
    """The main window's image counts and an open gallery don't watch the
    database, so refresh whichever of them are open."""
    for widget in QApplication.topLevelWidgets():
        name = type(widget).__name__
        try:
            if name == "DSOGalleryWindow" and widget.isVisible():
                widget._refresh_gallery()
            elif name == "MainWindow":
                widget._refresh_data()
        except Exception as e:
            logger.warning(f"Could not refresh {name} after adding an image: {e}")


def import_image(parent, file_path=None, dsodetailid=None, session=None):
    """Add an image to the collection - how every window does it (the gallery,
    the main window, an object's details, a finished session): the Add Image
    dialog, pre-filled with file_path, a DSO and/or a Session Manager session,
    then saved to its DSO and the other objects in it that were chosen.
    Returns the new userimages id for its DSO, or None if nothing was added."""
    dialog = AddImageDialog(parent)
    try:
        if dsodetailid is not None:
            dialog.select_dso(dsodetailid)  # Before the file, so it isn't guessed from it
        if file_path:
            dialog.set_file_path(file_path)
        if session:
            dialog.prefill_from_session(session)
        if dialog.exec() != QDialog.Accepted:
            return None
        image_data = dialog.get_image_data()
        hints = dialog.solve_hints()
        offer_later = not image_data['objects_known'] and dialog.offer_objects_later()
    finally:
        dialog.deleteLater()
    try:
        ids = insert_user_image(DatabaseManager(), image_data)
    except Exception as e:
        logger.error(f"Error adding image {image_data['image_path']}: {e}", exc_info=True)
        QMessageBox.critical(parent, "Error", f"Failed to add the image:\n{e}")
        return None
    refresh_open_image_views()

    message = (f"{os.path.basename(image_data['image_path'])} was added to "
               f"{_join_names(image_data['dso_names'])}.")
    if offer_later:
        # Added before its plate solve finished - offer the other objects in it then
        _ObjectsOffer(image_data, hints)
        message += "\n\nAny other objects in it will be offered when its plate solve finishes."
    QMessageBox.information(parent, "Image Added", message)
    return ids[0]


def _join_names(names):
    """'M 42', 'M 42 and M 43', 'M 42, M 43 and NGC 1977'."""
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


# An object in an image is checked by default if it's in Messier's catalog,
# or an NGC/IC object at least this fraction of the image's short side across
NOTABLE_OBJECT_SIZE = 0.05


def _is_notable(obj, data):
    name = obj.name or ""
    if name.startswith("M "):
        return True
    if not name.startswith(("NGC ", "IC ")) or not obj.size:
        return False
    return obj.size / 60 >= NOTABLE_OBJECT_SIZE * min(data.width, data.height) * data.degrees_per_pixel


def _fill_objects_list(list_widget, data, exclude, checks):
    """List the catalog objects in a plate-solved image (AnnotationData) as
    checkboxes, leaving out the dsodetailids in exclude; checks: dsodetailid ->
    checked, for ones the user already chose (others: checked if notable).
    Returns how many are listed."""
    list_widget.clear()
    seen = set(exclude)
    for obj in data.dsos:
        if obj.dsodetailid is None or obj.dsodetailid in seen:
            continue
        seen.add(obj.dsodetailid)
        text = obj.name
        if obj.size:
            text += f"  ({obj.size:.0f}′)" if obj.size >= 1 else f"  ({obj.size * 60:.0f}″)"  # Arcminutes
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, obj.dsodetailid)
        item.setData(Qt.UserRole + 1, obj.name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checks.get(obj.dsodetailid, _is_notable(obj, data)) else Qt.Unchecked)
        list_widget.addItem(item)
    return list_widget.count()


def _checked_objects(list_widget):
    """[(dsodetailid, name)] of the checked objects in a _fill_objects_list list."""
    items = (list_widget.item(row) for row in range(list_widget.count()))
    return [(item.data(Qt.UserRole), item.data(Qt.UserRole + 1)) for item in items
            if item.checkState() == Qt.Checked]


class _ObjectsOffer(QObject):
    """For an image added before its plate solve finished: waits for the solve
    (starting it if need be), then offers the other notable objects in it."""

    _waiting = set()  # Keeps the offers alive until they're done

    def __init__(self, image_data, hints):
        super().__init__()
        self.image_data = image_data
        self.path = image_data['image_path']
        self.hints = hints
        self._queued = False
        _ObjectsOffer._waiting.add(self)
        plate_solve_queue().changed.connect(self._check)
        self._check(self.path)

    def _check(self, path):
        if path != self.path:
            return
        queue = plate_solve_queue()
        state = queue.state(path)
        if state == "solved":
            data = annotation_store().get(path)
            if data is not None:
                self._done()
                self._offer(data)
        elif state in ("unsolved", "failed"):
            if self._queued or not queue.add(path, self.hints, self.image_data['dso_names'][0], first=True):
                self._done()  # Couldn't be solved, or the solve was stopped
            else:
                self._queued = True

    def _done(self):
        plate_solve_queue().changed.disconnect(self._check)
        _ObjectsOffer._waiting.discard(self)

    def _offer(self, data):
        attached = {self.image_data['dsodetailid'], *self.image_data['also_dsodetailids']}
        dialog = QDialog(QApplication.activeWindow())
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle("Other Objects in the Image")
        layout = QVBoxLayout(dialog)
        label = QLabel(f"{os.path.basename(self.path)} is plate solved. It also shows these objects - "
                       f"add the image to them too?")
        label.setWordWrap(True)
        layout.addWidget(label)
        objects = QListWidget()
        if not _fill_objects_list(objects, data, attached, {}) or not _checked_objects(objects):
            dialog.deleteLater()  # Nothing notable - not worth interrupting for
            return
        layout.addWidget(objects)
        buttons = QDialogButtonBox()
        buttons.addButton("Add to Checked", QDialogButtonBox.AcceptRole)
        buttons.addButton("Not Now", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def add():
            chosen = [dsodetailid for dsodetailid, _name in _checked_objects(objects)]
            if not chosen:
                return
            try:
                insert_user_image(DatabaseManager(), self.image_data, others=chosen)
            except Exception as e:
                logger.error(f"Error adding {self.path} to other objects: {e}", exc_info=True)
                QMessageBox.critical(dialog, "Error", f"Failed to add the image:\n{e}")
                return
            refresh_open_image_views()

        dialog.accepted.connect(add)
        dialog.resize(360, 320)
        dialog.show()


class ThumbnailCache:
    """Cache for storing generated thumbnails to avoid regeneration. Shared by
    the worker threads, so every access holds a lock."""

    def __init__(self, max_size=200):
        self._cache = {}  # (image_path, thumbnail_size) -> QImage
        self._max_size = max_size
        self._access_order = []  # Track access order for LRU eviction
        self._lock = threading.Lock()

    def get(self, key):
        """Get cached thumbnail for an (image path, size) key"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None

    def put(self, key, image):
        """Store thumbnail in cache"""
        with self._lock:
            if key in self._cache:
                # Update existing entry
                if key in self._access_order:
                    self._access_order.remove(key)
            elif len(self._cache) >= self._max_size:
                # Remove least recently used item
                if self._access_order:
                    lru_key = self._access_order.pop(0)
                    self._cache.pop(lru_key, None)

            self._cache[key] = image
            self._access_order.append(key)

    def clear(self):
        """Clear all cached thumbnails"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


# How many original images the thumbnail threads load at once - a big stack
# takes a few hundred MB, read in and decoded
_ORIGINAL_DECODE_SLOTS = threading.BoundedSemaphore(4)


class ThumbnailSignals(QObject):
    """Signals for ThumbnailRunnable (QRunnable doesn't support signals directly)"""
    thumbnail_ready = Signal(object, QImage)   # card, image
    thumbnail_error = Signal(object, str)      # card, error_message


class ThumbnailRunnable(QRunnable):
    """Runnable task for generating a single thumbnail in a thread pool.

    Works in QImage only - PySide holds the GIL for the whole of a QPixmap
    decode, so decoding large images as QPixmaps here froze the gallery for
    seconds each. The gallery makes the QPixmap on the UI thread."""

    # Map thumbnail sizes to size names for disk cache filenames
    SIZE_NAMES = {
        100: 'Small',
        150: 'Medium',
        300: 'Large',
        500: 'ExtraLarge'
    }

    def __init__(self, card, image_path, cache, signals, cancelled_flag, thumbnail_size=150):
        """
        Initialize thumbnail runnable

        Args:
            card: GalleryCard instance to update
            image_path: Path to image file
            cache: ThumbnailCache instance
            signals: ThumbnailSignals instance for emitting signals
            cancelled_flag: List with single boolean for cancellation check
            thumbnail_size: Size of thumbnail (width and height in pixels)
        """
        super().__init__()
        self.card = card
        self.image_path = image_path
        self.cache = cache
        self.signals = signals
        self.cancelled_flag = cancelled_flag
        self.thumbnail_size = thumbnail_size

    def _get_disk_cache_path(self, size=None):
        """Get the path for the disk-cached thumbnail file (of this size, by default)"""
        size = size or self.thumbnail_size
        directory = os.path.dirname(self.image_path)
        basename = os.path.basename(self.image_path)
        name_without_ext = os.path.splitext(basename)[0]
        size_name = self.SIZE_NAMES.get(size, f'{size}px')
        cache_filename = f"{name_without_ext}_{size_name}_Thumbnail.jpg"
        return os.path.join(directory, cache_filename)

    def _is_disk_cache_valid(self, cache_path):
        """Check if disk cache file exists and is newer than the original image"""
        if not os.path.exists(cache_path):
            return False
        try:
            cache_mtime = os.path.getmtime(cache_path)
            original_mtime = os.path.getmtime(self.image_path)
            return cache_mtime >= original_mtime
        except OSError:
            return False

    def _load_from_disk_cache(self, cache_path):
        """Load thumbnail from disk cache"""
        image, _error = read_image_file(cache_path, auto_transform=False)
        return image

    def _save_to_disk_cache(self, image, cache_path):
        """Save thumbnail to disk cache as JPEG at 85% quality"""
        try:
            write_image_file(image, cache_path, "JPEG", 85)
        except Exception:
            pass  # Silently fail if we can't save cache

    def _larger_disk_thumbnail(self):
        """A bigger thumbnail of this image already cached on disk, if any -
        scaling it down takes milliseconds, where decoding a large original
        (a big stack or FITS file) takes seconds."""
        for size in sorted(s for s in self.SIZE_NAMES if s > self.thumbnail_size):
            path = self._get_disk_cache_path(size)
            if self._is_disk_cache_valid(path):
                image = self._load_from_disk_cache(path)
                if image is not None:
                    return image
        return None

    def _decode_original(self, ext):
        """Decode the original image - at about thumbnail size where the format allows"""
        if ext in ['.fits', '.fit', '.fts', '.xisf']:
            return load_astro_qimage(self.image_path)
        # Not EXIF-rotated, so plate-solve annotations line up with it
        image, _error = read_image_file(self.image_path, max_dim=self.thumbnail_size, auto_transform=False)
        return image

    def run(self):
        """Generate thumbnail for single image"""
        # Check if cancelled before starting
        if self.cancelled_flag[0]:
            return

        try:
            cache_key = (self.image_path, self.thumbnail_size)

            # Check memory cache first
            if self.cache:
                cached_image = self.cache.get(cache_key)
                if cached_image is not None:
                    self.signals.thumbnail_ready.emit(self.card, cached_image)
                    return

            # Check if cancelled
            if self.cancelled_flag[0]:
                return

            # Check if disk caching is enabled
            settings = QSettings("CosmosCollection", "CosmosCollection")
            disk_cache_enabled = settings.value("cache_thumbnails_to_disk", True, type=bool)
            disk_cache_path = self._get_disk_cache_path() if disk_cache_enabled else None

            # Try loading from disk cache if enabled and valid
            if disk_cache_enabled and self._is_disk_cache_valid(disk_cache_path):
                disk_image = self._load_from_disk_cache(disk_cache_path)
                if disk_image is not None:
                    # Store in memory cache too
                    if self.cache:
                        self.cache.put(cache_key, disk_image)
                    self.signals.thumbnail_ready.emit(self.card, disk_image)
                    return

            # Check if cancelled
            if self.cancelled_flag[0]:
                return

            if not os.path.exists(self.image_path):
                self.signals.thumbnail_error.emit(self.card, "File Not Found")
                return

            # Check file size
            if os.path.getsize(self.image_path) == 0:
                self.signals.thumbnail_error.emit(self.card, "Empty File")
                return

            _, ext = os.path.splitext(self.image_path.lower())
            image = self._larger_disk_thumbnail() if disk_cache_enabled else None
            if image is None:
                with _ORIGINAL_DECODE_SLOTS:
                    if self.cancelled_flag[0]:
                        return
                    image = self._decode_original(ext)

            # Check if cancelled before emitting
            if self.cancelled_flag[0]:
                return

            if image is None:
                is_astro = ext in ['.fits', '.fit', '.fts', '.xisf']
                self.signals.thumbnail_error.emit(self.card, "Image Load Error" if is_astro else "Load Error")
                return

            # Scale to thumbnail size (gallery card size)
            thumbnail = image.scaled(self.thumbnail_size, self.thumbnail_size, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)

            # Cache the thumbnail in memory
            if self.cache:
                self.cache.put(cache_key, thumbnail)

            # Save to disk cache if enabled
            if disk_cache_enabled and disk_cache_path:
                self._save_to_disk_cache(thumbnail, disk_cache_path)

            self.signals.thumbnail_ready.emit(self.card, thumbnail)

        except Exception as e:
            self.signals.thumbnail_error.emit(self.card, f"Error: {str(e)[:20]}")


class DataLoaderSignals(QObject):
    """Signals for DataLoaderRunnable"""
    data_loaded = Signal(list)  # Emits list of loaded items
    load_error = Signal(str)    # Emits error message


class DataLoaderRunnable(QRunnable):
    """Runnable task for loading gallery data in background"""

    def __init__(self, signals):
        """
        Initialize data loader runnable

        Args:
            signals: DataLoaderSignals instance for emitting signals
        """
        super().__init__()
        self.signals = signals

    def _get_friendly_type_name(self, dso_type):
        """Convert DSO type code to user-friendly name"""
        type_mapping = {
            "GALXY": "Galaxy",
            "DRKNB": "Dark Nebula",
            "OPNCL": "Open Cluster",
            "PLNNB": "Planetary Nebula",
            "BRTNB": "Bright Nebula",
            "SNREM": "Supernova Remnant",
            "GALCL": "Galaxy Cluster",
            "GLOCL": "Globular Cluster",
            "CL+NB": "Cluster + Nebula",
            "GX+DN": "Galaxy + Dark Nebula",
            "ASTER": "Asterism",
            "2STAR": "Double Star",
            "3STAR": "Triple Star",
            "4STAR": "Quadruple Star",
            "1STAR": "Single Star",
            "QUASR": "Quasar",
            "NONEX": "Non-existent",
            "LMCCN": "LMC Cluster/Nebula",
            "LMCDN": "LMC Dark Nebula",
            "LMCGC": "LMC Globular Cluster",
            "LMCOC": "LMC Open Cluster",
            "SMCCN": "SMC Cluster/Nebula",
            "SMCDN": "SMC Dark Nebula",
            "SMCGC": "SMC Globular Cluster",
            "SMCOC": "SMC Open Cluster"
        }
        return type_mapping.get(dso_type, dso_type)

    def run(self):
        """Load gallery data from database"""
        import sqlite3
        from ResourceManager import ResourceManager

        try:
            # Create new SQLite connection in this thread (DatabaseManager is a singleton)
            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            from ResourceManager import attach_update_catalogs
            attach_update_catalogs(conn)

            # Ensure created_date column exists (migration for older databases)
            # Do this before setting row_factory
            # Note: ALTER TABLE cannot use CURRENT_TIMESTAMP as default, so we use NULL
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(userimages)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'created_date' not in columns:
                cursor.execute("ALTER TABLE userimages ADD COLUMN created_date TEXT")
                conn.commit()

            conn.row_factory = sqlite3.Row

            try:
                cursor = conn.cursor()

                # One row per attached image (an image of M 31 and M 32 is two
                # rows) - the window groups them into cards by object or by file
                query = f"""
                SELECT
                    d.id as dsodetailid,
                    ui.id as imageid,
                    ui.image_path,
                    {EquipmentMapper.select_sql('ui')},
                    ui.is_favorite,
                    d.dsotype,
                    d.constellation,
                    GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                        ORDER BY
                            CASE c.catalogue
                                WHEN 'M' THEN 1
                                WHEN 'NGC' THEN 2
                                WHEN 'IC' THEN 3
                                ELSE 4
                            END, c.designation) as name,
                    ui.created_date,
                    d.ra,
                    d.dec,
                    ui.date_taken
                FROM userimages ui
                INNER JOIN dsodetail d ON d.id = ui.dsodetailid
                INNER JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE ui.image_path IS NOT NULL AND ui.image_path != ''
                GROUP BY ui.id
                ORDER BY name
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                # Which collages show each image (for the card tooltips)
                try:
                    from CollageBuilder import collage_names_by_image
                    in_collages = collage_names_by_image(conn)
                except Exception as e:
                    logger.debug(f"Could not look up collages: {e}")
                    in_collages = {}

                # Convert rows to dictionaries
                items = []
                for row in rows:
                    item = {
                        'dsodetailid': row[0],
                        'imageid': row[1],
                        'image_path': row[2],
                        'is_favorite': row[9],
                        'dsotype': row[10] or '',
                        'constellation': row[11] or '',
                        'name': row[12] or 'Unknown',
                        'friendly_type': self._get_friendly_type_name(row[10] or ''),
                        'created_date': row[13] or '',
                        'ra_deg': row[14],
                        'dec_deg': row[15],
                        'date_taken': row[16] or '',
                        'collages': in_collages.get(row[1], [])
                    }
                    # telescope, camera, equipment (both as text), equipment_key ...
                    item.update(EquipmentMapper.image_equipment(tuple(row)[3:9]))
                    items.append(item)

                # Emit success signal with loaded data
                self.signals.data_loaded.emit(items)

            finally:
                # Close the connection
                conn.close()

        except Exception as e:
            # Emit error signal
            self.signals.load_error.emit(str(e))


# --- Opening an image or its DSO (shared with the Collage Builder) ---

_open_windows = []  # Windows opened here, kept alive until closed


def _keep(window):
    _open_windows.append(window)
    window.destroyed.connect(lambda *_args, w=window: _open_windows.remove(w) if w in _open_windows else None)


def _format_ra(ra_deg):
    """RA in degrees as HH:MM:SS.SS"""
    ra_hours = ra_deg / 15.0
    hours = int(ra_hours)
    minutes = int((ra_hours - hours) * 60)
    seconds = ((ra_hours - hours) * 60 - minutes) * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def _format_dec(dec_deg):
    """Dec in degrees as DD:MM:SS.S"""
    sign = '+' if dec_deg >= 0 else '-'
    dec_abs = abs(dec_deg)
    degrees = int(dec_abs)
    minutes = int((dec_abs - degrees) * 60)
    seconds = ((dec_abs - degrees) * 60 - minutes) * 60
    return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:04.1f}"


def open_image_viewer(parent, name, image_path, ra_deg=None, dec_deg=None, versions=None):
    """Show an image full size in the Image Viewer (loaded in the background,
    so large files don't freeze the UI). versions: the object's image paths
    (image_path among them) to step through there. Returns the viewer, or None."""
    try:
        from ImageViewer import ImageViewerWindow
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(parent, "Image Not Found",
                                f"Image file not found:\n{image_path}\n\n"
                                f"The file may have been moved or deleted.")
            return None
        # (pixmap, title, file_path, parent, dso_ra, dso_dec)
        viewer = ImageViewerWindow(None, name, image_path, parent, dso_ra=ra_deg, dso_dec=dec_deg, versions=versions)
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()
        return viewer
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"Failed to open image viewer:\n{str(e)}")
        return None


def open_dso_details(parent, dsodetailid, name=None):
    """Open the DSO Detail window for an object (by dsodetailid)."""
    try:
        from DSODetail import DSODetailWindow

        with DatabaseManager().get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness,
                       CAST(d.sizemin/60.0 AS REAL) as sizemin,
                       CAST(d.sizemax/60.0 AS REAL) as sizemax,
                       d.constellation, d.dsotype, d.dsoclass,
                       GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY
                           CASE c.catalogue
                               WHEN 'M' THEN 1
                               WHEN 'NGC' THEN 2
                               WHEN 'IC' THEN 3
                               ELSE 4
                           END, c.designation) as designations,
                       ui.image_path, ui.integration_time, ui.date_taken, ui.notes,
                       (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                LEFT JOIN userimages ui ON d.id = ui.dsodetailid AND ui.is_favorite = 1
                WHERE d.id = ?
                GROUP BY d.id
            """, (dsodetailid,))
            result = cursor.fetchone()

        if not result:
            QMessageBox.warning(parent, "Error", "Could not load DSO details")
            return None

        (obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max,
         constellation, dso_type, dso_class, designations, image_path, integration_time,
         date_taken, notes, image_count) = tuple(result)

        # Primary designation for catalogue and id
        primary_designation = designations.split(',')[0]
        catalogue, designation = primary_designation.strip().split(' ', 1)

        data = {
            "name": name or designations,
            "ra": _format_ra(ra),
            "dec": _format_dec(dec),
            "ra_deg": ra,
            "dec_deg": dec,
            "magnitude": magnitude,
            "surface_brightness": surface_brightness,
            "size_min": size_min if size_min else 0.0,
            "size_max": size_max if size_max else 0.0,
            "constellation": constellation,
            "dso_type": dso_type,
            "dso_class": dso_class,
            "designations": designations,
            "catalogue": catalogue,
            "id": designation,
            "dsodetailid": obj_id,
            "image_path": image_path,
            "integration_time": integration_time,
            "date_taken": date_taken,
            "notes": notes,
            "image_count": image_count
        }
        detail_window = DSODetailWindow(data)
        _keep(detail_window)
        detail_window.show()
        detail_window.raise_()
        detail_window.activateWindow()
        return detail_window
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"Failed to open DSO details:\n{str(e)}")
        return None


def _load_preview_pixmap(image_path, max_dim=200):
    """Load a small preview pixmap for an image path, including FITS/XISF files.

    Returns None if the file can't be read/decoded (e.g. unsupported format).
    """
    if not image_path or not os.path.exists(image_path):
        return None

    _, ext = os.path.splitext(image_path.lower())

    if ext in ('.fits', '.fit', '.fts', '.xisf'):
        return load_astro_pixmap(image_path, max_dim=max_dim)

    image, _error = read_image_file(image_path, max_dim=max_dim, auto_transform=False)
    if image is None:
        return None

    return QPixmap.fromImage(image).scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class AddImageDialog(WindowPositionMixin, QDialog):
    """Dialog for adding a new image to a DSO"""

    WINDOW_POSITION_KEY = "AddImageDialog"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Image to DSO")
        self.selected_file = None
        self.dso_data = []  # List of (dsodetailid, name) tuples, index-aligned with dso_combo
        self._dso_auto_selected = False
        self._capture_header = None  # FITS/XISF header dict for the selected file, or None
        self._telescope_auto_filled = False
        self._camera_auto_filled = False
        self._date_auto_filled = False
        self._integration_auto_filled = False
        self._equipment_info = None  # LibraryScanner.EquipmentInfo: the user's equipment and file-name matches
        self._equipment = None       # The telescope/camera to save, linked to Equipment on OK
        self._matching_sessions = []  # Session Manager sessions found for the current DSO
        self._dso_positions = {}  # dsodetailid -> (ra, dec) degrees
        self._dso_indexes = {}    # dsodetailid -> dso_combo index
        # Finding the other objects in the image by plate solving it
        self._solved_data = None  # The selected file's AnnotationData, once solved
        self._object_checks = {}  # dsodetailid -> checked, as the user left them
        self._solves_tried = set()  # (path, hint) solves started, so a failure isn't retried
        self._solves_queued = set()  # Paths this dialog queued - stopped if it's cancelled
        self._can_solve_blind = None  # ASTAP is available (no position needed)
        self._session_id = None       # The Session Manager session it came from, once one is chosen
        self._objects_timer = QTimer(self)
        self._objects_timer.setSingleShot(True)
        self._objects_timer.timeout.connect(self._update_objects)

        self.setAcceptDrops(True)
        self._init_ui()
        self._load_dso_list()
        self._load_equipment_list()
        self._update_preview()
        self.setup_window_position()
        plate_solve_queue().changed.connect(self._on_solve_changed)
        plate_solve_queue().progress.connect(self._on_solve_progress)

    def _init_ui(self):
        """Create the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Instructions
        instructions = QLabel("Select an image file and choose which DSO to attach it to.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # --- Image preview / drop zone + DSO selection -----------------
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # Grows with the dialog (both wider and taller) so enlarging the
        # window makes the thumbnail bigger instead of the text fields;
        # the loaded source pixmap is re-scaled to fit on every resize.
        self._preview_source_pixmap = None
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(150, 150)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setCursor(Qt.PointingHandCursor)
        self.preview_label.setToolTip("Click to browse, or drag & drop an image onto this dialog")
        themed_style(self.preview_label, lambda: f"""
            QLabel {{
                background-color: {COLORS['background_light']};
                border: 2px dashed {COLORS['border_light']};
                border-radius: 6px;
                color: {COLORS['text_secondary']};
                font-size: {font_size(9)};
                padding: 6px;
            }}
        """)
        self.preview_label.mousePressEvent = lambda event: self._browse_file()
        top_row.addWidget(self.preview_label, 1)

        # Fields column stays at its natural width (stretch 0) so extra
        # horizontal space is given to the preview above instead.
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected...")
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setMinimumWidth(200)
        self.file_path_edit.setMaximumWidth(260)
        file_layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)
        right_col.addLayout(file_layout)

        dso_label = QLabel("Attach to DSO:")
        right_col.addWidget(dso_label)

        # DSO selection with search
        self.dso_combo = QComboBox()
        self.dso_combo.setEditable(True)
        self.dso_combo.setInsertPolicy(QComboBox.NoInsert)
        self.dso_combo.lineEdit().setPlaceholderText("Search for DSO...")
        # Keep the closed combo box narrow regardless of how long individual
        # DSO entries are (some objects have many catalogue designations);
        # the popup list itself is widened separately so full names stay readable.
        self.dso_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.dso_combo.setMinimumContentsLength(20)
        self.dso_combo.setMinimumWidth(200)
        self.dso_combo.setMaximumWidth(260)
        self.dso_combo.view().setMinimumWidth(380)
        self.dso_combo.activated.connect(self._on_dso_manually_changed)
        right_col.addWidget(self.dso_combo)

        self.detected_label = QLabel("")
        themed_style(self.detected_label, lambda: f"color: {COLORS['info']}; font-size: {font_size(9)};")
        self.detected_label.setWordWrap(True)
        self.detected_label.setMaximumWidth(260)
        self.detected_label.hide()
        right_col.addWidget(self.detected_label)

        # Offers to import telescope/camera/date/integration time from an
        # existing Session Manager session for the same DSO, when the file's
        # own metadata didn't supply them (e.g. a PixelMath-composited XISF
        # result, which carries no acquisition metadata at all). Never
        # auto-applied - always an explicit click, matching this app's
        # existing "always confirm, never silently auto-attach" convention
        # for cross-feature data reuse (see SessionManager.DropMatchDialog).
        self.session_import_btn = QPushButton("Import from Session...")
        self.session_import_btn.setMaximumWidth(260)
        self.session_import_btn.clicked.connect(self._show_session_import_menu)
        self.session_import_btn.hide()
        right_col.addWidget(self.session_import_btn)

        # The other catalog objects in the image, found by plate solving it
        # in the background - the image can be added to them too
        self.objects_status = QLabel("")
        themed_style(self.objects_status, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        self.objects_status.setWordWrap(True)
        self.objects_status.setMaximumWidth(260)
        self.objects_status.hide()
        right_col.addWidget(self.objects_status)
        self.objects_list = QListWidget()
        self.objects_list.setMaximumWidth(260)
        self.objects_list.setMaximumHeight(130)
        self.objects_list.setToolTip("The image is also added to the checked objects")
        self.objects_list.itemChanged.connect(self._on_object_checked)
        self.objects_list.hide()
        right_col.addWidget(self.objects_list)
        right_col.addStretch()

        top_row.addLayout(right_col)
        # Give the image/DSO row the extra vertical space on a taller resize
        # too, so the preview grows in both directions.
        layout.addLayout(top_row, 1)

        # --- Optional capture details -----------------------------------
        # Capped to a fixed max width so resizing the dialog doesn't stretch
        # these fields - all the extra space goes to the preview instead.
        FIELD_MAX_WIDTH = 260

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.telescope_combo = QComboBox()
        self.telescope_combo.setEditable(True)
        self.telescope_combo.setInsertPolicy(QComboBox.NoInsert)
        self.telescope_combo.lineEdit().setPlaceholderText("e.g., 8\" SCT")
        self.telescope_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.telescope_combo.setMinimumContentsLength(18)
        self.telescope_combo.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Telescope:", self.telescope_combo)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.setInsertPolicy(QComboBox.NoInsert)
        self.camera_combo.lineEdit().setPlaceholderText("e.g., ASI294MC Pro")
        self.camera_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(18)
        self.camera_combo.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Camera:", self.camera_combo)

        self.integration_edit = QLineEdit()
        self.integration_edit.setPlaceholderText("e.g., 2h 30m")
        self.integration_edit.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Integration Time:", self.integration_edit)

        date_layout = QHBoxLayout()
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("e.g., 2024-01-15")
        self.date_edit.setMaximumWidth(FIELD_MAX_WIDTH - 70)
        date_layout.addWidget(self.date_edit)
        today_btn = QPushButton("Today")
        today_btn.setToolTip("Fill in today's date")
        today_btn.clicked.connect(self._fill_today_date)
        date_layout.addWidget(today_btn)
        date_layout.addStretch()
        form_layout.addRow("Date Taken:", date_layout)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes about this image")
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setMaximumWidth(FIELD_MAX_WIDTH)
        form_layout.addRow("Notes:", self.notes_edit)

        layout.addLayout(form_layout)

        # Inline validation feedback (shown instead of popping a dialog
        # for every missing field)
        self.status_label = QLabel("")
        themed_style(self.status_label, lambda: f"color: {COLORS['error']};")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setMinimumWidth(460)
        self.resize(460, self.sizeHint().height())

    def _load_dso_list(self):
        """Load all DSOs from database for the combo box"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT d.id as dsodetailid,
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                               ORDER BY CASE c.catalogue
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation) as name,
                           d.constellation,
                           d.dsotype,
                           d.ra,
                           d.dec
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    GROUP BY d.id
                    ORDER BY
                        CASE
                            WHEN name LIKE 'M %' THEN 1
                            WHEN name LIKE 'NGC %' THEN 2
                            WHEN name LIKE 'IC %' THEN 3
                            ELSE 4
                        END,
                        name
                """
                cursor.execute(query)
                rows = cursor.fetchall()

                # Clear and populate combo box
                self.dso_combo.clear()
                self.dso_data = []

                for row in rows:
                    dsodetailid, name, constellation, dsotype, ra, dec = row
                    display_text = f"{name} ({constellation})"
                    self._dso_indexes[dsodetailid] = self.dso_combo.count()
                    self.dso_combo.addItem(display_text, dsodetailid)
                    self.dso_data.append((dsodetailid, name))
                    if ra is not None and dec is not None:
                        self._dso_positions[dsodetailid] = (ra, dec)

                # Setup completer for search functionality
                completer = QCompleter([self.dso_combo.itemText(i) for i in range(self.dso_combo.count())])
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                completer.setFilterMode(Qt.MatchContains)
                self.dso_combo.setCompleter(completer)

                # Clear selection so placeholder text is shown
                self.dso_combo.setCurrentIndex(-1)
                self.dso_combo.lineEdit().clear()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load DSO list: {str(e)}")

    def _load_equipment_list(self):
        """Load user telescopes and cameras into their respective dropdowns"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Load telescopes
                cursor.execute("SELECT name FROM usertelescopes ORDER BY name")
                for row in cursor.fetchall():
                    self.telescope_combo.addItem(row[0])

                # Load cameras
                cursor.execute("SELECT name FROM userequipment WHERE equipment_type = 'camera' ORDER BY name")
                for row in cursor.fetchall():
                    self.camera_combo.addItem(row[0])

                # Which equipment the names in files mean (matched in the import wizard)
                import LibraryScanner
                self._equipment_info = LibraryScanner.load_equipment_info(conn)

            # Clear selection so placeholder text is shown
            self.telescope_combo.setCurrentIndex(-1)
            self.telescope_combo.lineEdit().clear()
            self.camera_combo.setCurrentIndex(-1)
            self.camera_combo.lineEdit().clear()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load equipment list: {str(e)}")

    def _equipment_name(self, kind, value, header=None):
        """A telescope/camera name from a file or session ('z61', 'ZWO ASI2600MC Pro')
        as the user's equipment is named: what it's matched to, else what the
        Import Wizard would suggest - so the gallery doesn't list the same gear
        under several names. kind: 'telescope' or 'camera'."""
        value = str(value).strip()
        if self._equipment_info is None:
            return value
        return EquipmentMapper.display_name(kind, value, self._equipment_info, header)

    def _browse_file(self):
        """Open file dialog to select an image"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            os.path.expanduser("~"),
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fts *.xisf);;"
            "PNG Files (*.png);;"
            "JPEG Files (*.jpg *.jpeg);;"
            "TIFF Files (*.tif *.tiff);;"
            "FITS Files (*.fits *.fit *.fts);;"
            "XISF Files (*.xisf);;"
            "All Files (*.*)"
        )
        if file_name:
            self.set_file_path(file_name)

    def set_file_path(self, path):
        """Set the selected file path, refresh the preview and try to auto-detect
        the DSO and capture info (telescope/camera/date) from the file's metadata"""
        if path != self.selected_file:
            self._stop_solves(keep=path)
            self._solved_data = None
            self._object_checks = {}
        self.selected_file = path
        self.file_path_edit.setText(path)
        self._clear_error(self.file_path_edit)
        self._capture_header = self._read_capture_header()
        self._update_preview()
        self._try_auto_detect_dso()
        self._try_auto_detect_capture_info()
        self._check_for_matching_sessions()
        self._objects_timer.start(0)

    def _read_capture_header(self):
        """Read FITS/XISF header keywords (OBJECT, TELESCOP, INSTRUME, DATE-OBS,
        ...) from the selected file, if it's one of those formats. Returns None
        for other formats, or if the header can't be read."""
        if not self.selected_file:
            return None
        ext = os.path.splitext(self.selected_file)[1].lower()
        try:
            if ext in ('.fits', '.fit', '.fts'):
                from SessionFileScanner import extract_fits_header
                return extract_fits_header(self.selected_file)
            elif ext == '.xisf':
                from SessionFileScanner import extract_xisf_header
                return extract_xisf_header(self.selected_file)
        except Exception as e:
            logger.debug(f"Could not read capture header from {self.selected_file}: {e}")
        return None

    def _update_preview(self):
        """(Re)load the preview source for the selected file and render it at the
        preview box's current size. Loaded once per file at a resolution large
        enough to still look sharp when the dialog is enlarged."""
        self._preview_source_pixmap = (
            _load_preview_pixmap(self.selected_file, max_dim=600) if self.selected_file else None
        )
        self._render_preview()

    def _render_preview(self):
        """Scale the cached preview source to fit the preview box's current size"""
        if self._preview_source_pixmap:
            scaled = self._preview_source_pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)
        elif self.selected_file:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Preview not\navailable")
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Drag && drop\nan image here\nor click to browse")

    def resizeEvent(self, event):
        """Re-scale the preview thumbnail to fill the enlarged/shrunk preview box"""
        super().resizeEvent(event)
        self._render_preview()

    def _match_dso_index(self, text):
        """Look for a DSO catalogue designation (e.g. 'M31', 'NGC 7000') embedded
        in an arbitrary piece of text - a filename, or a FITS/XISF header's
        OBJECT value."""
        # Keep hyphens (but drop other punctuation/whitespace) so distinct
        # designations that only differ by a hyphen - e.g. Messier "M 16"
        # vs. Minkowski "M 1-6" - don't collapse into the same "M16" string
        # and get confused for one another.
        text_norm = re.sub(r'[^A-Z0-9-]', '', text.upper())
        if not text_norm:
            return None

        best_index = None
        best_len = 0
        for index, (dsodetailid, name) in enumerate(self.dso_data):
            for designation in name.split(','):
                d_norm = re.sub(r'[^A-Z0-9-]', '', designation.upper())
                if len(d_norm) >= 2 and len(d_norm) > best_len and d_norm in text_norm:
                    best_index = index
                    best_len = len(d_norm)
        return best_index

    def _guess_dso_index_from_filename(self, file_path):
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return self._match_dso_index(basename)

    def _try_auto_detect_dso(self):
        """Auto-select the DSO combo, preferring the OBJECT name from the file's
        FITS/XISF header over a filename guess, without overriding a selection
        the user made themselves"""
        if not self.selected_file or (self.dso_combo.currentIndex() >= 0 and not self._dso_auto_selected):
            return

        source = None
        index = None
        object_name = self._capture_header.get('OBJECT') if self._capture_header else None
        if object_name:
            index = self._match_dso_index(str(object_name))
            if index is not None:
                source = "image metadata"
        if index is None:
            index = self._guess_dso_index_from_filename(self.selected_file)
            if index is not None:
                source = "filename"

        if index is not None:
            self.dso_combo.setCurrentIndex(index)
            self._dso_auto_selected = True
            self.detected_label.setText(f"Auto-detected from {source} — please verify this is correct.")
            self.detected_label.show()
        elif self._dso_auto_selected:
            self.dso_combo.setCurrentIndex(-1)
            self.dso_combo.lineEdit().clear()
            self._dso_auto_selected = False
            self.detected_label.hide()
        self._objects_timer.start(0)

    def _try_auto_detect_capture_info(self):
        """Fill telescope/camera/date from the file's FITS/XISF header, each
        only if that field is still empty - mirrors _try_auto_detect_dso's
        auto-vs-manual tracking so switching to a different dropped file
        replaces a previous auto-fill but never clobbers a manual edit."""
        header = self._capture_header or {}

        telescope = header.get('TELESCOP')
        if telescope and (self._telescope_auto_filled or not self.telescope_combo.currentText().strip()):
            self.telescope_combo.setCurrentText(self._equipment_name("telescope", telescope, header))
            self._telescope_auto_filled = True
        elif not telescope and self._telescope_auto_filled:
            self.telescope_combo.setCurrentText("")
            self._telescope_auto_filled = False

        camera = header.get('INSTRUME')
        if camera and (self._camera_auto_filled or not self.camera_combo.currentText().strip()):
            self.camera_combo.setCurrentText(self._equipment_name("camera", camera, header))
            self._camera_auto_filled = True
        elif not camera and self._camera_auto_filled:
            self.camera_combo.setCurrentText("")
            self._camera_auto_filled = False

        date_str = self._format_date_obs(header.get('DATE-OBS'))
        if date_str and (self._date_auto_filled or not self.date_edit.text().strip()):
            self.date_edit.setText(date_str)
            self._date_auto_filled = True
        elif not date_str and self._date_auto_filled:
            self.date_edit.setText("")
            self._date_auto_filled = False

    @staticmethod
    def _format_date_obs(date_obs):
        """Reformat a FITS/XISF DATE-OBS value (e.g. '2024-01-15T22:30:00.000')
        to just the date portion, matching this dialog's date_edit convention."""
        if not date_obs:
            return None
        return str(date_obs).strip()[:10] or None

    def _find_matching_sessions(self, dso_index):
        """Find Session Manager sessions for the DSO at dso_data[dso_index],
        matching by normalized designation (the same normalization
        _match_dso_index uses) since a session's dso_name is free text a user
        typed when logging it, not a catalog ID. Only returns sessions that
        have at least one importable field (telescope/camera/integration
        time) - a bare 'Planned' session with nothing filled in yet isn't
        worth suggesting."""
        if dso_index is None or dso_index < 0 or dso_index >= len(self.dso_data):
            return []

        _, designations_str = self.dso_data[dso_index]
        designations_norm = {
            re.sub(r'[^A-Z0-9-]', '', d.upper()) for d in designations_str.split(',')
        }

        matches = []
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # camera: its Equipment's name, else the name it was given
                cursor.execute(f"""
                    SELECT s.id, s.dso_name, s.session_date, tel.name as telescope_name,
                           s.camera_id, {EquipmentMapper.session_camera_sql('s')}, s.integration_seconds, s.status
                    FROM usersessions s
                    LEFT JOIN usertelescopes tel ON s.telescope_id = tel.id
                    ORDER BY s.session_date DESC, s.id DESC
                """)
                for (session_id, dso_name, session_date, telescope_name, camera_id, camera, integration_seconds,
                     status) in cursor.fetchall():
                    if not dso_name:
                        continue
                    name_norm = re.sub(r'[^A-Z0-9-]', '', dso_name.upper())
                    if name_norm not in designations_norm:
                        continue
                    if not (telescope_name or camera or integration_seconds):
                        continue
                    matches.append({
                        "id": session_id, "dso_name": dso_name, "session_date": session_date,
                        "telescope_name": telescope_name, "camera_id": camera_id, "camera": camera,
                        "integration_seconds": integration_seconds, "status": status,
                    })
        except Exception as e:
            logger.debug(f"Error finding matching sessions: {e}")

        return matches

    def _check_for_matching_sessions(self):
        """Show/hide the 'Import from Session...' button depending on whether
        the currently-selected DSO has any matching sessions with data to offer."""
        self._matching_sessions = self._find_matching_sessions(self.dso_combo.currentIndex())
        if self._matching_sessions:
            count = len(self._matching_sessions)
            self.session_import_btn.setText(
                "Import from Session..." if count == 1 else f"Import from Session... ({count} found)")
            self.session_import_btn.show()
        else:
            self.session_import_btn.hide()

    def _show_session_import_menu(self):
        """Let the user pick which matching session to import equipment/date/
        integration time from, when there's more than one - never guesses."""
        if not self._matching_sessions:
            return

        menu = QMenu(self)
        for session in self._matching_sessions:
            hours = (session["integration_seconds"] or 0) / 3600.0
            parts = [session["session_date"] or "Unknown date"]
            if session["telescope_name"]:
                parts.append(session["telescope_name"])
            if session["camera"]:
                parts.append(session["camera"])
            if hours > 0:
                parts.append(f"{hours:.1f}h")
            parts.append(f"({session['status']})")
            action = menu.addAction(" — ".join(parts))
            action.triggered.connect(lambda checked=False, s=session: self._import_session_data(s))

        menu.exec(self.session_import_btn.mapToGlobal(
            self.session_import_btn.rect().bottomLeft()))

    def prefill_from_session(self, session):
        """Pre-fill from a Session Manager session (the completion handoff): its
        DSO, then its telescope/camera/date/integration time. Call after
        set_file_path() so the session's values win over the file's metadata."""
        index = self._match_dso_index(session.get("dso_name") or "")
        if index is not None:
            self.dso_combo.setCurrentIndex(index)
            # Chosen from the session, not guessed - a later file won't replace it.
            self._dso_auto_selected = False
            self.detected_label.setText(f"From session '{session.get('dso_name')}' — please verify.")
            self.detected_label.show()
            self._clear_error(self.dso_combo)
            self._check_for_matching_sessions()
            self._objects_timer.start(0)
        self._import_session_data(session)

    def select_dso(self, dsodetailid):
        """Choose the DSO to add the image to, as if the user picked it."""
        index = self._dso_indexes.get(dsodetailid)
        if index is None:
            return
        self.dso_combo.setCurrentIndex(index)
        self._on_dso_manually_changed(index)

    def _import_session_data(self, session):
        """Apply a chosen session's telescope/camera/date/integration time.
        An explicit user click (via the menu above), so unlike the passive
        metadata auto-fill this always overwrites the fields - the user just
        told us which session's data they want. The image is linked to it."""
        self._session_id = session.get("id")
        if session["telescope_name"]:
            self.telescope_combo.setCurrentText(session["telescope_name"])
            self._telescope_auto_filled = True
        if session.get("camera"):
            # The session's camera in Equipment - else its name, matched like a file's
            info = self._equipment_info
            item = info.cameras.get(session.get("camera_id")) if info is not None else None
            self.camera_combo.setCurrentText(item["name"] if item else self._equipment_name("camera", session["camera"]))
            self._camera_auto_filled = True
        if session["session_date"]:
            self.date_edit.setText(str(session["session_date"])[:10])
            self._date_auto_filled = True
        if session["integration_seconds"]:
            self.integration_edit.setText(self._format_integration_seconds(session["integration_seconds"]))
            self._integration_auto_filled = True

        self._clear_error(self.telescope_combo)
        self._clear_error(self.camera_combo)

    @staticmethod
    def _format_integration_seconds(seconds):
        """Format a session's total integration_seconds as 'Xh Ym', matching
        this dialog's integration_edit placeholder convention (e.g. '2h 30m')."""
        total_minutes = round(seconds / 60.0)
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            return f"{hours}h {minutes}m"
        elif hours:
            return f"{hours}h"
        return f"{minutes}m"

    def _on_dso_manually_changed(self, index):
        """User picked a DSO themselves; stop treating the selection as a guess"""
        self._dso_auto_selected = False
        self.detected_label.hide()
        self._clear_error(self.dso_combo)
        self._check_for_matching_sessions()
        self._objects_timer.start(0)

    # --- The other objects in the image (plate solve) ---------------------

    def solve_hints(self):
        """Where to plate solve the selected file near: {"ra", "dec", "radius"}
        (degrees) from where its FITS/XISF header says it was pointed, else
        the chosen DSO's position - None if neither is known."""
        from SessionFileScanner import _file_coordinates
        position = _file_coordinates(self._capture_header) if self._capture_header else None
        if position is None:
            position = self._dso_positions.get(self.dso_combo.currentData())
        return {"ra": position[0], "dec": position[1], "radius": 15} if position else None

    def _solve_attempt(self, hints):
        return self.selected_file, (round(hints["ra"], 3), round(hints["dec"], 3)) if hints else None

    def offer_objects_later(self):
        """Whether to offer the other objects in the image after it's added
        (they weren't known yet): its plate solve is under way, or hasn't been
        tried near where it points."""
        if plate_solve_queue().state(self.selected_file) in ("queued", "solving", "checking", "solved"):
            return True
        hints = self.solve_hints()
        return hints is not None and self._solve_attempt(hints) not in self._solves_tried

    def _main_dso_name(self):
        """The chosen DSO's main designation ('M 42'), or ''."""
        index = self.dso_combo.currentIndex()
        return self.dso_data[index][1].split(',')[0].strip() if 0 <= index < len(self.dso_data) else ""

    def _blind_solving(self):
        """Whether the image can be solved without knowing where it points
        (ASTAP; astrometry.net could, but uploads - wait for the DSO instead)."""
        if self._can_solve_blind is None:
            from PlateSolver import PlateSolver, solver_order
            self._can_solve_blind = "astap" in solver_order() and PlateSolver().is_astap_available()
        return self._can_solve_blind

    def _show_objects_status(self, text, tooltip=""):
        self.objects_status.setText(text)
        self.objects_status.setToolTip(tooltip)
        self.objects_status.setVisible(bool(text))

    def _on_solve_changed(self, path):
        if path == self.selected_file and self._objects_timer is not None:
            self._objects_timer.start(0)

    def _on_solve_progress(self, path, _name, message):
        if path == self.selected_file and plate_solve_queue().state(path) == "solving":
            self._show_objects_status("Finding the objects in this image (plate solving)…", message)

    def _update_objects(self):
        """Show the other objects in the selected file once it's plate solved,
        starting the solve if it isn't (and can be)."""
        path = self.selected_file
        if not path:
            self._show_objects_status("")
            self.objects_list.hide()
            return
        queue = plate_solve_queue()
        state = queue.state(path)
        if state == "solved":
            data = annotation_store().get(path)
            if data is not None:
                self._show_objects(data)
                return
            state = "checking"  # Being read - ready() comes back here
        self._solved_data = None
        self.objects_list.hide()
        if state == "checking":
            self._show_objects_status("Checking for a plate solve…")
            return
        if state in ("queued", "solving"):
            self._show_objects_status("Finding the objects in this image (plate solving)…")
            return
        hints = self.solve_hints()
        attempt = self._solve_attempt(hints)
        if attempt not in self._solves_tried and (hints or self._blind_solving()):
            self._solves_tried.add(attempt)
            if queue.add(path, hints, self._main_dso_name() or os.path.basename(path), first=True):
                self._solves_queued.add(path)
                self._show_objects_status("Finding the objects in this image (plate solving)…")
                return
        if state == "failed":
            self._show_objects_status("Couldn't plate solve this image to find the other objects in it.",
                                      queue.failure(path) or "")
        elif hints is None:
            self._show_objects_status("Choose the DSO to also find the other objects in this image.")
        else:
            self._show_objects_status("")

    def _show_objects(self, data):
        """List the solved image's other objects (choosing its DSO from them,
        if none is chosen yet)."""
        self._solved_data = data
        if self.dso_combo.currentIndex() < 0:
            best = next((obj for obj in data.dsos if obj.dsodetailid in self._dso_indexes), None)
            if best is not None:
                self.dso_combo.setCurrentIndex(self._dso_indexes[best.dsodetailid])
                self._dso_auto_selected = True
                self.detected_label.setText("Found by plate solving the image — please verify this is correct.")
                self.detected_label.show()
                self._clear_error(self.dso_combo)
                self._check_for_matching_sessions()
        self.objects_list.blockSignals(True)
        count = _fill_objects_list(self.objects_list, data, {self.dso_combo.currentData()}, self._object_checks)
        self.objects_list.blockSignals(False)
        self.objects_list.setVisible(count > 0)
        self._show_objects_status("Also add it to these objects in the image:" if count
                                  else "No other catalog objects are in this image.")

    def _on_object_checked(self, item):
        self._object_checks[item.data(Qt.UserRole)] = item.checkState() == Qt.Checked

    def _stop_solves(self, keep=None):
        """Stop the plate solves this dialog started (but keep's), if still waiting or running."""
        stopping = [path for path in self._solves_queued if path != keep]
        if stopping:
            plate_solve_queue().stop(stopping)
        self._solves_queued.difference_update(stopping)

    def done(self, result):
        # Added: its solve carries on (the other objects are offered after it).
        # Cancelled: nobody is waiting for it.
        if self._objects_timer is not None:
            self._objects_timer.stop()
            self._objects_timer = None
            if result != QDialog.Accepted:
                self._stop_solves()
            plate_solve_queue().changed.disconnect(self._on_solve_changed)
            plate_solve_queue().progress.disconnect(self._on_solve_progress)
        super().done(result)

    def _fill_today_date(self):
        """Fill the date field with today's date"""
        self.date_edit.setText(datetime.now().strftime('%Y-%m-%d'))

    def _mark_error(self, widget):
        themed_style(widget, lambda: f"border: 1px solid {COLORS['error']};")

    def _clear_error(self, widget):
        widget.setStyleSheet("")

    def dragEnterEvent(self, event):
        """Accept drag if it is a single supported image file"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Set the dropped file as the selected image"""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                event.acceptProposedAction()
                self.set_file_path(file_path)

    def _validate_and_accept(self):
        """Validate inputs before accepting, showing inline feedback instead of popups"""
        errors = []
        self._clear_error(self.file_path_edit)
        self._clear_error(self.dso_combo)

        if not self.selected_file:
            errors.append("Select an image file.")
            self._mark_error(self.file_path_edit)
        elif not os.path.exists(self.selected_file):
            errors.append("The selected image file no longer exists.")
            self._mark_error(self.file_path_edit)

        if self.dso_combo.currentIndex() < 0:
            errors.append("Choose which DSO to attach the image to.")
            self._mark_error(self.dso_combo)

        if errors:
            self.status_label.setText("  •  ".join(errors))
            self.status_label.show()
            return

        # The telescope and camera, linked to the user's Equipment - asking
        # about any that aren't in it (cancelling stays here)
        from EquipmentManager import choose_equipment
        equipment = choose_equipment(self, self.telescope_combo.currentText(),
                                     self.camera_combo.currentText(), self._capture_header)
        if equipment is None:
            return
        self._equipment = equipment

        self.status_label.hide()
        self.accept()

    def get_image_data(self):
        """Return the entered image data: also_dsodetailids are the other objects
        in it chosen; objects_known: whether they were known (it was solved)."""
        known = self._solved_data is not None
        others = _checked_objects(self.objects_list) if known else []
        return {
            'dsodetailid': self.dso_combo.currentData(),
            'also_dsodetailids': [dsodetailid for dsodetailid, _name in others],
            'dso_names': [self._main_dso_name()] + [name for _dsodetailid, name in others],
            'objects_known': known,
            'image_path': self.selected_file,
            # {"telescope_id", "telescope_name", "camera_id", "camera_name"} (EquipmentMapper.py)
            'equipment': self._equipment or {
                'telescope_name': self.telescope_combo.currentText().strip() or None,
                'camera_name': self.camera_combo.currentText().strip() or None},
            'integration_time': self.integration_edit.text().strip(),
            'date_taken': self.date_edit.text().strip(),
            'notes': self.notes_edit.toPlainText().strip(),
            'session_id': self._session_id,
        }


class GalleryCard(QFrame):
    """Individual card widget displaying a DSO thumbnail and info"""

    double_clicked = Signal(dict)  # Emits item_data when double-clicked
    context_menu_requested = Signal(dict, object)  # Emits item_data and position
    stack_toggled = Signal(dict)  # An object's card: show or hide its other images after it

    def __init__(self, item_data, parent=None, thumbnail_size=150, expanded=False):
        """
        Initialize gallery card

        Args:
            item_data (dict): Dictionary describing the image shown
                - dsodetailid: DSO ID
                - imageid: userimages.id for this specific image
                - name: DSO name
                - dsotype: DSO type code
                - image_path: Path to this image
                - equipment: Equipment used for this image
                - role: 'stack' (an object's card, its images in 'stack', this
                  one first), 'version' (one of an expanded object's other
                  images, after its card) or 'image' (an image file - with
                  'rows' for each object it's attached to)
            thumbnail_size (int): Size of thumbnail in pixels (default 150)
            expanded: an object's card with its other images showing after it
        """
        super().__init__(parent)
        self.item_data = item_data
        self.thumbnail_size = thumbnail_size
        self._thumb = None                    # The plain thumbnail (shown as is, or annotated)
        self._annotations = (None, {}, None)  # AnnotationData, layers, plate solve state
        self._grouped = item_data.get('role') == 'version'  # Tinted with the object card it follows
        self.stack_button = None
        self._init_ui()
        if self.stack_button is not None:
            self.set_expanded(expanded)

    def _init_ui(self):
        """Create card layout with thumbnail and labels"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Thumbnail label (dynamic size)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        themed_style(self.thumbnail_label, lambda: f"""
            QLabel {{
                background-color: {COLORS['background_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
            }}
        """)

        # Placeholder text
        self.thumbnail_label.setText("Loading...")
        layout.addWidget(self.thumbnail_label)

        version = self.item_data.get('role') == 'version'
        file_name = os.path.basename(self.item_data.get('image_path') or '')

        # DSO name label - a version (after its object's card) is told apart by its file name
        name_label = QLabel(self.item_data.get('name', 'Unknown'))
        themed_style(name_label, lambda: f"font-weight: bold; font-size: {font_px(12)};")
        name_label.setAlignment(Qt.AlignCenter)
        if version:
            name_label.ensurePolished()  # The style sheet's font, for measuring
            name_label.setText(name_label.fontMetrics().elidedText(file_name, Qt.ElideMiddle, self.thumbnail_size))
        else:
            name_label.setWordWrap(True)
        layout.addWidget(name_label)

        # DSO type label - a version's date instead
        subtitle = (self.item_data.get('date_taken') if version else None) or self.item_data.get('friendly_type')
        type_label = QLabel(subtitle or 'Unknown')
        themed_style(type_label, lambda: f"font-size: {font_px(10)}; color: {COLORS['text_secondary']};")
        type_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(type_label)

        # Tooltip with per-image details, since an object's images otherwise look alike
        equipment = self.item_data.get('equipment', '').strip()
        tooltip_lines = [self.item_data.get('name', 'Unknown'), file_name]
        if equipment:
            tooltip_lines.append(f"Equipment: {equipment}")
        if self.item_data.get('date_taken'):
            tooltip_lines.append(f"Date: {self.item_data['date_taken']}")
        if self.item_data.get('is_favorite'):
            tooltip_lines.append("Favorite image")
        if self.item_data.get('collages'):
            tooltip_lines.append(f"In collages: {', '.join(self.item_data['collages'])}")
        stack = self.item_data.get('stack') or []
        if len(stack) > 1:
            shown = "the favorite" if self.item_data.get('is_favorite') else "the newest"
            tooltip_lines.append(f"\n{len(stack)} images of this object - {shown} shows")
        self._tooltip = '\n'.join(tooltip_lines)
        self.setToolTip(self._tooltip)

        # An object with more images: a badge on its thumbnail shows them after its card
        if len(stack) > 1:
            self.stack_button = QToolButton(self.thumbnail_label)
            self.stack_button.setCursor(Qt.PointingHandCursor)
            # Like the plate solve badge: legible on any image, in any theme
            self.stack_button.setStyleSheet(f"""
                QToolButton {{
                    background-color: rgba(0, 0, 0, 170); color: #ebebeb; border: none;
                    border-radius: 3px; padding: 1px 5px; font-size: {11 if self.thumbnail_size >= 150 else 10}px;
                }}
                QToolButton:hover {{ background-color: rgba(0, 0, 0, 220); color: white; }}
            """)
            self.stack_button.clicked.connect(lambda: self.stack_toggled.emit(self.item_data))

        # Card styling - card width is thumbnail size + padding (20px for margins and borders)
        card_width = self.thumbnail_size + 20
        self.setFixedWidth(card_width)
        self._apply_style()
        self.setCursor(Qt.PointingHandCursor)

    def _apply_style(self):
        """The card's frame - tinted with the accent while it's one of an
        object's images shown together."""
        grouped = self._grouped

        def build():
            if grouped:
                accent = QColor(COLORS['accent'])
                background = f"rgba({accent.red()}, {accent.green()}, {accent.blue()}, 45)"
                border = f"1px solid {COLORS['accent']}"
            else:
                background, border = COLORS['background_lighter'], f"1px solid {COLORS['border']}"
            return f"""
                GalleryCard {{
                    background-color: {background};
                    border: {border};
                    border-radius: 5px;
                }}
                GalleryCard:hover {{
                    border: 2px solid {COLORS['accent']};
                    background-color: {COLORS['background_hover']};
                }}
            """
        themed_style(self, build)

    def set_expanded(self, expanded):
        """An object's card: whether its other images show after it."""
        count = len(self.item_data.get('stack') or [])
        self.stack_button.setText(f"{count} ◂" if expanded else f"{count} ▸")
        self.stack_button.setToolTip(f"Hide the other {count - 1} images of this object" if expanded else
                                     f"{count} images of this object - show them all here.\n"
                                     f"Double-click to step through them in the Image Viewer.")
        self.stack_button.adjustSize()
        self.stack_button.move(self.thumbnail_size - self.stack_button.width() - 4, 4)
        self._grouped = expanded
        self._apply_style()

    def set_thumbnail(self, pixmap):
        """Update thumbnail with actual image"""
        if pixmap and not pixmap.isNull():
            # Scale to fit thumbnail size while maintaining aspect ratio
            self._thumb = pixmap.scaled(self.thumbnail_size, self.thumbnail_size, Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            self._show_thumbnail()
            self.thumbnail_label.setText("")  # Clear placeholder text

    def set_annotations(self, data, layers=None, state=None, failure=None):
        """Draw plate-solve annotations over the thumbnail (data: AnnotationData,
        None for none) and a badge when the image isn't plate solved (state:
        PlateSolveQueue.state(), None for no badge). failure: why its plate
        solve failed, for the tooltip."""
        self._annotations = (data, layers or {}, state)
        reason = f"\nPlate solve failed: {failure}" if state == "failed" and failure else ""
        self.setToolTip(self._tooltip + reason)
        self._show_thumbnail()

    def _show_thumbnail(self):
        """The thumbnail, annotated on a copy - the cached thumbnail stays plain."""
        if self._thumb is None:
            return
        data, layers, state = self._annotations
        badge = SOLVE_BADGES.get(state)
        if data is None and badge is None:
            self.thumbnail_label.setPixmap(self._thumb)
            return
        pixmap = QPixmap(self._thumb)
        painter = QPainter(pixmap)
        whole = QRectF(pixmap.rect())
        if data is not None:
            # Names from Medium thumbnails up
            paint_annotations(painter, data, layers, pixmap.width(), whole, whole,
                              font_px=8 + self.thumbnail_size / 100, names=self.thumbnail_size >= 150)
        if badge is not None:
            font = QFont()
            font.setPixelSize(10 if self.thumbnail_size >= 150 else 9)
            metrics = QFontMetricsF(font)
            box = QRectF(4, 0, metrics.horizontalAdvance(badge) + 8, metrics.height() + 4)
            box.moveBottom(pixmap.height() - 4)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(box, 3, 3)
            painter.setFont(font)
            painter.setPen(QColor(255, 190, 110) if state == "failed" else QColor(235, 235, 235))
            painter.drawText(box, Qt.AlignCenter, badge)
        painter.end()
        self.thumbnail_label.setPixmap(pixmap)

    def set_error(self, error_message):
        """Display error on card"""
        self.thumbnail_label.setText(f"Error:\n{error_message[:30]}")
        themed_style(self.thumbnail_label, lambda: f"""
            QLabel {{
                background-color: {COLORS['error_bg']};
                border: 1px solid {COLORS['error']};
                border-radius: 3px;
                color: {COLORS['error']};
            }}
        """)

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click - emit signal"""
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.item_data)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """Handle right-click context menu"""
        self.context_menu_requested.emit(self.item_data, event.globalPos())
        event.accept()


class DSOGalleryWindow(WindowPositionMixin, QMainWindow):
    """Main window for DSO Image Gallery"""

    WINDOW_POSITION_KEY = "DSOGallery"
    _SUPPORTED_DROP_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
    ANNOTATIONS_SETTING = "gallery_annotations"
    LAYERS_SETTING = "gallery_annotation_layers"

    def __init__(self):
        """Initialize DSO Image Gallery window"""
        super().__init__()
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowTitle("DSO Image Gallery - Cosmos Collection")
        self.resize(1200, 800)

        # Initialize data structures
        self.db_manager = DatabaseManager()
        self.all_items = []       # One per attached image (database row)
        self._files = []          # One per image file - its rows (one per object) in 'rows'
        self._objects = {}        # dsodetailid -> its rows
        self._expanded = set()    # Objects whose other images show after their card
        self.filtered_items = []  # The cards shown, in order
        self.data_loaded = False  # Track whether initial data load is complete
        self.current_columns = 1
        self.current_filters = {
            'search': '',
            'catalog': 'All',
            'type': 'All',
            'telescope': 'All',
            'camera': 'All',
            'show': SHOW_BY_OBJECT,
            'sort': 'Name (A-Z)'
        }

        # Restore previously used search/filter selections
        filter_settings = QSettings("CosmosCollection", "CosmosCollection")
        self.current_filters['search'] = filter_settings.value("gallery_filter_search", '', type=str)
        self.current_filters['catalog'] = filter_settings.value("gallery_filter_catalog", 'All', type=str)
        self.current_filters['type'] = filter_settings.value("gallery_filter_type", 'All', type=str)
        self.current_filters['telescope'] = filter_settings.value("gallery_filter_telescope", 'All', type=str)
        self.current_filters['camera'] = filter_settings.value("gallery_filter_camera", 'All', type=str)
        self.current_filters['show'] = filter_settings.value("gallery_filter_show", SHOW_BY_OBJECT, type=str)
        self.current_filters['sort'] = filter_settings.value("gallery_filter_sort", 'Name (A-Z)', type=str)

        # Thumbnail size options: Small (100), Medium (150), Large (200), Extra Large (250)
        self.thumbnail_size_options = {
            'Small': 100,
            'Medium': 150,
            'Large': 300,
            'Extra Large': 500
        }
        # Load thumbnail size from settings (default to Medium)
        settings = QSettings("CosmosCollection", "CosmosCollection")
        saved_size_name = settings.value("gallery_thumbnail_size", "Medium")
        self.thumbnail_size = self.thumbnail_size_options.get(saved_size_name, 150)

        # Thumbnail cache and thread pool
        self.thumbnail_cache = ThumbnailCache(max_size=200)
        self.thread_pool = QThreadPool.globalInstance()
        # Get thread count from user settings
        cpu_count = os.cpu_count() or 4
        default_threads = max(1, cpu_count - 2)
        thread_count = settings.value("max_threads", default_threads, type=int)
        self.thread_pool.setMaxThreadCount(thread_count)
        self.thumbnail_signals = ThumbnailSignals()
        self._thumbnail_signals_connected = False
        self.cancelled_flag = [False]  # Mutable flag for cancellation

        # Thumbnail loading progress tracking
        self.thumbnail_progress_dialog = None
        self.thumbnails_to_load = 0
        self.thumbnails_loaded = 0
        self.is_loading_thumbnails = False
        self._finished_cards = set()  # Cards counted in the progress (each once)

        # Data loader signals
        self.data_loader_signals = DataLoaderSignals()
        self.data_loader_signals.data_loaded.connect(self._on_data_loaded)
        self.data_loader_signals.load_error.connect(self._on_data_load_error)

        # Search debounce timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._apply_filters)

        # Resize debounce timer (longer delay to reduce rebuilds)
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._handle_resize)

        # The status text while resizing shows the column count instead
        self._status_before_resize = None

        # Count of outstanding QApplication.setOverrideCursor(WaitCursor) pushes
        # that still need a matching restoreOverrideCursor(). A plain boolean
        # can't survive overlapping triggers (e.g. typing quickly in the search
        # box while a previous grid populate is still batching in), so we use
        # a counter instead to keep every push paired with exactly one pop.
        self._pending_wait_cursors = 0

        # Lazy loading tracking
        self._queued_cards = set()  # Cards whose thumbnails have been queued
        self.scroll_debounce_timer = QTimer()
        self.scroll_debounce_timer.setSingleShot(True)
        self.scroll_debounce_timer.timeout.connect(self._on_scroll_debounced)
        self.visible_buffer_rows = 2  # Load this many extra rows above/below visible area

        # Annotations on plate-solved thumbnails (the gallery's own setting and layers)
        self.annotations_enabled = settings.value(self.ANNOTATIONS_SETTING, False, type=bool)
        self.annotation_layers = {key: settings.value(f"{self.LAYERS_SETTING}/{key}", default, type=bool)
                                  for key, default in DEFAULT_ANNOTATION_LAYERS.items()}
        self._cards_by_path = {}       # image path -> its cards
        self._changed_paths = set()    # Images whose annotations or solve state changed, to redraw
        self._solving_name = ""        # The shown image being plate solved
        self._solve_progress = ""      # ...and what its plate solve is doing
        self._run_results = [0, 0]     # Solved, failed - since the status bar last showed
        self._activity_hide = QTimer(self)  # Hides the status bar a moment after the wrap-up shows
        self._activity_hide.setSingleShot(True)
        self._activity_hide.setInterval(5000)
        self._activity_hide.timeout.connect(self._hide_activity)
        self._annotation_timer = QTimer(self)
        self._annotation_timer.setSingleShot(True)
        self._annotation_timer.setInterval(150)
        self._annotation_timer.timeout.connect(self._redraw_changed_annotations)
        plate_solve_queue().changed.connect(self._on_annotation_changed)
        plate_solve_queue().progress.connect(self._on_solve_progress)
        plate_solve_queue().finished.connect(self._on_solve_finished)

        # Enable drag-and-drop of image files from file manager
        self.setAcceptDrops(True)

        # Initialize UI
        self._init_ui()

        # Setup window position persistence
        self.setup_window_position()

        # Load data in background (defer grid population until data is loaded)
        self._initial_load_pending = True
        self._start_background_data_load()

    def _init_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Controls: one panel, a row of filters (which images show) over a row
        # of view options (how they show) - the window title names the gallery
        controls = QFrame()
        controls.setObjectName("galleryControls")
        themed_style(controls, lambda: f"""
            QFrame#galleryControls {{
                background-color: {COLORS['background_lighter']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
            }}
        """)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 6, 8, 6)
        controls_layout.setSpacing(6)
        filters_layout = QHBoxLayout()
        filters_layout.setSpacing(6)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search DSO name...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(180)
        self.search_input.setText(self.current_filters['search'])
        self.search_input.textChanged.connect(lambda: self.search_timer.start(300))
        filters_layout.addWidget(self.search_input)

        # Catalog filter
        filters_layout.addWidget(QLabel("Catalog:"))
        self.catalog_combo = QComboBox()
        self.catalog_combo.addItems(["All", "M", "NGC", "IC", "Sh2", "B", "Cr", "Mel"])
        if self.current_filters['catalog'] in [self.catalog_combo.itemText(i) for i in range(self.catalog_combo.count())]:
            self.catalog_combo.setCurrentText(self.current_filters['catalog'])
        self.catalog_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.catalog_combo)

        # Type filter
        filters_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("All")
        self.type_combo.setMinimumWidth(150)
        self.type_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.type_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.type_combo)

        # Telescope and camera filters
        self.telescope_filter = QComboBox()
        self.camera_filter = QComboBox()
        for label, combo in (("Telescope:", self.telescope_filter), ("Camera:", self.camera_filter)):
            filters_layout.addWidget(QLabel(label))
            combo.addItem("All")
            combo.setMinimumWidth(140)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.currentTextChanged.connect(self._on_filter_changed)
            filters_layout.addWidget(combo)

        # Show filter: one card per object, or every image (with the other filters, e.g. every Seestar image)
        filters_layout.addWidget(QLabel("Show:"))
        self.show_combo = QComboBox()
        self.show_combo.addItems([SHOW_BY_OBJECT, SHOW_EVERY_IMAGE])
        self.show_combo.setToolTip(f"{SHOW_BY_OBJECT}: each object's favorite image (else its newest), with a badge\n"
                                   f"to show the rest of its images that match the filters.\n"
                                   f"{SHOW_EVERY_IMAGE}: a card for each image that matches the filters.")
        if self.current_filters['show'] in (SHOW_BY_OBJECT, SHOW_EVERY_IMAGE):
            self.show_combo.setCurrentText(self.current_filters['show'])
        self.show_combo.currentTextChanged.connect(self._on_filter_changed)
        filters_layout.addWidget(self.show_combo)

        # Clear filters button
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_filters)
        filters_layout.addWidget(clear_btn)

        filters_layout.addStretch()

        # Add Image button - an action, apart from the filters
        add_image_btn = QPushButton("Add Image")
        add_image_btn.clicked.connect(lambda: import_image(self))
        filters_layout.addWidget(add_image_btn)
        controls_layout.addLayout(filters_layout)

        view_layout = QHBoxLayout()
        view_layout.setSpacing(6)

        # Sort dropdown
        view_layout.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name (A-Z)", "Name (Z-A)", "Date Added (Newest)", "Date Added (Oldest)", "Type", "Constellation"])
        self.sort_combo.setMinimumWidth(130)
        if self.current_filters['sort'] in [self.sort_combo.itemText(i) for i in range(self.sort_combo.count())]:
            self.sort_combo.setCurrentText(self.current_filters['sort'])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        view_layout.addWidget(self.sort_combo)

        # Thumbnail size selector
        view_layout.addWidget(QLabel("Thumbnail Size:"))
        self.size_combo = QComboBox()
        for size_name in self.thumbnail_size_options.keys():
            self.size_combo.addItem(size_name)
        # Set current selection based on loaded setting
        current_size_name = [k for k, v in self.thumbnail_size_options.items() if v == self.thumbnail_size]
        if current_size_name:
            self.size_combo.setCurrentText(current_size_name[0])
        self.size_combo.currentTextChanged.connect(self._on_thumbnail_size_changed)
        view_layout.addWidget(self.size_combo)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        view_layout.addSpacing(6)
        view_layout.addWidget(divider)
        view_layout.addSpacing(6)

        # Annotations: the objects in each plate-solved image, over its thumbnail
        self.annotations_check = QCheckBox("Show Annotations")
        self.annotations_check.setToolTip("Mark the objects in each plate-solved image.\n"
                                          "Names show from Medium thumbnails up.")
        self.annotations_check.setChecked(self.annotations_enabled)
        self.annotations_check.toggled.connect(self._on_annotations_toggled)
        view_layout.addWidget(self.annotations_check)
        self.layers_button = QToolButton()
        self.layers_button.setText("Layers")
        self.layers_button.setToolTip("What the annotations show")
        self.layers_button.setPopupMode(QToolButton.InstantPopup)
        layers_menu = QMenu(self.layers_button)
        self.layer_actions = {}
        for key, label in ANNOTATION_LAYERS:
            action = layers_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.annotation_layers[key])
            action.toggled.connect(self._on_annotation_layers_changed)
            self.layer_actions[key] = action
        # Dark nebulae are among the DSOs
        self.layer_actions["dark"].setEnabled(self.annotation_layers["dsos"])
        self.layer_actions["dsos"].toggled.connect(self.layer_actions["dark"].setEnabled)
        self.layers_button.setMenu(layers_menu)
        view_layout.addWidget(self.layers_button)
        self.solve_summary = QLabel()
        themed_style(self.solve_summary, lambda: f"color: {COLORS['text_secondary']};")
        # Changing text mustn't change the window's minimum width (that made
        # resizing jump) - it's cut short when there's no room (full text in the tooltip)
        self.solve_summary.setMinimumWidth(1)
        view_layout.addWidget(self.solve_summary)
        self.solve_button = QPushButton("Plate Solve Unsolved…")
        self.solve_button.setToolTip("Plate solve the shown images that aren't solved yet, in the background,\n"
                                     "so they can be annotated")
        self.solve_button.clicked.connect(self._solve_unsolved)
        view_layout.addWidget(self.solve_button)
        view_layout.addStretch()

        # Status (how many images show, loading progress)
        self.status_label = QLabel("Loading...")
        themed_style(self.status_label, lambda: f"color: {COLORS['text_secondary']};")
        self.status_label.setMinimumWidth(1)  # As the summary: its text changes while resizing
        view_layout.addWidget(self.status_label)
        controls_layout.addLayout(view_layout)
        main_layout.addWidget(controls)

        # Scroll area for grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for grid
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)

        self.scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll_area)

        # Status bar - only while plate solving or star name lookups run
        self.activity_label = QLabel()
        self.activity_label.setMinimumWidth(1)  # Long messages are cut short, not widening the window
        self.activity_label.setContentsMargins(6, 0, 0, 0)
        self.activity_busy = QProgressBar()
        self.activity_busy.setRange(0, 0)  # Busy, no percentage
        self.activity_busy.setTextVisible(False)
        self.activity_busy.setFixedSize(110, 12)
        self.activity_stop = QPushButton("Stop")
        self.activity_stop.setToolTip("Stop plate solving the images that are still waiting")
        self.activity_stop.clicked.connect(lambda: plate_solve_queue().stop(list(self._cards_by_path)))
        self.statusBar().addWidget(self.activity_label, 1)
        self.statusBar().addPermanentWidget(self.activity_busy)
        self.statusBar().addPermanentWidget(self.activity_stop)
        self.statusBar().hide()
        self._update_solve_summary()

        # Connect scroll bar to lazy loading
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)


    def _start_background_data_load(self):
        """Start loading gallery data in background thread"""
        loader = DataLoaderRunnable(self.data_loader_signals)
        self.thread_pool.start(loader)

    def _on_data_loaded(self, items):
        """Handle data loaded from background thread"""
        self.all_items = items
        self._group_images()
        self.data_loaded = True  # Mark data as loaded

        # Populate filter dropdowns (restores saved type/telescope/camera selection if still valid)
        self._populate_type_filter()
        self._populate_equipment_filters()

        # Apply restored search/filter/sort state now that data and options are available
        self._filter_and_sort_items()

        # Update status
        count = len(self._files)
        self.status_label.setText(f"Loaded {count} image{'s' if count != 1 else ''}")

        # Populate grid if window is already shown
        if not self._initial_load_pending:
            self._populate_grid()

    def _group_images(self):
        """Each image file once - an image of M 31 and M 32 is one file, named
        "M 31 & M 32" (as in the Collage Builder) - and each object's rows."""
        from CollageBuilder import join_names
        self._files, self._objects, by_file = [], {}, {}
        for row in self.all_items:
            key = _file_key(row['image_path']) or f"id:{row['imageid']}"
            entry = by_file.get(key)
            if entry is None:
                entry = by_file[key] = dict(row, rows=[])
                self._files.append(entry)
            elif any(other['dsodetailid'] == row['dsodetailid'] for other in entry['rows']):
                continue  # The same file attached twice (paths spelled differently)
            entry['rows'].append(row)
            self._objects.setdefault(row['dsodetailid'], []).append(row)
        for entry in self._files:
            rows = entry['rows']
            if len(rows) > 1:
                entry['name'] = join_names(_main_name(row['name']) for row in rows)
                entry['friendly_type'] = join_names(row['friendly_type'] for row in rows)
                entry['is_favorite'] = any(row['is_favorite'] for row in rows)
                entry['collages'] = list(dict.fromkeys(name for row in rows for name in row['collages']))

    def _on_data_load_error(self, error_message):
        """Handle error loading data from background thread"""
        self.status_label.setText(f"Error loading gallery: {error_message}")
        QMessageBox.critical(self, "Error", f"Failed to load DSO gallery:\n{error_message}")

    def _populate_type_filter(self):
        """Populate type filter dropdown with unique types from loaded data"""
        # Get unique friendly types
        types = set()
        for item in self.all_items:
            if item['friendly_type']:
                types.add(item['friendly_type'])

        # Sort and add to combo box
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        self.type_combo.addItem("All")
        for dso_type in sorted(types):
            self.type_combo.addItem(dso_type)

        # Restore saved filter selection if it still exists among the options
        saved_type = self.current_filters.get('type', 'All')
        if saved_type in [self.type_combo.itemText(i) for i in range(self.type_combo.count())]:
            self.type_combo.setCurrentText(saved_type)
        else:
            self.current_filters['type'] = 'All'
        self.type_combo.blockSignals(False)

        # Ensure dropdown view is wide enough to show full text
        self.type_combo.view().setMinimumWidth(self.type_combo.minimumSizeHint().width())

    def _populate_equipment_filters(self):
        """The telescope and camera filters: those the loaded images were taken with"""
        for combo, key in ((self.telescope_filter, 'telescope'), (self.camera_filter, 'camera')):
            names = sorted({item[key] for item in self.all_items if item[key]}, key=str.casefold)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            combo.addItems(names)

            # Restore saved filter selection if it still exists among the options
            if self.current_filters.get(key, 'All') in names:
                combo.setCurrentText(self.current_filters[key])
            else:
                self.current_filters[key] = 'All'
            combo.blockSignals(False)

            # Ensure dropdown view is wide enough to show full text
            combo.view().setMinimumWidth(combo.minimumSizeHint().width())

    def _grid_width(self):
        """Width the cards can use: the scroll area's, less a scrollbar whether
        or not one shows - so the column count can't flip back and forth as
        the scrollbar comes and goes with the number of rows."""
        return (self.scroll_area.width() - 2 * self.scroll_area.frameWidth()
                - self.scroll_area.verticalScrollBar().sizeHint().width())

    def _calculate_grid_columns(self):
        """How many cards fit across: n cards take n widths and n - 1 gaps."""
        card_width = self.thumbnail_size + 20  # GalleryCard's fixed width
        spacing = self.grid_layout.spacing()
        margins = self.grid_layout.contentsMargins()
        available = self._grid_width() - margins.left() - margins.right()
        return max(1, (available + spacing) // (card_width + spacing))

    def _push_wait_cursor(self):
        """Push a WaitCursor and track it so it's always paired with a restore."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._pending_wait_cursors += 1

    def _pop_wait_cursor(self):
        """Restore a previously pushed WaitCursor, if one is still outstanding."""
        if self._pending_wait_cursors > 0:
            QApplication.restoreOverrideCursor()
            self._pending_wait_cursors -= 1

    def _populate_grid(self):
        """Populate grid with gallery cards"""
        # Close any existing progress dialog
        self._close_thumbnail_progress()
        self.is_loading_thumbnails = False

        # Disable updates during grid rebuild to prevent excessive repainting
        self.grid_container.setUpdatesEnabled(False)

        # Clear existing cards
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards_by_path = {}
        self.cards = []
        self._update_solve_summary()

        # Check if there are items to display
        if not self.filtered_items:
            # Show different message if still loading data vs no results
            if not self.data_loaded:
                # Still loading initial data
                loading_label = QLabel("Loading DSO images from database...")
                themed_style(loading_label, lambda: f"font-size: {font_px(14)}; color: {COLORS['text_secondary']}; padding: 50px;")
            elif len(self.all_items) == 0:
                # Data loaded but database has no images
                loading_label = QLabel("No images in your database.\n\nAdd images to DSO objects to see them here.")
                themed_style(loading_label, lambda: f"font-size: {font_px(14)}; color: {COLORS['text_disabled']}; padding: 50px;")
            else:
                # Data loaded but no matches for current filters
                loading_label = QLabel("No DSO images found matching your filters")
                themed_style(loading_label, lambda: f"font-size: {font_px(14)}; color: {COLORS['text_disabled']}; padding: 50px;")
            loading_label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(loading_label, 0, 0)

            if self.all_items:
                self.status_label.setText(f"Showing 0 of {len(self._files)} images")

            # Re-enable updates
            self.grid_container.setUpdatesEnabled(True)
            return

        # Calculate columns based on window width
        cols = self._calculate_grid_columns()
        self.current_columns = cols

        # Show wait cursor during loading if many items
        showing = len(self.filtered_items)
        cursor_pushed = showing > 50
        if cursor_pushed:
            self._push_wait_cursor()

        # Set up progress tracking for thumbnail loading
        self.thumbnails_to_load = showing
        self.thumbnails_loaded = 0
        self.is_loading_thumbnails = True
        self._finished_cards = set()

        # Create and show progress dialog for loading thumbnails
        if showing > 0:
            self.thumbnail_progress_dialog = QProgressDialog(
                f"Loading thumbnails... (0/{showing})",
                "Hide",  # Thumbnails keep loading, with the progress in the status bar
                0,
                showing,
                self
            )
            self.thumbnail_progress_dialog.canceled.connect(self._hide_thumbnail_progress)
            self.thumbnail_progress_dialog.setWindowTitle("Loading Gallery")
            self.thumbnail_progress_dialog.setWindowModality(Qt.WindowModal)
            self.thumbnail_progress_dialog.setMinimumDuration(500)  # Only show if takes > 500ms
            self.thumbnail_progress_dialog.setMinimumWidth(400)
            self.thumbnail_progress_dialog.setMinimumHeight(120)
            themed_style(self.thumbnail_progress_dialog, lambda: f"""
                QProgressDialog {{
                    font-size: {font_size(12)};
                }}
                QProgressBar {{
                    min-height: 25px;
                    font-size: {font_size(11)};
                }}
                QLabel {{
                    font-size: {font_size(12)};
                }}
            """)
            self.thumbnail_progress_dialog.setValue(0)
            QApplication.processEvents()  # Ensure dialog can be displayed

        # Update status immediately
        self.status_label.setText(f"Loading {showing} card{'s' if showing != 1 else ''}...")

        # Create cards in batches to keep UI responsive
        self._create_cards_batch(0, cols, cursor_pushed=cursor_pushed)

    def _make_card(self, item):
        """A card for an item of filtered_items, at the current thumbnail size."""
        expanded = item.get('role') == 'stack' and item['dsodetailid'] in self._expanded
        card = GalleryCard(item, thumbnail_size=self.thumbnail_size, expanded=expanded)
        card.double_clicked.connect(self._on_card_double_clicked)
        card.context_menu_requested.connect(self._show_card_context_menu)
        card.stack_toggled.connect(lambda item: self._toggle_stack(item['dsodetailid']))
        self._cards_by_path.setdefault(item['image_path'], []).append(card)
        return card

    def _create_cards_batch(self, start_idx, cols, batch_size=15, cursor_pushed=False):
        """Create a batch of gallery cards to keep UI responsive"""
        end_idx = min(start_idx + batch_size, len(self.filtered_items))

        # Create cards for this batch
        cols = self.current_columns  # A resize may have changed it since the first batch
        for idx in range(start_idx, end_idx):
            card = self._make_card(self.filtered_items[idx])
            self.grid_layout.addWidget(card, idx // cols, idx % cols)
            self.cards.append(card)

        # Update progress in status bar
        total = len(self.filtered_items)
        self.status_label.setText(f"Loading gallery... {end_idx}/{total}")

        # If there are more cards to create, schedule next batch
        # Use QTimer.singleShot(0, ...) to allow UI events to process between batches
        if end_idx < len(self.filtered_items):
            QTimer.singleShot(0, lambda: self._create_cards_batch(end_idx, cols, batch_size, cursor_pushed))
        else:
            # All cards created - finalize grid layout
            self.grid_layout.setRowStretch(len(self.filtered_items) // cols + 1, 1)
            self.grid_layout.setColumnStretch(cols, 1)

            # Re-enable updates now that grid is built
            self.grid_container.setUpdatesEnabled(True)

            # Restore cursor if this batch chain pushed one
            if cursor_pushed:
                self._pop_wait_cursor()

            # Update status
            self.status_label.setText(self._showing_text())

            # Load thumbnails in background
            self._load_thumbnails()
            self._update_solve_summary()

    def _load_thumbnails(self):
        """Load all thumbnails, prioritizing visible cards first"""
        # Cancel any pending tasks (they will check the flag and exit early)
        self.cancelled_flag[0] = True
        # Create new cancellation flag for new batch of tasks
        self.cancelled_flag = [False]

        # Reset tracking for new grid
        self._queued_cards.clear()

        # Connect signals (disconnect first to avoid duplicates). Only attempt
        # disconnect if we know we're connected - PySide6 emits a
        # RuntimeWarning (not a catchable exception) when disconnecting a
        # signal with no matching connection, so a try/except can't suppress it.
        if self._thumbnail_signals_connected:
            self.thumbnail_signals.thumbnail_ready.disconnect(self._on_thumbnail_ready)
            self.thumbnail_signals.thumbnail_error.disconnect(self._on_thumbnail_error)

        self.thumbnail_signals.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumbnail_signals.thumbnail_error.connect(self._on_thumbnail_error)
        self._thumbnail_signals_connected = True

        # Load visible thumbnails first (priority)
        self._load_visible_thumbnails()

        # Schedule loading of remaining thumbnails after visible ones are queued
        QTimer.singleShot(200, self._load_remaining_thumbnails)

    def _get_visible_card_indices(self):
        """Calculate which card indices are currently visible in the viewport"""
        if not hasattr(self, 'cards') or not self.cards or self.current_columns == 0:
            return set()

        # Get scroll area viewport geometry
        viewport = self.scroll_area.viewport()
        viewport_height = viewport.height()
        scroll_pos = self.scroll_area.verticalScrollBar().value()

        # Estimate card height (thumbnail + name label ~20 + type label ~16 + margins ~30 + spacing)
        card_height = self.thumbnail_size + 70  # Approximate height of each card including spacing

        # Calculate visible row range
        first_visible_row = max(0, scroll_pos // card_height - self.visible_buffer_rows)
        last_visible_row = (scroll_pos + viewport_height) // card_height + self.visible_buffer_rows

        # Convert rows to card indices
        visible_indices = set()
        total_cards = len(self.cards)

        for row in range(first_visible_row, last_visible_row + 1):
            for col in range(self.current_columns):
                idx = row * self.current_columns + col
                if 0 <= idx < total_cards:
                    visible_indices.add(idx)

        return visible_indices

    def _queue_thumbnail(self, card):
        """Load a card's thumbnail in the background, unless it's already queued."""
        if card in self._queued_cards:
            return
        self._queued_cards.add(card)
        self.thread_pool.start(ThumbnailRunnable(card, card.item_data['image_path'], self.thumbnail_cache,
                                                 self.thumbnail_signals, self.cancelled_flag, self.thumbnail_size))

    def _load_visible_thumbnails(self):
        """Load thumbnails for currently visible cards that haven't been loaded yet"""
        for idx in self._get_visible_card_indices():
            self._queue_thumbnail(self.cards[idx])

    def _load_remaining_thumbnails(self, batch_start=0, batch_size=20):
        """Load remaining thumbnails that weren't in the initial visible set"""
        if not hasattr(self, 'cards') or not self.cards:
            return

        # Check if cancelled (grid was rebuilt)
        if self.cancelled_flag[0]:
            return

        total_cards = len(self.cards)
        batch_end = min(batch_start + batch_size, total_cards)

        # Queue thumbnails for this batch (skip already queued ones)
        for idx in range(batch_start, batch_end):
            self._queue_thumbnail(self.cards[idx])

        # Schedule next batch if there are more cards
        if batch_end < total_cards:
            # Small delay between batches to keep UI responsive
            QTimer.singleShot(50, lambda: self._load_remaining_thumbnails(batch_end, batch_size))

    def _on_scroll(self, value):
        """Handle scroll events - debounce and trigger lazy loading"""
        # Use debounce to avoid excessive loading during fast scrolling
        self.scroll_debounce_timer.start(100)  # 100ms debounce for smoother scrolling

    def _on_scroll_debounced(self):
        """Handle debounced scroll - load newly visible thumbnails"""
        self._load_visible_thumbnails()

    def _is_current_card(self, card):
        """Whether a thumbnail result is for a card in the grid now - not one
        from before a rebuild, whose result was already on its way."""
        return card in self._cards_by_path.get(card.item_data['image_path'], ())

    def _on_thumbnail_ready(self, card, image):
        """Handle thumbnail loaded successfully"""
        if not self._is_current_card(card):
            return
        try:
            card.set_thumbnail(QPixmap.fromImage(image))  # Pixmaps belong on the UI thread
            self._update_card_annotations(card)
        except RuntimeError:
            # Card was deleted, ignore
            pass

        self._update_thumbnail_progress(card)

    def _on_thumbnail_error(self, card, error_message):
        """Handle thumbnail load error"""
        if not self._is_current_card(card):
            return
        try:
            card.set_error(error_message)
        except RuntimeError:
            # Card was deleted, ignore
            pass

        self._update_thumbnail_progress(card)

    def _showing_text(self):
        """Status text for how many of the images are shown (counting those
        behind each object's card)"""
        files = {_file_key(row['image_path']) for card in self.filtered_items for row in card.get('stack') or [card]}
        total = len(self._files)
        images = (f"all {total} image{'s' if total != 1 else ''}" if len(files) == total
                  else f"{len(files)} of {total} images")
        if not self._by_object():
            return f"Showing {images}"
        objects = sum(card.get('role') == 'stack' for card in self.filtered_items)
        return f"Showing {objects} object{'s' if objects != 1 else ''} · {images}"

    def _set_status(self, text):
        """Show status text - kept for after the resize if one is showing the column count"""
        if self._status_before_resize is not None:
            self._status_before_resize = text
        else:
            self.status_label.setText(text)

    def _update_thumbnail_progress(self, card):
        """Update the thumbnail loading progress dialog"""
        if not self.is_loading_thumbnails or card in self._finished_cards:
            return

        self._finished_cards.add(card)
        self.thumbnails_loaded += 1
        progress = f"Loading thumbnails... ({self.thumbnails_loaded}/{self.thumbnails_to_load})"

        # Update dialog if it exists and is visible
        if self.thumbnail_progress_dialog is not None:
            try:
                self.thumbnail_progress_dialog.setValue(self.thumbnails_loaded)
                self.thumbnail_progress_dialog.setLabelText(progress)
            except (RuntimeError, AttributeError):
                # Dialog was deleted or not fully initialized
                pass
        elif self._grid_built():
            # The dialog was hidden: show the progress in the status bar instead
            self._set_status(f"{self._showing_text()} · {progress}")

        # Check if all thumbnails are complete
        if self.thumbnails_loaded >= self.thumbnails_to_load:
            self._finish_thumbnail_loading()

    def _hide_thumbnail_progress(self):
        """The progress dialog's Hide button: keep loading, with the progress in the status bar"""
        self.thumbnail_progress_dialog = None
        if self._grid_built():
            self._set_status(f"{self._showing_text()} · Loading thumbnails... "
                             f"({self.thumbnails_loaded}/{self.thumbnails_to_load})")

    def _close_thumbnail_progress(self):
        """Close the progress dialog - without it counting as Hide (closing emits canceled)"""
        dialog, self.thumbnail_progress_dialog = self.thumbnail_progress_dialog, None
        if dialog is not None:
            dialog.canceled.disconnect(self._hide_thumbnail_progress)
            dialog.close()

    def _finish_thumbnail_loading(self):
        """Close the thumbnail progress dialog and clean up"""
        was_hidden = self.thumbnail_progress_dialog is None
        self.is_loading_thumbnails = False
        self.thumbnails_to_load = 0
        self.thumbnails_loaded = 0

        self._close_thumbnail_progress()
        if was_hidden and self._grid_built():
            self._set_status(self._showing_text())

    def _on_card_double_clicked(self, item_data):
        """Open the image in the Image Viewer - which steps through its object's
        other images (those the filters show) too"""
        versions = [row['image_path'] for row in self._object_images(item_data['dsodetailid'])]
        self.image_viewer = open_image_viewer(self, item_data['name'], item_data['image_path'],
                                              item_data.get('ra_deg'), item_data.get('dec_deg'), versions)

    def _open_dso_details(self, item_data):
        """Open DSO detail window for the selected object"""
        open_dso_details(self, item_data['dsodetailid'], item_data['name'])

    def _show_card_context_menu(self, item_data, position):
        """Show context menu when right-clicking on a card"""
        context_menu = QMenu(self)

        # Add menu actions
        view_action = context_menu.addAction("View Full Image")
        view_action.triggered.connect(lambda: self._on_card_double_clicked(item_data))

        # An image of several objects: each one's details
        rows = item_data.get('rows') or [item_data]
        if len(rows) > 1:
            details_menu = context_menu.addMenu("View DSO Details")
            for row in rows:
                details_menu.addAction(row['name'], lambda checked=False, row=row: self._open_dso_details(row))
        else:
            details_action = context_menu.addAction("View DSO Details")
            details_action.triggered.connect(lambda: self._open_dso_details(item_data))

        # An object's other images, shown after its card
        key = item_data['dsodetailid']
        count = len(item_data.get('stack') or [])
        if item_data.get('role') == 'version':
            context_menu.addAction("Hide the Object's Other Images", lambda: self._toggle_stack(key))
        elif count > 1:
            label = f"Hide the Other {count - 1} Images" if key in self._expanded else f"Show All {count} Images Here"
            context_menu.addAction(label, lambda: self._toggle_stack(key))

        # Annotations need a plate solve
        if plate_solve_queue().state(item_data['image_path']) in ("unsolved", "failed"):
            context_menu.addAction("Plate Solve (for Annotations)", lambda: self._plate_solve([item_data]))

        # Collages: add this image to one, or open one that shows it
        if item_data.get('imageid') is not None:
            from CollageBuilder import add_collage_actions
            context_menu.addSeparator()
            add_collage_actions(context_menu, self, [item_data['imageid']])

        # Show menu at cursor position
        context_menu.exec(position)

    # Annotations

    def _on_annotations_toggled(self, enabled):
        self.annotations_enabled = enabled
        QSettings("CosmosCollection", "CosmosCollection").setValue(self.ANNOTATIONS_SETTING, enabled)
        self._annotate_all_cards()

    def _on_annotation_layers_changed(self):
        self.annotation_layers = {key: action.isChecked() for key, action in self.layer_actions.items()}
        settings = QSettings("CosmosCollection", "CosmosCollection")
        for key, value in self.annotation_layers.items():
            settings.setValue(f"{self.LAYERS_SETTING}/{key}", value)
        self._annotate_all_cards()

    def _annotate_all_cards(self):
        for cards in self._cards_by_path.values():
            for card in cards:
                self._update_card_annotations(card)
        self._update_solve_summary()

    def _update_card_annotations(self, card):
        """Show (or hide) a card's annotations and plate solve badge."""
        try:
            if not self.annotations_enabled:
                card.set_annotations(None)
                return
            path = card.item_data['image_path']
            queue = plate_solve_queue()
            data = annotation_store().get(path, want_stars=self.annotation_layers["stars"])
            card.set_annotations(data, self.annotation_layers, queue.state(path), queue.failure(path))
        except RuntimeError:
            pass  # The card was deleted (grid rebuilt)

    def _on_annotation_changed(self, path):
        """An image's annotations loaded, or its plate solve state changed -
        redraw it soon (many arrive together)."""
        if self.annotations_enabled and path in self._cards_by_path:
            self._changed_paths.add(path)
            self._annotation_timer.start()

    def _redraw_changed_annotations(self):
        paths, self._changed_paths = self._changed_paths, set()
        for path in paths:
            for card in self._cards_by_path.get(path, []):
                self._update_card_annotations(card)
        self._update_solve_summary()

    def _shown_paths(self):
        return list(dict.fromkeys(item['image_path'] for item in self.filtered_items))

    def _update_solve_summary(self):
        """How many of the shown images are plate solved, and the button to solve the rest."""
        enabled = self.annotations_enabled
        for widget in (self.layers_button, self.solve_summary):
            widget.setVisible(enabled)
        self._update_activity()
        if not enabled or not self.filtered_items:
            self.solve_summary.setText("")
            self.solve_button.setVisible(False)
            return
        queue = plate_solve_queue()
        paths = self._shown_paths()
        states = [queue.state(path) for path in paths]
        waiting = states.count("solving") + states.count("queued")
        unsolved = states.count("unsolved") + states.count("failed")
        text = f"{states.count('solved')} of {len(paths)} shown image{'s' if len(paths) != 1 else ''} plate solved"
        if states.count("checking"):
            text += " · checking…"
        self.solve_summary.setText(text)
        self.solve_summary.setToolTip(text)
        self.solve_button.setVisible(bool(waiting or unsolved))
        self.solve_button.setText("Stop Plate Solving" if waiting else f"Plate Solve {unsolved} Unsolved…")

    def _solve_unsolved(self):
        queue = plate_solve_queue()
        paths = self._shown_paths()
        if queue.waiting(paths):  # The button reads "Stop Plate Solving"
            queue.stop(paths)
            return
        items, seen = [], set()
        for item in self.filtered_items:
            path = item['image_path']
            if path not in seen and queue.state(path) in ("unsolved", "failed"):
                items.append(item)
                seen.add(path)
        if not items:
            return
        reply = QMessageBox.question(
            self, "Plate Solve",
            f"Plate solve {len(items)} image{'s' if len(items) != 1 else ''}?\n\n"
            f"{describe_solving()} You can keep browsing, and stop at any time.")
        if reply == QMessageBox.Yes:
            self._plate_solve(items)

    def _update_activity(self):
        """The status bar while shown images are plate solved or have their star
        names looked up: what's happening, with a busy bar (and Stop while
        solving). When it's all done, a wrap-up for a moment, then it hides."""
        paths = list(self._cards_by_path)
        waiting = plate_solve_queue().waiting(paths)
        parts = []
        if waiting:
            text = f"Plate solving {self._solving_name or 'an image'}"
            if waiting > 1:
                text += f" ({waiting - 1} more waiting)"
            if self._solve_progress:
                text += f" - {self._solve_progress}"
            parts.append(text)
        if self.annotations_enabled and self.annotation_layers["stars"]:
            done, total = annotation_store().star_progress(paths)
            if done < total:
                parts.append(f"Looking up star names online: {done} of {total}…")
        bar = self.statusBar()
        self.activity_stop.setVisible(bool(waiting))
        if parts:
            self._activity_hide.stop()
            text = "     ·     ".join(parts)
            self.activity_label.setText(text)
            self.activity_label.setToolTip(text)
            self.activity_busy.show()
            bar.show()
        elif bar.isVisible() and not self._activity_hide.isActive():
            solved, failed = self._run_results
            if solved or failed:
                text = f"Plate solving finished: {solved} solved"
                if failed:
                    text += f", {failed} couldn't be solved (see their thumbnails' tooltips)"
                self.activity_label.setText(text)
                self.activity_label.setToolTip(text)
                self.activity_busy.hide()
                self._activity_hide.start()
            else:
                self._hide_activity()

    def _hide_activity(self):
        self.statusBar().hide()
        self._run_results = [0, 0]

    def _plate_solve(self, items):
        """Queue images for plate solving, near their objects' positions."""
        for item in items:
            ra, dec = item.get('ra_deg'), item.get('dec_deg')
            hints = {"ra": ra, "dec": dec, "radius": 15} if ra is not None and dec is not None else None
            name = (item.get('name') or "").split(", ")[0]  # Its main designation ("M 8"), for the status bar
            plate_solve_queue().add(item['image_path'], hints, name)
        self._update_solve_summary()

    def _on_solve_progress(self, path, name, message):
        if path in self._cards_by_path:
            self._solving_name, self._solve_progress = name, message
            self._update_activity()

    def _on_solve_finished(self, path, solved, failure, name):
        if path in self._cards_by_path:
            self._run_results[0 if solved else 1] += 1
            self._solving_name = self._solve_progress = ""
            self._update_solve_summary()

    def _on_filter_changed(self):
        """Handle filter combo box changes"""
        self._apply_filters()

    def _on_thumbnail_size_changed(self, size_name):
        """Handle thumbnail size selector change"""
        new_size = self.thumbnail_size_options.get(size_name, 150)
        if new_size != self.thumbnail_size:
            self.thumbnail_size = new_size

            # Save setting
            settings = QSettings("CosmosCollection", "CosmosCollection")
            settings.setValue("gallery_thumbnail_size", size_name)

            # The memory cache is kept - it's keyed by size, so switching back is instant
            self._queued_cards.clear()

            # Cancel pending thumbnail tasks
            self.cancelled_flag[0] = True

            # Repopulate grid with new size (progress dialog shown by _populate_grid)
            self._populate_grid()

    def _apply_filters(self):
        """Apply all filters and refresh grid"""
        # Update current filter state
        self.current_filters['search'] = self.search_input.text().strip()
        self.current_filters['catalog'] = self.catalog_combo.currentText()
        self.current_filters['type'] = self.type_combo.currentText()
        self.current_filters['telescope'] = self.telescope_filter.currentText()
        self.current_filters['camera'] = self.camera_filter.currentText()
        self.current_filters['show'] = self.show_combo.currentText()
        self.current_filters['sort'] = self.sort_combo.currentText()

        # Persist filter/search state so it's remembered next time the gallery opens
        self._save_filter_settings()

        self._filter_and_sort_items()

        # Refresh grid
        self._populate_grid()

    def _save_filter_settings(self):
        """Persist current search/filter/sort state to settings"""
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue("gallery_filter_search", self.current_filters['search'])
        settings.setValue("gallery_filter_catalog", self.current_filters['catalog'])
        settings.setValue("gallery_filter_type", self.current_filters['type'])
        settings.setValue("gallery_filter_telescope", self.current_filters['telescope'])
        settings.setValue("gallery_filter_camera", self.current_filters['camera'])
        settings.setValue("gallery_filter_show", self.current_filters['show'])
        settings.setValue("gallery_filter_sort", self.current_filters['sort'])

    def _by_object(self):
        return self.current_filters['show'] != SHOW_EVERY_IMAGE

    def _filter_and_sort_items(self):
        """The cards to show (filtered_items), sorted: one per object, its images
        that match the filters behind it - then an expanded object's other
        images follow its card. Or one per image file that matches."""
        if self._by_object():
            self.filtered_items = []
            for dsodetailid, rows in self._objects.items():
                images = self._object_images(dsodetailid) if self._object_matches(rows[0]) else []
                if images:
                    self.filtered_items.append(dict(images[0], role='stack', stack=images))
        else:
            self.filtered_items = [dict(entry, role='image') for entry in self._files if self._image_matches(entry)
                                   and any(self._object_matches(row) for row in entry['rows'])]
        self._sort_items()
        if self._by_object() and self._expanded:
            cards, self.filtered_items = self.filtered_items, []
            for card in cards:
                self.filtered_items.append(card)
                if card['dsodetailid'] in self._expanded:
                    self.filtered_items.extend(self._version_items(card))

    def _object_images(self, dsodetailid):
        """An object's images that match the filters, as its rows: its favorite
        (else its newest) first, then newest first."""
        rows = sorted((row for row in self._objects.get(dsodetailid, []) if self._image_matches(row)),
                      key=_newest, reverse=True)
        rows.sort(key=lambda row: not row['is_favorite'])  # Stable: the rest stay newest first
        return rows

    @staticmethod
    def _version_items(card):
        """Items for the other images of an object's card, shown after it."""
        return [dict(row, role='version') for row in card['stack'][1:]]

    def _toggle_stack(self, dsodetailid):
        """Show an object's other images after its card, or hide them again.
        The cards are added or removed in place - the rest keep their thumbnails."""
        expanding = dsodetailid not in self._expanded
        if expanding:
            self._expanded.add(dsodetailid)
        else:
            self._expanded.discard(dsodetailid)
        head = next((i for i, card in enumerate(getattr(self, 'cards', None) or [])
                     if card.item_data.get('role') == 'stack' and card.item_data['dsodetailid'] == dsodetailid), None)
        if head is None or not self._grid_built():  # Not shown, or the grid is still being built
            self._filter_and_sort_items()
            self._populate_grid()
            return
        head_card = self.cards[head]
        if expanding:
            items = self._version_items(head_card.item_data)
            added = [self._make_card(item) for item in items]
            self.filtered_items[head + 1:head + 1] = items
            self.cards[head + 1:head + 1] = added
            if self.is_loading_thumbnails:
                self.thumbnails_to_load += len(added)
            for card in added:
                self._queue_thumbnail(card)
        else:
            end = head + 1
            while end < len(self.cards) and self.cards[end].item_data.get('role') == 'version' \
                    and self.cards[end].item_data['dsodetailid'] == dsodetailid:
                end += 1
            removed = self.cards[head + 1:end]
            del self.filtered_items[head + 1:end]
            del self.cards[head + 1:end]
            for card in removed:
                if self.is_loading_thumbnails and card not in self._finished_cards:
                    self.thumbnails_to_load -= 1
                self._queued_cards.discard(card)
                self._cards_by_path[card.item_data['image_path']].remove(card)
                if not self._cards_by_path[card.item_data['image_path']]:
                    del self._cards_by_path[card.item_data['image_path']]
                self.grid_layout.removeWidget(card)
                card.deleteLater()
        head_card.set_expanded(expanding)
        self._arrange_cards(self.current_columns)
        self._set_status(self._showing_text())
        self._update_solve_summary()
        if self.is_loading_thumbnails and self.thumbnails_loaded >= self.thumbnails_to_load:
            self._finish_thumbnail_loading()

    def _on_sort_changed(self):
        """Handle sort dropdown change"""
        self._apply_filters()

    def _sort_items(self):
        """Sort filtered items based on current sort selection"""
        sort_option = self.current_filters['sort']

        def added(x):  # An object's card: when its newest image was added
            return max(row.get('created_date') or '' for row in x.get('stack') or [x])

        if sort_option == "Name (A-Z)":
            self.filtered_items.sort(key=lambda x: x['name'].lower())
        elif sort_option == "Name (Z-A)":
            self.filtered_items.sort(key=lambda x: x['name'].lower(), reverse=True)
        elif sort_option == "Date Added (Newest)":
            self.filtered_items.sort(key=added, reverse=True)
        elif sort_option == "Date Added (Oldest)":
            self.filtered_items.sort(key=added)
        elif sort_option == "Type":
            self.filtered_items.sort(key=lambda x: (x['friendly_type'].lower(), x['name'].lower()))
        elif sort_option == "Constellation":
            self.filtered_items.sort(key=lambda x: (x['constellation'].lower(), x['name'].lower()))

    def _object_matches(self, row):
        """Whether an image's object (a row's) matches the search, catalog and type filters"""
        # Text search (case-insensitive substring match in name)
        search_text = self.current_filters['search']
        if search_text and search_text.lower() not in row['name'].lower():
            return False

        # Catalog filter (name starts with catalog prefix)
        catalog = self.current_filters['catalog']
        if catalog != 'All':
            # Check if name starts with the catalog (e.g., "M ", "NGC ")
            if not row['name'].startswith(catalog + ' '):
                return False

        # Type filter (friendly type name match)
        type_filter = self.current_filters['type']
        if type_filter != 'All':
            if row['friendly_type'] != type_filter:
                return False

        return True

    def _image_matches(self, item):
        """Whether an image matches the telescope and camera filters"""
        return all(self.current_filters[key] in ('All', item.get(key, '')) for key in ('telescope', 'camera'))

    def _clear_filters(self):
        """Clear all filters"""
        self.search_input.clear()
        self.catalog_combo.setCurrentText("All")
        self.type_combo.setCurrentText("All")
        self.telescope_filter.setCurrentText("All")
        self.camera_filter.setCurrentText("All")
        self.show_combo.setCurrentText(SHOW_BY_OBJECT)
        self._apply_filters()

    def _refresh_gallery(self):
        """Reload gallery data from database"""
        # Clear current data
        self.all_items = []
        self.filtered_items = []
        self.data_loaded = False
        self._queued_cards.clear()

        # Cancel pending thumbnail tasks
        self.cancelled_flag[0] = True

        # Clear thumbnail cache for this item (in case image changed)
        self.thumbnail_cache.clear()

        # Update status
        self.status_label.setText("Refreshing gallery...")

        # Reload data from database
        self._start_background_data_load()

    def showEvent(self, event):
        """Handle window show - populate grid on first show"""
        super().showEvent(event)
        if self._initial_load_pending:
            self._initial_load_pending = False
            # Use QTimer to ensure window is fully laid out
            QTimer.singleShot(0, self._populate_grid)
        else:
            # Trigger a resize check to ensure grid fits current window
            QTimer.singleShot(0, self._handle_resize)

    def _grid_built(self):
        cards = getattr(self, 'cards', None)
        return bool(cards) and len(cards) == len(self.filtered_items)

    def _handle_resize(self):
        """Once resizing pauses: put the cards in the new number of columns."""
        if self._status_before_resize is not None:
            self.status_label.setText(self._status_before_resize)
            self._status_before_resize = None
        if self.filtered_items:
            cols = self._calculate_grid_columns()
            if cols != self.current_columns:
                self._arrange_cards(cols)

    def _arrange_cards(self, cols):
        """Move the existing cards into cols columns. Nothing is rebuilt, so
        it's quick and their thumbnails and annotations stay - and no progress
        dialog pops up mid-drag (that ended Windows' resize, snapping the
        window back)."""
        self.current_columns = cols
        cards = getattr(self, 'cards', None)
        if not cards:
            return
        layout = self.grid_layout
        self.grid_container.setUpdatesEnabled(False)
        for row in range(layout.rowCount()):
            layout.setRowStretch(row, 0)
        for col in range(layout.columnCount()):
            layout.setColumnStretch(col, 0)
        for card in cards:
            layout.removeWidget(card)
        for index, card in enumerate(cards):
            layout.addWidget(card, index // cols, index % cols)
        if self._grid_built():  # Still being built: the last batch adds these
            layout.setRowStretch(len(cards) // cols + 1, 1)
            layout.setColumnStretch(cols, 1)
        self.grid_container.setUpdatesEnabled(True)
        self._load_visible_thumbnails()

    def resizeEvent(self, event):
        """Show the column count while resizing; rearrange when it pauses."""
        super().resizeEvent(event)
        if self._grid_built():
            cols = self._calculate_grid_columns()
            card_width = self.thumbnail_size + 20
            spacing = self.grid_layout.spacing()
            margins = self.grid_layout.contentsMargins()
            needed = (cols + 1) * (card_width + spacing) - spacing + margins.left() + margins.right() \
                - self._grid_width()
            if self._status_before_resize is None:
                self._status_before_resize = self.status_label.text()
            self.status_label.setText(f"Columns: {cols} · {needed}px wider for {cols + 1}")
        self.resize_timer.start(150)

    def dragEnterEvent(self, event):
        """Accept drag if it is a single supported image file"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in self._SUPPORTED_DROP_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Open AddImageDialog with the dropped file path pre-filled"""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in self._SUPPORTED_DROP_EXTENSIONS:
                event.acceptProposedAction()
                # Deferred: a dialog shown inside dropEvent keeps the source app's drag (e.g. Explorer) blocked
                QTimer.singleShot(0, lambda: import_image(self, file_path))

    def closeEvent(self, event):
        """Handle window close - cleanup thread pool and cursor"""
        # Restore any WaitCursor pushes that never got matched with a restore
        # (e.g. window closed while a grid populate or resize was still in flight)
        while self._pending_wait_cursors > 0:
            self._pop_wait_cursor()

        # Cancel all pending thumbnail tasks
        self.cancelled_flag[0] = True
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds for tasks to finish
        super().closeEvent(event)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = DSOGalleryWindow()
    window.show()
    sys.exit(app.exec())
