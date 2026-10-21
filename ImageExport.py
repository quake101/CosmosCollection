#!/usr/bin/env python3
"""
Image Export for Cosmos Collection
Every way an image leaves the Image Viewer, in one dialog: saved to a file,
set as the desktop background, copied to the clipboard, or made ready for
AstroBin (an upload-ready copy, its description on the clipboard, an
acquisition CSV and AstroBin's upload page - AstroBin has no upload API).

Each way is a Destination. The dialog only knows that interface, so more
(Flickr, Google Photos...) are added to DESTINATIONS without changing it.

The export is rendered here too: the image at the chosen size with the
annotations and details painted on it in image pixels - sized to the image,
so the preview is what's exported.
"""

import csv
import ctypes
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QSettings, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QWidget, QLabel, QPushButton,
                               QComboBox, QCheckBox, QListWidget, QListWidgetItem, QToolButton, QSpinBox,
                               QStackedWidget, QSizePolicy, QFileDialog, QMessageBox, QTableWidget,
                               QTableWidgetItem, QHeaderView, QDialogButtonBox, QAbstractItemView)

import ImageDetails
from AnnotationOverlay import paint_annotations
from Theme import COLORS, font_size, themed_style
from WindowPositionManager import WindowPositionManager, WindowPositionMixin

logger = logging.getLogger(__name__)


ASTRO_EXTENSIONS = ('.fits', '.fit', '.fts', '.xisf')

# (key, name) - the long edge it's exported at; never larger than the image
SIZES = [("full", "Full size"), ("screen", "Screen size"), ("4096", "4096 px"),
         ("2048", "2048 px - for the web"), ("1080", "1080 px - small")]
PREVIEW_PX = 1100   # The preview's long edge (it's painted like the export, just smaller)

LAST_DESTINATION_SETTING = "image_export/destination"
MORE_OPTIONS_SETTING = "image_export/more_options"
JPEG_QUALITY_SETTING = "image_export/jpeg_quality"
DEFAULT_JPEG_QUALITY = 95

ASTROBIN_UPLOAD_URL = "https://app.astrobin.com/uploader"
ASTROBIN_FILTERS_URL = "https://app.astrobin.com/equipment/explorer/filter"
ASTROBIN_FILTER_IDS_SETTING = "image_export/astrobin/filter_ids"

# Workers that outlived their dialog. A QThread destroyed while it runs
# crashes the app, so a reference is kept until it finishes.
_orphaned_workers = set()


def _settings():
    return QSettings("CosmosCollection", "CosmosCollection")


def is_astro_file(path):
    return Path(path or "").suffix.lower() in ASTRO_EXTENSIONS


def keep_running(worker, *signals):
    """Let a worker finish after its window closes: its signals dropped and
    the thread kept alive until it ends."""
    if worker is None or not worker.isRunning():
        return
    for signal in signals:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Nothing connected
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass
    _orphaned_workers.add(worker)
    worker.finished.connect(lambda: _orphaned_workers.discard(worker))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@dataclass
class ExportOptions:
    annotations: bool = False   # Painted on (only when the image has them)
    details: bool = False       # Painted on (only when any are known)
    size: str = "full"          # A SIZES key
    quality: int = DEFAULT_JPEG_QUALITY


def long_edge(size):
    """The long edge in pixels a size asks for - None for full size."""
    if size == "full":
        return None
    if size == "screen":
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        geometry = screen.geometry()
        return round(max(geometry.width(), geometry.height()) * screen.devicePixelRatio())
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def output_size(width, height, size):
    """(width, height) the image is exported at - never larger than it is."""
    edge = long_edge(size)
    if not edge or max(width, height) <= edge:
        return width, height
    scale = edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render(image, annotation_data=None, layers=None, lines=(), style=None):
    """image (a QImage at the size it's exported at) with the annotations and
    details painted on it, as a new QImage - taller with a details banner.
    QImage only, so it's safe off the GUI thread."""
    result = image.convertToFormat(QImage.Format_RGB32)
    if annotation_data is not None:
        painter = QPainter(result)
        whole = QRectF(result.rect())
        paint_annotations(painter, annotation_data, layers, result.width(), whole, whole)
        painter.end()
    if lines:
        result = ImageDetails.compose(result, list(lines), style)
    return result


# ---------------------------------------------------------------------------
# Saving and the desktop background (worker threads)
# ---------------------------------------------------------------------------

class ImageSaveWorker(QThread):
    """Encodes and writes a QImage to disk off the GUI thread (PNG compression
    of a large image takes seconds)"""
    image_saved = Signal(bool, str)  # success, save path

    def __init__(self, image, save_path, image_format, quality=-1):
        super().__init__()
        self.image = image
        self.save_path = save_path
        self.image_format = image_format
        self.quality = quality

    def run(self):
        try:
            from ImageLoader import write_image_file
            ok = write_image_file(self.image, self.save_path, self.image_format, self.quality)
        except Exception as e:
            logger.error(f"Failed to save image to {self.save_path}: {e}", exc_info=True)
            ok = False
        self.image_saved.emit(ok, self.save_path)


class BackgroundSetterWorker(QThread):
    """Writes the image as a PNG (if given one) and sets it as the desktop
    background off the GUI thread"""
    background_set = Signal(str, str)  # level ('success'/'warning'/'error'), message

    def __init__(self, file_path, image=None):
        super().__init__()
        self.file_path = file_path
        self.image = image  # QImage to write as a PNG first, or None to use file_path directly

    def run(self):
        try:
            if self.image is not None:
                abs_path = export_background_png(self.image, self.file_path)
                if not abs_path:
                    self.background_set.emit("error", "Failed to convert image to PNG for desktop background")
                    return
            else:
                abs_path = str(Path(self.file_path).resolve())
            level, message = apply_desktop_background(abs_path)
            self.background_set.emit(level, message)
        except Exception as e:
            logger.error(f"Failed to set desktop background: {e}", exc_info=True)
            self.background_set.emit("error", f"Failed to set desktop background: {str(e)}")


def background_support():
    """Why the desktop background can't be set here - None when it can."""
    system = platform.system()
    if system in ("Windows", "Darwin"):
        return None
    if system == "Linux":
        desktop = _linux_desktop_name()
        if any(name in desktop for name in ('kde', 'plasma', 'xfce', 'cinnamon', 'mate',
                                             'gnome', 'unity', 'budgie', 'pantheon')):
            return None
        return (f"Setting the background isn't supported on this desktop ({desktop or 'unknown'}). "
                f"Supported: KDE Plasma, GNOME, Cinnamon, MATE, XFCE.")
    return f"Setting the desktop background isn't supported on {system}."


def apply_desktop_background(abs_path):
    """Set abs_path as the desktop wallpaper. Safe to call from a worker thread.
    Returns (level, message) where level is 'success', 'warning' or 'error'."""
    success = ("success", "Desktop background updated.")

    if platform.system() == "Windows":
        # Windows implementation using ctypes
        SPI_SETDESKWALLPAPER = 20
        SPIF_UPDATEINIFILE = 0x01
        SPIF_SENDCHANGE = 0x02

        # Call SystemParametersInfoW to set the wallpaper
        result = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETDESKWALLPAPER,
            0,
            abs_path,
            SPIF_UPDATEINIFILE | SPIF_SENDCHANGE
        )
        return success if result else ("error", "Failed to set desktop background")

    if platform.system() == "Darwin":
        # macOS implementation
        script = f'''
        tell application "Finder"
            set desktop picture to POSIX file "{abs_path}"
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=True)
        return success

    if platform.system() == "Linux":
        desktop = _linux_desktop_name()
        if _set_linux_background(abs_path, desktop):
            return success
        return ("warning",
                f"Could not set background on this desktop environment "
                f"({desktop or 'unknown'}).\n"
                f"Supported: KDE Plasma, GNOME, Cinnamon, MATE, XFCE.")

    return ("warning", f"Setting desktop background is not supported on {platform.system()}")


def _linux_desktop_name():
    """Return the current desktop environment in lowercase, e.g. 'kde:plasma'
    or 'ubuntu:gnome' (XDG_CURRENT_DESKTOP can hold several names separated by ':')"""
    return (os.environ.get('XDG_CURRENT_DESKTOP')
            or os.environ.get('DESKTOP_SESSION')
            or '').lower()


def _set_linux_background(abs_path, desktop):
    """Set the wallpaper using the right tool for the Linux desktop environment.
    Returns True on success, False if the desktop isn't supported or the command failed."""
    uri = Path(abs_path).as_uri()

    def run(*args):
        subprocess.run(list(args), check=True, capture_output=True, timeout=15)

    try:
        if 'kde' in desktop or 'plasma' in desktop:
            # Plasma 5.24+ ships a CLI tool for this
            try:
                run("plasma-apply-wallpaperimage", abs_path)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Older Plasma - set it through a plasmashell script over D-Bus
            script = (
                "var allDesktops = desktops();"
                "for (var i = 0; i < allDesktops.length; i++) {"
                "  var d = allDesktops[i];"
                "  d.wallpaperPlugin = 'org.kde.image';"
                "  d.currentConfigGroup = ['Wallpaper', 'org.kde.image', 'General'];"
                f"  d.writeConfig('Image', {json.dumps(uri)});"
                "}"
            )
            for qdbus in ("qdbus6", "qdbus", "qdbus-qt5"):
                try:
                    run(qdbus, "org.kde.plasmashell", "/PlasmaShell",
                        "org.kde.PlasmaShell.evaluateScript", script)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            return False

        if 'xfce' in desktop:
            # Every monitor/workspace has its own last-image property
            result = subprocess.run(["xfconf-query", "-c", "xfce4-desktop", "-l"],
                                    check=True, capture_output=True, text=True, timeout=15)
            props = [p for p in result.stdout.split() if p.endswith('/last-image')]
            if not props:
                return False
            for prop in props:
                run("xfconf-query", "-c", "xfce4-desktop", "-p", prop, "-s", abs_path)
            return True

        if 'cinnamon' in desktop:
            run("gsettings", "set", "org.cinnamon.desktop.background", "picture-uri", uri)
            return True

        if 'mate' in desktop:
            run("gsettings", "set", "org.mate.background", "picture-filename", abs_path)
            return True

        if any(name in desktop for name in ('gnome', 'unity', 'budgie', 'pantheon')):
            run("gsettings", "set", "org.gnome.desktop.background", "picture-uri", uri)
            # GNOME 42+ uses a separate key in dark mode; older versions lack it
            try:
                run("gsettings", "set", "org.gnome.desktop.background", "picture-uri-dark", uri)
            except subprocess.CalledProcessError:
                pass
            return True

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to set background on desktop '{desktop}': {e}")
        return False

    logger.warning(f"Setting background not supported on desktop '{desktop}'")
    return False


def export_background_png(image, file_path):
    """Save image (a QImage) as a PNG next to the original file for use as a
    desktop background. The file has to persist since macOS/GNOME reference it
    directly. Safe to call from a worker thread. Returns the absolute path, or
    None on failure."""
    # "_background" suffix so an existing PNG of the same name isn't overwritten
    source = Path(file_path).resolve()
    png_path = source.with_name(f"{source.stem}_background.png")
    from ImageLoader import write_image_file
    if not write_image_file(image, str(png_path), "PNG"):
        logger.error(f"Failed to save background PNG to {png_path}")
        return None

    logger.info(f"Converted {file_path} to {png_path} for desktop background")
    return str(png_path)


def choose_save_path(parent, title, default_path, jpeg_default):
    """(path, "PNG" or "JPEG") from a save dialog - (None, None) if cancelled.
    Only PNG and JPEG can be written: a name with any other extension (e.g.
    .xisf) gets the chosen filter's, so its name matches its contents."""
    png_filter = "PNG Image (*.png)"
    jpeg_filter = "JPEG Image (*.jpg *.jpeg)"
    path, selected = QFileDialog.getSaveFileName(parent, title, str(default_path), f"{png_filter};;{jpeg_filter}",
                                                 jpeg_filter if jpeg_default else png_filter)
    if not path:
        return None, None
    ext = Path(path).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        ext = '.jpg' if selected == jpeg_filter else '.png'
        path = str(Path(path).with_suffix(ext))
        if Path(path).exists() and QMessageBox.question(
                parent, "Overwrite File",
                f"{Path(path).name} already exists.\nDo you want to replace it?") != QMessageBox.Yes:
            return None, None
    return path, ("JPEG" if ext in ('.jpg', '.jpeg') else "PNG")


# ---------------------------------------------------------------------------
# AstroBin's acquisition CSV
# ---------------------------------------------------------------------------

# AstroBin's long-exposure acquisition columns (welcome.astrobin.com/importing-acquisitions-from-csv),
# the ones the session knows. number and duration are required; a column nothing fills is left out.
ACQUISITION_COLUMNS = ["date", "filter", "number", "duration", "binning", "gain", "sensorCooling",
                       "darks", "flats", "bias"]
REQUIRED_COLUMNS = {"date", "filter", "number", "duration"}


def load_filter_ids():
    """{filter key (SessionObservations.normalize_filter): AstroBin filter ID}"""
    try:
        value = json.loads(_settings().value(ASTROBIN_FILTER_IDS_SETTING, "{}", type=str) or "{}")
    except (TypeError, ValueError):
        return {}
    return {str(k): str(v) for k, v in value.items() if v} if isinstance(value, dict) else {}


def save_filter_ids(ids):
    _settings().setValue(ASTROBIN_FILTER_IDS_SETTING, json.dumps({k: v for k, v in ids.items() if v}))


def parse_filter_id(text):
    """AstroBin's filter ID from what was typed: the number, or the address of
    the filter's page on AstroBin ('.../equipment/explorer/filter/123/...')."""
    text = (text or "").strip()
    match = re.search(r"/filter/(\d+)", text)
    if match:
        return match.group(1)
    return text if text.isdigit() else ""


def _number(value, decimals):
    """'300' or '0.25' - at most decimals places, no trailing zeros."""
    return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")


def _iso_date(value):
    try:
        return datetime.fromisoformat(str(value).strip()[:10]).date().isoformat()
    except (TypeError, ValueError):
        return ""


def session_filters(conn, session_id):
    """The names of the filters a session's lights were taken with, by name."""
    import SessionObservations as so
    names = {}
    for (name,) in conn.execute("SELECT DISTINCT filter_name FROM usersessionfiles "
                                "WHERE session_id = ? AND frame_type = 'Light'", (session_id,)):
        if name:
            names.setdefault(so.normalize_filter(name), name)
    for (name,) in conn.execute("""
            SELECT DISTINCT f.filter_name FROM usersessionobservationfilters f
            JOIN usersessionobservations o ON o.id = f.observation_id WHERE o.session_id = ?""", (session_id,)):
        if name:
            names.setdefault(so.normalize_filter(name), name)
    return sorted(names.values(), key=str.lower)


def acquisition_rows(conn, session_id, filter_ids=None):
    """AstroBin acquisition rows ({column: text}) for a session: one per night,
    filter and exposure, counted as the session counts them (files or the
    hand-logged rows). Lights not on any night have no date. Gain, binning and
    sensor temperature come from the light files; each night's calibration
    files are counted once - flats on the first row of their filter, darks on
    the first of their exposure, bias on the night's first. A filter without
    an AstroBin ID is left blank, to pick on AstroBin."""
    import SessionObservations as so
    filter_ids = filter_ids or {}
    observations = conn.execute("SELECT id, night_date FROM usersessionobservations WHERE session_id = ? "
                                "ORDER BY night_date", (session_id,)).fetchall()
    breakdowns = so.load_breakdowns(conn, [obs_id for obs_id, _night in observations])

    # Each group's light settings: (observation, filter key, exposure) -> gains, binnings, temperatures
    lights = {}
    for obs_id, filter_name, exposure, gain, binning, temperature in conn.execute("""
            SELECT observation_id, filter_name, exptime_seconds, gain, xbinning, ccd_temp FROM usersessionfiles
            WHERE session_id = ? AND frame_type = 'Light'""", (session_id,)):
        key = (obs_id, so.normalize_filter(filter_name), so.exposure_bucket(exposure))
        values = lights.setdefault(key, ([], [], []))
        for values_list, value in zip(values, (gain, binning, temperature)):
            if value is not None:
                values_list.append(value)

    # Each night's calibration: darks by exposure, flats by filter, bias
    calibration = {}
    for obs_id, frame_type, filter_name, exposure in conn.execute("""
            SELECT observation_id, frame_type, filter_name, exptime_seconds FROM usersessionfiles
            WHERE session_id = ? AND observation_id IS NOT NULL AND frame_type IN ('Dark', 'Flat', 'Bias')""",
                                                                  (session_id,)):
        night = calibration.setdefault(obs_id, {"Dark": Counter(), "Flat": Counter(), "Bias": Counter()})
        if frame_type == "Dark":
            night["Dark"][so.exposure_bucket(exposure)] += 1
        elif frame_type == "Flat":
            night["Flat"][so.normalize_filter(filter_name)] += 1
        else:
            night["Bias"][None] += 1

    groups = [(obs_id, night, row["filter_name"], row["exposure_seconds"], row["counted_subs"])
              for obs_id, night in observations for row in breakdowns.get(obs_id) or []]
    for filter_name, exposure, count in conn.execute("""
            SELECT filter_name, exptime_seconds, COUNT(*) FROM usersessionfiles
            WHERE session_id = ? AND observation_id IS NULL AND frame_type = 'Light'
            GROUP BY filter_name, exptime_seconds""", (session_id,)):
        groups.append((None, None, filter_name, exposure, count))

    rows = []
    for obs_id, night, filter_name, exposure, count in groups:
        if not count or count <= 0 or not exposure or exposure <= 0:
            continue
        filter_key, bucket = so.normalize_filter(filter_name), so.exposure_bucket(exposure)
        row = {"date": _iso_date(night) if night else "", "filter": filter_ids.get(filter_key, ""),
               "number": str(int(count)), "duration": _number(exposure, 4)}
        gains, binnings, temperatures = lights.get((obs_id, filter_key, bucket), ([], [], []))
        if gains:
            row["gain"] = _number(Counter(gains).most_common(1)[0][0], 2)
        binning = Counter(int(b) for b in binnings).most_common(1)[0][0] if binnings else None
        if binning in (1, 2, 3, 4):
            row["binning"] = str(binning)
        if temperatures:
            row["sensorCooling"] = str(int(round(statistics.median(temperatures))))
        night_calibration = calibration.get(obs_id)
        if night_calibration:
            for column, kind, key in (("darks", "Dark", bucket), ("flats", "Flat", filter_key),
                                      ("bias", "Bias", None)):
                taken = night_calibration[kind].pop(key, 0)
                if taken:
                    row[column] = str(taken)
        rows.append(row)
    return rows


def write_acquisition_csv(path, rows):
    """The rows as AstroBin's CSV: a header line, no spaces around the commas,
    only the columns something fills (AstroBin leaves out a missing one)."""
    columns = [c for c in ACQUISITION_COLUMNS if c in REQUIRED_COLUMNS or any(row.get(c) for row in rows)]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c, "") for c in columns])


# ---------------------------------------------------------------------------
# Destinations
# ---------------------------------------------------------------------------

@dataclass
class ExportContext:
    """What a destination is handed with the rendered image."""
    file_path: str
    title: str
    options: ExportOptions
    source_size: tuple            # The image's (width, height)
    output_size: tuple            # What it's exported at, before a banner
    details: dict = field(default=None)   # {"auto", "edits"} - None if not gathered

    @property
    def has_overlays(self):
        return self.options.annotations or self.options.details

    @property
    def resized(self):
        return tuple(self.source_size) != tuple(self.output_size)


class Destination:
    """One way to export. Subclasses fill in the names and run(); the dialog
    does the rest. A sign-in service (Flickr, Google Photos) would add its
    sign-in to status() and options, and upload in run() with a worker."""
    key = ""
    label = ""
    description = ""            # One line under the list, what it does
    action_label = "Export"
    default_size = "full"
    needs_details = False       # Wants the image's details gathered (e.g. a description)

    def __init__(self, dialog):
        self.dialog = dialog
        self.worker = None

    def status(self):
        """Why this can't be used now (shown instead of running it) - None when it can."""
        return None

    def more_widget(self):
        """Its own settings, under More options - None for none."""
        return None

    def wants_render(self, context):
        """False when the original file is used as it is."""
        return True

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def run(self, image, context):
        raise NotImplementedError

    def release(self):
        """The dialog is closing: running work finishes, unreported."""
        if self.busy():
            keep_running(self.worker, *self._worker_signals())
        self.worker = None

    def _worker_signals(self):
        return ()

    def _save(self, image, path, image_format, quality, on_saved):
        """Write image in a worker, then on_saved(ok, path)."""
        self.worker = ImageSaveWorker(image, path, image_format, quality if image_format == "JPEG" else -1)
        self.worker.image_saved.connect(on_saved)
        self.dialog.start_work(f"Saving {Path(path).name}...")
        self.worker.start()


class SaveToFile(Destination):
    key = "file"
    label = "Save to File"
    description = "Save a PNG or JPEG copy, with what you include on it."
    action_label = "Save..."

    def run(self, image, context):
        original = Path(context.file_path)
        options = context.options
        suffix = "_annotated" if options.annotations else ("_details" if options.details else "_export")
        jpeg = original.suffix.lower() in ('.jpg', '.jpeg')
        default = original.parent / f"{original.stem}{suffix}{'.jpg' if jpeg else '.png'}"
        path, image_format = choose_save_path(self.dialog, "Save Image", default, jpeg)
        if path:
            self._save(image, path, image_format, options.quality, self._on_saved)

    def _worker_signals(self):
        return (self.worker.image_saved,)

    def _on_saved(self, ok, path):
        if ok:
            self.dialog.report("success", f"Saved {Path(path).name}", path)
        else:
            self.dialog.report("error", f"Failed to save the image:\n{path}")


class DesktopBackground(Destination):
    key = "background"
    label = "Desktop Background"
    description = ("Use it as your desktop background. A copy with what you include, or of a FITS/XISF image, "
                   "is saved next to it as a PNG.")
    action_label = "Set as Background"

    def status(self):
        return background_support()

    def wants_render(self, context):
        # A plain JPEG/PNG/... is used as it is
        return context.has_overlays or context.resized or is_astro_file(context.file_path)

    def run(self, image, context):
        self.worker = BackgroundSetterWorker(context.file_path, image)
        self.worker.background_set.connect(self._on_set)
        self.dialog.start_work("Saving the background..." if image is not None else "Setting the background...")
        self.worker.start()

    def _worker_signals(self):
        return (self.worker.background_set,)

    def _on_set(self, level, message):
        self.dialog.report(level, message)


class CopyToClipboard(Destination):
    key = "clipboard"
    label = "Copy to Clipboard"
    description = "Copy it, to paste into a forum post, chat or another app."
    action_label = "Copy"
    default_size = "2048"

    def run(self, image, context):
        QGuiApplication.clipboard().setImage(image)
        self.dialog.report("success", f"Copied the image ({image.width()} × {image.height()}) to the clipboard")


class AstroBinAssist(Destination):
    """AstroBin has no upload API: this gets everything ready for its uploader."""
    key = "astrobin"
    label = "AstroBin"
    description = ("Save an upload-ready copy, copy its capture details for the description, write the "
                   "acquisition CSV and open AstroBin's uploader.")
    action_label = "Prepare for AstroBin..."
    needs_details = True

    OPTIONS = [("open_page", "Open AstroBin's uploader", True),
               ("copy_description", "Copy the capture details for the description", True),
               ("write_csv", "Write the acquisition CSV (for AstroBin's Import CSV)", True)]

    def __init__(self, dialog):
        super().__init__(dialog)
        self.checks = {}
        self._context = None

    def _option(self, key):
        check = self.checks.get(key)
        if check is not None:
            return check.isChecked()
        default = next(on for k, _name, on in self.OPTIONS if k == key)
        return _settings().value(f"image_export/astrobin/{key}", default, type=bool)

    def more_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for key, name, _on in self.OPTIONS:
            check = QCheckBox(name)
            check.setChecked(self._option(key))
            check.toggled.connect(lambda on, key=key: _settings().setValue(f"image_export/astrobin/{key}", on))
            layout.addWidget(check)
            self.checks[key] = check
        ids_button = QPushButton("AstroBin Filter IDs...")
        ids_button.setToolTip("AstroBin's CSV names each filter by its AstroBin ID - set them once")
        ids_button.clicked.connect(self._edit_filter_ids)
        row = QHBoxLayout()
        row.addWidget(ids_button)
        row.addStretch()
        layout.addLayout(row)
        return widget

    def _session_id(self, conn):
        return ImageDetails.session_id_for(conn, self.dialog.file_path())

    def _edit_filter_ids(self):
        try:
            with _connect() as conn:
                session_id = self._session_id(conn)
                filters = session_filters(conn, session_id) if session_id else []
        except Exception as e:
            logger.error(f"Couldn't read the session's filters: {e}", exc_info=True)
            filters = []
        if not filters:
            QMessageBox.information(self.dialog, "AstroBin Filter IDs",
                                    "This image isn't linked to a session with filters, so there are none to set.")
            return
        AstroBinFilterDialog(filters, self.dialog).exec()

    def run(self, image, context):
        original = Path(context.file_path)
        path, image_format = choose_save_path(self.dialog, "Save for AstroBin",
                                              original.parent / f"{original.stem}_astrobin.jpg", True)
        if path:
            self._context = context
            self._save(image, path, image_format, context.options.quality, self._on_saved)

    def _worker_signals(self):
        return (self.worker.image_saved,)

    def _on_saved(self, ok, path):
        if not ok:
            self.dialog.report("error", f"Failed to save the image:\n{path}")
            return
        done = [f"Saved {Path(path).name}"]
        if self._option("write_csv"):
            done.append(self._write_csv(path))
        if self._option("copy_description"):
            details = self._context.details or {}
            text = ImageDetails.description_text(details.get("auto"), details.get("edits"))
            if text:
                QGuiApplication.clipboard().setText(text)
                done.append("the capture details are on the clipboard for the description")
        if self._option("open_page"):
            from UrlOpener import open_url
            open_url(ASTROBIN_UPLOAD_URL)
        self.dialog.report("success", "; ".join(part for part in done if part), path)

    def _write_csv(self, image_path):
        """Write the acquisition CSV next to the image - what to say about it."""
        try:
            with _connect() as conn:
                session_id = self._session_id(conn)
                rows = acquisition_rows(conn, session_id, load_filter_ids()) if session_id else []
        except Exception as e:
            logger.error(f"Couldn't build the acquisition CSV: {e}", exc_info=True)
            return "couldn't read the session for the acquisition CSV"
        if not rows:
            return "no acquisition CSV (the image isn't linked to a session with subs)"
        csv_path = Path(image_path).with_name(f"{Path(image_path).stem}_acquisition.csv")
        try:
            write_acquisition_csv(csv_path, rows)
        except OSError as e:
            logger.error(f"Couldn't write {csv_path}: {e}")
            return f"couldn't write {csv_path.name}"
        missing = any(not row["filter"] for row in rows)
        return f"wrote {csv_path.name}" + (" (set AstroBin Filter IDs to fill its filters)" if missing else "")


DESTINATIONS = [SaveToFile, DesktopBackground, CopyToClipboard, AstroBinAssist]


class _connect:
    """A database connection of its own, closed afterwards."""

    def __enter__(self):
        import sqlite3
        from ResourceManager import ResourceManager
        self.conn = sqlite3.connect(str(ResourceManager.get_database_path()))
        return self.conn

    def __exit__(self, *_exc):
        self.conn.close()
        return False


class AstroBinFilterDialog(QDialog):
    """AstroBin's ID for each filter of a session - its CSV names filters by them."""

    def __init__(self, filters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AstroBin Filter IDs - Cosmos Collection")
        self.resize(460, 340)
        import SessionObservations as so
        self.keys = [so.normalize_filter(name) for name in filters]
        ids = load_filter_ids()

        layout = QVBoxLayout(self)
        intro = QLabel(f"AstroBin's acquisition CSV names each filter by its AstroBin ID. Find your filter in "
                       f"<a href=\"{ASTROBIN_FILTERS_URL}\">AstroBin's equipment explorer</a> and paste its "
                       f"page address or number. Filters left blank are picked on AstroBin.")
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        layout.addWidget(intro)
        self.table = QTableWidget(len(filters), 2)
        self.table.setHorizontalHeaderLabels(["Filter", "AstroBin ID"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        for row, (name, key) in enumerate(zip(filters, self.keys)):
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem(ids.get(key, "")))
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self):
        ids = load_filter_ids()
        for row, key in enumerate(self.keys):
            item = self.table.item(row, 1)
            ids[key] = parse_filter_id(item.text() if item else "")
        save_filter_ids(ids)
        self.accept()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class _Preview(QLabel):
    """The rendered preview, fitted to the space it has."""

    def __init__(self):
        super().__init__()
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # It never asks to grow
        themed_style(self, lambda: f"background-color: black; color: #aaaaaa; font-size: {font_size(11)};")

    def show_image(self, image):
        self._pixmap = QPixmap.fromImage(image)
        self._fit()

    def show_message(self, text):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        room = self.size() - QSize(12, 12)
        self.setPixmap(self._pixmap.scaled(room.expandedTo(QSize(1, 1)), Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))


class ExportDialog(WindowPositionMixin, QDialog):
    """Every way to export the viewer's image - non-modal beside it. The
    include checkboxes start from what the viewer shows and can be changed
    here without changing the viewer; each destination remembers its size.

    The viewer provides: original_pixmap, file_path, title, details_style,
    export_annotations() -> (state "ready"/"loading"/"unsolved", AnnotationData),
    export_layers(), export_details() -> {"auto", "edits"} or None while
    gathered, annotations_enabled, and _open_details_dialog()."""
    WINDOW_POSITION_KEY = "ImageExportDialog"

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.setWindowTitle("Export Image - Cosmos Collection")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.resize(900, 600)
        self.setup_window_position()
        self.destinations = [cls(self) for cls in DESTINATIONS]
        self._working = None        # The destination whose work is running
        self._preview_base = None   # (image key, QImage scaled for the preview)
        self._released = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)

        layout = QHBoxLayout(self)
        self.destination_list = QListWidget()
        self.destination_list.setFixedWidth(190)
        for destination in self.destinations:
            item = QListWidgetItem(destination.label)
            item.setToolTip(destination.description)
            item.setSizeHint(QSize(0, 34))
            self.destination_list.addItem(item)
        layout.addWidget(self.destination_list)

        right = QVBoxLayout()
        layout.addLayout(right, 1)
        self.preview = _Preview()
        right.addWidget(self.preview, 1)
        self.size_note = QLabel()
        self.size_note.setAlignment(Qt.AlignCenter)
        themed_style(self.size_note, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        right.addWidget(self.size_note)
        self.description = QLabel()
        self.description.setWordWrap(True)
        right.addWidget(self.description)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        includes = QHBoxLayout()
        self.annotations_check = QCheckBox("Annotations")
        self.annotations_check.setChecked(bool(viewer.annotations_enabled))
        self.annotations_check.toggled.connect(self.refresh)
        includes.addWidget(self.annotations_check)
        self.details_check = QCheckBox("Details")
        self.details_check.setChecked(bool(viewer.details_style["enabled"]))
        self.details_check.toggled.connect(self.refresh)
        includes.addWidget(self.details_check)
        details_button = QPushButton("Details...")
        details_button.setToolTip("Choose the details shown and how they look")
        details_button.clicked.connect(viewer._open_details_dialog)
        includes.addWidget(details_button)
        includes.addStretch()
        form.addRow("Include:", includes)
        self.include_note = QLabel()
        self.include_note.setWordWrap(True)
        themed_style(self.include_note, lambda: f"color: {COLORS['text_secondary']}; font-size: {font_size(9)};")
        form.addRow("", self.include_note)
        self.size_combo = QComboBox()
        for key, name in SIZES:
            if key == "screen" and long_edge("screen"):
                name = f"Screen size - {long_edge('screen')} px"
            self.size_combo.addItem(name, key)
        self.size_combo.setToolTip("The long edge it's exported at - never larger than the image")
        self.size_combo.currentIndexChanged.connect(self._on_size_changed)
        form.addRow("Size:", self.size_combo)
        right.addLayout(form)

        # More options: tucked away, most people never need them
        self.more_toggle = QToolButton()
        self.more_toggle.setText("More options")
        self.more_toggle.setCheckable(True)
        self.more_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.more_toggle.setArrowType(Qt.RightArrow)
        self.more_toggle.setStyleSheet("QToolButton { border: none; }")
        self.more_toggle.toggled.connect(self._toggle_more)
        right.addWidget(self.more_toggle)
        self.more = QWidget()
        more_layout = QVBoxLayout(self.more)
        more_layout.setContentsMargins(20, 0, 0, 0)
        more_form = QFormLayout()
        more_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setSuffix("%")
        self.quality_spin.setToolTip("Higher keeps more of the faint detail, in a larger file")
        self.quality_spin.setValue(_settings().value(JPEG_QUALITY_SETTING, DEFAULT_JPEG_QUALITY, type=int))
        self.quality_spin.valueChanged.connect(lambda value: _settings().setValue(JPEG_QUALITY_SETTING, value))
        more_form.addRow("JPEG quality:", self.quality_spin)
        more_layout.addLayout(more_form)
        self.more_stack = QStackedWidget()
        for destination in self.destinations:
            self.more_stack.addWidget(destination.more_widget() or QWidget())
        more_layout.addWidget(self.more_stack)
        right.addWidget(self.more)

        buttons = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.linkActivated.connect(self._show_in_folder)
        buttons.addWidget(self.status_label, 1)
        self.action_button = QPushButton()
        self.action_button.setDefault(True)
        self.action_button.clicked.connect(self._run)
        buttons.addWidget(self.action_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        right.addLayout(buttons)

        more_shown = _settings().value(MORE_OPTIONS_SETTING, False, type=bool)
        self.more_toggle.setChecked(more_shown)
        self._toggle_more(more_shown)
        last = _settings().value(LAST_DESTINATION_SETTING, "", type=str)
        index = next((i for i, d in enumerate(self.destinations) if d.key == last), 0)
        self.destination_list.currentRowChanged.connect(self._on_destination_changed)
        self.destination_list.setCurrentRow(index)

    # The viewer's side

    def file_path(self):
        return self.viewer.file_path

    def follow_viewer(self):
        """The viewer's Show toggles changed - include what it shows now."""
        self.annotations_check.setChecked(bool(self.viewer.annotations_enabled))
        self.details_check.setChecked(bool(self.viewer.details_style["enabled"]))
        self.refresh()

    def refresh(self):
        """Something the export shows changed (loaded, another image, a toggle)."""
        self._update_controls()
        self._preview_timer.start(120)

    # State

    def _current(self):
        row = self.destination_list.currentRow()
        return self.destinations[row] if 0 <= row < len(self.destinations) else self.destinations[0]

    def _state(self):
        """What the export carries now: {"annotations": data or None, "lines",
        "details", "waiting": what's still loading or None, "notes"}."""
        viewer = self.viewer
        state = {"annotations": None, "lines": [], "details": None, "waiting": None, "notes": []}
        if viewer.original_pixmap is None:
            state["waiting"] = "Loading the image..."
            return state
        if self.annotations_check.isChecked():
            status, data = viewer.export_annotations()
            if status == "unsolved":
                state["notes"].append("Not plate solved, so there are no annotations - use Plate Solve in the "
                                      "viewer to find its objects.")
            elif status == "loading":
                state["waiting"] = "Loading the annotations..."
            else:
                state["annotations"] = data
        details = viewer.export_details()  # Gathered for the details and descriptions
        state["details"] = details
        if self.details_check.isChecked():
            if details is None:
                state["waiting"] = state["waiting"] or "Gathering the details..."
            else:
                state["lines"] = ImageDetails.detail_lines(details["auto"], details["edits"], viewer.details_style)
                if not state["lines"]:
                    state["notes"].append("None of the chosen details are known for this image - use "
                                          "Details... to choose others, or type them in.")
        if details is None and self._current().needs_details:
            state["waiting"] = state["waiting"] or "Gathering the details..."
        return state

    def _options(self, state):
        return ExportOptions(annotations=state["annotations"] is not None, details=bool(state["lines"]),
                             size=self.size_combo.currentData() or "full", quality=self.quality_spin.value())

    def _sizes(self, state):
        """(the image's size, its export size before a banner, the banner's height)"""
        pixmap = self.viewer.original_pixmap
        source = (pixmap.width(), pixmap.height())
        target = output_size(*source, self.size_combo.currentData() or "full")
        banner = ImageDetails.banner_height(state["lines"], target, self.viewer.details_style) \
            if state["lines"] else 0
        return source, target, banner

    def _update_controls(self):
        destination = self._current()
        state = self._state()
        self.description.setText(destination.description)
        self.action_button.setText(destination.action_label)
        self.include_note.setText(" ".join(state["notes"]))
        self.include_note.setVisible(bool(state["notes"]))
        if self.viewer.original_pixmap is not None:
            source, target, banner = self._sizes(state)
            text = f"{source[0]} × {source[1]} px"
            if target != source or banner:
                text += f"  →  {target[0]} × {target[1] + banner} px"
            self.size_note.setText(text + ("  (with the details banner)" if banner else ""))
        else:
            self.size_note.setText("")
        problem = destination.status()
        working = self._working is not None and self._working.busy()
        self.action_button.setEnabled(not (problem or state["waiting"] or working))
        if problem:
            self._set_status("warning", problem)
        elif state["waiting"] and not working:
            self._set_status(None, state["waiting"])
        elif not working and self._status_waiting:
            self._set_status(None, "")

    _status_waiting = False

    def _set_status(self, level, text, path=None):
        """The status line: text in the level's color, with a link to the file's folder."""
        self._status_waiting = level is None and bool(text)
        html = escape(text).replace("\n", "<br>")
        if path:
            html += f" &nbsp;<a href=\"{escape(path)}\">Show in folder</a>"
        self.status_label.setText(html)
        self.status_label.setToolTip(text)
        themed_style(self.status_label, lambda: f"color: {COLORS[level] if level else COLORS['text_secondary']};")

    # Preview

    def _render_preview(self):
        pixmap = self.viewer.original_pixmap
        if pixmap is None or pixmap.isNull():
            self.preview.show_message("Loading the image...")
            return
        key = (self.viewer.file_path, pixmap.cacheKey())
        if self._preview_base is None or self._preview_base[0] != key:
            small = pixmap
            if max(pixmap.width(), pixmap.height()) > PREVIEW_PX:
                small = pixmap.scaled(PREVIEW_PX, PREVIEW_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._preview_base = (key, small.toImage())
        state = self._state()
        image = render(self._preview_base[1], state["annotations"], self.viewer.export_layers(), state["lines"],
                       self.viewer.details_style)
        self.preview.show_image(image)

    # Controls

    def _on_destination_changed(self, row):
        destination = self._current()
        self.more_stack.setCurrentIndex(max(0, row))
        size = _settings().value(f"image_export/{destination.key}/size", destination.default_size, type=str)
        self.size_combo.blockSignals(True)
        self.size_combo.setCurrentIndex(max(0, self.size_combo.findData(size)))
        self.size_combo.blockSignals(False)
        if not (self._working is not None and self._working.busy()):
            self._set_status(None, "")
        self.refresh()

    def _on_size_changed(self, _index):
        _settings().setValue(f"image_export/{self._current().key}/size", self.size_combo.currentData())
        self.refresh()

    def _toggle_more(self, shown):
        self.more.setVisible(shown)
        self.more_toggle.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)
        _settings().setValue(MORE_OPTIONS_SETTING, shown)

    def _show_in_folder(self, path):
        from ResourceManager import ResourceManager
        if not ResourceManager.open_file_manager(path):
            QMessageBox.critical(self, "Error", "Failed to open the file's folder")

    # Running

    def _run(self):
        destination = self._current()
        state = self._state()
        if destination.status() or state["waiting"] or (self._working is not None and self._working.busy()):
            return
        options = self._options(state)
        source, target, _banner = self._sizes(state)
        context = ExportContext(self.viewer.file_path, self.viewer.title, options, source, target, state["details"])
        _settings().setValue(LAST_DESTINATION_SETTING, destination.key)
        image = None
        if destination.wants_render(context):
            self.setCursor(Qt.BusyCursor)
            try:
                base = self.viewer.original_pixmap.toImage()
                if target != source:
                    base = base.scaled(target[0], target[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                image = render(base, state["annotations"], self.viewer.export_layers(), state["lines"],
                               self.viewer.details_style)
            except Exception as e:
                logger.error(f"Couldn't render the export: {e}", exc_info=True)
                self.unsetCursor()
                QMessageBox.critical(self, "Error", f"Couldn't prepare the image: {e}")
                return
            self.unsetCursor()
        self._working = destination
        try:
            destination.run(image, context)
        except Exception as e:
            logger.error(f"Export to {destination.label} failed: {e}", exc_info=True)
            self.report("error", f"Couldn't export the image: {e}")

    def start_work(self, message):
        """A destination's work started (it reports when it's done)."""
        self.setCursor(Qt.BusyCursor)
        self._set_status(None, message)
        self._status_waiting = False
        self.action_button.setEnabled(False)

    def report(self, level, message, path=None):
        """A destination's result: 'success', 'warning' or 'error' - the last
        two as a message too, so they aren't missed. path: the file made."""
        self.unsetCursor()
        self._working = None
        self._set_status(level, message, path)
        self._update_controls()
        if level == "error":
            QMessageBox.critical(self, "Export Failed", message)
        elif level == "warning":
            QMessageBox.warning(self, "Export", message)

    # Closing

    def _release(self):
        if not self._released:
            self._released = True
            self._preview_timer.stop()
            for destination in self.destinations:
                destination.release()

    def done(self, result):
        """Closed (the close button, Escape or the viewer closing): work that's
        running finishes in the background."""
        self._release()
        WindowPositionManager.save_window_position(self, self.WINDOW_POSITION_KEY)
        super().done(result)
        self.deleteLater()
