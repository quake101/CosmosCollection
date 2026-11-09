#!/usr/bin/env python3
"""
DSO Detail Window for Cosmos Collection
Provides detailed object information, image management, and visibility data
"""

import logging
import os
from typing import Optional, Dict

from PySide6.QtCore import Qt, Signal, QObject, QTimer, QEvent, QThread
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QLineEdit,
    QComboBox, QTextEdit, QGroupBox, QPushButton, QGridLayout,
    QToolBar, QMessageBox, QMenu, QFileDialog, QProgressBar, QFrame
)

from cosmoscollection.equipment import EquipmentMapper
from cosmoscollection.sessions.SessionFileScanner import SUPPORTED_EXTENSIONS
from cosmoscollection.planning.Observer import active_site
from cosmoscollection.core.DatabaseManager import DatabaseManager, set_favorite_image
from cosmoscollection.ui.WindowPositionManager import WindowPositionManager
from cosmoscollection.ui.WindowLifetime import release_on_close, window_is_open
from cosmoscollection.ui.Theme import COLORS, adapt_color, font_size, themed_style, themed_text
from cosmoscollection.nina.NINAIntegration import NINAIntegration
from cosmoscollection.ui import DSOActions
from cosmoscollection.ui.ZoomableImageView import ZoomableImageView
from cosmoscollection.ui.DSOActions import TargetListActionsMixin
from cosmoscollection.core.UrlOpener import open_url
from cosmoscollection.core.Formatting import format_dec, format_magnitude, format_ra

# Set up logging
logger = logging.getLogger(__name__)

# Check for optional DSO Visibility Calculator availability
try:
    from cosmoscollection.planning import DSOVisibilityCalculator
    VISIBILITY_AVAILABLE = True
except ImportError:
    VISIBILITY_AVAILABLE = False
    logger.warning("DSOVisibilityCalculator.py not found. Visibility calculator will be disabled.")


class SimbadQueryThread(QThread):
    """Background thread for querying SIMBAD without blocking the UI"""
    query_complete = Signal(object, str, float, float)
    query_failed = Signal(str, str)

    def __init__(self, object_name, ra_deg, dec_deg, dso_type):
        super().__init__()
        self.object_name = object_name
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.dso_type = dso_type

    def run(self):
        """Query SIMBAD in background thread"""
        from cosmoscollection.catalog.SimbadQuery import SimbadUnavailable
        try:
            from cosmoscollection.catalog.SimbadQuery import query
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            logger.debug(f"Querying SIMBAD for emission data: {self.object_name} at RA={self.ra_deg}, Dec={self.dec_deg}")

            result = None
            if self.ra_deg is not None and self.dec_deg is not None:
                coords = SkyCoord(ra=self.ra_deg*u.degree, dec=self.dec_deg*u.degree, frame='icrs')
                result = query(lambda simbad: simbad.query_region(coords, radius=3*u.arcmin),
                               fields=('U', 'B', 'V', 'R', 'I', 'sp', 'otype'))
                if result is not None and len(result) > 0:
                    logger.debug(f"SIMBAD data found for {self.object_name} at coordinates")
                else:
                    logger.debug(f"No SIMBAD data found for {self.object_name} at coordinates")
            else:
                logger.debug(f"No coordinates available for {self.object_name}, skipping SIMBAD query")

            self.query_complete.emit(result, self.object_name, self.ra_deg, self.dec_deg)

        except SimbadUnavailable as e:
            self.query_failed.emit(self.object_name, str(e))
        except Exception as e:
            logger.error(f"SIMBAD query failed for {self.object_name}: {type(e).__name__}: {str(e)}", exc_info=True)
            self.query_failed.emit(self.object_name, "SIMBAD query failed")


# Image loader threads whose result is no longer wanted (the user switched image
# or closed the window). Referenced here until they finish, since destroying a
# running QThread crashes.
_orphaned_loader_threads = set()


class ImageCache:
    _instance = None
    _cache: Dict[str, QPixmap] = {}
    _max_size = 10

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCache, cls).__new__(cls)
        return cls._instance

    def get(self, path: str) -> Optional[QPixmap]:
        if path in self._cache:
            return self._cache[path]
        return None

    def put(self, path: str, pixmap: QPixmap):
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = pixmap

    def clear(self):
        self._cache.clear()


class VisibilityCalculationWorker(QObject):
    """Works out an object's viewing seasons in a background thread."""
    finished = Signal(str, str)  # (the seasons as HTML, their months: "Nov-Dec, Jan-Mar")
    error = Signal(str)

    def __init__(self, lat: float, lon: float, ra_deg: float, dec_deg: float, object_name: str):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.object_name = object_name

        from cosmoscollection.planning.Visibility import DSOVisibilityCalculator
        self.calculator = DSOVisibilityCalculator(lat, lon)

        # lat/lon are the active location's, so its custom horizon applies
        from cosmoscollection.planning.HorizonProfile import load_active_horizon
        self.horizon = load_active_horizon()

    def calculate_visibility(self):
        """Calculate visibility seasons in background thread"""
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from cosmoscollection.planning.Visibility import season_months, seasonal_visibility

            coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)
            seasons = seasonal_visibility(self.calculator, coord, 30, self.horizon)
            if seasons:
                criteria = ">30° altitude and above your custom horizon" if self.horizon else ">30° altitude"
                visibility_text = (f"Best viewing seasons ({criteria} in dark sky):<br>"
                                   + "<br>".join(f"{first:%B %d} - {last:%B %d}" for first, last in seasons)
                                   + "<br><br><small>Use Visibility Calculator for detailed nightly times.</small>")
            else:
                visibility_text = "Object not optimally visible from your location this year."
            self.finished.emit(visibility_text, season_months(seasons))
        except Exception as e:
            logger.error(f"Error calculating visibility: {str(e)}", exc_info=True)
            self.error.emit(f"Error calculating viewing season information:<br>{str(e)}")


class DSODetailWindow(TargetListActionsMixin, QDialog):
    """
    Detail window for DSO objects with image support including FITS files.
    """

    def __init__(self, data: dict, parent=None):
        super().__init__(None)
        release_on_close(self)  # Freed once closed (see WindowLifetime)
        logger.debug(f"Creating DSODetailWindow for {data['name']}")
        from cosmoscollection.catalog.DSONames import display_name
        self.setWindowTitle(f"{display_name(data['name'], data.get('dsodetailid'), data.get('designations'))}"
                            " - DSO Detail - Cosmos Collection")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.NonModal)
        self.resize(1000, 800)
        WindowPositionManager.restore_window_position(self, "ObjectDetail")
        self.setAcceptDrops(True)

        self.data = data
        self.current_image_path = None
        self.original_pixmap = None
        self.image_cache = ImageCache()
        self.db_manager = DatabaseManager()

        self.image_loader_thread = None
        self.simbad_query_thread = None
        self._shown = False  # The heavy work starts after the first show (showEvent)

        self.user_images = []
        self.current_image_index = 0
        self.best_months = ""  # The viewing seasons' months, once worked out

        self.ra_str, self.dec_str = format_ra(data["ra_deg"]), format_dec(data["dec_deg"])

        logger.debug(f"About to set up UI for {data['name']}")
        self._setup_ui()

    def showEvent(self, event):
        """The heavy work starts once, after the first show. Restoring the window
        from minimized shows it again - that re-ran everything: a second
        visibility thread replaced one still running (a fatal Qt error), the
        UI waited on the running SIMBAD query, and the images reloaded."""
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            QTimer.singleShot(200, self._defer_heavy_calculations)

    def _defer_heavy_calculations(self):
        """Perform heavy calculations after window is shown"""
        try:
            with DatabaseManager().get_connection() as conn:
                site = active_site(conn)
            if site is not None:
                QTimer.singleShot(500, self._set_season_label_from_location)
            else:
                self.season_label.setText("Set your location in Settings to see viewing season.")
        except Exception as e:
            logger.error(f"Error checking location in deferred calculations: {str(e)}")
            self.season_label.setText("Set your location in Settings to see viewing season.")

        QTimer.singleShot(300, self._load_user_images)
        QTimer.singleShot(400, self._query_emission_data)

    def _get_update_sources_html(self):
        """Return HTML listing update-database sources that contributed NAME designations to this DSO.

        Returns an empty string when no update databases are loaded or this DSO has no NAME entries.
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT d.id FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ? AND c.designation = ?
                """, (self.data['catalogue'], self.data['id']))
                row = cursor.fetchone()
                if not row:
                    return ""
                dsodetailid = str(row[0])

                cursor.execute("""
                    SELECT DISTINCT source FROM name_provenance_all
                    WHERE dsodetailid = ?
                """, (dsodetailid,))
                source_strings = [r[0] for r in cursor.fetchall()]
                if not source_strings:
                    return ""

                # Source strings can be combined, e.g. "OpenNGC+Wikidata"
                labels = set()
                for s in source_strings:
                    for part in s.split("+"):
                        labels.add(part.strip())

                placeholders = ",".join("?" * len(labels))
                cursor.execute(f"""
                    SELECT source, key, value FROM attribution_all
                    WHERE source IN ({placeholders}) AND key IN ('name', 'url', 'license')
                """, list(labels))
                attr_rows = cursor.fetchall()
        except Exception:
            return ""

        sources = {}
        for source, key, value in attr_rows:
            sources.setdefault(source, {})[key] = value

        if not sources:
            return ""

        parts = []
        for info in sorted(sources.values(), key=lambda x: x.get('name', '')):
            name = info.get('name', '')
            url = info.get('url', '')
            license_ = info.get('license', '')
            entry = f'<a href="{url}">{name}</a>' if url else name
            if license_:
                entry += f" ({license_})"
            parts.append(entry)

        return "<br><b>Name Sources:</b> " + " &middot; ".join(parts)

    def _query_emission_data(self):
        """Query SIMBAD for emission line and spectral information"""
        try:
            dso_type = self.data.get('dso_type', '')
            if dso_type not in ['BRTNB', 'CL+NB', 'PLNNB', 'SNREM']:
                self.emission_label.setText("Not applicable (not an emission nebula)")
                themed_style(self.emission_label, lambda: f"color: {COLORS['text_disabled']};")
                return

            from cosmoscollection.core.OnlineLimits import is_offline
            if is_offline():
                self.emission_label.setText("Not looked up - offline mode is on")
                themed_style(self.emission_label, lambda: f"color: {COLORS['text_disabled']};")
                return

            object_name = self.data.get('name', '')
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')

            self.emission_label.setText("Querying SIMBAD...")
            themed_style(self.emission_label, lambda: f"color: {COLORS['text_disabled']};")

            if self.simbad_query_thread and self.simbad_query_thread.isRunning():
                self.simbad_query_thread.quit()
                self.simbad_query_thread.wait()

            self.simbad_query_thread = SimbadQueryThread(object_name, ra_deg, dec_deg, dso_type)
            self.simbad_query_thread.query_complete.connect(self._on_simbad_query_complete)
            self.simbad_query_thread.query_failed.connect(self._on_simbad_query_failed)
            self.simbad_query_thread.start()

            logger.debug(f"Started background SIMBAD query for {object_name}")

        except Exception as e:
            logger.error(f"Error initiating SIMBAD query: {str(e)}", exc_info=True)
            self.emission_label.setText("Error querying SIMBAD")
            themed_style(self.emission_label, lambda: f"color: {COLORS['error']};")

    def _on_simbad_query_complete(self, result, object_name, ra_deg, dec_deg):
        """Handle SIMBAD query completion"""
        try:
            dso_type = self.data.get('dso_type', '')
            emission_info = self._parse_emission_info(dso_type, result)

            if emission_info:
                # The emission lines are colored per theme, so rebuild on theme changes
                themed_text(self.emission_label,
                            lambda dso_type=dso_type, result=result: self._parse_emission_info(dso_type, result))
                themed_style(self.emission_label, lambda: f"color: {COLORS['text']};")
            else:
                self.emission_label.setText("No specific emission line data available")
                themed_style(self.emission_label, lambda: f"color: {COLORS['text_disabled']};")

        except Exception as e:
            logger.error(f"Error processing SIMBAD result: {str(e)}", exc_info=True)
            self.emission_label.setText("Error processing SIMBAD data")
            themed_style(self.emission_label, lambda: f"color: {COLORS['error']};")

    def _on_simbad_query_failed(self, object_name, message):
        """Handle SIMBAD query failure - message says why, for the user"""
        logger.warning(f"SIMBAD query failed for {object_name}: {message}")
        self.emission_label.setText(message)
        themed_style(self.emission_label, lambda: f"color: {COLORS['error']};")

    @staticmethod
    def _emission_line_color(line):
        """Display color for an emission line, hinting at its wavelength"""
        if 'Hα' in line or 'Hβ' in line:
            return adapt_color('#ff6b6b')
        elif 'OIII' in line:
            return adapt_color('#6bcdff')
        elif 'SII' in line:
            return adapt_color('#ff9966')
        elif 'NII' in line:
            return adapt_color('#ff8888')
        return COLORS['text']

    def _parse_emission_info(self, dso_type, simbad_result):
        """Parse emission line information based on DSO type and SIMBAD data"""
        emission_lines = {
            'BRTNB': {
                'primary': ['Hα (656.3 nm)', 'Hβ (486.1 nm)'],
                'secondary': ['OIII (495.9, 500.7 nm)', 'SII (671.6, 673.1 nm)', 'NII (658.3 nm)'],
                'description': 'Emission nebula - typically rich in hydrogen and oxygen'
            },
            'CL+NB': {
                'primary': ['Hα (656.3 nm)'],
                'secondary': ['OIII (495.9, 500.7 nm)', 'Hβ (486.1 nm)'],
                'description': 'Star cluster with associated emission nebulosity'
            },
            'PLNNB': {
                'primary': ['OIII (495.9, 500.7 nm)', 'Hα (656.3 nm)'],
                'secondary': ['Hβ (486.1 nm)', 'NII (658.3 nm)'],
                'description': 'Planetary nebula - strong in oxygen and hydrogen'
            },
            'SNREM': {
                'primary': ['OIII (495.9, 500.7 nm)', 'SII (671.6, 673.1 nm)'],
                'secondary': ['Hα (656.3 nm)', 'Hβ (486.1 nm)'],
                'description': 'Supernova remnant - often oxygen and sulfur rich'
            }
        }

        if dso_type not in emission_lines:
            return None

        info = emission_lines[dso_type]
        text = ""

        if simbad_result is not None and len(simbad_result) > 0:
            row = simbad_result[0]
            text += "<b>SIMBAD Data:</b>"

            if 'OTYPE' in row.colnames and row['OTYPE']:
                otype = str(row['OTYPE']).strip()
                text += f"<b>Object Type:</b> {otype}<br>"

            if 'SP_TYPE' in row.colnames and row['SP_TYPE']:
                sp_type = str(row['SP_TYPE']).strip()
                if sp_type and sp_type != '--':
                    text += f"<b>Spectral Type:</b> {sp_type}<br>"

            flux_data = []
            flux_bands = [
                (['U', 'FLUX_U', 'flux_U'], 'U'),
                (['B', 'FLUX_B', 'flux_B'], 'B'),
                (['V', 'FLUX_V', 'flux_V'], 'V'),
                (['R', 'FLUX_R', 'flux_R'], 'R'),
                (['I', 'FLUX_I', 'flux_I'], 'I')
            ]

            for flux_cols, band in flux_bands:
                flux_val = None
                for flux_col in flux_cols:
                    if flux_col in row.colnames and row[flux_col] is not None:
                        try:
                            flux_val = float(row[flux_col])
                            if not (flux_val == 0 or str(flux_val) == 'nan'):
                                break
                        except (ValueError, TypeError):
                            pass
                if flux_val is not None and not (flux_val == 0 or str(flux_val) == 'nan'):
                    flux_data.append(f"{band}={flux_val:.2f}")

            if flux_data:
                text += f"<b>Photometry:</b> {', '.join(flux_data)} mag<br>"

            text += "<br>"

        text += f"<b>{info['description']}</b><br><br>"
        text += "<b>Primary Emission Lines:</b><br>"
        for line in info['primary']:
            color = self._emission_line_color(line)
            text += f"<span style='color: {color};'>• {line}</span><br>"

        text += "<br><b>Secondary Lines:</b><br>"
        for line in info['secondary']:
            color = self._emission_line_color(line)
            text += f"<span style='color: {color};'>• {line}</span><br>"

        text += "<br><i>Useful for Hα, OIII, and SII narrowband imaging</i>"

        return text

    def _setup_ui(self):
        """Set up the UI components - called after window is shown"""
        from PySide6.QtWidgets import QToolButton, QMenu
        from PySide6.QtGui import QAction

        logger.debug(f"_setup_ui called for {self.data['name']}")
        try:
            # Create menu bar using QToolBar with proper menu buttons
            menubar = QToolBar()
            menubar.setMovable(False)
            menubar.setStyleSheet("QToolBar { border: 0px; spacing: 3px; }")

            # Add Visibility Calculator button (if available)
            if VISIBILITY_AVAILABLE:
                visibility_button = QToolButton()
                visibility_button.setText("Visibility Calculator")
                visibility_button.setToolTip("Open the Visibility Calculator to see when this object is visible")
                visibility_button.clicked.connect(self._open_visibility_calculator)
                menubar.addWidget(visibility_button)

            # Add Aladin Lite button
            aladin_button = QToolButton()
            aladin_button.setText("FOV Simulator")
            aladin_button.setToolTip("Open interactive sky atlas with telescope field of view simulator")
            aladin_button.clicked.connect(lambda: self._open_aladin_lite(self.data))
            menubar.addWidget(aladin_button)

            # Add Wikipedia button
            wikipedia_button = QToolButton()
            wikipedia_button.setText("Wikipedia")
            wikipedia_button.setToolTip("Open the Wikipedia page for this object in your browser")
            wikipedia_button.clicked.connect(self._open_wikipedia)
            menubar.addWidget(wikipedia_button)

            # Target List menu: Add, or Remove / Open in, as the object is listed
            target_menu = QMenu("Target List", self)
            self.add_target_list_actions(target_menu)

            # Create Target List button with menu
            target_button = QToolButton()
            target_button.setText("Target List")
            target_button.setToolTip("Manage your observing target list")
            target_button.setMenu(target_menu)
            target_button.setPopupMode(QToolButton.InstantPopup)
            menubar.addWidget(target_button)

            # Add NINA button if integration is enabled
            if NINAIntegration.is_enabled():
                nina_menu = QMenu(self)

                framing_action = QAction("Send to Framing Assistant", self)
                framing_action.triggered.connect(lambda: DSOActions.send_to_nina(self.data, self))
                nina_menu.addAction(framing_action)

                slew_action = QAction("Slew to Target", self)
                slew_action.triggered.connect(lambda: DSOActions.slew_to(self.data, self))
                nina_menu.addAction(slew_action)

                nina_button = QToolButton()
                nina_button.setText("NINA")
                nina_button.setToolTip("Send to NINA")
                nina_button.setMenu(nina_menu)
                nina_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
                menubar.addWidget(nina_button)

            # Create main vertical layout to hold menubar and content
            window_layout = QVBoxLayout()
            window_layout.setContentsMargins(0, 0, 0, 0)
            window_layout.setSpacing(0)
            window_layout.addWidget(menubar)

            # Create main horizontal layout for content
            main_layout = QHBoxLayout()

            # Left side - Image placeholder and information
            left_layout = QVBoxLayout()

            # Image area (larger)
            image_layout = QVBoxLayout()

            # Create a container for the image and zoom controls
            image_container = QWidget()
            image_container_layout = QVBoxLayout()

            # Add zoom controls
            zoom_layout = QHBoxLayout()
            zoom_out_button = QPushButton("-")
            themed_style(zoom_out_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
            zoom_out_button.clicked.connect(lambda: self.image_view.zoom_by(1 / ZoomableImageView.ZOOM_STEP))
            zoom_layout.addWidget(zoom_out_button)

            zoom_in_button = QPushButton("+")
            themed_style(zoom_in_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
            zoom_in_button.clicked.connect(lambda: self.image_view.zoom_by(ZoomableImageView.ZOOM_STEP))
            zoom_layout.addWidget(zoom_in_button)

            reset_button = QPushButton("Reset")
            reset_button.setFixedSize(60, 30)
            reset_button.clicked.connect(lambda: self.image_view.fit())
            zoom_layout.addWidget(reset_button)

            # Add image navigation controls
            nav_separator = QLabel("|")
            themed_style(nav_separator, lambda: f"font-size: {font_size(12)}; color: {COLORS['border_light']}; padding: 0 5px;")
            zoom_layout.addWidget(nav_separator)

            self.prev_image_button = QPushButton("<-")
            themed_style(self.prev_image_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
            self.prev_image_button.clicked.connect(self._previous_image)
            self.prev_image_button.setToolTip("Previous image")
            zoom_layout.addWidget(self.prev_image_button)

            self.image_counter_label = QLabel("1/1")
            themed_style(self.image_counter_label, lambda: f"font-size: {font_size(10)}; color: {COLORS['border_light']}; padding: 0 5px;")
            self.image_counter_label.setMinimumWidth(40)
            self.image_counter_label.setAlignment(Qt.AlignCenter)
            zoom_layout.addWidget(self.image_counter_label)

            self.next_image_button = QPushButton("->")
            themed_style(self.next_image_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
            self.next_image_button.clicked.connect(self._next_image)
            self.next_image_button.setToolTip("Next image")
            zoom_layout.addWidget(self.next_image_button)

            # Add image button
            add_separator = QLabel("|")
            themed_style(add_separator, lambda: f"font-size: {font_size(12)}; color: {COLORS['border_light']}; padding: 0 5px;")
            zoom_layout.addWidget(add_separator)

            self.add_image_button = QPushButton("+")
            self.add_image_button.clicked.connect(self._add_user_image)
            self.add_image_button.setToolTip("Add new image")
            themed_style(self.add_image_button, lambda: f"QPushButton {{ color: {COLORS['success']}; font-size: {font_size(12)}; }}")
            zoom_layout.addWidget(self.add_image_button)

            # Delete image button
            self.delete_image_button = QPushButton("X")
            self.delete_image_button.clicked.connect(self._delete_current_image)
            self.delete_image_button.setToolTip("Delete current image")
            themed_style(self.delete_image_button, lambda: f"QPushButton {{ color: {COLORS['error']}; font-size: {font_size(12)}; }}")
            zoom_layout.addWidget(self.delete_image_button)

            # Favorite image button
            self.favorite_button = QPushButton("*")
            self.favorite_button.clicked.connect(self._toggle_favorite)
            self.favorite_button.setToolTip("Mark as favorite")
            themed_style(self.favorite_button, lambda: f"QPushButton {{ font-size: {font_size(12)}; }}")
            zoom_layout.addWidget(self.favorite_button)

            # Collages: add this image to one, or open one that shows it
            self.collage_button = QPushButton("Collage")
            self.collage_button.setToolTip("Add this image to a collage, or open a collage that shows it")
            collage_menu = QMenu(self.collage_button)
            collage_menu.aboutToShow.connect(lambda: self._fill_collage_menu(collage_menu))
            self.collage_button.setMenu(collage_menu)
            zoom_layout.addWidget(self.collage_button)

            zoom_layout.addStretch()
            image_container_layout.addLayout(zoom_layout)

            # The image: wheel zooms (from fit up to 4x), drag pans, double-click
            # opens the image viewer
            self.image_view = ZoomableImageView("Loading...", min_fit_ratio=1.0, max_fit_ratio=4.0,
                                                toggle_on_double_click=False, keep_zoom=False, message_points=14)
            self.image_view.setMinimumSize(600, 400)
            self.image_view.setFrameShape(QFrame.NoFrame)
            self.image_view.setStyleSheet("background: transparent;")  # The window's color around the image
            self.image_view.double_clicked.connect(self._open_image_viewer)
            image_container_layout.addWidget(self.image_view)

            # Loading an image: how much of the file is read, then busy while it decodes
            self.image_progress = QProgressBar()
            self.image_progress.setTextVisible(False)
            self.image_progress.setFixedHeight(8)
            self.image_progress.hide()
            image_container_layout.addWidget(self.image_progress)

            # Which collages show this image (links open the Collage Builder on them)
            self.collages_label = QLabel()
            self.collages_label.setTextFormat(Qt.RichText)
            self.collages_label.linkActivated.connect(self._open_collage)
            themed_style(self.collages_label, lambda: f"color: {COLORS['text_secondary']};")
            self.collages_label.hide()
            image_container_layout.addWidget(self.collages_label)

            image_container.setLayout(image_container_layout)
            image_layout.addWidget(image_container)
            left_layout.addLayout(image_layout, stretch=2)  # Give image area more stretch

            # Create container for image information form
            self.info_form_container = QGroupBox("Image Information")
            themed_style(self.info_form_container, lambda: f"QGroupBox:title {{ subcontrol-position: top center; font-size: {font_size(16)}; font-weight: bold; }}")
            info_form_layout = QGridLayout()
            info_form_layout.setSpacing(5)  # Reduce spacing between elements
            info_form_layout.setVerticalSpacing(5)
            info_form_layout.setHorizontalSpacing(10)

            # Row 0: Integration Time and Date (side by side)
            integration_label = QLabel("Integration:")
            self.integration_edit = QLineEdit()
            self.integration_edit.setPlaceholderText("e.g., 2h 30m")
            info_form_layout.addWidget(integration_label, 0, 0)
            info_form_layout.addWidget(self.integration_edit, 0, 1)

            date_label = QLabel("Date:")
            self.date_edit = QLineEdit()
            self.date_edit.setPlaceholderText("e.g., 2024-03-15")
            info_form_layout.addWidget(date_label, 0, 2)
            info_form_layout.addWidget(self.date_edit, 0, 3)

            # Row 1: Telescope and Camera (side by side) - your equipment, or any name typed
            for column, (label, placeholder) in enumerate((("Telescope:", "e.g., 8\" SCT"),
                                                           ("Camera:", "e.g., ASI294MC Pro"))):
                combo = QComboBox()
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.NoInsert)
                combo.lineEdit().setPlaceholderText(placeholder)
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
                info_form_layout.addWidget(QLabel(label), 1, column * 2)
                info_form_layout.addWidget(combo, 1, column * 2 + 1)
                if column == 0:
                    self.telescope_combo = combo
                else:
                    self.camera_combo = combo

            # Row 2: Notes (label and text edit on same row)
            notes_label = QLabel("Notes:")
            notes_label.setAlignment(Qt.AlignTop)  # Align label to top
            self.notes_edit = QTextEdit()
            self.notes_edit.setPlaceholderText("Additional notes about the image")
            self.notes_edit.setMaximumHeight(60)  # Limit notes height
            info_form_layout.addWidget(notes_label, 2, 0)
            info_form_layout.addWidget(self.notes_edit, 2, 1, 1, 3)  # Span 3 columns

            # Row 3: Save button
            self.save_button = QPushButton("Save Image Information")
            self.save_button.clicked.connect(self._save_image_info)
            info_form_layout.addWidget(self.save_button, 3, 0, 1, 4)  # Span all columns

            self.info_form_container.setLayout(info_form_layout)
            left_layout.addWidget(self.info_form_container, stretch=1)  # Give info area less stretch

            # Relocate image button (initially hidden)
            self.relocate_button = QPushButton("Relocate Image")
            self.relocate_button.clicked.connect(self._relocate_image)
            self.relocate_button.setVisible(False)
            left_layout.addWidget(self.relocate_button)

            # Load telescopes and cameras into the dropdowns after UI is set up
            self._load_equipment()

            # Add left layout to a container widget
            left_container = QWidget()
            left_container.setLayout(left_layout)
            main_layout.addWidget(left_container)

            # Right side - Object Information
            right_layout = QVBoxLayout()

            # Create Object information groupbox
            object_info_groupbox = QGroupBox(self.data["name"])
            themed_style(object_info_groupbox, lambda: f"QGroupBox:title {{ subcontrol-position: top center; font-size: {font_size(28)}; font-weight: bold; }}")
            object_info_layout = QVBoxLayout()

            # Add Object information with proper null handling
            magnitude_str = format_magnitude(self.data['magnitude'], decimals=2, unknown="Unknown")
            surface_brightness = format_magnitude(self.data['surface_brightness'], decimals=2)  # Same stand-ins
            surface_brightness_str = f"{surface_brightness} mag/arcmin2" if surface_brightness else "Unknown"

            # Handle size information
            size_min = self.data['size_min'] if self.data['size_min'] is not None else 0
            size_max = self.data['size_max'] if self.data['size_max'] is not None else 0
            if size_min > 0 or size_max > 0:
                size_str = f"{size_min:.1f}' - {size_max:.1f}'"
            else:
                size_str = "Unknown"

            object_info_text = (
                f"<b>Right Ascension:</b> {self.ra_str}<br>"
                f"<b>Declination:</b> {self.dec_str}<br>"
                f"<b>Constellation:</b> {self.data['constellation'] or 'Unknown'}<br><br>"
                f"<b>Magnitude:</b> {magnitude_str}<br>"
                f"<b>Surface Brightness:</b> {surface_brightness_str}<br>"
                f"<b>Size:</b> {size_str}<br>"
                f"<b>Type:</b> {self.data['dso_type'] or 'Unknown'}<br>"
                f"<b>Class:</b> {self.data['dso_class'] or 'Unknown'}<br><br>"
                f"<b>Other Designations:</b><br>"
            )

            # Add object designations with proper formatting
            logger.debug(f"Designations data: {self.data.get('designations')}")
            designations_str = self.data.get('designations')
            if designations_str and designations_str.strip():
                designations = [d.strip() for d in designations_str.split(',') if d.strip()]
                logger.debug(f"Split designations: {designations}")

                # Get the current primary designation (as shown in the title)
                primary_name = self.data.get('name', '').strip()

                # Show all designations except the current primary one
                other_designations = []
                seen = set()
                for designation in designations:
                    if designation and designation != primary_name and designation not in seen:
                        other_designations.append(designation)
                        seen.add(designation)

                # Display other designations
                if other_designations:
                    for designation in other_designations:
                        object_info_text += f"{designation}<br>"
                else:
                    object_info_text += "None<br>"
            else:
                object_info_text += "None<br>"

            object_info_text += "<br><b>Source:</b> N.I.N.A. Database"
            update_sources = self._get_update_sources_html()
            if update_sources:
                object_info_text += update_sources

            object_info_label = QLabel(object_info_text)
            object_info_label.setAlignment(Qt.AlignLeft)
            object_info_label.setWordWrap(True)
            object_info_label.setOpenExternalLinks(True)
            object_info_layout.addWidget(object_info_label)

            object_info_groupbox.setLayout(object_info_layout)
            right_layout.addWidget(object_info_groupbox)

            # --- Emission Lines / Gases GroupBox ---
            self.emission_groupbox = QGroupBox("Emission Lines / Gases")
            themed_style(self.emission_groupbox, lambda: f"QGroupBox:title {{ subcontrol-position: top center; font-size: {font_size(16)}; font-weight: bold; }}")
            emission_layout = QVBoxLayout()

            self.emission_label = QLabel("Loading emission data from SIMBAD...")
            self.emission_label.setAlignment(Qt.AlignLeft)
            self.emission_label.setWordWrap(True)
            themed_style(self.emission_label, lambda: f"color: {COLORS['text_disabled']}; font-style: italic;")
            emission_layout.addWidget(self.emission_label)

            # Add SIMBAD link using coordinates
            ra_deg = self.data.get('ra_deg')
            dec_deg = self.data.get('dec_deg')
            if ra_deg is not None and dec_deg is not None:
                # Format coordinates for SIMBAD URL (decimal degrees)
                simbad_url = f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra_deg}+{dec_deg}&Radius=3&Radius.unit=arcmin"
                simbad_link_label = QLabel(f"<br><a href='{simbad_url}'>View more details in SIMBAD</a>")
                simbad_link_label.setAlignment(Qt.AlignLeft)
                simbad_link_label.setOpenExternalLinks(True)  # Enable clickable links
                emission_layout.addWidget(simbad_link_label)

            self.emission_groupbox.setLayout(emission_layout)
            right_layout.addWidget(self.emission_groupbox)

            # --- Season / Dates GroupBox ---
            season_groupbox = QGroupBox("Viewing Season / Dates")
            themed_style(season_groupbox, lambda: f"QGroupBox:title {{ subcontrol-position: top center; font-size: {font_size(16)}; font-weight: bold; }}")
            season_layout = QVBoxLayout()
            self.season_label = QLabel("")
            self.season_label.setAlignment(Qt.AlignLeft)
            self.season_label.setWordWrap(True)
            season_layout.addWidget(self.season_label)
            season_groupbox.setLayout(season_layout)
            right_layout.addWidget(season_groupbox)
            # User location loaded below; will update groupbox visibility and season

            # Add some spacing
            right_layout.addStretch()

            # Check if DSO is already in target list and update menu action visibility
            self.update_target_list_actions()

            # Add right layout to a container widget
            right_container = QWidget()
            right_container.setLayout(right_layout)
            main_layout.addWidget(right_container)

            # Add the main content layout to the window layout
            window_layout.addLayout(main_layout)

            # Set the window layout
            self.setLayout(window_layout)

            # Defer image loading to improve initial window performance
            if self.data.get('image_path'):
                # Show image information form but defer actual image loading
                self.info_form_container.setVisible(True)
            else:
                # Hide image information form
                self.info_form_container.setVisible(False)

                self.season_label.setText("Loading location information...")

                # Ensure the window is properly sized before showing
                self.adjustSize()
                logger.debug("DSODetailWindow setup complete")
        except Exception as e:
            logger.error(f"Error in _setup_ui: {str(e)}", exc_info=True)

    def _open_visibility_calculator(self):
        """Open the DSO Visibility Calculator with the current object pre-loaded"""
        if not VISIBILITY_AVAILABLE:
            QMessageBox.warning(self, "Feature Unavailable",
                                "DSO Visibility Calculator is not available. "
                                "Please ensure DSOVisibilityCalculator.py is in the same directory.")
            return
        DSOActions.open_visibility(self.data, self)

    def _load_equipment(self):
        """Load the user's telescopes and cameras into their dropdowns"""
        try:
            with DatabaseManager().get_connection() as conn:
                for combo, query in ((self.telescope_combo, "SELECT name FROM usertelescopes ORDER BY name"),
                                     (self.camera_combo, "SELECT name FROM userequipment "
                                                         "WHERE equipment_type = 'camera' ORDER BY name")):
                    combo.clear()
                    for (name,) in conn.execute(query).fetchall():
                        combo.addItem(name)
                    combo.setCurrentIndex(-1)
        except Exception as e:
            logger.error(f"Error loading telescopes and cameras: {e}")

    def _set_season_label_from_location(self):
        """
        Set the season_label text with the visibility season/dates string based on user location.
        Reads location directly from the database. Uses background thread for heavy calculations.
        """
        try:
            with DatabaseManager().get_connection() as conn:
                site = active_site(conn)
            if site is None:
                self.season_label.setText("Set your location in Settings to see viewing season.")
                return

            lat, lon = float(site.lat), float(site.lon)
            ra_deg = self.data.get("ra_deg")
            dec_deg = self.data.get("dec_deg")
            object_name = self.data.get("name")

            # Show loading message
            self.season_label.setText("Calculating viewing seasons...")

            # Create worker thread for visibility calculation
            self.visibility_thread = QThread()
            self.visibility_worker = VisibilityCalculationWorker(lat, lon, ra_deg, dec_deg, object_name)

            # Move worker to thread
            self.visibility_worker.moveToThread(self.visibility_thread)

            # Connect signals
            self.visibility_thread.started.connect(self.visibility_worker.calculate_visibility)
            self.visibility_worker.finished.connect(self._on_visibility_calculated)
            self.visibility_worker.error.connect(self._on_visibility_error)
            self.visibility_worker.finished.connect(self.visibility_thread.quit)
            self.visibility_worker.error.connect(self.visibility_thread.quit)
            self.visibility_thread.finished.connect(self.visibility_thread.deleteLater)

            # Start the thread
            self.visibility_thread.start()

        except Exception as e:
            logger.error(f"Error setting up visibility calculation: {str(e)}")
            self.season_label.setText(f"Error setting up viewing season calculation:<br>{str(e)}")

    def _on_visibility_calculated(self, visibility_text, best_months):
        """Handle completion of visibility calculation"""
        self.season_label.setText(visibility_text)
        self.best_months = best_months
        logger.debug("Visibility calculation completed in background thread")

    def _on_visibility_error(self, error_message):
        """Handle error from visibility calculation"""
        self.season_label.setText(error_message)
        logger.error("Visibility calculation failed in background thread")

    def _save_image_info(self):
        """Save the current image information to the database"""
        try:
            # Only save if we have images and a valid current image
            if not self.user_images or self.current_image_index >= len(self.user_images):
                QMessageBox.warning(self, "No Image", "No image selected to save information for.")
                return

            current_image = self.user_images[self.current_image_index]
            image_id = current_image['id']

            # The telescope and camera, linked to the user's Equipment - asking
            # about any that aren't in it
            from cosmoscollection.equipment.EquipmentManager import choose_equipment
            equipment = choose_equipment(self, self.telescope_combo.currentText(),
                                         self.camera_combo.currentText())
            if equipment is None:
                return

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Update the specific image information
                cursor.execute("""
                    UPDATE userimages
                    SET integration_time = ?,
                        telescope_id = ?,
                        telescope_name = ?,
                        camera_id = ?,
                        camera_name = ?,
                        equipment = NULL,
                        date_taken = ?,
                        notes = ?
                    WHERE id = ?
                """, (
                    self.integration_edit.text().strip(),
                    equipment['telescope_id'],
                    equipment['telescope_name'],
                    equipment['camera_id'],
                    equipment['camera_name'],
                    self.date_edit.text().strip(),
                    self.notes_edit.toPlainText().strip(),
                    image_id
                ))
                conn.commit()
                shown = EquipmentMapper.image_equipment(tuple(conn.execute(
                    f"SELECT {EquipmentMapper.select_sql('ui')} FROM userimages ui WHERE ui.id = ?",
                    (image_id,)).fetchone()))

                # Update the current image data in our local list
                current_image.update({
                    'integration_time': self.integration_edit.text().strip(),
                    'telescope': shown['telescope'],
                    'camera': shown['camera'],
                    'date_taken': self.date_edit.text().strip(),
                    'notes': self.notes_edit.toPlainText().strip()
                })
                self.telescope_combo.setCurrentText(shown['telescope'])
                self.camera_combo.setCurrentText(shown['camera'])

                logger.debug(f"Successfully updated image information for image {image_id}")

                QMessageBox.information(self, "Success", "Image information saved successfully!")

        except Exception as e:
            logger.error(f"Error saving image information: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save image information: {str(e)}")

    def _add_user_image(self, file_name=None):
        """Add an image to this object with the Add Image dialog (file_name:
        pre-selected, e.g. dropped), then show it."""
        from cosmoscollection.imaging.DSOGallery import import_image
        try:
            with self.db_manager.get_connection() as conn:
                row = conn.execute("""
                    SELECT d.id
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ? AND c.designation = ?
                """, (self.data['catalogue'], self.data['id'])).fetchone()
        except Exception as e:
            logger.error(f"Error finding {self.data['name']} to add an image: {e}", exc_info=True)
            row = None
        if not row:
            QMessageBox.critical(self, "Error",
                                 f"Could not add image: no database entry was found for {self.data['name']}.")
            return

        new_image_id = import_image(self, file_name or None, dsodetailid=row[0])
        if new_image_id is None:
            return
        # Show the new image (it may have been added to another object instead)
        self._load_user_images()
        for i, img in enumerate(self.user_images):
            if img['id'] == new_image_id:
                self.image_view.set_image(None, "Loading image...")
                self.current_image_index = i
                self._load_user_image(img['image_path'])
                self._load_current_image_info()
                self._update_image_navigation()
                self.info_form_container.setVisible(True)
                break

    def _load_user_image(self, image_path):
        """Load and display a user image with caching (supports FITS files) - async version"""
        try:
            # Store the current image path
            self.current_image_path = image_path

            # Drop any load still in progress for a previous image - first, so
            # it can't replace what's shown now (a cached image, or why this
            # one can't be shown) when it finishes
            self._abandon_image_loader_thread()

            logger.debug(f"Attempting to load image: {image_path}")

            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                self.image_view.set_image(None, f"Image file not found:\n{os.path.basename(image_path)}\n\nClick 'Relocate Image' button to find the new location")
                self.image_view.setToolTip(f"File path: {image_path}")
                self._show_relocate_button()
                return

            # Check file size
            try:
                file_size = os.path.getsize(image_path)
                logger.debug(f"Image file size: {file_size} bytes")
                if file_size == 0:
                    logger.error(f"Image file is empty: {image_path}")
                    self.image_view.set_image(None, f"Image file is empty:\n{os.path.basename(image_path)}")
                    self.image_view.setToolTip(f"File path: {image_path}")
                    return
            except OSError as e:
                logger.error(f"Error checking file size: {e}")
                self.image_view.set_image(None, f"Cannot access image file:\n{os.path.basename(image_path)}")
                self.image_view.setToolTip(f"File path: {image_path}\nError: {str(e)}")
                return

            # Check cache first
            cached_pixmap = self.image_cache.get(image_path)
            if cached_pixmap:
                logger.debug("Using cached image")
                self.original_pixmap = cached_pixmap
                self._display_loaded_image()
                return

            # Load in a background thread (all formats) so large files don't
            # freeze the UI - the same loader as the Image Viewer, with progress
            from cosmoscollection.imaging.ImageViewer import ImageLoadWorker
            logger.info(f"Image loading: {os.path.basename(image_path)} - {file_size / (1024 * 1024):.1f}MB")
            self.image_view.set_image(None, f"Loading {os.path.basename(image_path)}...")
            self.image_progress.setRange(0, 0)  # Busy until the first progress arrives
            self.image_progress.show()
            self.image_loader_thread = ImageLoadWorker(image_path)
            self.image_loader_thread.image_loaded.connect(self._on_image_loaded)
            self.image_loader_thread.load_failed.connect(self._on_image_load_failed)
            self.image_loader_thread.progress.connect(self._on_image_load_progress)
            self.image_loader_thread.start()
            logger.debug(f"Started background loading for: {image_path}")

        except Exception as e:
            logger.error(f"Error initiating image load: {str(e)}", exc_info=True)
            self.image_view.set_image(None, f"Error loading image:\n{str(e)}")

    def _on_image_load_progress(self, image_path, done, total):
        """How much of the file is read, under the image and in its place -
        then decoding, which can't report progress."""
        if image_path != self.current_image_path:
            return  # From a load that was dropped (another image was chosen)
        name = os.path.basename(image_path)
        if done < total:
            from cosmoscollection.imaging.ImageViewer import _megabytes
            self.image_progress.setRange(0, 1000)
            self.image_progress.setValue(int(done * 1000 / total) if total else 0)
            percent = int(done * 100 / total) if total else 0
            self.image_view.set_image(None, f"Loading {name}...\n\n{percent}% ({_megabytes(done)} of {_megabytes(total)})")
        else:
            self.image_progress.setRange(0, 0)
            self.image_view.set_image(None, f"Loading {name}...\n\nDecoding...")

    def _on_image_loaded(self, qimage, image_path):
        """Handle image loaded signal from background thread"""
        if image_path != self.current_image_path:
            return  # From a load that was dropped (another image was chosen)
        self.image_progress.hide()
        try:
            logger.info(f"Full image received: {qimage.width()}x{qimage.height()}")

            # Convert QImage to QPixmap (must be done on main thread)
            self.original_pixmap = QPixmap.fromImage(qimage)

            if self.original_pixmap.isNull():
                logger.error(f"Failed to convert QImage to QPixmap for: {image_path}")
                self._on_image_load_failed("Failed to convert image", image_path)
                return

            logger.info(f"Full resolution image converted to pixmap: {self.original_pixmap.width()}x{self.original_pixmap.height()}")

            # Cache the loaded image
            self.image_cache.put(image_path, self.original_pixmap)

            # Display the image
            self._display_loaded_image()
            logger.info("Full resolution image displayed")

        except Exception as e:
            logger.error(f"Error handling loaded image: {str(e)}", exc_info=True)
            self._on_image_load_failed(str(e), image_path)

    def _on_image_load_failed(self, error_message, image_path):
        """Handle image load failure from background thread"""
        if image_path != self.current_image_path:
            return
        self.image_progress.hide()
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext in SUPPORTED_EXTENSIONS:
            self.image_view.set_image(None, f"Failed to load FITS/XISF file:\n{os.path.basename(image_path)}\n\nRequires astropy library for FITS support\nError: {error_message}")
        else:
            self.image_view.set_image(None, f"Failed to load image:\n{os.path.basename(image_path)}\n\nError: {error_message}")

        self.image_view.setToolTip(f"File path: {image_path}")
        logger.error(f"Image load failed: {image_path} - {error_message}")

    def _display_loaded_image(self):
        """Show the loaded image, fitted to the view"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        self.image_view.set_image(self.original_pixmap)
        self.image_view.setToolTip("Double click to open in the image viewer")
        logger.debug("Image loaded and displayed successfully")

    def _open_aladin_lite(self, data):
        """The FOV Simulator for this object - the one already open, if it is."""
        if window_is_open(getattr(self, 'aladin_window', None)):
            self.aladin_window.raise_()
            self.aladin_window.activateWindow()
        else:
            self.aladin_window = DSOActions.open_fov(data, self)

    def _open_wikipedia(self):
        """Open Wikipedia page for the current DSO in the default browser"""
        try:
            import re

            dso_name = self.data.get('name', '').strip()

            if not dso_name:
                QMessageBox.warning(self, "No Name", "Cannot open Wikipedia: DSO name not available.")
                return

            dso_name = re.sub(r'\s*\([^)]*\)', '', dso_name).strip()

            if re.match(r'^M\s*\d+$', dso_name):
                number = re.search(r'\d+', dso_name).group()
                wiki_name = f'Messier_{number}'
            elif re.match(r'^Sh2[\s\-]*(\d+)$', dso_name, re.IGNORECASE):
                match = re.match(r'^Sh2[\s\-]*(\d+)$', dso_name, re.IGNORECASE)
                number = match.group(1)
                wiki_name = f'Sh_2-{number}'
            else:
                wiki_name = dso_name.replace(' ', '_')

            wiki_url = f"https://en.wikipedia.org/wiki/{wiki_name}"

            logger.debug(f"Opening Wikipedia page: {wiki_url}")
            open_url(wiki_url)

        except Exception as e:
            logger.error(f"Error opening Wikipedia: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Wikipedia: {str(e)}")

    def _open_image_viewer(self):
        """Open the current image in a new window - which steps through the
        object's other images too"""
        if self.original_pixmap is not None:
            try:
                from cosmoscollection.imaging.ImageViewer import ImageViewerWindow
                from cosmoscollection.catalog.DSONames import display_name

                ra_deg = self.data.get("ra_deg")
                dec_deg = self.data.get("dec_deg")
                title = display_name(self.data["name"], self.data.get("dsodetailid"), self.data.get("designations"))
                # No pixmap: the viewer loads the file itself (in the background), which
                # it needs to do to step through the versions
                versions = [img['image_path'] for img in self.user_images]
                viewer = ImageViewerWindow(
                    None, title, self.current_image_path, self,
                    dso_ra=ra_deg, dso_dec=dec_deg, versions=versions
                )
                viewer.setModal(False)
                viewer.show()
                logger.debug(f"Opened image viewer for {self.data['name']} (RA={ra_deg}, Dec={dec_deg})")
            except Exception as e:
                logger.error(f"Error opening image viewer: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to open image viewer: {str(e)}")

    def _load_user_images(self):
        """Load all user images for this DSO from the database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT d.id
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE c.catalogue = ? AND c.designation = ?
                """, (self.data['catalogue'], self.data['id']))
                result = cursor.fetchone()

                if result:
                    dsodetailid = result[0]

                    cursor.execute(f"""
                        SELECT ui.id, ui.image_path, ui.integration_time, ui.date_taken, ui.notes, ui.is_favorite,
                               {EquipmentMapper.select_sql('ui')}
                        FROM userimages ui
                        WHERE ui.dsodetailid = ?
                        ORDER BY ui.is_favorite DESC, ui.id ASC
                    """, (dsodetailid,))

                    images = cursor.fetchall()

                    self.user_images = []
                    for row in images:
                        img_id, image_path, integration_time, date_taken, notes, is_favorite = tuple(row)[:6]
                        shown = EquipmentMapper.image_equipment(tuple(row)[6:])
                        self.user_images.append({
                            'id': img_id,
                            'dsodetailid': dsodetailid,
                            'image_path': image_path,
                            'integration_time': integration_time,
                            'telescope': shown['telescope'],
                            'camera': shown['camera'],
                            'date_taken': date_taken,
                            'notes': notes,
                            'is_favorite': is_favorite
                        })

                    logger.debug(f"Loaded {len(self.user_images)} images for {self.data['name']}")

                    self._update_image_navigation()

                    if self.user_images:
                        self.image_view.set_image(None, "Loading image...")

                        self.current_image_index = 0
                        current_image = self.user_images[self.current_image_index]
                        self._load_user_image(current_image['image_path'])
                        self._load_current_image_info()
                        self.info_form_container.setVisible(True)
                    else:
                        self.image_view.set_image(None, "No image attached to this DSO.")
                        self.info_form_container.setVisible(False)
                else:
                    logger.error(f"Could not find dsodetailid for {self.data['name']}")
                    self.image_view.set_image(None, "No image attached to this DSO.")
                    self.info_form_container.setVisible(False)

        except Exception as e:
            logger.error(f"Error loading user images: {str(e)}", exc_info=True)
            self.image_view.set_image(None, "Error loading images.", error=True)
            self.info_form_container.setVisible(False)

    def _update_image_navigation(self):
        """Update the image navigation controls based on current state"""
        image_count = len(self.user_images)

        if image_count == 0:
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.delete_image_button.setEnabled(False)
            self.image_counter_label.setText("0/0")
            self._update_collage_links()
        elif image_count == 1:
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            self.delete_image_button.setEnabled(True)
            self.image_counter_label.setText("1/1")
        else:
            self.prev_image_button.setEnabled(self.current_image_index > 0)
            self.next_image_button.setEnabled(self.current_image_index < image_count - 1)
            self.delete_image_button.setEnabled(True)
            self.image_counter_label.setText(f"{self.current_image_index + 1}/{image_count}")

    def _previous_image(self):
        """Navigate to the previous image"""
        if self.user_images and self.current_image_index > 0:
            self.image_view.set_image(None, "Loading image...")

            self.current_image_index -= 1
            current_image = self.user_images[self.current_image_index]
            self._load_user_image(current_image['image_path'])
            self._load_current_image_info()
            self._update_image_navigation()
            logger.debug(f"Navigated to previous image: {self.current_image_index + 1}/{len(self.user_images)}")

    def _next_image(self):
        """Navigate to the next image"""
        if self.user_images and self.current_image_index < len(self.user_images) - 1:
            self.image_view.set_image(None, "Loading image...")

            self.current_image_index += 1
            current_image = self.user_images[self.current_image_index]
            self._load_user_image(current_image['image_path'])
            self._load_current_image_info()
            self._update_image_navigation()
            logger.debug(f"Navigated to next image: {self.current_image_index + 1}/{len(self.user_images)}")

    def _load_current_image_info(self):
        """Load the current image's information into the form fields"""
        if self.user_images and 0 <= self.current_image_index < len(self.user_images):
            current_image = self.user_images[self.current_image_index]
            self.integration_edit.setText(current_image.get('integration_time', ''))
            self.telescope_combo.setCurrentText(current_image.get('telescope', ''))
            self.camera_combo.setCurrentText(current_image.get('camera', ''))
            self.date_edit.setText(current_image.get('date_taken', ''))
            self.notes_edit.setText(current_image.get('notes', ''))

            is_favorite = current_image.get('is_favorite', 0)
            if is_favorite:
                self.favorite_button.setText("*")
                themed_style(self.favorite_button, lambda: f"QPushButton {{ color: {COLORS['favorite']}; font-size: {font_size(12)}; }}")
                self.favorite_button.setToolTip("Unmark as favorite")
            else:
                self.favorite_button.setText("*")
                themed_style(self.favorite_button, lambda: f"QPushButton {{ color: {COLORS['text_disabled']}; font-size: {font_size(12)}; }}")
                self.favorite_button.setToolTip("Mark as favorite")
        self._update_collage_links()

    def _current_image_id(self):
        if self.user_images and 0 <= self.current_image_index < len(self.user_images):
            return self.user_images[self.current_image_index].get('id')
        return None

    def _fill_collage_menu(self, menu):
        """Add to Collage / Show in Collage for the image shown."""
        menu.clear()
        image_id = self._current_image_id()
        if image_id is None:
            menu.addAction("No image to add").setEnabled(False)
            return
        from cosmoscollection.imaging.CollageBuilder import add_collage_actions
        add_collage_actions(menu, self, [image_id])

    def _update_collage_links(self):
        """'In collages: Messier 2, Galaxies' under the image - each a link."""
        if not hasattr(self, "collages_label"):
            return
        image_id = self._current_image_id()
        using = []
        if image_id is not None:
            try:
                from cosmoscollection.imaging.CollageBuilder import collages_using
                with self.db_manager.get_connection() as conn:
                    using = collages_using(conn, [image_id])
            except Exception as e:
                logger.debug(f"Could not look up collages: {e}")
        if using:
            from html import escape
            links = ", ".join(f"<a href='{collage_id}'>{escape(name)}</a>" for collage_id, name in using)
            self.collages_label.setText(f"In collages: {links}")
        self.collages_label.setVisible(bool(using))

    def changeEvent(self, event):
        # Back from the Collage Builder (an image may have been added): refresh the links
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._update_collage_links()
        super().changeEvent(event)

    def _open_collage(self, link):
        from cosmoscollection.imaging.CollageBuilder import open_collage_builder
        image_id = self._current_image_id()
        open_collage_builder(self, int(link), [image_id] if image_id is not None else None)

    def _toggle_favorite(self):
        """Toggle the favorite status of the current image"""
        if not self.user_images or self.current_image_index < 0 or self.current_image_index >= len(self.user_images):
            return

        current_image = self.user_images[self.current_image_index]
        image_id = current_image.get('id')

        if not image_id:
            logger.error("Cannot toggle favorite: image has no ID")
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                new_favorite = 0 if current_image.get('is_favorite', 0) else 1
                set_favorite_image(cursor, image_id, bool(new_favorite))
                conn.commit()

                current_image['is_favorite'] = new_favorite

                if new_favorite == 1:
                    for img in self.user_images:
                        if img['id'] != image_id:
                            img['is_favorite'] = 0

                self._load_current_image_info()
                logger.debug(f"Toggled favorite status for image {image_id} to {new_favorite}")

        except Exception as e:
            logger.error(f"Error toggling favorite: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to toggle favorite: {str(e)}")

    def _show_relocate_button(self):
        """Show the relocate button when an image cannot be found"""
        self.relocate_button.setVisible(True)
        logger.debug("Relocate button shown due to missing image file")

    def _relocate_image(self):
        """Allow user to select new location for missing image file"""
        try:
            if not hasattr(self, 'current_image_path') or not self.current_image_path:
                QMessageBox.warning(self, "Error", "No image selected for relocation.")
                return

            if not self.user_images or self.current_image_index < 0 or self.current_image_index >= len(self.user_images):
                QMessageBox.warning(self, "Error", "No current image to relocate.")
                return

            current_image = self.user_images[self.current_image_index]
            image_id = current_image.get('id')

            if not image_id:
                QMessageBox.warning(self, "Error", "Cannot identify current image for relocation.")
                return

            original_filename = os.path.basename(self.current_image_path)

            new_image_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select new location for {original_filename}",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.fits *.fit *.fts *.xisf);;All Files (*)"
            )

            if new_image_path:
                try:
                    self.db_manager.execute_update(
                        "UPDATE userimages SET image_path = ? WHERE id = ?",
                        (new_image_path, image_id)
                    )

                    self.current_image_path = new_image_path
                    current_image['image_path'] = new_image_path

                    self.relocate_button.setVisible(False)
                    self.image_view.set_image(None, "Loading image...")
                    self._load_user_image(new_image_path)

                    QMessageBox.information(self, "Success", f"Image location updated successfully!")
                    logger.info(f"Image relocated successfully to {new_image_path}")

                except Exception as db_error:
                    logger.error(f"Database error during image relocation: {str(db_error)}")
                    QMessageBox.critical(self, "Database Error", f"Failed to update image location: {str(db_error)}")

        except Exception as e:
            logger.error(f"Error relocating image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to relocate image: {str(e)}")

    def _delete_current_image(self):
        """Delete the current image from the database and update the display"""
        try:
            if not self.user_images:
                QMessageBox.warning(self, "No Images", "No images available to delete.")
                return

            if self.current_image_index < 0 or self.current_image_index >= len(self.user_images):
                QMessageBox.warning(self, "Error", "No current image selected for deletion.")
                return

            current_image = self.user_images[self.current_image_index]
            image_id = current_image.get('id')
            image_path = current_image.get('image_path', 'Unknown')

            if not image_id:
                QMessageBox.warning(self, "Error", "Cannot identify current image for deletion.")
                return

            filename = os.path.basename(image_path) if image_path != 'Unknown' else 'this image'

            reply = QMessageBox.question(
                self, "Delete Image",
                f"Are you sure you want to delete '{filename}'?\n\nThis will remove the image record from the database.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM userimages WHERE id = ?", (image_id,))
                conn.commit()

                del self.user_images[self.current_image_index]

                if self.user_images:
                    if self.current_image_index >= len(self.user_images):
                        self.current_image_index = len(self.user_images) - 1

                    current_image = self.user_images[self.current_image_index]
                    self._load_user_image(current_image['image_path'])
                    self._load_current_image_info()
                else:
                    self.current_image_index = 0
                    self._abandon_image_loader_thread()
                    self.current_image_path = None
                    self.image_view.set_image(None, "No Image Loaded")
                    self._clear_image_info()

                self._update_image_navigation()

                QMessageBox.information(self, "Success", f"'{filename}' has been removed from the database.")
                logger.info(f"Deleted image with ID {image_id}: {filename}")

        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to delete image: {str(e)}")

    def _clear_image_info(self):
        """Clear all image information fields"""
        self.integration_edit.setText("")
        self.telescope_combo.setCurrentText("")
        self.camera_combo.setCurrentText("")
        self.date_edit.setText("")
        self.notes_edit.setText("")

    def target_list_data(self):
        """What the Add Target dialog is given: the object, with its best months
        once the viewing seasons are worked out."""
        data = dict(self.data)
        if self.best_months:
            data['best_months'] = self.best_months
        return data

    def _abandon_image_loader_thread(self):
        """Stop a running image load: its read stops (decoding can't be
        interrupted), its result is dropped, and the thread object is kept
        alive until it finishes - destroying a running QThread crashes."""
        thread = self.image_loader_thread
        if thread and thread.isRunning():
            thread.cancel()
            thread.image_loaded.disconnect()
            thread.load_failed.disconnect()
            thread.progress.disconnect()
            _orphaned_loader_threads.add(thread)
            thread.finished.connect(lambda: _orphaned_loader_threads.discard(thread))
        self.image_loader_thread = None
        if hasattr(self, 'image_progress'):
            self.image_progress.hide()

    def closeEvent(self, event):
        """Save window position when closing"""
        self._abandon_image_loader_thread()
        WindowPositionManager.save_window_position(self, "ObjectDetail")
        event.accept()

    def dragEnterEvent(self, event):
        """Accept drag if it is a single supported image file"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                from cosmoscollection.imaging.DSOGallery import SUPPORTED_IMAGE_EXTENSIONS
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Add the dropped file as a new image for this object"""
        from cosmoscollection.imaging.DSOGallery import SUPPORTED_IMAGE_EXTENSIONS
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                event.acceptProposedAction()
                # Deferred: a dialog shown inside dropEvent keeps the source app's drag (e.g. Explorer) blocked
                QTimer.singleShot(0, lambda: self._add_user_image(file_path))
