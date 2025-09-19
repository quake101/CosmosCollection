#!/usr/bin/env python3
"""
Best DSO Tonight Calculator
Calculates and displays the best Deep Sky Objects visible from user's location tonight
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QGroupBox, QMessageBox,
                               QHeaderView, QProgressBar, QSpinBox, QComboBox, QMenu)

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import pytz
import warnings

warnings.filterwarnings('ignore')

from DatabaseManager import DatabaseManager

class DSOCalculationThread(QThread):
    """
    Thread for calculating best DSOs for tonight.
    
    Uses coordinate-based visibility calculations for reliability and consistency
    with other tools (avoids object name resolution issues like 'sh2 142' vs 'sh2-142').
    """
    progress = Signal(int)
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, min_altitude=30, max_magnitude=12.0, selected_catalogs=None, dso_limit=200, selected_dso_types=None):
        super().__init__()
        self.min_altitude = min_altitude
        self.max_magnitude = max_magnitude
        self.selected_catalogs = selected_catalogs or []
        self.selected_dso_types = selected_dso_types or []
        self.dso_limit = dso_limit
        
        # Use centralized calculator
        try:
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            # Initialize with location info - the calculator will load from database if no params provided
            self.calculator = DSOVisibilityCalculator()
            self.location = self.calculator.location
            self.local_tz = self.calculator.timezone
        except (ImportError, Exception) as e:
            print(f"Warning: Could not initialize DSOVisibilityCalculator: {e}")
            self.calculator = None
            self.location = self.setup_location()
            self.local_tz = self.setup_timezone()

    def setup_location(self):
        """Set up the observer location from database"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    lat, lon = row
                    if lat is not None and lon is not None:
                        lat = lat * u.deg
                        lon = lon * u.deg
                        height = 250 * u.m
                        return EarthLocation(lat=lat, lon=lon, height=height)
        except Exception:
            pass
        return None

    def setup_timezone(self):
        """Set up the user's timezone from database"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row and row[0]:
                    return pytz.timezone(row[0])
        except Exception:
            pass
        return pytz.UTC

    def calculate_tonight_visibility(self, dso_info):
        """Calculate visibility for a DSO tonight using centralized calculator with coordinates"""
        try:
            if self.calculator is None:
                # Fallback to original method if centralized calculator not available
                return self._calculate_tonight_visibility_fallback(dso_info)
            
            # Get tonight's date
            now = datetime.now(self.local_tz)
            tonight_date = now.strftime('%Y-%m-%d')
            
            # Use coordinate-based calculation for reliability (avoids name resolution issues)
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            # Create coordinate object from DSO data
            dso_coord = SkyCoord(ra=dso_info["ra_deg"] * u.deg, dec=dso_info["dec_deg"] * u.deg)
            
            # Use coordinate-based calculation
            time_range, dso_altaz, sun_altaz = self.calculator.calculate_altaz_over_time(
                dso_coord, tonight_date, 12)
            
            # Find optimal viewing times using same criteria
            optimal_times = self.calculator.find_optimal_viewing_times(
                dso_altaz, sun_altaz, self.min_altitude)
            
            # Create results structure compatible with existing code
            results = {
                "optimal_times": optimal_times,
                "dso_altaz": dso_altaz,
                "time_range": time_range,
                "sun_altaz": sun_altaz,
                "timezone": self.local_tz
            }
            
            if "error" in results or not np.any(results.get("optimal_times", [])):
                return None
            
            # Extract relevant information
            optimal_times = results["optimal_times"]
            dso_altaz = results["dso_altaz"]
            time_range = results["time_range"]
            
            # Calculate metrics
            max_altitude = np.max(dso_altaz.alt.deg[optimal_times])
            visible_hours = np.sum(optimal_times) * 0.25  # 15-minute intervals
            
            # Find optimal viewing time (mid-point of viewing window)
            optimal_indices = np.where(optimal_times)[0]
            if len(optimal_indices) > 0:
                mid_idx = optimal_indices[len(optimal_indices)//2]
                optimal_time_utc = time_range[mid_idx].datetime.replace(tzinfo=pytz.UTC)
                optimal_time_local = optimal_time_utc.astimezone(self.local_tz)
                optimal_altitude = dso_altaz.alt.deg[mid_idx]
                optimal_azimuth = dso_altaz.az.deg[mid_idx]
            else:
                return None
            
            return {
                "dso_info": dso_info,
                "max_altitude": max_altitude,
                "visible_hours": visible_hours,
                "optimal_time": optimal_time_local,
                "optimal_altitude": optimal_altitude,
                "optimal_azimuth": optimal_azimuth,
                "coordinates": dso_coord
            }
            
        except Exception:
            return None
    
    def _calculate_tonight_visibility_fallback(self, dso_info):
        """Fallback method using original calculations if centralized calculator unavailable"""
        try:
            # Get tonight's date range (sunset to sunrise)
            now = datetime.now(self.local_tz)
            tonight_start = now.replace(hour=18, minute=0, second=0, microsecond=0)
            
            # Convert to astropy Time objects
            start_time = Time(tonight_start.astimezone(pytz.UTC).replace(tzinfo=None))
            time_range = start_time + np.linspace(0, 12, 48) * u.hour  # Every 15 minutes
            
            # Get DSO coordinates from database data (coordinate-based for reliability)
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                dso_coord = SkyCoord(ra=dso_info["ra_deg"] * u.deg, dec=dso_info["dec_deg"] * u.deg)
            except Exception:
                return None
            
            # Calculate altitude/azimuth
            altaz_frame = AltAz(obstime=time_range, location=self.location)
            dso_altaz = dso_coord.transform_to(altaz_frame)
            
            # Calculate sun position
            sun = get_sun(time_range)
            sun_altaz = sun.transform_to(altaz_frame)
            
            # Find when object is visible (above minimum altitude and sun is down)
            dso_visible = dso_altaz.alt.deg > self.min_altitude
            dark_sky = sun_altaz.alt.deg < -12  # Astronomical twilight
            optimal_times = dso_visible & dark_sky
            
            if not np.any(optimal_times):
                return None
                
            # Calculate visibility metrics
            max_altitude = np.max(dso_altaz.alt.deg[optimal_times])
            visible_hours = np.sum(optimal_times) * 0.25  # 15-minute intervals
            
            # Find optimal viewing time
            optimal_indices = np.where(optimal_times)[0]
            if len(optimal_indices) > 0:
                mid_idx = optimal_indices[len(optimal_indices)//2]
                optimal_time_utc = time_range[mid_idx].datetime.replace(tzinfo=pytz.UTC)
                optimal_time_local = optimal_time_utc.astimezone(self.local_tz)
                optimal_altitude = dso_altaz.alt.deg[mid_idx]
                optimal_azimuth = dso_altaz.az.deg[mid_idx]
            else:
                return None
            
            return {
                "dso_info": dso_info,
                "max_altitude": max_altitude,
                "visible_hours": visible_hours,
                "optimal_time": optimal_time_local,
                "optimal_altitude": optimal_altitude,
                "optimal_azimuth": optimal_azimuth,
                "coordinates": dso_coord
            }
            
        except Exception:
            return None

    def azimuth_to_direction(self, az):
        """Convert azimuth to cardinal direction using centralized method"""
        if self.calculator:
            return self.calculator.azimuth_to_direction(az)
        else:
            # Fallback implementation
            directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            idx = int((az + 11.25) / 22.5) % 16
            return directions[idx]

    def get_available_catalogs(self):
        """Get list of available catalogs from database"""
        try:
            from ResourceManager import ResourceManager
            import sqlite3
            
            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT catalogue FROM cataloguenr ORDER BY catalogue")
            catalogs = [row[0] for row in cursor.fetchall()]
            conn.close()
            return catalogs
        except Exception as e:
            print(f"Error getting catalogs: {e}")
            return []

    def load_dsos_from_database(self):
        """Load DSOs from the database with thread-safe connection"""
        try:
            # Import here to avoid circular imports and create fresh connection in thread
            from ResourceManager import ResourceManager
            import sqlite3
            
            # Create a new database connection in this thread
            db_path = ResourceManager.get_database_path()
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            
            # Build the query with catalog filtering if specified
            base_query = """
                SELECT DISTINCT
                    (
                        SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ')
                        FROM cataloguenr c
                        WHERE c.dsodetailid = d.id
                        LIMIT 1
                    ) as name,
                    d.dsotype as type,
                    d.constellation,
                    d.magnitude,
                    d.surfacebrightness,
                    CAST(d.sizemin/60.0 AS REAL) as sizemin,
                    CAST(d.sizemax/60.0 AS REAL) as sizemax,
                    d.dsoclass,
                    d.ra as ra_deg,
                    d.dec as dec_deg,
                    (
                        SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ')
                        FROM cataloguenr c
                        WHERE c.dsodetailid = d.id
                    ) as designations
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE d.magnitude IS NOT NULL 
                    AND d.magnitude <= ?
                    AND d.constellation IS NOT NULL
                    AND d.dsotype IS NOT NULL
                    AND d.ra IS NOT NULL
                    AND d.dec IS NOT NULL
            """
            
            params = [self.max_magnitude]
            
            # Add catalog filtering if catalogs are selected
            if self.selected_catalogs:
                catalog_placeholders = ','.join('?' * len(self.selected_catalogs))
                base_query += f" AND c.catalogue IN ({catalog_placeholders})"
                params.extend(self.selected_catalogs)
            
            # Add DSO type filtering if types are selected
            if self.selected_dso_types:
                type_placeholders = ','.join('?' * len(self.selected_dso_types))
                base_query += f" AND d.dsotype IN ({type_placeholders})"
                params.extend(self.selected_dso_types)
            
            base_query += f" ORDER BY d.magnitude ASC LIMIT {self.dso_limit}"
            
            cursor.execute(base_query, params)
            
            rows = cursor.fetchall()
            dsos = []
            for row in rows:
                name, dso_type, constellation, magnitude, surface_brightness, size_min, size_max, dso_class, ra_deg, dec_deg, designations = row
                if name and magnitude is not None and ra_deg is not None and dec_deg is not None:
                    # Take the first designation as the primary name
                    primary_name = name.split(',')[0].strip()
                    dsos.append({
                        "name": primary_name,
                        "type": dso_type or "Unknown",
                        "constellation": constellation or "Unknown",
                        "magnitude": float(magnitude),
                        "surface_brightness": float(surface_brightness) if surface_brightness is not None else 0.0,
                        "size_min": float(size_min) if size_min is not None else 0.0,
                        "size_max": float(size_max) if size_max is not None else 0.0,
                        "dso_class": dso_class or "Unknown",
                        "ra_deg": float(ra_deg),
                        "dec_deg": float(dec_deg),
                        "designations": designations or name
                    })
            
            conn.close()
            return dsos
            
        except Exception as e:
            print(f"Error loading DSOs from database: {e}")
            return []
    
    def calculate_dso_batch(self, dso_batch):
        """Calculate visibility for a batch of DSOs"""
        batch_results = []
        for dso_info in dso_batch:
            result = self.calculate_tonight_visibility(dso_info)
            if result:
                result["direction"] = self.azimuth_to_direction(result["optimal_azimuth"])
                batch_results.append(result)
        return batch_results

    def run(self):
        """Main calculation thread with parallel processing"""
        try:
            if self.location is None:
                self.error_occurred.emit("Observer location not configured")
                return
            
            # Load DSOs from database
            dso_catalog = self.load_dsos_from_database()
            
            if not dso_catalog:
                self.error_occurred.emit("No DSOs found in database")
                return
            
            visible_dsos = []
            total_dsos = len(dso_catalog)
            
            # Determine number of threads (use CPU count but limit to reasonable maximum)
            import os
            max_workers = min(os.cpu_count() or 4, 8)  # Use CPU count, max 8 threads
            
            # Split DSOs into batches for parallel processing
            batch_size = max(1, len(dso_catalog) // max_workers)
            dso_batches = [dso_catalog[i:i + batch_size] for i in range(0, len(dso_catalog), batch_size)]
            
            completed_count = 0
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {executor.submit(self.calculate_dso_batch, batch): batch for batch in dso_batches}
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        visible_dsos.extend(batch_results)
                        
                        # Update progress based on completed batches
                        completed_count += len(future_to_batch[future])
                        progress = int((completed_count * 100) / total_dsos)
                        self.progress.emit(min(progress, 100))  # Ensure we don't exceed 100%
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
            
            # Sort by combination of altitude and magnitude (lower magnitude is better)
            visible_dsos.sort(key=lambda x: (-x["max_altitude"] + x["dso_info"]["magnitude"]), reverse=False)
            
            self.result_ready.emit(visible_dsos)
            
        except Exception as e:
            self.error_occurred.emit(f"Calculation error: {str(e)}")


class BestDSOTonightWindow(QMainWindow):
    """Main window for Best DSO Tonight calculator"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Best DSO Tonight - Cosmos Collection")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QSpinBox, QComboBox {
                background-color: #404040;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #404040;
                border: 1px solid #666666;
                color: #ffffff;
                gridline-color: #666666;
                selection-background-color: #0078d4;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #555555;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #555555;
                color: #ffffff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QLabel {
                color: #ffffff;
            }
            QProgressBar {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 3px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        
        self.calc_thread = None
        self.available_catalogs = []
        self.visible_dsos_data = []  # Store DSO data for detail window
        self.init_ui()
        self.load_location_info()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("Best Deep Sky Objects for Tonight")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(header_label)
        
        # Location info
        self.location_group = QGroupBox("Observer Location")
        location_layout = QVBoxLayout(self.location_group)
        self.location_label = QLabel("Loading location...")
        location_layout.addWidget(self.location_label)
        main_layout.addWidget(self.location_group)
        
        # Settings
        settings_group = QGroupBox("Calculation Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # First row of settings
        settings_row1 = QHBoxLayout()
        
        # Minimum altitude
        settings_row1.addWidget(QLabel("Min Altitude:"))
        self.min_altitude_spin = QSpinBox()
        self.min_altitude_spin.setRange(10, 90)
        self.min_altitude_spin.setValue(30)
        self.min_altitude_spin.setSuffix("°")
        settings_row1.addWidget(self.min_altitude_spin)
        
        # Maximum magnitude
        settings_row1.addWidget(QLabel("Max Magnitude:"))
        self.max_magnitude_combo = QComboBox()
        self.max_magnitude_combo.addItems(["8.0", "10.0", "12.0", "14.0", "16.0"])
        self.max_magnitude_combo.setCurrentText("12.0")
        settings_row1.addWidget(self.max_magnitude_combo)
        
        # DSO Limit
        settings_row1.addWidget(QLabel("DSO Limit:"))
        self.dso_limit_spin = QSpinBox()
        self.dso_limit_spin.setRange(50, 1000)
        self.dso_limit_spin.setValue(200)
        self.dso_limit_spin.setSingleStep(50)
        settings_row1.addWidget(self.dso_limit_spin)
        
        settings_row1.addStretch()
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate Best DSOs Tonight")
        self.calculate_btn.clicked.connect(self.calculate_best_dsos)
        settings_row1.addWidget(self.calculate_btn)
        
        settings_layout.addLayout(settings_row1)
        
        # Second row - Catalog and DSO Type selection
        catalog_row = QHBoxLayout()
        catalog_row.addWidget(QLabel("Catalog:"))
        
        # Catalog dropdown selection
        self.catalog_combo = QComboBox()
        self.load_catalog_options()
        
        # Add "All Catalogs" option and individual catalogs
        self.catalog_combo.addItem("All Catalogs")
        for catalog in sorted(self.available_catalogs):
            self.catalog_combo.addItem(catalog)
        
        self.catalog_combo.setCurrentText("All Catalogs")
        catalog_row.addWidget(self.catalog_combo)
        
        # DSO Type filter
        catalog_row.addWidget(QLabel("Type:"))
        self.dso_type_combo = QComboBox()
        self.load_dso_type_options()
        catalog_row.addWidget(self.dso_type_combo)
        
        catalog_row.addStretch()
        settings_layout.addLayout(catalog_row)
        
        main_layout.addWidget(settings_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Results table
        results_group = QGroupBox("Tonight's Best DSOs")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "DSO", "Type", "Constellation", "Magnitude", 
            "Max Alt.", "Best Time", "Direction", "Visible Hours"
        ])
        
        # Enable sorting and disable editing
        self.results_table.setSortingEnabled(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Set column widths
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        header.setSectionResizeMode(6, QHeaderView.Fixed)
        header.setSectionResizeMode(7, QHeaderView.Fixed)
        
        self.results_table.setColumnWidth(0, 80)
        self.results_table.setColumnWidth(3, 70)
        self.results_table.setColumnWidth(4, 70)
        self.results_table.setColumnWidth(5, 80)
        self.results_table.setColumnWidth(6, 70)
        self.results_table.setColumnWidth(7, 90)
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Connect double-click handler
        self.results_table.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Enable context menu
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_context_menu)
        
        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_group)
        
        # Status label
        self.status_label = QLabel("Ready to calculate best DSOs for tonight")
        main_layout.addWidget(self.status_label)

    def load_location_info(self):
        """Load and display location information"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon, location_name, timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                
                if row:
                    lat, lon, location_name, timezone = row
                    if lat is not None and lon is not None:
                        lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
                        lon_str = f"{abs(lon):.2f}°{'W' if lon < 0 else 'E'}"
                        display_name = location_name if location_name else "User Location"
                        
                        location_text = f"{display_name} - {lat_str}, {lon_str}"
                        if timezone:
                            try:
                                tz_obj = pytz.timezone(timezone)
                                now = datetime.now(tz_obj)
                                tz_abbrev = now.strftime('%Z')
                                location_text += f" ({tz_abbrev})"
                            except Exception:
                                pass
                        
                        self.location_label.setText(location_text)
                        self.calculate_btn.setEnabled(True)
                        return
                
                # No location configured
                self.location_label.setText("Location not configured - Please set location in main application")
                self.calculate_btn.setEnabled(False)
                
        except Exception:
            self.location_label.setText("Error loading location")
            self.calculate_btn.setEnabled(False)

    def load_catalog_options(self):
        """Load available catalogs from database"""
        try:
            # Create a temporary calculation thread to get catalogs
            temp_thread = DSOCalculationThread()
            self.available_catalogs = temp_thread.get_available_catalogs()
        except Exception as e:
            print(f"Error loading catalogs: {e}")
            self.available_catalogs = ["M", "NGC", "IC"]  # Fallback default catalogs
    
    def load_dso_type_options(self):
        """Load available DSO types from database with friendly names"""
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT dsotype FROM dsodetail WHERE dsotype IS NOT NULL ORDER BY dsotype")
                dso_types = [row[0] for row in cursor.fetchall()]

                # Add "All Types" option first
                self.dso_type_combo.addItem("All Types")

                # Add DSO types with friendly names, ordered by frequency/popularity
                priority_types = [
                    "GALXY", "DRKNB", "OPNCL", "PLNNB", "BRTNB", "SNREM",
                    "GALCL", "GLOCL", "ASTER", "2STAR", "CL+NB", "GX+DN",
                    "3STAR", "4STAR", "1STAR", "LMCOC", "LMCCN", "LMCGC",
                    "LMCDN", "SMCGC", "SMCCN", "SMCOC", "SMCDN", "QUASR", "NONEX"
                ]

                # Add priority types first if they exist in database
                for dso_type in priority_types:
                    if dso_type in dso_types:
                        friendly_name = self._get_friendly_type_name(dso_type)
                        self.dso_type_combo.addItem(friendly_name, dso_type)

                # Add any remaining types that weren't in priority list
                for dso_type in sorted(dso_types):
                    if dso_type not in priority_types:
                        friendly_name = self._get_friendly_type_name(dso_type)
                        self.dso_type_combo.addItem(friendly_name, dso_type)

                self.dso_type_combo.setCurrentText("All Types")
        except Exception as e:
            print(f"Error loading DSO type options: {e}")
            # Add default option if database query fails
            self.dso_type_combo.addItem("All Types")
    

    def calculate_best_dsos(self):
        """Start the calculation of best DSOs for tonight"""
        if self.calc_thread and self.calc_thread.isRunning():
            return
        
        # Get settings
        min_altitude = self.min_altitude_spin.value()
        max_magnitude = float(self.max_magnitude_combo.currentText())
        dso_limit = self.dso_limit_spin.value()
        
        # Get selected catalog
        selected_catalog = self.catalog_combo.currentText()
        selected_catalogs = [] if selected_catalog == "All Catalogs" else [selected_catalog]
        
        # Get selected DSO type
        selected_dso_type_text = self.dso_type_combo.currentText()
        if selected_dso_type_text == "All Types":
            selected_dso_types = []
        else:
            # Get the database code from the combo box data
            selected_dso_type = self.dso_type_combo.currentData()
            selected_dso_types = [selected_dso_type] if selected_dso_type else []
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Calculating...")
        
        # Create status text
        catalog_text = "all catalogs" if not selected_catalogs else f"{selected_catalog} catalog"
        type_text = "all types" if not selected_dso_types else f"{selected_dso_type} objects"
        self.status_label.setText(f"Calculating visibility for {type_text} from {catalog_text}...")
        
        # Start calculation thread
        self.calc_thread = DSOCalculationThread(min_altitude, max_magnitude, selected_catalogs, dso_limit, selected_dso_types)
        self.calc_thread.progress.connect(self.progress_bar.setValue)
        self.calc_thread.result_ready.connect(self.display_results)
        self.calc_thread.error_occurred.connect(self.handle_error)
        self.calc_thread.start()

    def display_results(self, visible_dsos):
        """Display the calculation results in the table"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Best DSOs Tonight")
        
        if not visible_dsos:
            self.status_label.setText("No DSOs meet the visibility criteria for tonight")
            self.results_table.setRowCount(0)
            return
        
        self.status_label.setText(f"Found {len(visible_dsos)} visible DSOs for tonight")
        
        # Disable sorting temporarily while populating
        self.results_table.setSortingEnabled(False)
        
        # Populate table
        self.results_table.setRowCount(len(visible_dsos))
        
        # Store the DSO data for later use in double-click handler
        self.visible_dsos_data = visible_dsos
        
        for row, dso_data in enumerate(visible_dsos):
            dso_info = dso_data["dso_info"]
            
            # DSO name - store DSO data in item for sorting
            name_item = QTableWidgetItem(dso_info["name"])
            name_item.setData(Qt.UserRole, dso_data)  # Store full DSO data
            self.results_table.setItem(row, 0, name_item)
            
            # Type - use friendly name
            friendly_type = self._get_friendly_type_name(dso_info["type"])
            self.results_table.setItem(row, 1, QTableWidgetItem(friendly_type))
            
            # Constellation
            self.results_table.setItem(row, 2, QTableWidgetItem(dso_info["constellation"]))
            
            # Magnitude - use numeric sorting
            mag_item = QTableWidgetItem()
            mag_item.setData(Qt.DisplayRole, f"{dso_info['magnitude']:.1f}")
            mag_item.setData(Qt.UserRole, dso_info['magnitude'])  # Store numeric value for sorting
            mag_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 3, mag_item)
            
            # Maximum altitude - use numeric sorting
            alt_item = QTableWidgetItem()
            alt_item.setData(Qt.DisplayRole, f"{dso_data['max_altitude']:.0f}°")
            alt_item.setData(Qt.UserRole, dso_data['max_altitude'])  # Store numeric value for sorting
            alt_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 4, alt_item)
            
            # Best time - store time as sortable value
            time_str = dso_data["optimal_time"].strftime("%H:%M")
            time_item = QTableWidgetItem()
            time_item.setData(Qt.DisplayRole, time_str)
            time_item.setData(Qt.UserRole, dso_data["optimal_time"].hour * 60 + dso_data["optimal_time"].minute)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 5, time_item)
            
            # Direction
            dir_item = QTableWidgetItem(dso_data["direction"])
            dir_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 6, dir_item)
            
            # Visible hours - use numeric sorting
            hours_item = QTableWidgetItem()
            hours_item.setData(Qt.DisplayRole, f"{dso_data['visible_hours']:.1f}h")
            hours_item.setData(Qt.UserRole, dso_data['visible_hours'])  # Store numeric value for sorting
            hours_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 7, hours_item)
        
        # Re-enable sorting
        self.results_table.setSortingEnabled(True)

    def on_item_double_clicked(self, item):
        """Handle double-click on table item to show object details"""
        try:
            # Get DSO data from the name item (column 0) of the clicked row
            name_item = self.results_table.item(item.row(), 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    self.show_object_detail(dso_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open object details: {e}")
    
    def show_object_detail(self, dso_data):
        """Show detailed information for the selected DSO"""
        try:
            # Import ObjectDetailWindow from Main module
            import sys
            import os
            
            # Add the directory containing Main.py to the Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from main import ObjectDetailWindow
            
            # Get DSO coordinates from astropy result
            coordinates = dso_data.get("coordinates")
            if coordinates:
                ra_deg = coordinates.ra.degree
                dec_deg = coordinates.dec.degree
            else:
                # Fallback - try to get coordinates again
                from astropy.coordinates import SkyCoord
                try:
                    coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                    ra_deg = coord.ra.degree
                    dec_deg = coord.dec.degree
                except:
                    ra_deg = 0.0
                    dec_deg = 0.0
            
            # Create the data dictionary expected by ObjectDetailWindow
            detail_data = {
                "name": dso_data["dso_info"]["name"],
                "type": dso_data["dso_info"]["type"],
                "constellation": dso_data["dso_info"]["constellation"],
                "magnitude": dso_data["dso_info"]["magnitude"],
                "surface_brightness": dso_data["dso_info"]["surface_brightness"],
                "size_min": dso_data["dso_info"]["size_min"],
                "size_max": dso_data["dso_info"]["size_max"],
                "dso_type": dso_data["dso_info"]["type"],
                "dso_class": dso_data["dso_info"]["dso_class"],
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "catalogue": dso_data["dso_info"]["name"].split()[0],  # Extract catalog prefix
                "id": dso_data["dso_info"]["name"].split()[1] if len(dso_data["dso_info"]["name"].split()) > 1 else "",
                "designations": dso_data["dso_info"]["designations"],
                # Add visibility info as additional data
                "visibility_info": {
                    "max_altitude": dso_data["max_altitude"],
                    "optimal_time": dso_data["optimal_time"],
                    "direction": dso_data["direction"],
                    "visible_hours": dso_data["visible_hours"]
                }
            }
            
            # Create and show the detail window
            detail_window = ObjectDetailWindow(detail_data, self)
            detail_window.show()
            
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not load ObjectDetailWindow: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open object details: {e}")

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
        return type_mapping.get(dso_type, dso_type)  # Return original if not found

    def handle_error(self, error_msg):
        """Handle calculation errors"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Best DSOs Tonight")
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.warning(self, "Calculation Error", error_msg)

    def _show_context_menu(self, position):
        """Show context menu when right-clicking on the DSO table"""
        # Get the item at the clicked position
        index = self.results_table.indexAt(position)
        if not index.isValid():
            return  # No item at this position

        # Get the row number
        row = index.row()
        if row < 0 or row >= self.results_table.rowCount():
            return

        # Create context menu
        context_menu = QMenu(self)

        # Apply dark theme styling to the context menu
        context_menu.setStyleSheet("""
            QMenu {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 2px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 8px 16px;
                border: none;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QMenu::item:hover {
                background-color: #0078d4;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #666666;
                margin: 2px 8px;
            }
        """)

        # Add menu actions
        details_action = context_menu.addAction("View DSO Details")
        details_action.triggered.connect(lambda: self._context_view_details(row))

        visibility_action = context_menu.addAction("Open DSO Visibility Calculator")
        visibility_action.triggered.connect(lambda: self._context_open_visibility(row))

        aladin_action = context_menu.addAction("Open in Aladin Lite")
        aladin_action.triggered.connect(lambda: self._context_open_aladin(row))

        context_menu.addSeparator()

        target_action = context_menu.addAction("Add to Target List")
        target_action.triggered.connect(lambda: self._context_add_to_target_list(row))

        # Show the menu at the clicked position
        context_menu.exec(self.results_table.mapToGlobal(position))

    def _context_view_details(self, row):
        """View DSO details from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    self.show_object_detail(dso_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open DSO details: {str(e)}")

    def _context_open_visibility(self, row):
        """Open DSO Visibility Calculator from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    dso_name = dso_data["dso_info"]["name"]

                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_name)
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Import and open DSO Visibility Calculator
                    from DSOVisibilityCalculator import DSOVisibilityApp

                    # Store reference to prevent garbage collection
                    self.visibility_window = DSOVisibilityApp()

                    # Use coordinates instead of name for more reliable calculation
                    # Format coordinates as a string that astropy can parse
                    coord_string = f"{ra_deg:.6f} {dec_deg:+.6f}"

                    # Pre-populate with the coordinates
                    if hasattr(self.visibility_window, 'dso_input'):
                        self.visibility_window.dso_input.setText(coord_string)

                    # Show the window immediately
                    self.visibility_window.show()
                    self.visibility_window.raise_()
                    self.visibility_window.activateWindow()

                    # Automatically trigger calculation after a short delay
                    if hasattr(self.visibility_window, 'calculate_visibility'):
                        QTimer.singleShot(500, self.visibility_window.calculate_visibility)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open DSO Visibility Calculator: {str(e)}")

    def _context_open_aladin(self, row):
        """Open Aladin Lite from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Create data dictionary for Aladin Lite
                    detail_data = {
                        'name': dso_data["dso_info"]["name"],
                        'ra_deg': ra_deg,
                        'dec_deg': dec_deg,
                        'size_min': dso_data["dso_info"]["size_min"],
                        'size_max': dso_data["dso_info"]["size_max"],
                        'dsodetailid': dso_data["dso_info"]["name"]
                    }

                    # Import and open Aladin Lite window
                    from main import AladinLiteWindow
                    aladin_window = AladinLiteWindow(detail_data, self)
                    aladin_window.setModal(False)
                    aladin_window.show()
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

    def _context_add_to_target_list(self, row):
        """Add DSO to target list from context menu"""
        try:
            # Get DSO data from the name item (column 0) of the selected row
            name_item = self.results_table.item(row, 0)
            if name_item:
                dso_data = name_item.data(Qt.UserRole)
                if dso_data:
                    # Get coordinates
                    coordinates = dso_data.get("coordinates")
                    if coordinates:
                        ra_deg = coordinates.ra.degree
                        dec_deg = coordinates.dec.degree
                    else:
                        # Fallback - try to get coordinates again
                        from astropy.coordinates import SkyCoord
                        try:
                            coord = SkyCoord.from_name(dso_data["dso_info"]["name"])
                            ra_deg = coord.ra.degree
                            dec_deg = coord.dec.degree
                        except:
                            ra_deg = 0.0
                            dec_deg = 0.0

                    # Create data dictionary for target list
                    target_data = {
                        'name': dso_data["dso_info"]["name"],
                        'ra_deg': ra_deg,
                        'dec_deg': dec_deg,
                        'magnitude': dso_data["dso_info"]["magnitude"],
                        'size_min': dso_data["dso_info"]["size_min"],
                        'size_max': dso_data["dso_info"]["size_max"],
                        'constellation': dso_data["dso_info"]["constellation"],
                        'dso_type': dso_data["dso_info"]["type"],
                        'dso_class': dso_data["dso_info"]["dso_class"]
                    }

                    # Import and open Target List window, then add the DSO
                    from DSOTargetList import DSOTargetListWindow
                    if not hasattr(self, 'target_list_window') or not self.target_list_window.isVisible():
                        self.target_list_window = DSOTargetListWindow()

                    self.target_list_window.show()
                    self.target_list_window.raise_()
                    self.target_list_window.activateWindow()

                    # Add the DSO to the target list
                    self.target_list_window.add_target_from_dso(target_data)
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve DSO data from selected row")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add to target list: {str(e)}")


def main():
    """Main entry point for the application"""
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = BestDSOTonightWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()