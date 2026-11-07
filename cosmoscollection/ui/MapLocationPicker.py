"""Map dialog for picking an observing location (Leaflet map in a web view)."""

import json
import logging
import threading

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cosmoscollection.ui.Theme import COLORS, font_size, themed_style

logger = logging.getLogger(__name__)


# --- Map Location Picker Dialog ---
class MapBridge(QObject):
    """Bridge object for Qt-JavaScript communication in map picker"""

    searchCompleted = Signal(str)  # Signal to send search results back to JavaScript
    geocodeCompleted = Signal(str)  # Signal to send geocode results back to JavaScript

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    @Slot(float, float, str)
    def selectLocation(self, lat, lon, location_name):
        """Called from JavaScript when user selects a location on the map"""
        logger.debug(f"MapBridge.selectLocation called: lat={lat}, lon={lon}, name={location_name}")
        self.dialog._on_location_selected(lat, lon, location_name)

    @Slot(str)
    def searchLocationFromPython(self, query):
        """Search for location using Python (to avoid CORS issues)"""
        logger.debug(f"MapBridge.searchLocationFromPython called: query={query}")
        self._run_in_background(self.searchCompleted, self.dialog._search_location_python, query)

    @Slot(float, float)
    def reverseGeocodeFromPython(self, lat, lon):
        """Reverse geocode using Python (to avoid CORS issues)"""
        logger.debug(f"MapBridge.reverseGeocodeFromPython called: lat={lat}, lon={lon}")
        self._run_in_background(self.geocodeCompleted, self.dialog._reverse_geocode_python, lat, lon)

    @staticmethod
    def _run_in_background(result_signal, lookup, *args):
        """Run a Nominatim lookup off the UI thread (it can take up to its 10s timeout);
        the result signal is delivered back to the web channel on the UI thread."""
        def worker():
            result = lookup(*args)
            logger.debug(f"Emitting lookup result: {result}")
            try:
                result_signal.emit(result)
            except RuntimeError:
                pass  # Dialog closed while the lookup was running

        threading.Thread(target=worker, daemon=True).start()


class MapLocationPickerDialog(QDialog):
    """Dialog for selecting location coordinates from an interactive map"""

    def __init__(self, initial_lat=40.7128, initial_lon=-74.0060, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Location from Map - Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(900, 700)

        # Store initial coordinates
        self.initial_lat = initial_lat
        self.initial_lon = initial_lon

        # Selected coordinates (initialize with initial values so user can accept without clicking)
        self.selected_lat = initial_lat
        self.selected_lon = initial_lon
        self.selected_location_name = ""

        # Web view and bridge
        self.web_view = None
        self.bridge = None
        self.channel = None
        self._renderer_crashed = False  # Set by _on_render_process_terminated; suppresses the generic load-failed message
        self._gpu_reset_bar = None   # Shown when the map broke after a graphics reset
        self._init_failed = False  # Set when the map can't be created at all; exec() then returns immediately

        self._setup_ui()

        # Update coordinates display with initial position
        lat_str = f"{abs(initial_lat):.6f}°{'N' if initial_lat >= 0 else 'S'}"
        lon_str = f"{abs(initial_lon):.6f}°{'W' if initial_lon < 0 else 'E'}"
        self.coords_label.setText(f"Current position: {lat_str}, {lon_str}\nClick on the map to change location")

        # Create web view immediately instead of deferring
        self._load_map()

    def _setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()

        # Search section
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter city name, address, or landmark...")
        self.search_input.returnPressed.connect(self._on_search_clicked)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._on_search_clicked)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)

        # Placeholder for map (will be replaced by web view in _load_map)
        self.map_container = QWidget()
        self.map_container.setMinimumSize(800, 500)
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.map_container, 1)

        # Coordinates display
        self.coords_label = QLabel("Click on the map to select a location")
        self.coords_label.setAlignment(Qt.AlignCenter)
        themed_style(self.coords_label, lambda: f"QLabel {{ font-size: {font_size(12)}; padding: 10px; }}")
        layout.addWidget(self.coords_label)

        # Help text
        help_text = QLabel("Tip: You can pan, zoom, and search for locations. Click anywhere on the map to select coordinates.")
        help_text.setWordWrap(True)
        themed_style(help_text, lambda: f"QLabel {{ color: {COLORS['text_disabled']}; font-size: {font_size(9)}; padding: 5px; }}")
        help_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(help_text)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.select_button = QPushButton("Select Location")
        self.select_button.clicked.connect(self._on_select_clicked)
        self.select_button.setEnabled(True)  # Enabled since we have initial coordinates
        # Don't set as default to prevent event propagation to parent dialog
        # self.select_button.setDefault(True)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_map_html(self, lat, lon, zoom=10):
        """Create HTML with Leaflet map and JavaScript bridge"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Location Picker</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/qwebchannel@6.2.0/qwebchannel.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 100vh; }}
        .info-box {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            max-width: 500px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box" id="infoBox">Click on the map to select a location</div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var qt_bridge = null;
        var map = null;
        var marker = null;
        var searchTimeout = null;
        var bridgeReady = false;
        var currentGeocodeLat = null;
        var currentGeocodeLon = null;

        // Initialize QWebChannel for Qt-JavaScript communication
        new QWebChannel(qt.webChannelTransport, function(channel) {{
            qt_bridge = channel.objects.qt_bridge;
            bridgeReady = true;
            console.log("Qt bridge initialized successfully");

            // Connect to search results signal
            qt_bridge.searchCompleted.connect(function(resultJson) {{
                console.log("Received search results:", resultJson);
                handleSearchResults(resultJson);
            }});

            // Connect to geocode results signal
            qt_bridge.geocodeCompleted.connect(function(resultJson) {{
                console.log("Received geocode results:", resultJson);
                handleGeocodeResults(resultJson);
            }});
        }});

        // Initialize Leaflet map
        try {{
            map = L.map('map').setView([{lat}, {lon}], {zoom});

            // Add OpenStreetMap tiles
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 19
            }}).addTo(map);

            // Add initial marker at starting position
            marker = L.marker([{lat}, {lon}], {{
                draggable: true
            }}).addTo(map);

            // Handle marker drag
            marker.on('dragend', function(e) {{
                var latlng = e.target.getLatLng();
                onLocationSelected(latlng.lat, latlng.lng);
            }});

            // Handle map clicks
            map.on('click', function(e) {{
                var lat = e.latlng.lat;
                var lon = e.latlng.lng;

                // Update or create marker
                if (marker) {{
                    marker.setLatLng(e.latlng);
                }} else {{
                    marker = L.marker(e.latlng, {{
                        draggable: true
                    }}).addTo(map);

                    marker.on('dragend', function(e) {{
                        var latlng = e.target.getLatLng();
                        onLocationSelected(latlng.lat, latlng.lng);
                    }});
                }}

                onLocationSelected(lat, lon);
            }});

            document.getElementById('infoBox').innerHTML = 'Map loaded successfully - Click to select location';
        }} catch (e) {{
            console.error("Failed to initialize map:", e);
            document.getElementById('infoBox').innerHTML = 'Error loading map: ' + e.message;
        }}

        function onLocationSelected(lat, lon) {{
            // Update info box
            document.getElementById('infoBox').innerHTML =
                'Selected: ' + lat.toFixed(6) + '°, ' + lon.toFixed(6) + '° - Getting location name...';

            // Reverse geocode to get location name
            reverseGeocode(lat, lon);

            // Also send coordinates immediately to Qt (in case reverse geocode fails)
            sendToQt(lat, lon, "");
        }}

        function sendToQt(lat, lon, locationName) {{
            if (bridgeReady && qt_bridge) {{
                try {{
                    qt_bridge.selectLocation(lat, lon, locationName);
                    console.log("Sent to Qt: " + lat + ", " + lon);
                }} catch(e) {{
                    console.error("Error calling Qt bridge:", e);
                }}
            }} else {{
                console.warn("Qt bridge not ready yet, retrying in 100ms...");
                setTimeout(function() {{
                    sendToQt(lat, lon, locationName);
                }}, 100);
            }}
        }}

        function reverseGeocode(lat, lon) {{
            // Store current coordinates for when we receive the result
            currentGeocodeLat = lat;
            currentGeocodeLon = lon;

            // Use Python bridge to reverse geocode (avoids CORS issues)
            if (bridgeReady && qt_bridge && qt_bridge.reverseGeocodeFromPython) {{
                qt_bridge.reverseGeocodeFromPython(lat, lon);
            }} else {{
                console.warn('Bridge not ready or reverseGeocodeFromPython not available');
                document.getElementById('infoBox').innerHTML =
                    'Selected: ' + lat.toFixed(6) + '°, ' + lon.toFixed(6) + '°';
                // Send coordinates without name
                sendToQt(lat, lon, "");
            }}
        }}

        function handleGeocodeResults(resultJson) {{
            try {{
                var data = JSON.parse(resultJson);
                var lat = currentGeocodeLat;
                var lon = currentGeocodeLon;
                var locationName = data.display_name || "";

                // Update info box
                document.getElementById('infoBox').innerHTML =
                    'Selected: ' + lat.toFixed(6) + '°, ' + lon.toFixed(6) + '°<br>' +
                    '<small>' + locationName + '</small>';

                // Send to Qt with location name
                sendToQt(lat, lon, locationName);
            }} catch(error) {{
                console.error('Error handling geocode results:', error);
                document.getElementById('infoBox').innerHTML =
                    'Selected: ' + currentGeocodeLat.toFixed(6) + '°, ' + currentGeocodeLon.toFixed(6) + '°';
                sendToQt(currentGeocodeLat, currentGeocodeLon, "");
            }}
        }}

        // Search function (called from Qt)
        function searchLocation(query) {{
            if (!query || query.trim() === '') {{
                return;
            }}

            document.getElementById('infoBox').innerHTML = 'Searching for: ' + query + '...';

            // Use Python bridge to search (avoids CORS issues)
            if (bridgeReady && qt_bridge && qt_bridge.searchLocationFromPython) {{
                qt_bridge.searchLocationFromPython(query);
            }} else {{
                console.error('Bridge not ready or searchLocationFromPython not available');
                document.getElementById('infoBox').innerHTML = 'Search unavailable. Please try again in a moment.';
            }}
        }}

        function handleSearchResults(resultJson) {{
            try {{
                var data = JSON.parse(resultJson);

                if (data && data.length > 0) {{
                    var result = data[0];
                    var lat = parseFloat(result.lat);
                    var lon = parseFloat(result.lon);

                    // Pan to location
                    map.setView([lat, lon], 13);

                    // Update marker
                    if (marker) {{
                        marker.setLatLng([lat, lon]);
                    }} else {{
                        marker = L.marker([lat, lon], {{
                            draggable: true
                        }}).addTo(map);

                        marker.on('dragend', function(e) {{
                            var latlng = e.target.getLatLng();
                            onLocationSelected(latlng.lat, latlng.lng);
                        }});
                    }}

                    // Select this location
                    onLocationSelected(lat, lon);
                }} else {{
                    document.getElementById('infoBox').innerHTML = 'No results found';
                }}
            }} catch(error) {{
                console.error('Error handling search results:', error);
                document.getElementById('infoBox').innerHTML = 'Search failed. Please try again.';
            }}
        }}
    </script>
</body>
</html>"""
        return html_template

    def _load_map(self):
        """Create and load the web view with the map"""
        try:
            # Import QtWebEngine components
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
                from PySide6.QtWebEngineCore import QWebEngineSettings
                from PySide6.QtWebChannel import QWebChannel
            except ImportError as ie:
                QMessageBox.warning(self, "Feature Unavailable",
                    "Map picker requires QtWebEngine which is not available.\n\n"
                    "You can still enter coordinates manually.")
                logger.error(f"QtWebEngine not available: {ie}")
                self._init_failed = True
                return

            # Create web view
            self.web_view = QWebEngineView()
            self.web_view.setMinimumSize(800, 500)
            self.web_view.setContextMenuPolicy(Qt.DefaultContextMenu)

            # Configure web settings
            settings = self.web_view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

            # Enable developer tools for debugging (optional - don't fail if unavailable)
            try:
                settings.setAttribute(QWebEngineSettings.WebAttribute.DeveloperExtrasEnabled, True)
                logger.debug("Developer tools enabled for map picker")
            except AttributeError:
                logger.debug("DeveloperExtrasEnabled not available in this Qt version")
            except Exception as e:
                logger.debug(f"Could not enable developer tools: {e}")

            # Set up QWebChannel for Qt-JavaScript bridge BEFORE loading HTML
            self.bridge = MapBridge(self)
            self.channel = QWebChannel()
            self.channel.registerObject("qt_bridge", self.bridge)
            self.web_view.page().setWebChannel(self.channel)

            # Generate and load HTML
            html = self._create_map_html(self.initial_lat, self.initial_lon)
            self.web_view.setHtml(html)

            # Add load finished handler to check if everything loaded
            self.web_view.loadFinished.connect(self._on_map_loaded)

            # Catch GPU/renderer process crashes (see FOVSimulator's AladinLiteWindow for the
            # same pattern) so a crash shows a clear message instead of a silently dead dialog.
            self._renderer_crashed = False
            self.web_view.page().renderProcessTerminated.connect(self._on_render_process_terminated)

            # Add web view to map container
            self.map_layout.addWidget(self.web_view)

            logger.debug("Map view created successfully")
            logger.debug("_load_map() completed successfully - dialog should still be open")

        except Exception as e:
            logger.error(f"Failed to create map view: {e}", exc_info=True)
            QMessageBox.critical(self, "Error",
                f"Failed to load map picker: {str(e)}\n\n"
                "Please enter coordinates manually.")
            self._init_failed = True

    def exec(self):
        # _load_map runs from __init__, before the caller's exec(); reject() there
        # would be undone by exec() showing the dialog anyway
        if self._init_failed:
            return QDialog.Rejected
        return super().exec()

    def _on_map_loaded(self, success):
        """Called when the map page finishes loading"""
        if success:
            logger.debug("Map page loaded successfully")

            # Test basic JavaScript execution
            self.web_view.page().runJavaScript(
                "1 + 1",
                lambda result: logger.debug(f"JavaScript execution test (1+1): {result}")
            )

            # Check if QWebChannel is available
            self.web_view.page().runJavaScript(
                "typeof QWebChannel !== 'undefined'",
                lambda result: logger.debug(f"QWebChannel available: {result}")
            )

            # Check if qt object is available
            self.web_view.page().runJavaScript(
                "typeof qt !== 'undefined'",
                lambda result: logger.debug(f"qt object available: {result}")
            )

            # Check bridge status after a delay (give it time to initialize)
            QTimer.singleShot(2000, self._check_bridge_status)

            # No WebGL here usually means the web views' graphics are broken - after a
            # graphics reset, the map may not draw either
            self.web_view.page().runJavaScript(
                "(function () { var c = document.createElement('canvas');"
                " return !!(c.getContext('webgl2') || c.getContext('webgl')); })()",
                lambda ok: None if ok else self._check_gpu_reset()
            )
        else:
            # If the renderer already crashed, _on_render_process_terminated has already shown
            # a more accurate message -- don't stomp it with the generic one.
            if not self._renderer_crashed:
                logger.error("Map page failed to load")
                QMessageBox.warning(self, "Map Failed to Load",
                    "The map failed to load, possibly due to a network issue.\n\n"
                    "You can still enter coordinates manually.")
                self.reject()

    def _on_render_process_terminated(self, termination_status, exit_code):
        """Handle GPU/renderer process crash without crashing the whole app or leaving a dead dialog"""
        if self._renderer_crashed:
            # Chromium can fire this signal again (e.g. an automatic reload attempt) before
            # reject() below has actually torn the dialog down. Only handle the first one.
            return
        from PySide6.QtWebEngineCore import QWebEnginePage
        status_names = {
            QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus: "Normal",
            QWebEnginePage.RenderProcessTerminationStatus.AbnormalTerminationStatus: "Abnormal",
            QWebEnginePage.RenderProcessTerminationStatus.CrashedTerminationStatus: "Crashed",
            QWebEnginePage.RenderProcessTerminationStatus.KilledTerminationStatus: "Killed",
        }
        status_name = status_names.get(termination_status, f"Unknown({termination_status})")
        logger.error(f"Map picker renderer process terminated: {status_name}, exit code: {exit_code}")
        self._renderer_crashed = True
        if self._check_gpu_reset():
            return  # The bar offers Try Again / Restart App instead of closing the map

        QMessageBox.warning(self, "Map Renderer Crashed",
            "The map's renderer process crashed (usually a GPU driver issue).\n\n"
            "You can still enter coordinates manually.")
        self.reject()

    def _check_gpu_reset(self):
        """The map failed: if the graphics card was reset since the app started, say so
        and offer Try Again / Restart App (see core/GpuReset.py). True if shown."""
        from cosmoscollection.core.GpuReset import was_reset
        if not was_reset():
            return False
        if self._gpu_reset_bar is None:
            from cosmoscollection.ui.GpuResetBar import GpuResetBar
            self._gpu_reset_bar = GpuResetBar("the map", self._retry_after_gpu_reset,
                                              before_restart=self.reject)
            self.layout().insertWidget(1, self._gpu_reset_bar)  # Below the search row
        self._gpu_reset_bar.show()
        return True

    def _retry_after_gpu_reset(self):
        """Try Again on the graphics reset bar: load the map again where it was."""
        if self.web_view is None:
            return
        self._renderer_crashed = False
        self.web_view.setHtml(self._create_map_html(self.selected_lat, self.selected_lon))

    def _check_bridge_status(self):
        """Check if the JavaScript bridge is ready"""
        def log_bridge_status(result):
            logger.debug(f"Bridge ready status: {result}")
            if not result:
                logger.error("Bridge failed to initialize! Trying to manually trigger...")
                # Try to manually test the bridge by calling selectLocation
                self.web_view.page().runJavaScript(
                    "if (qt_bridge && qt_bridge.selectLocation) { qt_bridge.selectLocation(40.7128, -74.0060, 'Test Location'); } else { console.log('Bridge not available'); }",
                    lambda r: logger.debug(f"Manual bridge test result: {r}")
                )

        self.web_view.page().runJavaScript("bridgeReady", log_bridge_status)

    def _on_location_selected(self, lat, lon, location_name=""):
        """Called when user selects a location (from JavaScript bridge)"""
        logger.debug(f"_on_location_selected called: lat={lat}, lon={lon}, name={location_name}")
        self.selected_lat = lat
        self.selected_lon = lon
        self.selected_location_name = location_name

        # Update display
        lat_str = f"{abs(lat):.6f}°{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.6f}°{'W' if lon < 0 else 'E'}"

        if location_name:
            # Truncate long location names
            display_name = location_name if len(location_name) <= 60 else location_name[:57] + "..."
            self.coords_label.setText(f"Selected: {lat_str}, {lon_str}\n{display_name}")
        else:
            self.coords_label.setText(f"Selected: {lat_str}, {lon_str}")

        # Enable select button
        self.select_button.setEnabled(True)

        logger.debug(f"Location selected: {lat}, {lon} - {location_name}")

    def _search_location_python(self, query):
        """Search for location using Python's urllib to avoid CORS issues"""
        import urllib.request
        import urllib.parse
        import json
        from cosmoscollection.core.OnlineLimits import is_offline

        if is_offline():
            return json.dumps([])
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_query}&limit=1"

            # Create request with User-Agent header (required by Nominatim)
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'CosmosCollection/1.0'}
            )

            # Make the request
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                logger.debug(f"Search results: {data}")
                return json.dumps(data)  # Return as JSON string

        except Exception as e:
            logger.error(f"Error searching location: {e}")
            return json.dumps([])  # Return empty array on error

    def _reverse_geocode_python(self, lat, lon):
        """Reverse geocode using Python's urllib to avoid CORS issues"""
        import urllib.request
        import json
        from cosmoscollection.core.OnlineLimits import is_offline

        if is_offline():
            return json.dumps({})
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"

            # Create request with User-Agent header (required by Nominatim)
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'CosmosCollection/1.0'}
            )

            # Make the request
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                logger.debug(f"Reverse geocode result: {data}")
                return json.dumps(data)  # Return as JSON string

        except Exception as e:
            logger.error(f"Error reverse geocoding: {e}")
            return json.dumps({})  # Return empty object on error

    def _on_search_clicked(self):
        """Handle search button click"""
        query = self.search_input.text().strip()
        if query and self.web_view:
            # Call JavaScript function which will call back to Python for the actual search
            self.web_view.page().runJavaScript(f"searchLocation({json.dumps(query)})")

    def _on_select_clicked(self):
        """Handle Select Location button click"""
        self.accept()

    def _on_cancel_clicked(self):
        """Handle Cancel button click"""
        self.reject()

    def get_selected_coordinates(self):
        """Get the selected coordinates"""
        if self.selected_lat is not None and self.selected_lon is not None:
            return (self.selected_lat, self.selected_lon, self.selected_location_name)
        return None
