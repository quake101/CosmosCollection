#!/usr/bin/env python3
"""
NINA Integration Module for Cosmos Collection
Provides reusable NINA Advanced API integration functionality
https://github.com/christian-photo/ninaAPI
"""

import json
import logging
import urllib.request
import urllib.parse
import urllib.error

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

# Set up logging
logger = logging.getLogger(__name__)


class NINAIntegration:
    """
    Static class providing NINA Advanced API integration functionality.

    Settings keys used:
    - nina_integration_enabled (bool, default: False)
    - nina_api_host (str, default: "localhost")
    - nina_api_port (int, default: 1888)
    """

    @staticmethod
    def is_enabled():
        """
        Check if NINA integration is enabled in settings.

        Returns:
            bool: True if NINA integration is enabled, False otherwise
        """
        settings = QSettings("CosmosCollection", "CosmosCollection")
        return settings.value("nina_integration_enabled", False, type=bool)

    @staticmethod
    def get_settings():
        """
        Get NINA API host and port from QSettings.

        Returns:
            tuple: (host: str, port: int)
        """
        settings = QSettings("CosmosCollection", "CosmosCollection")
        host = settings.value("nina_api_host", "localhost", type=str)
        port = settings.value("nina_api_port", 1888, type=int)
        return host, port

    @staticmethod
    def test_connection(host, port):
        """
        Test the connection to NINA's Advanced API.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            tuple: (success: bool, message: str, version: str or None)
        """
        url = f"http://{host}:{port}/v2/api/version"

        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))

                if result.get('Success'):
                    version_info = result.get('Response', 'Unknown')
                    return True, f"Successfully connected to NINA!\n\nAPI Version: {version_info}", version_info
                else:
                    error_msg = result.get('Error', 'Unknown error')
                    return False, f"NINA returned an error: {error_msg}", None

        except urllib.error.URLError as e:
            return False, (
                f"Could not connect to NINA at {host}:{port}\n\n"
                "Please ensure:\n"
                "- NINA is running\n"
                "- The Advanced API plugin is installed and enabled\n"
                f"- The host ({host}) and port ({port}) are correct"
            ), None
        except Exception as e:
            logger.error(f"Error testing NINA connection: {e}")
            return False, f"Connection test failed: {str(e)}", None

    @staticmethod
    def send_to_framing_assistant(ra_deg, dec_deg, target_name, parent_widget=None):
        """
        Send coordinates to NINA Framing Assistant and switch to the framing tab.

        Args:
            ra_deg: Right Ascension in degrees
            dec_deg: Declination in degrees
            target_name: Name of the target (for logging)
            parent_widget: Parent widget for message boxes (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        if ra_deg is None or dec_deg is None:
            if parent_widget:
                QMessageBox.warning(parent_widget, "Error", "Target coordinates not available")
            logger.warning(f"Cannot send {target_name} to NINA: coordinates not available")
            return False

        host, port = NINAIntegration.get_settings()

        # Build NINA API URL for setting coordinates
        base_url = f"http://{host}:{port}/v2/api/framing/set-coordinates"
        params = urllib.parse.urlencode({
            'RAangle': ra_deg,
            'DecAngle': dec_deg
        })
        url = f"{base_url}?{params}"

        logger.debug(f"Sending {target_name} to NINA: RA={ra_deg}, Dec={dec_deg}")

        try:
            # Send coordinates to framing assistant
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))

                if result.get('Success'):
                    logger.info(f"Sent {target_name} to NINA Framing Assistant")

                    # Switch to the Framing tab
                    switch_url = f"http://{host}:{port}/v2/api/application/switch-tab?tab=framing"
                    try:
                        switch_request = urllib.request.Request(switch_url)
                        with urllib.request.urlopen(switch_request, timeout=5) as switch_response:
                            switch_result = json.loads(switch_response.read().decode('utf-8'))
                            if switch_result.get('Success'):
                                logger.debug("Switched NINA to Framing tab")
                            else:
                                logger.warning(f"Could not switch to Framing tab: {switch_result.get('Error', 'Unknown')}")
                    except Exception as e:
                        logger.warning(f"Could not switch to Framing tab: {e}")

                    return True
                else:
                    error_msg = result.get('Error', 'Unknown error')
                    if parent_widget:
                        QMessageBox.warning(parent_widget, "NINA Error", f"NINA returned an error: {error_msg}")
                    logger.warning(f"NINA error when sending {target_name}: {error_msg}")
                    return False

        except urllib.error.URLError as e:
            logger.warning(f"Could not connect to NINA: {e}")
            if parent_widget:
                QMessageBox.warning(
                    parent_widget, "Connection Error",
                    "Could not connect to NINA.\n\n"
                    "Please ensure:\n"
                    "- NINA is running\n"
                    "- The Advanced API plugin is enabled\n"
                    "- The Framing Assistant tab has been opened at least once"
                )
            return False
        except Exception as e:
            logger.error(f"Error sending to NINA: {e}")
            if parent_widget:
                QMessageBox.warning(parent_widget, "Error", f"Failed to send to NINA: {str(e)}")
            return False

    # -------------------------------------------------------------------------
    # Dashboard API Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def get_camera_info(host, port):
        """
        Get camera equipment information from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Camera info dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/camera/info"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting camera info: {e}")
            return None

    @staticmethod
    def get_mount_info(host, port):
        """
        Get mount equipment information from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Mount info dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/mount/info"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting mount info: {e}")
            return None

    @staticmethod
    def get_guider_info(host, port):
        """
        Get guider equipment information from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Guider info dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/guider/info"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting guider info: {e}")
            return None

    @staticmethod
    def get_image_count(host, port):
        """
        Get the count of images by probing for the highest valid index.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            int: Number of images (highest index + 1), or 0 if no images
        """
        # The /v2/api/image/count endpoint is unreliable (returns 500)
        # Instead, probe for images by trying indices until we get a 404
        # Use binary search for efficiency

        # First check if there are any images at all
        if not NINAIntegration._image_exists(host, port, 0):
            return 0

        # Binary search to find the highest valid index
        low, high = 0, 1

        # First, find an upper bound by doubling
        while NINAIntegration._image_exists(host, port, high):
            low = high
            high *= 2
            if high > 1000:  # Safety limit
                break

        # Binary search between low and high
        while low < high - 1:
            mid = (low + high) // 2
            if NINAIntegration._image_exists(host, port, mid):
                low = mid
            else:
                high = mid

        # low is now the highest valid index
        count = low + 1
        logger.debug(f"Probed image count: {count} (highest index: {low})")
        return count

    @staticmethod
    def _image_exists(host, port, index):
        """Check if an image exists at the given index."""
        url = f"http://{host}:{port}/v2/api/image/thumbnail/{index}?size=50&stream=true"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=3) as response:
                content_type = response.headers.get('Content-Type', '')
                # Read a small amount to verify it's an image
                data = response.read(100)
                exists = 'image' in content_type and len(data) > 0
                logger.debug(f"Image exists check index {index}: {exists} (content-type: {content_type})")
                return exists
        except urllib.error.HTTPError as e:
            logger.debug(f"Image exists check index {index}: False (HTTP {e.code})")
            return False
        except Exception as e:
            logger.debug(f"Image exists check index {index}: False ({e})")
            return False

    @staticmethod
    def get_image_thumbnail(host, port, index=0, size=300):
        """
        Get an image thumbnail from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            index: Image index (0 = most recent, or use negative for latest)
            size: Thumbnail size in pixels (default 300)

        Returns:
            tuple: (bytes image data, dict metadata) on success, (None, None) on failure
        """
        # The API uses /image/thumbnail/{index} with size as query param
        # Try index -1 for latest, or 0 for first
        url = f"http://{host}:{port}/v2/api/image/thumbnail/{index}?size={size}&stream=true"

        logger.debug(f"Fetching image thumbnail from: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as response:
                content_type = response.headers.get('Content-Type', '')
                logger.debug(f"Thumbnail response Content-Type: {content_type}")
                if 'image' in content_type:
                    data = response.read()
                    logger.debug(f"Thumbnail data size: {len(data)} bytes")
                    return data, {}
                # API returned JSON - log it to see the error
                data = response.read().decode('utf-8')
                logger.debug(f"Thumbnail JSON response: {data[:500]}")
                return None, None
        except urllib.error.HTTPError as e:
            # Try alternative URL format without stream parameter
            if "stream" in url:
                alt_url = f"http://{host}:{port}/v2/api/image/thumbnail/{index}?size={size}"
                logger.debug(f"Trying alternative URL: {alt_url}")
                try:
                    request = urllib.request.Request(alt_url)
                    with urllib.request.urlopen(request, timeout=10) as response:
                        content_type = response.headers.get('Content-Type', '')
                        if 'image' in content_type:
                            data = response.read()
                            logger.debug(f"Alt thumbnail data size: {len(data)} bytes")
                            return data, {}
                except Exception:
                    pass
            logger.debug(f"HTTP Error getting image thumbnail: {e.code} {e.reason} for URL {url}")
            return None, None
        except Exception as e:
            logger.debug(f"Error getting image thumbnail: {e}")
            return None, None

    @staticmethod
    def get_livestack_status(host, port):
        """
        Get live stacking status from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Livestack status dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/livestack/status"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting livestack status: {e}")
            return None

    @staticmethod
    def get_livestack_image(host, port):
        """
        Get the current live-stacked image from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bytes: Image data (JPEG) on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/livestack/image"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    return response.read()
                return None
        except Exception as e:
            logger.debug(f"Error getting livestack image: {e}")
            return None

    @staticmethod
    def get_guiding_graph_data(host, port):
        """
        Get guiding graph data (RA/Dec deviations) from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            list: List of guiding data points on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/guider/graph"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, list) else None
                return None
        except urllib.error.HTTPError as e:
            # 404 is expected when guider is not connected or guiding not active
            if e.code != 404:
                logger.debug(f"Error getting guiding graph data: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error getting guiding graph data: {e}")
            return None
