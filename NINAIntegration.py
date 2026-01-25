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
