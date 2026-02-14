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

    @staticmethod
    def slew_to_coordinates(ra_deg, dec_deg, target_name, parent_widget=None):
        """
        Slew mount to specified coordinates with confirmation dialog.

        Args:
            ra_deg: Right Ascension in degrees
            dec_deg: Declination in degrees
            target_name: Name of the target (for display)
            parent_widget: Parent widget for message boxes (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        if ra_deg is None or dec_deg is None:
            if parent_widget:
                QMessageBox.warning(parent_widget, "Error", "Target coordinates not available")
            logger.warning(f"Cannot slew to {target_name}: coordinates not available")
            return False

        host, port = NINAIntegration.get_settings()

        # Format coordinates for display
        ra_h = ra_deg / 15.0
        ra_hours = int(ra_h)
        ra_min = int((ra_h - ra_hours) * 60)
        ra_sec = ((ra_h - ra_hours) * 60 - ra_min) * 60

        dec_sign = '+' if dec_deg >= 0 else '-'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_m = int((dec_abs - dec_d) * 60)
        dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60

        coord_str = f"RA: {ra_hours:02d}h {ra_min:02d}m {ra_sec:05.2f}s\nDec: {dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:04.1f}\""

        # Show confirmation dialog
        if parent_widget:
            reply = QMessageBox.question(
                parent_widget,
                "Confirm Slew",
                f"Slew mount to {target_name}?\n\n{coord_str}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return False

        logger.info(f"Slewing mount to {target_name}: RA={ra_deg}, Dec={dec_deg}")

        try:
            success = NINAIntegration.slew_mount(host, port, ra_deg, dec_deg, wait_for_result=True)

            if success:
                logger.info(f"Slew to {target_name} completed successfully")
                if parent_widget:
                    QMessageBox.information(
                        parent_widget,
                        "Slew Complete",
                        f"Mount slew to {target_name} completed."
                    )
                return True
            else:
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Slew Failed",
                        f"Failed to slew to {target_name}.\n\n"
                        "Please check:\n"
                        "- Mount is connected in NINA\n"
                        "- Mount is not parked\n"
                        "- No other slew operation is in progress"
                    )
                return False

        except urllib.error.HTTPError as e:
            if e.code == 409:
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Slew Failed",
                        "Mount is not available for slewing.\n\n"
                        "Please check:\n"
                        "- Mount is connected in NINA\n"
                        "- Mount is not parked"
                    )
            else:
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Slew Failed",
                        f"HTTP error {e.code} when slewing.\n\n"
                        f"Error: {e.reason}"
                    )
            logger.error(f"HTTP error slewing to {target_name}: {e.code} {e.reason}")
            return False

        except urllib.error.URLError as e:
            logger.warning(f"Could not connect to NINA: {e}")
            if parent_widget:
                QMessageBox.warning(
                    parent_widget,
                    "Connection Error",
                    "Could not connect to NINA.\n\n"
                    "Please ensure:\n"
                    "- NINA is running\n"
                    "- The Advanced API plugin is enabled"
                )
            return False

        except Exception as e:
            logger.error(f"Error slewing to {target_name}: {e}")
            if parent_widget:
                QMessageBox.warning(
                    parent_widget,
                    "Error",
                    f"Failed to slew to target: {str(e)}"
                )
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
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting camera info: {e}")
            return None

    @staticmethod
    def set_camera_cooling(host, port, enabled, temperature=None):
        """
        Enable or disable camera cooling.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            enabled: True to enable cooling, False to disable
            temperature: Target temperature in Celsius (optional, only used when enabling)

        Returns:
            bool: True on success, False on failure
        """
        if enabled:
            if temperature is not None:
                # Use minutes=-1 for default duration
                url = f"http://{host}:{port}/v2/api/equipment/camera/cool?temperature={temperature}&minutes=-1"
        else:
            # Cancel cooling
            url = f"http://{host}:{port}/v2/api/equipment/camera/cool?cancel=true"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to set camera cooling: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error setting camera cooling: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error setting camera cooling: {e}")
            return False

    @staticmethod
    def set_camera_dew_heater(host, port, enabled):
        """
        Enable or disable camera dew heater.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            enabled: True to enable dew heater, False to disable

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/camera/dew-heater?power={'true' if enabled else 'false'}"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to set dew heater: {result.get('Error', 'Unknown error')}")
                return success
        except Exception as e:
            logger.error(f"Error setting dew heater: {e}")
            return False

    @staticmethod
    def home_mount(host, port):
        """
        Home the mount.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/mount/home"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to home mount: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error homing mount: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error homing mount: {e}")
            return False

    @staticmethod
    def park_mount(host, port):
        """
        Park the mount.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/mount/park"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to park mount: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error parking mount: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error parking mount: {e}")
            return False

    @staticmethod
    def slew_mount(host, port, ra_deg, dec_deg, wait_for_result=False):
        """
        Slew the mount to specified coordinates.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            ra_deg: Right Ascension in degrees
            dec_deg: Declination in degrees
            wait_for_result: Whether to wait for slew to complete

        Returns:
            bool: True on success, False on failure
        """
        params = [f"ra={ra_deg}", f"dec={dec_deg}"]
        if wait_for_result:
            params.append("waitForResult=true")

        url = f"http://{host}:{port}/v2/api/equipment/mount/slew?{'&'.join(params)}"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            # Longer timeout if waiting for result
            timeout = 120 if wait_for_result else 10
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to slew mount: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error slewing mount: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error slewing mount: {e}")
            return False

    @staticmethod
    def unpark_mount(host, port):
        """
        Unpark the mount.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/mount/unpark"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to unpark mount: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error unparking mount: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error unparking mount: {e}")
            return False

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
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting mount info: {e}")
            return None

    @staticmethod
    def start_guiding(host, port, calibrate=False):
        """
        Start guiding.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            calibrate: Whether to force calibration before guiding

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/guider/start"
        if calibrate:
            url += "?calibrate=true"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to start guiding: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error starting guiding: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error starting guiding: {e}")
            return False

    @staticmethod
    def stop_guiding(host, port):
        """
        Stop guiding.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/guider/stop"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to stop guiding: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error stopping guiding: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error stopping guiding: {e}")
            return False

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
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting guider info: {e}")
            return None

    @staticmethod
    def get_filterwheel_info(host, port):
        """
        Get filter wheel equipment information from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Filter wheel info dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/filterwheel/info"
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting filter wheel info: {e}")
            return None

    @staticmethod
    def change_filter(host, port, filter_id):
        """
        Change the active filter on the filter wheel.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            filter_id: The ID of the filter to change to

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/filterwheel/change-filter?filterId={filter_id}"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to change filter: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error changing filter: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error changing filter: {e}")
            return False

    @staticmethod
    def capture_image(host, port, duration=None, gain=None, save=True, image_type="SNAPSHOT"):
        """
        Start a camera capture.

        Args:
            host: NINA host
            port: NINA port
            duration: Exposure duration in seconds (optional)
            gain: Camera gain (optional)
            save: Save image to disk (default True)
            image_type: LIGHT, DARK, BIAS, FLAT, or SNAPSHOT

        Returns:
            bool: True on success, False on failure
        """
        params = [f"imageType={image_type}", f"save={'true' if save else 'false'}"]
        if duration is not None:
            params.append(f"duration={duration}")
        if gain is not None:
            params.append(f"gain={gain}")

        url = f"http://{host}:{port}/v2/api/equipment/camera/capture?{'&'.join(params)}"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to capture image: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error capturing image: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return False

    @staticmethod
    def abort_exposure(host, port):
        """
        Abort the current camera exposure.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/camera/abort-exposure"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to abort exposure: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error aborting exposure: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error aborting exposure: {e}")
            return False

    @staticmethod
    def start_autofocus(host, port):
        """
        Start an autofocus run.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/focuser/auto-focus"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to start autofocus: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error starting autofocus: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error starting autofocus: {e}")
            return False

    @staticmethod
    def cancel_autofocus(host, port):
        """
        Cancel a running autofocus.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            bool: True on success, False on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/focuser/auto-focus?cancel=true"

        logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.debug(f"API Response: {result}")
                success = result.get('Success', False)
                if not success:
                    logger.warning(f"Failed to cancel autofocus: {result.get('Error', 'Unknown error')}")
                return success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            logger.error(f"HTTP Error canceling autofocus: {e.code} {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Error canceling autofocus: {e}")
            return False

    @staticmethod
    def get_focuser_info(host, port):
        """
        Get focuser equipment information from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Focuser info dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/focuser/info"
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting focuser info: {e}")
            return None

    @staticmethod
    def get_capture_statistics(host, port):
        """
        Get capture statistics from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Capture statistics dict on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/camera/capture/statistics"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    resp = result.get('Response')
                    return resp if isinstance(resp, dict) else None
                return None
        except Exception as e:
            logger.debug(f"Error getting capture statistics: {e}")
            return None

    @staticmethod
    def get_image_statistics(host, port, index=None):
        """
        Get image history/statistics from NINA using the image-history endpoint.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            index: Optional image index to get stats for a specific image.
                   If None, gets the latest image (all=false).

        Returns:
            dict: Image history entry on success, None on failure.
                  Keys include: Stars, HFR, HFRStDev, Median, Mean, StDev, Min, Max,
                  ExposureTime, Filter, Gain, Offset, Temperature, TargetName,
                  ImageType, Filename, Date, CameraName, TelescopeName, etc.
        """
        params = []
        if index is not None:
            params.append(f"index={index}")
        else:
            params.append("all=false")
        params.append("imageType=LIGHT")
        query = "&".join(params)
        url = f"http://{host}:{port}/v2/api/image-history?{query}"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                resp = result.get('Response')
                if isinstance(resp, list) and len(resp) > 0:
                    return resp[-1]  # Return the last (most recent) entry
                elif isinstance(resp, dict):
                    return resp
                return None
        except Exception as e:
            logger.debug(f"Error getting image history: {e}")
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
    def get_image_thumbnail(host, port, index=0, size=300, quality=-1, size_wh=None):
        """
        Get an image thumbnail from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            index: Image index (0 = most recent, or use negative for latest)
            size: Thumbnail size in pixels (default 300, used as fallback)
            quality: JPEG quality 1-100, or -1 for PNG (default -1)
            size_wh: Size as "WxH" string (e.g. "800x600"). If provided, uses
                     resize=true with this size instead of the simple size param.

        Returns:
            tuple: (bytes image data, dict metadata) on success, (None, None) on failure
        """
        # Build URL based on quality and size parameters
        if size_wh and quality != -1:
            # JPEG with resize
            url = (f"http://{host}:{port}/v2/api/image/thumbnail/{index}"
                   f"?quality={quality}&stream=true&resize=true&size={size_wh}&debayer=true")
        elif size_wh:
            # PNG with resize - extract width from WxH for size param
            w = size_wh.split('x')[0] if 'x' in size_wh else size_wh
            url = f"http://{host}:{port}/v2/api/image/thumbnail/{index}?size={w}&stream=true&debayer=true"
        else:
            url = f"http://{host}:{port}/v2/api/image/thumbnail/{index}?size={size}&stream=true&debayer=true"

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
    def get_image(host, port, index=0, quality=-1, size_wh=None, progress_callback=None):
        """
        Get a full image from NINA using the /image/{index} endpoint.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            index: Image index (0 = most recent)
            quality: JPEG quality 1-100, or -1 for PNG (default -1)
            size_wh: Size as "WxH" string (e.g. "800x600"). If provided, uses
                     resize=true with this size.
            progress_callback: Optional callable(bytes_received, total_bytes).
                               total_bytes is -1 if Content-Length is not available.

        Returns:
            tuple: (bytes image data, dict metadata) on success, (None, None) on failure
        """
        params = ["stream=true", "debayer=true"]
        if quality != -1:
            params.append(f"quality={quality}")
        if size_wh:
            params.append("resize=true")
            params.append(f"size={size_wh}")
        query = "&".join(params)
        url = f"http://{host}:{port}/v2/api/image/{index}?{query}"

        logger.debug(f"Fetching image from: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=15) as response:
                content_type = response.headers.get('Content-Type', '')
                logger.debug(f"Image response Content-Type: {content_type}")
                if 'image' in content_type:
                    total = int(response.headers.get('Content-Length', -1))
                    if progress_callback and total > 0:
                        chunks = []
                        received = 0
                        chunk_size = 65536
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            received += len(chunk)
                            progress_callback(received, total)
                        data = b''.join(chunks)
                    else:
                        if progress_callback:
                            progress_callback(0, -1)
                        data = response.read()
                    logger.debug(f"Image data size: {len(data)} bytes")
                    return data, {}
                data = response.read().decode('utf-8')
                logger.debug(f"Image JSON response: {data[:500]}")
                return None, None
        except urllib.error.HTTPError as e:
            logger.debug(f"HTTP Error getting image: {e.code} {e.reason} for URL {url}")
            return None, None
        except Exception as e:
            logger.debug(f"Error getting image: {e}")
            return None, None

    @staticmethod
    def get_prepared_image(host, port, quality=80, size_wh="800x600"):
        """
        Get the camera's current prepared image (what NINA shows in its imaging tab).

        Useful for framing, focusing, and monitoring the camera in real-time.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            quality: JPEG quality 1-100 (default 80)
            size_wh: Size as "WxH" string (default "800x600")

        Returns:
            bytes: Image data on success, None on failure
        """
        url = (f"http://{host}:{port}/v2/api/prepared-image"
               f"?quality={quality}&resize=true&size={size_wh}"
               f"&stream=true&autoPrepare=true&debayer=true")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=3) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    data = response.read()
                    return data
                return None
        except Exception as e:
            logger.debug(f"Error getting prepared image: {e}")
            return None

    @staticmethod
    def get_livestack_status(host, port):
        """
        Get live stacking status from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            dict: Livestack status dict with 'running' key on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/livestack/status"
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response')
                    # API returns "running" or "stopped" string
                    if isinstance(resp, str):
                        return {'running': resp.lower() == 'running'}
                    # Handle dict response for compatibility
                    elif isinstance(resp, dict):
                        return resp
                return None
        except Exception as e:
            logger.debug(f"Error getting livestack status: {e}")
            return None

    @staticmethod
    def get_livestack_available(host, port):
        """
        Get available livestack targets and filters from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            list: List of dicts with 'Target' and 'Filter' keys, empty list on failure
        """
        try:
            available_url = f"http://{host}:{port}/v2/api/livestack/image/available"
            request = urllib.request.Request(available_url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success') and result.get('Response'):
                    return result.get('Response', [])
                return []
        except Exception as e:
            logger.debug(f"Error getting livestack available: {e}")
            return []

    @staticmethod
    def resolve_livestack_selection(stacks, target=None, filter_name=None):
        """
        Resolve which livestack target/filter to use from available stacks.

        Args:
            stacks: List of available stack dicts with 'Target' and 'Filter' keys
            target: Optional target name (if None, uses first available)
            filter_name: Optional filter name (if None, prefers RGB then first available)

        Returns:
            tuple: (str, str) - Resolved target name and filter name
                   (None, None) on failure
        """
        if not stacks:
            return None, None

        selected = None

        if target and filter_name:
            for stack in stacks:
                if stack.get('Target') == target and stack.get('Filter') == filter_name:
                    selected = stack
                    break
        elif target:
            matching = [s for s in stacks if s.get('Target') == target]
            if matching:
                for stack in matching:
                    if stack.get('Filter') == 'RGB':
                        selected = stack
                        break
                if not selected:
                    selected = matching[0]
        elif filter_name:
            for stack in stacks:
                if stack.get('Filter') == filter_name:
                    selected = stack
                    break

        if not selected:
            for stack in stacks:
                if stack.get('Filter') == 'RGB':
                    selected = stack
                    break
            if not selected:
                selected = stacks[0]

        actual_target = selected.get('Target', '')
        actual_filter = selected.get('Filter', '')
        if not actual_target or not actual_filter:
            return None, None
        return actual_target, actual_filter

    @staticmethod
    def fetch_livestack_image_data(host, port, target, filter_name, quality=100, size="800x600",
                                    progress_callback=None):
        """
        Fetch the livestack image for a resolved target/filter.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            target: Resolved target name
            filter_name: Resolved filter name
            quality: JPEG quality 1-100, or -1 for PNG (default 100)
            size: Size as "WxH" string (default "800x600")
            progress_callback: Optional callable(bytes_received, total_bytes).
                               total_bytes is -1 if Content-Length is not available.

        Returns:
            bytes: Image data on success, None on failure
        """
        try:
            encoded_target = urllib.parse.quote(target, safe='')
            encoded_filter = urllib.parse.quote(filter_name, safe='')
            image_url = (f"http://{host}:{port}/v2/api/livestack/image/"
                         f"{encoded_target}/{encoded_filter}"
                         f"?quality={quality}&stream=true&resize=true&size={size}")
            request = urllib.request.Request(image_url)
            with urllib.request.urlopen(request, timeout=10) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    total = int(response.headers.get('Content-Length', -1))
                    if progress_callback and total > 0:
                        chunks = []
                        received = 0
                        chunk_size = 65536
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            received += len(chunk)
                            progress_callback(received, total)
                        data = b''.join(chunks)
                    else:
                        data = response.read()
                    logger.debug(f"API Response: livestack image data, {len(data)} bytes")
                    return data
                return None
        except Exception as e:
            logger.debug(f"Error fetching livestack image: {e}")
            return None

    @staticmethod
    def get_livestack_image(host, port, target=None, filter_name=None):
        """
        Get the current live-stacked image from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number
            target: Optional target name to fetch (if None, uses first available)
            filter_name: Optional filter name to fetch (if None, prefers RGB then first available)

        Returns:
            tuple: (bytes, str, str) - Image data (JPEG), target name, filter name on success
                   (None, None, None) on failure
        """
        stacks = NINAIntegration.get_livestack_available(host, port)
        actual_target, actual_filter = NINAIntegration.resolve_livestack_selection(
            stacks, target, filter_name
        )
        if not actual_target:
            return None, None, None
        image_data = NINAIntegration.fetch_livestack_image_data(
            host, port, actual_target, actual_filter
        )
        if image_data:
            return image_data, actual_target, actual_filter
        return None, None, None

    @staticmethod
    def get_livestack_info(host, port, target, filter_name):
        """Get live stack info (stack count, etc.) from NINA."""
        try:
            encoded_target = urllib.parse.quote(target, safe='')
            encoded_filter = urllib.parse.quote(filter_name, safe='')
            url = (f"http://{host}:{port}/v2/api/livestack/image/"
                   f"{encoded_target}/{encoded_filter}/info")
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data.get('Success') and data.get('Response'):
                    logger.debug(f"API Response: livestack info for {target}/{filter_name}")
                    return data['Response']
                return None
        except Exception as e:
            logger.debug(f"Error getting livestack info: {e}")
            return None

    @staticmethod
    def get_event_history(host, port):
        """
        Get event history from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            list: List of event dicts with 'Event', 'Time', and optional data fields
        """
        url = f"http://{host}:{port}/v2/api/event-history"
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('Success'):
                    return result.get('Response', [])
                return []
        except Exception as e:
            logger.debug(f"Error getting event history: {e}")
            return []

    @staticmethod
    def get_guiding_graph_data(host, port):
        """
        Get guiding graph data (RA/Dec deviations) from NINA.

        Args:
            host: The hostname or IP address of the NINA instance
            port: The API port number

        Returns:
            list: List of guiding data points (GuideSteps) on success, None on failure
        """
        url = f"http://{host}:{port}/v2/api/equipment/guider/graph"
        #logger.debug(f"API Request: {url}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                #logger.debug(f"API Response: {result}")
                if result.get('Success'):
                    resp = result.get('Response', {})
                    if isinstance(resp, dict):
                        # Response contains GuideSteps array with the graph data
                        guide_steps = resp.get('GuideSteps', [])
                        return guide_steps if isinstance(guide_steps, list) else None
                    return None
                return None
        except urllib.error.HTTPError as e:
            # 404 is expected when guider is not connected or guiding not active
            if e.code != 404:
                logger.debug(f"Error getting guiding graph data: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error getting guiding graph data: {e}")
            return None
