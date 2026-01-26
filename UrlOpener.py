#!/usr/bin/env python3
"""
URL Opener utility for Cosmos Collection
Provides cross-platform URL opening that avoids shell-related issues on Linux
"""

import logging
import subprocess
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

logger = logging.getLogger(__name__)


def open_url(url):
    """
    Open a URL in the default browser.

    On Linux, uses xdg-open directly to avoid shell-related issues
    (e.g., readline symbol errors on Arch Linux).
    On other platforms, uses Qt's QDesktopServices.

    Args:
        url: URL string or QUrl object to open

    Returns:
        bool: True if successful, False otherwise
    """
    # Convert QUrl to string if needed
    if isinstance(url, QUrl):
        url_str = url.toString()
    else:
        url_str = str(url)

    try:
        if sys.platform.startswith('linux'):
            # On Linux, use xdg-open directly to avoid shell/readline issues
            # Using start_new_session=True to fully detach from parent process
            subprocess.Popen(
                ['xdg-open', url_str],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            logger.debug(f"Opened URL via xdg-open: {url_str}")
            return True
        else:
            # On Windows/macOS, Qt's QDesktopServices works fine
            qurl = QUrl(url_str) if not isinstance(url, QUrl) else url
            result = QDesktopServices.openUrl(qurl)
            if result:
                logger.debug(f"Opened URL via QDesktopServices: {url_str}")
            else:
                logger.warning(f"QDesktopServices.openUrl returned False for: {url_str}")
            return result

    except Exception as e:
        logger.error(f"Error opening URL {url_str}: {e}")
        # Fallback to QDesktopServices
        try:
            qurl = QUrl(url_str) if not isinstance(url, QUrl) else url
            return QDesktopServices.openUrl(qurl)
        except Exception as fallback_e:
            logger.error(f"Fallback URL open also failed: {fallback_e}")
            return False
