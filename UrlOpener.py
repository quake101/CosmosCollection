#!/usr/bin/env python3
"""
URL Opener utility for Cosmos Collection
Provides cross-platform URL opening that avoids shell-related issues on Linux
"""

import logging
import os
import shutil
import subprocess
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

logger = logging.getLogger(__name__)


def open_url(url):
    """
    Open a URL in the default browser.

    On Linux, uses direct browser invocation or gio/xdg-open with a clean
    environment to avoid shell-related issues (e.g., readline symbol errors
    on Arch Linux).
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
            return _open_url_linux(url_str)
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
        return False


def clean_subprocess_env():
    """
    Environment for launching external programs, without the variables this
    app sets for its own use:
    - QTWEBENGINE_CHROMIUM_FLAGS (main.py) is read by every QtWebEngine app,
      e.g. PixInsight - its --enable-logging makes each of their Chromium
      helper processes flash a console window on Windows (and --log-file
      would send their logs into this app's chromium_debug.log).
    - SSL/CA bundle variables and XDG_CACHE_HOME point at this app's own
      certificates and cache folder.
    On Linux a PyInstaller-frozen build also exports its bundled
    LD_LIBRARY_PATH (and friends) to child processes, which causes library
    conflicts (e.g. readline symbol errors) in them, so those are removed too.
    """
    env = os.environ.copy()
    for var in ['QTWEBENGINE_CHROMIUM_FLAGS', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
                'XDG_CACHE_HOME']:
        env.pop(var, None)
    if sys.platform.startswith('linux'):
        for var in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
            env.pop(var, None)
    return env


def _open_url_linux(url_str):
    """
    Open URL on Linux with workarounds for shell/readline issues.

    Tries multiple methods in order of preference:
    1. gio open (modern GNOME/freedesktop method)
    2. xdg-open with clean environment
    3. Direct browser invocation
    """
    clean_env = clean_subprocess_env()

    # Method 1: Try gio open (GNOME/modern freedesktop)
    if shutil.which('gio'):
        try:
            subprocess.Popen(
                ['gio', 'open', url_str],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=clean_env,
                shell=False
            )
            logger.debug(f"Opened URL via gio: {url_str}")
            return True
        except Exception as e:
            logger.debug(f"gio open failed: {e}, trying xdg-open")

    # Method 2: Try xdg-open with clean environment
    if shutil.which('xdg-open'):
        try:
            subprocess.Popen(
                ['xdg-open', url_str],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=clean_env,
                shell=False
            )
            logger.debug(f"Opened URL via xdg-open: {url_str}")
            return True
        except Exception as e:
            logger.debug(f"xdg-open failed: {e}, trying direct browser")

    # Method 3: Try common browsers directly
    browsers = ['firefox', 'chromium', 'google-chrome', 'brave', 'vivaldi', 'opera']
    for browser in browsers:
        if shutil.which(browser):
            try:
                subprocess.Popen(
                    [browser, url_str],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=clean_env,
                    shell=False
                )
                logger.debug(f"Opened URL via {browser}: {url_str}")
                return True
            except Exception as e:
                logger.debug(f"{browser} failed: {e}")
                continue

    # Method 4: Last resort - QDesktopServices (may still have issues)
    logger.warning("All Linux URL open methods failed, falling back to QDesktopServices")
    try:
        return QDesktopServices.openUrl(QUrl(url_str))
    except Exception as e:
        logger.error(f"QDesktopServices fallback also failed: {e}")
        return False
