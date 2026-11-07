#!/usr/bin/env python3
"""
Notices when the graphics card was reset while the app runs, and restarts the app.

A graphics reset (Windows switching Auto HDR on for a game, HDR switched on or off with
Win+Alt+B, a driver update or recovery) can leave the app's web views - the FOV
Simulator's sky, the map picker - unable to draw until the app restarts. Sometimes
Chromium recovers by itself, so a reset alone isn't reported: a web view that then fails
asks was_reset() and, if so, offers Try Again and Restart App (GpuResetBar).

Windows only: a DirectX (DXGI) factory made at startup stops being "current" once the
graphics card is reset or changed. Elsewhere was_reset() is always False.

    start_watch()                      # once, at startup
    if was_reset(): ...                # a web view failed - was it the reset?
    restart_app({"tool": "fov", "data": data})   # quit, then start again
    relaunch_if_requested()            # after app.exec() returns
    take_reopen()                      # in the new app: the tool to open again
"""

import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid

logger = logging.getLogger(__name__)

# The new app waits for the old one to finish exiting before it starts (see Bootstrap.py)
RESTART_WAIT_ENV = "COSMOS_RESTART_WAIT_PID"
_REOPEN_KEY = "restart/reopen"  # QSettings: the tool to open again after the restart (JSON)

_IID_IDXGIFactory1 = "770aae78-f26f-4dba-a829-253c83d1b387"
_IS_CURRENT = 13  # IDXGIFactory1::IsCurrent: IUnknown (3) + IDXGIObject (4) + IDXGIFactory (5) + EnumAdapters1

_factory = None          # IDXGIFactory1 pointer (c_void_p), kept for the life of the app
_is_current = None       # Its IsCurrent(), callable
_reset = False           # Seen a reset (stays True: the web views stay broken until a restart)
_restart_requested = False


def start_watch():
    """Make the DirectX factory to watch. Call once at startup, on the main thread."""
    global _factory, _is_current
    if sys.platform != "win32" or _factory is not None:
        return
    try:
        dxgi = ctypes.WinDLL("dxgi")
        iid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(_IID_IDXGIFactory1).bytes_le)
        factory = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory))
        if hr != 0 or not factory:
            logger.debug(f"Graphics reset check unavailable (CreateDXGIFactory1: {hr & 0xFFFFFFFF:#010x})")
            return
        vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        is_current = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(vtable[_IS_CURRENT])
        _factory, _is_current = factory, is_current
    except Exception as e:
        logger.debug(f"Graphics reset check unavailable: {e}")


def was_reset():
    """True if the graphics card was reset or changed since the app started."""
    global _reset
    if _reset or _is_current is None:
        return _reset
    try:
        if not _is_current(_factory):
            _reset = True
            logger.warning("The graphics card was reset or changed since the app started")
    except Exception as e:
        logger.debug(f"Graphics reset check failed: {e}")
    return _reset


# --------------------------------------------------------------------
# Restart
# --------------------------------------------------------------------
def restart_app(reopen=None):
    """Quit the app, then start it again once this one has exited.

    reopen: what to open again in the new app, e.g. {"tool": "fov", "data": {...}}
    (JSON-friendly; dropped if it isn't)."""
    global _restart_requested
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    settings = QSettings("CosmosCollection", "CosmosCollection")
    text = None
    if reopen:
        try:
            text = json.dumps(reopen)
        except (TypeError, ValueError) as e:
            logger.debug(f"Restart: can't reopen {reopen.get('tool')}: {e}")
    if text:
        settings.setValue(_REOPEN_KEY, text)
    else:
        settings.remove(_REOPEN_KEY)
    settings.sync()

    _restart_requested = True
    logger.info("Restarting the app")
    app = QApplication.instance()
    # The main window's quit stops its timers, the tray icon and the database first
    for widget in app.topLevelWidgets():
        quit_app = getattr(widget, "_quit_application", None)
        if callable(quit_app):
            quit_app()
            return
    app.quit()


def relaunch_if_requested():
    """Start the app again if restart_app() asked. Call after app.exec() returns."""
    if not _restart_requested:
        return
    if getattr(sys, "frozen", False):
        cmd = [sys.executable] + sys.argv[1:]
    else:
        cmd = [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]
    env = dict(os.environ)
    env[RESTART_WAIT_ENV] = str(os.getpid())
    kwargs = {"cwd": os.getcwd(), "env": env, "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
        # Out of this app's systemd scope, which is torn down when it exits (see AppUpdater.py)
        systemd_run = shutil.which("systemd-run")
        if systemd_run:
            cmd = [systemd_run, "--user", "--scope", "--quiet", "--collect", "--"] + cmd
    try:
        subprocess.Popen(cmd, **kwargs)
        logger.info("Started the app again")
    except OSError as e:
        logger.error(f"Could not start the app again: {e}")


def take_reopen():
    """The tool restart_app() asked to open again ({"tool": ..., ...}), or None. Only once."""
    from PySide6.QtCore import QSettings
    settings = QSettings("CosmosCollection", "CosmosCollection")
    text = settings.value(_REOPEN_KEY)
    if text is None:
        return None
    settings.remove(_REOPEN_KEY)
    try:
        note = json.loads(text)
    except (TypeError, ValueError):
        return None
    return note if isinstance(note, dict) else None


def wait_for_previous_app():
    """In an app started by relaunch_if_requested(): wait (up to 20 s) for the old one to
    finish exiting, so the two don't share the web views' storage or the database. Called
    from Bootstrap.py before anything else starts."""
    pid_text = os.environ.pop(RESTART_WAIT_ENV, None)
    if not pid_text:
        return
    try:
        pid = int(pid_text)
    except ValueError:
        return
    timeout_s = 20
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:  # None: it's gone already
            kernel32.WaitForSingleObject(ctypes.c_void_p(handle), timeout_s * 1000)
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    else:
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.1)
