"""The app's process setup, which must happen before numpy, Qt or any network
module loads.

Imported first by main.py when it starts the app (not the command line);
importing it does the setup. The part the command line shares is in
core/ProcessSetup.py."""

import logging
import os
import sys

# Started again by Restart App: wait for the old app to finish exiting first, so the two
# don't share the database or the web views' storage (see core/GpuReset.py)
from cosmoscollection.core.GpuReset import RESTART_WAIT_ENV  # noqa: E402 - standard library only
if os.environ.get(RESTART_WAIT_ENV):
    from cosmoscollection.core.GpuReset import wait_for_previous_app
    wait_for_previous_app()

# OpenBLAS thread cap, SSL certificates and astroquery's cache folder
import cosmoscollection.core.ProcessSetup  # noqa: E402,F401

# QtWebEngine (Chromium) flags. Set here, before Qt starts, and nowhere else: a later
# update once overwrote these and brought the crashes back.
#   --no-sandbox: Chromium's sandbox failed to start here (renderer exit code 18), so the
#     FOV Simulator's sky and the map picker never drew. Applies to every web view.
#   --enable-logging --log-level=1: Chromium writes its fatal-error message to stderr just
#     before it aborts - the only clue when the app crashes inside app.exec() while a sky
#     view is drawing. Level 1 keeps warnings and errors, not the info lines (every web page
#     console message). QtWebEngine ignores --log-file, so stderr is pointed at
#     chromium_debug.log below.
#   Linux only - --disable-gpu-compositing --disable-gpu-rasterization --disable-zero-copy:
#     some Mesa builds (e.g. CachyOS, especially under Wayland) draw web views as garbled or
#     solid black. These keep drawing on the CPU (WebGL is unaffected). Elsewhere they'd only
#     cost frame rate.
from cosmoscollection.core.ResourceManager import ResourceManager as _ResourceManager
_chromium_log_path = os.path.join(str(_ResourceManager.get_data_dir()), 'chromium_debug.log')
_gpu_workarounds = ('--disable-gpu-compositing --disable-gpu-rasterization --disable-zero-copy '
                    if sys.platform.startswith('linux') else '')
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = (
    f'--no-sandbox {_gpu_workarounds}'
    f'--enable-logging --log-level=1'
)

# Windows: no Qt vsync helper thread (QDxgiVSyncService). When the graphics device is reset
# while the app runs (Auto HDR switching on for a game, an HDR toggle, a driver reset), it was
# being torn down when the app aborted on opening the FOV Simulator (2026-11-06,
# "QDxgiVSyncService not destroyed in time" then "Fatal Python error: Aborted"). Without it Qt
# paces window updates with a timer instead. A value already in the environment wins.
if sys.platform == 'win32':
    os.environ.setdefault('QT_D3D_NO_VBLANK_THREAD', '1')

# The packaged app (and pythonw) has no console, so Chromium's stderr went nowhere. Point the
# process's stderr at chromium_debug.log; Chromium's GPU and renderer processes inherit it.
# Appended, not overwritten, so the messages from a crash survive the next start. From a
# terminal they already show there.
if sys.stderr is None:
    try:
        os.makedirs(os.path.dirname(_chromium_log_path), exist_ok=True)
        if os.path.exists(_chromium_log_path) and os.path.getsize(_chromium_log_path) > 2 * 1024 * 1024:
            os.replace(_chromium_log_path, _chromium_log_path + '.old')
        _chromium_log_file = open(_chromium_log_path, 'a', encoding='utf-8', buffering=1)
        import datetime as _datetime
        _chromium_log_file.write(f"\n--- session started {_datetime.datetime.now().isoformat()} ---\n")
        os.dup2(_chromium_log_file.fileno(), 2)
    except OSError:
        pass

# Set up logging: DEBUG when running from source; packaged builds log INFO unless
# the "Enable log file" setting raises it to DEBUG (see Startup.py)
logging.basicConfig(level=logging.INFO if getattr(sys, 'frozen', False) else logging.DEBUG)
logger = logging.getLogger(__name__)


# Log the SSL and astroquery cache setup (ProcessSetup.py)
if getattr(sys, 'frozen', False):
    logger.info(f"Running in PyInstaller bundle. SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE')}")
    logger.info(f"Astroquery cache directory set to: {os.environ.get('XDG_CACHE_HOME')}")
else:
    logger.debug("Running in normal Python environment")
