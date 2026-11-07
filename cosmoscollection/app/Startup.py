"""App startup: command-line mode or the GUI (splash, crash log, pre-warm, main window)."""

import faulthandler
import logging
import os
import sys
import threading

from PySide6.QtCore import QCoreApplication, QSettings, QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from cosmoscollection.app.StartupLoadingDialog import StartupLoadingDialog
from cosmoscollection.core.DatabaseManager import DatabaseManager
from cosmoscollection.core.ResourceManager import ResourceManager
from cosmoscollection.ui.Theme import apply_theme

from cosmoscollection.app.CommandLine import parse_cli_arguments, run_cli_command
from cosmoscollection.app.DSOData import InitialDataLoadWorker
from cosmoscollection.app.MainWindow import MainWindow

logger = logging.getLogger(__name__)


# Import astroquery at module level so PyInstaller detects it
try:
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROQUERY_AVAILABLE = True
except ImportError as e:
    ASTROQUERY_AVAILABLE = False
    print(f"Warning: astroquery not available: {e}")

# Check for optional DSO Visibility Calculator availability
try:
    from cosmoscollection.planning import DSOVisibilityCalculator
    VISIBILITY_AVAILABLE = True
except ImportError:
    VISIBILITY_AVAILABLE = False
    logging.warning("DSOVisibilityCalculator.py not found. Visibility calculator will be disabled.")


def main():
    """Start the app: run a command-line command if one was given, otherwise the GUI."""
    # Parse CLI arguments first
    cli_args, remaining_args = parse_cli_arguments()

    # Check if a CLI command was requested
    cli_result = run_cli_command(cli_args)
    if cli_result is not None:
        # A CLI command was run, exit with appropriate code
        sys.exit(0 if cli_result else 1)

    # No CLI command - continue with GUI startup
    # (QTWEBENGINE_CHROMIUM_FLAGS is already set by Bootstrap.py, imported first.)

    # Initialize QtWebEngine BEFORE QApplication: it needs shared OpenGL contexts, which
    # can only be enabled before the application object exists. (This used to import a
    # nonexistent QtWebEngineCore.QtWebEngineCore name, so it always failed silently and
    # QtWebEngine was first loaded later, from a dialog, after QApplication.)
    try:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        from PySide6 import QtWebEngineWidgets  # noqa: F401
        logger.debug("QtWebEngine initialized with shared OpenGL contexts")
    except ImportError:
        logger.warning("QtWebEngineWidgets not available - WebGL may not work")

    app = QApplication(sys.argv)

    # Set application icon
    icon_path = str(ResourceManager.get_icon_path())
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)

    # Apply global dark theme before any UI paints
    apply_theme(app)

    # Show a loading dialog immediately so the user gets feedback instead of
    # a blank screen during the pre-warm/database-load gap below.
    splash = StartupLoadingDialog()
    splash.show()
    QApplication.processEvents()

    # Always-on crash log: faulthandler writes a native traceback to this file
    # even when the process is killed by a C-level crash (segfault, access violation).
    try:
        _crash_log_dir = ResourceManager.get_data_dir()
        os.makedirs(_crash_log_dir, exist_ok=True)
        _crash_log_path = os.path.join(_crash_log_dir, "crash.log")
        _crash_log_file = open(_crash_log_path, "a", encoding="utf-8")
        import datetime
        _crash_log_file.write(f"\n--- session started {datetime.datetime.now().isoformat()} ---\n")
        _crash_log_file.flush()
        faulthandler.enable(_crash_log_file)
        # faulthandler.enable() only registers SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL by default --
        # it misses SIGTRAP, which is how e.g. a hardened-glibc/OpenBLAS/Chromium-GPU-process
        # trap can take the whole process down. Register it too so that shows up here instead
        # of vanishing with nothing but a "session started" line. Unix only: Windows has
        # neither SIGTRAP nor faulthandler.register() (enable() above covers its crashes).
        import signal
        if hasattr(faulthandler, "register") and hasattr(signal, "SIGTRAP"):
            try:
                faulthandler.register(signal.SIGTRAP, file=_crash_log_file, all_threads=True)
            except Exception as e:
                logger.debug(f"Could not register SIGTRAP handler: {e}")
        logger.debug(f"Crash log: {_crash_log_path}")
    except Exception as e:
        logger.warning(f"Could not enable crash log: {e}")

    # With no console, Qt sends its warnings - and the qFatal message it prints just before
    # aborting ("Fatal Python error: Aborted" in crash.log) - to the Windows debugger only.
    # Send them to stderr, which is chromium_debug.log then (see Bootstrap.py).
    if sys.stderr is None:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
        import datetime

        def _qt_message_to_stderr(msg_type, context, message):
            if msg_type == QtMsgType.QtDebugMsg or msg_type == QtMsgType.QtInfoMsg:
                return
            try:
                line = f"{datetime.datetime.now().isoformat()} Qt {msg_type.name}: {message}\n"
                os.write(2, line.encode('utf-8', 'replace'))
            except OSError:
                pass

        qInstallMessageHandler(_qt_message_to_stderr)

    # Configure file logging if enabled (must be after QApplication for QSettings to work on Windows)
    try:
        settings = QSettings("CosmosCollection", "CosmosCollection")
        if settings.value("enable_logfile", False, type=bool):
            from logging.handlers import RotatingFileHandler
            log_dir = ResourceManager.get_data_dir()
            log_file_path = os.path.join(log_dir, "CosmosCollection.log")
            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            # The log file is for troubleshooting, so capture debug detail in it
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info(f"File logging enabled: {log_file_path}")
    except Exception as e:
        logger.warning(f"Could not configure file logging: {e}")
        QMessageBox.warning(None, "Logging Error",
            f"Could not create log file:\n{e}\n\n"
            "Logging to file has been disabled for this session.")

    # Set global WebEngine profile settings
    try:
        from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
        profile = QWebEngineProfile.defaultProfile()
        settings = profile.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        logger.debug("Global WebEngine profile configured with WebGL support")
    except Exception as e:
        logger.warning(f"Could not configure global WebEngine profile: {e}")

    splash.set_status("Preparing astronomy engine...")

    # Pre-warm astropy/erfa/numpy on the main thread, before any background thread (weather
    # fetch, DSO visibility calc, ...) gets a chance to trigger the same first-time lazy
    # imports/C-extension dispatch setup concurrently. A rare native crash (Fatal Python
    # error: Illegal instruction, mid garbage-collection) has been observed at more than one
    # such first-call site -- astropy.units.cds's first import during an IERS table parse,
    # and separately astropy.time.Time._apply/replicate inside get_sun() -- both only ever
    # seen inside the background WeatherForecast thread. Rather than chase each first-call
    # site individually, exercise the actual functions that thread runs (with throwaway data)
    # here, single-threaded, before any worker thread has started, so whatever one-time
    # native setup they trigger happens in a context that reproducibly hasn't crashed.
    try:
        from astropy.utils.iers import IERS_Auto
        IERS_Auto.open()
        from cosmoscollection.planning.WeatherForecast import calculate_sun_altitudes, calculate_moon_phase
        import datetime as _dt
        _now = _dt.datetime.utcnow()
        calculate_sun_altitudes(0.0, 0.0, [_now])
        calculate_moon_phase(_now)

        # Also pre-warm DSOVisibilityCalculator (Best DSO Tonight, DSO Visibility
        # Calculator, Session Manager). It pulls in matplotlib (backend init, font
        # cache) on top of the astropy/erfa cost above, and exercises the same
        # ICRS<->AltAz transform graph via a different call path - importing and
        # exercising it here, in this same single-threaded pre-warm block, means
        # that cost (observed at ~8s cold) is paid once at startup instead of
        # freezing whichever of those windows the user happens to open first.
        from cosmoscollection.planning.DSOVisibilityCalculator import DSOVisibilityCalculator
        from astropy.coordinates import SkyCoord
        import astropy.units as _u
        _calc = DSOVisibilityCalculator(location_lat=0.0, location_lon=0.0, timezone="UTC")
        _calc.calculate_visibility_for_coordinates(
            SkyCoord(ra=0.0 * _u.deg, dec=0.0 * _u.deg), _now.strftime("%Y-%m-%d")
        )

        logger.debug("Pre-warmed astropy/erfa machinery")
    except Exception as e:
        logger.warning(f"Could not pre-warm astropy machinery (will retry lazily later): {e}")

    # pytz checks a zone name against its list of installed zones, which it builds the first
    # time by opening every zone file (~600). That froze the window that first used a real
    # time zone - 2.5 s for Best DSO Tonight with the app on a network drive. Build it now,
    # in the background: it's only file reads under pytz's own lock, so a lookup made before
    # it finishes just waits for the rest.
    def _fill_pytz_zone_list():
        try:
            import pytz
            len(pytz.all_timezones_set)
        except Exception as e:
            logger.debug(f"Could not pre-load the pytz time zone list: {e}")

    threading.Thread(target=_fill_pytz_zone_list, name="pytz-zone-list", daemon=True).start()

    splash.set_status("Loading database...")

    # Initialize database manager and get data
    db_manager = DatabaseManager()

    # Held for the life of the app (main() runs until the event loop ends)
    initial_loader = None
    window = None

    def on_initial_data_loaded(dso_data, catalogs):
        """Handle initial data loaded from background thread"""
        nonlocal window
        try:
            logger.debug(f"Initial data loaded in background: {len(dso_data)} DSOs")

            # Create and show the main window with loaded data
            window = MainWindow(dso_data, catalogs)
            window.show()
            splash.close()

            # Check if location is configured after window is shown
            QTimer.singleShot(500, window._check_location_on_startup)
            # Restarted from a broken view (Restart App): open that tool again
            QTimer.singleShot(800, window._reopen_after_restart)

        except Exception as e:
            logger.error(f"Error creating main window: {e}", exc_info=True)
            splash.close()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
            # Stop the event loop; sys.exit() raised inside a Qt slot doesn't reliably exit
            app.exit(1)

    def on_initial_load_failed(error_msg):
        """Handle initial data load failure"""
        from PySide6.QtWidgets import QMessageBox
        logger.error(f"Failed to load initial data: {error_msg}")
        splash.close()
        QMessageBox.critical(None, "Error", f"Failed to load DSO data from database:\n{error_msg}")
        app.exit(1)

    try:
        # Check if catalogs directory exists, create if needed
        catalogs_dir = str(ResourceManager.get_resource_path('catalogs'))
        if not os.path.exists(catalogs_dir):
            os.makedirs(catalogs_dir)

        # Start background thread to load initial data
        logger.debug("Starting background thread to load initial DSO data")
        initial_loader = InitialDataLoadWorker()
        initial_loader.data_loaded.connect(on_initial_data_loaded)
        initial_loader.load_failed.connect(on_initial_load_failed)
        initial_loader.start()

        # Freezes, slow memory cleanups and memory use go to the log file
        from cosmoscollection.core.Diagnostics import start_diagnostics
        start_diagnostics(app)

        # Notices a graphics reset, for web views that break afterward (Restart App)
        from cosmoscollection.core import GpuReset
        GpuReset.start_watch()

        exit_code = app.exec()
        GpuReset.relaunch_if_requested()
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        splash.close()
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
        sys.exit(1)
    finally:
        db_manager.close()
