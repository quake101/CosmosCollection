"""
Progress dialog for the standalone updater helper (see CosmosUpdater.py).

Reuses PySide6 rather than a platform-specific GUI toolkit - it's already a
hard dependency of the main app, so this doesn't add a second GUI stack to
the project, and the same dialog code works on every platform Cosmos
Collection ships for. The QThread + Signal split below mirrors
AppUpdater.py's DownloadWorker: the actual update steps run off the UI
thread so the dialog stays responsive, and status text crosses threads via
a Qt signal instead of touching widgets directly from the worker.
"""

import os
import sys

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QProgressDialog


def _resource_path(*parts):
    """Path to a bundled resource, working both frozen (onefile build -
    files are extracted under sys._MEIPASS) and running from source (this
    file lives in updater/, resources are one directory up)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    return os.path.join(base, *parts)


class UpdateWorker(QThread):
    """Runs run_update(status_cb) off the UI thread. run_update is expected
    to catch its own errors and always return an exit code (matching
    CosmosUpdater's original main()) rather than raise - the broad except
    here is just a last-resort safety net so a truly unexpected bug can't
    hang the dialog forever."""

    status = Signal(str)
    done = Signal(int)

    def __init__(self, run_update, parent=None):
        super().__init__(parent)
        self._run_update = run_update

    def run(self):
        try:
            code = self._run_update(self.status.emit)
        except Exception as e:
            self.status.emit(f"Update failed: {e}")
            code = 1
        self.done.emit(code)


def run_with_progress(run_update) -> int:
    """Shows a small progress dialog while run_update(status_cb) executes
    on a background thread; status_cb(text) updates the dialog's label.
    Returns run_update's exit code.

    Falls back to running run_update synchronously with no UI if Qt can't
    initialize (e.g. no display available) - the update itself must never
    be blocked by a GUI problem."""
    try:
        app = QApplication.instance() or QApplication(sys.argv)
    except Exception:
        app = None

    if app is None:
        try:
            return run_update(lambda _text: None)
        except Exception:
            return 1

    icon = QIcon(_resource_path("images", "CosmosCollectionUpdater.png"))
    app.setWindowIcon(icon)

    progress = QProgressDialog("Waiting for Cosmos Collection to close...", None, 0, 0)
    progress.setWindowTitle("Updating Cosmos Collection")
    progress.setWindowIcon(icon)
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setCancelButton(None)  # can't safely cancel mid file-swap
    progress.show()

    result = {"code": 1}

    worker = UpdateWorker(run_update)
    worker.status.connect(progress.setLabelText)

    def _on_done(code):
        result["code"] = code
        if code == 0:
            progress.close()
            app.quit()
        else:
            # Leave the failure message on screen briefly rather than
            # vanishing the dialog the instant something goes wrong.
            QTimer.singleShot(4000, lambda: (progress.close(), app.quit()))

    worker.done.connect(_on_done)
    worker.start()

    app.exec()
    worker.wait()
    return result["code"]
