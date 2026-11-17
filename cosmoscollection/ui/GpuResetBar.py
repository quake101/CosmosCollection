#!/usr/bin/env python3
"""A bar for a web view after the graphics card was reset (see core/GpuReset.py): says
why, with Try Again / Try Anyway (draw the view - works when Chromium has recovered) and
Restart App (always works).

Shown before the view draws (ahead=True: drawing it then can close the app, so it waits
for Try Anyway), or after it failed to draw (set_ahead(False): Try Again).

    bar = GpuResetBar("the sky view", on_try_again=self._draw, reopen={...}, ahead=True)
    layout.insertWidget(0, bar)
"""

import json
import logging

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from cosmoscollection.ui.Theme import COLORS, themed_style

logger = logging.getLogger(__name__)


def termination_status_name(status):
    """'Crashed' etc. for a web page's renderProcessTerminated status, for the log."""
    from PySide6.QtWebEngineCore import QWebEnginePage
    names = {
        QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus: "Normal",
        QWebEnginePage.RenderProcessTerminationStatus.AbnormalTerminationStatus: "Abnormal",
        QWebEnginePage.RenderProcessTerminationStatus.CrashedTerminationStatus: "Crashed",
        QWebEnginePage.RenderProcessTerminationStatus.KilledTerminationStatus: "Killed",
    }
    return names.get(status, f"Unknown({status})")


# {available, version, vendor, renderer} or {available: false, error} as JSON
_WEBGL_CHECK_JS = """
(function () {
    var canvas = document.createElement('canvas');
    var gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
        return JSON.stringify({available: false, error: 'WebGL not available'});
    }
    var debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    return JSON.stringify({
        available: true,
        version: (typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext)
            ? 'WebGL2' : 'WebGL1',
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown',
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown'
    });
})();
"""


def check_webgl(page, on_result):
    """Ask a loaded web page whether WebGL works - without it the web views'
    graphics are usually broken (e.g. after a graphics reset). on_result gets
    {"available": bool, "version", "vendor", "renderer"} or {"available": False, "error"}.
    WebGL working after a reset means Chromium recovered (GpuReset.recovered())."""
    def parse(result):
        try:
            info = json.loads(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Couldn't read the WebGL check: {e} ({result!r})")
            return
        if info.get("available"):
            from cosmoscollection.core.GpuReset import recovered
            recovered()
        on_result(info)

    try:
        page.runJavaScript(_WEBGL_CHECK_JS, parse)
    except Exception as e:
        logger.error(f"Couldn't check WebGL: {e}")


class GpuResetBar(QFrame):
    def __init__(self, what, on_try_again, reopen=None, before_restart=None, ahead=False, parent=None):
        """what: what can't draw ("the sky view"); on_try_again(): draw it (again);
        reopen: passed to restart_app() (the tool to open again); before_restart(): e.g.
        close a dialog first; ahead: shown before it draws (see set_ahead())."""
        super().__init__(parent)
        self._what = what
        self._on_try_again = on_try_again
        self._reopen = reopen
        self._before_restart = before_restart

        self.setObjectName("gpuResetBar")
        themed_style(self, lambda: f"#gpuResetBar {{ background-color: {COLORS['background_light']};"
                                   f" border: 1px solid {COLORS['warning']}; border-radius: 4px; }}")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        self._label = QLabel()
        self._label.setWordWrap(True)
        row.addWidget(self._label, 1)

        self._try_button = QPushButton()
        self._try_button.clicked.connect(self._try_again)
        row.addWidget(self._try_button)
        restart = QPushButton("Restart App")
        restart.setToolTip("Close Cosmos Collection and start it again")
        restart.clicked.connect(self._restart)
        row.addWidget(restart)
        self.set_ahead(ahead)

    def set_ahead(self, ahead):
        """ahead: the view hasn't drawn yet (drawing it may close the app); otherwise it
        failed to draw."""
        if ahead:
            self._label.setText(
                f"Your graphics card was reset since the app started (for example, HDR turned on "
                f"for a game). Showing {self._what} now could close the app. Restart the app to "
                f"fix it, or try anyway.")
            self._try_button.setText("Try Anyway")
            self._try_button.setToolTip("Show it now. Works if the graphics have recovered; "
                                        "if not, the app may close.")
        else:
            self._label.setText(
                f"Your graphics card was reset (for example, HDR turned on for a game), so "
                f"{self._what} can't draw. Try again, or restart the app to fix it.")
            self._try_button.setText("Try Again")
            self._try_button.setToolTip("Load it again. Works if the graphics have recovered by now.")

    def _try_again(self):
        self.hide()
        self._on_try_again()

    def _restart(self):
        from cosmoscollection.core.GpuReset import restart_app
        if self._before_restart:
            self._before_restart()
        restart_app(self._reopen)
