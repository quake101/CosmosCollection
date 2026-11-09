#!/usr/bin/env python3
"""A bar for a web view that broke after the graphics card was reset (see core/GpuReset.py):
says why, with Try Again (reload the view - works when Chromium has recovered) and Restart
App (always works).

    bar = GpuResetBar("the sky view", on_try_again=self._reload, reopen={...})
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
    {"available": bool, "version", "vendor", "renderer"} or {"available": False, "error"}."""
    def parse(result):
        try:
            info = json.loads(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Couldn't read the WebGL check: {e} ({result!r})")
            return
        on_result(info)

    try:
        page.runJavaScript(_WEBGL_CHECK_JS, parse)
    except Exception as e:
        logger.error(f"Couldn't check WebGL: {e}")


class GpuResetBar(QFrame):
    def __init__(self, what, on_try_again, reopen=None, before_restart=None, parent=None):
        """what: what can't draw ("the sky view"); on_try_again(): reload it;
        reopen: passed to restart_app() (the tool to open again); before_restart(): e.g.
        close a dialog first."""
        super().__init__(parent)
        self._on_try_again = on_try_again
        self._reopen = reopen
        self._before_restart = before_restart

        self.setObjectName("gpuResetBar")
        themed_style(self, lambda: f"#gpuResetBar {{ background-color: {COLORS['background_light']};"
                                   f" border: 1px solid {COLORS['warning']}; border-radius: 4px; }}")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        label = QLabel(
            f"Your graphics card was reset (for example, HDR turned on for a game), so {what} "
            f"can't draw. Try again, or restart the app to fix it.")
        label.setWordWrap(True)
        row.addWidget(label, 1)

        try_again = QPushButton("Try Again")
        try_again.setToolTip("Load it again. Works if the graphics have recovered by now.")
        try_again.clicked.connect(self._try_again)
        row.addWidget(try_again)
        restart = QPushButton("Restart App")
        restart.setToolTip("Close Cosmos Collection and start it again")
        restart.clicked.connect(self._restart)
        row.addWidget(restart)

    def _try_again(self):
        self.hide()
        self._on_try_again()

    def _restart(self):
        from cosmoscollection.core.GpuReset import restart_app
        if self._before_restart:
            self._before_restart()
        restart_app(self._reopen)
