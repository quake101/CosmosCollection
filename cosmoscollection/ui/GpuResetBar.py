#!/usr/bin/env python3
"""A bar for a web view that broke after the graphics card was reset (see core/GpuReset.py):
says why, with Try Again (reload the view - works when Chromium has recovered) and Restart
App (always works).

    bar = GpuResetBar("the sky view", on_try_again=self._reload, reopen={...})
    layout.insertWidget(0, bar)
"""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from cosmoscollection.ui.Theme import COLORS, themed_style


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
