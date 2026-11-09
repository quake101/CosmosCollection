#!/usr/bin/env python3
"""A "More options" / "Advanced" toggle: an arrow and text that shows or hides
the settings tucked under it, which most people never need.

    self.more = QWidget()
    self.more_toggle = disclosure_toggle("More options", self.more)
    layout.addWidget(self.more_toggle)
    ...
    layout.addWidget(self.more)

The content starts hidden; setChecked(True) on the toggle shows it.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton


def disclosure_toggle(text, content):
    """A flat checkable button with a right/down arrow that hides/shows content."""
    toggle = QToolButton()
    toggle.setText(text)
    toggle.setCheckable(True)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.RightArrow)
    toggle.setStyleSheet("QToolButton { border: none; }")

    def show(shown):
        content.setVisible(shown)
        toggle.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)

    toggle.toggled.connect(show)
    content.setVisible(False)
    return toggle
