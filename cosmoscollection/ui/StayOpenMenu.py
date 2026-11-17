"""
StayOpenMenu.py - A QMenu that stays open while its checkable actions are
clicked, so several can be ticked in one go (the main window's filters, Best
DSO Tonight's directions).
"""

from PySide6.QtWidgets import QMenu


class StayOpenMenu(QMenu):
    """A QMenu that stays open when checkable actions are clicked."""
    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action and action.isCheckable():
            action.trigger()
        else:
            super().mouseReleaseEvent(event)
