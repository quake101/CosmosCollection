"""
Window Position Manager
Manages saving and restoring window positions using QSettings
"""
from PySide6.QtCore import QSettings, QPoint, QRect
from PySide6.QtWidgets import QApplication


class WindowPositionManager:
    """Manages window position persistence using QSettings"""

    @staticmethod
    def get_settings():
        """Get QSettings instance for the application"""
        return QSettings("CosmosCollection", "CosmosCollection")

    @staticmethod
    def save_window_position(window, window_key):
        """
        Save window position and size to settings

        Args:
            window: QMainWindow or QDialog instance
            window_key: Unique key to identify this window type (e.g., 'DSOTargetList')
        """
        settings = WindowPositionManager.get_settings()
        settings.setValue(f"{window_key}/geometry", window.saveGeometry())
        settings.setValue(f"{window_key}/pos", window.pos())
        settings.setValue(f"{window_key}/size", window.size())

    @staticmethod
    def restore_window_position(window, window_key):
        """
        Restore window position and size from settings, or center if first time

        Args:
            window: QMainWindow or QDialog instance
            window_key: Unique key to identify this window type (e.g., 'DSOTargetList')

        Returns:
            bool: True if position was restored, False if centered (first time)
        """
        settings = WindowPositionManager.get_settings()

        # Try to restore saved geometry first
        geometry = settings.value(f"{window_key}/geometry")
        restored = bool(geometry) and window.restoreGeometry(geometry)

        # Fallback: Try to restore position and size separately
        if not restored:
            pos = settings.value(f"{window_key}/pos")
            size = settings.value(f"{window_key}/size")
            if pos is not None and size is not None:
                window.move(pos)
                window.resize(size)
                restored = True

        if not restored:
            # First time opening - center the window
            WindowPositionManager.center_window(window)
            return False

        # Saved on a monitor that's since been unplugged or turned off: a modal
        # dialog there would block the app with nothing to see or close
        if not WindowPositionManager._on_a_screen(window):
            WindowPositionManager.center_window(window)
        return True

    @staticmethod
    def _on_a_screen(window, min_width=100, min_height=50):
        """Whether enough of the window shows on a connected screen to see and drag it."""
        geometry = window.geometry()
        for screen in QApplication.screens():
            visible = screen.availableGeometry().intersected(geometry)
            if visible.width() >= min(min_width, geometry.width()) and visible.height() >= min(min_height, geometry.height()):
                return True
        return False

    @staticmethod
    def center_window(window):
        """
        Center window on its parent's screen (the primary screen without one),
        shrunk to fit if it's bigger than that screen

        Args:
            window: QMainWindow or QDialog instance
        """
        parent = window.parentWidget()
        screen = (parent.window().screen() if parent is not None else None) or QApplication.primaryScreen()
        available = screen.availableGeometry()
        frame = window.frameGeometry()
        # Room for the title bar and borders, which a not-yet-shown window doesn't have yet
        border_w, border_h = max(frame.width() - window.width(), 16), max(frame.height() - window.height(), 40)
        window.resize(min(window.width(), available.width() - border_w),
                      min(window.height(), available.height() - border_h))
        window_geometry = QRect(0, 0, window.width() + border_w, window.height() + border_h)
        window_geometry.moveCenter(available.center())
        window.move(window_geometry.topLeft())


class WindowPositionMixin:
    """
    Mixin class to add position persistence to any QMainWindow or QDialog

    Usage:
        class MyWindow(WindowPositionMixin, QMainWindow):
            WINDOW_POSITION_KEY = "MyWindow"

            def __init__(self):
                super().__init__()
                self.setup_window_position()
    """

    # Subclasses should define this
    WINDOW_POSITION_KEY = None

    def setup_window_position(self):
        """Initialize window position management"""
        if self.WINDOW_POSITION_KEY is None:
            raise ValueError("WINDOW_POSITION_KEY must be defined in subclass")

        # Restore saved position or center if first time
        WindowPositionManager.restore_window_position(self, self.WINDOW_POSITION_KEY)

    def closeEvent(self, event):
        """Save window position when closing"""
        if self.WINDOW_POSITION_KEY:
            WindowPositionManager.save_window_position(self, self.WINDOW_POSITION_KEY)

        # Call parent closeEvent if it exists
        if hasattr(super(), 'closeEvent'):
            super().closeEvent(event)
        else:
            event.accept()

    def moveEvent(self, event):
        """Save window position when moved"""
        if self.WINDOW_POSITION_KEY and self.isVisible():
            WindowPositionManager.save_window_position(self, self.WINDOW_POSITION_KEY)

        # Call parent moveEvent if it exists
        if hasattr(super(), 'moveEvent'):
            super().moveEvent(event)

    def resizeEvent(self, event):
        """Save window size when resized"""
        if self.WINDOW_POSITION_KEY and self.isVisible():
            WindowPositionManager.save_window_position(self, self.WINDOW_POSITION_KEY)

        # Call parent resizeEvent if it exists
        if hasattr(super(), 'resizeEvent'):
            super().resizeEvent(event)
