"""
Window Position Manager
Manages saving and restoring window positions using QSettings
"""
from PySide6.QtCore import QSettings, QPoint
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
        if geometry:
            window.restoreGeometry(geometry)
            return True

        # Fallback: Try to restore position and size separately
        pos = settings.value(f"{window_key}/pos")
        size = settings.value(f"{window_key}/size")

        if pos is not None and size is not None:
            window.move(pos)
            window.resize(size)
            return True

        # First time opening - center the window
        WindowPositionManager.center_window(window)
        return False

    @staticmethod
    def center_window(window):
        """
        Center window on screen

        Args:
            window: QMainWindow or QDialog instance
        """
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = window.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
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
