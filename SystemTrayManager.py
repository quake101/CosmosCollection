#!/usr/bin/env python3
"""
System Tray Manager for Cosmos Collection
Provides system tray icon with mini weather forecast and quick actions
"""

import logging
import sys
from typing import List, Optional, Callable

from PySide6.QtCore import QObject, Signal, QSettings
from PySide6.QtGui import QIcon, QAction, QCursor
from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QApplication, QWidget

logger = logging.getLogger(__name__)


class SystemTrayManager(QObject):
    """Manages the system tray icon, menu, and tooltip for Cosmos Collection"""

    # Signals for communicating with the main window
    restore_requested = Signal()
    quit_requested = Signal()
    action_triggered = Signal(str)  # Emits action name: "best_dso", "target_list", "weather", "gallery"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._menu: Optional[QMenu] = None
        self._is_available = False
        self._first_minimize = True

    @property
    def is_available(self) -> bool:
        """Check if system tray is available on this platform"""
        return self._is_available

    def setup(self, icon: QIcon) -> bool:
        """
        Initialize the system tray icon and menu.

        Args:
            icon: The QIcon to use for the tray icon

        Returns:
            True if setup was successful, False otherwise
        """
        # Check if system tray is available
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray is not available on this platform")
            self._is_available = False
            return False

        try:
            # Create tray icon
            self._tray_icon = QSystemTrayIcon(icon, self)
            self._tray_icon.setToolTip("Cosmos Collection")

            # Create context menu — must have a QWidget parent so Windows has a
            # valid HWND when rendering hover highlights while the main window is hidden.
            parent_widget = self.parent() if isinstance(self.parent(), QWidget) else None
            self._menu = QMenu(parent_widget)
            self._create_menu()

            # On Windows, deliberately NOT calling setContextMenu(): when the
            # main window is hidden there is no foreground window, and
            # Windows' menu-tracking then misreads the still-logically-down
            # right mouse button as a click on whatever item the cursor first
            # touches while moving into the menu — items fire on hover instead
            # of on click. We pop the menu ourselves in _on_tray_activated()
            # so we can call SetForegroundWindow first, which is Microsoft's
            # documented fix for tray-icon context menus (see
            # Shell_NotifyIcon/TrackPopupMenu remarks). This does not
            # show/restore any window — it only hands our process the OS
            # input-focus context a normal foreground app would already have,
            # exactly like Explorer does for its own tray icons on every
            # right-click.
            #
            # On Linux/macOS we do use setContextMenu(), letting the platform
            # (e.g. the desktop environment's StatusNotifierItem handling)
            # position and show the menu natively above the tray icon.
            # Manually popping up at QCursor.pos() instead is unreliable
            # there — under Wayland compositors in particular (common on
            # distros like CachyOS running KDE Plasma or Hyprland), the
            # global cursor position isn't reliably queryable, which is why
            # the menu could show up centered on screen instead of above the
            # icon.
            if sys.platform != "win32":
                self._tray_icon.setContextMenu(self._menu)

            # Connect signals
            self._tray_icon.activated.connect(self._on_tray_activated)

            self._is_available = True
            logger.debug("System tray initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup system tray: {e}", exc_info=True)
            self._is_available = False
            return False

    def _create_menu(self):
        """Create the context menu for the tray icon"""
        if not self._menu:
            return

        self._menu.clear()

        # Show/Restore action
        show_action = QAction("Show Cosmos Collection", self._menu)
        show_action.triggered.connect(self._on_show_clicked)
        self._menu.addAction(show_action)

        self._menu.addSeparator()

        # Quick access actions
        best_dso_action = QAction("Best DSO Tonight", self._menu)
        best_dso_action.triggered.connect(lambda: self._on_action_clicked("best_dso"))
        self._menu.addAction(best_dso_action)

        target_list_action = QAction("Target List", self._menu)
        target_list_action.triggered.connect(lambda: self._on_action_clicked("target_list"))
        self._menu.addAction(target_list_action)

        weather_action = QAction("Weather Forecast", self._menu)
        weather_action.triggered.connect(lambda: self._on_action_clicked("weather"))
        self._menu.addAction(weather_action)

        gallery_action = QAction("Image Gallery", self._menu)
        gallery_action.triggered.connect(lambda: self._on_action_clicked("gallery"))
        self._menu.addAction(gallery_action)

        nina_dashboard_action = QAction("NINA Dashboard", self._menu)
        nina_dashboard_action.triggered.connect(lambda: self._on_action_clicked("nina_dashboard"))
        self._menu.addAction(nina_dashboard_action)

        self._menu.addSeparator()

        # Quit action
        quit_action = QAction("Quit", self._menu)
        quit_action.triggered.connect(self._on_quit_clicked)
        self._menu.addAction(quit_action)

    def show(self):
        """Show the tray icon"""
        if self._tray_icon and self._is_available:
            self._tray_icon.show()
            logger.debug("System tray icon shown")

            # Show balloon message on first minimize
            if self._first_minimize:
                self._first_minimize = False
                settings = QSettings("CosmosCollection", "CosmosCollection")
                if not settings.value("tray_notification_shown", False, type=bool):
                    self._tray_icon.showMessage(
                        "Cosmos Collection",
                        "The application is still running in the system tray.\n"
                        "Double-click the icon to restore, or right-click for quick actions.",
                        QSystemTrayIcon.Information,
                        3000
                    )
                    settings.setValue("tray_notification_shown", True)

    def hide(self):
        """Hide the tray icon"""
        if self._tray_icon:
            self._tray_icon.hide()
            logger.debug("System tray icon hidden")

    def update_tooltip(self, weather_data: List) -> None:
        """
        Update the tray icon tooltip with weather forecast summary.

        Args:
            weather_data: List of DailyWeatherSummary objects (first 3 days will be used)
        """
        if not self._tray_icon:
            return

        try:
            # Windows tooltip limit is ~127 chars, so keep it compact
            # Format: "Cosmos Collection\nTue: ★78 ☁15% 🌔85%\nWed: ★45 ☁65%..."
            tooltip_lines = ["Cosmos Collection"]

            if weather_data and len(weather_data) > 0:
                # Show up to 3 days in compact format
                for summary in weather_data[:3]:
                    # Short day name
                    day_str = summary.date.strftime("%a")

                    # Astro score with star
                    score = summary.astro_score

                    # Cloud cover
                    cloud_pct = int(summary.tonight_avg_cloud_cover)

                    # Moon info (compact)
                    moon_str = ""
                    if summary.moon_phase:
                        moon_emoji = summary.moon_phase.phase_emoji
                        moon_pct = int(summary.moon_phase.illumination)
                        moon_str = f" {moon_emoji}{moon_pct}%"

                    # Compact format: "Tue: ★78 ☁15% 🌔85%"
                    line = f"{day_str}: ★{score} ☁{cloud_pct}%{moon_str}"
                    tooltip_lines.append(line)
            else:
                tooltip_lines.append("No weather data")
                tooltip_lines.append("Right-click → Weather")

            tooltip_text = "\n".join(tooltip_lines)
            self._tray_icon.setToolTip(tooltip_text)
            logger.debug("Tray tooltip updated with weather data")

        except Exception as e:
            logger.error(f"Failed to update tray tooltip: {e}", exc_info=True)
            self._tray_icon.setToolTip("Cosmos Collection")

    def cleanup(self):
        """Clean up resources before application exit"""
        if self._tray_icon:
            self._tray_icon.hide()
            self._tray_icon = None
        if self._menu:
            self._menu.deleteLater()
            self._menu = None
        logger.debug("System tray cleaned up")

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """Handle tray icon activation (click, double-click, etc.)"""
        logger.debug(f"Tray icon activated with reason: {reason}")
        if reason == QSystemTrayIcon.DoubleClick:
            self.restore_requested.emit()
        elif reason == QSystemTrayIcon.Context and sys.platform == "win32":
            # On other platforms the menu is already attached via
            # setContextMenu() in setup(), and the platform/desktop
            # environment shows it itself, positioned above the tray icon.
            self._popup_menu()

    def _popup_menu(self):
        """
        Show the right-click menu, taking the OS foreground-window handoff
        first so Windows' menu tracking works correctly when no app window
        is currently visible/active. This does not show or restore any
        window — see the comment in setup() for why it's needed.

        Windows-only: other platforms rely on setContextMenu() instead (see
        setup()), since QCursor.pos() isn't a reliable menu-anchor position
        everywhere else (notably under Wayland).
        """
        if not self._menu:
            return

        try:
            import ctypes
            hwnd = int(self._menu.winId())
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception as e:
            logger.debug(f"SetForegroundWindow before tray menu popup failed: {e}")

        self._menu.popup(QCursor.pos())

    def _on_show_clicked(self):
        """Handle Show action clicked"""
        self.restore_requested.emit()

    def _on_action_clicked(self, action_name: str):
        """Handle quick action clicked"""
        # Just trigger the action - tools open independently without restoring main window
        self.action_triggered.emit(action_name)

    def _on_quit_clicked(self):
        """Handle Quit action clicked"""
        logger.debug(f"Tray icon quit triggered.")
        self.quit_requested.emit()
