"""The About dialog."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QToolTip, QVBoxLayout

from cosmoscollection.core.ResourceManager import ResourceManager
from cosmoscollection.ui.Theme import COLORS, font_size, themed_style

from cosmoscollection.app.UpdateDialogs import offer_update_and_apply, run_update_check


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Cosmos Collection")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        # Set minimum size to accommodate content on all platforms
        self.setMinimumSize(450, 420)

        # Create main layout with proper margins and spacing
        About_layout = QVBoxLayout(self)
        About_layout.setContentsMargins(30, 30, 30, 30)
        About_layout.setSpacing(20)

        # Create icon label
        icon_label = QLabel()
        icon_path = ResourceManager.get_icon_path()
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            # Scale the icon to a reasonable size (64x64)
            scaled_pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        About_layout.addWidget(icon_label)

        # Title
        title_label = QLabel("Cosmos Collection")
        title_label.setAlignment(Qt.AlignCenter)
        themed_style(title_label, lambda: f"font-size: {font_size(18)}; font-weight: bold; color: {COLORS['text']};")
        About_layout.addWidget(title_label)

        # Version information - the local version now, the latest release once GitHub answers
        try:
            from cosmoscollection.version import version_manager
            version_text = f"Local Version: {version_manager.get_local_version()}\nChecking for the latest release…"
            run_update_check(self, self._show_version_details)
        except ImportError:
            version_text = "Version information not available"

        self.version_label = version_label = QLabel(version_text)
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setWordWrap(True)  # Allow text wrapping if needed
        themed_style(version_label, lambda: f"font-size: {font_size(10)}; color: {COLORS['text_secondary']}; margin: 5px 0px; font-family: monospace;")
        About_layout.addWidget(version_label)

        # Description
        desc_label = QLabel("A personal astrophotography catalog and session planning tools for organizing and exploring your celestial images.")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        themed_style(desc_label, lambda: f"font-size: {font_size(11)}; color: {COLORS['text_secondary']}; margin: 10px 0px;")
        About_layout.addWidget(desc_label)

        # GitHub links: the repository, and sponsoring the project
        link_label = QLabel(
            f'<a href="https://github.com/quake101/CosmosCollection" style="color: {COLORS["link"]};">Visit GitHub Repository</a>'
            f' &nbsp;·&nbsp; '
            f'<a href="https://github.com/sponsors/quake101" style="color: {COLORS["link"]};">Sponsor on GitHub &#9829;</a>'
            f' &nbsp;·&nbsp; '
            f'<a href="https://buymeacoffee.com/quake101" style="color: {COLORS["link"]};">Buy me a Beer? &#127866;</a>')
        # QLabel has no per-anchor tooltips, so show one for whichever link is hovered
        link_tooltips = {
            "https://github.com/quake101/CosmosCollection": "View the source code, report issues and request features",
            "https://github.com/sponsors/quake101": "Sponsoring helps keep Cosmos Collection going",
            "https://buymeacoffee.com/quake101": "Send a one-time thank-you to support development",
        }
        link_label.linkHovered.connect(
            lambda url: QToolTip.showText(QCursor.pos(), link_tooltips.get(url, ""), link_label)
            if url else QToolTip.hideText())
        link_label.setAlignment(Qt.AlignCenter)
        link_label.setOpenExternalLinks(True)
        link_label.setTextFormat(Qt.RichText)
        themed_style(link_label, lambda: f"font-size: {font_size(10)};")
        About_layout.addWidget(link_label)

        # Add fixed stretch to prevent layout jumping
        About_layout.addStretch(1)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Check for updates button
        try:
            from cosmoscollection.version import check_for_updates
            self.update_button = QPushButton("Check for Updates")
            self.update_button.setFixedSize(130, 30)
            self.update_button.clicked.connect(self._check_updates)
            button_layout.addWidget(self.update_button)
        except ImportError:
            pass

        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedSize(80, 30)
        close_button.clicked.connect(self.close)
        close_button.setDefault(True)
        button_layout.addWidget(close_button)

        button_layout.addStretch()
        About_layout.addLayout(button_layout)

    def _show_version_details(self, version_info):
        from cosmoscollection.version import version_manager
        self.version_label.setText(version_manager.get_detailed_version_info(version_info))

    def _check_updates(self):
        """Ask GitHub again (in the background) and show the result"""
        from cosmoscollection.core.OnlineLimits import is_offline, notify_offline
        if is_offline():
            notify_offline(self, "Checking for updates")
            return
        self.update_button.setEnabled(False)
        self.update_button.setText("Checking…")
        run_update_check(self, self._on_update_check, refresh=True)

    def _on_update_check(self, version_info):
        """Show the result of Check for Updates"""
        self.update_button.setEnabled(True)
        self.update_button.setText("Check for Updates")
        self._show_version_details(version_info)
        try:
            from PySide6.QtWidgets import QMessageBox

            if not version_info['github_available']:
                QMessageBox.information(self, "Update Check",
                    "Unable to check for updates. Please check your internet connection.")
                return

            if version_info['update_available']:
                offer_update_and_apply(self, version_info)
            else:
                QMessageBox.information(self, "No Updates",
                    f"You are running the latest version ({version_info['local_version']}).")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Error checking for updates: {str(e)}")
