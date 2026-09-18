#!/usr/bin/env python3
"""
Startup loading dialog shown while CosmosCollection initializes (astropy
pre-warm, database load) so the user sees feedback instead of a blank
screen during the gap before the main window exists.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout

from ResourceManager import ResourceManager
from Theme import COLORS


class StartupLoadingDialog(QDialog):
    """Frameless-style splash shown from app launch until MainWindow is ready."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Starting CosmosCollection")
        self.setFixedSize(420, 160)

        # Remove close button - dialog is dismissed programmatically only
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        logo_label = QLabel()
        pixmap = QPixmap(str(ResourceManager.get_icon_path()))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        self.status_label = QLabel("Starting CosmosCollection...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 11pt;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
            }}
            QProgressBar {{
                background-color: {COLORS['background_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)

        self._center_on_screen()

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        self.move(
            geometry.center().x() - self.width() // 2,
            geometry.center().y() - self.height() // 2,
        )

    def set_status(self, text: str):
        """Update the status text and force an immediate repaint.

        Needed because most of startup runs synchronously on the main
        thread, so without processEvents() the label change wouldn't
        actually paint until that work finished.
        """
        self.status_label.setText(text)
        QApplication.processEvents()
