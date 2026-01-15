#!/usr/bin/env python3
"""
Global Theme for Cosmos Collection
Provides consistent dark theme styling across the entire application
"""

# Color palette
COLORS = {
    'background': '#2b2b2b',
    'background_light': '#353535',
    'background_lighter': '#404040',
    'background_hover': '#4a4a4a',
    'border': '#555555',
    'border_light': '#666666',
    'text': '#ffffff',
    'text_secondary': '#cccccc',
    'text_disabled': '#888888',
    'accent': '#0078d4',
    'accent_hover': '#106ebe',
    'accent_pressed': '#005a9e',
    'error': '#ff4444',
    'error_bg': '#4a2020',
    'warning': '#ffcc00',
    'success': '#44ff44',
    'info': '#88ccff',
    'favorite': '#FFD700',
}

# Global stylesheet applied to entire application
DARK_THEME = f"""
    /* === Global Defaults === */
    QWidget {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
        font-family: "Segoe UI", Arial, sans-serif;
    }}

    /* === Main Window === */
    QMainWindow {{
        background-color: {COLORS['background']};
    }}

    /* === Group Boxes === */
    QGroupBox {{
        background-color: {COLORS['background_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        margin-top: 10px;
        padding: 10px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {COLORS['accent']};
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 3px;
        min-height: 28px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['accent_hover']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['accent_pressed']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['border']};
        color: {COLORS['text_disabled']};
    }}

    /* === Text Inputs === */
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border_light']};
        padding: 5px;
        border-radius: 3px;
        selection-background-color: {COLORS['accent']};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {COLORS['accent']};
    }}
    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
        background-color: {COLORS['background_light']};
        color: {COLORS['text_disabled']};
    }}

    /* === Combo Boxes === */
    QComboBox {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border_light']};
        padding: 5px;
        border-radius: 3px;
        min-height: 20px;
    }}
    QComboBox:focus {{
        border: 1px solid {COLORS['accent']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid {COLORS['text']};
        margin-right: 5px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        selection-background-color: {COLORS['accent']};
        border: 1px solid {COLORS['border_light']};
        padding: 2px;
    }}
    QComboBox QAbstractItemView::item {{
        padding: 5px;
        min-height: 20px;
    }}

    /* === Scroll Areas === */
    QScrollArea {{
        border: none;
        background-color: {COLORS['background']};
    }}

    /* === Scroll Bars === */
    QScrollBar:vertical {{
        background-color: {COLORS['background_light']};
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background-color: {COLORS['border_light']};
        border-radius: 6px;
        min-height: 30px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['text_disabled']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background-color: {COLORS['background_light']};
        height: 12px;
        border-radius: 6px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {COLORS['border_light']};
        border-radius: 6px;
        min-width: 30px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {COLORS['text_disabled']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* === Labels === */
    QLabel {{
        color: {COLORS['text']};
        background-color: transparent;
    }}

    /* === Check Boxes === */
    QCheckBox {{
        color: {COLORS['text']};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {COLORS['border_light']};
        border-radius: 3px;
        background-color: {COLORS['background_lighter']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLORS['accent']};
        border-color: {COLORS['accent']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {COLORS['accent']};
    }}

    /* === Radio Buttons === */
    QRadioButton {{
        color: {COLORS['text']};
        spacing: 8px;
    }}
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {COLORS['border_light']};
        border-radius: 9px;
        background-color: {COLORS['background_lighter']};
    }}
    QRadioButton::indicator:checked {{
        background-color: {COLORS['accent']};
        border-color: {COLORS['accent']};
    }}
    QRadioButton::indicator:hover {{
        border-color: {COLORS['accent']};
    }}

    /* === Tab Widget === */
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        background-color: {COLORS['background_light']};
    }}
    QTabBar::tab {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        padding: 8px 16px;
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {COLORS['background_light']};
        border-bottom: 1px solid {COLORS['background_light']};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {COLORS['background_hover']};
    }}

    /* === Tables === */
    QTableWidget, QTableView {{
        background-color: {COLORS['background_light']};
        alternate-background-color: {COLORS['background_lighter']};
        gridline-color: {COLORS['border']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
    }}
    QTableWidget::item, QTableView::item {{
        padding: 5px;
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background-color: {COLORS['accent']};
    }}
    QHeaderView::section {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        padding: 8px;
        border: none;
        border-right: 1px solid {COLORS['border']};
        border-bottom: 1px solid {COLORS['border']};
        font-weight: bold;
    }}

    /* === List Widget === */
    QListWidget {{
        background-color: {COLORS['background_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
    }}
    QListWidget::item {{
        padding: 5px;
    }}
    QListWidget::item:selected {{
        background-color: {COLORS['accent']};
    }}
    QListWidget::item:hover:!selected {{
        background-color: {COLORS['background_hover']};
    }}

    /* === Progress Bar === */
    QProgressBar {{
        background-color: {COLORS['background_lighter']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        text-align: center;
        color: {COLORS['text']};
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['accent']};
        border-radius: 2px;
    }}

    /* === Slider === */
    QSlider::groove:horizontal {{
        background-color: {COLORS['background_lighter']};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background-color: {COLORS['accent']};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background-color: {COLORS['accent_hover']};
    }}

    /* === Menu Bar === */
    QMenuBar {{
        background-color: {COLORS['background_light']};
        color: {COLORS['text']};
        border-bottom: 1px solid {COLORS['border']};
    }}
    QMenuBar::item {{
        padding: 6px 12px;
    }}
    QMenuBar::item:selected {{
        background-color: {COLORS['accent']};
    }}

    /* === Menus === */
    QMenu {{
        background-color: {COLORS['background_lighter']};
        border: 1px solid {COLORS['border']};
        padding: 5px;
    }}
    QMenu::item {{
        padding: 8px 30px 8px 20px;
    }}
    QMenu::item:selected {{
        background-color: {COLORS['accent']};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {COLORS['border']};
        margin: 5px 10px;
    }}

    /* === Tool Tips === */
    QToolTip {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        padding: 5px;
        border-radius: 3px;
    }}

    /* === Dialogs === */
    QDialog {{
        background-color: {COLORS['background']};
    }}

    /* === Message Box === */
    QMessageBox {{
        background-color: {COLORS['background']};
    }}
    QMessageBox QLabel {{
        color: {COLORS['text']};
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {COLORS['border']};
    }}
    QSplitter::handle:hover {{
        background-color: {COLORS['accent']};
    }}

    /* === Status Bar === */
    QStatusBar {{
        background-color: {COLORS['background_light']};
        color: {COLORS['text_secondary']};
        border-top: 1px solid {COLORS['border']};
    }}

    /* === Frame === */
    QFrame[frameShape="4"], /* HLine */
    QFrame[frameShape="5"]  /* VLine */ {{
        color: {COLORS['border']};
    }}

    /* === Calendar Widget === */
    QCalendarWidget {{
        background-color: {COLORS['background_light']};
    }}
    QCalendarWidget QToolButton {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        border: none;
        border-radius: 3px;
        padding: 5px;
    }}
    QCalendarWidget QToolButton:hover {{
        background-color: {COLORS['background_hover']};
    }}
    QCalendarWidget QMenu {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
    }}
    QCalendarWidget QSpinBox {{
        background-color: {COLORS['background_lighter']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border_light']};
    }}
    QCalendarWidget QAbstractItemView {{
        background-color: {COLORS['background_light']};
        selection-background-color: {COLORS['accent']};
        selection-color: {COLORS['text']};
    }}
    QCalendarWidget QWidget#qt_calendar_navigationbar {{
        background-color: {COLORS['background_lighter']};
    }}
"""


def apply_theme(app):
    """
    Apply the dark theme to the application

    Args:
        app: QApplication instance
    """
    app.setStyleSheet(DARK_THEME)


def get_color(color_name):
    """
    Get a color value from the palette

    Args:
        color_name: Key from COLORS dict

    Returns:
        Color hex string or None if not found
    """
    return COLORS.get(color_name)
