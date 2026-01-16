#!/usr/bin/env python3
"""
Global Theme for Cosmos Collection
Provides consistent modern dark theme styling across the entire application.
"""

# --------------------------------------------------------------------
# 1. Color palette
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# 2. Modern Dark Theme (Qt Style Sheet)
# --------------------------------------------------------------------
DARK_THEME = f"""
/* Root variables ------------------------------------------------- */
:root {{
    --bg-base:        {COLORS['background']};
    --bg-light:       {COLORS['background_light']};
    --accent:         {COLORS['accent']};
    --accent-hover:   {COLORS['accent_hover']};
    --text-main:      {COLORS['text']};
    --text-sub:       {COLORS['text_secondary']};
    --border-light:   {COLORS['border_light']};
}}

/* Global widget styling ---------------------------------------- */
QWidget {{
    background-color: var(--bg-base);
    color:            var(--text-main);
    font-family:      "Segoe UI", Arial, sans-serif;
    font-size:        11pt;
    line-height:     1.4em;
}}

QLineEdit, QTextEdit, QPlainTextEdit,
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: var(--bg-light);
    color:            var(--text-main);
    border:           1px solid var(--border-light);
    padding:          5px;
    border-radius:    3px;
}}
/* Focus ring */
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: var(--accent);
}}

/* Buttons ------------------------------------------------------- */
QPushButton {{
    background: transparent;
    color:      var(--text-main);
    border:     1px solid var(--accent);
    padding:    6px 12px;
    border-radius: 4px;
}}
QPushButton:hover {{ background-color: rgba(0,120,212,0.08); }}
QPushButton:pressed {{ background-color: rgba(0,120,212,0.16); }}
QPushButton:disabled {{
    color:      var(--text-sub);
    border-color: var(--border-light);
}}

/* Tool buttons (toolbar icons) --------------------------------- */
QToolButton {{
    background: transparent;
    border:     none;
}}
QToolButton:hover {{ background: rgba(0,120,212,0.08); }}
QToolButton:pressed {{ background: rgba(0,120,212,0.16); }}

/* Tab widget ---------------------------------------------------- */
QTabBar::tab {{
    background-color: var(--bg-light);
    color:            var(--text-main);
    padding:          6px 12px;
    border:           1px solid transparent;
}}
QTabBar::tab:selected, QTabBar::tab:hover {{
    background-color: var(--bg-base);
}}
QTabWidget::pane {{
    background: var(--bg-light);
    border-top-left-radius: 0;
}}

/* Tooltips ----------------------------------------------------- */
QToolTip {{
    background-color: var(--bg-light);
    color:            var(--text-main);
    border:           1px solid var(--accent);
    padding:          4px 8px;
    border-radius:    3px;
}}

/* Combo boxes ---------------------------------------------------- */
QComboBox {{
    background-color: var(--bg-light);
    color:            var(--text-main);
    border:           1px solid var(--border-light);
    padding:          2px 25px 2px 6px;
    border-radius:    3px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: none;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid var(--text-main);
    margin-right: 5px;
}}
/* Popup view */
QComboBox QAbstractItemView {{
    background-color: var(--bg-light);
    color:            var(--text-main);
    selection-background-color: var(--accent);
    border:           1px solid var(--border-light);
}}

/* Sliders ------------------------------------------------------- */
QSlider::groove:horizontal {{
    background-color: var(--bg-light);
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: var(--accent);
    width: 12px; height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background-color: var(--accent-hover); }}

/* Progress bars ------------------------------------------------- */
QProgressBar {{
    background-color: var(--bg-light);
    border: 1px solid var(--border-light);
    text-align:center;
}}
QProgressBar::chunk {{
    background-color: var(--accent);
    border-radius: 2px;
}}

/* Scrollbars ----------------------------------------------------- */
QScrollBar:vertical, QScrollBar:horizontal {{
    background-color: transparent;
    width: 12px; height: 12px;
}}
QScrollBar::handle {{
    background-color: rgba(255,255,255,0.2);
    border-radius: 6px;
}}
QScrollBar::handle:hover {{ background-color: var(--accent); }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    height: 0; width: 0;
}}

/* Status bar ----------------------------------------------------- */
QStatusBar {{
    background-color: var(--bg-light);
    color:            var(--text-sub);
    border-top: 1px solid var(--border-light);
}}

/* Check boxes & radio buttons ----------------------------------- */
QCheckBox, QRadioButton {{
    color: var(--text-main); spacing: 8px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px; height: 18px;
    border: 1px solid var(--border-light);
    background-color: var(--bg-light);
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: var(--accent); border-color: var(--accent);
}}

/* Tables -------------------------------------------------------- */
QTableWidget, QTableView {{
    background-color: var(--bg-light);
    alternate-background-color: var(--bg-base);
    gridline-color: var(--border-light);
}}
QHeaderView::section {{
    background-color: var(--bg-base);
    color:            var(--text-main);
    padding:          8px;
    border-bottom: 1px solid var(--border-light);
}}

/* List widget ---------------------------------------------------- */
QListWidget {{
    background-color: var(--bg-light);
}}
QListWidget::item:selected {{
    background-color: var(--accent);
}}

/* Calendar Widget ----------------------------------------------- */
QCalendarWidget {{
    background-color: var(--bg-light);
}}
"""

# --------------------------------------------------------------------
# 3. Helper functions – unchanged
# --------------------------------------------------------------------
def apply_theme(app):
    """Apply the modern dark theme to the application."""
    app.setStyleSheet(DARK_THEME)

def get_color(color_name):
    """Return a hex string from COLORS dict or None if missing."""
    return COLORS.get(color_name)
