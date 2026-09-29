#!/usr/bin/env python3
"""
Global Theme for Cosmos Collection
Builds the application's colors, palette, font and Qt style sheet from the
user's theme settings (mode, accent color and text options).

Other modules read colors from the shared COLORS dict. apply_theme() refreshes
it in place, so windows opened after a theme change pick up the new colors.
"""

import os
import tempfile
from dataclasses import dataclass, asdict

from PySide6.QtCore import QObject, QPointF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPalette

# --------------------------------------------------------------------
# 1. Theme settings
# --------------------------------------------------------------------
MODES = {
    'system': 'System',
    'dark': 'Dark',
    'light': 'Light',
    'night': 'Night Vision',
}

DEFAULT_ACCENT = '#0078d4'

ACCENT_PRESETS = {
    'Blue': '#0078d4',
    'Purple': '#8a5cd6',
    'Teal': '#00a3a3',
    'Green': '#2e9e44',
    'Amber': '#d99a00',
    'Orange': '#e0622a',
    'Red': '#d13438',
    'Pink': '#d6408f',
}

FONT_SIZE_MIN = 8
FONT_SIZE_MAX = 16

_SETTINGS_ORG = "CosmosCollection"
_SETTINGS_APP = "CosmosCollection"


@dataclass
class ThemeSettings:
    """User-selectable theme options.

    An empty font_family or a font_size of 0 means "use the platform default".
    """
    mode: str = 'dark'
    accent: str = DEFAULT_ACCENT
    font_family: str = ''
    font_size: int = 0
    bold_headings: bool = False


def load_settings():
    """Read the saved theme settings, falling back to defaults."""
    defaults = ThemeSettings()
    settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    mode = settings.value("theme/mode", defaults.mode, type=str)
    accent = settings.value("theme/accent", defaults.accent, type=str)
    font_size = settings.value("theme/font_size", defaults.font_size, type=int)
    return ThemeSettings(
        mode=mode if mode in MODES else defaults.mode,
        accent=accent if QColor.isValidColorName(accent) else defaults.accent,
        font_family=settings.value("theme/font_family", defaults.font_family, type=str),
        font_size=font_size if font_size == 0 or FONT_SIZE_MIN <= font_size <= FONT_SIZE_MAX else 0,
        bold_headings=settings.value("theme/bold_headings", defaults.bold_headings, type=bool),
    )


def save_settings(theme_settings):
    """Persist theme settings to QSettings."""
    settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    for key, value in asdict(theme_settings).items():
        settings.setValue(f"theme/{key}", value)


# --------------------------------------------------------------------
# 2. Color palettes
# --------------------------------------------------------------------
# 'overlay' is the RGB used for translucent handles (e.g. scrollbars).
# A palette may pin 'accent' and 'text_on_accent'; otherwise they come from
# the user's accent color.
_BASE_PALETTES = {
    'dark': {
        'background': '#2b2b2b',
        'background_light': '#353535',
        'background_lighter': '#404040',
        'background_hover': '#4a4a4a',
        'border': '#555555',
        'border_light': '#666666',
        'text': '#ffffff',
        'text_secondary': '#cccccc',
        'text_disabled': '#888888',
        'error': '#ff4444',
        'error_bg': '#4a2020',
        'warning': '#ffcc00',
        'success': '#44ff44',
        'info': '#88ccff',
        'favorite': '#FFD700',
        'overlay': (255, 255, 255),
    },
    'light': {
        'background': '#f3f3f3',
        'background_light': '#ffffff',
        'background_lighter': '#e6e6e6',
        'background_hover': '#dadada',
        'border': '#c4c4c4',
        'border_light': '#a8a8a8',
        'text': '#1a1a1a',
        'text_secondary': '#454545',
        'text_disabled': '#8a8a8a',
        'error': '#c62828',
        'error_bg': '#fde4e4',
        'warning': '#a86400',
        'success': '#2e7d32',
        'info': '#0b62a8',
        'favorite': '#b8860b',
        'overlay': (0, 0, 0),
    },
    # Red-only, low-brightness palette that preserves dark-adapted vision.
    # Status colors are told apart by brightness rather than hue.
    'night': {
        'background': '#0a0000',
        'background_light': '#140000',
        'background_lighter': '#1e0000',
        'background_hover': '#2a0000',
        'border': '#3a0000',
        'border_light': '#520000',
        'text': '#c41e1e',
        'text_secondary': '#961616',
        'text_disabled': '#5c0e0e',
        'error': '#ff3b3b',
        'error_bg': '#330000',
        'warning': '#e03030',
        'success': '#a81c1c',
        'info': '#8a1414',
        'favorite': '#d02828',
        'accent': '#7a0000',
        'text_on_accent': '#ff4a4a',
        'overlay': (196, 30, 30),
    },
}


def platform_color_scheme():
    """Return 'dark' or 'light' for the operating system's color scheme."""
    app = QGuiApplication.instance()
    if app is not None and app.styleHints().colorScheme() == Qt.ColorScheme.Light:
        return 'light'
    return 'dark'


def resolve_mode(theme_settings):
    """Return the concrete palette name ('dark', 'light' or 'night')."""
    if theme_settings.mode == 'system':
        return platform_color_scheme()
    return theme_settings.mode if theme_settings.mode in _BASE_PALETTES else 'dark'


def _text_on(color):
    """Black or white, whichever reads better on the given background."""
    c = QColor(color)
    luminance = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
    return '#000000' if luminance > 0.55 else '#ffffff'


def _rgba(color, alpha):
    """CSS rgba() string for a color (hex string or RGB tuple) at the given alpha."""
    if isinstance(color, tuple):
        r, g, b = color
    else:
        c = QColor(color)
        r, g, b = c.red(), c.green(), c.blue()
    return f"rgba({r},{g},{b},{alpha})"


def build_colors(theme_settings):
    """Return the full color dict for the given settings."""
    colors = dict(_BASE_PALETTES[resolve_mode(theme_settings)])
    if 'accent' not in colors:
        colors['accent'] = QColor(theme_settings.accent).name()
    if 'text_on_accent' not in colors:
        colors['text_on_accent'] = _text_on(colors['accent'])
    accent = QColor(colors['accent'])
    colors['accent_hover'] = accent.darker(112).name()
    colors['accent_pressed'] = accent.darker(135).name()
    return colors


# --------------------------------------------------------------------
# 3. Palette, font and Qt style sheet
# --------------------------------------------------------------------
def build_palette(colors):
    """QPalette matching the color dict, for widgets no style sheet rule covers."""
    palette = QPalette()
    roles = {
        QPalette.Window: colors['background'],
        QPalette.WindowText: colors['text'],
        QPalette.Base: colors['background_light'],
        QPalette.AlternateBase: colors['background'],
        QPalette.Text: colors['text'],
        QPalette.PlaceholderText: colors['text_disabled'],
        QPalette.Button: colors['background_lighter'],
        QPalette.ButtonText: colors['text'],
        QPalette.BrightText: colors['error'],
        QPalette.Highlight: colors['accent'],
        QPalette.HighlightedText: colors['text_on_accent'],
        QPalette.ToolTipBase: colors['background_light'],
        QPalette.ToolTipText: colors['text'],
        QPalette.Link: colors['accent'],
        QPalette.LinkVisited: colors['accent_pressed'],
        QPalette.Light: colors['background_hover'],
        QPalette.Midlight: colors['background_lighter'],
        QPalette.Mid: colors['border'],
        QPalette.Dark: colors['border'],
        QPalette.Shadow: colors['background'],
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(colors['text_disabled']))
    return palette


# Captured the first time a font is built, before the theme changes the app font.
_platform_font = None


def platform_font():
    """The platform's default application font (before any theme font)."""
    global _platform_font
    if _platform_font is None:
        _platform_font = QFont(QGuiApplication.font())
    return QFont(_platform_font)


def build_font(theme_settings):
    """Application font for the given settings."""
    font = platform_font()
    if theme_settings.font_family:
        font.setFamily(theme_settings.font_family)
    if theme_settings.font_size:
        font.setPointSize(theme_settings.font_size)
    return font


def _arrow_image(direction, color):
    """Path to a small triangle PNG ('up' or 'down') in the given color.

    Style sheets can only draw arrow subcontrols from image files, so these
    are rendered once per color into a temp folder and reused.
    """
    folder = os.path.join(tempfile.gettempdir(), "CosmosCollection", "theme")
    path = os.path.join(folder, f"arrow_{direction}_{QColor(color).name()[1:]}.png")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        # Drawn at 2x the size used in the style sheet so it stays sharp on HiDPI screens.
        image = QImage(16, 10, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        points = [QPointF(0, 10), QPointF(16, 10), QPointF(8, 0)] if direction == 'up' \
            else [QPointF(0, 0), QPointF(16, 0), QPointF(8, 10)]
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        painter.drawPolygon(points)
        painter.end()
        image.save(path)
    return path.replace("\\", "/")


def build_stylesheet(colors, theme_settings):
    """Qt style sheet for the color dict and text options."""
    c = colors
    hover_bg = _rgba(c['accent'], 0.10)
    pressed_bg = _rgba(c['accent'], 0.20)
    handle = _rgba(c['overlay'], 0.25)

    qss = f"""
/* Inputs ------------------------------------------------------- */
QLineEdit, QTextEdit, QPlainTextEdit,
QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit {{
    background-color: {c['background_light']};
    color:            {c['text']};
    border:           1px solid {c['border_light']};
    padding:          4px;
    border-radius:    3px;
    selection-background-color: {c['accent']};
    selection-color:  {c['text_on_accent']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QDateEdit:focus, QTimeEdit:focus, QDateTimeEdit:focus {{
    border-color: {c['accent']};
}}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c['text_disabled']};
}}

/* Buttons ------------------------------------------------------ */
QPushButton {{
    background: transparent;
    color:      {c['text']};
    border:     1px solid {c['accent']};
    padding:    5px 12px;
    border-radius: 4px;
}}
QPushButton:hover {{ background-color: {hover_bg}; }}
QPushButton:pressed, QPushButton:checked {{ background-color: {pressed_bg}; }}
QPushButton:default {{ border-width: 2px; }}
QPushButton:disabled {{
    color:        {c['text_disabled']};
    border-color: {c['border']};
}}

/* Tool buttons (toolbar icons) --------------------------------- */
QToolButton {{
    background: transparent;
    border:     none;
    padding:    3px;
    border-radius: 3px;
}}
QToolButton:hover {{ background: {hover_bg}; }}
QToolButton:pressed, QToolButton:checked {{ background: {pressed_bg}; }}

/* Tab widget --------------------------------------------------- */
QTabWidget::pane {{
    border: 1px solid {c['border']};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {c['background_light']};
    color:            {c['text_secondary']};
    padding:          6px 12px;
    border:           1px solid {c['border']};
    border-bottom:    none;
}}
QTabBar::tab:selected {{
    background-color: {c['background']};
    color:            {c['text']};
    border-top:       2px solid {c['accent']};
}}
QTabBar::tab:hover:!selected {{
    background-color: {c['background_hover']};
}}

/* Tooltips ----------------------------------------------------- */
QToolTip {{
    background-color: {c['background_light']};
    color:            {c['text']};
    border:           1px solid {c['accent']};
    padding:          4px 8px;
}}

/* Combo boxes -------------------------------------------------- */
QComboBox {{
    padding: 2px 25px 2px 6px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border: none;
}}
QComboBox::down-arrow {{
    image: url("{_arrow_image('down', c['text'])}");
    width: 8px; height: 5px;
    margin-right: 5px;
}}
QComboBox::down-arrow:disabled {{
    image: url("{_arrow_image('down', c['text_disabled'])}");
}}

/* Spin box buttons --------------------------------------------- */
QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit {{
    padding-right: 18px;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 16px;
    border: none;
    background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {hover_bg};
}}
QAbstractSpinBox::up-arrow {{
    image: url("{_arrow_image('up', c['text'])}");
    width: 8px; height: 5px;
}}
QAbstractSpinBox::down-arrow {{
    image: url("{_arrow_image('down', c['text'])}");
    width: 8px; height: 5px;
}}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url("{_arrow_image('up', c['text_disabled'])}");
}}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url("{_arrow_image('down', c['text_disabled'])}");
}}
QComboBox QAbstractItemView {{
    background-color: {c['background_light']};
    color:            {c['text']};
    selection-background-color: {c['accent']};
    selection-color:  {c['text_on_accent']};
    border:           1px solid {c['border_light']};
}}

/* Sliders ------------------------------------------------------ */
QSlider::groove:horizontal {{
    background-color: {c['background_lighter']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background-color: {c['accent']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: {c['accent']};
    width: 12px; height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background-color: {c['accent_hover']}; }}

/* Progress bars ------------------------------------------------ */
QProgressBar {{
    background-color: {c['background_light']};
    border: 1px solid {c['border_light']};
    border-radius: 3px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {c['accent']};
    border-radius: 2px;
}}

/* Scrollbars --------------------------------------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {handle};
    min-height: 24px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background-color: {handle};
    min-width: 24px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:hover {{ background-color: {c['accent']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* Status bar --------------------------------------------------- */
QStatusBar {{
    background-color: {c['background_light']};
    color:            {c['text_secondary']};
    border-top: 1px solid {c['border_light']};
}}

/* Check boxes & radio buttons ---------------------------------- */
QCheckBox, QRadioButton {{
    spacing: 8px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {c['border_light']};
    background-color: {c['background_light']};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c['accent']};
    border-color: {c['accent']};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {c['border']};
    background-color: {c['background']};
}}

/* Tables & lists ----------------------------------------------- */
QTableWidget, QTableView, QTreeView, QTreeWidget, QListView, QListWidget {{
    color: {c['text']};
    background-color: {c['background_light']};
    alternate-background-color: {c['background']};
    gridline-color: {c['border']};
    selection-background-color: {c['accent']};
    selection-color: {c['text_on_accent']};
}}
QHeaderView::section {{
    background-color: {c['background']};
    color:            {c['text']};
    padding:          6px;
    border: none;
    border-right:  1px solid {c['border']};
    border-bottom: 1px solid {c['border_light']};
}}

/* Group boxes -------------------------------------------------- */
QGroupBox {{
    border: 1px solid {c['border']};
    border-radius: 4px;
    margin-top: 1.1em;
    padding-top: 0.4em;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {c['text']};
}}

/* Menus -------------------------------------------------------- */
QMenu {{
    background-color: {c['background_light']};
    border: 1px solid {c['border']};
}}
QMenu::item:selected {{
    background-color: {c['accent']};
    color: {c['text_on_accent']};
}}
QMenu::separator {{
    height: 1px;
    background: {c['border']};
    margin: 4px 8px;
}}

/* Dock widgets ------------------------------------------------- */
QDockWidget::title {{
    background-color: {c['background_light']};
    color: {c['text']};
    padding: 6px 10px;
    border-bottom: 1px solid {c['border']};
    text-align: left;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    border: none;
    background: transparent;
    padding: 2px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {pressed_bg};
}}
"""
    # Headings are set explicitly either way, so a widget-level preview of
    # this style sheet overrides whatever the application style sheet uses.
    heading_weight = 'bold' if theme_settings.bold_headings else 'normal'
    qss += f"""
/* Headings ----------------------------------------------------- */
QGroupBox, QDockWidget, QHeaderView::section {{ font-weight: {heading_weight}; }}
"""
    if theme_settings.bold_headings:
        # Keep the bold heading font from propagating to the contents.
        qss += "QGroupBox > *, QDockWidget > * { font-weight: normal; }\n"
    return qss


# --------------------------------------------------------------------
# 4. Applying the theme
# --------------------------------------------------------------------
# Shared color dict read by every module. Refreshed in place by apply_theme().
COLORS = build_colors(ThemeSettings())


class ThemeManager(QObject):
    """Emits theme_changed after a theme is applied, and follows the OS
    color scheme while the 'system' mode is selected."""

    theme_changed = Signal()

    def __init__(self, app):
        super().__init__(app)
        self.settings = ThemeSettings()
        app.styleHints().colorSchemeChanged.connect(self._on_color_scheme_changed)

    def _on_color_scheme_changed(self, _scheme):
        if self.settings.mode == 'system':
            apply_theme(QGuiApplication.instance(), self.settings)


_manager = None
_fusion_set = False


def theme_manager():
    """The application's ThemeManager (requires a QApplication)."""
    global _manager
    if _manager is None:
        _manager = ThemeManager(QGuiApplication.instance())
    return _manager


def apply_theme(app, theme_settings=None):
    """Apply the theme (the saved settings if none given) to the application."""
    if theme_settings is None:
        theme_settings = load_settings()
    colors = build_colors(theme_settings)
    COLORS.clear()
    COLORS.update(colors)

    # Fusion renders palettes and style sheets consistently in every mode.
    # (Once a style sheet is set, app.style() is Qt's style sheet wrapper, so
    # track this with a flag rather than checking the style's name.)
    global _fusion_set
    if not _fusion_set:
        app.setStyle("Fusion")
        _fusion_set = True
    app.setPalette(build_palette(colors))
    app.setFont(build_font(theme_settings))
    app.setStyleSheet(build_stylesheet(colors, theme_settings))

    manager = theme_manager()
    manager.settings = theme_settings
    manager.theme_changed.emit()


def get_color(color_name):
    """Return a hex string from COLORS dict or None if missing."""
    return COLORS.get(color_name)
