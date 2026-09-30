#!/usr/bin/env python3
"""
Global Theme for Cosmos Collection
Builds the application's colors, palette, font and Qt style sheet from the
user's theme settings (mode, accent color and text options).

Other modules read colors from the shared COLORS dict. apply_theme() refreshes
it in place, so windows opened after a theme change pick up the new colors.
"""

import logging
import os
import tempfile
import weakref
from dataclasses import dataclass, asdict

from PySide6.QtCore import QObject, QPointF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPalette, QPen

logger = logging.getLogger(__name__)

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

CHECKBOX_STYLES = {
    'check': 'Checkmark',
    'filled': 'Filled',
}

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
    checkbox_style: str = 'check'  # a key of CHECKBOX_STYLES


def load_settings():
    """Read the saved theme settings, falling back to defaults."""
    defaults = ThemeSettings()
    settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    mode = settings.value("theme/mode", defaults.mode, type=str)
    accent = settings.value("theme/accent", defaults.accent, type=str)
    font_size = settings.value("theme/font_size", defaults.font_size, type=int)
    checkbox_style = settings.value("theme/checkbox_style", defaults.checkbox_style, type=str)
    return ThemeSettings(
        mode=mode if mode in MODES else defaults.mode,
        accent=accent if QColor.isValidColorName(accent) else defaults.accent,
        font_family=settings.value("theme/font_family", defaults.font_family, type=str),
        font_size=font_size if font_size == 0 or FONT_SIZE_MIN <= font_size <= FONT_SIZE_MAX else 0,
        bold_headings=settings.value("theme/bold_headings", defaults.bold_headings, type=bool),
        checkbox_style=checkbox_style if checkbox_style in CHECKBOX_STYLES else defaults.checkbox_style,
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
        'link': '#6ea8fe',
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
        'warning': '#945800',
        'success': '#2e7d32',
        'info': '#0b62a8',
        'favorite': '#7e5c08',
        'link': '#0b62a8',
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
        # Pure red tops out near 5:1 contrast on black, so readable text stays
        # in the ~180-255 red range; disabled text is deliberately dimmer.
        'text': '#d22020',
        'text_secondary': '#b41c1c',
        'text_disabled': '#701212',
        'error': '#ff2a2a',
        'error_bg': '#330000',
        'warning': '#e62424',
        'success': '#c81e1e',
        'info': '#b41c1c',
        'favorite': '#dc2424',
        'link': '#e02626',
        'accent': '#7a0000',
        'text_on_accent': '#ff3030',
        'overlay': (210, 32, 32),
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


def _luminance(color):
    """WCAG relative luminance of a color (0 = black, 1 = white)."""
    def linear(v):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    c = QColor(color)
    return 0.2126 * linear(c.redF()) + 0.7152 * linear(c.greenF()) + 0.0722 * linear(c.blueF())


def _contrast(a, b):
    """WCAG contrast ratio between two colors (1 to 21)."""
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _pick_text(background, preferred, alternative, minimum):
    """preferred if it reaches the minimum contrast on background, otherwise
    whichever of the two contrasts more."""
    if _contrast(preferred, background) >= minimum:
        return preferred
    return max((preferred, alternative), key=lambda c: _contrast(c, background))


def _text_on(color):
    """White or black text for the given background (white when it reads well)."""
    return _pick_text(color, '#ffffff', '#000000', 4.5)


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
        QPalette.Link: colors['link'],
        QPalette.LinkVisited: colors['link'],
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


def font_scale_for(theme_settings):
    """How much the chosen text size enlarges text relative to the platform
    default (1.0 when the default size is used)."""
    base = platform_font().pointSizeF()
    chosen = build_font(theme_settings).pointSizeF()
    return chosen / base if base > 0 and chosen > 0 else 1.0


# Fixed font sizes in the code (style sheets, charts) were designed at the
# platform's default size; they're scaled by this so the text size setting
# reaches them too. Set by apply_theme().
_font_scale = 1.0


def font_size(points):
    """A fixed style sheet font size in pt, scaled with the theme's text size,
    e.g. f"font-size: {font_size(9)};" -> "font-size: 9pt;" at the default size."""
    return f"{round(points * _font_scale * 2) / 2:g}pt"


def font_px(pixels):
    """Like font_size, for sizes given in px."""
    return f"{round(pixels * _font_scale)}px"


def chart_font_size(points):
    """A fixed matplotlib font size (points), scaled with the theme's text size."""
    return points * _font_scale


def _theme_image(kind, color):
    """Path to a small PNG of a style sheet glyph in the given color.

    kind: 'up' / 'down' (arrows, drawn 16x10 for 8x5), or 'check' / 'dot' /
    'dash' (check box and radio button marks, drawn 32x32 for a 16px box).
    Style sheets can only draw these subcontrols from image files, so they're
    rendered once per color into a temp folder and reused. Everything is drawn
    at 2x the size it's shown at, so it stays sharp on HiDPI screens.
    """
    folder = os.path.join(tempfile.gettempdir(), "CosmosCollection", "theme")
    prefix = f"arrow_{kind}" if kind in ('up', 'down') else kind
    path = os.path.join(folder, f"{prefix}_{QColor(color).name()[1:]}.png")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        size = (16, 10) if kind in ('up', 'down') else (32, 32)
        image = QImage(*size, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        if kind in ('up', 'down'):
            points = [QPointF(0, 10), QPointF(16, 10), QPointF(8, 0)] if kind == 'up' \
                else [QPointF(0, 0), QPointF(16, 0), QPointF(8, 10)]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color))
            painter.drawPolygon(points)
        elif kind == 'dot':
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(16, 16), 6.5, 6.5)
        else:
            pen = QPen(QColor(color), 4.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            if kind == 'check':
                painter.drawPolyline([QPointF(8, 16.5), QPointF(13.5, 22), QPointF(24, 10)])
            else:  # dash
                painter.drawLine(QPointF(9, 16), QPointF(23, 16))
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
    image: url("{_theme_image('down', c['text'])}");
    width: 8px; height: 5px;
    margin-right: 5px;
}}
QComboBox::down-arrow:disabled {{
    image: url("{_theme_image('down', c['text_disabled'])}");
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
    image: url("{_theme_image('up', c['text'])}");
    width: 8px; height: 5px;
}}
QAbstractSpinBox::down-arrow {{
    image: url("{_theme_image('down', c['text'])}");
    width: 8px; height: 5px;
}}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url("{_theme_image('up', c['text_disabled'])}");
}}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url("{_theme_image('down', c['text_disabled'])}");
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
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c['accent']};
}}
QCheckBox::indicator:checked, QCheckBox::indicator:indeterminate,
QRadioButton::indicator:checked {{
    background-color: {c['accent']};
    border-color: {c['accent']};
}}
QCheckBox::indicator:indeterminate {{
    image: url("{_theme_image('dash', c['text_on_accent'])}");
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {c['border']};
    background-color: {c['background']};
}}
QCheckBox::indicator:checked:disabled, QCheckBox::indicator:indeterminate:disabled,
QRadioButton::indicator:checked:disabled {{
    border-color: {c['border']};
    background-color: {c['border']};
}}
QCheckBox::indicator:indeterminate:disabled {{
    image: url("{_theme_image('dash', c['text_disabled'])}");
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
    # Check box style: 'filled' marks a checked box by its accent fill alone;
    # 'check' adds a checkmark (and a dot in radio buttons). Filled resets the
    # image explicitly so a widget-level preview overrides the app style sheet.
    if theme_settings.checkbox_style == 'check':
        qss += f"""
/* Check box marks ---------------------------------------------- */
QCheckBox::indicator:checked {{
    image: url("{_theme_image('check', c['text_on_accent'])}");
}}
QRadioButton::indicator:checked {{
    image: url("{_theme_image('dot', c['text_on_accent'])}");
}}
QCheckBox::indicator:checked:disabled {{
    image: url("{_theme_image('check', c['text_disabled'])}");
}}
QRadioButton::indicator:checked:disabled {{
    image: url("{_theme_image('dot', c['text_disabled'])}");
}}
"""
    else:
        qss += """
QCheckBox::indicator:checked, QRadioButton::indicator:checked,
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled { image: none; }
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


# --------------------------------------------------------------------
# Window title bars (Windows)
# --------------------------------------------------------------------
# The title bar is drawn by Windows, not Qt. Windows 11 lets an app set each
# window's title bar colors (DwmSetWindowAttribute); older versions ignore the
# color attributes, and other platforms skip this entirely.
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36
_DWMWA_COLOR_DEFAULT = 0xFFFFFFFF  # let Windows choose the color

# Bumped whenever title bars need restyling, so windows styled for an older
# theme are redone (see style_title_bar).
_title_bar_generation = 0


def _colorref(color):
    """A color as a Windows COLORREF (0x00BBGGRR)."""
    c = QColor(color)
    return c.red() | (c.green() << 8) | (c.blue() << 16)


def style_title_bar(window):
    """Match a top-level window's title bar to the theme: dark or light to
    follow the app mode, and fully colored in Night Vision so no bright strip
    spoils dark adaptation. window is a QWindow (widget.windowHandle())."""
    # Only for real Windows windows - other Qt platforms (e.g. offscreen) have
    # no native title bar, and their window ids aren't Windows handles
    if QGuiApplication.platformName() != 'windows' or window is None \
            or window.type() not in (Qt.Window, Qt.Dialog):
        return
    hwnd = int(window.winId())
    stamp = f"{_title_bar_generation}:{hwnd}"
    if window.property('_theme_title_bar') == stamp:
        return  # already styled for this theme (and this native window)
    if _current_mode == 'night':
        colors = (_colorref(COLORS['background']), _colorref(COLORS['text']), _colorref(COLORS['border']))
    else:
        colors = (_DWMWA_COLOR_DEFAULT,) * 3
    values = zip((_DWMWA_USE_IMMERSIVE_DARK_MODE, _DWMWA_CAPTION_COLOR, _DWMWA_TEXT_COLOR, _DWMWA_BORDER_COLOR),
                 (int(_current_mode != 'light'),) + colors)
    try:
        import ctypes
        dwmapi, user32 = ctypes.windll.dwmapi, ctypes.windll.user32
        for attribute, value in values:
            data = ctypes.c_uint32(value)
            # Unsupported attributes (older Windows) just return an error code
            dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), attribute, ctypes.byref(data), ctypes.sizeof(data))
        # Redraw the frame so an already-visible title bar picks up the change
        swp_flags = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020  # NOSIZE|NOMOVE|NOZORDER|NOACTIVATE|FRAMECHANGED
        user32.SetWindowPos(ctypes.c_void_p(hwnd), None, 0, 0, 0, 0, swp_flags)
    except (AttributeError, OSError) as e:
        logger.debug(f"Could not style window title bar: {e}")
        return
    window.setProperty('_theme_title_bar', stamp)


def _restyle_title_bars():
    """Restyle the title bars of all visible windows for the current theme."""
    global _title_bar_generation
    _title_bar_generation += 1
    for window in QGuiApplication.topLevelWindows():
        if window.isVisible():
            style_title_bar(window)


class ThemeManager(QObject):
    """Emits theme_changed after a theme is applied, follows the OS color
    scheme while the 'system' mode is selected, and keeps window title bars
    styled."""

    theme_changed = Signal()

    def __init__(self, app):
        super().__init__(app)
        self.settings = ThemeSettings()
        app.styleHints().colorSchemeChanged.connect(self._on_color_scheme_changed)
        # New windows get their title bar styled when first activated (a
        # per-window hook, rather than an app-wide event filter that would run
        # for every event)
        app.focusWindowChanged.connect(style_title_bar)

    def _on_color_scheme_changed(self, _scheme):
        if self.settings.mode == 'system':
            apply_theme(QGuiApplication.instance(), self.settings)
        else:
            # Qt resets title bars to the OS scheme; put the theme's back after it
            QTimer.singleShot(0, _restyle_title_bars)


_manager = None
_fusion_set = False
_current_mode = 'dark'  # resolved palette name of the applied theme


def theme_manager():
    """The application's ThemeManager (requires a QApplication)."""
    global _manager
    if _manager is None:
        _manager = ThemeManager(QGuiApplication.instance())
    return _manager


def apply_theme(app, theme_settings=None):
    """Apply the theme (the saved settings if none given) to the application."""
    global _current_mode, _font_scale
    if theme_settings is None:
        theme_settings = load_settings()
    colors = build_colors(theme_settings)
    COLORS.clear()
    COLORS.update(colors)
    _current_mode = resolve_mode(theme_settings)
    _font_scale = font_scale_for(theme_settings)

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
    _apply_chart_theme()
    _refresh_themed_styles()
    _restyle_title_bars()

    manager = theme_manager()
    manager.settings = theme_settings
    manager.theme_changed.emit()


def get_color(color_name):
    """Return a hex string from COLORS dict or None if missing."""
    return COLORS.get(color_name)


# Widgets with a style sheet or text built from theme colors. Each keeps its own
# bindings in widget._theme_bindings ({setter: (getter, build, applied)}):
# build functions often bind the window that owns the widget, and holding them
# here instead would keep closed windows alive. This way the set is weak and
# they're collected normally.
_themed_widgets = weakref.WeakSet()


def _bind(widget, setter, getter, build):
    value = build()
    getattr(widget, setter)(value)
    bindings = widget.__dict__.setdefault('_theme_bindings', {})
    bindings[setter] = (getter, build, value)
    _themed_widgets.add(widget)


def themed_style(widget, build):
    """Set a style sheet built from theme colors, and rebuild it on theme changes.

    build is a callable returning the style sheet, e.g.
        themed_style(label, lambda: f"color: {COLORS['text_secondary']};")
    It's called now and again after each theme change, so windows that stay
    open follow the theme. If the widget's style sheet is later set some other
    way, that newer style wins and the widget stops being rebuilt.
    """
    _bind(widget, 'setStyleSheet', 'styleSheet', build)


def themed_text(label, build):
    """Like themed_style, for label text (rich text) that embeds theme colors."""
    _bind(label, 'setText', 'text', build)


def _refresh_themed_styles():
    """Rebuild every registered style sheet and text with the current colors."""
    for widget in list(_themed_widgets):
        try:
            bindings = widget._theme_bindings
            for setter, (getter, build, applied) in list(bindings.items()):
                if getattr(widget, getter)() != applied:
                    # Set since by other code - that value is the current one
                    del bindings[setter]
                    continue
                value = build()
                getattr(widget, setter)(value)
                bindings[setter] = (getter, build, value)
        except RuntimeError:
            # The Qt widget was deleted while its Python wrapper lingered
            _themed_widgets.discard(widget)
            continue
        except Exception as e:
            # A failing rebuild must not stop the rest of the theme change
            logger.warning(f"Could not restyle {type(widget).__name__} for the new theme: {e}")
            _themed_widgets.discard(widget)
            continue
        if not bindings:
            _themed_widgets.discard(widget)


# --------------------------------------------------------------------
# 5. Adapting fixed colors to the current theme
# --------------------------------------------------------------------
# Some colors carry meaning through their hue (emission lines, green-to-red
# scales, chart series), so they can't simply become COLORS keys. These helpers
# keep that meaning while staying readable in light mode and red-only in
# Night Vision. They read the applied theme, so call them when building UI.

def current_mode():
    """Resolved palette name of the applied theme: 'dark', 'light' or 'night'."""
    return _current_mode


def _themed_hue(color, rank=None):
    """The color for the current mode before any contrast adjustment: unchanged,
    except Night Vision maps it to red. The red's brightness follows the
    color's brightness, or rank (0-1) for colors on an ordered scale whose
    order is carried by hue (such as green-to-red), since hue is lost."""
    c = QColor(color)
    if _current_mode == 'night':
        if rank is None:
            rank = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
        red = int(70 + 185 * rank)
        c = QColor(red, int(red * 0.12), int(red * 0.12))
    return c


def adapt_color(color, on_dark=False, background=None, rank=None, minimum=None):
    """Hex color adjusted to read on its background, keeping its hue meaning.

    The background defaults to the window background, or black when on_dark
    says the color sits on a fixed dark surface such as an image view. The
    color's lightness is shifted only as far as needed to reach 4.5:1 contrast
    (3:1 in Night Vision, which maps colors to red first), so colors that
    already read well are returned unchanged. rank: see _themed_hue. minimum
    overrides the target contrast (e.g. 3.0 for chart lines, which aren't text).
    """
    c = _themed_hue(color, rank)
    if background is None:
        background = '#000000' if on_dark else COLORS['background']
    night = _current_mode == 'night'
    if minimum is None:
        minimum = 3.0 if night else 4.5
    lighten = _luminance(background) < 0.18
    hue, saturation, lightness, _alpha = c.getHslF()
    # Past 0.5 lightness, red starts turning pink - stay red in Night Vision.
    max_lightness = 0.5 if night else 1.0
    for _ in range(40):
        # Small margin so rounding to a hex color can't dip below the minimum.
        if _contrast(c, background) >= minimum + 0.05:
            break
        lightness = min(max_lightness, lightness + 0.02) if lighten else max(0.0, lightness - 0.02)
        c.setHslF(hue, saturation, lightness)
    return c.name()


def tint(color, amount, rank=None):
    """Hex color blending the color into the panel background, for colored
    table cells and highlights. amount 0 = background, 1 = color.
    rank: see _themed_hue."""
    if _current_mode == 'night':
        # Red-only text only reads on dim backgrounds, so keep tints dim.
        amount *= 0.55
    base = QColor(COLORS['background_light'])
    top = _themed_hue(color, rank)
    return QColor(
        round(base.red() + (top.red() - base.red()) * amount),
        round(base.green() + (top.green() - base.green()) * amount),
        round(base.blue() + (top.blue() - base.blue()) * amount),
    ).name()


def contrast_text(background):
    """Hex text color that reads on the given background color."""
    if _current_mode == 'night':
        return _pick_text(background, COLORS['text_on_accent'], '#000000', 3.0)
    return _text_on(background)


# --------------------------------------------------------------------
# 6. Charts (matplotlib)
# --------------------------------------------------------------------
# apply_theme() sets matplotlib's defaults (backgrounds, text, ticks, grid,
# legend), so charts only choose their data colors - through chart_color().
# Charts are drawn with the colors current at draw time, so a chart that stays
# open should redraw on theme_manager().theme_changed.

# Sky shading behind time-of-night charts, (color, alpha) from day to night.
_SKY_SHADES = {
    'dark': {
        'day': ('#4a4a3a', 0.5),            # warm tint
        'civil': ('#2a3a4a', 0.6),          # light blue-gray
        'nautical': ('#1a2535', 0.7),       # medium blue
        'astronomical': ('#101520', 0.8),   # dark blue
        'night': ('#080a10', 0.9),          # very dark blue/black
    },
    'light': {
        'day': ('#f7f1dc', 1.0),
        'civil': ('#e3e9f2', 1.0),
        'nautical': ('#d2dbe9', 1.0),
        'astronomical': ('#c1cddf', 1.0),
        'night': ('#b0bfd6', 1.0),
    },
    'night': {
        'day': ('#1c0000', 1.0),
        'civil': ('#140000', 1.0),
        'nautical': ('#0e0000', 1.0),
        'astronomical': ('#080000', 1.0),
        'night': ('#030000', 1.0),
    },
}

# The background a chart line must stand out against in the worst case:
# the lightest plot background in dark palettes, the darkest in light.
_CHART_LINE_REFERENCE = {'dark': '#404040', 'light': '#b0bfd6', 'night': '#1c0000'}


def sky_shade(level):
    """(color, alpha) for sky shading: 'day', 'civil', 'nautical',
    'astronomical' or 'night'."""
    return _SKY_SHADES[_current_mode][level]


def chart_color(color, rank=None):
    """Hex color for chart data (lines, markers), adapted to the current theme
    so it stands out from the plot background, including sky shading.
    rank: see _themed_hue."""
    return adapt_color(color, background=_CHART_LINE_REFERENCE[_current_mode],
                       rank=rank, minimum=3.0)


def chart_background():
    """Figure (outer) background color for charts."""
    return COLORS['background']


def _apply_chart_theme():
    """Point matplotlib's defaults at the current theme colors."""
    try:
        import matplotlib
        from cycler import cycler
    except ImportError:
        return
    # Series drawn without an explicit color use matplotlib's default cycle
    default_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    matplotlib.rcParams.update({
        'font.size': chart_font_size(10),  # matplotlib's default, scaled with the text size
        'axes.prop_cycle': cycler(color=[chart_color(c) for c in default_cycle]),
        'figure.facecolor': COLORS['background'],
        'figure.edgecolor': COLORS['background'],
        'savefig.facecolor': COLORS['background'],
        'savefig.edgecolor': COLORS['background'],
        'axes.facecolor': COLORS['background_light'],
        'axes.edgecolor': COLORS['border_light'],
        'axes.labelcolor': COLORS['text'],
        'axes.titlecolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text_secondary'],
        'ytick.color': COLORS['text_secondary'],
        'grid.color': COLORS['border_light'],
        'legend.facecolor': COLORS['background_lighter'],
        'legend.edgecolor': COLORS['border_light'],
        'legend.labelcolor': COLORS['text'],
        'patch.edgecolor': COLORS['border_light'],
    })
