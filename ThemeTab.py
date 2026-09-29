#!/usr/bin/env python3
"""
Theme tab for the Settings dialog: mode, accent color and text options, with a
live preview. The preview is styled on its own, so nothing outside it changes
until the dialog is saved.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QColorDialog, QComboBox,
    QFontComboBox, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QToolButton, QVBoxLayout, QWidget,
)

from Theme import (
    ACCENT_PRESETS, COLORS, FONT_SIZE_MAX, FONT_SIZE_MIN, MODES, ThemeSettings,
    build_colors, build_font, build_stylesheet, platform_color_scheme, platform_font,
)

SWATCH_SIZE = 26
PREVIEW_DELAY_MS = 80


class ThemeSettingsTab(QWidget):
    """Controls for ThemeSettings plus a preview of the result."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._accent = ACCENT_PRESETS['Blue']
        self._custom_accent = None
        # The family the font combo showed after load/reset, and the stored value
        # it stands for, so an untouched combo keeps its stored value ('' = default)
        # even when the platform font isn't listed under the same name.
        self._shown_family = ''
        self._shown_family_value = ''

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self._update_preview)

        layout = QHBoxLayout(self)
        layout.addWidget(self._build_controls())
        layout.addWidget(self._build_preview(), 1)

    # ----------------------------------------------------------------
    # Controls
    # ----------------------------------------------------------------
    def _build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(340)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 6, 0)
        scroll.setWidget(controls)

        # Mode
        mode_group = QGroupBox("Mode")
        mode_layout = QGridLayout(mode_group)
        self.mode_buttons = {}
        self.mode_button_group = QButtonGroup(self)
        for i, (mode, label) in enumerate(MODES.items()):
            if mode == 'system':
                label = f"{label} (currently {MODES[platform_color_scheme()]})"
            button = QRadioButton(label)
            self.mode_buttons[mode] = button
            self.mode_button_group.addButton(button)
            mode_layout.addWidget(button, i // 2, i % 2)
        self.mode_buttons['system'].setToolTip("Follow the Windows light/dark setting")
        self.mode_buttons['night'].setToolTip(
            "Dim, red-only colors that preserve dark-adapted vision at the telescope"
        )
        self.mode_button_group.buttonToggled.connect(self._on_mode_toggled)
        controls_layout.addWidget(mode_group)

        # Accent color
        accent_group = QGroupBox("Accent Color")
        accent_layout = QVBoxLayout(accent_group)
        swatch_layout = QHBoxLayout()
        swatch_layout.setSpacing(6)
        self.accent_button_group = QButtonGroup(self)
        self.preset_swatches = {}
        for name, color in ACCENT_PRESETS.items():
            swatch = self._make_swatch(color, f"{name} ({color})")
            swatch.clicked.connect(lambda _checked=False, c=color: self._select_accent(c))
            self.preset_swatches[QColor(color).name()] = swatch
            swatch_layout.addWidget(swatch)
        self.custom_swatch = self._make_swatch(ACCENT_PRESETS['Blue'], "Custom color")
        self.custom_swatch.clicked.connect(lambda: self._select_accent(self._custom_accent))
        self.custom_swatch.hide()
        swatch_layout.addWidget(self.custom_swatch)
        swatch_layout.addStretch()
        accent_layout.addLayout(swatch_layout)

        custom_layout = QHBoxLayout()
        self.custom_accent_btn = QPushButton("Custom...")
        self.custom_accent_btn.setToolTip("Choose any accent color")
        self.custom_accent_btn.clicked.connect(self._choose_custom_accent)
        custom_layout.addWidget(self.custom_accent_btn)
        custom_layout.addStretch()
        accent_layout.addLayout(custom_layout)

        self.night_accent_note = QLabel("Night Vision uses a fixed deep-red accent.")
        self.night_accent_note.setWordWrap(True)
        self.night_accent_note.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; }}")
        self.night_accent_note.hide()
        accent_layout.addWidget(self.night_accent_note)
        controls_layout.addWidget(accent_group)

        # Text
        text_group = QGroupBox("Text")
        text_layout = QFormLayout(text_group)
        self.font_combo = QFontComboBox()
        self.font_combo.setToolTip("Font used throughout the application")
        self.font_combo.currentFontChanged.connect(self._schedule_preview)
        text_layout.addRow("Font:", self.font_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(FONT_SIZE_MIN, FONT_SIZE_MAX)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.setToolTip(
            "Base text size. Some labels use their own fixed size and\n"
            "won't follow this setting yet."
        )
        self.font_size_spin.valueChanged.connect(self._schedule_preview)
        text_layout.addRow("Size:", self.font_size_spin)

        self.bold_headings_checkbox = QCheckBox("Bold headings")
        self.bold_headings_checkbox.setToolTip("Bold group box titles, dock titles and table headers")
        self.bold_headings_checkbox.toggled.connect(self._schedule_preview)
        text_layout.addRow(self.bold_headings_checkbox)
        controls_layout.addWidget(text_group)

        # Reset + note
        reset_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Restore the default theme (not saved until you click Save)")
        reset_btn.clicked.connect(lambda: self.load(ThemeSettings()))
        reset_layout.addWidget(reset_btn)
        reset_layout.addStretch()
        controls_layout.addLayout(reset_layout)

        reopen_note = QLabel(
            "The theme is applied when you click Save. Some colors in windows "
            "that are already open update the next time they're opened."
        )
        reopen_note.setWordWrap(True)
        reopen_note.setStyleSheet(f"QLabel {{ color: {COLORS['text_disabled']}; font-size: 9pt; }}")
        controls_layout.addWidget(reopen_note)
        controls_layout.addStretch()
        return scroll

    def _make_swatch(self, color, tooltip):
        swatch = QToolButton()
        swatch.setCheckable(True)
        swatch.setFixedSize(SWATCH_SIZE, SWATCH_SIZE)
        swatch.setCursor(Qt.PointingHandCursor)
        swatch.setToolTip(tooltip)
        self._set_swatch_color(swatch, color)
        self.accent_button_group.addButton(swatch)
        return swatch

    def _set_swatch_color(self, swatch, color):
        radius = SWATCH_SIZE // 2
        swatch.setStyleSheet(f"""
            QToolButton {{
                background-color: {color};
                border: 2px solid {COLORS['background']};
                border-radius: {radius}px;
            }}
            QToolButton:hover {{ border-color: {COLORS['border_light']}; }}
            QToolButton:checked {{ border-color: {COLORS['text']}; }}
            QToolButton:disabled {{ background-color: {COLORS['border']}; }}
        """)

    def _select_accent(self, color):
        color = QColor(color).name()
        self._accent = color
        swatch = self.preset_swatches.get(color)
        if swatch is None:
            self._custom_accent = color
            self._set_swatch_color(self.custom_swatch, color)
            self.custom_swatch.setToolTip(f"Custom ({color})")
            self.custom_swatch.show()
            swatch = self.custom_swatch
        swatch.setChecked(True)
        self._schedule_preview()

    def _choose_custom_accent(self):
        color = QColorDialog.getColor(QColor(self._accent), self, "Choose Accent Color")
        if color.isValid():
            self._select_accent(color.name())

    def _selected_mode(self):
        for mode, button in self.mode_buttons.items():
            if button.isChecked():
                return mode
        return ThemeSettings().mode

    def _on_mode_toggled(self, _button, checked):
        if not checked:
            return
        night = self._selected_mode() == 'night'
        for swatch in self.accent_button_group.buttons():
            swatch.setEnabled(not night)
        self.custom_accent_btn.setEnabled(not night)
        self.night_accent_note.setVisible(night)
        self._schedule_preview()

    def _set_font_family(self, family):
        self.font_combo.setCurrentFont(QFont(family or platform_font().family()))
        self._shown_family = self.font_combo.currentFont().family()
        self._shown_family_value = family

    # ----------------------------------------------------------------
    # Loading and reading settings
    # ----------------------------------------------------------------
    def load(self, theme_settings):
        """Show the given settings in the controls and preview."""
        self.mode_buttons.get(theme_settings.mode, self.mode_buttons['dark']).setChecked(True)
        self._select_accent(theme_settings.accent)
        self._set_font_family(theme_settings.font_family)
        default_size = platform_font().pointSize()
        size = theme_settings.font_size or default_size
        self.font_size_spin.setValue(min(max(size, FONT_SIZE_MIN), FONT_SIZE_MAX))
        self.bold_headings_checkbox.setChecked(theme_settings.bold_headings)
        self._update_preview()

    def theme_settings(self):
        """The ThemeSettings chosen in the controls."""
        family = self.font_combo.currentFont().family()
        if family == self._shown_family:
            font_family = self._shown_family_value
        else:
            font_family = '' if family == platform_font().family() else family
        size = self.font_size_spin.value()
        return ThemeSettings(
            mode=self._selected_mode(),
            accent=self._accent,
            font_family=font_family,
            font_size=0 if size == platform_font().pointSize() else size,
            bold_headings=self.bold_headings_checkbox.isChecked(),
        )

    # ----------------------------------------------------------------
    # Preview
    # ----------------------------------------------------------------
    def _build_preview(self):
        group = QGroupBox("Preview")
        group_layout = QVBoxLayout(group)

        # The preview is styled through this scroll area's style sheet, which
        # reaches every sample widget but nothing outside it.
        self.preview = QScrollArea()
        self.preview.setWidgetResizable(True)
        self.preview.setFrameShape(QFrame.NoFrame)
        group_layout.addWidget(self.preview)

        page = QWidget()
        page.setObjectName("previewPage")
        page_layout = QVBoxLayout(page)
        self.preview.setWidget(page)

        session = QGroupBox("Tonight's Session")
        session_layout = QVBoxLayout(session)
        session_layout.addWidget(QLabel("M31 - Andromeda Galaxy"))
        secondary = QLabel("Galaxy in Andromeda, magnitude 3.4")
        secondary.setObjectName("previewSecondary")
        session_layout.addWidget(secondary)
        help_text = QLabel("Help text looks like this.")
        help_text.setObjectName("previewHelp")
        session_layout.addWidget(help_text)

        inputs_layout = QHBoxLayout()
        target_edit = QLineEdit("M31")
        filter_combo = QComboBox()
        filter_combo.addItems(["Luminance", "Red", "Green", "Blue", "H-alpha"])
        exposure_spin = QSpinBox()
        exposure_spin.setRange(1, 1800)
        exposure_spin.setValue(300)
        exposure_spin.setSuffix(" s")
        inputs_layout.addWidget(target_edit, 1)
        inputs_layout.addWidget(filter_combo, 1)
        inputs_layout.addWidget(exposure_spin)
        session_layout.addLayout(inputs_layout)

        options_layout = QHBoxLayout()
        dither = QCheckBox("Dither")
        dither.setChecked(True)
        options_layout.addWidget(dither)
        options_layout.addWidget(QCheckBox("Meridian flip"))
        autofocus = QRadioButton("Autofocus")
        autofocus.setChecked(True)
        options_layout.addWidget(autofocus)
        options_layout.addStretch()
        session_layout.addLayout(options_layout)

        slider = QSlider(Qt.Horizontal)
        slider.setValue(60)
        session_layout.addWidget(slider)
        progress = QProgressBar()
        progress.setValue(65)
        session_layout.addWidget(progress)

        buttons_layout = QHBoxLayout()
        start_btn = QPushButton("Start")
        start_btn.setDefault(True)
        park_btn = QPushButton("Park")
        park_btn.setEnabled(False)
        buttons_layout.addWidget(start_btn)
        buttons_layout.addWidget(QPushButton("Cancel"))
        buttons_layout.addWidget(park_btn)
        buttons_layout.addStretch()
        session_layout.addLayout(buttons_layout)
        page_layout.addWidget(session)

        table = QTableWidget(3, 3)
        table.setHorizontalHeaderLabels(["Target", "Filter", "Frames"])
        for row, values in enumerate([("M31", "L", "40"), ("M42", "Ha", "25"), ("NGC 7000", "OIII", "18")]):
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.selectRow(1)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.preview_table = table
        page_layout.addWidget(table)

        status_layout = QHBoxLayout()
        for name, text in [("Success", "Connected"), ("Warning", "Clouds"),
                           ("Error", "Disconnected"), ("Info", "Info")]:
            label = QLabel(f"● {text}")
            label.setObjectName(f"preview{name}")
            status_layout.addWidget(label)
        status_layout.addStretch()
        page_layout.addLayout(status_layout)
        page_layout.addStretch()
        return group

    def _schedule_preview(self, *_args):
        self._preview_timer.start()

    def _update_preview(self):
        self._preview_timer.stop()
        theme_settings = self.theme_settings()
        colors = build_colors(theme_settings)
        self.preview.setStyleSheet(
            build_stylesheet(colors, theme_settings)
            + self._preview_base_rules(colors, build_font(theme_settings))
        )
        # Fit the table to its rows so it doesn't scroll at larger font sizes.
        table = self.preview_table
        table.resizeRowsToContents()
        table.setFixedHeight(table.horizontalHeader().sizeHint().height()
                             + sum(table.rowHeight(r) for r in range(table.rowCount()))
                             + 2 * table.frameWidth())

    @staticmethod
    def _preview_base_rules(colors, font):
        """Background, text color and font for the preview, plus the sample
        labels that other windows color from COLORS.

        While an application style sheet is set, Qt doesn't propagate a
        widget's palette or font to its children, so the preview can't use
        build_palette()/build_font() directly and states them here instead.
        """
        family = font.family().replace('"', '')
        return f"""
QWidget {{
    color: {colors['text']};
    font-family: "{family}";
    font-size: {font.pointSize()}pt;
}}
#previewPage {{ background-color: {colors['background']}; }}
#previewSecondary {{ color: {colors['text_secondary']}; }}
#previewHelp {{ color: {colors['text_disabled']}; }}
#previewSuccess {{ color: {colors['success']}; }}
#previewWarning {{ color: {colors['warning']}; }}
#previewError {{ color: {colors['error']}; }}
#previewInfo {{ color: {colors['info']}; }}
"""
