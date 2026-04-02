"""Wiederverwendbares Kaderblick-Theme für Qt-Anwendungen.

Unterstuetzte Bindings:
- PyQt5
- PyQt6
- PySide6

Das Modul setzt Palette und Stylesheet zentral füur die gesamte Anwendung
und stellt optional einen einfachen Brand-Header bereit.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

def _import_qt_modules():
    for binding in ("PyQt5", "PyQt6", "PySide6"):
        try:
            qt_core = importlib.import_module(f"{binding}.QtCore")
            qt_gui = importlib.import_module(f"{binding}.QtGui")
            qt_widgets = importlib.import_module(f"{binding}.QtWidgets")
            return qt_core, qt_gui, qt_widgets
        except ImportError:
            continue
    raise ImportError("Es wurde kein unterstuetztes Qt-Binding gefunden.")


QtCore, QtGui, QtWidgets = _import_qt_modules()

QSize = QtCore.QSize
Qt = QtCore.Qt
QColor = QtGui.QColor
QFont = QtGui.QFont
QFontDatabase = QtGui.QFontDatabase
QFontMetrics = QtGui.QFontMetrics
QPainter = QtGui.QPainter
QPalette = QtGui.QPalette
QApplication = QtWidgets.QApplication
QFrame = cast(type[Any], QtWidgets.QFrame)
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QPushButton = QtWidgets.QPushButton
QSizePolicy = QtWidgets.QSizePolicy
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = cast(type[Any], QtWidgets.QWidget)


PRIMARY_GREEN = "#06B62E"
PRIMARY_GREEN_DARK = "#059B28"
PRIMARY_GREEN_SOFT = "#E9F8EC"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F7FAF7"
APP_BG = "#F3F5F2"
BORDER = "#D6E3D6"
TEXT = "#18212B"
MUTED = "#667582"
BRAND_HEADER_TEXT = "rgba(255, 255, 255, 0.92)"
UI_FONT_CANDIDATES = ["Roboto Flex", "Inter", "Montserrat", "Helvetica Neue", "Arial"]
BRAND_FONT_FALLBACKS = ["Impact", "Arial Black", "Sans Serif"]
THEME_APPLIED_PROPERTY = "_kaderblickThemeApplied"
BRAND_K_SIZE_DELTA = 1.7

_loaded_brand_family: str | None = None


def _palette_group(name: str):
    group_enum = getattr(QPalette, "ColorGroup", None)
    return getattr(group_enum, name) if group_enum is not None else getattr(QPalette, name)


def _palette_role(name: str):
    role_enum = getattr(QPalette, "ColorRole", None)
    return getattr(role_enum, name) if role_enum is not None else getattr(QPalette, name)


def _available_font_families() -> set[str]:
    database = QFontDatabase()
    return set(database.families())


def _default_ui_font() -> QFont:
    available = _available_font_families()
    fallback_family = QApplication.font().family()
    family = next((candidate for candidate in UI_FONT_CANDIDATES if candidate in available), fallback_family)
    font = QFont(family)
    font.setPointSize(10)
    return font


def _font_weight_bold():
    weight_enum = getattr(QFont, "Weight", None)
    return getattr(weight_enum, "Bold") if weight_enum is not None else QFont.Bold


def _spacing_type_absolute():
    spacing_enum = getattr(QFont, "SpacingType", None)
    return getattr(spacing_enum, "AbsoluteSpacing") if spacing_enum is not None else QFont.AbsoluteSpacing


def _text_antialiasing_hint():
    render_enum = getattr(QPainter, "RenderHint", None)
    return getattr(render_enum, "TextAntialiasing") if render_enum is not None else QPainter.TextAntialiasing


def _antialiasing_hint():
    render_enum = getattr(QPainter, "RenderHint", None)
    return getattr(render_enum, "Antialiasing") if render_enum is not None else QPainter.Antialiasing


def brand_wordmark_font() -> QFont:
    family = _loaded_brand_family or next(iter(BRAND_FONT_FALLBACKS), "Sans Serif")
    font = QFont(family)
    font.setPointSize(22)
    font.setWeight(_font_weight_bold())
    font.setLetterSpacing(_spacing_type_absolute(), 0.6)
    return font


def _ensure_brand_font_loaded() -> None:
    global _loaded_brand_family
    if _loaded_brand_family is not None:
        return

    font_path = Path(__file__).resolve().parent.parent / "assets" / "ImpactLTStd.woff2"
    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _loaded_brand_family = families[0]
                return

    available = _available_font_families()
    for fallback in BRAND_FONT_FALLBACKS:
        if fallback in available:
            _loaded_brand_family = fallback
            return
    _loaded_brand_family = QApplication.font().family()


class BrandWordmarkWidget(QWidget):
    def __init__(self, text: str = "KADERBLICK", parent: Any = None):
        super().__init__(parent)
        self._text = text
        self._first = text[:1]
        self._rest = text[1:]
        self.setObjectName("brandWordmark")
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        except AttributeError:
            self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFont(brand_wordmark_font())
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        first_font = QFont(self.font())
        first_font.setPointSizeF(first_font.pointSizeF() + BRAND_K_SIZE_DELTA)
        first_metrics = QFontMetrics(first_font)
        rest_metrics = QFontMetrics(self.font())
        width = first_metrics.horizontalAdvance(self._first) + rest_metrics.horizontalAdvance(self._rest)
        height = max(first_metrics.height(), rest_metrics.height())
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, _event) -> None:
        first_font = QFont(self.font())
        first_font.setPointSizeF(first_font.pointSizeF() + BRAND_K_SIZE_DELTA)
        first_metrics = QFontMetrics(first_font)
        rest_metrics = QFontMetrics(self.font())

        painter = QPainter(self)
        painter.setRenderHint(_text_antialiasing_hint(), True)
        painter.setRenderHint(_antialiasing_hint(), True)

        top = (self.height() - max(first_metrics.height(), rest_metrics.height())) // 2
        first_baseline_y = top + first_metrics.ascent()
        rest_baseline_y = top + rest_metrics.ascent()

        painter.setFont(first_font)
        painter.setPen(QColor("#018707"))
        painter.drawText(0, first_baseline_y, self._first)

        painter.setFont(self.font())
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(first_metrics.horizontalAdvance(self._first), rest_baseline_y, self._rest)
        painter.end()


def build_palette(seed: QPalette | None = None) -> QPalette:
    palette = QPalette(seed or QPalette())
    text = QColor(TEXT)
    surface = QColor(SURFACE)
    surface_alt = QColor(SURFACE_ALT)
    app_bg = QColor(APP_BG)
    selection = QColor("#D8F2DE")
    muted = QColor(MUTED)

    for group_name in ("Active", "Inactive", "Disabled"):
        group = _palette_group(group_name)
        palette.setColor(group, _palette_role("WindowText"), text)
        palette.setColor(group, _palette_role("Text"), text)
        palette.setColor(group, _palette_role("ButtonText"), text)
        palette.setColor(group, _palette_role("Base"), surface)
        palette.setColor(group, _palette_role("AlternateBase"), surface_alt)
        palette.setColor(group, _palette_role("Window"), app_bg)
        palette.setColor(group, _palette_role("Button"), surface)
        palette.setColor(group, _palette_role("Highlight"), selection)
        palette.setColor(group, _palette_role("HighlightedText"), text)
        palette.setColor(group, _palette_role("PlaceholderText"), muted)

    return palette


def build_stylesheet() -> str:
    return f"""
    QWidget {{
        background: {APP_BG};
        color: {TEXT};
    }}

    QMainWindow, QDialog {{
        background: {APP_BG};
    }}

    QLabel {{
        background: transparent;
    }}

    QMenuBar {{
        background: {PRIMARY_GREEN};
        color: white;
        border: none;
        padding: 4px 12px;
        font-weight: 600;
    }}

    QMenuBar::item {{
        background: transparent;
        padding: 8px 12px;
        margin: 2px 4px;
        border-radius: 14px;
    }}

    QMenuBar::item:selected {{
        background: rgba(255, 255, 255, 0.16);
    }}

    QMenu {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 6px;
    }}

    QMenu::item {{
        padding: 8px 12px;
        border-radius: 8px;
    }}

    QMenu::item:selected {{
        background: {PRIMARY_GREEN_SOFT};
        color: {PRIMARY_GREEN_DARK};
    }}

    QToolBar {{
        background: {PRIMARY_GREEN};
        border: none;
        spacing: 6px;
        padding: 10px 14px;
    }}

    QToolBar::separator {{
        background: rgba(255, 255, 255, 0.18);
        width: 1px;
        margin: 6px 10px;
    }}

    QToolBar QToolButton {{
        background: transparent;
        color: white;
        border: 1px solid transparent;
        border-radius: 16px;
        padding: 8px 12px;
        font-weight: 600;
    }}

    QToolBar QToolButton:hover {{
        background: rgba(255, 255, 255, 0.14);
    }}

    QToolBar QToolButton:pressed {{
        background: rgba(0, 0, 0, 0.12);
    }}

    QToolBar QToolButton#qt_toolbar_ext_button {{
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.28);
        min-width: 28px;
        padding: 8px;
    }}

    QToolBar QToolButton#qt_toolbar_ext_button:hover {{
        background: rgba(255, 255, 255, 0.28);
    }}

    QSplitter::handle {{
        background: transparent;
    }}

    QGroupBox,
    QTableWidget,
    QTextEdit,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QTimeEdit,
    QAbstractSpinBox,
    QProgressBar {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
    }}

    QGroupBox {{
        margin-top: 10px;
        padding: 14px 12px 12px 12px;
        font-weight: 700;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {TEXT};
    }}

    QLineEdit,
    QComboBox,
    QSpinBox,
    QTimeEdit,
    QAbstractSpinBox,
    QTextEdit {{
        padding: 8px 10px;
        selection-background-color: #D8F2DE;
        selection-color: {TEXT};
    }}

    QTableWidget {{
        alternate-background-color: {SURFACE_ALT};
        gridline-color: #E8EFE8;
        selection-background-color: #D8F2DE;
        selection-color: {TEXT};
        padding: 4px;
    }}

    QHeaderView::section {{
        background: #EFF7F0;
        color: {TEXT};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 10px 12px;
        font-weight: 700;
    }}

    QPushButton {{
        background: {PRIMARY_GREEN};
        color: white;
        border: none;
        border-radius: 12px;
        padding: 8px 14px;
        font-weight: 700;
    }}

    QPushButton:hover {{
        background: {PRIMARY_GREEN_DARK};
    }}

    QPushButton:disabled {{
        background: #B9C9BA;
        color: #F3F7F3;
    }}

    QPushButton#primaryActionButton {{
        min-height: 40px;
        font-size: 14px;
        border-radius: 14px;
        padding: 10px 18px;
    }}

    QProgressBar {{
        background: #ECF2ED;
        min-height: 20px;
        text-align: center;
        font-weight: 600;
    }}

    QProgressBar::chunk {{
        background: {PRIMARY_GREEN};
        border-radius: 10px;
    }}

    QCheckBox {{
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 8px;
        border: 1px solid #9FB69F;
        background: {SURFACE};
    }}

    QCheckBox::indicator:checked {{
        background: {PRIMARY_GREEN};
        border-color: {PRIMARY_GREEN_DARK};
    }}

    QLabel#mutedText,
    QLabel#statusMetaLabel {{
        color: {MUTED};
    }}

    QLabel#summaryLabel {{
        font-weight: 700;
        padding: 4px 2px;
    }}

    QLabel#logoPreview {{
        background: #EEF4EE;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 6px;
    }}

    QFrame#brandBanner[tone="brand"] {{
        background: {PRIMARY_GREEN};
        border: none;
        border-radius: 0;
    }}

    QFrame#brandBanner[tone="surface"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 18px;
    }}

    QFrame#brandBanner QFrame#brandBannerSeparator {{
        background: rgba(255, 255, 255, 0.18);
        min-width: 1px;
        max-width: 1px;
    }}

    QFrame#brandBanner QPushButton[variant="banner"] {{
        background: transparent;
        color: white;
        border: 1px solid transparent;
        border-radius: 16px;
        padding: 8px 12px;
        font-weight: 600;
    }}

    QFrame#brandBanner QPushButton[variant="banner"]:hover {{
        background: rgba(255, 255, 255, 0.14);
    }}

    QFrame#brandBanner QPushButton[variant="banner"]:pressed {{
        background: rgba(0, 0, 0, 0.12);
    }}

    QFrame#brandBanner QPushButton[variant="banner-primary"] {{
        background: rgba(255, 255, 255, 0.14);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 8px 12px;
        font-weight: 700;
    }}

    QFrame#brandBanner QPushButton[variant="banner-primary"]:hover {{
        background: rgba(255, 255, 255, 0.24);
    }}

    QWidget#brandWordmark {{
        background: transparent;
    }}

    QLabel#brandTagline[tone="brand"] {{
        color: {BRAND_HEADER_TEXT};
        background: transparent;
        font-size: 11px;
        font-weight: 500;
    }}

    QLabel#brandTagline[tone="surface"] {{
        color: {MUTED};
        background: transparent;
        font-size: 11px;
        font-weight: 500;
    }}
    """


def apply_application_theme(app: QApplication | None = None) -> None:
    app = app or QApplication.instance()
    if app is None:
        return

    _ensure_brand_font_loaded()
    if not bool(app.property(THEME_APPLIED_PROPERTY)):
        app.setStyle("Fusion")
        app.setFont(_default_ui_font())
        app.setPalette(build_palette(app.palette()))
        app.setStyleSheet(build_stylesheet())
        app.setProperty(THEME_APPLIED_PROPERTY, True)


class BrandHeaderWidget(QFrame):
    def __init__(
        self,
        title: str = "KADERBLICK",
        subtitle: str = "Video Combiner",
        parent: Any = None,
        tone: str = "brand",
    ):
        super().__init__(parent)
        self.setObjectName("brandBanner")
        self.setProperty("tone", tone)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(64)
        self.setMaximumHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        self._actions_layout = QHBoxLayout()
        self._actions_layout.setContentsMargins(0, 0, 0, 0)
        self._actions_layout.setSpacing(6)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)

        title_widget = BrandWordmarkWidget(title)
        text_layout.addWidget(title_widget)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("brandTagline")
        subtitle_label.setProperty("tone", tone)
        subtitle_label.setVisible(bool(subtitle.strip()))
        text_layout.addWidget(subtitle_label)

        layout.addLayout(text_layout)
        layout.addStretch(1)
        layout.addLayout(self._actions_layout)

    def add_action(self, text: str, callback, *, primary: bool = False):
        button = QPushButton(text)
        button.setProperty("variant", "banner-primary" if primary else "banner")
        button.clicked.connect(callback)
        self._actions_layout.addWidget(button)
        return button

    def add_separator(self):
        separator = QFrame()
        separator.setObjectName("brandBannerSeparator")
        separator.setFrameShape(QFrame.VLine)
        self._actions_layout.addWidget(separator)
        return separator

    def sizeHint(self) -> QSize:
        return QSize(960, 64)

    def minimumSizeHint(self) -> QSize:
        return QSize(480, 64)