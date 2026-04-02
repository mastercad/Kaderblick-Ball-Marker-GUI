import logging
import os
import sys

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from shared.kaderblick_qt_theme import apply_application_theme

# Logging konfigurieren – ball_detector gibt diagnostische Infos aus
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Kaderblick - BallMarker GUI")
    app.setDesktopFileName("kaderblick-ballmarker-gui")
    apply_application_theme(app)

    # App-Icon mit mehreren Auflösungen setzen
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        png = os.path.join(assets_dir, f"icon_{size}.png")
        if os.path.isfile(png):
            icon.addFile(png, QSize(size, size))
    svg = os.path.join(assets_dir, "icon.svg")
    if os.path.isfile(svg):
        icon.addFile(svg)
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
