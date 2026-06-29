import logging
import sys

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from shared.app_paths import resource_path
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
    assets_dir = resource_path("assets")
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        png = assets_dir / f"icon_{size}.png"
        if png.is_file():
            icon.addFile(str(png), QSize(size, size))
    svg = assets_dir / "icon.svg"
    if svg.is_file():
        icon.addFile(str(svg))
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
