import logging
import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow

# Logging konfigurieren – ball_detector gibt diagnostische Infos aus
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
