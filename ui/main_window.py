
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QMenuBar, QFileDialog, QMessageBox
from PySide6.QtGui import QAction
from ui.video_panel import VideoPanel
from ui.timeline_widget import TimelineWidget
from model.session import Session
from autosave.autosave import Autosave
from export.exporter import export_markers

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ballmarker")
        self.session = Session()
        self.autosave = Autosave(self.session)
        from ui.video_graphics_panel import VideoGraphicsPanel
        self.left_panel = VideoGraphicsPanel(self.session)
        self.right_panel = VideoGraphicsPanel(self.session)
        self.timeline = TimelineWidget(self.session, self.left_panel, self.right_panel)
        layout = QVBoxLayout()
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.left_panel)
        video_layout.addWidget(self.right_panel)
        layout.addLayout(video_layout)
        layout.addWidget(self.timeline)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self._setup_menu()
        self.autosave.start()

    def _setup_menu(self):
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("Datei")
        export_action = QAction("Exportieren", self)
        export_action.triggered.connect(self.export)
        file_menu.addAction(export_action)
        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.undo)
        file_menu.addAction(undo_action)
        redo_action = QAction("Redo", self)
        redo_action.triggered.connect(self.redo)
        file_menu.addAction(redo_action)
        self.setMenuBar(menubar)

    def export(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Exportieren", "", "JSON Files (*.json)")
        if filename:
            export_markers(self.session.markers, filename)
            QMessageBox.information(self, "Export", f"Export erfolgreich: {filename}")

    def undo(self):
        self.session.undo()
        self.left_panel.view.viewport().update()
        self.right_panel.view.viewport().update()
        self.timeline.update()

    def redo(self):
        self.session.redo()
        self.left_panel.view.viewport().update()
        self.right_panel.view.viewport().update()
        self.timeline.update()
