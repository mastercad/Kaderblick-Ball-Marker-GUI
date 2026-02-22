
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent, QFont
from PySide6.QtCore import Qt

class TimelineWidget(QWidget):
    def __init__(self, session, left_panel, right_panel):
        super().__init__()
        self.session = session
        self.left_panel = left_panel
        self.right_panel = right_panel
        self.setMinimumHeight(80)
        self.selected_frame = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(20, self.height() // 2, self.width() - 20, self.height() // 2)
        # Korrekt: Panel hat total_frames(), nicht QMediaPlayer
        if hasattr(self.left_panel, 'total_frames') and callable(getattr(self.left_panel, 'total_frames')):
            total_frames = max(1, self.left_panel.total_frames())
        else:
            total_frames = 1
        # Marker als Punkte
        for marker in self.session.markers:
            x = 20 + (self.width() - 40) * (marker.frame_index / total_frames)
            color = QColor(0, 180, 0) if marker.type == "manual" else QColor(220, 60, 60)
            painter.setPen(QPen(color, 8))
            painter.drawPoint(int(x), self.height() // 2)
        # Aktueller Frame
        painter.setPen(QPen(QColor(60, 120, 220), 2))
        x = 20 + (self.width() - 40) * (self.selected_frame / total_frames)
        painter.drawLine(int(x), self.height() // 2 - 20, int(x), self.height() // 2 + 20)
        painter.setFont(QFont("Arial", 10))
        painter.drawText(10, self.height() - 10, f"Frame: {self.selected_frame}")

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if hasattr(self.left_panel, 'total_frames') and callable(getattr(self.left_panel, 'total_frames')):
                total_frames = max(1, self.left_panel.total_frames())
            else:
                total_frames = 1
            frame = int((event.x() - 20) / (self.width() - 40) * total_frames)
            self.selected_frame = max(0, min(frame, total_frames - 1))
            if hasattr(self.left_panel, 'set_frame'):
                self.left_panel.set_frame(self.selected_frame)
            if hasattr(self.right_panel, 'set_frame'):
                self.right_panel.set_frame(self.selected_frame)
            self.update()
