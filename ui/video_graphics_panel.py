from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsItem, QVBoxLayout, QWidget, QPushButton
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtCore import Qt, QRectF, QEvent
from PySide6.QtGui import QPainter, QMouseEvent, QColor

class VideoGraphicsPanel(QWidget):
    def wheelEvent(self, event):
        pos = self.view.mapToScene(event.position().toPoint())
        for marker, item in self.marker_items.items():
            if item.isUnderMouse():
                delta = event.angleDelta().y() / 1200.0
                video_rect = self.video_item.boundingRect()
                r = marker.radius * min(video_rect.width(), video_rect.height())
                new_r = max(5, min(50, r + delta * 10))
                item.setRect(item.rect().center().x()-new_r/2, item.rect().center().y()-new_r/2, new_r, new_r)
                marker.radius = new_r / min(video_rect.width(), video_rect.height())
                if hasattr(self.session, 'resize_marker'):
                    self.session.resize_marker(marker, marker.radius)
                break
        super().wheelEvent(event)

    def __init__(self, session):
        super(VideoGraphicsPanel, self).__init__()
        self.session = session
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.player.setVideoOutput(self.video_item)
        self.markers = []  # Marker-Objekte
        self.marker_items = {}  # Marker-Objekt -> QGraphicsEllipseItem
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        layout = QVBoxLayout()
        self.open_btn = QPushButton("Video öffnen")
        layout.addWidget(self.open_btn)
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.open_btn.clicked.connect(self.open_video)

    def set_frame(self, frame):
        # Setzt die Position im Video entsprechend dem Frame
        if self.player is not None:
            # Annahme: 30 fps
            self.player.setPosition(int(frame * 1000 / 30))


    def total_frames(self):
        # Dummy: Gibt 1 zurück, falls kein Video geladen
        if self.player is not None and self.player.duration() > 0:
            # Annahme: 30 fps
            return int(self.player.duration() / 1000 * 30)
        return 1

    def open_video(self):
        from PySide6.QtWidgets import QFileDialog
        from PySide6.QtCore import QUrl
        filename, _ = QFileDialog.getOpenFileName(self, "Video öffnen", "", "Video Files (*.mp4 *.avi *.mov)")
        if filename:
            self.player.setSource(QUrl.fromLocalFile(filename))
            self.video_item.setSize(self.view.size())
            self.player.setPosition(0)
            self.player.pause()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_item.setSize(self.view.size())

    def eventFilter(self, obj, event):
        from model.marker import Marker
        if obj is self.view.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.LeftButton:
                    pos = self.view.mapToScene(event.pos())
                    video_rect = self.video_item.boundingRect()
                    # Prüfe, ob auf einen Marker geklickt wurde
                    for m, item in self.marker_items.items():
                        if item.isUnderMouse():
                            self._dragged_marker = m
                            self._drag_offset = pos - item.rect().center()
                            self._prev_marker_pos = m.position
                            return True
                    # Sonst neuen Marker setzen
                    if video_rect.contains(pos):
                        frame = int(self.player.position() / 40)
                        timestamp = self.player.position()
                        video_file = self.player.source().toString() if hasattr(self.player, 'source') else ''
                        marker = Marker(video_file, frame, timestamp, (pos.x()/video_rect.width(), pos.y()/video_rect.height()), 0.05, "manual")
                        r = marker.radius * min(video_rect.width(), video_rect.height())
                        item = QGraphicsEllipseItem(QRectF(pos.x()-r, pos.y()-r, 2*r, 2*r))
                        item.setBrush(QColor(255,0,0,128))  # 128 = semi-transparent
                        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
                        self.scene.addItem(item)
                        self.markers.append(marker)
                        self.marker_items[marker] = item
                        if hasattr(self.session, 'add_marker'):
                            self.session.add_marker(marker)
                        self._dragged_marker = marker
                        self._drag_offset = pos - item.rect().center()
                        return True
            elif event.type() == QEvent.Type.MouseMove:
                if hasattr(self, '_dragged_marker') and self._dragged_marker:
                    pos = self.view.mapToScene(event.pos())
                    rect = self.marker_items[self._dragged_marker].rect()
                    new_center = pos - self._drag_offset
                    self.marker_items[self._dragged_marker].setRect(new_center.x()-rect.width()/2, new_center.y()-rect.height()/2, rect.width(), rect.height())
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if hasattr(self, '_dragged_marker') and self._dragged_marker:
                    # Undo/Redo für Marker verschieben
                    if hasattr(self.session, 'move_marker') and hasattr(self, '_prev_marker_pos'):
                        new_center = self.marker_items[self._dragged_marker].rect().center()
                        self.session.move_marker(self._dragged_marker, (new_center.x(), new_center.y()))
                        self._prev_marker_pos = None
                    self._dragged_marker = None
                    self._drag_offset = None
                    return True
        return False

