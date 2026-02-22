from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor, QMouseEvent, QWheelEvent, QCursor
from PySide6.QtCore import Qt
from model.marker import Marker
import sys

class MarkerOverlay(QWidget):
    def __init__(self, session, player):
        super().__init__()
        self.session = session
        self.player = player
        self.selected_marker = None
        self.drag_start = None
        self.zoom = 1.0
        self.pan = [0, 0]
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show()
        self.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        print(f"[DEBUG] showEvent: Overlay sichtbar, Größe: {self.width()}x{self.height()}, Parent: {self.parent()} (Parent-Klasse: {self.parent().__class__.__name__})")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        if self.width() == 0 or self.height() == 0:
            print(f"[DEBUG] paintEvent: Overlaygröße ist 0, Marker können nicht gezeichnet werden.")
            return
        # Sichtbarkeit erzwingen: grellgelber, undurchsichtiger Hintergrund
        painter.setBrush(QColor(255, 255, 0, 255))  # Gelb, komplett undurchsichtig
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        # Sehr dicker Rahmen
        painter.setPen(QPen(QColor(255, 0, 0, 255), 12))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.rect())
        print(f"[DEBUG] paintEvent: GELBES OVERLAY gezeichnet (Overlaygröße: {self.rect().width()}x{self.rect().height()})")
        if not self.session.markers:
            print(f"[DEBUG] paintEvent: Keine Marker vorhanden.")
        # DEBUG: Testkreis in der Mitte
        center = self.rect().center()
        radius = min(self.rect().width(), self.rect().height()) // 8
        painter.setPen(QPen(QColor(255, 0, 0), 6))
        painter.setBrush(QColor(255, 0, 0, 255))
        painter.drawEllipse(center, radius, radius)
        print(f"[DEBUG] paintEvent: Testkreis in der Mitte gezeichnet (Overlaygröße: {self.rect().width()}x{self.rect().height()})")
        for marker in self.session.markers:
            color = QColor(0, 180, 0) if marker.type == "manual" else QColor(220, 60, 60)
            pen = QPen(color, 2)
            painter.setPen(pen)
            r = marker.radius * min(self.width(), self.height())
            x = marker.position[0] * self.width() * self.zoom + self.pan[0]
            y = marker.position[1] * self.height() * self.zoom + self.pan[1]
            painter.setBrush(QColor(0, 0, 0, 0))
            if r < 1:
                print(f"[DEBUG] paintEvent: Marker-Radius zu klein (r={r}), Marker nicht sichtbar.")
            painter.drawEllipse(int(x - r), int(y - r), int(2 * r), int(2 * r))
            print(f"[DEBUG] paintEvent: marker pos=({x},{y}), radius={r}, type={marker.type}")
            if marker.type == "interpolated":
                painter.setPen(QPen(QColor(220, 60, 60, 120), 2, Qt.PenStyle.DashLine))
                painter.drawEllipse(int(x - r), int(y - r), int(2 * r), int(2 * r))

    def mousePressEvent(self, event: QMouseEvent):
        pos = self._to_marker_space(event.x(), event.y())
        print(f"[DEBUG] mousePressEvent: pos={pos}, button={event.button()}, modifiers={event.modifiers()}, session markers={len(self.session.markers)}")
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.drag_start = [event.x(), event.y()]
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                print(f"[DEBUG] Pan start: drag_start={self.drag_start}")
                return
            for marker in self.session.markers:
                mx, my = self._to_marker_space(marker.position[0] * self.width(), marker.position[1] * self.height())
                mr = marker.radius * min(self.width(), self.height())
                if (pos[0] - mx) ** 2 + (pos[1] - my) ** 2 < mr ** 2:
                    self.selected_marker = marker
                    self.drag_start = pos
                    self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                    print(f"[DEBUG] Marker selected: marker={marker}, drag_start={self.drag_start}")
                    return
            frame = self.player.current_frame()
            timestamp = self.player.current_timestamp()
            video_file = self.player.video_file
            marker = Marker(video_file, frame, timestamp, (pos[0]/self.width(), pos[1]/self.height()), 0.05, "manual")
            self.session.add_marker(marker)
            self.update()
            print(f"[DEBUG] Marker gesetzt: video_file={video_file}, frame={frame}, timestamp={timestamp}, pos={pos}, marker={marker}")
        elif event.button() == Qt.MouseButton.RightButton:
            for marker in self.session.markers:
                mx, my = self._to_marker_space(marker.position[0] * self.width(), marker.position[1] * self.height())
                mr = marker.radius * min(self.width(), self.height())
                if (pos[0] - mx) ** 2 + (pos[1] - my) ** 2 < mr ** 2:
                    self.session.remove_marker(marker)
                    self.update()
                    print(f"[DEBUG] Marker gelöscht: marker={marker}")
                    return

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = self._to_marker_space(event.x(), event.y())
        if self.selected_marker and self.drag_start:
            dx = pos[0] - self.drag_start[0]
            dy = pos[1] - self.drag_start[1]
            new_x = self.selected_marker.position[0] + dx/self.width()
            new_y = self.selected_marker.position[1] + dy/self.height()
            self.session.move_marker(self.selected_marker, (new_x, new_y))
            self.drag_start = pos
            self.update()
            print(f"[DEBUG] Marker verschoben: marker={self.selected_marker}, new_pos=({new_x},{new_y})")
        elif self.drag_start and event.buttons() & Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Pan bei Zoom
            self.pan[0] += event.x() - self.drag_start[0]
            self.pan[1] += event.y() - self.drag_start[1]
            self.drag_start = [event.x(), event.y()]
            self.update()
            print(f"[DEBUG] Pan: pan={self.pan}, drag_start={self.drag_start}")
        else:
            hovered = False
            for marker in self.session.markers:
                mx, my = self._to_marker_space(marker.position[0] * self.width(), marker.position[1] * self.height())
                mr = marker.radius * min(self.width(), self.height())
                if (pos[0] - mx) ** 2 + (pos[1] - my) ** 2 < mr ** 2:
                    self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    hovered = True
                    break
            if not hovered:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.selected_marker = None
        self.drag_start = None
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def wheelEvent(self, event: QWheelEvent):
        pos = event.position()
        x, y = pos.x(), pos.y()
        marker_pos = self._to_marker_space(x, y)
        print(f"[DEBUG] wheelEvent: pos={marker_pos}, angleDelta={event.angleDelta().y()}, modifiers={event.modifiers()}, zoom={self.zoom}")
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Zoom
            old_zoom = self.zoom
            self.zoom += event.angleDelta().y() / 1200.0
            self.zoom = max(0.5, min(3.0, self.zoom))
            self.update()
            print(f"[DEBUG] Zoom: old_zoom={old_zoom}, new_zoom={self.zoom}")
        elif self.selected_marker:
            # Markergröße ändern
            delta = event.angleDelta().y() / 1200.0
            new_radius = max(0.01, min(0.2, self.selected_marker.radius + delta))
            self.session.resize_marker(self.selected_marker, new_radius)
            self.update()
            print(f"[DEBUG] Markergröße geändert: marker={self.selected_marker}, new_radius={new_radius}")

    def _to_marker_space(self, x, y):
        # Berücksichtigt Zoom und Pan
        return ((x - self.pan[0]) / self.zoom, (y - self.pan[1]) / self.zoom)
