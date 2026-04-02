"""
Zoombare, pannbare QGraphicsView für die Kalibrierung.

Verarbeitet:
  - Mausrad → Zoom
  - Mittlere Taste → Pan
  - Linksklick (verzögert) → neuer Punkt
  - Doppelklick → Punkt auf Linie einfügen
  - Rechtsklick auf Punkt → entfernen
"""

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QTransform
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from calibration.drag_point import DragPoint


class CalibrationView(QGraphicsView):
    """Zoombare, pannbare GraphicsView für die Kalibrierung."""

    point_clicked = Signal(float, float)            # Klick-Position in Bildkoordinaten
    point_remove_requested = Signal(int)              # Index des zu löschenden Punkts
    line_insert_requested = Signal(float, float)      # Doppelklick-Position

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._panning = False
        self._pan_start = QPointF()
        self._zoom = 1.0
        self._active_mode: str = ""  # Aktueller Modus (nur dessen Punkte reagieren)

        # Timer um Einzelklick von Doppelklick zu unterscheiden
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(250)
        self._click_timer.timeout.connect(self._emit_pending_click)
        self._pending_click: QPointF | None = None

    # ── Zoom ─────────────────────────────────────────────────────

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(20.0, self._zoom))
        self.setTransform(QTransform.fromScale(self._zoom, self._zoom))
        event.accept()

    # ── Mouse-Events ─────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.pos())
            if isinstance(item, DragPoint) and item.mode == self._active_mode:
                self.point_remove_requested.emit(item.index)
                event.accept()
                return
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, DragPoint) and item.mode == self._active_mode:
                super().mousePressEvent(event)
                return
            # Klick verzögern, damit Doppelklick ihn abfangen kann
            scene_pos = self.mapToScene(event.pos())
            self._pending_click = scene_pos
            self._click_timer.start()
            event.accept()
            return

        super().mousePressEvent(event)

    def _emit_pending_click(self):
        """Feuert den verzögerten Einzelklick, falls kein Doppelklick kam."""
        if self._pending_click is not None:
            self.point_clicked.emit(self._pending_click.x(), self._pending_click.y())
            self._pending_click = None

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_timer.stop()
            self._pending_click = None
            item = self.itemAt(event.pos())
            if isinstance(item, DragPoint) and item.mode == self._active_mode:
                event.accept()
                return
            scene_pos = self.mapToScene(event.pos())
            self.line_insert_requested.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Hilfsmethoden ────────────────────────────────────────────

    def fit_image(self):
        """Setzt Zoom zurück und passt die Ansicht an die Szene an."""
        self._zoom = 1.0
        self.setTransform(QTransform())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
