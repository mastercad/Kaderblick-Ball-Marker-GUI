"""
Verschiebbarer Punkt (DragPoint) für die Kalibrierungs-Szene.

Ein QGraphicsEllipseItem, das per Maus gezogen werden kann
und seinen neuen Standort per Callback meldet.
"""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsTextItem,
)

from typing import Callable, Optional


class DragPoint(QGraphicsEllipseItem):
    """Ein verschiebbarer Punkt in der Kalibrierungsszene.

    Attributes:
        index: Laufende Nummer innerhalb des aktuellen Modus.
        mode:  Name des Modus, zu dem dieser Punkt gehört.
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        color: QColor,
        index: int,
        on_moved: Optional[Callable[[int, QPointF], None]] = None,
        parent=None,
        mode: str = "",
    ):
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setPos(x, y)
        self.setPen(QPen(Qt.GlobalColor.white, 2))
        self.setBrush(QBrush(color))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(20)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.index = index
        self._on_moved = on_moved
        self._radius = radius
        self.mode = mode

        # Nummer-Label
        self._label = QGraphicsTextItem(str(index + 1), self)
        self._label.setDefaultTextColor(color)
        font = self._label.font()
        font.setPointSize(max(8, int(radius)))
        font.setBold(True)
        self._label.setFont(font)
        self._label.setPos(radius + 2, -radius - 4)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self._on_moved:
            self._on_moved(self.index, value)
        return super().itemChange(change, value)
