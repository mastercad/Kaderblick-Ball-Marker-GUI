"""
Szenen-Renderer für die Kalibrierung.

Verantwortlich für das Zeichnen aller Punkte, Verbindungslinien,
Polygone und Ellipsen auf der QGraphicsScene.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
)

from calibration.calibration_modes import CLOSED_MODES, COLORS, FLAG_MODES, ModeSpec
from calibration.drag_point import DragPoint
from calibration.field_calibration import FieldCalibrationData


class SceneRenderer:
    """Zeichnet Kalibrierungspunkte und Verbindungslinien in eine QGraphicsScene.

    Hält keine eigene Kopie der Kalibrierungsdaten – greift per Callback
    auf die aktuelle Punktliste zu (``get_points``).
    """

    def __init__(
        self,
        scene: QGraphicsScene,
        get_points: Callable[[str], List[List[int]]],
        on_point_moved: Callable[[str, int, QPointF], None],
        point_radius: float = 10.0,
    ):
        self._scene = scene
        self._get_points = get_points
        self._on_point_moved = on_point_moved
        self.point_radius = point_radius

        # Verwaltung gezeichneter Elemente
        self._point_items: Dict[str, List[DragPoint]] = {}
        self._line_items: List[QGraphicsItem] = []

    # ── Öffentliche API ──────────────────────────────────────────

    def redraw_all(self, modes: List[ModeSpec], current_mode_name: str, data: FieldCalibrationData):
        """Zeichnet alle Punkte und Linien komplett neu."""
        self._clear_points()
        self._clear_lines()

        for mode_name, _, _, _ in modes:
            pts = self._get_points(mode_name)
            if not pts:
                continue

            color = COLORS.get(mode_name, QColor(200, 200, 200, 200))
            is_active = (mode_name == current_mode_name)

            point_items: List[DragPoint] = []
            for i, pt in enumerate(pts):
                point_color = COLORS["active_point"] if is_active else color
                item = DragPoint(
                    pt[0], pt[1], self.point_radius, point_color, i,
                    on_moved=lambda idx, pos, m=mode_name: self._on_point_moved(m, idx, pos),
                    mode=mode_name,
                )
                if not is_active:
                    item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                    item.setCursor(Qt.CursorShape.ArrowCursor)
                    item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(item)
                point_items.append(item)
            self._point_items[mode_name] = point_items

        self._redraw_lines(modes, data)

    def redraw_lines(self, modes: List[ModeSpec], data: FieldCalibrationData):
        """Zeichnet nur die Verbindungslinien/Polygone neu (z. B. nach Drag)."""
        self._redraw_lines(modes, data)

    # ── Interne Zeichenmethoden ─────────────────────────────────

    def _clear_points(self):
        for items in self._point_items.values():
            for item in items:
                self._scene.removeItem(item)
        self._point_items.clear()

    def _clear_lines(self):
        for item in self._line_items:
            self._scene.removeItem(item)
        self._line_items.clear()

    def _redraw_lines(self, modes: List[ModeSpec], data: FieldCalibrationData):
        self._clear_lines()
        pen_width = max(2, int(self.point_radius / 3))

        for mode_name, _, _, _ in modes:
            pts = self._get_points(mode_name)
            if len(pts) < 2:
                continue

            if mode_name in FLAG_MODES:
                continue

            color = COLORS.get(mode_name, QColor(200, 200, 200, 200))
            pen = QPen(color, pen_width)
            pen.setCosmetic(True)

            is_closed = mode_name in CLOSED_MODES

            if is_closed:
                polygon = QPolygonF([QPointF(p[0], p[1]) for p in pts])
                item = QGraphicsPolygonItem(polygon)
                item.setPen(pen)
                if len(pts) >= 3:
                    fill = QColor(color)
                    fill.setAlpha(30)
                    item.setBrush(QBrush(fill))
                else:
                    item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            else:
                path = QPainterPath()
                path.moveTo(pts[0][0], pts[0][1])
                for p in pts[1:]:
                    path.lineTo(p[0], p[1])
                item = QGraphicsPathItem(path)
                item.setPen(pen)

            item.setZValue(10)
            self._scene.addItem(item)
            self._line_items.append(item)

        # Ellipse für center_ellipse
        self._draw_ellipse(data)

    def _draw_ellipse(self, data: FieldCalibrationData):
        """Zeichnet die Mittelkreis-Ellipse wenn alle 3 Punkte vorhanden sind."""
        if not data.center_circle_center:
            return
        center = data.center_circle_center
        color = COLORS["center_ellipse"]

        if data.center_circle_horizontal and data.center_circle_vertical:
            h_pt = data.center_circle_horizontal
            v_pt = data.center_circle_vertical
            axis_h = int(np.sqrt((h_pt[0] - center[0]) ** 2 + (h_pt[1] - center[1]) ** 2))
            axis_v = int(np.sqrt((v_pt[0] - center[0]) ** 2 + (v_pt[1] - center[1]) ** 2))

            ellipse = QGraphicsEllipseItem(
                center[0] - axis_h, center[1] - axis_v,
                axis_h * 2, axis_v * 2,
            )
            pen = QPen(color, max(2, int(self.point_radius / 3)))
            pen.setCosmetic(True)
            ellipse.setPen(pen)
            ellipse.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ellipse.setZValue(10)
            self._scene.addItem(ellipse)
            self._line_items.append(ellipse)
