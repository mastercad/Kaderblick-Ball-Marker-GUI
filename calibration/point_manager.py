"""
Punkt-Verwaltung für die Kalibrierungsdaten.

CRUD-Operationen (Hinzufügen, Entfernen, Einfügen, Leeren, Verschieben)
für die verschiedenen Kalibrierungsmodi.  Arbeitet direkt auf einer
``FieldCalibrationData``-Instanz.
"""

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF

from calibration.calibration_modes import FLAG_MODES, CLOSED_MODES, ModeSpec
from calibration.field_calibration import FieldCalibrationData


class PointManager:
    """Verwaltet Kalibrierungspunkte für alle Modi.

    Args:
        data: Die zugrunde liegende Kalibrierungsdaten-Instanz.
    """

    def __init__(self, data: FieldCalibrationData):
        self._data = data

    @property
    def data(self) -> FieldCalibrationData:
        return self._data

    @data.setter
    def data(self, value: FieldCalibrationData):
        self._data = value

    # ── Punkt-Zugriff ────────────────────────────────────────────

    def points_for_mode(self, mode: str) -> List[List[int]]:
        """Gibt die Punktliste für den übergebenen Modus zurück (Live-Referenz)."""
        if mode == "field_boundary":
            return self._data.field_boundary
        elif mode == "center_line":
            return self._data.center_line
        elif mode == "center_ellipse":
            pts: List[List[int]] = []
            if self._data.center_circle_center:
                pts.append(self._data.center_circle_center)
            if self._data.center_circle_horizontal:
                pts.append(self._data.center_circle_horizontal)
            if self._data.center_circle_vertical:
                pts.append(self._data.center_circle_vertical)
            return pts
        elif mode == "center_half_ellipse":
            return self._data.center_half_ellipse_points
        elif mode == "penalty_left":
            return self._data.penalty_area_left
        elif mode == "penalty_right":
            return self._data.penalty_area_right
        elif mode == "corner_flags":
            return self._data.corner_flags
        elif mode == "center_line_flags":
            return self._data.center_line_flags
        return []

    def set_points_for_mode(self, mode: str, pts: List[List[int]]):
        """Ersetzt die Punktliste eines Modus komplett."""
        if mode == "field_boundary":
            self._data.field_boundary = pts
        elif mode == "center_line":
            self._data.center_line = pts
        elif mode == "center_half_ellipse":
            self._data.center_half_ellipse_points = pts
        elif mode == "penalty_left":
            self._data.penalty_area_left = pts
        elif mode == "penalty_right":
            self._data.penalty_area_right = pts
        elif mode == "corner_flags":
            self._data.corner_flags = pts
        elif mode == "center_line_flags":
            self._data.center_line_flags = pts
        elif mode == "center_ellipse":
            if len(pts) > 0:
                self._data.center_circle_center = pts[0]
            if len(pts) > 1:
                self._data.center_circle_horizontal = pts[1]
            if len(pts) > 2:
                self._data.center_circle_vertical = pts[2]

    # ── Punkt hinzufügen ─────────────────────────────────────────

    def add_point(self, mode: str, x: float, y: float, max_pts: int) -> bool:
        """Fügt einen Punkt hinzu. Gibt True zurück wenn der Modus damit voll ist.

        Returns:
            True wenn der Modus nach dem Hinzufügen abgeschlossen ist
            (z.B. center_ellipse mit 3 Punkten).
        """
        if mode == "done":
            return False

        current_pts = self.points_for_mode(mode)
        if max_pts > 0 and len(current_pts) >= max_pts:
            return False

        pt = [int(round(x)), int(round(y))]

        if mode == "center_ellipse":
            if not self._data.center_circle_center:
                self._data.center_circle_center = pt
            elif not self._data.center_circle_horizontal:
                self._data.center_circle_horizontal = pt
            elif not self._data.center_circle_vertical:
                self._data.center_circle_vertical = pt
                return True  # Modus abgeschlossen
        else:
            current_pts.append(pt)

        return False

    # ── Letzten Punkt entfernen ──────────────────────────────────

    def remove_last_point(self, mode: str):
        """Entfernt den zuletzt gesetzten Punkt des Modus."""
        if mode == "done":
            return

        if mode == "center_ellipse":
            if self._data.center_circle_vertical:
                self._data.center_circle_vertical = None
            elif self._data.center_circle_horizontal:
                self._data.center_circle_horizontal = None
            elif self._data.center_circle_center:
                self._data.center_circle_center = None
        else:
            pts = self.points_for_mode(mode)
            if pts:
                pts.pop()

    # ── Punkt per Index entfernen ────────────────────────────────

    def remove_point_at(self, mode: str, index: int):
        """Entfernt einen Punkt an gegebener Indexposition (z.B. per Rechtsklick)."""
        if mode == "done":
            return

        if mode == "center_ellipse":
            if index == 2:
                self._data.center_circle_vertical = None
            elif index == 1:
                self._data.center_circle_horizontal = None
                self._data.center_circle_vertical = None
            elif index == 0:
                self._data.center_circle_center = None
                self._data.center_circle_horizontal = None
                self._data.center_circle_vertical = None
        else:
            pts = self.points_for_mode(mode)
            if 0 <= index < len(pts):
                pts.pop(index)

    # ── Punkt auf Linie einfügen ────────────────────────────────

    def insert_on_line(self, mode: str, x: float, y: float,
                       max_pts: int, threshold: float) -> bool:
        """Fügt einen Punkt auf dem nächstgelegenen Liniensegment ein.

        Returns:
            True wenn ein Punkt erfolgreich eingefügt wurde.
        """
        if mode in ("done", "center_ellipse") or mode in FLAG_MODES:
            return False

        pts = self.points_for_mode(mode)
        if len(pts) < 2:
            return False

        if max_pts > 0 and len(pts) >= max_pts:
            return False

        click = np.array([x, y])
        is_closed = mode in CLOSED_MODES
        n_segments = len(pts) if is_closed else len(pts) - 1

        best_dist = float("inf")
        best_seg = -1

        for i in range(n_segments):
            a = np.array(pts[i], dtype=float)
            b = np.array(pts[(i + 1) % len(pts)], dtype=float)
            ab = b - a
            ab_len_sq = float(np.dot(ab, ab))
            if ab_len_sq < 1e-9:
                continue
            t = float(np.dot(click - a, ab)) / ab_len_sq
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            dist = float(np.linalg.norm(click - proj))
            if dist < best_dist:
                best_dist = dist
                best_seg = i

        if best_seg < 0 or best_dist > threshold:
            return False

        pt = [int(round(x)), int(round(y))]
        pts.insert(best_seg + 1, pt)
        return True

    # ── Alle Punkte eines Modus leeren ──────────────────────────

    def clear_mode(self, mode: str):
        """Löscht alle Punkte des angegebenen Modus."""
        if mode == "done":
            return

        if mode == "center_ellipse":
            self._data.center_circle_center = None
            self._data.center_circle_horizontal = None
            self._data.center_circle_vertical = None
        else:
            pts = self.points_for_mode(mode)
            pts.clear()

    # ── Punkt verschieben ────────────────────────────────────────

    def move_point(self, mode: str, index: int, new_pos: QPointF):
        """Aktualisiert die Position eines Punkts nach einem Drag."""
        pt = [int(round(new_pos.x())), int(round(new_pos.y()))]

        if mode == "center_ellipse":
            if index == 0:
                self._data.center_circle_center = pt
            elif index == 1:
                self._data.center_circle_horizontal = pt
            elif index == 2:
                self._data.center_circle_vertical = pt
        else:
            pts = self.points_for_mode(mode)
            if index < len(pts):
                pts[index] = pt
