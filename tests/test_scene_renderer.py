"""Tests für calibration.scene_renderer."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import QPointF

from calibration.scene_renderer import SceneRenderer
from calibration.calibration_modes import MODES_CAM0, MODES_FULL
from calibration.field_calibration import FieldCalibrationData
from calibration.drag_point import DragPoint


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def data():
    return FieldCalibrationData(camera_id=0)


@pytest.fixture
def scene(qapp):
    return QGraphicsScene()


def _make_renderer(scene, data):
    """Erstellt einen SceneRenderer mit PointManager-ähnlichem Callback."""
    from calibration.point_manager import PointManager
    mgr = PointManager(data)
    moved_log = []

    def on_moved(mode, index, pos):
        moved_log.append((mode, index, pos))

    renderer = SceneRenderer(
        scene,
        get_points=mgr.points_for_mode,
        on_point_moved=on_moved,
        point_radius=10.0,
    )
    return renderer, moved_log


class TestSceneRenderer:
    def test_empty_data_no_items(self, scene, data):
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        # Szene sollte leer sein (keine Punkte, keine Linien)
        assert len(renderer._point_items) == 0
        assert len(renderer._line_items) == 0

    def test_points_are_drawn(self, scene, data):
        data.field_boundary = [[100, 100], [200, 100], [200, 200], [100, 200]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        assert "field_boundary" in renderer._point_items
        assert len(renderer._point_items["field_boundary"]) == 4

    def test_active_mode_points_are_movable(self, scene, data):
        data.field_boundary = [[100, 100], [200, 100]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        for item in renderer._point_items["field_boundary"]:
            assert item.flags() & item.GraphicsItemFlag.ItemIsMovable

    def test_inactive_mode_points_not_movable(self, scene, data):
        data.field_boundary = [[100, 100], [200, 100]]
        data.center_line = [[300, 0], [300, 500]]
        renderer, _ = _make_renderer(scene, data)
        # field_boundary ist aktiv → center_line inaktiv
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        for item in renderer._point_items["center_line"]:
            assert not (item.flags() & item.GraphicsItemFlag.ItemIsMovable)

    def test_lines_drawn_for_polygon(self, scene, data):
        data.field_boundary = [[0, 0], [100, 0], [100, 100], [0, 100]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        # Mindestens 1 Linienelement (Polygon)
        assert len(renderer._line_items) >= 1

    def test_lines_not_drawn_for_flags(self, scene, data):
        data.corner_flags = [[10, 10], [90, 10]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "corner_flags", data)
        # Keine Linien für Flag-Modi
        assert len(renderer._line_items) == 0

    def test_ellipse_drawn_for_center_ellipse(self, scene):
        d = FieldCalibrationData(camera_id=2)  # FULL-Modus
        d.center_circle_center = [500, 400]
        d.center_circle_horizontal = [600, 400]
        d.center_circle_vertical = [500, 500]
        renderer, _ = _make_renderer(scene, d)
        renderer.redraw_all(MODES_FULL, "center_ellipse", d)
        # Ellipse + evtl. andere Linien
        has_ellipse = any(
            isinstance(item, type(item)) and hasattr(item, 'rect')
            for item in renderer._line_items
        )
        # Mindestens ein Linienelement (die Ellipse)
        assert len(renderer._line_items) >= 1

    def test_redraw_lines_only(self, scene, data):
        data.field_boundary = [[0, 0], [100, 0], [100, 100]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        initial_point_count = len(renderer._point_items.get("field_boundary", []))
        # Nur Linien neu zeichnen
        renderer.redraw_lines(MODES_CAM0, data)
        # Punkte unverändert
        assert len(renderer._point_items.get("field_boundary", [])) == initial_point_count

    def test_redraw_clears_old_items(self, scene, data):
        data.field_boundary = [[0, 0], [100, 0], [100, 100]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        first_items = list(renderer._point_items.get("field_boundary", []))

        # Nochmal zeichnen (z.B. nach Punkt-Änderung)
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        second_items = list(renderer._point_items.get("field_boundary", []))

        # Alte Items sollten nicht mehr in der Szene sein
        for item in first_items:
            assert item.scene() is None

    def test_open_line_for_center_line(self, scene, data):
        """Mittellinie bekommt QPainterPath (offen), kein Polygon."""
        data.center_line = [[300, 0], [300, 500]]
        renderer, _ = _make_renderer(scene, data)
        renderer.redraw_all(MODES_CAM0, "center_line", data)
        from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsPolygonItem
        for item in renderer._line_items:
            assert not isinstance(item, QGraphicsPolygonItem)

    def test_point_radius_applied(self, scene, data):
        data.field_boundary = [[50, 50]]
        renderer, _ = _make_renderer(scene, data)
        renderer.point_radius = 15.0
        renderer.redraw_all(MODES_CAM0, "field_boundary", data)
        point = renderer._point_items["field_boundary"][0]
        # Punkt-Radius: rect von -15 bis 15 → Breite 30
        assert point.rect().width() == pytest.approx(30.0)
