"""Tests für calibration.point_manager."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from calibration.field_calibration import FieldCalibrationData
from calibration.point_manager import PointManager


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
def mgr(data):
    return PointManager(data)


# ── points_for_mode ─────────────────────────────────────────────

class TestPointsForMode:
    def test_field_boundary_empty(self, mgr):
        assert mgr.points_for_mode("field_boundary") == []

    def test_field_boundary_returns_live_reference(self, mgr, data):
        data.field_boundary.append([10, 20])
        pts = mgr.points_for_mode("field_boundary")
        assert pts == [[10, 20]]
        # Mutation durch Referenz
        pts.append([30, 40])
        assert data.field_boundary == [[10, 20], [30, 40]]

    def test_center_line(self, mgr, data):
        data.center_line = [[1, 2], [3, 4]]
        assert mgr.points_for_mode("center_line") == [[1, 2], [3, 4]]

    def test_center_half_ellipse(self, mgr, data):
        data.center_half_ellipse_points = [[5, 6]]
        assert mgr.points_for_mode("center_half_ellipse") == [[5, 6]]

    def test_penalty_left(self, mgr, data):
        data.penalty_area_left = [[7, 8]]
        assert mgr.points_for_mode("penalty_left") == [[7, 8]]

    def test_penalty_right(self, mgr, data):
        data.penalty_area_right = [[9, 10]]
        assert mgr.points_for_mode("penalty_right") == [[9, 10]]

    def test_corner_flags(self, mgr, data):
        data.corner_flags = [[11, 12]]
        assert mgr.points_for_mode("corner_flags") == [[11, 12]]

    def test_center_line_flags(self, mgr, data):
        data.center_line_flags = [[13, 14]]
        assert mgr.points_for_mode("center_line_flags") == [[13, 14]]

    def test_center_ellipse_empty(self, mgr):
        assert mgr.points_for_mode("center_ellipse") == []

    def test_center_ellipse_partial(self, mgr, data):
        data.center_circle_center = [100, 200]
        assert mgr.points_for_mode("center_ellipse") == [[100, 200]]

    def test_center_ellipse_full(self, mgr, data):
        data.center_circle_center = [100, 200]
        data.center_circle_horizontal = [150, 200]
        data.center_circle_vertical = [100, 250]
        assert len(mgr.points_for_mode("center_ellipse")) == 3

    def test_unknown_mode_returns_empty(self, mgr):
        assert mgr.points_for_mode("nonexistent") == []


# ── add_point ────────────────────────────────────────────────────

class TestAddPoint:
    def test_add_to_field_boundary(self, mgr, data):
        mgr.add_point("field_boundary", 100.7, 200.3, 0)
        assert data.field_boundary == [[101, 200]]

    def test_add_multiple(self, mgr, data):
        mgr.add_point("field_boundary", 10, 20, 0)
        mgr.add_point("field_boundary", 30, 40, 0)
        assert len(data.field_boundary) == 2

    def test_max_points_respected(self, mgr, data):
        data.corner_flags = [[1, 1], [2, 2]]
        result = mgr.add_point("corner_flags", 3, 3, 2)
        assert result is False
        assert len(data.corner_flags) == 2

    def test_add_center_ellipse_first_point(self, mgr, data):
        mgr.add_point("center_ellipse", 50, 60, 3)
        assert data.center_circle_center == [50, 60]
        assert data.center_circle_horizontal is None

    def test_add_center_ellipse_second_point(self, mgr, data):
        data.center_circle_center = [50, 60]
        mgr.add_point("center_ellipse", 80, 60, 3)
        assert data.center_circle_horizontal == [80, 60]

    def test_add_center_ellipse_third_returns_true(self, mgr, data):
        data.center_circle_center = [50, 60]
        data.center_circle_horizontal = [80, 60]
        result = mgr.add_point("center_ellipse", 50, 90, 3)
        assert result is True
        assert data.center_circle_vertical == [50, 90]

    def test_done_mode_ignored(self, mgr, data):
        result = mgr.add_point("done", 10, 20, 0)
        assert result is False

    def test_rounds_coordinates(self, mgr, data):
        mgr.add_point("center_line", 10.5, 20.4, 0)
        assert data.center_line[-1] == [10, 20]  # round(10.5)=10, round(20.4)=20

        mgr.add_point("center_line", 10.6, 20.5, 0)
        assert data.center_line[-1] == [11, 20]  # round(10.6)=11, round(20.5)=20


# ── remove_last_point ────────────────────────────────────────────

class TestRemoveLastPoint:
    def test_remove_from_list(self, mgr, data):
        data.field_boundary = [[1, 2], [3, 4], [5, 6]]
        mgr.remove_last_point("field_boundary")
        assert data.field_boundary == [[1, 2], [3, 4]]

    def test_remove_from_empty_list(self, mgr, data):
        mgr.remove_last_point("field_boundary")  # Sollte keinen Fehler werfen

    def test_remove_center_ellipse_vertical(self, mgr, data):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        data.center_circle_vertical = [10, 40]
        mgr.remove_last_point("center_ellipse")
        assert data.center_circle_vertical is None
        assert data.center_circle_horizontal is not None

    def test_remove_center_ellipse_horizontal(self, mgr, data):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        mgr.remove_last_point("center_ellipse")
        assert data.center_circle_horizontal is None
        assert data.center_circle_center is not None

    def test_remove_center_ellipse_center(self, mgr, data):
        data.center_circle_center = [10, 20]
        mgr.remove_last_point("center_ellipse")
        assert data.center_circle_center is None

    def test_done_mode_noop(self, mgr):
        mgr.remove_last_point("done")  # Kein Fehler


# ── remove_point_at ──────────────────────────────────────────────

class TestRemovePointAt:
    def test_remove_middle_point(self, mgr, data):
        data.field_boundary = [[1, 2], [3, 4], [5, 6]]
        mgr.remove_point_at("field_boundary", 1)
        assert data.field_boundary == [[1, 2], [5, 6]]

    def test_remove_first_point(self, mgr, data):
        data.center_line = [[10, 20], [30, 40]]
        mgr.remove_point_at("center_line", 0)
        assert data.center_line == [[30, 40]]

    def test_index_out_of_range(self, mgr, data):
        data.field_boundary = [[1, 2]]
        mgr.remove_point_at("field_boundary", 5)  # Kein Fehler
        assert data.field_boundary == [[1, 2]]

    def test_center_ellipse_remove_index_0_clears_all(self, mgr, data):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        data.center_circle_vertical = [10, 40]
        mgr.remove_point_at("center_ellipse", 0)
        assert data.center_circle_center is None
        assert data.center_circle_horizontal is None
        assert data.center_circle_vertical is None

    def test_center_ellipse_remove_index_1(self, mgr, data):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        data.center_circle_vertical = [10, 40]
        mgr.remove_point_at("center_ellipse", 1)
        assert data.center_circle_center is not None
        assert data.center_circle_horizontal is None
        assert data.center_circle_vertical is None

    def test_center_ellipse_remove_index_2(self, mgr, data):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        data.center_circle_vertical = [10, 40]
        mgr.remove_point_at("center_ellipse", 2)
        assert data.center_circle_vertical is None
        assert data.center_circle_horizontal is not None

    def test_done_mode_noop(self, mgr):
        mgr.remove_point_at("done", 0)  # Kein Fehler


# ── insert_on_line ───────────────────────────────────────────────

class TestInsertOnLine:
    def test_insert_between_two_points(self, mgr, data):
        data.field_boundary = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Punkt nahe der oberen Kante (zwischen [0,0] und [100,0])
        result = mgr.insert_on_line("field_boundary", 50, 5, 0, threshold=50.0)
        assert result is True
        assert len(data.field_boundary) == 5
        assert data.field_boundary[1] == [50, 5]

    def test_insert_too_far_from_line(self, mgr, data):
        data.center_line = [[0, 0], [100, 0]]
        result = mgr.insert_on_line("center_line", 50, 200, 0, threshold=50.0)
        assert result is False
        assert len(data.center_line) == 2

    def test_insert_not_enough_points(self, mgr, data):
        data.center_line = [[0, 0]]
        result = mgr.insert_on_line("center_line", 50, 0, 0, threshold=50.0)
        assert result is False

    def test_done_mode_returns_false(self, mgr, data):
        result = mgr.insert_on_line("done", 50, 50, 0, threshold=50.0)
        assert result is False

    def test_center_ellipse_returns_false(self, mgr, data):
        result = mgr.insert_on_line("center_ellipse", 50, 50, 0, threshold=50.0)
        assert result is False

    def test_flag_mode_returns_false(self, mgr, data):
        data.corner_flags = [[0, 0], [100, 0]]
        result = mgr.insert_on_line("corner_flags", 50, 0, 0, threshold=50.0)
        assert result is False

    def test_max_pts_respected(self, mgr, data):
        data.center_line = [[0, 0], [100, 0]]
        result = mgr.insert_on_line("center_line", 50, 0, 2, threshold=50.0)
        assert result is False

    def test_insert_on_closed_polygon(self, mgr, data):
        """Geschlossene Form: Einfügen auf dem Segment zwischen letztem und erstem Punkt."""
        data.penalty_area_left = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Punkt nahe der linken Kante (zwischen [0,100] und [0,0])
        result = mgr.insert_on_line("penalty_left", 5, 50, 0, threshold=50.0)
        assert result is True
        assert len(data.penalty_area_left) == 5


# ── clear_mode ───────────────────────────────────────────────────

class TestClearMode:
    def test_clear_list_mode(self, mgr, data):
        data.field_boundary = [[1, 2], [3, 4]]
        mgr.clear_mode("field_boundary")
        assert data.field_boundary == []

    def test_clear_center_ellipse(self, mgr, data):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        data.center_circle_vertical = [10, 40]
        mgr.clear_mode("center_ellipse")
        assert data.center_circle_center is None
        assert data.center_circle_horizontal is None
        assert data.center_circle_vertical is None

    def test_clear_done_noop(self, mgr):
        mgr.clear_mode("done")  # Kein Fehler


# ── move_point ───────────────────────────────────────────────────

class TestMovePoint:
    def test_move_list_point(self, mgr, data, qapp):
        data.field_boundary = [[10, 20], [30, 40]]
        mgr.move_point("field_boundary", 0, QPointF(15.7, 25.3))
        assert data.field_boundary[0] == [16, 25]

    def test_move_center_ellipse_center(self, mgr, data, qapp):
        data.center_circle_center = [10, 20]
        mgr.move_point("center_ellipse", 0, QPointF(50, 60))
        assert data.center_circle_center == [50, 60]

    def test_move_center_ellipse_horizontal(self, mgr, data, qapp):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        mgr.move_point("center_ellipse", 1, QPointF(40, 20))
        assert data.center_circle_horizontal == [40, 20]

    def test_move_center_ellipse_vertical(self, mgr, data, qapp):
        data.center_circle_center = [10, 20]
        data.center_circle_horizontal = [30, 20]
        data.center_circle_vertical = [10, 40]
        mgr.move_point("center_ellipse", 2, QPointF(10, 50))
        assert data.center_circle_vertical == [10, 50]

    def test_move_out_of_range_noop(self, mgr, data, qapp):
        data.field_boundary = [[10, 20]]
        mgr.move_point("field_boundary", 5, QPointF(99, 99))
        assert data.field_boundary == [[10, 20]]


# ── set_points_for_mode ─────────────────────────────────────────

class TestSetPointsForMode:
    def test_set_field_boundary(self, mgr, data):
        mgr.set_points_for_mode("field_boundary", [[1, 2], [3, 4]])
        assert data.field_boundary == [[1, 2], [3, 4]]

    def test_set_center_ellipse(self, mgr, data):
        mgr.set_points_for_mode("center_ellipse", [[10, 20], [30, 20], [10, 40]])
        assert data.center_circle_center == [10, 20]
        assert data.center_circle_horizontal == [30, 20]
        assert data.center_circle_vertical == [10, 40]

    def test_set_corner_flags(self, mgr, data):
        mgr.set_points_for_mode("corner_flags", [[5, 5], [95, 5]])
        assert data.corner_flags == [[5, 5], [95, 5]]


# ── data property ────────────────────────────────────────────────

class TestDataProperty:
    def test_get_data(self, mgr, data):
        assert mgr.data is data

    def test_set_data(self, mgr):
        new_data = FieldCalibrationData(camera_id=1)
        new_data.field_boundary = [[99, 99]]
        mgr.data = new_data
        assert mgr.points_for_mode("field_boundary") == [[99, 99]]
