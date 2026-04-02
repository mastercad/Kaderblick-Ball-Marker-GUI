"""Tests für calibration.drag_point."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor

from calibration.drag_point import DragPoint


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def scene(qapp):
    return QGraphicsScene()


class TestDragPoint:
    def test_creation(self, scene):
        point = DragPoint(100, 200, 10.0, QColor(255, 0, 0), 0, mode="test")
        scene.addItem(point)
        assert point.pos().x() == 100
        assert point.pos().y() == 200
        assert point.index == 0
        assert point.mode == "test"

    def test_default_mode_is_empty(self, scene):
        point = DragPoint(0, 0, 5.0, QColor(0, 255, 0), 0)
        assert point.mode == ""

    def test_is_movable_by_default(self, scene):
        point = DragPoint(0, 0, 5.0, QColor(0, 0, 255), 0, mode="field_boundary")
        assert point.flags() & point.GraphicsItemFlag.ItemIsMovable

    def test_sends_geometry_changes(self, scene):
        point = DragPoint(0, 0, 5.0, QColor(0, 0, 255), 0)
        assert point.flags() & point.GraphicsItemFlag.ItemSendsGeometryChanges

    def test_label_shows_index_plus_one(self, scene):
        point = DragPoint(0, 0, 5.0, QColor(255, 255, 0), 3, mode="center_line")
        label_text = point._label.toPlainText()
        assert label_text == "4"  # index 3 → Label "4"

    def test_z_value(self, scene):
        point = DragPoint(0, 0, 5.0, QColor(255, 0, 0), 0)
        assert point.zValue() == 20

    def test_on_moved_callback(self, scene):
        moved_calls = []

        def on_moved(idx, pos):
            moved_calls.append((idx, pos.x(), pos.y()))

        point = DragPoint(10, 20, 5.0, QColor(255, 0, 0), 2, on_moved=on_moved)
        scene.addItem(point)
        # Trigger position change
        point.setPos(30, 40)
        assert len(moved_calls) == 1
        assert moved_calls[0][0] == 2
        assert moved_calls[0][1] == 30.0
        assert moved_calls[0][2] == 40.0

    def test_multiple_points_different_indices(self, scene):
        points = []
        for i in range(5):
            p = DragPoint(i * 10, i * 10, 5.0, QColor(0, 0, 0), i, mode="test")
            scene.addItem(p)
            points.append(p)
        for i, p in enumerate(points):
            assert p.index == i
            assert p._label.toPlainText() == str(i + 1)
