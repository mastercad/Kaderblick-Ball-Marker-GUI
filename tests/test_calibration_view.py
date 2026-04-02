"""Tests für calibration.calibration_view."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QTransform

from calibration.calibration_view import CalibrationView
from calibration.drag_point import DragPoint


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def scene(qapp):
    s = QGraphicsScene()
    s.setSceneRect(0, 0, 1920, 1080)
    return s


@pytest.fixture
def view(scene):
    v = CalibrationView(scene)
    return v


class TestCalibrationView:
    def test_creation(self, view):
        assert view._zoom == 1.0
        assert view._panning is False
        assert view._active_mode == ""

    def test_fit_image_resets_zoom(self, view):
        view._zoom = 5.0
        view.fit_image()
        assert view._zoom == 1.0

    def test_active_mode_can_be_set(self, view):
        view._active_mode = "field_boundary"
        assert view._active_mode == "field_boundary"

    def test_has_signals(self, view):
        """Prüft dass die Signale existieren."""
        assert hasattr(view, "point_clicked")
        assert hasattr(view, "point_remove_requested")
        assert hasattr(view, "line_insert_requested")

    def test_initial_drag_mode(self, view):
        from PySide6.QtWidgets import QGraphicsView
        assert view.dragMode() == QGraphicsView.DragMode.NoDrag

    def test_click_timer_is_singleshot(self, view):
        assert view._click_timer.isSingleShot()
        assert view._click_timer.interval() == 250

    def test_pending_click_initially_none(self, view):
        assert view._pending_click is None
