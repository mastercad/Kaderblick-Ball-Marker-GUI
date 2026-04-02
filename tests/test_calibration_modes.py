"""Tests für calibration.calibration_modes."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from calibration.calibration_modes import (
    CLOSED_MODES,
    COLORS,
    DONE_MODE,
    FLAG_MODES,
    MODES_CAM0,
    MODES_CAM1,
    MODES_FULL,
    ModeSpec,
    current_mode,
    modes_for_camera,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── modes_for_camera ────────────────────────────────────────────

class TestModesForCamera:
    def test_cam0_returns_cam0_modes(self):
        assert modes_for_camera(0) is MODES_CAM0

    def test_cam1_returns_cam1_modes(self):
        assert modes_for_camera(1) is MODES_CAM1

    def test_unknown_camera_returns_full(self):
        assert modes_for_camera(2) is MODES_FULL
        assert modes_for_camera(-1) is MODES_FULL
        assert modes_for_camera(99) is MODES_FULL


# ── current_mode ────────────────────────────────────────────────

class TestCurrentMode:
    def test_first_mode(self):
        mode = current_mode(MODES_CAM0, 0)
        assert mode[0] == "field_boundary"

    def test_last_valid_index(self):
        mode = current_mode(MODES_CAM0, len(MODES_CAM0) - 1)
        assert mode[0] != "done"

    def test_index_past_end_returns_done(self):
        mode = current_mode(MODES_CAM0, len(MODES_CAM0))
        assert mode == DONE_MODE
        assert mode[0] == "done"

    def test_way_past_end_returns_done(self):
        mode = current_mode(MODES_CAM0, 1000)
        assert mode == DONE_MODE


# ── Modus-Definitionen ────────────────────────────────────────

class TestModeDefinitions:
    @pytest.mark.parametrize("modes", [MODES_CAM0, MODES_CAM1, MODES_FULL])
    def test_all_modes_are_4_tuples(self, modes):
        for m in modes:
            assert len(m) == 4
            name, desc, min_pts, max_pts = m
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert isinstance(min_pts, int)
            assert isinstance(max_pts, int)
            assert min_pts >= 0
            assert max_pts >= 0

    def test_cam0_starts_with_field_boundary(self):
        assert MODES_CAM0[0][0] == "field_boundary"

    def test_cam1_starts_with_field_boundary(self):
        assert MODES_CAM1[0][0] == "field_boundary"

    def test_cam0_has_penalty_left(self):
        names = [m[0] for m in MODES_CAM0]
        assert "penalty_left" in names

    def test_cam1_has_penalty_right(self):
        names = [m[0] for m in MODES_CAM1]
        assert "penalty_right" in names

    def test_full_has_both_penalties(self):
        names = [m[0] for m in MODES_FULL]
        assert "penalty_left" in names
        assert "penalty_right" in names

    def test_cam0_has_flags(self):
        names = [m[0] for m in MODES_CAM0]
        assert "corner_flags" in names
        assert "center_line_flags" in names

    def test_cam1_has_flags(self):
        names = [m[0] for m in MODES_CAM1]
        assert "corner_flags" in names
        assert "center_line_flags" in names


# ── Konstanten ──────────────────────────────────────────────────

class TestConstants:
    def test_closed_modes_contains_expected(self):
        assert "field_boundary" in CLOSED_MODES
        assert "penalty_left" in CLOSED_MODES
        assert "penalty_right" in CLOSED_MODES

    def test_flag_modes_contains_expected(self):
        assert "corner_flags" in FLAG_MODES
        assert "center_line_flags" in FLAG_MODES

    def test_colors_has_all_modes(self, qapp):
        required = {
            "field_boundary", "center_line", "center_half_ellipse",
            "center_ellipse", "penalty_left", "penalty_right",
            "corner_flags", "center_line_flags", "active_point",
        }
        assert required.issubset(set(COLORS.keys()))

    def test_colors_are_qcolor(self, qapp):
        for key, color in COLORS.items():
            assert isinstance(color, QColor), f"{key} is not QColor"

    def test_done_mode_structure(self):
        assert DONE_MODE[0] == "done"
        assert len(DONE_MODE) == 4
