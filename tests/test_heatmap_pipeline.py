import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from detection.motion_candidates import (
    expected_ball_radius_px,
    find_motion_candidates,
    motion_support_score,
    _estimate_global_shift,
)
from detection.temporal_tracker import TemporalBallTracker
from training.export_heatmap_data import _load_markers


def test_expected_ball_radius_increases_towards_bottom():
    top = expected_ball_radius_px(50, 1000)
    bottom = expected_ball_radius_px(950, 1000)

    assert top < bottom
    assert 2.0 <= top <= 11.0
    assert 2.0 <= bottom <= 11.0


def test_motion_candidates_find_small_moving_ball():
    rng = np.random.default_rng(123)
    texture = rng.integers(0, 35, size=(120, 160, 1), dtype=np.uint8)
    base = np.repeat(texture, 3, axis=2)
    prev_frame = base.copy()
    cur_frame = base.copy()
    next_frame = base.copy()
    for frame in (prev_frame, cur_frame, next_frame):
        cv2.rectangle(frame, (10, 10), (35, 30), (70, 70, 70), -1)
        cv2.line(frame, (0, 100), (150, 35), (90, 90, 90), 2)

    cv2.circle(prev_frame, (45, 60), 4, (255, 255, 255), -1)
    cv2.circle(cur_frame, (55, 60), 4, (255, 255, 255), -1)
    cv2.circle(next_frame, (65, 60), 4, (255, 255, 255), -1)

    candidates = find_motion_candidates(prev_frame, cur_frame, next_frame)

    assert candidates
    assert any(abs(candidate.x - 55) < 20 and abs(candidate.y - 60) < 10 for candidate in candidates)
    assert motion_support_score(55, 60, candidates, 20) > 0.0


def test_motion_candidates_respect_field_mask():
    prev_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cur_frame = np.zeros_like(prev_frame)
    next_frame = np.zeros_like(prev_frame)

    cv2.circle(prev_frame, (80, 80), 4, (255, 255, 255), -1)
    cv2.circle(cur_frame, (82, 80), 4, (255, 255, 255), -1)
    cv2.circle(next_frame, (84, 80), 4, (255, 255, 255), -1)

    field = np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]], dtype=np.int32)
    candidates = find_motion_candidates(
        prev_frame,
        cur_frame,
        next_frame,
        field_boundary=field,
        field_boundary_wh=(100, 100),
        field_margin_px=0,
    )

    assert candidates == []


def test_global_camera_shift_is_estimated():
    reference = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(reference, (20, 20), (80, 70), 180, -1)
    cv2.line(reference, (0, 100), (150, 30), 220, 2)
    matrix = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -4.0]], dtype=np.float32)
    shifted = cv2.warpAffine(reference, matrix, (160, 120), borderMode=cv2.BORDER_REPLICATE)

    dx, dy = _estimate_global_shift(reference, shifted)

    assert abs(dx - 7.0) < 0.75
    assert abs(dy + 4.0) < 0.75


def test_large_camera_jolt_disables_motion_candidates():
    base = np.zeros((140, 180, 3), dtype=np.uint8)
    cv2.rectangle(base, (15, 15), (80, 80), (100, 100, 100), -1)
    cv2.line(base, (0, 120), (170, 30), (150, 150, 150), 2)

    matrix_prev = np.array([[1.0, 0.0, -28.0], [0.0, 1.0, 12.0]], dtype=np.float32)
    matrix_next = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, -14.0]], dtype=np.float32)
    prev_frame = cv2.warpAffine(base, matrix_prev, (180, 140), borderMode=cv2.BORDER_REPLICATE)
    cur_frame = base.copy()
    next_frame = cv2.warpAffine(base, matrix_next, (180, 140), borderMode=cv2.BORDER_REPLICATE)
    cv2.circle(cur_frame, (90, 70), 4, (255, 255, 255), -1)

    candidates = find_motion_candidates(prev_frame, cur_frame, next_frame)

    assert candidates == []


def test_light_unstable_camera_sway_disables_motion_candidates():
    base = np.zeros((140, 180, 3), dtype=np.uint8)
    cv2.rectangle(base, (20, 20), (90, 80), (95, 95, 95), -1)
    cv2.line(base, (5, 125), (175, 35), (170, 170, 170), 2)
    cv2.circle(base, (135, 95), 12, (120, 120, 120), -1)

    center = (90, 70)
    prev_matrix = cv2.getRotationMatrix2D(center, -1.7, 1.0)
    next_matrix = cv2.getRotationMatrix2D(center, 1.9, 1.0)
    prev_matrix[:, 2] += (-2.0, 1.5)
    next_matrix[:, 2] += (2.0, -1.0)
    prev_frame = cv2.warpAffine(base, prev_matrix, (180, 140), borderMode=cv2.BORDER_REPLICATE)
    cur_frame = base.copy()
    next_frame = cv2.warpAffine(base, next_matrix, (180, 140), borderMode=cv2.BORDER_REPLICATE)
    cv2.circle(cur_frame, (92, 68), 4, (255, 255, 255), -1)

    candidates = find_motion_candidates(prev_frame, cur_frame, next_frame)

    assert candidates == []


def test_temporal_tracker_prefers_plausible_nearby_detection():
    tracker = TemporalBallTracker(gate=0.2)
    tracker.update((0.50, 0.50, 0.004, 0.9), 10)
    tracker.update((0.52, 0.50, 0.004, 0.9), 11)

    selected = tracker.select(
        [
            (0.54, 0.50, 0.004, 0.55),
            (0.90, 0.10, 0.004, 0.65),
        ],
        12,
    )

    assert selected is not None
    assert selected[0] == 0.54
    assert selected[1] == 0.50


def test_heatmap_marker_loader_keeps_exclusion_as_hard_negative(tmp_path):
    marker_json = tmp_path / "markers.json"
    marker_json.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_file": "file:///tmp/test.mp4",
                        "frames": [
                            {
                                "frame_index": 12,
                                "markers": [
                                    {
                                        "type": "manual",
                                        "position": {"x": 0.4, "y": 0.5},
                                        "radius": 0.01,
                                    },
                                    {
                                        "type": "exclusion",
                                        "position": {"x": 0.7, "y": 0.8},
                                        "radius": 0.02,
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    markers = _load_markers(str(marker_json))

    assert [marker["type"] for marker in markers] == ["manual", "exclusion"]
