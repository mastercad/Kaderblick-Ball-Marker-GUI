"""Motion candidates for tiny ball detection.

The heatmap model answers "where does this look like a ball?". This module adds
"where is a small object moving plausibly?" so tiny balls are not judged from a
single still image alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionCandidate:
    x: float
    y: float
    radius: float
    score: float
    area: float


@dataclass(frozen=True)
class GlobalMotion:
    dx: float
    dy: float
    response: float
    stable: bool


def point_in_field_pixels(
    x: float,
    y: float,
    frame_w: int,
    frame_h: int,
    field_boundary: Optional[np.ndarray],
    field_boundary_wh: Optional[tuple[int, int]],
    margin_px: int,
) -> bool:
    if field_boundary is None or field_boundary_wh is None:
        return True
    fw, fh = field_boundary_wh
    px = (x / max(1, frame_w)) * fw
    py = (y / max(1, frame_h)) * fh
    dist = cv2.pointPolygonTest(field_boundary, (px, py), measureDist=True)
    return dist >= -margin_px


def expected_ball_radius_px(y: float, frame_h: int, min_radius: float = 2.0, max_radius: float = 11.0) -> float:
    """Approximate perspective prior: lower image areas usually contain larger balls."""
    if frame_h <= 0:
        return min_radius
    t = max(0.0, min(1.0, y / frame_h))
    return min_radius + (max_radius - min_radius) * (t * t)


def radius_score(radius: float, expected_radius: float) -> float:
    tolerance = max(2.0, expected_radius * 1.2)
    return float(np.exp(-((radius - expected_radius) ** 2) / (2.0 * tolerance * tolerance)))


def _estimate_global_motion(reference_gray: np.ndarray, moving_gray: np.ndarray) -> GlobalMotion:
    """Estimate camera translation between two frames on a downscaled image."""
    ref = reference_gray.astype(np.float32)
    mov = moving_gray.astype(np.float32)
    max_side = max(ref.shape[:2])
    scale = 1.0
    if max_side > 640:
        scale = 640.0 / max_side
        ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mov = cv2.resize(mov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ref, mov, window)
    except cv2.error:
        return GlobalMotion(0.0, 0.0, 0.0, False)

    dx /= scale
    dy /= scale
    shift = (dx * dx + dy * dy) ** 0.5
    stable = shift <= 1.0 or (
        response >= 0.12 and shift <= max(12.0, min(reference_gray.shape[:2]) * 0.035)
    )
    return GlobalMotion(dx, dy, float(response), stable)


def _estimate_global_shift(reference_gray: np.ndarray, moving_gray: np.ndarray) -> tuple[float, float]:
    """Compatibility wrapper used by tests and callers that only need the shift."""
    motion = _estimate_global_motion(reference_gray, moving_gray)
    return motion.dx, motion.dy


def _background_residual_stable(reference_gray: np.ndarray, aligned_gray: np.ndarray) -> bool:
    """Reject motion if too much fixed background still changes after alignment."""
    diff = cv2.absdiff(reference_gray, aligned_gray)
    if diff.size == 0:
        return False

    # Ignore the strongest tiny foreground changes; broad residuals indicate
    # camera sway, rolling shutter, vibration, or non-translational movement.
    p85 = float(np.percentile(diff, 85))
    changed_fraction = float(np.mean(diff > 18))
    return p85 <= 10.0 and changed_fraction <= 0.035


def _align_to_reference(reference_gray: np.ndarray, moving_gray: np.ndarray) -> tuple[np.ndarray, bool]:
    motion = _estimate_global_motion(reference_gray, moving_gray)
    if not motion.stable:
        return moving_gray, False
    dx, dy = motion.dx, motion.dy
    if abs(dx) < 0.25 and abs(dy) < 0.25:
        return moving_gray, True
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        moving_gray,
        matrix,
        (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, _background_residual_stable(reference_gray, aligned)


def _motion_mask(prev_frame: np.ndarray, cur_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cur_gray = cv2.GaussianBlur(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    next_gray = cv2.GaussianBlur(cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    prev_aligned, prev_stable = _align_to_reference(cur_gray, prev_gray)
    next_aligned, next_stable = _align_to_reference(cur_gray, next_gray)
    if not prev_stable or not next_stable:
        return np.zeros_like(cur_gray, dtype=np.uint8)

    diff_prev = cv2.absdiff(cur_gray, prev_aligned)
    diff_next = cv2.absdiff(next_aligned, cur_gray)
    motion = cv2.max(diff_prev, diff_next)

    _, mask = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def find_motion_candidates(
    prev_frame: np.ndarray,
    cur_frame: np.ndarray,
    next_frame: np.ndarray,
    max_candidates: int = 80,
    field_boundary: Optional[np.ndarray] = None,
    field_boundary_wh: Optional[tuple[int, int]] = None,
    field_margin_px: int = 150,
) -> list[MotionCandidate]:
    """Find small moving components that could be the ball."""
    frame_h, frame_w = cur_frame.shape[:2]
    mask = _motion_mask(prev_frame, cur_frame, next_frame)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[MotionCandidate] = []
    # A tiny moving ball creates a short motion trail across frames, not one
    # perfect static dot. Radius remains the stronger size guard below.
    max_area = max(420.0, frame_w * frame_h * 0.00045)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 2.0 or area > max_area:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 1.0 or radius > 18.0:
            continue
        if not point_in_field_pixels(x, y, frame_w, frame_h, field_boundary, field_boundary_wh, field_margin_px):
            continue

        expected = expected_ball_radius_px(y, frame_h)
        size_score = radius_score(radius, expected)
        compactness = min(1.0, area / max(1.0, np.pi * radius * radius))
        score = 0.65 * size_score + 0.35 * compactness
        candidates.append(MotionCandidate(x=x, y=y, radius=radius, score=score, area=area))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def motion_support_score(
    x: float,
    y: float,
    candidates: list[MotionCandidate],
    support_radius_px: float,
) -> float:
    """Return the strongest nearby motion support for a proposed ball center."""
    if not candidates:
        return 0.0
    best = 0.0
    support_radius_px = max(4.0, support_radius_px)
    for candidate in candidates:
        dist = ((x - candidate.x) ** 2 + (y - candidate.y) ** 2) ** 0.5
        proximity = float(np.exp(-(dist * dist) / (2.0 * support_radius_px * support_radius_px)))
        best = max(best, proximity * candidate.score)
    return best
