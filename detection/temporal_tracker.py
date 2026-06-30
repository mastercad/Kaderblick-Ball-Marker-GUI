"""Temporal ball tracker for stabilising frame-by-frame detections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BallTrack:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.004
    confidence: float = 0.0
    last_frame: int = -1
    misses: int = 0

    def predict(self, frame_index: int) -> tuple[float, float]:
        if self.last_frame < 0:
            return self.x, self.y
        dt = max(1, frame_index - self.last_frame)
        return self.x + self.vx * dt, self.y + self.vy * dt


class TemporalBallTracker:
    """Small alpha-beta tracker tuned for one ball trajectory."""

    def __init__(self, max_misses: int = 12, gate: float = 0.09):
        self.track: BallTrack | None = None
        self.max_misses = max_misses
        self.gate = gate

    def prediction_score(self, x: float, y: float, frame_index: int) -> float:
        if self.track is None or self.track.confidence <= 0:
            return 0.0
        px, py = self.track.predict(frame_index)
        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if dist > self.gate:
            return 0.0
        return max(0.0, 1.0 - dist / self.gate) * min(1.0, self.track.confidence)

    def select(self, detections: list[tuple[float, float, float, float]], frame_index: int):
        if not detections:
            self.mark_missed()
            return None
        if self.track is None:
            return detections[0]

        best = None
        best_score = -1.0
        for det in detections:
            x, y, _r, score = det
            temporal = self.prediction_score(x, y, frame_index)
            combined = 0.72 * score + 0.28 * temporal
            if combined > best_score:
                best = det
                best_score = combined
        return best

    def update(self, detection: tuple[float, float, float, float] | None, frame_index: int):
        if detection is None:
            self.mark_missed()
            return self.track

        x, y, radius, score = detection
        if self.track is None or self.track.last_frame < 0:
            self.track = BallTrack(
                x=x,
                y=y,
                radius=radius,
                confidence=score,
                last_frame=frame_index,
                misses=0,
            )
            return self.track

        dt = max(1, frame_index - self.track.last_frame)
        measured_vx = (x - self.track.x) / dt
        measured_vy = (y - self.track.y) / dt
        alpha = 0.72
        beta = 0.34
        self.track.x = alpha * x + (1.0 - alpha) * (self.track.x + self.track.vx * dt)
        self.track.y = alpha * y + (1.0 - alpha) * (self.track.y + self.track.vy * dt)
        self.track.vx = beta * measured_vx + (1.0 - beta) * self.track.vx
        self.track.vy = beta * measured_vy + (1.0 - beta) * self.track.vy
        self.track.radius = 0.8 * radius + 0.2 * self.track.radius
        self.track.confidence = min(1.0, 0.65 * self.track.confidence + 0.45 * score)
        self.track.last_frame = frame_index
        self.track.misses = 0
        return self.track

    def mark_missed(self):
        if self.track is None:
            return
        self.track.misses += 1
        self.track.confidence *= 0.75
        if self.track.misses > self.max_misses or self.track.confidence < 0.08:
            self.track = None
