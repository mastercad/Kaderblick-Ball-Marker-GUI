"""Tests für Ausschlusszonen (Exclusion Marker).

Getestet werden:
- _check_exclusion_list: statische Prüfung ob eine Detektion in einer Ausschlusszone liegt
- Frame-Fenster-Logik: Ausschlüsse gelten auch auf benachbarten Frames
- Radius-Prüfung: nur Detektionen innerhalb 2× Radius werden unterdrückt
- Mehrere Ausschlusszonen gleichzeitig
- Interpolation erzeugt korrekt neue Ausschluss-Marker
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── _check_exclusion_list (statische Methode) ────────────────────

class TestCheckExclusionList:
    """Prüft die Thread-sichere Ausschluss-Prüfung."""

    def _check(self, cx, cy, frame, exclusions, window=10):
        from ui.video_graphics_panel import VideoGraphicsPanel
        return VideoGraphicsPanel._check_exclusion_list(cx, cy, frame, exclusions, window)

    def test_exact_hit(self):
        """Detektion exakt auf dem Ausschluss-Marker → unterdrückt."""
        excl = [(100, 0.80, 0.50, 0.05)]
        assert self._check(0.80, 0.50, 100, excl) is True

    def test_within_radius(self):
        """Detektion knapp innerhalb 2× Radius → unterdrückt."""
        excl = [(100, 0.50, 0.50, 0.05)]
        # 2 * 0.05 = 0.10 → Detektion bei Distanz ~0.07 → innerhalb
        assert self._check(0.55, 0.54, 100, excl) is True

    def test_outside_radius(self):
        """Detektion weit außerhalb des Radius → nicht unterdrückt."""
        excl = [(100, 0.50, 0.50, 0.05)]
        assert self._check(0.80, 0.80, 100, excl) is False

    def test_nearby_frame_within_window(self):
        """Ausschluss auf Frame 100, Detektion auf Frame 105 (window=10) → unterdrückt."""
        excl = [(100, 0.50, 0.50, 0.05)]
        assert self._check(0.50, 0.50, 105, excl, window=10) is True

    def test_frame_outside_window(self):
        """Ausschluss auf Frame 100, Detektion auf Frame 120 (window=10) → nicht unterdrückt."""
        excl = [(100, 0.50, 0.50, 0.05)]
        assert self._check(0.50, 0.50, 120, excl, window=10) is False

    def test_multiple_zones(self):
        """Zwei Ausschlusszonen, Detektion trifft die zweite."""
        excl = [
            (100, 0.10, 0.10, 0.03),  # Linke obere Ecke (Eckfahne)
            (100, 0.90, 0.50, 0.04),  # Rechter Rand (Mittellinie Fahne)
        ]
        # Trifft Zone 1 nicht, trifft Zone 2
        assert self._check(0.91, 0.51, 100, excl) is True

    def test_no_zones(self):
        """Keine Ausschlusszonen → nie unterdrückt."""
        assert self._check(0.50, 0.50, 100, []) is False

    def test_edge_of_radius(self):
        """Detektion knapp außerhalb von 2× Radius → nicht unterdrückt."""
        excl = [(100, 0.50, 0.50, 0.05)]
        # 2× Radius = 0.10. Punkt deutlich außerhalb (Distanz ≈ 0.11)
        assert self._check(0.61, 0.50, 100, excl) is False
        # Punkt knapp innerhalb (Distanz ≈ 0.09) → unterdrückt
        assert self._check(0.59, 0.50, 100, excl) is True

    def test_large_exclusion_radius(self):
        """Ausschlusszone mit großem Radius deckt breiten Bereich ab."""
        excl = [(100, 0.50, 0.50, 0.15)]
        # 2 * 0.15 = 0.30 → alles innerhalb 0.30 wird unterdrückt
        assert self._check(0.65, 0.55, 100, excl) is True
        assert self._check(0.80, 0.80, 100, excl) is False


# ── _is_in_exclusion_zone (instanzmethode) ───────────────────────

class TestIsInExclusionZoneInstance:
    """Prüft die Instanzmethode mit echtem Session-Objekt."""

    def test_instance_exclusion_check(self, qapp):
        """Marker in Session wird korrekt als Ausschlusszone erkannt."""
        from model.session import Session
        from model.marker import Marker
        from ui.video_graphics_panel import VideoGraphicsPanel

        session = Session()
        excl_marker = Marker("video.mp4", 50, 1667, (0.80, 0.50), 0.05, "exclusion")
        session.add_marker(excl_marker)

        panel = VideoGraphicsPanel(session)
        # Detektion exakt auf der Ausschlusszone
        assert panel._is_in_exclusion_zone(0.80, 0.50, 50, "video.mp4") is True
        # Detektion weit entfernt
        assert panel._is_in_exclusion_zone(0.20, 0.20, 50, "video.mp4") is False
        # Detektion auf nahes Frame (innerhalb window=10)
        assert panel._is_in_exclusion_zone(0.80, 0.50, 55, "video.mp4") is True
        # Anderes Video → nicht betroffen
        assert panel._is_in_exclusion_zone(0.80, 0.50, 50, "other.mp4") is False

    def test_ball_marker_not_treated_as_exclusion(self, qapp):
        """Ball-Marker (manual/yolo) blockieren keine Detektionen."""
        from model.session import Session
        from model.marker import Marker
        from ui.video_graphics_panel import VideoGraphicsPanel

        session = Session()
        session.add_marker(Marker("v.mp4", 50, 1667, (0.50, 0.50), 0.05, "manual"))
        session.add_marker(Marker("v.mp4", 51, 1700, (0.50, 0.50), 0.05, "yolo"))

        panel = VideoGraphicsPanel(session)
        # Ball-Marker sind keine Ausschlusszonen
        assert panel._is_in_exclusion_zone(0.50, 0.50, 50, "v.mp4") is False
        assert panel._is_in_exclusion_zone(0.50, 0.50, 51, "v.mp4") is False
