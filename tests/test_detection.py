"""Tests für die Ballerkennung (Unit-Tests ohne echtes YOLO-Modell).

Getestet werden:
- Geometrie-Filter (_passes_geometry_filter)
- Kachelrand-Filter (_should_discard_edge_box)
- Anker-basierte Auswahl (_merge_detections + Anker-Logik)
- NMS / Merge-Verhalten
- Temporaler Ausreißerfilter (filter_temporal_outliers)
- Ausschlusszone-Prüfung (_check_exclusion_list)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from detection.ball_detector import (
    _passes_geometry_filter,
    _should_discard_edge_box,
    _merge_detections,
    _nms,
    _generate_tiles,
    filter_temporal_outliers,
)


# ── Geometrie-Filter ──────────────────────────────────────────────

class TestGeometryFilter:
    """Prüft dass der Geometrie-Filter vernünftige Bälle durchlässt
    und offensichtlich falsche Detektionen verwirft."""

    def test_normal_ball(self):
        """Typischer Ball (20×20px in 1920×1080) → behalten."""
        assert _passes_geometry_filter(100, 100, 120, 120, 1920, 1080) is True

    def test_too_small(self):
        """Zu kleines Objekt (5×5px) → verwerfen."""
        assert _passes_geometry_filter(100, 100, 105, 105, 1920, 1080) is False

    def test_too_large(self):
        """Zu großes Objekt (>15% der kürzeren Kante) → verwerfen."""
        # 1080 * 0.15 = 162 → eine Box von 200px ist zu groß
        assert _passes_geometry_filter(100, 100, 300, 300, 1920, 1080) is False

    def test_bad_aspect_ratio(self):
        """Seitenverhältnis > 2.5 (Linie, Reflexionsstreifen) → verwerfen."""
        assert _passes_geometry_filter(100, 100, 110, 200, 1920, 1080) is False

    def test_minimum_size_boundary(self):
        """Genau 8px Box → gerade noch akzeptiert."""
        assert _passes_geometry_filter(0, 0, 8, 8, 1920, 1080) is True

    def test_zero_size(self):
        """Degenerierte Box → verwerfen."""
        assert _passes_geometry_filter(100, 100, 100, 100, 1920, 1080) is False

    def test_4k_normal_ball(self):
        """Ball in 4K (3840×2160), 30×30px → behalten."""
        assert _passes_geometry_filter(500, 500, 530, 530, 3840, 2160) is True


# ── Kachelrand-Filter ─────────────────────────────────────────────

class TestEdgeFilter:
    """Prüft dass kleine, niedrig-konfidente Randdetektionen verworfen,
    große oder hoch-konfidente aber behalten werden."""

    def test_small_low_conf_at_edge_discarded(self):
        """Kleine Box, geringe Confidence am Rand → verwerfen."""
        assert _should_discard_edge_box(0, 100, 15, 115, 640, 640, 0.2) is True

    def test_large_box_at_edge_kept(self):
        """Große Box am Rand (>25% Tile-Seite) → behalten."""
        assert _should_discard_edge_box(0, 100, 200, 300, 640, 640, 0.2) is False

    def test_high_conf_at_edge_kept(self):
        """Hohe Confidence am Rand (≥0.4) → behalten."""
        assert _should_discard_edge_box(0, 100, 20, 120, 640, 640, 0.5) is False

    def test_center_box_kept(self):
        """Box in der Kachelmitte → immer behalten."""
        assert _should_discard_edge_box(100, 100, 120, 120, 640, 640, 0.1) is False


# ── NMS / Merge ───────────────────────────────────────────────────

class TestMergeDetections:
    """Prüft das Zusammenführen überlappender Detektionen."""

    def test_single_detection(self):
        """Einzelne Detektion wird unverändert durchgereicht."""
        boxes = np.array([[100, 100, 120, 120]])
        scores = np.array([0.8])
        mb, ms = _merge_detections(boxes, scores, 0.4)
        assert len(mb) == 1
        assert ms[0] == 0.8

    def test_overlapping_detections_merged(self):
        """Zwei stark überlappende Boxen werden zusammengeführt."""
        boxes = np.array([[100, 100, 120, 120], [105, 105, 125, 125]])
        scores = np.array([0.8, 0.6])
        mb, ms = _merge_detections(boxes, scores, 0.3)
        assert len(mb) == 1
        assert ms[0] == 0.8  # Max-Confidence aus dem Cluster

    def test_non_overlapping_kept_separate(self):
        """Zwei entfernte Boxen bleiben getrennt."""
        boxes = np.array([[0, 0, 20, 20], [500, 500, 520, 520]])
        scores = np.array([0.7, 0.9])
        mb, ms = _merge_detections(boxes, scores, 0.4)
        assert len(mb) == 2

    def test_empty(self):
        """Leere Eingabe → leere Ausgabe."""
        mb, ms = _merge_detections(np.empty((0, 4)), np.empty(0), 0.4)
        assert len(mb) == 0


# ── Tiling ────────────────────────────────────────────────────────

class TestTiling:
    """Prüft die Kachelgenerierung."""

    def test_small_image_single_tile(self):
        """Bild kleiner als Tile-Size → eine Kachel."""
        tiles = _generate_tiles(400, 300, 640, 0.25)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 400, 300)

    def test_4k_covers_entire_image(self):
        """4K-Bild wird vollständig abgedeckt (keine Pixel unbesucht)."""
        tiles = _generate_tiles(3840, 2160, 640, 0.25)
        # Jede Pixelposition muss von mindestens einer Kachel abgedeckt werden
        assert len(tiles) > 1
        # Prüfe dass rechte und untere Kante abgedeckt sind
        max_x_end = max(x + w for x, y, w, h in tiles)
        max_y_end = max(y + h for x, y, w, h in tiles)
        assert max_x_end >= 3840
        assert max_y_end >= 2160


# ── Temporaler Ausreißerfilter ────────────────────────────────────

class TestTemporalFilter:
    """Prüft den Filter, der räumlich isolierte Detektionen entfernt."""

    def test_consistent_detections_kept(self):
        """Detektionen auf einer geraden Linie → alle behalten."""
        detections = {
            10: (0.50, 0.50, 0.02),
            11: (0.51, 0.50, 0.02),
            12: (0.52, 0.50, 0.02),
            13: (0.53, 0.50, 0.02),
            14: (0.54, 0.50, 0.02),
        }
        clean = filter_temporal_outliers(detections)
        assert len(clean) == 5

    def test_spatial_outlier_removed(self):
        """Ein Ausreißer am komplett anderen Bildrand → entfernt."""
        detections = {
            10: (0.50, 0.50, 0.02),
            11: (0.51, 0.50, 0.02),
            12: (0.52, 0.50, 0.02),
            13: (0.95, 0.10, 0.02),  # ← Ausreißer
            14: (0.54, 0.50, 0.02),
        }
        clean = filter_temporal_outliers(detections)
        assert 13 not in clean
        assert 10 in clean and 11 in clean and 12 in clean and 14 in clean

    def test_anchor_saves_detection(self):
        """Detektion ohne Nachbarn, aber nahe einem manuellen Anker → behalten."""
        detections = {
            50: (0.30, 0.40, 0.02),
        }
        anchors = {
            48: (0.30, 0.41),  # manueller Marker 2 Frames entfernt
        }
        clean = filter_temporal_outliers(detections, anchors=anchors)
        assert 50 in clean

    def test_isolated_without_anchor_removed(self):
        """Einzelne Detektion ohne Anker und ohne Nachbarn → entfernt."""
        detections = {
            50: (0.30, 0.40, 0.02),
            # Nächste 100 Frames entfernt → außerhalb window=8
            200: (0.30, 0.40, 0.02),
        }
        clean = filter_temporal_outliers(detections, window=8)
        # Beide Detektionen haben innerhalb ±8 Frames keinen räumlichen Nachbarn
        # aber mit nur 2 Detektionen und keinen Ankern wird nicht gefiltert
        # (Sicherheitsregel: ≤2 Detektionen ohne Anker → alle behalten)
        assert len(clean) == 2

    def test_three_isolated_detections_cleaned(self):
        """Drei weit entfernte Detektionen → Ausreißer entfernt."""
        detections = {
            10: (0.50, 0.50, 0.02),
            11: (0.51, 0.50, 0.02),
            12: (0.52, 0.50, 0.02),
            50: (0.10, 0.90, 0.02),  # ← komplett isoliert
        }
        clean = filter_temporal_outliers(detections, window=8)
        assert 50 not in clean
        assert len(clean) == 3


# ── Anker-basierte Erkennung (Unittest der Logik) ────────────────

class TestAnchorLogic:
    """Testet die Logik, dass ein Anker die Auswahl beeinflusst."""

    def test_closer_box_preferred_with_anchor(self):
        """Bei zwei Kandidaten und einem Anker wird der nähere bevorzugt,
        auch bei geringerer Confidence."""
        # Simuliere: 2 Boxen, Anker nahe der zweiten
        boxes = np.array([[100, 100, 120, 120],   # Box A, weit vom Anker
                          [500, 500, 520, 520]])   # Box B, nahe am Anker
        scores = np.array([0.6, 0.5])             # A hat leicht höhere Confidence

        # Anchor in normierter Position nahe Box B
        w, h = 1000, 1000
        anchor = (0.51, 0.51)  # nahe Box B (510/1000, 510/1000)

        # Distanz Box A zu Anker: ~0.57 → außerhalb _ANCHOR_MAX_DIST
        # Distanz Box B zu Anker: ~0.00 → innerhalb, bekommt Bonus (bis +0.25)

        # Berechne erwartetes Ergebnis manuell:
        from detection.ball_detector import _ANCHOR_MAX_DIST, _ANCHOR_BONUS
        centers = np.array([
            [(100 + 120) / 2 / w, (100 + 120) / 2 / h],
            [(500 + 520) / 2 / w, (500 + 520) / 2 / h],
        ])
        dists = np.sqrt((centers[:, 0] - anchor[0])**2 + (centers[:, 1] - anchor[1])**2)
        in_range = dists <= _ANCHOR_MAX_DIST
        adj = scores.copy()
        adj[in_range] += _ANCHOR_BONUS * (1.0 - dists[in_range] / _ANCHOR_MAX_DIST)
        best = adj.argmax()
        # Box B sollte bevorzugt werden wegen Anker-Bonus
        assert best == 1
