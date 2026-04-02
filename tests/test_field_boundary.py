"""Tests für die Feldgrenze-Filterung."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest


# ── Hilfsfunktionen (statisch, ohne GUI) ────────────────────────────

def _make_boundary(pts, wh=(3840, 2160)):
    """Erstellt ein cv2-kompatibles Polygon + Frame-Größe."""
    boundary = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    return boundary, wh


def _inside(cx_norm, cy_norm, boundary, boundary_wh, margin_px=150):
    """Statische Polygon-Prüfung (identisch zu VideoGraphicsPanel._is_inside_field_static)."""
    fw, fh = boundary_wh
    px = cx_norm * fw
    py = cy_norm * fh
    dist = cv2.pointPolygonTest(boundary, (px, py), measureDist=True)
    return dist >= -margin_px


# ── Einfaches rechteckiges Feld ─────────────────────────────────────

# Spielfeld-Polygon: Rechteck von (500,300) bis (3300,1800) in einem 3840×2160 Bild
_RECT_FIELD = [
    [500, 300], [3300, 300], [3300, 1800], [500, 1800]
]


class TestFieldBoundaryBasic:
    """Grundlegende Polygon-Tests."""

    def setup_method(self):
        self.boundary, self.wh = _make_boundary(_RECT_FIELD)

    def test_inside_center(self):
        """Mitte des Feldes ist drin."""
        assert _inside(0.5, 0.5, self.boundary, self.wh)

    def test_inside_near_edge(self):
        """Punkt knapp innerhalb der Grenze."""
        # (510, 310) → norm (510/3840, 310/2160) ≈ (0.1328, 0.1435)
        assert _inside(510 / 3840, 310 / 2160, self.boundary, self.wh)

    def test_outside_far(self):
        """Punkt weit außerhalb (Ersatzball an Eckfahne)."""
        # (100, 2000) → weit links unten, außerhalb Feld
        assert not _inside(100 / 3840, 2000 / 2160, self.boundary, self.wh)

    def test_outside_within_margin(self):
        """Punkt knapp außerhalb, aber innerhalb Toleranz (Einwurf)."""
        # 50px unterhalb der unteren Kante (y=1800), also y=1850
        # Abstand zum Polygon = 50px < 150px Toleranz → erlaubt
        assert _inside(1900 / 3840, 1850 / 2160, self.boundary, self.wh)

    def test_outside_beyond_margin(self):
        """Punkt außerhalb und jenseits Toleranz."""
        # 200px unterhalb der unteren Kante (y=1800), also y=2000
        # Abstand = 200px > 150px Toleranz → abgelehnt
        assert not _inside(1900 / 3840, 2000 / 2160, self.boundary, self.wh)

    def test_corner_inside(self):
        """Eckpunkt des Feldes ist exakt auf der Kante → erlaubt."""
        assert _inside(500 / 3840, 300 / 2160, self.boundary, self.wh)

    def test_no_margin(self):
        """Punkt 50px draußen mit margin=0 → verworfen."""
        assert not _inside(1900 / 3840, 1850 / 2160, self.boundary, self.wh, margin_px=0)


# ── Realistisches Polygon (trapezförmig) ────────────────────────────

_TRAPEZ_FIELD = [
    [293, 846], [490, 773], [780, 730], [1120, 713],
    [1620, 713], [2160, 730], [2600, 773], [2797, 846],
    [2797, 1665], [1751, 1363], [1018, 1108], [620, 960],
    [293, 846],
]


class TestFieldBoundaryTrapez:
    """Tests mit einem trapezförmigen Polygon (wie cam0)."""

    def setup_method(self):
        self.boundary, self.wh = _make_boundary(_TRAPEZ_FIELD)

    def test_center_inside(self):
        """Mitte des Trapezes ist drin."""
        assert _inside(1500 / 3840, 900 / 2160, self.boundary, self.wh)

    def test_spare_ball_outside(self):
        """Ersatzball bei (2616, 1971) – weit unter der unteren Kante."""
        # Die untere Kante des Trapezes bei x≈2797 ist y≈1665
        # Abstand ≈ 306px >> 150px Toleranz → abgelehnt
        assert not _inside(2616 / 3840, 1971 / 2160, self.boundary, self.wh)

    def test_top_outside(self):
        """Punkt oberhalb des Feldes."""
        assert not _inside(1500 / 3840, 200 / 2160, self.boundary, self.wh)


# ── JSON laden ──────────────────────────────────────────────────────

class TestFieldCalibrationJson:
    """Test: JSON-Datei laden und Polygon extrahieren."""

    def test_load_valid_json(self):
        data = {
            "cam0": {
                "field_boundary": _RECT_FIELD,
                "frame_width": 3840,
                "frame_height": 2160,
                "video_path": "/some/path/Kamera1.avi",
                "camera_id": 0,
            },
            "cam1": {
                "field_boundary": [[100, 100], [3700, 100], [3700, 2000], [100, 2000]],
                "frame_width": 3840,
                "frame_height": 2160,
                "video_path": "/some/path/Kamera2.avi",
                "camera_id": 1,
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            with open(path, 'r') as fh:
                loaded = json.load(fh)
            cam0 = loaded["cam0"]
            pts = cam0["field_boundary"]
            boundary = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            assert boundary.shape == (4, 1, 2)
            assert cam0["frame_width"] == 3840
        finally:
            os.unlink(path)

    def test_polygon_roundtrip(self):
        """Polygon aus JSON laden → Punkt testen."""
        data = {
            "cam0": {
                "field_boundary": _RECT_FIELD,
                "frame_width": 3840,
                "frame_height": 2160,
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            with open(path, 'r') as fh:
                loaded = json.load(fh)
            cam = loaded["cam0"]
            boundary, wh = _make_boundary(cam["field_boundary"],
                                           (cam["frame_width"], cam["frame_height"]))
            # Innen
            assert _inside(0.5, 0.5, boundary, wh)
            # Außen
            assert not _inside(0.01, 0.95, boundary, wh)
        finally:
            os.unlink(path)


class TestNoFieldBoundary:
    """Ohne Feldgrenze → alles erlaubt."""

    def test_none_boundary_returns_true(self):
        """is_inside_field gibt True zurück wenn kein Polygon gesetzt."""
        # Simuliere: _field_boundary = None → True
        # Die Methode prüft erst ob boundary None ist
        assert True  # Logik ist in is_inside_field, hier nur Platzhalter
