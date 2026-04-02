"""
Feldkalibrierungs-Datenmodell und Persistenz.

Unterstützt Dual-Kamera-Setup (cam0 = linke Hälfte, cam1 = rechte Hälfte)
sowie beliebig viele Punkte pro Linie (Fischauge-Korrektur).
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("field_calibration")


@dataclass
class FieldCalibrationData:
    """Speichert manuell gesetzte Feldmarkierungen für eine Kameraseite."""

    # Spielfeldrand als Polygon (beliebig viele Punkte, im Uhrzeigersinn)
    field_boundary: List[List[int]] = field(default_factory=list)

    # Alte 4-Ecken Kompatibilität
    corners: List[List[int]] = field(default_factory=list)

    # Mittellinie (beliebig viele Punkte für gekrümmte Linie)
    center_line: List[List[int]] = field(default_factory=list)

    # Mittelkreis als Ellipse
    center_circle_center: Optional[List[int]] = None
    center_circle_horizontal: Optional[List[int]] = None
    center_circle_vertical: Optional[List[int]] = None

    # Halbellipse für Mittelkreis (Dual-Kamera)
    center_half_ellipse_points: List[List[int]] = field(default_factory=list)

    # Alte Kompatibilität
    center_circle_edge: Optional[List[int]] = None

    # Strafraum links / rechts
    penalty_area_left: List[List[int]] = field(default_factory=list)
    penalty_area_right: List[List[int]] = field(default_factory=list)

    # Torraum links / rechts (optional)
    goal_area_left: List[List[int]] = field(default_factory=list)
    goal_area_right: List[List[int]] = field(default_factory=list)

    # Eckfahnen (bis zu 4 Punkte: sichtbare Eckfahnen)
    corner_flags: List[List[int]] = field(default_factory=list)

    # Mittellinienfahnen (2 Punkte: obere und untere Seitenlinie)
    center_line_flags: List[List[int]] = field(default_factory=list)

    # Metadaten
    frame_width: int = 0
    frame_height: int = 0
    video_path: str = ""
    camera_id: int = 0  # 0 = links, 1 = rechts

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FieldCalibrationData":
        """Erstellt Instanz aus Dictionary (JSON-kompatibel)."""
        d = dict(data)
        # Listen von Listen normalisieren
        for key in [
            "corners", "field_boundary", "center_line",
            "penalty_area_left", "penalty_area_right",
            "goal_area_left", "goal_area_right",
            "center_half_ellipse_points",
            "corner_flags", "center_line_flags",
        ]:
            if key in d and d[key]:
                d[key] = [list(p) for p in d[key]]

        for key in [
            "center_circle_center", "center_circle_edge",
            "center_circle_horizontal", "center_circle_vertical",
        ]:
            if d.get(key):
                d[key] = list(d[key])

        # Unbekannte Schlüssel entfernen
        known = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in known}

        return cls(**d)

    def is_valid(self) -> bool:
        return len(self.field_boundary) >= 3 or len(self.corners) >= 4

    def get_boundary_points(self) -> List[List[int]]:
        if self.field_boundary:
            return self.field_boundary
        return self.corners

    def get_field_mask(self) -> Optional[np.ndarray]:
        boundary = self.get_boundary_points()
        if len(boundary) < 3:
            return None
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        corners_np = np.array(boundary, dtype=np.int32)
        cv2.fillPoly(mask, [corners_np], 255)
        return mask

    def get_ellipse_params(self):
        """Gibt Ellipsen-Parameter (center, (axis_h, axis_v), angle) oder None."""
        if not self.center_circle_center:
            return None
        center = tuple(self.center_circle_center)
        if self.center_circle_horizontal and self.center_circle_vertical:
            axis_h = int(np.sqrt(
                (self.center_circle_horizontal[0] - center[0]) ** 2
                + (self.center_circle_horizontal[1] - center[1]) ** 2
            ))
            axis_v = int(np.sqrt(
                (self.center_circle_vertical[0] - center[0]) ** 2
                + (self.center_circle_vertical[1] - center[1]) ** 2
            ))
            return (center, (axis_h, axis_v), 0)
        if self.center_circle_edge:
            radius = int(np.sqrt(
                (self.center_circle_edge[0] - center[0]) ** 2
                + (self.center_circle_edge[1] - center[1]) ** 2
            ))
            return (center, (radius, radius), 0)
        return None


# ── Persistenz ─────────────────────────────────────────────────


def save_calibration(data: FieldCalibrationData, output_path: str):
    """Speichert Kalibrierungsdaten (zusammen mit anderen Kameraseiten) in eine JSON-Datei."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = f"cam{data.camera_id}"

    all_data: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except Exception:
            all_data = {}

    all_data[key] = data.to_dict()
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(path))
    log.info("Kalibrierung gespeichert: Schlüssel '%s' → %s", key, output_path)


def load_calibration(input_path: str, camera_id: int = 0) -> Optional[FieldCalibrationData]:
    """Lädt Kalibrierungsdaten für eine bestimmte Kamera.

    Backward-kompatibel: alte Struktur (direkt die Daten) wird ebenfalls gelesen.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log.error("Kalibrierung nicht lesbar: %s – %s", input_path, exc)
        return None

    if isinstance(data, dict) and any(k.startswith("cam") for k in data):
        key = f"cam{camera_id}"
        if key in data:
            return FieldCalibrationData.from_dict(data[key])
        log.warning("Kein Eintrag '%s' in %s", key, input_path)
        return None

    # Alte Struktur
    return FieldCalibrationData.from_dict(data)


def load_all_calibrations(input_path: str) -> Dict[int, FieldCalibrationData]:
    """Lädt alle Kamera-Kalibrierungen aus einer Datei."""
    result: Dict[int, FieldCalibrationData] = {}
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log.error("Kalibrierung nicht lesbar: %s – %s", input_path, exc)
        return result

    if isinstance(data, dict):
        for key, val in data.items():
            if key.startswith("cam") and isinstance(val, dict):
                try:
                    cam_id = int(key.replace("cam", ""))
                    result[cam_id] = FieldCalibrationData.from_dict(val)
                except (ValueError, TypeError):
                    pass
    return result
