"""
Modus-Definitionen und Farbpalette für die Spielfeld-Kalibrierung.

Definiert die Kalibrierungsschritte (Modi) je Kamera-Setup
sowie die Farbzuordnungen für die verschiedenen Spielfeldelemente.
"""

from typing import List, Tuple

from PySide6.QtGui import QColor


# ── Farbpalette ────────────────────────────────────────────────

COLORS = {
    "field_boundary": QColor(0, 255, 0, 200),
    "center_line": QColor(255, 255, 0, 200),
    "center_half_ellipse": QColor(255, 0, 255, 200),
    "center_ellipse": QColor(255, 0, 255, 200),
    "penalty_left": QColor(0, 165, 255, 200),
    "penalty_right": QColor(255, 165, 0, 200),
    "corner_flags": QColor(0, 255, 255, 200),
    "center_line_flags": QColor(180, 255, 100, 200),
    "active_point": QColor(255, 0, 0, 255),
}

# Modus-Tupel: (mode_name, description, min_points, max_points)  max=0 → unbegrenzt
ModeSpec = Tuple[str, str, int, int]


# ── Kamera-spezifische Modi ────────────────────────────────────

MODES_CAM0: List[ModeSpec] = [
    ("field_boundary", "SPIELFELDRAND – Sichtbare Außenlinie der LINKEN Hälfte (mind. 4). Weiter = nächster Modus", 4, 0),
    ("center_line", "MITTELLINIE – Am RECHTEN Bildrand (mind. 2 Punkte)", 2, 0),
    ("center_half_ellipse", "MITTELKREIS (linke Hälfte) – Punkte entlang des sichtbaren Bogens", 5, 0),
    ("penalty_left", "STRAFRAUM LINKS – 4 Ecken im Uhrzeigersinn", 4, 0),
    ("corner_flags", "ECKFAHNEN – Sichtbare Eckfahnen markieren (max. 2)", 1, 2),
    ("center_line_flags", "MITTELLINIENFAHNEN – Obere und untere Fahne (max. 2)", 1, 2),
]

MODES_CAM1: List[ModeSpec] = [
    ("field_boundary", "SPIELFELDRAND – Sichtbare Außenlinie der RECHTEN Hälfte (mind. 4). Weiter = nächster Modus", 4, 0),
    ("center_line", "MITTELLINIE – Am LINKEN Bildrand (mind. 2 Punkte)", 2, 0),
    ("center_half_ellipse", "MITTELKREIS (rechte Hälfte) – Punkte entlang des sichtbaren Bogens", 5, 0),
    ("penalty_right", "STRAFRAUM RECHTS – 4 Ecken im Uhrzeigersinn", 4, 0),
    ("corner_flags", "ECKFAHNEN – Sichtbare Eckfahnen markieren (max. 2)", 1, 2),
    ("center_line_flags", "MITTELLINIENFAHNEN – Obere und untere Fahne (max. 2)", 1, 2),
]

MODES_FULL: List[ModeSpec] = [
    ("field_boundary", "SPIELFELDRAND – Klicke entlang der Außenlinie (mind. 4)", 4, 0),
    ("center_line", "MITTELLINIE – Klicke entlang der Linie (mind. 2)", 2, 0),
    ("center_ellipse", "MITTELKREIS – 1) Zentrum  2) Hor. Rand  3) Vert. Rand", 3, 3),
    ("penalty_left", "STRAFRAUM LINKS – Klicke entlang der Linie", 3, 0),
    ("penalty_right", "STRAFRAUM RECHTS – Klicke entlang der Linie", 3, 0),
    ("corner_flags", "ECKFAHNEN – Alle 4 sichtbaren Eckfahnen markieren", 1, 4),
    ("center_line_flags", "MITTELLINIENFAHNEN – Obere und untere Fahne", 1, 2),
]

# Modi, die geschlossene Formen (Polygone) zeichnen
CLOSED_MODES = frozenset({"field_boundary", "penalty_left", "penalty_right"})

# Modi, die aus einzelnen Punkten bestehen (keine Verbindungslinien)
FLAG_MODES = frozenset({"corner_flags", "center_line_flags"})

# Sentinel-Modus für "alle Schritte abgeschlossen"
DONE_MODE: ModeSpec = ("done", "FERTIG – Speichern oder weitere Punkte anpassen", 0, 0)


def modes_for_camera(camera_id: int) -> List[ModeSpec]:
    """Gibt die Modus-Liste für die angegebene Kamera zurück."""
    if camera_id == 0:
        return MODES_CAM0
    elif camera_id == 1:
        return MODES_CAM1
    return MODES_FULL


def current_mode(modes: List[ModeSpec], index: int) -> ModeSpec:
    """Gibt den aktuellen Modus oder DONE_MODE zurück."""
    if index < len(modes):
        return modes[index]
    return DONE_MODE
