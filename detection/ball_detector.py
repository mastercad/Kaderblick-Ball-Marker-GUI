"""YOLO-basierte Ballerkennung für einzelne Frames.

Nutzt Tiling (SAHI-Prinzip): Das 4K-Bild wird in überlappende Kacheln
aufgeteilt, YOLO läuft auf jeder Kachel in voller Auflösung, Ergebnisse
werden zusammengeführt und per NMS dedupliziert.
"""

from __future__ import annotations

import os
import json
import threading
from typing import Optional

import cv2
import numpy as np

from shared.app_paths import runtime_path
from shared.python_runtime import apply_external_python_paths

# COCO-Klasse 32 = "sports ball"
_BALL_CLASS = 32
_MODEL_DIR = runtime_path("models")
_MODEL_NAME = str(_MODEL_DIR / "yolo11l.pt")

# Custom-Modell (Fine-Tuned): Wird automatisch bevorzugt, falls vorhanden.
# Klasse 0 = "ball" (einzige Klasse im Custom-Dataset)
_CUSTOM_MODEL_NAME = str(runtime_path("models", "ballmarker_custom.pt"))
_MODEL_SELECTION_PATH = runtime_path("model_selection.json")

# Tiling-Parameter
_TILE_SIZE = 640       # Kachelgröße in Pixel (1× Skala)
_TILE_SIZE_HIRES = 320 # Kachelgröße für 2× Hochauflösung (wird auf 640 hochskaliert)
_TILE_SIZE_ULTRA = 160 # Kachelgröße für 4× Hochauflösung (wird auf 640 hochskaliert)
_TILE_OVERLAP = 0.25   # 25% Überlappung zwischen Kacheln
_NMS_IOU = 0.4         # IoU-Schwelle für Non-Maximum-Suppression
_EDGE_MARGIN = 8       # Detektionen innerhalb dieser Pixel am Kachelrand verwerfen

# Geometrische Filter (Pixel-Werte, bezogen auf das Originalbild)
_MIN_BOX_PX = 3        # Mindestgröße der Bounding-Box (kürzere Seite) – Ball bei 90m ≈ 4px
_MIN_BOX_PX_HIRES = 2  # 2×-Suchpass: ultralytics skaliert Boxen auf Crop-Koordinaten zurück
_MIN_BOX_PX_ULTRA = 1  # 4×-Suchpass für extrem kleine Bälle
_MAX_BOX_RATIO = 0.35  # Erlaubt auch nahe Bälle direkt vor der Kamera
_MAX_ASPECT = 2.5      # Max. Seitenverhältnis der Box (Ball ≈ 1:1, aber Bewegungsunschärfe erlauben)

# Lokaler Kontrast-Check: Kandidat muss sich farblich vom Umfeld abheben
_SALIENCY_MIN_DIFF = 18.0  # Minimale Farbabweichung (L*a*b*-Distanz) zum Umfeld
_SALIENCY_PAD = 2.0        # Faktor: Umfeld-Region = Pad × BBox-Größe

# Anker-Parameter (Positions-Kontinuität zwischen Frames)
_ANCHOR_MAX_DIST = 0.12  # Max. normierte Distanz zum Anker (≈ 12% der Bildbreite)
_ANCHOR_BONUS = 0.25     # Confidence-Bonus für die dem Anker nächste Detektion

# Singleton – Modell wird nur einmal geladen
_model = None
_standard_fallback_model = None
_model_lock = threading.Lock()
_model_is_custom = False  # True wenn Custom-Modell geladen
_model_path = ""          # Pfad des aktuell geladenen Modells
_model_mode = ""          # auto, standard, custom

# Öffentliches Pfad-Verzeichnis für GUI-Zugriff
CUSTOM_MODEL_PATH = _CUSTOM_MODEL_NAME


def _load_model_mode() -> str:
    if _model_mode:
        return _model_mode
    try:
        with open(_MODEL_SELECTION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        mode = data.get("model_mode", "auto")
        if mode in {"auto", "standard", "custom"}:
            return mode
    except Exception:
        pass
    return "auto"


def _save_model_mode(mode: str) -> None:
    os.makedirs(_MODEL_SELECTION_PATH.parent, exist_ok=True)
    with open(_MODEL_SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump({"model_mode": mode}, f, indent=2)


def _selected_model_path() -> tuple[str, bool, str]:
    """Returns (path, is_custom, mode) for the current model selection."""
    mode = _load_model_mode()
    if mode == "standard":
        return _MODEL_NAME, False, mode
    if mode == "custom":
        if os.path.isfile(_CUSTOM_MODEL_NAME):
            return _CUSTOM_MODEL_NAME, True, mode
        return _MODEL_NAME, False, "standard"
    if os.path.isfile(_CUSTOM_MODEL_NAME):
        return _CUSTOM_MODEL_NAME, True, mode
    return _MODEL_NAME, False, mode


def _get_model():
    """Lädt das YOLO-Modell (lazy, thread-safe).
    
    Bevorzugt automatisch das Custom-Modell (ballmarker_custom.pt),
    falls vorhanden. Fällt sonst auf yolo11l.pt zurück.
    """
    global _model, _model_is_custom, _model_path, _model_mode
    apply_external_python_paths()
    if _model is None:
        with _model_lock:
            if _model is None:
                apply_external_python_paths()
                from ultralytics import YOLO
                model_path, is_custom, mode = _selected_model_path()
                _model_is_custom = is_custom
                _model_mode = mode
                if is_custom:
                    print(f"[YOLO] Custom-Modell geladen: {model_path}")
                m = YOLO(model_path)
                try:
                    m.fuse()
                except (AttributeError, RuntimeError):
                    if hasattr(m, 'model') and hasattr(m.model, 'fuse'):
                        m.model.fuse = lambda *a, **kw: m.model
                _model = m
                _model_path = model_path
    return _model


def _load_yolo_model(path: str):
    apply_external_python_paths()
    from ultralytics import YOLO

    m = YOLO(path)
    try:
        m.fuse()
    except (AttributeError, RuntimeError):
        if hasattr(m, 'model') and hasattr(m.model, 'fuse'):
            m.model.fuse = lambda *a, **kw: m.model
    return m


def _get_standard_fallback_model():
    """Lädt das Standardmodell separat für Fallbacks bei schwachen Custom-Modellen."""
    global _standard_fallback_model
    if _standard_fallback_model is None:
        with _model_lock:
            if _standard_fallback_model is None:
                _standard_fallback_model = _load_yolo_model(_MODEL_NAME)
    return _standard_fallback_model


def load_custom_model(path: str) -> bool:
    """Lädt ein benutzerdefiniertes YOLO-Modell.
    
    Args:
        path: Pfad zur .pt-Datei.
        
    Returns:
        True bei Erfolg.
    """
    global _model, _model_is_custom, _model_path, _model_mode
    apply_external_python_paths()
    with _model_lock:
        from ultralytics import YOLO
        m = YOLO(path)
        try:
            m.fuse()
        except (AttributeError, RuntimeError):
            if hasattr(m, 'model') and hasattr(m.model, 'fuse'):
                m.model.fuse = lambda *a, **kw: m.model
        _model = m
        _model_is_custom = True
        _model_path = path
        _model_mode = "custom"
        _save_model_mode("custom")
        print(f"[YOLO] Custom-Modell geladen: {path}")
    return True


def use_standard_model() -> None:
    """Wählt dauerhaft das Standard-COCO-Modell aus."""
    global _model, _model_is_custom, _model_path, _model_mode
    with _model_lock:
        _model = None
        _model_is_custom = False
        _model_path = ""
        _model_mode = "standard"
        _save_model_mode("standard")


def use_auto_model_selection() -> None:
    """Wählt automatisch Custom-Modell, falls vorhanden, sonst Standard."""
    global _model, _model_is_custom, _model_path, _model_mode
    with _model_lock:
        _model = None
        _model_is_custom = False
        _model_path = ""
        _model_mode = "auto"
        _save_model_mode("auto")


def is_custom_model() -> bool:
    """Gibt zurück, ob aktuell ein Custom-Modell geladen ist."""
    return _model_is_custom


def active_model_info(load_if_needed: bool = False) -> dict:
    """Gibt Informationen zum aktiven oder voraussichtlich genutzten Modell zurück."""
    if load_if_needed:
        _get_model()

    mode = _load_model_mode()
    if _model_path:
        path = _model_path
        is_custom = _model_is_custom
        loaded = True
    else:
        path, is_custom, mode = _selected_model_path()
        loaded = False

    return {
        "path": path,
        "name": os.path.basename(path),
        "is_custom": is_custom,
        "loaded": loaded,
        "mode": mode,
    }


def reset_loaded_model() -> None:
    """Forces YOLO to be loaded again on the next detection."""
    global _model, _model_is_custom, _model_path
    with _model_lock:
        _model = None
        _model_is_custom = False
        _model_path = ""


def runtime_status() -> dict:
    """Returns information about the active Torch runtime."""
    apply_external_python_paths()
    status = {
        "external_paths": [],
        "torch_version": "",
        "torch_file": "",
        "cuda_available": False,
        "cuda_version": "",
        "cuda_device": "",
        "error": "",
    }
    try:
        from shared.python_runtime import configured_external_package_paths

        status["external_paths"] = configured_external_package_paths()
        import torch

        status["torch_version"] = getattr(torch, "__version__", "")
        status["torch_file"] = getattr(torch, "__file__", "")
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
        if status["cuda_available"]:
            status["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        status["error"] = str(exc)
    return status


def get_ball_class() -> int:
    """Gibt die richtige Klassen-ID zurück, je nach Modell.
    
    Custom-Modell: Klasse 0 (einzige Klasse: ball)
    COCO-Modell: Klasse 32 (sports ball)
    """
    return 0 if _model_is_custom else _BALL_CLASS


def _generate_tiles(img_w: int, img_h: int, tile_size: int, overlap: float):
    """Erzeugt (x_off, y_off, w, h) Kacheln mit Überlappung."""
    step = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            tw = min(tile_size, img_w - x)
            th = min(tile_size, img_h - y)
            tiles.append((x, y, tw, th))
    return tiles


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Einfache Non-Maximum-Suppression. boxes: Nx4 (x1,y1,x2,y2)."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        mask = iou < iou_thresh
        order = rest[mask]
    return keep


def _box_touches_edge(x1, y1, x2, y2, tile_w, tile_h, margin=_EDGE_MARGIN):
    """Prüft ob eine Bounding-Box den Kachelrand berührt (= wahrscheinlich abgeschnitten)."""
    return x1 <= margin or y1 <= margin or x2 >= tile_w - margin or y2 >= tile_h - margin


def _should_discard_edge_box(bx1, by1, bx2, by2, tw, th, conf):
    """Entscheidet ob eine Rand-Detektion verworfen werden soll.

    Kleine, niedrig-konfidente Boxen am Rand sind wahrscheinlich abgeschnitten
    und werden verworfen.  Große Boxen (>25% der Kachelseite) oder solche
    mit hoher Confidence (≥0.4) werden behalten."""
    if not _box_touches_edge(bx1, by1, bx2, by2, tw, th):
        return False  # Nicht am Rand → behalten
    bw = bx2 - bx1
    bh = by2 - by1
    # Große Box → echt (Ball nah an Kamera)
    if max(bw, bh) > min(tw, th) * 0.25:
        return False
    # Hohe Confidence → echt
    if conf >= 0.4:
        return False
    # Kleine, niedrig-konfidente Box am Rand → verwerfen
    return True


def _merge_detections(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """NMS + gewichtete Mittelung: Detektionen desselben Balls werden
    per Confidence-gewichtetem Mittel zusammengeführt statt nur die
    beste Box zu nehmen.

    Returns: (merged_boxes Nx4, merged_scores N)
    """
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0)

    order = scores.argsort()[::-1]
    merged_boxes = []
    merged_scores = []

    used = np.zeros(len(boxes), dtype=bool)

    while np.any(~used[order]):
        # Nächste unbenutzste Box mit höchster Confidence
        idx = -1
        for o in order:
            if not used[o]:
                idx = o
                break
        if idx == -1:
            break
        used[idx] = True

        # Finde alle überlappenden Boxen
        cluster_indices = [idx]
        for o in order:
            if used[o]:
                continue
            # IoU berechnen
            xx1 = max(boxes[idx, 0], boxes[o, 0])
            yy1 = max(boxes[idx, 1], boxes[o, 1])
            xx2 = min(boxes[idx, 2], boxes[o, 2])
            yy2 = min(boxes[idx, 3], boxes[o, 3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            area_a = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
            area_b = (boxes[o, 2] - boxes[o, 0]) * (boxes[o, 3] - boxes[o, 1])
            iou = inter / (area_a + area_b - inter + 1e-6)
            if iou >= iou_thresh:
                cluster_indices.append(o)
                used[o] = True

        # Confidence-gewichtetes Mittel der Cluster-Boxen
        cluster_boxes = boxes[cluster_indices]
        cluster_scores = scores[cluster_indices]
        weights = cluster_scores / cluster_scores.sum()
        avg_box = (cluster_boxes * weights[:, None]).sum(axis=0)
        merged_boxes.append(avg_box)
        merged_scores.append(cluster_scores.max())

    return np.array(merged_boxes), np.array(merged_scores)


def _verify_local_contrast(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           min_diff: float = _SALIENCY_MIN_DIFF,
                           pad_factor: float = _SALIENCY_PAD) -> tuple[bool, float]:
    """Prüft ob eine Detektion sich farblich vom Umfeld abhebt.

    Vergleicht die mittlere Farbe (im L*a*b*-Raum) der Detection-Box
    mit dem umgebenden Ring.  Gras-auf-Gras hat nahezu identische Farbe
    und wird verworfen.

    Returns:
        (passed, diff) – ob der Check bestanden wurde und die gemessene Differenz.
    """
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False, 0.0

    # Innere Region (die Detektion selbst)
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(w, int(x2))
    iy2 = min(h, int(y2))
    inner = frame[iy1:iy2, ix1:ix2]
    if inner.size == 0:
        return False, 0.0

    # Äußere Region (Ring um die Detektion)
    pw = int(bw * pad_factor)
    ph = int(bh * pad_factor)
    ox1 = max(0, ix1 - pw)
    oy1 = max(0, iy1 - ph)
    ox2 = min(w, ix2 + pw)
    oy2 = min(h, iy2 + ph)
    outer_full = frame[oy1:oy2, ox1:ox2]
    if outer_full.size == 0:
        return True, 999.0  # Kein Umfeld → nicht filtern

    # Maske: nur der Ring (outer minus inner)
    mask = np.ones(outer_full.shape[:2], dtype=np.uint8)
    rel_x1 = ix1 - ox1
    rel_y1 = iy1 - oy1
    rel_x2 = rel_x1 + (ix2 - ix1)
    rel_y2 = rel_y1 + (iy2 - iy1)
    mask[rel_y1:rel_y2, rel_x1:rel_x2] = 0

    # In L*a*b* konvertieren (perceptuell gleichmäßiger)
    inner_lab = cv2.cvtColor(inner, cv2.COLOR_BGR2Lab).astype(np.float32)
    outer_lab = cv2.cvtColor(outer_full, cv2.COLOR_BGR2Lab).astype(np.float32)

    inner_mean = inner_lab.mean(axis=(0, 1))
    # Gewichtetes Mittel nur über den Ring
    ring_pixels = outer_lab[mask == 1]
    if ring_pixels.size == 0:
        return True, 999.0
    outer_mean = ring_pixels.mean(axis=0)

    # Euklidische Distanz im L*a*b*-Raum
    diff = float(np.sqrt(((inner_mean - outer_mean) ** 2).sum()))
    return diff >= min_diff, diff


def _looks_like_orange_cone(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Heuristik gegen orange Trainingshütchen als Ball-Fehlkennung."""
    h, w = frame.shape[:2]
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(w, int(x2))
    iy2 = min(h, int(y2))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    orange = ((hue >= 5) & (hue <= 28) & (sat >= 80) & (val >= 80))
    return float(orange.mean()) >= 0.25


def _field_context_score(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Anteil feldtypischer Pixel im Ring um eine Kandidatenbox.

    Weiße Stutzen/Trikotte sind oft hell und kompakt, liegen aber direkt in
    Spieler-Silhouetten. Ein Ball auf dem Feld hat dagegen meistens Rasen bzw.
    trockene Rasenfarbe im Umfeld.
    """
    h, w = frame.shape[:2]
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    pad = max(10, int(max(bw, bh) * 4))

    ox1 = max(0, int(x1) - pad)
    oy1 = max(0, int(y1) - pad)
    ox2 = min(w, int(x2) + pad)
    oy2 = min(h, int(y2) + pad)
    ring = frame[oy1:oy2, ox1:ox2]
    if ring.size == 0:
        return 0.0

    mask = np.ones(ring.shape[:2], dtype=bool)
    ix1 = max(0, int(x1) - ox1)
    iy1 = max(0, int(y1) - oy1)
    ix2 = min(mask.shape[1], int(x2) - ox1)
    iy2 = min(mask.shape[0], int(y2) - oy1)
    mask[iy1:iy2, ix1:ix2] = False

    hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Grün bis gelbgrün plus trockener gelb/brauner Rasen.
    green_grass = (hue >= 30) & (hue <= 95) & (sat >= 25) & (val >= 35)
    dry_grass = (hue >= 12) & (hue < 30) & (sat >= 25) & (val >= 45)
    field_like = (green_grass | dry_grass) & mask
    ring_pixels = int(mask.sum())
    if ring_pixels <= 0:
        return 0.0
    return float(field_like.sum() / ring_pixels)


def _looks_attached_to_player(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Heuristik für helle Kandidaten, die Teil eines Spielers sind."""
    field_score = _field_context_score(frame, x1, y1, x2, y2)
    if field_score >= 0.32:
        return False

    h, w = frame.shape[:2]
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    pad_x = max(8, bw * 3)
    pad_y = max(12, bh * 5)
    ox1 = max(0, int(x1) - pad_x)
    oy1 = max(0, int(y1) - pad_y)
    ox2 = min(w, int(x2) + pad_x)
    oy2 = min(h, int(y2) + pad_y)
    region = frame[oy1:oy2, ox1:ox2]
    if region.size == 0:
        return field_score < 0.20

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark = val < 70
    skin_or_cloth = ((hue >= 0) & (hue <= 25) & (sat >= 35) & (val >= 60)) | (sat >= 80)
    non_field_mass = float((dark | skin_or_cloth).mean())
    return non_field_mass >= 0.18


def _fallback_bright_ball_candidates(
    frame: np.ndarray,
    img_w: int,
    img_h: int,
    field_boundary: Optional[np.ndarray] = None,
    field_boundary_wh: Optional[tuple[int, int]] = None,
    field_margin_px: int = 150,
) -> tuple[list[list[float]], list[float]]:
    """Findet helle, kompakte Ballkandidaten als Fallback, wenn YOLO nichts liefert.

    Das ist bewusst konservativ: weiße/helle runde Blobs auf dem Feld, keine
    großen Trikotflächen, keine Liniensegmente, keine orangefarbenen Hütchen.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Weiße/helle Ballflächen. Linien sind auch hell, werden danach über Form
    # und Größe reduziert.
    mask = ((hsv[:, :, 2] >= 150) & (hsv[:, :, 1] <= 95)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[float]] = []
    scores: list[float] = []
    min_side = min(img_w, img_h)
    max_box = max(18, min_side * 0.04)

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 3 or bh < 3:
            continue
        if bw > max_box or bh > max_box:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 1.8:
            continue
        area = cv2.contourArea(contour)
        rect_area = bw * bh
        if rect_area <= 0:
            continue
        fill = area / rect_area
        if fill < 0.25:
            continue

        x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)
        if _looks_like_orange_cone(frame, int(x1), int(y1), int(x2), int(y2)):
            continue
        field_score = _field_context_score(frame, int(x1), int(y1), int(x2), int(y2))
        if field_score < 0.28 or _looks_attached_to_player(frame, int(x1), int(y1), int(x2), int(y2)):
            continue
        passed, diff = _verify_local_contrast(frame, int(x1), int(y1), int(x2), int(y2), min_diff=8.0)
        if not passed:
            continue

        if field_boundary is not None and field_boundary_wh is not None:
            fw, fh = field_boundary_wh
            cx_norm = ((x1 + x2) / 2) / img_w
            cy_norm = ((y1 + y2) / 2) / img_h
            px = cx_norm * fw
            py = cy_norm * fh
            dist = cv2.pointPolygonTest(field_boundary, (px, py), measureDist=True)
            if dist < -field_margin_px:
                continue

        # Score ist keine YOLO-Confidence, sondern eine plausible Sortierung.
        compactness = min(1.0, fill)
        contrast_score = min(1.0, diff / 80.0)
        scores.append(0.12 + 0.12 * compactness + 0.10 * contrast_score + 0.18 * field_score)
        boxes.append([x1, y1, x2, y2])

    if len(boxes) > 40:
        order = np.array(scores).argsort()[::-1][:40]
        boxes = [boxes[i] for i in order]
        scores = [scores[i] for i in order]

    return boxes, scores


def _passes_geometry_filter(x1, y1, x2, y2, img_w, img_h, min_box_px: int = _MIN_BOX_PX):
    """Prüft ob eine Detektion die Größen- und Seitenverhältnis-Filter besteht."""
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    # Zu klein (Rauschen, Wassertropfen, Reflexionspunkte)
    if min(bw, bh) < min_box_px:
        return False
    # Zu groß (kein Ball)
    min_side = min(img_w, img_h)
    if max(bw, bh) > min_side * _MAX_BOX_RATIO:
        return False
    # Seitenverhältnis zu weit von 1:1 (Reflexionsstreifen, Linien)
    aspect = max(bw, bh) / min(bw, bh)
    if aspect > _MAX_ASPECT:
        return False
    return True


def _run_tile_pass(model, frame, img_w, img_h, tiles, imgsz, conf, log, frame_index,
                   scale_label="", class_id: int | None = None,
                   min_box_px: int = _MIN_BOX_PX):
    """Führt einen Tiling-Durchlauf aus und gibt (boxes, scores) in Originalkoordinaten zurück.

    Bei Tiles kleiner als imgsz skaliert YOLO intern hoch → effektive Vergrößerung.
    Z.B. 320×320 Crop → imgsz=640 ergibt 2× Zoom.
    """
    boxes = []
    scores = []
    for (tx, ty, tw, th) in tiles:
        crop = frame[ty:ty+th, tx:tx+tw]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        classes = [get_ball_class() if class_id is None else class_id]
        results = model.predict(crop, conf=conf, classes=classes,
                                imgsz=imgsz, verbose=False)
        if not results or len(results[0].boxes) == 0:
            continue
        # Skalierungsfaktor: YOLO-Koordinaten beziehen sich auf das Crop,
        # nicht auf das hochskalierte Bild (ultralytics gibt Originalkoordinaten zurück)
        for box in results[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            box_conf = box.conf.item()
            # Kleine, niedrig-konfidente Detektionen am Kachelrand verwerfen
            if _should_discard_edge_box(bx1, by1, bx2, by2, tw, th, box_conf):
                continue
            # Zurück auf Original-Koordinaten
            gx1, gy1, gx2, gy2 = bx1 + tx, by1 + ty, bx2 + tx, by2 + ty
            # Geometrie-Filter (Größe + Seitenverhältnis)
            if not _passes_geometry_filter(gx1, gy1, gx2, gy2, img_w, img_h, min_box_px=min_box_px):
                continue
            boxes.append([gx1, gy1, gx2, gy2])
            scores.append(box_conf)
    if boxes:
        log.info("[detect] Frame %d [%s]: %d Kandidaten aus %d Kacheln",
                 frame_index, scale_label, len(boxes), len(tiles))
    else:
        log.info("[detect] Frame %d [%s]: 0 Kandidaten aus %d Kacheln",
                 frame_index, scale_label, len(tiles))
    return boxes, scores


def _boxes_to_results(
    boxes: np.ndarray | list,
    scores: np.ndarray | list,
    img_w: int,
    img_h: int,
    max_results: int = 8,
) -> list[tuple[float, float, float]]:
    """Konvertiert Boxen in normierte Marker-Ergebnisse, nach Score sortiert."""
    if len(boxes) == 0:
        return []
    boxes_arr = np.array(boxes)
    scores_arr = np.array(scores)
    order = scores_arr.argsort()[::-1][:max_results]
    results = []
    min_side = min(img_w, img_h)
    for idx in order:
        x1, y1, x2, y2 = boxes_arr[idx]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        radius_px = max(x2 - x1, y2 - y1) / 2
        results.append((cx, cy, radius_px / min_side))
    return results


def detect_ball_in_frame(
    video_path: str,
    frame_index: int,
    fps: float,
    conf: float = 0.20,
    anchor: Optional[tuple[float, float]] = None,
    field_boundary: Optional[np.ndarray] = None,
    field_boundary_wh: Optional[tuple[int, int]] = None,
    field_margin_px: int = 150,
    return_candidates: bool = False,
    max_candidates: int = 8,
    return_details: bool = False,
) -> Optional[tuple[float, float, float]] | list[tuple[float, float, float]]:
    """Erkennt einen Ball im gegebenen Frame per Tiling.

    Das Bild wird in überlappende 640×640-Kacheln aufgeteilt.
    YOLO läuft auf jeder Kachel in voller Auflösung.

    Args:
        anchor: Optionale (norm_x, norm_y) der letzten bekannten Ballposition.
                Wenn gesetzt, wird die dem Anker nächste Detektion bevorzugt
                und Detektionen zu weit vom Anker entfernt verworfen.
        field_boundary: Optionales cv2-Polygon (Nx1x2 int32) der Feldgrenze.
                        Kandidaten außerhalb (+margin) werden VOR der Auswahl
                        entfernt.
        field_boundary_wh: (width, height) des Kalibrierungsbildes.
        field_margin_px: Toleranz in Pixeln außerhalb der Grenze (für Einwürfe).

    Returns:
        (norm_x, norm_y, norm_radius), None wenn kein Ball gefunden, oder bei
        return_candidates=True eine Liste plausibler Kandidaten.
        Alle Werte sind auf 0..1 normiert (relativ zur Bildgröße).
    """
    import logging
    log = logging.getLogger("ball_detector")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("[detect] Video nicht geöffnet: %s", video_path)
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning("[detect] Frame %d nicht lesbar (ret=%s)", frame_index, ret)
            return None
    finally:
        cap.release()

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        log.warning("[detect] Frame hat ungültige Größe: %dx%d", w, h)
        return None

    log.info("[detect] Frame %d gelesen: %dx%d, conf=%.2f, anchor=%s",
             frame_index, w, h, conf, anchor)

    model = _get_model()
    source = "yolo"

    # Bei kleinen Bildern (≤ tile_size) direkt ohne Tiling
    if max(w, h) <= _TILE_SIZE:
        result = _detect_single(model, frame, w, h, conf, anchor,
                                field_boundary, field_boundary_wh, field_margin_px)
        if result is None and _model_is_custom:
            log.warning(
                "[detect] Frame %d: Custom-Modell fand nichts; versuche Standardmodell als Fallback",
                frame_index,
            )
            result = _detect_single(
                _get_standard_fallback_model(), frame, w, h, conf, anchor,
                field_boundary, field_boundary_wh, field_margin_px,
                class_id=_BALL_CLASS,
            )
        if result is None:
            boxes, scores = _fallback_bright_ball_candidates(
                frame, w, h, field_boundary, field_boundary_wh, field_margin_px
            )
            if boxes:
                source = "fallback"
                if return_candidates:
                    candidates = _boxes_to_results(boxes, scores, w, h, max_candidates)
                    if return_details:
                        return {"source": source, "candidates": candidates}
                    return candidates
                best = int(np.argmax(np.array(scores)))
                x1, y1, x2, y2 = boxes[best]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                radius_px = max(x2 - x1, y2 - y1) / 2
                result = (cx, cy, radius_px / min(w, h))
        if return_candidates:
            candidates = [result] if result is not None else []
            if return_details:
                return {"source": source, "candidates": candidates}
            return candidates
        if return_details:
            return {"source": source, "candidates": [result] if result is not None else []}
        return result

    # ── Tiling: 1×-Skala (640×640 Crops, 1:1 Auflösung) ─────────
    tiles = _generate_tiles(w, h, _TILE_SIZE, _TILE_OVERLAP)
    all_boxes, all_scores = _run_tile_pass(
        model, frame, w, h, tiles, _TILE_SIZE, conf, log, frame_index, scale_label="1×")

    # ── Tiling: 2×-Skala (320×320 Crops → auf 640 hochskaliert) ──
    # Kleine Bälle (4-8px) werden auf 8-16px vergrößert → YOLO-detektierbar
    # Niedrigere Confidence (0.20) weil hochskalierte kleine Objekte
    # grundsätzlich schlechtere Scores bekommen
    _HIRES_CONF = max(0.08, conf - 0.10)
    tiles_hires = _generate_tiles(w, h, _TILE_SIZE_HIRES, _TILE_OVERLAP)
    hires_boxes, hires_scores = _run_tile_pass(
        model, frame, w, h, tiles_hires, _TILE_SIZE, _HIRES_CONF, log,
        frame_index, scale_label="2×", min_box_px=_MIN_BOX_PX_HIRES)

    if hires_boxes:
        log.info("[detect] Frame %d: 2×-Durchlauf fand %d zusätzliche Kandidaten",
                 frame_index, len(hires_boxes))
        all_boxes.extend(hires_boxes)
        all_scores.extend(hires_scores)

    # ── Tiling: 4×-Skala (160×160 Crops → auf 640 hochskaliert) ──
    # Für 4K-Übersichtskameras aus ~8m Höhe. Ein 4px-Ball wird für YOLO
    # effektiv zu ~16px. Das ist deutlich langsamer, läuft deshalb nur,
    # wenn die normalen YOLO-Pässe gar nichts gefunden haben.
    tiles_ultra = None
    if not all_boxes:
        _ULTRA_CONF = max(0.03, conf - 0.15)
        tiles_ultra = _generate_tiles(w, h, _TILE_SIZE_ULTRA, _TILE_OVERLAP)
        ultra_boxes, ultra_scores = _run_tile_pass(
            model, frame, w, h, tiles_ultra, _TILE_SIZE, _ULTRA_CONF, log,
            frame_index, scale_label="4×", min_box_px=_MIN_BOX_PX_ULTRA)
        if ultra_boxes:
            log.info("[detect] Frame %d: 4×-Durchlauf fand %d Kandidaten",
                     frame_index, len(ultra_boxes))
            all_boxes.extend(ultra_boxes)
            all_scores.extend(ultra_scores)

    if not all_boxes and _model_is_custom:
        log.warning(
            "[detect] Frame %d: Custom-Modell fand nichts; versuche Standardmodell als Fallback",
            frame_index,
        )
        fallback_model = _get_standard_fallback_model()
        all_boxes, all_scores = _run_tile_pass(
            fallback_model, frame, w, h, tiles, _TILE_SIZE, conf, log,
            frame_index, scale_label="1× Standard-Fallback", class_id=_BALL_CLASS)
        hires_boxes, hires_scores = _run_tile_pass(
            fallback_model, frame, w, h, tiles_hires, _TILE_SIZE, _HIRES_CONF, log,
            frame_index, scale_label="2× Standard-Fallback", class_id=_BALL_CLASS,
            min_box_px=_MIN_BOX_PX_HIRES)
        if hires_boxes:
            all_boxes.extend(hires_boxes)
            all_scores.extend(hires_scores)
        if not all_boxes:
            _ULTRA_CONF = max(0.03, conf - 0.15)
            if tiles_ultra is None:
                tiles_ultra = _generate_tiles(w, h, _TILE_SIZE_ULTRA, _TILE_OVERLAP)
            ultra_boxes, ultra_scores = _run_tile_pass(
                fallback_model, frame, w, h, tiles_ultra, _TILE_SIZE, _ULTRA_CONF, log,
                frame_index, scale_label="4× Standard-Fallback", class_id=_BALL_CLASS,
                min_box_px=_MIN_BOX_PX_ULTRA)
            if ultra_boxes:
                all_boxes.extend(ultra_boxes)
                all_scores.extend(ultra_scores)

    if not all_boxes:
        log.warning("[detect] Frame %d: Keine YOLO-Detektion; versuche hellen Ball-Fallback",
                    frame_index)
        all_boxes, all_scores = _fallback_bright_ball_candidates(
            frame, w, h, field_boundary, field_boundary_wh, field_margin_px
        )
        source = "fallback"
        if not all_boxes:
            log.warning("[detect] Frame %d: Keine Detektion nach Tiling (%d+%d Kacheln, conf=%.2f)",
                        frame_index, len(tiles), len(tiles_hires), conf)
            if return_details:
                return {"source": "none", "candidates": []}
            return [] if return_candidates else None

    log.info("[detect] Frame %d: %d Kandidaten nach Tiling+Filter", frame_index, len(all_boxes))

    # ── Lokaler Kontrast-Check (Gras-auf-Gras eliminieren) ────────
    contrast_keep = []
    for i, (gx1, gy1, gx2, gy2) in enumerate(all_boxes):
        if _looks_like_orange_cone(frame, int(gx1), int(gy1), int(gx2), int(gy2)):
            log.info("[detect] Frame %d: Kandidat bei (%.0f,%.0f)-(%.0f,%.0f) "
                     "sieht wie orangefarbenes Hütchen aus → verworfen",
                     frame_index, gx1, gy1, gx2, gy2)
            continue
        field_score = _field_context_score(frame, int(gx1), int(gy1), int(gx2), int(gy2))
        if _looks_attached_to_player(frame, int(gx1), int(gy1), int(gx2), int(gy2)):
            log.info("[detect] Frame %d: Kandidat bei (%.0f,%.0f)-(%.0f,%.0f) "
                     "liegt zu stark in Spieler-Kontext (field_score=%.2f) → verworfen",
                     frame_index, gx1, gy1, gx2, gy2, field_score)
            continue
        passed, diff = _verify_local_contrast(frame, int(gx1), int(gy1), int(gx2), int(gy2))
        if passed:
            contrast_keep.append(i)
        else:
            log.info("[detect] Frame %d: Kandidat bei (%.0f,%.0f)-(%.0f,%.0f) conf=%.3f "
                     "hat zu wenig Kontrast zum Umfeld (diff=%.1f < %.1f) → verworfen",
                     frame_index, gx1, gy1, gx2, gy2, all_scores[i], diff, _SALIENCY_MIN_DIFF)
    if not contrast_keep:
        log.warning("[detect] Frame %d: Alle %d Kandidaten bei Farb/Kontrast-Check durchgefallen; "
                    "versuche hellen Ball-Fallback", frame_index, len(all_boxes))
        all_boxes, all_scores = _fallback_bright_ball_candidates(
            frame, w, h, field_boundary, field_boundary_wh, field_margin_px
        )
        source = "fallback"
        if not all_boxes:
            if return_details:
                return {"source": "none", "candidates": []}
            return [] if return_candidates else None
        contrast_keep = list(range(len(all_boxes)))
    if len(contrast_keep) < len(all_boxes):
        log.info("[detect] Frame %d: %d/%d Kandidaten nach Kontrast-Check",
                 frame_index, len(contrast_keep), len(all_boxes))
    all_boxes = [all_boxes[i] for i in contrast_keep]
    all_scores = [all_scores[i] for i in contrast_keep]

    boxes_arr = np.array(all_boxes)
    scores_arr = np.array(all_scores)

    # Zusammenführen: NMS + gewichtete Mittelung der Positionen
    merged_boxes, merged_scores = _merge_detections(boxes_arr, scores_arr, _NMS_IOU)
    if len(merged_boxes) == 0:
        log.warning("[detect] Frame %d: Keine Kandidaten nach NMS-Merge", frame_index)
        if return_details:
            return {"source": "none", "candidates": []}
        return [] if return_candidates else None

    log.info("[detect] Frame %d: %d Kandidaten nach Merge, scores=%s",
             frame_index, len(merged_boxes),
             [f'{s:.3f}' for s in merged_scores])

    # ── Feldgrenze-Filter (VOR Anker-Auswahl) ─────────────────────
    if field_boundary is not None and field_boundary_wh is not None:
        fw, fh = field_boundary_wh
        keep_mask = np.ones(len(merged_boxes), dtype=bool)
        for i in range(len(merged_boxes)):
            bx1, by1, bx2, by2 = merged_boxes[i]
            cx_norm = ((bx1 + bx2) / 2) / w
            cy_norm = ((by1 + by2) / 2) / h
            px = cx_norm * fw
            py = cy_norm * fh
            dist = cv2.pointPolygonTest(field_boundary, (px, py), measureDist=True)
            if dist < -field_margin_px:
                keep_mask[i] = False
                log.info("[detect] Frame %d: Kandidat %d (%.3f,%.3f) außerhalb Feldgrenze "
                         "(dist=%.0fpx, margin=%dpx) → verworfen",
                         frame_index, i, cx_norm, cy_norm, -dist, field_margin_px)
        if not np.any(keep_mask):
            log.warning("[detect] Frame %d: Alle %d Kandidaten außerhalb Feldgrenze → kein Ball",
                        frame_index, len(merged_boxes))
            if return_details:
                return {"source": "none", "candidates": []}
            return [] if return_candidates else None
        merged_boxes = merged_boxes[keep_mask]
        merged_scores = merged_scores[keep_mask]
        log.info("[detect] Frame %d: %d Kandidaten nach Feldgrenze-Filter",
                 frame_index, len(merged_boxes))

    if return_candidates and anchor is None:
        candidates = _boxes_to_results(merged_boxes, merged_scores, w, h, max_candidates)
        if return_details:
            return {"source": source, "candidates": candidates}
        return candidates

    # ── Anker-basierte Auswahl ────────────────────────────────────
    # Wenn eine vorherige Ballposition bekannt ist, bevorzuge die
    # Detektion, die am nächsten daran liegt.  Verwerfe Kandidaten,
    # die zu weit weg sind (wahrscheinlich Fehlerkennungen).

    if anchor is not None and len(merged_boxes) > 1:
        ax, ay = anchor
        # Normiertes Zentrum jeder Kandidaten-Box berechnen
        centers = np.column_stack([
            ((merged_boxes[:, 0] + merged_boxes[:, 2]) / 2) / w,
            ((merged_boxes[:, 1] + merged_boxes[:, 3]) / 2) / h,
        ])
        dists = np.sqrt((centers[:, 0] - ax) ** 2 + (centers[:, 1] - ay) ** 2)

        # Kandidaten innerhalb des erlaubten Radius
        in_range = dists <= _ANCHOR_MAX_DIST
        if np.any(in_range):
            # Unter den nahen Kandidaten: Confidence + Nähe-Bonus
            adj_scores = merged_scores.copy()
            adj_scores[in_range] += _ANCHOR_BONUS * (1.0 - dists[in_range] / _ANCHOR_MAX_DIST)
            best = adj_scores.argmax()
        else:
            # Alle zu weit weg → nimm trotzdem den besten, aber nur
            # wenn mindestens einer sehr hohe Confidence hat (>0.6)
            if merged_scores.max() >= 0.6:
                best = merged_scores.argmax()
            else:
                # Alles unplausibel → kein Ball in diesem Frame
                log.warning("[detect] Frame %d: Alle %d Kandidaten zu weit vom Anker "
                            "(%.3f,%.3f), max_conf=%.3f < 0.6 → verworfen",
                            frame_index, len(merged_boxes), ax, ay,
                            merged_scores.max())
                if return_details:
                    return {"source": "none", "candidates": []}
                return [] if return_candidates else None
    elif anchor is not None and len(merged_boxes) == 1:
        # Nur ein Kandidat: Distanz prüfen
        x1, y1, x2, y2 = merged_boxes[0]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        dist = ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5
        if dist > _ANCHOR_MAX_DIST and merged_scores[0] < 0.6:
            log.warning("[detect] Frame %d: Einziger Kandidat zu weit vom Anker "
                        "(dist=%.3f > %.3f, conf=%.3f < 0.6) → verworfen",
                        frame_index, dist, _ANCHOR_MAX_DIST, merged_scores[0])
            if return_details:
                return {"source": "none", "candidates": []}
            return [] if return_candidates else None
        best = 0
    else:
        # Kein Anker → höchste Confidence
        best = merged_scores.argmax()

    x1, y1, x2, y2 = merged_boxes[best]

    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    box_w = x2 - x1
    box_h = y2 - y1
    radius_px = max(box_w, box_h) / 2
    min_side = min(w, h)
    norm_radius = radius_px / min_side

    result = (cx, cy, norm_radius)
    if return_details:
        return {"source": source, "candidates": [result]}
    return result


def _detect_single(model, frame, w, h, conf, anchor=None,
                   field_boundary=None, field_boundary_wh=None, field_margin_px=150,
                   class_id: int | None = None):
    """Erkennung ohne Tiling (für kleine Bilder)."""
    classes = [get_ball_class() if class_id is None else class_id]
    results = model.predict(frame, conf=conf, classes=classes,
                            imgsz=_TILE_SIZE, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    # Geometrie-Filter anwenden
    valid = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        if _passes_geometry_filter(x1, y1, x2, y2, w, h):
            valid.append(i)
    if not valid:
        return None
    # Kontrast-Filter (Gras-auf-Gras eliminieren)
    contrast_valid = []
    for i in valid:
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        passed, diff = _verify_local_contrast(frame, int(x1), int(y1), int(x2), int(y2))
        if passed:
            contrast_valid.append(i)
    if not contrast_valid:
        return None
    valid = contrast_valid
    # Feldgrenze-Filter
    if field_boundary is not None and field_boundary_wh is not None:
        fw, fh = field_boundary_wh
        filtered = []
        for i in valid:
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx_norm = ((x1 + x2) / 2) / w
            cy_norm = ((y1 + y2) / 2) / h
            px = cx_norm * fw
            py = cy_norm * fh
            dist = cv2.pointPolygonTest(field_boundary, (px, py), measureDist=True)
            if dist >= -field_margin_px:
                filtered.append(i)
        if not filtered:
            return None
        valid = filtered
    # Anker-basierte Auswahl
    if anchor is not None and len(valid) > 1:
        def _score(i):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            dist = ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5
            bonus = _ANCHOR_BONUS * max(0, 1.0 - dist / _ANCHOR_MAX_DIST)
            return boxes.conf[i].item() + bonus
        best_idx = max(valid, key=_score)
    else:
        best_idx = max(valid, key=lambda i: boxes.conf[i].item())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    radius_px = max(x2 - x1, y2 - y1) / 2
    norm_radius = radius_px / min(w, h)
    return (cx, cy, norm_radius)


# ── Temporaler Konsistenz-Filter (für Batch-Erkennung) ──────────

def filter_temporal_outliers(
    detections: dict[int, tuple[float, float, float]],
    anchors: Optional[dict[int, tuple[float, float]]] = None,
    max_jump: float = 0.15,
    window: int = 8,
    min_neighbors: int = 1,
) -> dict[int, tuple[float, float, float]]:
    """Entfernt räumlich isolierte Detektionen (wahrscheinlich Fehlerkennungen).

    Eine Detektion gilt als Ausreißer, wenn sie in einem Fenster von
    ±`window` Frames keinen Nachbarn hat, dessen Position weniger als
    `max_jump` (normierte Distanz) entfernt ist.

    Manuelle Marker (über `anchors`) werden als vertrauenswürdige
    Referenzpunkte miteinbezogen, aber nie gelöscht.

    Args:
        detections: {frame_index: (norm_x, norm_y, norm_radius)}
        anchors:    {frame_index: (norm_x, norm_y)} vertrauenswürdige Positionen
                    (z.B. manuell gesetzte Marker). Werden als Nachbarn gewertet,
                    aber selbst nie entfernt.
        max_jump:   Maximale normierte euklidische Distanz zu einem Nachbarn
                    (0.15 ≈ 15% der Bildbreite pro Fenster-Spanne).
        window:     Anzahl Frames in jede Richtung für Nachbarsuche.
        min_neighbors: Mindestanzahl räumlich naher Nachbarn im Fenster.

    Returns:
        Bereinigtes dict ohne Ausreißer.
    """
    if anchors is None:
        anchors = {}

    # Alle Referenzpunkte (Detektionen + Anker) als Lookup
    all_pos: dict[int, tuple[float, float]] = {}
    for f, (x, y, _) in detections.items():
        all_pos[f] = (x, y)
    for f, (x, y) in anchors.items():
        all_pos[f] = (x, y)  # Anker überschreiben ggf. – sind vertrauenswürdiger

    if len(detections) <= 2 and not anchors:
        return detections  # Zu wenig Daten zum Filtern

    all_frames_sorted = sorted(all_pos.keys())
    detection_frames = set(detections.keys())
    keep = {}

    for f in sorted(detection_frames):
        cx, cy = all_pos[f]
        neighbors = 0

        # Binärsuche: Position im Gesamt-Array finden
        import bisect
        idx = bisect.bisect_left(all_frames_sorted, f)

        # In beide Richtungen suchen
        for direction in (-1, 1):
            j = idx + direction if direction > 0 else idx - 1
            while 0 <= j < len(all_frames_sorted):
                fn = all_frames_sorted[j]
                if abs(fn - f) > window:
                    break
                if fn != f:
                    nx, ny = all_pos[fn]
                    dist = ((cx - nx) ** 2 + (cy - ny) ** 2) ** 0.5
                    frame_gap = abs(fn - f)
                    allowed = max_jump * (frame_gap / window)
                    if dist <= allowed:
                        neighbors += 1
                        if neighbors >= min_neighbors:
                            break
                j += direction
            if neighbors >= min_neighbors:
                break

        if neighbors >= min_neighbors:
            keep[f] = detections[f]

    return keep
