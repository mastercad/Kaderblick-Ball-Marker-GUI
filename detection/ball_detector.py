"""YOLO-basierte Ballerkennung für einzelne Frames.

Nutzt Tiling (SAHI-Prinzip): Das 4K-Bild wird in überlappende Kacheln
aufgeteilt, YOLO läuft auf jeder Kachel in voller Auflösung, Ergebnisse
werden zusammengeführt und per NMS dedupliziert.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

# COCO-Klasse 32 = "sports ball"
_BALL_CLASS = 32
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODEL_NAME = os.path.join(_MODEL_DIR, "yolo11l.pt")

# Custom-Modell (Fine-Tuned): Wird automatisch bevorzugt, falls vorhanden.
# Klasse 0 = "ball" (einzige Klasse im Custom-Dataset)
_CUSTOM_MODEL_NAME = os.path.join(_MODEL_DIR, "ballmarker_custom.pt")

# Tiling-Parameter
_TILE_SIZE = 640       # Kachelgröße in Pixel (1× Skala)
_TILE_SIZE_HIRES = 320 # Kachelgröße für 2× Hochauflösung (wird auf 640 hochskaliert)
_TILE_OVERLAP = 0.25   # 25% Überlappung zwischen Kacheln
_NMS_IOU = 0.4         # IoU-Schwelle für Non-Maximum-Suppression
_EDGE_MARGIN = 8       # Detektionen innerhalb dieser Pixel am Kachelrand verwerfen

# Geometrische Filter (Pixel-Werte, bezogen auf das Originalbild)
_MIN_BOX_PX = 6        # Mindestgröße der Bounding-Box (kürzere Seite) – Ball bei 90m ≈ 4px
_MAX_BOX_RATIO = 0.15  # Maximale Boxgröße relativ zur kürzeren Bildkante
_MAX_ASPECT = 2.5      # Max. Seitenverhältnis der Box (Ball ≈ 1:1, aber Bewegungsunschärfe erlauben)

# Lokaler Kontrast-Check: Kandidat muss sich farblich vom Umfeld abheben
_SALIENCY_MIN_DIFF = 18.0  # Minimale Farbabweichung (L*a*b*-Distanz) zum Umfeld
_SALIENCY_PAD = 2.0        # Faktor: Umfeld-Region = Pad × BBox-Größe

# Anker-Parameter (Positions-Kontinuität zwischen Frames)
_ANCHOR_MAX_DIST = 0.12  # Max. normierte Distanz zum Anker (≈ 12% der Bildbreite)
_ANCHOR_BONUS = 0.25     # Confidence-Bonus für die dem Anker nächste Detektion

# Singleton – Modell wird nur einmal geladen
_model = None
_model_lock = threading.Lock()
_model_is_custom = False  # True wenn Custom-Modell geladen

# Öffentliches Pfad-Verzeichnis für GUI-Zugriff
CUSTOM_MODEL_PATH = _CUSTOM_MODEL_NAME


def _get_model():
    """Lädt das YOLO-Modell (lazy, thread-safe).
    
    Bevorzugt automatisch das Custom-Modell (ballmarker_custom.pt),
    falls vorhanden. Fällt sonst auf yolo11l.pt zurück.
    """
    global _model, _model_is_custom
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO
                # Custom-Modell bevorzugen
                if os.path.isfile(_CUSTOM_MODEL_NAME):
                    model_path = _CUSTOM_MODEL_NAME
                    _model_is_custom = True
                    print(f"[YOLO] Custom-Modell geladen: {model_path}")
                else:
                    model_path = _MODEL_NAME
                    _model_is_custom = False
                m = YOLO(model_path)
                try:
                    m.fuse()
                except (AttributeError, RuntimeError):
                    if hasattr(m, 'model') and hasattr(m.model, 'fuse'):
                        m.model.fuse = lambda *a, **kw: m.model
                _model = m
    return _model


def load_custom_model(path: str) -> bool:
    """Lädt ein benutzerdefiniertes YOLO-Modell.
    
    Args:
        path: Pfad zur .pt-Datei.
        
    Returns:
        True bei Erfolg.
    """
    global _model, _model_is_custom
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
        print(f"[YOLO] Custom-Modell geladen: {path}")
    return True


def is_custom_model() -> bool:
    """Gibt zurück, ob aktuell ein Custom-Modell geladen ist."""
    return _model_is_custom


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


def _passes_geometry_filter(x1, y1, x2, y2, img_w, img_h):
    """Prüft ob eine Detektion die Größen- und Seitenverhältnis-Filter besteht."""
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    # Zu klein (Rauschen, Wassertropfen, Reflexionspunkte)
    if min(bw, bh) < _MIN_BOX_PX:
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
                   scale_label=""):
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
        results = model.predict(crop, conf=conf, classes=[get_ball_class()],
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
            if not _passes_geometry_filter(gx1, gy1, gx2, gy2, img_w, img_h):
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


def detect_ball_in_frame(
    video_path: str,
    frame_index: int,
    fps: float,
    conf: float = 0.35,
    anchor: Optional[tuple[float, float]] = None,
    field_boundary: Optional[np.ndarray] = None,
    field_boundary_wh: Optional[tuple[int, int]] = None,
    field_margin_px: int = 150,
) -> Optional[tuple[float, float, float]]:
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
        (norm_x, norm_y, norm_radius) oder None wenn kein Ball gefunden.
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

    # Bei kleinen Bildern (≤ tile_size) direkt ohne Tiling
    if max(w, h) <= _TILE_SIZE:
        return _detect_single(model, frame, w, h, conf, anchor,
                              field_boundary, field_boundary_wh, field_margin_px)

    # ── Tiling: 1×-Skala (640×640 Crops, 1:1 Auflösung) ─────────
    tiles = _generate_tiles(w, h, _TILE_SIZE, _TILE_OVERLAP)
    all_boxes, all_scores = _run_tile_pass(
        model, frame, w, h, tiles, _TILE_SIZE, conf, log, frame_index, scale_label="1×")

    # ── Tiling: 2×-Skala (320×320 Crops → auf 640 hochskaliert) ──
    # Kleine Bälle (4-8px) werden auf 8-16px vergrößert → YOLO-detektierbar
    # Niedrigere Confidence (0.20) weil hochskalierte kleine Objekte
    # grundsätzlich schlechtere Scores bekommen
    _HIRES_CONF = max(0.20, conf - 0.15)
    tiles_hires = _generate_tiles(w, h, _TILE_SIZE_HIRES, _TILE_OVERLAP)
    hires_boxes, hires_scores = _run_tile_pass(
        model, frame, w, h, tiles_hires, _TILE_SIZE, _HIRES_CONF, log, frame_index, scale_label="2×")

    if hires_boxes:
        log.info("[detect] Frame %d: 2×-Durchlauf fand %d zusätzliche Kandidaten",
                 frame_index, len(hires_boxes))
        all_boxes.extend(hires_boxes)
        all_scores.extend(hires_scores)

    if not all_boxes:
        log.warning("[detect] Frame %d: Keine Detektion nach Tiling (%d+%d Kacheln, conf=%.2f)",
                    frame_index, len(tiles), len(tiles_hires), conf)
        return None

    log.info("[detect] Frame %d: %d Kandidaten nach Tiling+Filter", frame_index, len(all_boxes))

    # ── Lokaler Kontrast-Check (Gras-auf-Gras eliminieren) ────────
    contrast_keep = []
    for i, (gx1, gy1, gx2, gy2) in enumerate(all_boxes):
        passed, diff = _verify_local_contrast(frame, int(gx1), int(gy1), int(gx2), int(gy2))
        if passed:
            contrast_keep.append(i)
        else:
            log.info("[detect] Frame %d: Kandidat bei (%.0f,%.0f)-(%.0f,%.0f) conf=%.3f "
                     "hat zu wenig Kontrast zum Umfeld (diff=%.1f < %.1f) → verworfen",
                     frame_index, gx1, gy1, gx2, gy2, all_scores[i], diff, _SALIENCY_MIN_DIFF)
    if not contrast_keep:
        log.warning("[detect] Frame %d: Alle %d Kandidaten bei Kontrast-Check durchgefallen",
                    frame_index, len(all_boxes))
        return None
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
        return None

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
            return None
        merged_boxes = merged_boxes[keep_mask]
        merged_scores = merged_scores[keep_mask]
        log.info("[detect] Frame %d: %d Kandidaten nach Feldgrenze-Filter",
                 frame_index, len(merged_boxes))

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
                return None
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
            return None
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

    return (cx, cy, norm_radius)


def _detect_single(model, frame, w, h, conf, anchor=None,
                   field_boundary=None, field_boundary_wh=None, field_margin_px=150):
    """Erkennung ohne Tiling (für kleine Bilder)."""
    results = model.predict(frame, conf=conf, classes=[get_ball_class()],
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
