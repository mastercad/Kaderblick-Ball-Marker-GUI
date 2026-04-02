#!/usr/bin/env python3
"""Diagnose-Skript: Was sieht YOLO wirklich auf einem bestimmten Frame?

Leichtgewichtig — läuft auf CPU, gibt Text aus, kein GUI.
Aufruf:  python debug_detection.py
"""

import sys, gc, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # forciere CPU — spart GPU-RAM

import cv2
import numpy as np

# ── Konfiguration ────────────────────────────────────────────────
VIDEO = "/media/andreas/Seagate Expansion Drive/Videos/Fussballverein Wurgwitz/Testspiel 22.02.2026/Kamera1/aufnahme_2026-02-07_12-57-56.avi"
FRAME = 101309
BALL_CLASS = 32       # COCO "sports ball"

# ── Frame laden ──────────────────────────────────────────────────
print(f"Video: {VIDEO}")
print(f"Frame: {FRAME}\n")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit("FEHLER: Video konnte nicht geöffnet werden!")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {W}x{H}, {fps:.2f} FPS, {total} Frames")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ret, frame = cap.read()
cap.release()
if not ret or frame is None:
    sys.exit(f"FEHLER: Frame {FRAME} nicht lesbar!")

h, w = frame.shape[:2]
print(f"Frame gelesen: {w}x{h}\n")

# ── YOLO laden (CPU) ─────────────────────────────────────────────
from ultralytics import YOLO
print("Lade YOLO yolo11s.pt (CPU)…")
model = YOLO("yolo11s.pt")
print(f"Modell: {len(model.names)} Klassen, class {BALL_CLASS} = '{model.names.get(BALL_CLASS, '?')}'\n")

# ── Hilfsfunktion ─────────────────────────────────────────────────
def run_predict(img, conf, classes=None, imgsz=640):
    kw = dict(conf=conf, imgsz=imgsz, verbose=False, device="cpu")
    if classes is not None:
        kw["classes"] = classes
    res = model.predict(img, **kw)
    return res[0].boxes if res and len(res[0].boxes) > 0 else None

def fmt_box(b, ox=0, oy=0):
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    c = b.conf.item()
    cls = int(b.cls.item())
    bw, bh = x2 - x1, y2 - y1
    name = model.names.get(cls, f"cls{cls}")
    ball = " ◀ BALL" if cls == BALL_CLASS else ""
    return (f"{name:12s} conf={c:.3f}  box=({x1+ox:.0f},{y1+oy:.0f})-"
            f"({x2+ox:.0f},{y2+oy:.0f})  {bw:.0f}x{bh:.0f}{ball}")

# ═══════════════════════════════════════════════════════════════════
# TEST 1 — Vollbild, alle Klassen, conf=0.05, imgsz=1280
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Vollbild — alle Klassen, conf=0.05, imgsz=1280")
print("=" * 70)
boxes = run_predict(frame, 0.05, imgsz=1280)
if boxes:
    ball_count = sum(1 for i in range(len(boxes)) if int(boxes.cls[i].item()) == BALL_CLASS)
    print(f"  {len(boxes)} Detektionen, davon {ball_count} Ball(s):")
    for i in range(min(len(boxes), 30)):
        print(f"  [{i:2d}] {fmt_box(boxes[i])}")
else:
    print("  Keine Detektionen!")
del boxes; gc.collect()
print()

# ═══════════════════════════════════════════════════════════════════
# TEST 2 — Vollbild, nur Ball, conf=0.01, verschiedene imgsz
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Vollbild — nur Ball, conf=0.01, verschiedene imgsz")
print("=" * 70)
for imgsz in [640, 960, 1280, 1920]:
    boxes = run_predict(frame, 0.01, classes=[BALL_CLASS], imgsz=imgsz)
    if boxes:
        confs = [f"{boxes.conf[i].item():.4f}" for i in range(len(boxes))]
        print(f"  imgsz={imgsz:5d}: {len(boxes)} Ball(s), conf={confs}")
        for i in range(len(boxes)):
            print(f"    {fmt_box(boxes[i])}")
    else:
        print(f"  imgsz={imgsz:5d}: 0 Bälle")
    del boxes; gc.collect()
print()

# ═══════════════════════════════════════════════════════════════════
# TEST 3 — Tiling (640×640), nur Ball, conf=0.01
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: Tiling 640×640 — nur Ball, conf=0.01")
print("=" * 70)
from detection.ball_detector import (
    _generate_tiles, _should_discard_edge_box, _passes_geometry_filter,
    _MIN_BOX_PX, _MAX_BOX_RATIO, _MAX_ASPECT,
)
tiles = _generate_tiles(w, h, 640, 0.25)
print(f"  {len(tiles)} Kacheln")

found = 0
for tx, ty, tw, th in tiles:
    crop = frame[ty:ty+th, tx:tx+tw]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        continue
    boxes = run_predict(crop, 0.01, classes=[BALL_CLASS], imgsz=640)
    if not boxes:
        continue
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = boxes[i].xyxy[0].tolist()
        conf_val = boxes[i].conf.item()
        gx1, gy1 = bx1 + tx, by1 + ty
        gx2, gy2 = bx2 + tx, by2 + ty
        bw, bh = bx2 - bx1, by2 - by1

        edge = _should_discard_edge_box(bx1, by1, bx2, by2, tw, th, conf_val)
        geom = _passes_geometry_filter(gx1, gy1, gx2, gy2, w, h)
        tag = "[EDGE-FILTERED]" if edge else ("[GEOM-FILTERED]" if not geom else "[PASS ✓]")

        found += 1
        print(f"  Tile({tx},{ty}): conf={conf_val:.4f}  "
              f"local=({bx1:.0f},{by1:.0f})-({bx2:.0f},{by2:.0f})  "
              f"global=({gx1:.0f},{gy1:.0f})-({gx2:.0f},{gy2:.0f})  "
              f"{bw:.0f}x{bh:.0f} {tag}")
    del boxes; gc.collect()

if found == 0:
    print("  Keine Ball-Detektionen in irgendeiner Kachel!")
print()

# ═══════════════════════════════════════════════════════════════════
# TEST 4 — Manueller Crop um bekannte Ball-Position
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: Manueller Crop 640×640 um Bildmitte — alle Klassen, conf=0.01")
print("=" * 70)
cx, cy = w // 2, h // 2
x1c = max(0, cx - 320)
y1c = max(0, cy - 320)
crop_center = frame[y1c:y1c+640, x1c:x1c+640]
boxes = run_predict(crop_center, 0.01, imgsz=640)
if boxes:
    print(f"  {len(boxes)} Detektionen im Zentrum-Crop:")
    for i in range(min(len(boxes), 15)):
        print(f"  [{i:2d}] {fmt_box(boxes[i], x1c, y1c)}")
else:
    print("  Keine Detektionen im Zentrum-Crop!")
del boxes; gc.collect()
print()

# ═══════════════════════════════════════════════════════════════════
# TEST 5 — Filter-Schwellwerte anzeigen
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"Aktuelle Filter-Schwellwerte:")
print(f"  _MIN_BOX_PX    = {_MIN_BOX_PX}")
print(f"  _MAX_BOX_RATIO = {_MAX_BOX_RATIO}  → max {_MAX_BOX_RATIO * min(w,h):.0f} px bei {min(w,h)} min_side")
print(f"  _MAX_ASPECT    = {_MAX_ASPECT}")
print("=" * 70)

print("\nFertig.")
