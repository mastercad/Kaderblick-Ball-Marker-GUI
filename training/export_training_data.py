"""Exportiert manuell markierte Bälle als YOLO-Trainingsdaten.

Liest die Marker aus einer ballmarker.json-Exportdatei, extrahiert die
entsprechenden Frames aus den Videos und erzeugt ein YOLO-kompatibles
Dataset mit images/, labels/ und dataset.yaml.

YOLO-Annotationsformat (pro Zeile):
    class_id  cx  cy  w  h
Alle Werte sind auf 0..1 normiert (relativ zur Bildgröße).
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from PySide6.QtCore import QUrl


def _load_markers(json_path: str) -> list[dict]:
    """Lädt Marker aus einer ballmarker.json-Exportdatei."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markers = []
    for video_entry in data.get("videos", []):
        video_file = video_entry["video_file"]
        for frame_entry in video_entry.get("frames", []):
            frame_idx = frame_entry["frame_index"]
            for m in frame_entry.get("markers", []):
                marker_type = m.get("type", "manual")
                # Nur Ball-Marker (keine Ausschlusszonen)
                if marker_type == "exclusion":
                    continue
                pos = m["position"]
                markers.append({
                    "video_file": video_file,
                    "frame_index": frame_idx,
                    "cx": pos["x"],
                    "cy": pos["y"],
                    "radius": m["radius"],
                    "type": marker_type,
                })
    return markers


def _video_url_to_path(video_url: str) -> str:
    """Konvertiert eine file://-URL in einen lokalen Pfad."""
    if video_url.startswith("file://"):
        return QUrl(video_url).toLocalFile()
    return video_url


def export_yolo_dataset(
    json_path: str,
    output_dir: str,
    val_split: float = 0.15,
    box_scale: float = 2.5,
    seed: int = 42,
) -> dict:
    """Exportiert Marker als YOLO-Trainingsdaten.

    Args:
        json_path: Pfad zur ballmarker.json-Exportdatei.
        output_dir: Zielverzeichnis für das Dataset.
        val_split: Anteil der Frames für Validierung (0.0 .. 0.5).
        box_scale: Faktor für die BBox-Größe relativ zum Marker-Radius.
                   2.5 = BBox ist 2.5× so groß wie der Marker-Durchmesser
                   (gibt YOLO etwas Kontext um den Ball).
        seed: Random-Seed für reproduzierbare Train/Val-Aufteilung.

    Returns:
        Dict mit Statistiken: {total_frames, total_markers, train, val, skipped}
    """
    markers = _load_markers(json_path)
    if not markers:
        raise ValueError(f"Keine Ball-Marker in {json_path} gefunden.")

    # Gruppiere nach (video, frame)
    by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for m in markers:
        key = (m["video_file"], m["frame_index"])
        by_frame[key].append(m)

    # Verzeichnisstruktur anlegen
    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Train/Val-Split (Frame-basiert, nicht Marker-basiert)
    frame_keys = sorted(by_frame.keys())
    random.seed(seed)
    random.shuffle(frame_keys)
    n_val = max(1, int(len(frame_keys) * val_split))
    val_keys = set(frame_keys[:n_val])

    stats = {"total_frames": 0, "total_markers": 0, "train": 0, "val": 0, "skipped": 0}
    video_caps: dict[str, cv2.VideoCapture] = {}

    try:
        for video_url, frame_idx in frame_keys:
            video_path = _video_url_to_path(video_url)

            # Video öffnen (einmal pro Video, gecacht)
            if video_url not in video_caps:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"[WARN] Video nicht öffenbar: {video_path}")
                    stats["skipped"] += len(by_frame[(video_url, frame_idx)])
                    continue
                video_caps[video_url] = cap

            cap = video_caps[video_url]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[WARN] Frame {frame_idx} nicht lesbar aus {video_path}")
                stats["skipped"] += len(by_frame[(video_url, frame_idx)])
                continue

            h, w = frame.shape[:2]
            split = "val" if (video_url, frame_idx) in val_keys else "train"

            # Eindeutiger Dateiname: video-hash_frame.jpg
            video_hash = hex(hash(video_url) & 0xFFFFFFFF)[2:]
            img_name = f"{video_hash}_{frame_idx:06d}.jpg"

            # Bild speichern
            img_path = out / "images" / split / img_name
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # YOLO-Annotation schreiben
            label_path = out / "labels" / split / img_name.replace(".jpg", ".txt")
            frame_markers = by_frame[(video_url, frame_idx)]
            lines = []
            for m in frame_markers:
                cx = m["cx"]
                cy = m["cy"]
                # BBox-Größe: Marker-Radius × box_scale → Durchmesser
                # radius ist normiert auf min(w,h), wir brauchen es relativ zu w und h
                min_side = min(w, h)
                radius_px = m["radius"] * min_side
                box_px = radius_px * box_scale * 2  # Durchmesser × Skala
                bw = box_px / w  # normiert auf Bildbreite
                bh = box_px / h  # normiert auf Bildhöhe
                # Clamp
                bw = min(1.0, max(0.005, bw))
                bh = min(1.0, max(0.005, bh))
                # class_id = 0 (einzige Klasse: ball)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats["total_markers"] += 1

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            stats["total_frames"] += 1
            stats[split] += 1

    finally:
        for cap in video_caps.values():
            cap.release()

    # dataset.yaml schreiben
    yaml_path = out / "dataset.yaml"
    yaml_content = f"""# Ballmarker YOLO Training Dataset
# Automatisch generiert aus {os.path.basename(json_path)}
#
# Trainiere mit:
#   yolo detect train data={yaml_path} model=yolo11l.pt epochs=100 imgsz=640

path: {out.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: ball
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Zusammenfassung
    print(f"\n{'='*60}")
    print(f"YOLO-Dataset exportiert nach: {out.resolve()}")
    print(f"  Frames gesamt:  {stats['total_frames']}")
    print(f"  Marker gesamt:  {stats['total_markers']}")
    print(f"  Train:          {stats['train']} Frames")
    print(f"  Val:            {stats['val']} Frames")
    print(f"  Übersprungen:   {stats['skipped']} Marker")
    print(f"  dataset.yaml:   {yaml_path}")
    print(f"{'='*60}")

    return stats


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Exportiert ballmarker.json → YOLO-Trainingsdaten")
    parser.add_argument("json_path", help="Pfad zur ballmarker.json")
    parser.add_argument("-o", "--output", default="data/yolo_dataset",
                        help="Ausgabeverzeichnis (default: data/yolo_dataset)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validierungs-Anteil (default: 0.15)")
    parser.add_argument("--box-scale", type=float, default=2.5,
                        help="BBox-Skalierung relativ zum Marker (default: 2.5)")
    args = parser.parse_args()

    export_yolo_dataset(args.json_path, args.output,
                        val_split=args.val_split, box_scale=args.box_scale)


if __name__ == "__main__":
    main()
