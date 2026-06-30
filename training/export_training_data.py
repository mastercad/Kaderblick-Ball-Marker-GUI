"""Exportiert manuell markierte Bälle als YOLO-Trainingsdaten.

Liest die Marker aus einer ballmarker.json-Exportdatei, extrahiert
ballzentrierte Ausschnitte aus den Videos und erzeugt ein YOLO-kompatibles
Dataset mit images/, labels/ und dataset.yaml. Die Ausschnitte bewahren kleine
Bälle in ihrer Original-Pixelgröße, statt ganze 4K-Frames auf YOLO-Format
herunterzurechnen.

YOLO-Annotationsformat (pro Zeile):
    class_id  cx  cy  w  h
Alle Werte sind auf 0..1 normiert (relativ zur Bildgröße).
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Callable
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


def _crop_bounds(cx_px: float, cy_px: float, width: int, height: int, crop_size: int):
    """Berechnet einen möglichst zentrierten Bildausschnitt um einen Marker."""
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    x1 = int(round(cx_px - crop_w / 2))
    y1 = int(round(cy_px - crop_h / 2))
    x1 = min(max(0, x1), max(0, width - crop_w))
    y1 = min(max(0, y1), max(0, height - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def _marker_in_crop(marker: dict, frame_w: int, frame_h: int,
                    crop_x1: int, crop_y1: int, crop_x2: int, crop_y2: int,
                    margin_px: int = 24) -> bool:
    marker_x = marker["cx"] * frame_w
    marker_y = marker["cy"] * frame_h
    return (
        crop_x1 - margin_px <= marker_x < crop_x2 + margin_px
        and crop_y1 - margin_px <= marker_y < crop_y2 + margin_px
    )


def _marker_label_in_crop(
    marker: dict,
    frame_w: int,
    frame_h: int,
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    box_scale: float,
) -> str | None:
    """Erzeugt eine YOLO-Zeile für einen Marker, falls er im Ausschnitt liegt."""
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    marker_x = marker["cx"] * frame_w
    marker_y = marker["cy"] * frame_h

    if not (crop_x1 <= marker_x < crop_x2 and crop_y1 <= marker_y < crop_y2):
        return None

    min_side = min(frame_w, frame_h)
    radius_px = marker["radius"] * min_side
    box_px = max(2.0, radius_px * box_scale * 2)
    half_box = box_px / 2

    box_x1 = max(crop_x1, marker_x - half_box)
    box_y1 = max(crop_y1, marker_y - half_box)
    box_x2 = min(crop_x2, marker_x + half_box)
    box_y2 = min(crop_y2, marker_y + half_box)
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return None

    cx = ((box_x1 + box_x2) / 2 - crop_x1) / crop_w
    cy = ((box_y1 + box_y2) / 2 - crop_y1) / crop_h
    bw = (box_x2 - box_x1) / crop_w
    bh = (box_y2 - box_y1) / crop_h

    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    bw = min(1.0, max(0.005, bw))
    bh = min(1.0, max(0.005, bh))
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export_yolo_dataset(
    json_path: str,
    output_dir: str,
    val_split: float = 0.15,
    box_scale: float = 2.5,
    crop_size: int = 640,
    crop_sizes: Sequence[int] | None = None,
    negative_crops_per_frame: int = 2,
    seed: int = 42,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict:
    """Exportiert Marker als YOLO-Trainingsdaten.

    Args:
        json_path: Pfad zur ballmarker.json-Exportdatei.
        output_dir: Zielverzeichnis für das Dataset.
        val_split: Anteil der Frames für Validierung (0.0 .. 0.5).
        box_scale: Faktor für die BBox-Größe relativ zum Marker-Radius.
                   2.5 = BBox ist 2.5× so groß wie der Marker-Durchmesser
                   (gibt YOLO etwas Kontext um den Ball).
        crop_size: Größe der ballzentrierten Trainingsausschnitte in Pixeln.
                   640 bewahrt kleine Bälle in Originalgröße und passt zur
                   YOLO-Standard-Trainingsgröße.
        crop_sizes: Mehrere Ausschnittgrößen. Standard: 640, 320 und 160,
                    passend zur späteren 1×-/2×-/4×-Kachelsuche.
        negative_crops_per_frame: Zusätzliche Ausschnitte ohne Ball je Quellframe.
                                  Hilft gegen Fehlalarme auf Linien, Schuhen,
                                  Reflexionen und Grasstrukturen.
        seed: Random-Seed für reproduzierbare Train/Val-Aufteilung.
        progress_callback: Optionaler Callback für GUI/CLI-Fortschritt:
                           (aktueller_quellframe, gesamt, detailtext).
        cancel_callback: Optionaler Callback; True bricht den Export ab.

    Returns:
        Dict mit Statistiken: {total_frames, total_markers, train, val, skipped}
    """
    markers = _load_markers(json_path)
    if not markers:
        raise ValueError(f"Keine Ball-Marker in {json_path} gefunden.")

    if crop_sizes is None:
        crop_sizes = (crop_size, 320, 160)
    crop_sizes = tuple(sorted({int(s) for s in crop_sizes if int(s) > 0}, reverse=True))
    if not crop_sizes:
        raise ValueError("Mindestens eine gültige Ausschnittgröße ist erforderlich.")

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
    rng = random.Random(seed)
    rng.shuffle(frame_keys)
    n_val = max(1, int(len(frame_keys) * val_split))
    val_keys = set(frame_keys[:n_val])

    stats = {
        "total_frames": 0,
        "total_markers": 0,
        "positive_crops": 0,
        "negative_crops": 0,
        "source_frames": len(frame_keys),
        "crop_sizes": list(crop_sizes),
        "train": 0,
        "val": 0,
        "skipped": 0,
    }
    video_caps: dict[str, cv2.VideoCapture] = {}
    total_source_frames = len(frame_keys)

    try:
        for current_index, (video_url, frame_idx) in enumerate(frame_keys, start=1):
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError("Export wurde abgebrochen.")

            video_path = _video_url_to_path(video_url)
            if progress_callback is not None:
                progress_callback(
                    current_index - 1,
                    total_source_frames,
                    f"Frame {frame_idx} wird aus {os.path.basename(video_path)} gelesen...",
                )

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

            frame_markers = by_frame[(video_url, frame_idx)]
            if progress_callback is not None:
                progress_callback(
                    current_index - 1,
                    total_source_frames,
                    f"Erzeuge Ausschnitte für Frame {frame_idx} ({len(frame_markers)} Marker)...",
                )

            # Eindeutiger Dateiname: video-hash_frame_marker_crop.jpg
            video_hash = hex(hash(video_url) & 0xFFFFFFFF)[2:]
            for crop_size_value in crop_sizes:
                for marker_idx, marker in enumerate(frame_markers):
                    marker_x = marker["cx"] * w
                    marker_y = marker["cy"] * h
                    crop_x1, crop_y1, crop_x2, crop_y2 = _crop_bounds(
                        marker_x, marker_y, w, h, crop_size_value
                    )
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size == 0:
                        stats["skipped"] += 1
                        continue

                    lines = []
                    for label_marker in frame_markers:
                        line = _marker_label_in_crop(
                            label_marker,
                            w,
                            h,
                            crop_x1,
                            crop_y1,
                            crop_x2,
                            crop_y2,
                            box_scale,
                        )
                        if line is not None:
                            lines.append(line)

                    if not lines:
                        stats["skipped"] += 1
                        continue

                    img_name = f"{video_hash}_{frame_idx:06d}_{marker_idx:02d}_{crop_size_value}.jpg"
                    img_path = out / "images" / split / img_name
                    cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    label_path = out / "labels" / split / img_name.replace(".jpg", ".txt")
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")

                    stats["total_frames"] += 1
                    stats["positive_crops"] += 1
                    stats["total_markers"] += len(lines)
                    stats[split] += 1

                # Hintergrund-Ausschnitte ohne Ball: wichtig gegen Fehlalarme.
                negative_written = 0
                attempts = 0
                while negative_written < negative_crops_per_frame and attempts < negative_crops_per_frame * 20:
                    attempts += 1
                    crop_w = min(crop_size_value, w)
                    crop_h = min(crop_size_value, h)
                    if w == crop_w:
                        x1 = 0
                    else:
                        x1 = rng.randint(0, w - crop_w)
                    if h == crop_h:
                        y1 = 0
                    else:
                        y1 = rng.randint(0, h - crop_h)
                    x2 = x1 + crop_w
                    y2 = y1 + crop_h
                    if any(_marker_in_crop(m, w, h, x1, y1, x2, y2) for m in frame_markers):
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    img_name = f"{video_hash}_{frame_idx:06d}_neg{negative_written:02d}_{crop_size_value}.jpg"
                    img_path = out / "images" / split / img_name
                    cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    label_path = out / "labels" / split / img_name.replace(".jpg", ".txt")
                    with open(label_path, "w", encoding="utf-8"):
                        pass

                    stats["total_frames"] += 1
                    stats["negative_crops"] += 1
                    stats[split] += 1
                    negative_written += 1

            if progress_callback is not None:
                progress_callback(
                    current_index,
                    total_source_frames,
                    f"{stats['total_frames']} Ausschnitte erzeugt",
                )

    finally:
        for cap in video_caps.values():
            cap.release()

    # dataset.yaml schreiben
    yaml_path = out / "dataset.yaml"
    yaml_content = f"""# Ballmarker YOLO Training Dataset
# Automatisch generiert aus {os.path.basename(json_path)}
# Trainingsbilder sind ballzentrierte Ausschnitte in Originalauflösung.
# Größen: {", ".join(str(s) for s in crop_sizes)} px

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
    print(f"  Ausschnitte:    {stats['total_frames']}")
    print(f"  Positiv:        {stats['positive_crops']} Ausschnitte")
    print(f"  Hintergrund:    {stats['negative_crops']} Ausschnitte")
    print(f"  Quell-Frames:   {stats['source_frames']}")
    print(f"  Marker/Labels:  {stats['total_markers']}")
    print(f"  Train:          {stats['train']} Ausschnitte")
    print(f"  Val:            {stats['val']} Ausschnitte")
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
    parser.add_argument("--crop-size", type=int, default=640,
                        help="Größe der ballzentrierten Ausschnitte (default: 640)")
    parser.add_argument("--no-negative-crops", action="store_true",
                        help="Keine zusätzlichen Hintergrund-Ausschnitte erzeugen")
    args = parser.parse_args()

    export_yolo_dataset(args.json_path, args.output,
                        val_split=args.val_split, box_scale=args.box_scale,
                        crop_size=args.crop_size,
                        negative_crops_per_frame=0 if args.no_negative_crops else 2)


if __name__ == "__main__":
    main()
