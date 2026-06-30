"""Exportiert Ballmarker-Daten fuer Heatmap-basiertes Balltraining.

Dieser Export ist fuer sehr kleine Baelle in 4K-Uebersichtskameras gedacht:
statt Bounding-Boxes werden Bildsequenzen und Centerpoint-Heatmaps erzeugt.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QUrl


def _load_markers(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markers = []
    for video_entry in data.get("videos", []):
        video_file = video_entry["video_file"]
        for frame_entry in video_entry.get("frames", []):
            frame_idx = int(frame_entry["frame_index"])
            for marker in frame_entry.get("markers", []):
                pos = marker["position"]
                markers.append({
                    "video_file": video_file,
                    "frame_index": frame_idx,
                    "cx": float(pos["x"]),
                    "cy": float(pos["y"]),
                    "radius": float(marker["radius"]),
                    "type": marker.get("type", "manual"),
                })
    return markers


def _video_url_to_path(video_url: str) -> str:
    if video_url.startswith("file://"):
        return QUrl(video_url).toLocalFile()
    return video_url


def _read_frame(cap: cv2.VideoCapture, frame_idx: int, fallback_shape=None):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total:
        frame_idx = max(0, min(total - 1, frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret and frame is not None:
        return frame
    if fallback_shape is not None:
        return np.zeros(fallback_shape, dtype=np.uint8)
    return None


def _crop_bounds(cx: float, cy: float, frame_w: int, frame_h: int, crop_size: int):
    crop_w = min(crop_size, frame_w)
    crop_h = min(crop_size, frame_h)
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = min(max(0, x1), max(0, frame_w - crop_w))
    y1 = min(max(0, y1), max(0, frame_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def _resize_to_square(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.shape[0] == image_size and image.shape[1] == image_size:
        return image
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _make_heatmap(size: int, cx: float | None, cy: float | None, sigma: float) -> np.ndarray:
    heatmap = np.zeros((size, size), dtype=np.float32)
    if cx is None or cy is None:
        return heatmap

    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def _marker_in_crop(marker: dict, frame_w: int, frame_h: int, x1: int, y1: int, x2: int, y2: int, margin: int = 32) -> bool:
    mx = marker["cx"] * frame_w
    my = marker["cy"] * frame_h
    return x1 - margin <= mx < x2 + margin and y1 - margin <= my < y2 + margin


def export_heatmap_dataset(
    json_path: str,
    output_dir: str,
    image_size: int = 512,
    crop_size: int = 512,
    frame_offsets: tuple[int, int, int] = (-2, 0, 2),
    val_split: float = 0.15,
    negatives_per_positive: int = 2,
    jitter_fraction: float = 0.35,
    sigma_px: float = 4.0,
    seed: int = 42,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict:
    """Erzeugt ein Heatmap-Dataset aus markierten Ballpositionen.

    Ausgabe:
      output_dir/samples/train/*.npz
      output_dir/samples/val/*.npz
      output_dir/metadata.json

    Jede NPZ-Datei enthaelt:
      frames: uint8, shape (T, H, W, 3), RGB
      heatmap: float16, shape (H, W)
      has_ball: uint8
    """
    markers = _load_markers(json_path)
    if not markers:
        raise ValueError(f"Keine Ball-Marker in {json_path} gefunden.")

    positives_by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    negatives_by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for marker in markers:
        key = (marker["video_file"], marker["frame_index"])
        if marker.get("type") == "exclusion":
            negatives_by_frame[key].append(marker)
        else:
            positives_by_frame[key].append(marker)

    keys = sorted(set(positives_by_frame.keys()) | set(negatives_by_frame.keys()))
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_count = max(1, int(len(keys) * val_split))
    val_keys = set(keys[:val_count])

    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "samples" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "positive": 0,
        "negative": 0,
        "hard_negative": 0,
        "train": 0,
        "val": 0,
        "source_frames": len(keys),
        "skipped": 0,
        "image_size": image_size,
        "crop_size": crop_size,
        "frame_offsets": list(frame_offsets),
    }

    caps: dict[str, cv2.VideoCapture] = {}
    try:
        total_steps = len(keys)
        for index, (video_url, frame_idx) in enumerate(keys, start=1):
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError("Export wurde abgebrochen.")

            video_path = _video_url_to_path(video_url)
            if progress_callback is not None:
                progress_callback(index - 1, total_steps, f"Heatmap-Samples aus Frame {frame_idx}...")

            if video_url not in caps:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    stats["skipped"] += len(positives_by_frame[(video_url, frame_idx)])
                    continue
                caps[video_url] = cap
            cap = caps[video_url]

            center_frame = _read_frame(cap, frame_idx)
            if center_frame is None:
                stats["skipped"] += len(positives_by_frame[(video_url, frame_idx)])
                continue
            frame_h, frame_w = center_frame.shape[:2]
            split = "val" if (video_url, frame_idx) in val_keys else "train"
            video_hash = hex(hash(video_url) & 0xFFFFFFFF)[2:]

            markers_on_frame = positives_by_frame[(video_url, frame_idx)]
            hard_negatives_on_frame = negatives_by_frame[(video_url, frame_idx)]
            for marker_idx, marker in enumerate(markers_on_frame):
                ball_x = marker["cx"] * frame_w
                ball_y = marker["cy"] * frame_h
                jitter = crop_size * jitter_fraction
                crop_cx = ball_x + rng.uniform(-jitter, jitter)
                crop_cy = ball_y + rng.uniform(-jitter, jitter)
                x1, y1, x2, y2 = _crop_bounds(crop_cx, crop_cy, frame_w, frame_h, crop_size)

                frames = []
                fallback_shape = center_frame.shape
                for offset in frame_offsets:
                    frame = _read_frame(cap, frame_idx + offset, fallback_shape=fallback_shape)
                    crop = frame[y1:y2, x1:x2]
                    crop = _resize_to_square(crop, image_size)
                    frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                local_x = (ball_x - x1) / max(1, x2 - x1) * image_size
                local_y = (ball_y - y1) / max(1, y2 - y1) * image_size
                if not (0 <= local_x < image_size and 0 <= local_y < image_size):
                    stats["skipped"] += 1
                    continue

                heatmap = _make_heatmap(image_size, local_x, local_y, sigma_px)
                name = f"{video_hash}_{frame_idx:06d}_{marker_idx:02d}_pos.npz"
                np.savez_compressed(
                    out / "samples" / split / name,
                    frames=np.stack(frames).astype(np.uint8),
                    heatmap=heatmap.astype(np.float16),
                    has_ball=np.uint8(1),
                )
                stats["positive"] += 1
                stats[split] += 1

            for hard_idx, marker in enumerate(hard_negatives_on_frame):
                neg_x = marker["cx"] * frame_w
                neg_y = marker["cy"] * frame_h
                x1, y1, x2, y2 = _crop_bounds(neg_x, neg_y, frame_w, frame_h, crop_size)

                frames = []
                fallback_shape = center_frame.shape
                for offset in frame_offsets:
                    frame = _read_frame(cap, frame_idx + offset, fallback_shape=fallback_shape)
                    crop = _resize_to_square(frame[y1:y2, x1:x2], image_size)
                    frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                name = f"{video_hash}_{frame_idx:06d}_{hard_idx:02d}_hardneg.npz"
                np.savez_compressed(
                    out / "samples" / split / name,
                    frames=np.stack(frames).astype(np.uint8),
                    heatmap=np.zeros((image_size, image_size), dtype=np.float16),
                    has_ball=np.uint8(0),
                )
                stats["negative"] += 1
                stats["hard_negative"] = stats.get("hard_negative", 0) + 1
                stats[split] += 1

            # Hard negatives: field crops without a marked ball.
            negative_written = 0
            attempts = 0
            while negative_written < negatives_per_positive * max(1, len(markers_on_frame)) and attempts < 100:
                attempts += 1
                crop_w = min(crop_size, frame_w)
                crop_h = min(crop_size, frame_h)
                x1 = rng.randint(0, max(0, frame_w - crop_w)) if frame_w > crop_w else 0
                y1 = rng.randint(0, max(0, frame_h - crop_h)) if frame_h > crop_h else 0
                x2 = x1 + crop_w
                y2 = y1 + crop_h
                if any(_marker_in_crop(m, frame_w, frame_h, x1, y1, x2, y2) for m in markers_on_frame):
                    continue

                frames = []
                fallback_shape = center_frame.shape
                for offset in frame_offsets:
                    frame = _read_frame(cap, frame_idx + offset, fallback_shape=fallback_shape)
                    crop = _resize_to_square(frame[y1:y2, x1:x2], image_size)
                    frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                name = f"{video_hash}_{frame_idx:06d}_{negative_written:02d}_neg.npz"
                np.savez_compressed(
                    out / "samples" / split / name,
                    frames=np.stack(frames).astype(np.uint8),
                    heatmap=np.zeros((image_size, image_size), dtype=np.float16),
                    has_ball=np.uint8(0),
                )
                stats["negative"] += 1
                stats[split] += 1
                negative_written += 1

            if progress_callback is not None:
                progress_callback(index, total_steps, f"{stats['positive']} positive, {stats['negative']} negative Samples")
    finally:
        for cap in caps.values():
            cap.release()

    metadata = dict(stats)
    metadata["version"] = 1
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exportiert Ballmarker-Daten als Heatmap-Dataset")
    parser.add_argument("json_path", help="Pfad zur ballmarker.json")
    parser.add_argument("-o", "--output", default="data/heatmap_dataset")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--negatives", type=int, default=2)
    args = parser.parse_args()

    stats = export_heatmap_dataset(
        args.json_path,
        args.output,
        image_size=args.image_size,
        crop_size=args.crop_size,
        negatives_per_positive=args.negatives,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
