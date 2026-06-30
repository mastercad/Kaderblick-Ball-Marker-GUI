"""Heatmap-basierte Ball-Erkennung fuer kleine Baelle in 4K-Uebersichtsbildern."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.app_paths import runtime_path
from shared.python_runtime import apply_external_python_paths


MODEL_PATH = runtime_path("models", "ballmarker_heatmap.pt")


def _torch():
    apply_external_python_paths()
    import torch
    return torch


def _conv_block(torch, in_ch: int, out_ch: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class HeatmapBallNet:
    """Factory wrapper to avoid importing torch before external paths are applied."""

    @staticmethod
    def create(in_channels: int = 9, base_channels: int = 24):
        torch = _torch()
        nn = torch.nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = _conv_block(torch, in_channels, base_channels)
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = _conv_block(torch, base_channels, base_channels * 2)
                self.pool2 = nn.MaxPool2d(2)
                self.enc3 = _conv_block(torch, base_channels * 2, base_channels * 4)
                self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
                self.dec2 = _conv_block(torch, base_channels * 4, base_channels * 2)
                self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
                self.dec1 = _conv_block(torch, base_channels * 2, base_channels)
                self.out = nn.Conv2d(base_channels, 1, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool1(e1))
                e3 = self.enc3(self.pool2(e2))
                d2 = self.up2(e3)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))
                d1 = self.up1(d2)
                d1 = self.dec1(torch.cat([d1, e1], dim=1))
                return self.out(d1)

        return _Net()


_model = None


def reset_heatmap_model():
    global _model
    _model = None


def heatmap_model_available(path: str | os.PathLike = MODEL_PATH) -> bool:
    return Path(path).is_file()


def _load_model(path: str | os.PathLike = MODEL_PATH, device: str | None = None):
    global _model
    torch = _torch()
    if _model is not None:
        return _model
    if not Path(path).is_file():
        raise FileNotFoundError(f"Heatmap-Modell nicht gefunden: {path}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    model = HeatmapBallNet.create(in_channels=int(checkpoint.get("in_channels", 9)))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    _model = (model, device, checkpoint)
    return _model


def _read_sequence(cap: cv2.VideoCapture, frame_index: int, offsets: tuple[int, ...]):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    fallback = None
    for offset in offsets:
        idx = frame_index + offset
        if total:
            idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            if fallback is None:
                return None
            frame = np.zeros_like(fallback)
        fallback = frame
        frames.append(frame)
    return frames


def _generate_tiles(width: int, height: int, tile_size: int, overlap: float):
    step = max(1, int(tile_size * (1 - overlap)))
    tiles = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            tw = min(tile_size, width - x)
            th = min(tile_size, height - y)
            if tw < tile_size // 2 or th < tile_size // 2:
                continue
            tiles.append((x, y, tw, th))
    return tiles


def _point_in_field(cx: float, cy: float, frame_w: int, frame_h: int,
                    field_boundary: Optional[np.ndarray],
                    field_boundary_wh: Optional[tuple[int, int]],
                    margin_px: int) -> bool:
    if field_boundary is None or field_boundary_wh is None:
        return True
    fw, fh = field_boundary_wh
    px = (cx / frame_w) * fw
    py = (cy / frame_h) * fh
    dist = cv2.pointPolygonTest(field_boundary, (px, py), measureDist=True)
    return dist >= -margin_px


def detect_ball_heatmap_in_frame(
    video_path: str,
    frame_index: int,
    model_path: str | os.PathLike = MODEL_PATH,
    frame_offsets: tuple[int, int, int] = (-2, 0, 2),
    tile_size: int = 512,
    overlap: float = 0.5,
    threshold: float = 0.35,
    max_candidates: int = 5,
    field_boundary: Optional[np.ndarray] = None,
    field_boundary_wh: Optional[tuple[int, int]] = None,
    field_margin_px: int = 150,
) -> list[tuple[float, float, float, float]]:
    """Erkennt Ballzentren per Heatmap.

    Returns:
        Liste von (norm_x, norm_y, norm_radius, score), nach Score sortiert.
    """
    torch = _torch()
    model, device, _checkpoint = _load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        frames = _read_sequence(cap, frame_index, frame_offsets)
    finally:
        cap.release()
    if not frames:
        return []

    h, w = frames[len(frames) // 2].shape[:2]
    tiles = _generate_tiles(w, h, tile_size, overlap)
    detections = []

    with torch.no_grad():
        for x, y, tw, th in tiles:
            crops = []
            for frame in frames:
                crop = frame[y:y + th, x:x + tw]
                crop = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
            arr = np.concatenate(crops, axis=2).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
            logits = model(tensor)
            heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            score = float(heatmap.max())
            if score < threshold:
                continue
            py, px = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
            gx = x + (px / max(1, tile_size - 1)) * tw
            gy = y + (py / max(1, tile_size - 1)) * th
            if not _point_in_field(gx, gy, w, h, field_boundary, field_boundary_wh, field_margin_px):
                continue
            detections.append((gx / w, gy / h, 5.0 / min(w, h), score))

    detections.sort(key=lambda item: item[3], reverse=True)
    kept = []
    for det in detections:
        x, y, radius, score = det
        if any(((x - k[0]) ** 2 + (y - k[1]) ** 2) ** 0.5 < 0.015 for k in kept):
            continue
        kept.append(det)
        if len(kept) >= max_candidates:
            break
    return kept
