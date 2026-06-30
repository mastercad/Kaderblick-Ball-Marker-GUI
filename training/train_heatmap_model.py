"""Trainiert den Heatmap-Balldetektor fuer kleine Baelle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from shared.app_paths import runtime_path
from shared.python_runtime import apply_external_python_paths


def _torch():
    apply_external_python_paths()
    import torch
    return torch


class HeatmapDataset:
    def __init__(self, root: str | os.PathLike, split: str):
        self.root = Path(root) / "samples" / split
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"Keine Heatmap-Samples gefunden: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        torch = _torch()
        data = np.load(self.files[index])
        frames = data["frames"].astype(np.float32) / 255.0
        heatmap = data["heatmap"].astype(np.float32)
        x = np.concatenate([frame for frame in frames], axis=2)
        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = torch.from_numpy(heatmap).unsqueeze(0)
        has_ball = torch.tensor(float(data["has_ball"]), dtype=torch.float32)
        return x, y, has_ball


def _loss_fn(torch, logits, target, has_ball):
    # Sparse Heatmaps: BCE stabilisiert, MSE hilft beim Peak.
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(80.0, device=logits.device)
    )
    pred = torch.sigmoid(logits)
    mse = torch.nn.functional.mse_loss(pred, target)
    # Negative Samples sollen insgesamt niedrige Heatmap-Energie haben.
    energy = pred.mean(dim=(1, 2, 3))
    neg_loss = ((1.0 - has_ball) * energy).mean()
    return bce + 2.0 * mse + 0.5 * neg_loss


def train_heatmap_model(
    dataset_dir: str,
    output_path: str | None = None,
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str | None = None,
    status_callback=None,
):
    torch = _torch()
    from torch.utils.data import DataLoader
    from detection.heatmap_ball_detector import HeatmapBallNet

    if output_path is None:
        output_path = str(runtime_path("models", "ballmarker_heatmap.pt"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = HeatmapDataset(dataset_dir, "train")
    try:
        val_ds = HeatmapDataset(dataset_dir, "val")
    except FileNotFoundError:
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_x, _sample_y, _has_ball = train_ds[0]
    model = HeatmapBallNet.create(in_channels=sample_x.shape[0]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    os.makedirs(Path(output_path).parent, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, has_ball in train_loader:
            x = x.to(device)
            y = y.to(device)
            has_ball = has_ball.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = _loss_fn(torch, logits, y, has_ball)
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * x.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, has_ball in val_loader:
                x = x.to(device)
                y = y.to(device)
                has_ball = has_ball.to(device)
                val_loss += float(_loss_fn(torch, model(x), y, has_ball).item()) * x.shape[0]
        val_loss /= len(val_ds)

        message = f"Epoche {epoch:03d}/{epochs}: train={train_loss:.5f} val={val_loss:.5f}"
        print(message)
        if status_callback is not None:
            status_callback(epoch, epochs, message)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "in_channels": sample_x.shape[0],
                "best_val_loss": best_val,
                "epochs": epoch,
            }, output_path)
            saved_message = f"Gespeichert: {output_path}"
            print(f"  {saved_message}")
            if status_callback is not None:
                status_callback(epoch, epochs, saved_message)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Trainiert den Heatmap-Balldetektor")
    parser.add_argument("dataset_dir")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-e", "--epochs", type=int, default=40)
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_heatmap_model(
        args.dataset_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
