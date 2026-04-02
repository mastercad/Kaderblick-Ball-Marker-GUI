"""Fine-Tuning eines YOLO-Modells auf manuell markierte Ballpositionen.

Voraussetzung: Trainingsdaten wurden mit export_training_data.py exportiert.

Nutzung:
    python -m training.train_model data/yolo_dataset/dataset.yaml

Optional:
    python -m training.train_model data/yolo_dataset/dataset.yaml --epochs 150 --batch 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def train(
    dataset_yaml: str,
    base_model: str | None = None,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    output_dir: str = "models",
    project: str = "training/runs",
    name: str = "ballmarker_finetune",
    device: str | None = None,
):
    """Trainiert/Fine-Tuned ein YOLO-Modell.

    Args:
        dataset_yaml: Pfad zur dataset.yaml (von export_training_data.py).
        base_model: Basis-Modell zum Fine-Tuning (default: models/yolo11l.pt).
        epochs: Anzahl Trainings-Epochen.
        batch: Batch-Größe (kleiner = weniger VRAM nötig).
        imgsz: Eingabegröße für YOLO.
        output_dir: Zielverzeichnis für das finale Modell.
        project: Verzeichnis für Trainings-Runs/Logs.
        name: Name des Trainings-Runs.
        device: CUDA-Device ('0', 'cpu', etc.). None = auto.
    """
    from ultralytics import YOLO

    # Basis-Modell bestimmen
    if base_model is None:
        model_dir = Path(__file__).resolve().parent.parent / "models"
        base_model = str(model_dir / "yolo11l.pt")
        if not os.path.isfile(base_model):
            print(f"[ERROR] Basis-Modell nicht gefunden: {base_model}")
            print("        Lade es z.B. mit: yolo detect predict model=yolo11l.pt")
            sys.exit(1)

    if not os.path.isfile(dataset_yaml):
        print(f"[ERROR] Dataset nicht gefunden: {dataset_yaml}")
        print("        Erst exportieren: python -m training.export_training_data <ballmarker.json>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"YOLO Fine-Tuning")
    print(f"  Basis-Modell:   {base_model}")
    print(f"  Dataset:        {dataset_yaml}")
    print(f"  Epochen:        {epochs}")
    print(f"  Batch-Größe:    {batch}")
    print(f"  Bildgröße:      {imgsz}")
    print(f"  Device:         {device or 'auto'}")
    print(f"{'='*60}\n")

    model = YOLO(base_model)

    # Training starten
    train_kwargs = dict(
        data=os.path.abspath(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True,
        # Fine-Tuning-Parameter: Niedrige LR damit vortrainierte Gewichte
        # nicht zerstört werden
        lr0=0.001,          # Initiale Lernrate (deutlich niedriger als default 0.01)
        lrf=0.01,           # Finale LR = lr0 × lrf
        warmup_epochs=3,    # Warmup-Phase
        # Augmentation: Farbvariationen + Flip (aber kein extremes Mosaic
        # bei kleinen Objekten)
        mosaic=0.5,         # Mosaic nur in 50% der Batches
        mixup=0.0,          # Kein Mixup (verwischt kleine Bälle)
        scale=0.3,          # Zufälliger Zoom ±30%
        fliplr=0.5,         # Horizontaler Flip
        flipud=0.0,         # Kein vertikaler Flip
        hsv_h=0.015,        # Leichte Farbvariation
        hsv_s=0.3,
        hsv_v=0.3,
        # Freeze: Die ersten N Layer einfrieren (Feature-Extraktor behalten,
        # nur den Detection-Head fine-tunen)
        freeze=10,
        # Speichern
        save=True,
        save_period=25,     # Checkpoint alle 25 Epochen
        plots=True,
    )
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)

    # Bestes Modell kopieren
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.is_file():
        os.makedirs(output_dir, exist_ok=True)
        target = Path(output_dir) / "ballmarker_custom.pt"
        import shutil
        shutil.copy2(str(best_pt), str(target))
        print(f"\n{'='*60}")
        print(f"Training abgeschlossen!")
        print(f"  Bestes Modell: {target}")
        print(f"  Runs/Logs:     {Path(project) / name}")
        print(f"\nZum Verwenden: Im Ballmarker-GUI unter")
        print(f"  Werkzeuge → Eigenes Modell laden → {target}")
        print(f"{'='*60}")
    else:
        print(f"[WARN] Bestes Modell nicht gefunden unter {best_pt}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-Tuning eines YOLO-Modells auf Ballmarker-Daten")
    parser.add_argument("dataset_yaml",
                        help="Pfad zur dataset.yaml")
    parser.add_argument("-m", "--model", default=None,
                        help="Basis-Modell (default: models/yolo11l.pt)")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Epochen (default: 100)")
    parser.add_argument("-b", "--batch", type=int, default=16,
                        help="Batch-Größe (default: 16, kleiner bei wenig VRAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Bildgröße (default: 640)")
    parser.add_argument("--device", default=None,
                        help="Device: '0' für GPU, 'cpu' für CPU")
    parser.add_argument("-o", "--output", default="models",
                        help="Zielverzeichnis für Modell (default: models/)")
    args = parser.parse_args()

    train(
        dataset_yaml=args.dataset_yaml,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
