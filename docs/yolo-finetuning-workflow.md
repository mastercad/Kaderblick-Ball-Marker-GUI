# YOLO Fine-Tuning mit eigenen Ballmarkierungen

## Überblick

Das Standard-YOLO-Modell (`yolo11l.pt`) ist auf dem COCO-Datensatz trainiert und erkennt
"sports ball" (Klasse 32). In 4K-Aufnahmen mit 155° Weitwinkel sind Bälle in 70–90 m
Entfernung nur ca. 4–8 Pixel groß – zu klein für das generische Modell.

Die Lösung: **Bälle manuell markieren und YOLO mit diesen Daten fine-tunen**,
damit das Modell die spezifische Kamera-Perspektive und Ballgröße lernt.

---

## Voraussetzungen

- **Python-Umgebung** mit installiertem `ultralytics`-Paket (ist in `requirements.txt`)
- **GPU empfohlen** (NVIDIA mit CUDA) – Training auf CPU ist möglich, aber sehr langsam
- **Basis-Modell** `models/yolo11l.pt` muss vorhanden sein
- **Manuell markierte Bälle** im Ballmarker-GUI (je mehr Frames, desto besser)

### Wie viele Marker brauche ich?

| Marker-Anzahl | Erwartetes Ergebnis |
|---------------|---------------------|
| 50–100        | Erste Verbesserung, aber noch viele Fehldetektionen |
| 200–500       | Brauchbare Erkennung für ähnliche Aufnahmen |
| 500–1000+     | Solide Erkennung, auch bei unterschiedlichen Bedingungen |

**Tipp:** Am besten Frames aus verschiedenen Situationen markieren:
- Verschiedene Positionen auf dem Spielfeld
- Ball in Bewegung und ruhend
- Unterschiedliche Lichtverhältnisse (Sonne, Schatten, bewölkt)
- Ball auf Rasen, in der Luft, nahe an Spielern

---

## Schritt-für-Schritt-Anleitung

### 1. Bälle im GUI markieren

1. Video(s) im Ballmarker-GUI laden
2. Frame für Frame durchgehen und Bälle per Mausklick markieren
3. **Nur Ballpositionen markieren** – keine Ausschlusszonen (die werden automatisch gefiltert)
4. Per `Tab` / `Shift+Tab` kann man prüfen, ob alle Marker auf dem aktuellen Frame stimmen

### 2. Trainingsdaten exportieren

#### Variante A: Über das GUI (empfohlen)

1. **Werkzeuge → Trainingsdaten exportieren…**
2. Zielverzeichnis wählen (z.B. `data/yolo_dataset`)
3. Das Tool extrahiert automatisch:
   - Die markierten Frames als JPEG-Bilder
   - YOLO-Annotationsdateien (`.txt`) mit Ballpositionen
   - Eine `dataset.yaml` für das Training
4. Es erscheint eine Zusammenfassung mit Anzahl der Frames und Marker

#### Variante B: Über die Kommandozeile

Zuerst Marker als JSON exportieren (`Datei → Exportieren` oder `Strg+S`), dann:

```bash
# Virtual Environment aktivieren
source .venv/bin/activate

# Trainingsdaten exportieren
python -m training.export_training_data data/ballmarker.json -o data/yolo_dataset
```

**Optionale Parameter:**

| Parameter      | Default | Beschreibung |
|----------------|---------|--------------|
| `-o, --output` | `data/yolo_dataset` | Ausgabeverzeichnis |
| `--val-split`  | `0.15`  | Anteil der Frames für Validierung (15%) |
| `--box-scale`  | `2.5`   | Bounding-Box-Größe relativ zum Marker-Radius |

### 3. Dataset prüfen

Nach dem Export sollte das Verzeichnis so aussehen:

```
data/yolo_dataset/
├── dataset.yaml         ← Konfiguration für YOLO
├── images/
│   ├── train/           ← Trainingsbilder (85%)
│   │   ├── a1b2c3d4_000042.jpg
│   │   ├── a1b2c3d4_000108.jpg
│   │   └── ...
│   └── val/             ← Validierungsbilder (15%)
│       └── ...
└── labels/
    ├── train/           ← Annotationen (YOLO-Format)
    │   ├── a1b2c3d4_000042.txt
    │   └── ...
    └── val/
        └── ...
```

Jede `.txt`-Datei enthält pro Zeile eine Annotation:
```
0 0.452300 0.318700 0.008500 0.015100
```
Format: `klasse_id  cx  cy  breite  höhe` (alles normiert auf 0–1)

### 4. Training starten

```bash
# Virtual Environment aktivieren (falls noch nicht aktiv)
source .venv/bin/activate

# Training starten
python -m training.train_model data/yolo_dataset/dataset.yaml
```

**Wichtige Optionen:**

| Parameter        | Default | Beschreibung |
|------------------|---------|--------------|
| `-e, --epochs`   | `100`   | Anzahl Trainings-Epochen |
| `-b, --batch`    | `16`    | Batch-Größe (bei wenig VRAM auf 4 oder 8 reduzieren) |
| `-m, --model`    | `models/yolo11l.pt` | Anderes Basis-Modell verwenden |
| `--device`       | auto    | `0` für erste GPU, `cpu` für CPU |
| `--imgsz`        | `640`   | Eingabebildgröße |
| `-o, --output`   | `models` | Zielverzeichnis für fertiges Modell |

**Beispiele:**

```bash
# Standard-Training (GPU wird automatisch erkannt)
python -m training.train_model data/yolo_dataset/dataset.yaml

# Wenig VRAM (z.B. 4 GB): kleinere Batch-Größe
python -m training.train_model data/yolo_dataset/dataset.yaml --batch 4

# Mehr Epochen für bessere Ergebnisse
python -m training.train_model data/yolo_dataset/dataset.yaml --epochs 200

# Nur CPU (sehr langsam, aber möglich)
python -m training.train_model data/yolo_dataset/dataset.yaml --device cpu --batch 4

# Kleineres Basis-Modell (schnelleres Training, etwas weniger genau)
python -m training.train_model data/yolo_dataset/dataset.yaml -m models/yolo11s.pt
```

### 5. Training beobachten

Während des Trainings werden Logs und Grafiken gespeichert unter:
```
training/runs/ballmarker_finetune/
├── weights/
│   ├── best.pt          ← Bestes Modell (niedrigster Val-Loss)
│   └── last.pt          ← Letzter Checkpoint
├── results.csv          ← Metriken pro Epoche
├── results.png          ← Loss-Kurven als Grafik
├── confusion_matrix.png
└── ...
```

**Worauf achten:**
- **Val-Loss** sollte über die Epochen sinken und sich stabilisieren
- Wenn Val-Loss wieder steigt → **Overfitting** (weniger Epochen oder mehr Daten)
- **mAP50** (mean Average Precision) sollte möglichst hoch sein (>0.5 ist gut)

### 6. Modell verwenden

Nach erfolgreichem Training wird das beste Modell automatisch nach
`models/ballmarker_custom.pt` kopiert.

#### Automatisch beim Start

Beim nächsten Start des Ballmarker-GUI wird `ballmarker_custom.pt` automatisch
erkannt und anstelle von `yolo11l.pt` geladen. In der Konsole erscheint:
```
[YOLO] Custom-Modell geladen: .../models/ballmarker_custom.pt
```

#### Manuell ein anderes Modell laden

1. **Werkzeuge → Eigenes Modell laden…**
2. Beliebige `.pt`-Datei auswählen
3. Ab sofort nutzt die YOLO-Erkennung dieses Modell

#### Unterschied zum Standard-Modell

| Eigenschaft | Standard (yolo11l.pt) | Custom (ballmarker_custom.pt) |
|-------------|----------------------|-------------------------------|
| Klassen     | 80 COCO-Klassen      | 1 Klasse: "ball"              |
| Klassen-ID  | 32 (sports ball)     | 0 (ball)                      |
| Optimiert   | Allgemein            | Spezifisch für deine Kamera   |

Der Detektor wechselt automatisch die Klassen-ID je nach geladenem Modell.

---

## Tipps & Troubleshooting

### Training bricht ab mit "CUDA out of memory"
→ Batch-Größe reduzieren: `--batch 4` oder `--batch 2`

### Training ist extrem langsam
→ Prüfen ob GPU erkannt wird: `python -c "import torch; print(torch.cuda.is_available())"`
→ Falls `False`: CUDA/cuDNN installieren oder `--device cpu` nutzen

### Modell erkennt nichts nach dem Training
- Zu wenige Trainingsdaten → mehr Frames markieren (mind. 200+)
- Overfitting → weniger Epochen, z.B. `--epochs 50`
- Falsche Marker → prüfen ob die Marker wirklich auf dem Ball sitzen

### Modell produziert viele Fehlerkennungen
- Mehr negative Beispiele nötig (Frames ohne Ball, aber mit ähnlichen Objekten)
- Ausschlusszonen im GUI setzen, dann YOLO-Erkennung nochmal laufen lassen
- Confidence-Schwelle in der YOLO-Erkennung erhöhen

### Zurück zum Standard-Modell
- `models/ballmarker_custom.pt` löschen oder umbenennen
- Beim nächsten Start wird automatisch `yolo11l.pt` verwendet

---

## Iterativer Workflow

Das Training kann iterativ verbessert werden:

```
┌─────────────────────────────────────────────┐
│  1. Bälle manuell markieren (im GUI)        │
│                                             │
│  2. Trainingsdaten exportieren              │
│                                             │
│  3. Modell trainieren                       │
│                                             │
│  4. YOLO-Erkennung im GUI testen            │
│     (Strg+D auf einzelnen Frames)           │
│                                             │
│  5. Fehlende Bälle nachmarkieren            │
│     Falsche Detektionen korrigieren         │
│                                             │
│  └──→ Zurück zu Schritt 2                   │
└─────────────────────────────────────────────┘
```

Mit jeder Iteration wird das Modell besser, weil es mehr und
vielfältigere Trainingsdaten bekommt.

---

## Dateien & Verzeichnisse

| Pfad | Beschreibung |
|------|-------------|
| `training/export_training_data.py` | Exportiert Marker → YOLO-Dataset |
| `training/train_model.py` | Fine-Tuning-Skript |
| `models/yolo11l.pt` | Standard-YOLO-Modell (COCO) |
| `models/ballmarker_custom.pt` | Eigenes fine-tuned Modell (nach Training) |
| `data/yolo_dataset/` | Exportierte Trainingsdaten |
| `training/runs/` | Trainings-Logs und Checkpoints |
