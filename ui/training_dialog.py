"""GUI dialog for exporting a YOLO dataset and starting model training."""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from shared.app_paths import runtime_path


class _TrainingWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, options: dict):
        super().__init__()
        self._options = options

    def run(self):
        try:
            from training.train_model import train

            self.status.emit("Training wird vorbereitet...")
            result = train(**self._options)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingDialog(QDialog):
    """Starts YOLO fine-tuning from inside the desktop application."""

    _TOOLTIPS = {
        "dataset": (
            "Hier liegen die aus den Markierungen erzeugten Trainingsbilder. "
            "Dieser Pfad wird automatisch gesetzt."
        ),
        "base_model": (
            "Das vorhandene YOLO-Modell, das als Grundlage verbessert wird. "
            "Normalerweise bitte unverändert lassen."
        ),
        "epochs": (
            "Wie oft das Programm alle Trainingsbilder durcharbeitet. "
            "Mehr Epochen können die Erkennung verbessern, dauern aber länger "
            "und können bei sehr kleinen Datensätzen irgendwann schlechter werden. "
            "100 ist ein guter Startwert."
        ),
        "batch": (
            "Wie viele Bilder gleichzeitig verarbeitet werden. "
            "Größer ist oft schneller, braucht aber mehr Grafikspeicher. "
            "Wenn das Training mit Speicherfehler abbricht, diesen Wert senken "
            "(z. B. 8 oder 4)."
        ),
        "imgsz": (
            "Die exportierten Trainingsbilder sind bereits ballzentrierte Ausschnitte "
            "in Originalauflösung. 640 bedeutet hier: ein 640px-Ausschnitt um den Ball, "
            "nicht ein heruntergerechnetes 4K-Gesamtbild. Diesen Wert normalerweise "
            "bei 640 lassen; größere Werte rechnen die Ausschnitte hoch und brauchen "
            "mehr Speicher."
        ),
        "device": (
            "Legt fest, ob automatisch gewählt wird, die Grafikkarte genutzt wird "
            "oder nur der Prozessor. Automatisch ist für die meisten Fälle richtig."
        ),
        "start": (
            "Startet das interne Training. Die Anwendung bleibt offen; das kann "
            "je nach Anzahl der Markierungen und Rechnerleistung länger dauern."
        ),
        "load": (
            "Aktiviert das frisch trainierte Modell für die Ball-Erkennung in dieser Sitzung."
        ),
    }

    def __init__(self, parent=None, dataset_yaml: str = "", export_stats: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Modell trainieren")
        self.setMinimumWidth(560)
        self._thread: QThread | None = None
        self._worker: _TrainingWorker | None = None
        self._is_running = False
        self._best_model = ""

        layout = QVBoxLayout(self)

        if export_stats:
            summary = QLabel(
                "Trainingsdaten wurden vorbereitet:\n"
                f"Ausschnitte: {export_stats.get('total_frames', 0)} | "
                f"mit Ball: {export_stats.get('positive_crops', 0)} | "
                f"ohne Ball: {export_stats.get('negative_crops', 0)} | "
                f"Quell-Frames: {export_stats.get('source_frames', 0)} | "
                f"Train: {export_stats.get('train', 0)} | "
                f"Val: {export_stats.get('val', 0)}"
            )
            summary.setWordWrap(True)
            layout.addWidget(summary)

        settings_group = QGroupBox("Training")
        form = QFormLayout(settings_group)

        self._dataset_edit = QLineEdit(dataset_yaml)
        self._dataset_edit.setReadOnly(True)
        self._dataset_edit.setToolTip(self._TOOLTIPS["dataset"])
        form.addRow(self._label("Dataset:", "dataset"), self._dataset_edit)

        base_row = QHBoxLayout()
        self._base_model_edit = QLineEdit(str(runtime_path("models", "yolo11l.pt")))
        self._base_model_edit.setToolTip(self._TOOLTIPS["base_model"])
        self._base_btn = QPushButton("Auswählen...")
        self._base_btn.setToolTip("Nur ändern, wenn ein anderes vorhandenes YOLO-Modell als Grundlage genutzt werden soll.")
        self._base_btn.clicked.connect(self._select_base_model)
        base_row.addWidget(self._base_model_edit, stretch=1)
        base_row.addWidget(self._base_btn)
        form.addRow(self._label("Basis-Modell:", "base_model"), base_row)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1000)
        self._epochs_spin.setValue(100)
        self._epochs_spin.setToolTip(self._TOOLTIPS["epochs"])
        form.addRow(self._label("Epochen:", "epochs"), self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(16)
        self._batch_spin.setToolTip(self._TOOLTIPS["batch"])
        form.addRow(self._label("Batch-Größe:", "batch"), self._batch_spin)

        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(320, 1920)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        self._imgsz_spin.setToolTip(self._TOOLTIPS["imgsz"])
        form.addRow(self._label("Trainingsbildgröße:", "imgsz"), self._imgsz_spin)

        self._device_combo = QComboBox()
        self._device_combo.addItem("Automatisch", None)
        self._device_combo.addItem("GPU", "0")
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.setToolTip(self._TOOLTIPS["device"])
        form.addRow(self._label("Rechnen auf:", "device"), self._device_combo)

        layout.addWidget(settings_group)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        layout.addWidget(self._log)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._start_btn = self._buttons.addButton("Training starten", QDialogButtonBox.ButtonRole.AcceptRole)
        self._load_btn = self._buttons.addButton("Modell verwenden", QDialogButtonBox.ButtonRole.ActionRole)
        self._load_btn.setEnabled(False)
        self._start_btn.setToolTip(self._TOOLTIPS["start"])
        self._load_btn.setToolTip(self._TOOLTIPS["load"])
        self._buttons.rejected.connect(self.reject)
        self._start_btn.clicked.connect(self._start_training)
        self._load_btn.clicked.connect(self._load_model)
        layout.addWidget(self._buttons)

        self._append("Bereit. Das Training kann je nach Hardware deutlich dauern.")

    def _label(self, text: str, tooltip_key: str) -> QLabel:
        label = QLabel(text)
        label.setToolTip(self._TOOLTIPS[tooltip_key])
        return label

    def _select_base_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Basis-Modell wählen",
            os.path.dirname(self._base_model_edit.text()),
            "PyTorch Modell (*.pt)",
        )
        if path:
            self._base_model_edit.setText(path)

    def _start_training(self):
        if self._is_running:
            return

        dataset_yaml = self._dataset_edit.text().strip()
        base_model = self._base_model_edit.text().strip()
        if not os.path.isfile(dataset_yaml):
            QMessageBox.warning(self, "Training", "Dataset-Datei wurde nicht gefunden.")
            return
        if base_model and not os.path.isfile(base_model):
            QMessageBox.warning(self, "Training", "Basis-Modell wurde nicht gefunden.")
            return

        options = {
            "dataset_yaml": dataset_yaml,
            "base_model": base_model or None,
            "epochs": self._epochs_spin.value(),
            "batch": self._batch_spin.value(),
            "imgsz": self._imgsz_spin.value(),
            "output_dir": str(runtime_path("models")),
            "project": str(runtime_path("training", "runs")),
            "name": "ballmarker_finetune",
            "device": self._device_combo.currentData(),
        }

        self._set_running(True)
        self._progress.setRange(0, 0)
        self._append("Training gestartet...")

        self._thread = QThread(self)
        self._worker = _TrainingWorker(options)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self._append)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()

    def _on_finished(self, result: dict):
        self._best_model = result.get("best_model", "")
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._set_running(False)

        if self._best_model:
            self._append(f"Training abgeschlossen. Modell: {self._best_model}")
            self._load_btn.setEnabled(True)
        else:
            self._append("Training abgeschlossen, aber es wurde kein bestes Modell gefunden.")

    def _on_failed(self, message: str):
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._set_running(False)
        self._append(f"Training fehlgeschlagen: {message}")
        QMessageBox.critical(self, "Training fehlgeschlagen", message)

    def _load_model(self):
        if not self._best_model:
            return
        try:
            from detection.ball_detector import load_custom_model

            load_custom_model(self._best_model)
            self._append("Modell ist aktiv und wird für die YOLO-Erkennung verwendet.")
            QMessageBox.information(self, "Modell aktiv", "Das neue Modell wird jetzt verwendet.")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden:\n{exc}")

    def _set_running(self, running: bool):
        self._is_running = running
        self._start_btn.setEnabled(not running)
        for widget in (
            self._base_model_edit,
            self._base_btn,
            self._epochs_spin,
            self._batch_spin,
            self._imgsz_spin,
            self._device_combo,
        ):
            widget.setEnabled(not running)

    def _append(self, text: str):
        self._log.append(text)

    def _clear_thread_refs(self):
        self._thread = None
        self._worker = None

    def reject(self):
        if self._is_running:
            QMessageBox.information(
                self,
                "Training läuft",
                "Bitte warten, bis das Training abgeschlossen ist.",
            )
            return
        super().reject()

    def closeEvent(self, event):
        if self._is_running:
            QMessageBox.information(
                self,
                "Training läuft",
                "Bitte warten, bis das Training abgeschlossen ist.",
            )
            event.ignore()
            return
        super().closeEvent(event)
