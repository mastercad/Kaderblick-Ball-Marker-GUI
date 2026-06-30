"""GUI dialog for training the heatmap ball detector."""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)


class _HeatmapTrainingWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, options: dict):
        super().__init__()
        self._options = options

    def run(self):
        try:
            from training.train_heatmap_model import train_heatmap_model

            path = train_heatmap_model(
                status_callback=self.progress.emit,
                **self._options,
            )
            self.finished.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))


class HeatmapTrainingDialog(QDialog):
    """Starts heatmap training from inside the desktop application."""

    _TOOLTIPS = {
        "dataset": (
            "Ordner mit den Heatmap-Trainingsdaten. Darin liegen Sequenzen aus mehreren "
            "Frames, Ziel-Heatmaps und Negativbeispiele."
        ),
        "epochs": (
            "Wie oft alle Trainingsbeispiele durchlaufen werden. Mehr Epochen können "
            "helfen, dauern aber länger. Bei wenigen Markierungen nicht extrem hoch setzen."
        ),
        "batch": (
            "Wie viele Ausschnitte gleichzeitig verarbeitet werden. Höher ist schneller, "
            "braucht aber mehr Grafikspeicher. Bei Speicherfehlern auf 2 oder 1 senken."
        ),
        "device": (
            "Automatisch nutzt eine Grafikkarte, falls verfügbar. CPU funktioniert, ist "
            "aber deutlich langsamer."
        ),
        "start": (
            "Startet das interne Heatmap-Training. Danach nutzt der Heatmap-Button das "
            "neu gespeicherte Modell."
        ),
    }

    def __init__(self, parent=None, dataset_dir: str = "", export_stats: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Heatmap-Modell trainieren")
        self.setMinimumWidth(560)
        self._thread: QThread | None = None
        self._worker: _HeatmapTrainingWorker | None = None
        self._is_running = False
        self._model_path = ""

        layout = QVBoxLayout(self)

        if export_stats:
            summary = QLabel(
                "Heatmap-Trainingsdaten wurden vorbereitet:\n"
                f"Ball-Samples: {export_stats.get('positive', 0)} | "
                f"Negativsamples: {export_stats.get('negative', 0)} | "
                f"Quell-Frames: {export_stats.get('source_frames', 0)} | "
                f"Train: {export_stats.get('train', 0)} | "
                f"Val: {export_stats.get('val', 0)}"
            )
            summary.setWordWrap(True)
            layout.addWidget(summary)

        settings_group = QGroupBox("Training")
        form = QFormLayout(settings_group)

        self._dataset_edit = QLineEdit(dataset_dir)
        self._dataset_edit.setReadOnly(True)
        self._dataset_edit.setToolTip(self._TOOLTIPS["dataset"])
        form.addRow(self._label("Dataset-Ordner:", "dataset"), self._dataset_edit)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(40)
        self._epochs_spin.setToolTip(self._TOOLTIPS["epochs"])
        form.addRow(self._label("Epochen:", "epochs"), self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 32)
        self._batch_spin.setValue(4)
        self._batch_spin.setToolTip(self._TOOLTIPS["batch"])
        form.addRow(self._label("Batch-Größe:", "batch"), self._batch_spin)

        self._device_combo = QComboBox()
        self._device_combo.addItem("Automatisch", None)
        self._device_combo.addItem("GPU", "cuda")
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
        self._log.setMinimumHeight(130)
        layout.addWidget(self._log)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._start_btn = self._buttons.addButton("Training starten", QDialogButtonBox.ButtonRole.AcceptRole)
        self._start_btn.setToolTip(self._TOOLTIPS["start"])
        self._buttons.rejected.connect(self.reject)
        self._start_btn.clicked.connect(self._start_training)
        layout.addWidget(self._buttons)

        self._append("Bereit. Für zuverlässige Ergebnisse braucht der Detektor viele echte Ball- und Negativbeispiele.")

    def _label(self, text: str, tooltip_key: str) -> QLabel:
        label = QLabel(text)
        label.setToolTip(self._TOOLTIPS[tooltip_key])
        return label

    def _start_training(self):
        if self._is_running:
            return

        dataset_dir = self._dataset_edit.text().strip()
        if not os.path.isdir(os.path.join(dataset_dir, "samples", "train")):
            QMessageBox.warning(self, "Heatmap-Training", "Heatmap-Dataset wurde nicht gefunden.")
            return

        options = {
            "dataset_dir": dataset_dir,
            "epochs": self._epochs_spin.value(),
            "batch_size": self._batch_spin.value(),
            "device": self._device_combo.currentData(),
        }

        self._set_running(True)
        self._progress.setRange(0, self._epochs_spin.value())
        self._progress.setValue(0)
        self._append("Heatmap-Training gestartet...")

        self._thread = QThread(self)
        self._worker = _HeatmapTrainingWorker(options)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()

    def _on_progress(self, current: int, total: int, message: str):
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(min(current, total))
        self._append(message)

    def _on_finished(self, path: str):
        self._model_path = path
        self._progress.setValue(self._progress.maximum())
        self._set_running(False)
        self._append(f"Training abgeschlossen. Heatmap-Modell: {path}")
        QMessageBox.information(
            self,
            "Heatmap-Modell bereit",
            "Das Heatmap-Modell wurde gespeichert und wird vom Heatmap-Button verwendet.",
        )

    def _on_failed(self, message: str):
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._set_running(False)
        self._append(f"Training fehlgeschlagen: {message}")
        QMessageBox.critical(self, "Heatmap-Training fehlgeschlagen", message)

    def _set_running(self, running: bool):
        self._is_running = running
        self._start_btn.setEnabled(not running)
        self._epochs_spin.setEnabled(not running)
        self._batch_spin.setEnabled(not running)
        self._device_combo.setEnabled(not running)

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
