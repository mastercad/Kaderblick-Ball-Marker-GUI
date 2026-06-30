"""Progress dialog for exporting YOLO training data."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
)


class _TrainingExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, json_path: str, output_dir: str):
        super().__init__()
        self._json_path = json_path
        self._output_dir = output_dir
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            from training.export_training_data import export_yolo_dataset

            stats = export_yolo_dataset(
                self._json_path,
                self._output_dir,
                progress_callback=self.progress.emit,
                cancel_callback=lambda: self._cancel_requested,
            )
            self.finished.emit(stats)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingDataExportDialog(QDialog):
    """Shows progress while training data is exported in a background thread."""

    def __init__(self, parent=None, json_path: str = "", output_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Trainingsdaten exportieren")
        self.setMinimumWidth(520)
        self._thread: QThread | None = None
        self._worker: _TrainingExportWorker | None = None
        self._running = False
        self.stats: dict | None = None
        self.was_cancelled = False
        self.error_message = ""

        layout = QVBoxLayout(self)

        self._status_label = QLabel("Export wird vorbereitet...")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        layout.addWidget(self._log)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._close_btn = self._buttons.addButton(
            "Schließen", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._cancel_btn = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self._close_btn.setEnabled(False)
        self._buttons.rejected.connect(self._cancel_or_close)
        self._close_btn.clicked.connect(self.accept)
        layout.addWidget(self._buttons)

        self._start(json_path, output_dir)

    def _start(self, json_path: str, output_dir: str):
        self._running = True
        self._append("Export gestartet.")
        self._thread = QThread(self)
        self._worker = _TrainingExportWorker(json_path, output_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()

    def _on_progress(self, current: int, total: int, detail: str):
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(current)
        self._status_label.setText(detail)
        self._append(detail)

    def _on_finished(self, stats: dict):
        self.stats = stats
        self._running = False
        total = max(1, stats.get("source_frames", 1))
        self._progress.setRange(0, total)
        self._progress.setValue(total)
        self._status_label.setText("Export abgeschlossen.")
        self._append(
            f"Fertig: {stats.get('total_frames', 0)} Ausschnitte erzeugt "
            f"({stats.get('positive_crops', 0)} mit Ball, "
            f"{stats.get('negative_crops', 0)} ohne Ball)."
        )
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _on_failed(self, message: str):
        self.error_message = message
        self._running = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status_label.setText("Export fehlgeschlagen.")
        self._append(f"Fehler: {message}")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)
        QMessageBox.critical(self, "Export fehlgeschlagen", message)

    def _on_cancelled(self):
        self.was_cancelled = True
        self._running = False
        self._status_label.setText("Export abgebrochen.")
        self._append("Export wurde abgebrochen.")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _cancel_or_close(self):
        if not self._running:
            self.reject()
            return
        self._status_label.setText("Export wird abgebrochen...")
        self._append("Abbruch angefordert...")
        self._cancel_btn.setEnabled(False)
        if self._worker is not None:
            self._worker.request_cancel()

    def _append(self, text: str):
        if self._log.toPlainText().splitlines()[-1:] == [text]:
            return
        self._log.append(text)

    def _clear_thread_refs(self):
        self._thread = None
        self._worker = None

    def closeEvent(self, event):
        if self._running:
            self._cancel_or_close()
            event.ignore()
            return
        super().closeEvent(event)
