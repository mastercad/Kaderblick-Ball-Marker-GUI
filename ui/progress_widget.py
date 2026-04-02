"""Fortschrittsanzeige für die Statusleiste.

Zeigt einen Fortschrittsbalken mit Text für zeitintensive Prozesse
wie Batch-YOLO-Erkennung oder Keyframe-Analyse.
Unterstützt mehrere gleichzeitige Tasks.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal


class ProgressTask:
    """Beschreibt einen laufenden Task mit Fortschritt."""

    def __init__(self, task_id: str, label: str, total: int = 0, cancellable: bool = False):
        self.task_id = task_id
        self.label = label
        self.total = total
        self.current = 0
        self.detail = ""
        self.cancellable = cancellable
        self.finished = False


class ProgressWidget(QWidget):
    """Widget für die Statusleiste: Fortschrittsbalken + Text + optionaler Abbrechen-Button.

    Nutzung:
        pw = ProgressWidget()
        statusbar.addPermanentWidget(pw)

        pw.start_task("batch_yolo", "YOLO-Erkennung", total=2400, cancellable=True)
        pw.update_task("batch_yolo", current=50, detail="23 erkannt")
        pw.finish_task("batch_yolo", "Fertig: 500 Bälle erkannt")
    """

    cancel_requested = Signal(str)  # task_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: dict[str, ProgressTask] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        self._label = QLabel("")
        self._label.setMinimumWidth(120)
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setFixedWidth(200)
        self._bar.setFixedHeight(16)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        layout.addWidget(self._bar)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: gray;")
        layout.addWidget(self._detail_label)

        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setFixedSize(20, 20)
        self._cancel_btn.setToolTip("Abbrechen")
        self._cancel_btn.setFlat(True)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn)

        self.hide()

    # ── Öffentliche API ───────────────────────────────────────────

    def start_task(self, task_id: str, label: str, total: int = 0, cancellable: bool = False):
        """Startet einen neuen Task. total=0 → unbestimmter Fortschritt."""
        task = ProgressTask(task_id, label, total, cancellable)
        self._tasks[task_id] = task
        self._refresh()

    def update_task(self, task_id: str, current: int = -1, detail: str = ""):
        """Aktualisiert den Fortschritt eines Tasks."""
        task = self._tasks.get(task_id)
        if not task:
            return
        if current >= 0:
            task.current = current
        if detail:
            task.detail = detail
        self._refresh()

    def finish_task(self, task_id: str, message: str = "", auto_hide_ms: int = 3000):
        """Markiert einen Task als abgeschlossen.

        message: Abschlussmeldung (wird kurz angezeigt, dann verschwindet der Task).
        auto_hide_ms: Nach wie vielen ms der Task ausgeblendet wird (0 = sofort).
        """
        task = self._tasks.get(task_id)
        if not task:
            return
        task.finished = True
        task.current = task.total
        if message:
            task.detail = message
        self._refresh()

        if auto_hide_ms > 0:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(auto_hide_ms, lambda tid=task_id: self._remove_task(tid))
        else:
            self._remove_task(task_id)

    def cancel_task(self, task_id: str):
        """Entfernt einen Task sofort (z.B. nach manuellem Abbruch)."""
        self._remove_task(task_id)

    def has_task(self, task_id: str) -> bool:
        """Prüft ob ein Task existiert."""
        return task_id in self._tasks

    # ── Intern ────────────────────────────────────────────────────

    def _remove_task(self, task_id: str):
        self._tasks.pop(task_id, None)
        self._refresh()

    def _active_task(self) -> ProgressTask | None:
        """Gibt den aktuell anzuzeigenden Task zurück (letzter nicht-fertiger, oder letzter fertiger)."""
        # Bevorzuge nicht-fertige Tasks
        for task in reversed(list(self._tasks.values())):
            if not task.finished:
                return task
        # Sonst den letzten fertigen
        for task in reversed(list(self._tasks.values())):
            return task
        return None

    def _refresh(self):
        """Aktualisiert die Darstellung basierend auf dem aktiven Task."""
        task = self._active_task()
        if task is None:
            self.hide()
            return

        self.show()
        self._label.setText(task.label)

        if task.total > 0:
            self._bar.setMaximum(task.total)
            self._bar.setValue(task.current)
            self._bar.setFormat(f"{task.current}/{task.total}")
            self._bar.show()
        else:
            # Unbestimmter Fortschritt (pulsierend)
            self._bar.setMaximum(0)
            self._bar.setValue(0)
            self._bar.show()

        if task.finished:
            self._bar.setFormat("✓")
            if task.total > 0:
                self._bar.setValue(task.total)

        self._detail_label.setText(task.detail)
        self._detail_label.setVisible(bool(task.detail))

        self._cancel_btn.setVisible(task.cancellable and not task.finished)

    def _on_cancel(self):
        task = self._active_task()
        if task and task.cancellable:
            self.cancel_requested.emit(task.task_id)
