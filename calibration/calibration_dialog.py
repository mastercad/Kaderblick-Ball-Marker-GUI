"""
Qt-basierter Dialog für die manuelle Spielfeld-Kalibrierung.

Zeigt einen Videoframe an und erlaubt interaktives Setzen von
Spielfeldmarkierungen (Feldrand, Mittellinie, Mittelkreis, Strafräume).
Unterstützt Dual-Kamera-Setup mit kameraspezifischen Modi.

Die eigentliche Logik ist in spezialisierte Module aufgeteilt:
  - calibration_modes   – Modi-Definitionen und Farbpalette
  - drag_point           – verschiebbarer Punkt (DragPoint)
  - calibration_view     – zoombare/pannbare GraphicsView
  - scene_renderer       – Zeichnen von Punkten/Linien/Polygonen
  - point_manager        – CRUD-Operationen auf Kalibrierungspunkten
"""

import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from calibration.calibration_modes import (
    COLORS,
    DONE_MODE,
    FLAG_MODES,
    current_mode,
    modes_for_camera,
)
from calibration.calibration_view import CalibrationView
from calibration.field_calibration import (
    FieldCalibrationData,
    load_calibration,
    save_calibration,
)
from calibration.point_manager import PointManager
from calibration.scene_renderer import SceneRenderer

log = logging.getLogger("field_calibration_dialog")


class FieldCalibrationDialog(QDialog):
    """
    Interaktiver Dialog für die manuelle Spielfeld-Kalibrierung.

    Zeigt einen Videoframe an und erlaubt das Setzen von Markierungen
    per Mausklick. Unterstützt Drag & Drop bestehender Punkte.
    """

    calibration_saved = Signal(str)  # Pfad zur gespeicherten Datei

    def __init__(self, parent=None, video_path: str = "", camera_id: int = 0,
                 frame_index: int = 0, calibration_path: str = "",
                 video_paths: Optional[Dict[int, str]] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Feldkalibrierung – Kamera {camera_id}")
        self.setMinimumSize(1024, 700)
        self.resize(1400, 900)

        # Video-Pfade pro Kamera
        self._video_paths: Dict[int, str] = video_paths or {}
        if video_path and camera_id not in self._video_paths:
            self._video_paths[camera_id] = video_path
        self._video_path = self._video_paths.get(camera_id, video_path)
        self._camera_id = camera_id
        self._frame_index = frame_index
        self._calibration_path = calibration_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "field_calibration.json",
        )

        # Kalibrierungsdaten + Punkt-Manager
        self._data = FieldCalibrationData(camera_id=camera_id)
        self._point_mgr = PointManager(self._data)
        self._current_mode_idx = 0

        # Grafik
        self._scene = QGraphicsScene()
        self._view = CalibrationView(self._scene, self)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._point_radius = 10.0

        # Renderer
        self._renderer = SceneRenderer(
            self._scene,
            get_points=self._point_mgr.points_for_mode,
            on_point_moved=self._on_point_moved,
            point_radius=self._point_radius,
        )

        # Bild laden
        self._frame_image: Optional[np.ndarray] = None
        self._load_frame()

        # Vorhandene Kalibrierung laden
        self._try_load_existing()

        # UI aufbauen
        self._build_ui()
        self._redraw_all()
        self._update_status()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Obere Steuerleiste
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Kamera:"))
        self._cam_combo = QComboBox()
        self._cam_combo.addItem("Kamera 0 (Links)", 0)
        self._cam_combo.addItem("Kamera 1 (Rechts)", 1)
        self._cam_combo.setCurrentIndex(self._camera_id)
        self._cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        top_bar.addWidget(self._cam_combo)

        top_bar.addSpacing(20)

        self._prev_mode_btn = QPushButton("◀ Vorheriger Modus")
        self._prev_mode_btn.clicked.connect(self._prev_mode)
        top_bar.addWidget(self._prev_mode_btn)

        self._mode_label = QLabel()
        self._mode_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_bar.addWidget(self._mode_label, stretch=1)

        self._next_mode_btn = QPushButton("Nächster Modus ▶")
        self._next_mode_btn.clicked.connect(self._next_mode)
        top_bar.addWidget(self._next_mode_btn)

        main_layout.addLayout(top_bar)

        # Info-Zeile
        info_bar = QHBoxLayout()
        self._points_label = QLabel()
        self._points_label.setStyleSheet("font-size: 12px; color: #666;")
        info_bar.addWidget(self._points_label)

        info_bar.addStretch()

        self._undo_btn = QPushButton("↩ Letzten Punkt entfernen (Z)")
        self._undo_btn.clicked.connect(self._remove_last_point)
        info_bar.addWidget(self._undo_btn)

        self._clear_mode_btn = QPushButton("Modus leeren")
        self._clear_mode_btn.clicked.connect(self._clear_current_mode)
        info_bar.addWidget(self._clear_mode_btn)

        self._fit_btn = QPushButton("⊞ Einpassen")
        self._fit_btn.clicked.connect(self._view.fit_image)
        info_bar.addWidget(self._fit_btn)

        main_layout.addLayout(info_bar)

        # Grafik-View
        self._view.point_clicked.connect(self._on_point_clicked)
        self._view.point_remove_requested.connect(self._on_point_remove_requested)
        self._view.line_insert_requested.connect(self._on_line_insert_requested)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self._view, stretch=1)

        # Hilfe
        help_text = (
            "Linksklick = Punkt setzen  |  Doppelklick auf Linie = Punkt einfügen  |  "
            "Rechtsklick auf Punkt = entfernen  |  Punkte ziehen = verschieben  |  "
            "Z = letzten Punkt entfernen  |  Mausrad = Zoom  |  "
            "Mittlere Taste = Verschieben"
        )
        help_label = QLabel(help_text)
        help_label.setStyleSheet("font-size: 11px; color: #888; padding: 4px;")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(help_label)

        # Untere Button-Leiste
        bottom_bar = QHBoxLayout()

        self._import_btn = QPushButton("📂 Kalibrierung importieren…")
        self._import_btn.clicked.connect(self._import_calibration)
        bottom_bar.addWidget(self._import_btn)

        self._export_btn = QPushButton("💾 Kalibrierung exportieren…")
        self._export_btn.clicked.connect(self._export_calibration)
        bottom_bar.addWidget(self._export_btn)

        bottom_bar.addStretch()

        self._save_btn = QPushButton("✅ Speichern && Schließen")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._save_and_close)
        bottom_bar.addWidget(self._save_btn)

        self._cancel_btn = QPushButton("Abbrechen")
        self._cancel_btn.clicked.connect(self.reject)
        bottom_bar.addWidget(self._cancel_btn)

        main_layout.addLayout(bottom_bar)

    # ── Frame laden ──────────────────────────────────────────────

    def _load_frame(self):
        """Lädt einen Frame aus dem Video per cv2."""
        if not self._video_path or not os.path.isfile(self._video_path):
            return
        try:
            cap = cv2.VideoCapture(self._video_path)
            if self._frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_index)
            ret, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if ret:
                self._frame_image = frame
                self._data.frame_width = w
                self._data.frame_height = h
                self._data.video_path = self._video_path

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                self._pixmap_item = self._scene.addPixmap(pixmap)
                self._pixmap_item.setZValue(0)
                self._scene.setSceneRect(QRectF(0, 0, w, h))

                self._point_radius = max(8, min(20, w / 200))
                self._renderer.point_radius = self._point_radius

                QTimer.singleShot(100, self._view.fit_image)

        except Exception as exc:
            log.error("Frame konnte nicht geladen werden: %s", exc)

    # ── Vorhandene Kalibrierung laden ────────────────────────────

    def _try_load_existing(self):
        """Versucht vorhandene Kalibrierungsdaten zu laden."""
        if os.path.isfile(self._calibration_path):
            existing = load_calibration(self._calibration_path, self._camera_id)
            if existing:
                self._data = existing
                self._point_mgr.data = existing
                if self._frame_image is not None:
                    h, w = self._frame_image.shape[:2]
                    self._data.frame_width = w
                    self._data.frame_height = h
                    self._data.video_path = self._video_path
                self._data.camera_id = self._camera_id
                log.info("Vorhandene Kalibrierung für Kamera %d geladen", self._camera_id)

    # ── Modi ─────────────────────────────────────────────────────

    def _modes(self):
        return modes_for_camera(self._camera_id)

    def _current_mode(self):
        return current_mode(self._modes(), self._current_mode_idx)

    def _next_mode(self):
        modes = self._modes()
        if self._current_mode_idx < len(modes):
            self._current_mode_idx += 1
        self._update_status()
        self._redraw_all()

    def _prev_mode(self):
        if self._current_mode_idx > 0:
            self._current_mode_idx -= 1
        self._update_status()
        self._redraw_all()

    def _on_camera_changed(self, index):
        new_cam = self._cam_combo.itemData(index)
        if new_cam == self._camera_id:
            return

        self._camera_id = new_cam
        self._data = FieldCalibrationData(camera_id=new_cam)
        self._point_mgr.data = self._data

        new_path = self._video_paths.get(new_cam, "")
        if new_path and new_path != self._video_path:
            self._video_path = new_path
            self._frame_index = 0
            self._load_frame()
        elif new_path:
            self._video_path = new_path

        if self._frame_image is not None:
            h, w = self._frame_image.shape[:2]
            self._data.frame_width = w
            self._data.frame_height = h
            self._data.video_path = self._video_path

        self.setWindowTitle(f"Feldkalibrierung – Kamera {new_cam}")

        self._try_load_existing()
        self._current_mode_idx = 0
        self._update_status()
        self._redraw_all()

    # ── Punkt-Ereignisse (delegiert an PointManager) ─────────────

    def _on_point_clicked(self, x: float, y: float):
        mode, _, min_pts, max_pts = self._current_mode()
        completed = self._point_mgr.add_point(mode, x, y, max_pts)
        if completed:
            self._next_mode()
            return
        self._update_status()
        self._redraw_all()

    def _remove_last_point(self):
        mode = self._current_mode()[0]
        self._point_mgr.remove_last_point(mode)
        self._update_status()
        self._redraw_all()

    def _on_point_remove_requested(self, index: int):
        mode = self._current_mode()[0]
        self._point_mgr.remove_point_at(mode, index)
        self._update_status()
        self._redraw_all()

    def _on_line_insert_requested(self, x: float, y: float):
        mode, _, _, max_pts = self._current_mode()
        threshold = max(30.0, self._point_radius * 5)
        if self._point_mgr.insert_on_line(mode, x, y, max_pts, threshold):
            self._update_status()
            self._redraw_all()

    def _clear_current_mode(self):
        mode = self._current_mode()[0]
        self._point_mgr.clear_mode(mode)
        self._update_status()
        self._redraw_all()

    def _on_point_moved(self, mode: str, index: int, new_pos):
        self._point_mgr.move_point(mode, index, new_pos)
        self._renderer.redraw_lines(self._modes(), self._data)

    # ── Zeichnen ────────────────────────────────────────────────

    def _redraw_all(self):
        current_mode_name = self._current_mode()[0]
        self._renderer.redraw_all(self._modes(), current_mode_name, self._data)
        self._view._active_mode = current_mode_name

    def _update_status(self):
        """Aktualisiert Modus-Label und Punkt-Zähler."""
        mode, desc, min_pts, max_pts = self._current_mode()
        modes = self._modes()
        total = len(modes)

        if mode == "done":
            self._mode_label.setText(f"[✓] {desc}")
        else:
            idx = self._current_mode_idx + 1
            self._mode_label.setText(f"[{idx}/{total}] {desc}")

        self._prev_mode_btn.setEnabled(self._current_mode_idx > 0)
        self._next_mode_btn.setEnabled(self._current_mode_idx < len(modes))

        pts = self._point_mgr.points_for_mode(mode)
        pts_text = f"Punkte: {len(pts)}"
        if max_pts > 0:
            pts_text += f" / {max_pts}"
        elif min_pts > 0:
            pts_text += f" (min. {min_pts})"
        if len(pts) >= min_pts and mode != "done":
            pts_text += "  ✓"
        self._points_label.setText(pts_text)

    # ── Tastatur ────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Z:
            self._remove_last_point()
            event.accept()
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self._next_mode()
            event.accept()
        elif key == Qt.Key.Key_N:
            self._next_mode()
            event.accept()
        elif key == Qt.Key.Key_B:
            self._prev_mode()
            event.accept()
        elif key == Qt.Key.Key_0:
            self._view.fit_image()
            event.accept()
        else:
            super().keyPressEvent(event)

    # ── Import / Export ─────────────────────────────────────────

    def _import_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Kalibrierung importieren", "",
            "JSON-Dateien (*.json);;Alle Dateien (*)")
        if not path:
            return
        data = load_calibration(path, self._camera_id)
        if data:
            self._data = data
            self._point_mgr.data = data
            self._data.camera_id = self._camera_id
            if self._frame_image is not None:
                h, w = self._frame_image.shape[:2]
                self._data.frame_width = w
                self._data.frame_height = h
            self._current_mode_idx = 0
            self._update_status()
            self._redraw_all()
            log.info("Kalibrierung importiert von %s", path)
        else:
            QMessageBox.warning(
                self, "Import",
                f"Keine Kalibrierung für Kamera {self._camera_id} in der Datei gefunden.")

    def _export_calibration(self):
        if not self._data.is_valid():
            QMessageBox.warning(
                self, "Export",
                "Kalibrierung nicht gültig.\n"
                "Mindestens 3 Spielfeldrand-Punkte erforderlich.")
            return

        default_name = "field_calibration.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Kalibrierung exportieren",
            os.path.join(os.path.dirname(self._calibration_path), default_name),
            "JSON-Dateien (*.json)")
        if not path:
            return

        self._finalize_data()
        save_calibration(self._data, path)
        self.calibration_saved.emit(path)
        log.info("Kalibrierung exportiert nach %s", path)
        QMessageBox.information(
            self, "Export",
            f"Kalibrierung für Kamera {self._camera_id} exportiert:\n{path}")

    # ── Speichern ───────────────────────────────────────────────

    def _finalize_data(self):
        """Bereitet Daten vor dem Speichern auf."""
        if self._data.field_boundary and not self._data.corners:
            if len(self._data.field_boundary) >= 4:
                self._data.corners = self._data.field_boundary[:4]

    def _save_and_close(self):
        if not self._data.is_valid():
            reply = QMessageBox.question(
                self, "Ungültige Kalibrierung",
                "Die Kalibrierung hat weniger als 3 Spielfeldrand-Punkte.\n"
                "Trotzdem speichern?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._finalize_data()
        save_calibration(self._data, self._calibration_path)
        self.calibration_saved.emit(self._calibration_path)
        self.accept()

    def get_calibration_data(self) -> FieldCalibrationData:
        return self._data

    def get_calibration_path(self) -> str:
        return self._calibration_path
