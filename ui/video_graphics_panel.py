from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsItem, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaMetaData
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtCore import Qt, QRectF, QEvent, Signal, Slot, QSize, QPointF, QTimer
from PySide6.QtGui import QPainter, QMouseEvent, QColor, QTransform, QPen, QPolygonF, QBrush
from video.fps_detect import FpsDetector

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

# Konstanten für Marker-Größen (normiert, relativ zur kürzeren Videoseite)
DEFAULT_MARKER_RADIUS = 0.05   # 5% der kürzeren Kante
MIN_MARKER_RADIUS = 0.003      # 0.3% der kürzeren Kante
MAX_MARKER_RADIUS = 0.50       # 50% der kürzeren Kante

# Zoom-Konstanten
ZOOM_FACTOR_STEP = 1.15        # Pro Mausrad-Stufe 15% rein/raus
MIN_ZOOM = 1.0                 # Nicht kleiner als 1:1
MAX_ZOOM = 10.0                # Max 10-fach vergrößern


class VideoGraphicsPanel(QWidget):
    # Signal: wird nach dem Laden eines Videos emittiert, enthält die native Auflösung
    video_loaded = Signal(QSize)
    # Signal: Keyframe-Analyse läuft / ist fertig
    keyframes_status = Signal(str)   # Statustext (leer = fertig)
    keyframes_ready = Signal()       # echte Keyframes verfügbar
    # Signal: Statusmeldung für Operationen (YOLO, Interpolation, ...)
    status_message = Signal(str)
    # Signal: Strukturierter Fortschritt (task_id, current, total, detail)
    batch_progress = Signal(str, int, int, str)
    # Signal: Task gestartet (task_id, label, total, cancellable)
    task_started = Signal(str, str, int, bool)
    # Signal: Task beendet (task_id, message)
    task_finished = Signal(str, str)
    # Signal: Marker wurden hinzugefügt/entfernt/geändert
    markers_changed = Signal()

    def __init__(self, session):
        super(VideoGraphicsPanel, self).__init__()
        self.session = session
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.player.setVideoOutput(self.video_item)
        self.markers = []  # Marker-Objekte
        self.marker_items = {}  # Marker-Objekt -> QGraphicsEllipseItem
        self._zoom_level = 1.0  # Aktueller Zoom-Faktor
        self._panning = False   # Panning-Modus aktiv?
        self._pan_start = QPointF()
        self._highlight_index = -1  # Index für Marker-Zyklen auf aktuellem Frame
        self._highlight_items = []  # temporäre QGraphicsEllipseItems (Highlight-Ringe)
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._remove_highlight)
        self._fps_detector = FpsDetector()
        self._fps_detector.player = self.player
        self.player.metaDataChanged.connect(self._fps_detector.on_metadata_changed)

        # ── Feldgrenze (Polygon aus field_calibration.json) ──
        self._field_boundary: np.ndarray | None = None   # Nx1x2 int32 (cv2 contour)
        self._field_boundary_wh: tuple[int, int] | None = None  # (w, h) des Kalibrierungsbildes
        self._field_margin_px: int = 150  # Toleranz in Pixel (Einwürfe erlauben)
        self._field_boundary_item = None  # QGraphicsPolygonItem für Overlay
        self._field_calibration_path: str | None = None  # Pfad zur JSON für Autosave

        # ── Performance-Optimierung ──
        # Nur die nötigsten Bereiche neu zeichnen
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        # Painter-State nicht bei jedem Item speichern/wiederherstellen
        self.view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        # Scrollbars nur bei Zoom anzeigen
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Cache deaktivieren (Video-Frames ändern sich ständig)
        self.view.setCacheMode(QGraphicsView.CacheModeFlag.CacheNone)
        # Drag-Modus deaktivieren
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Kein Frame um die View
        self.view.setFrameShape(QGraphicsView.Shape.NoFrame)
        # Antialiasing nur für Marker, nicht für Video
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Video-Item: nativeSize nutzen und Aspect-Ratio beibehalten
        self.video_item.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        # Auf nativeSize-Änderung reagieren (sobald Video Frames liefert)
        self.video_item.nativeSizeChanged.connect(self._on_native_size_changed)
        # Anker für Zoom = Mausposition
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # ── Frame-Label + Marker-Statistik ──
        self._frame_label = QLabel("Frame: -")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_label = QLabel("")
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_label.setStyleSheet("font-size: 11px; color: #555;")

        layout = QVBoxLayout()
        self.open_btn = QPushButton("Video öffnen")
        layout.addWidget(self.open_btn)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self._frame_label)
        layout.addWidget(self._stats_label)
        self.setLayout(layout)
        self.open_btn.clicked.connect(self.open_video)

        # Position-Tracking für Frame-Anzeige
        self.player.positionChanged.connect(self._on_position_changed)

        # Seek-Schutz: Marker erst anzeigen wenn das Videobild gerendert wurde
        self._seeking = False
        self.player.videoSink().videoFrameChanged.connect(self._on_video_frame_changed)

    # ── YOLO-Ballerkennung ────────────────────────────────────────

    def detect_ball(self):
        """YOLO-Erkennung auf dem aktuellen Frame – erstellt Marker."""
        if not self.has_video:
            return
        self.pause()
        from PySide6.QtCore import QUrl
        path = QUrl(self.player.source().toString()).toLocalFile()
        frame_idx = self.current_frame()

        self.status_message.emit("Ball wird erkannt …")

        import threading
        threading.Thread(
            target=self._detect_ball_bg_safe,
            args=(path, frame_idx, self.fps, self.player.source().toString()),
            daemon=True,
        ).start()

    def _detect_ball_bg(self, path, frame_idx, fps, video_id):
        """Nicht verwendet – siehe _detect_ball_bg_safe."""
        pass

    # ── Ausschlusszone-Prüfung ─────────────────────────────────────

    def _is_in_exclusion_zone(self, cx, cy, frame_idx, video_id, window=10):
        """Prüft ob (cx, cy) in einer Ausschlusszone liegt.

        Berücksichtigt Ausschluss-Marker auf dem exakten Frame und
        in einem Fenster von ±window Frames (für nicht-interpolierte Zonen).
        """
        for m in self.session.markers:
            if m.video_file != video_id or m.type != "exclusion":
                continue
            if abs(m.frame_index - frame_idx) > window:
                continue
            dist = ((cx - m.position[0]) ** 2 + (cy - m.position[1]) ** 2) ** 0.5
            if dist < m.radius * 2.0:
                return True
        return False

    @staticmethod
    def _check_exclusion_list(cx, cy, frame_idx, exclusions, window=10):
        """Prüft ob (cx, cy) in einer vorberechneten Ausschluss-Liste liegt.

        exclusions: list von (frame_index, x, y, radius) Tupeln.
        Geeignet für Thread-sichere Batch-Prüfung.
        """
        for ef, ex, ey, er in exclusions:
            if abs(ef - frame_idx) > window:
                continue
            dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            if dist < er * 2.0:
                return True
        return False

    # ── Feldgrenze laden und prüfen ───────────────────────────────

    def load_field_calibration(self, json_path: str, camera_key: str | None = None):
        """Lädt die Feldgrenze aus einer field_calibration.json.

        camera_key: z.B. "cam0", "cam1". Wenn None, wird anhand des
        Videodateinamens automatisch gematch.
        """
        log = logging.getLogger("field_boundary")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.error("Feldkalibrierung nicht lesbar: %s – %s", json_path, exc)
            self.status_message.emit(f"Fehler: Kalibrierung nicht lesbar ({exc})")
            return False

        # Kamera zuordnen -------------------------------------------------
        if camera_key and camera_key in data:
            cam = data[camera_key]
        else:
            cam = self._match_camera(data)

        if cam is None:
            log.warning("Keine passende Kamera in %s gefunden", json_path)
            self.status_message.emit("Keine passende Kamera in Kalibrierung gefunden")
            return False

        pts = cam.get("field_boundary")
        if not pts or len(pts) < 3:
            log.warning("field_boundary fehlt oder zu wenig Punkte")
            self.status_message.emit("Feldkalibrierung: Kein gültiges Polygon gefunden")
            return False

        fw = cam.get("frame_width", 3840)
        fh = cam.get("frame_height", 2160)

        self._field_boundary = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        self._field_boundary_wh = (fw, fh)
        self._field_calibration_path = json_path  # Pfad merken für Autosave
        self._draw_field_boundary_overlay()

        cam_id = cam.get("camera_id", camera_key or "?")
        log.info("Feldgrenze geladen: %d Punkte, Kamera=%s, Frame=%dx%d, Toleranz=%dpx",
                 len(pts), cam_id, fw, fh, self._field_margin_px)
        self.status_message.emit(
            f"Feldgrenze geladen (Kamera {cam_id}, {len(pts)} Punkte, ±{self._field_margin_px}px Toleranz)")
        return True

    def _match_camera(self, cal_data: dict) -> dict | None:
        """Versucht die passende Kamera-Sektion anhand des Videonamens zu finden."""
        if not self.has_video:
            return None
        from PySide6.QtCore import QUrl
        video_path = QUrl(self.player.source().toString()).toLocalFile()
        video_stem = Path(video_path).stem.lower()

        for key in sorted(cal_data.keys()):
            cam = cal_data[key]
            if not isinstance(cam, dict):
                continue
            # Vergleich über video_path (Stem, ohne Extension)
            cal_path = cam.get("video_path", "")
            if cal_path:
                cal_stem = Path(cal_path).stem.lower()
                if cal_stem == video_stem:
                    return cam
            # Fallback: Kamera-ID im Dateinamen suchen
            cam_id = str(cam.get("camera_id", ""))
            if cam_id and (f"kamera{cam_id}" in video_stem or f"cam{cam_id}" in video_stem):
                return cam
        return None

    def is_inside_field(self, cx_norm: float, cy_norm: float) -> bool:
        """Prüft ob eine normierte Position (0..1) innerhalb der Feldgrenze liegt.

        Berücksichtigt _field_margin_px als Toleranz (für Einwürfe etc.).
        Gibt True zurück wenn keine Feldgrenze geladen ist (= kein Filter aktiv).
        """
        if self._field_boundary is None or self._field_boundary_wh is None:
            return True  # Kein Filter → alles erlaubt

        fw, fh = self._field_boundary_wh
        px = cx_norm * fw
        py = cy_norm * fh

        dist = cv2.pointPolygonTest(self._field_boundary, (px, py), measureDist=True)
        # dist > 0: innerhalb, dist == 0: auf Kante, dist < 0: außerhalb (Abstand)
        return dist >= -self._field_margin_px

    @staticmethod
    def _is_inside_field_static(cx_norm: float, cy_norm: float,
                                boundary: np.ndarray, boundary_wh: tuple[int, int],
                                margin_px: int) -> bool:
        """Thread-sichere statische Variante der Feldgrenze-Prüfung."""
        fw, fh = boundary_wh
        px = cx_norm * fw
        py = cy_norm * fh
        dist = cv2.pointPolygonTest(boundary, (px, py), measureDist=True)
        return dist >= -margin_px

    def clear_field_boundary(self):
        """Entfernt die geladene Feldgrenze."""
        self._field_boundary = None
        self._field_boundary_wh = None
        self._field_calibration_path = None
        if self._field_boundary_item is not None:
            self.scene.removeItem(self._field_boundary_item)
            self._field_boundary_item = None
        self.status_message.emit("Feldgrenze entfernt")

    def _draw_field_boundary_overlay(self):
        """Zeichnet die Feldgrenze als halbtransparentes Polygon-Overlay auf dem Video."""
        # Altes Overlay entfernen
        if self._field_boundary_item is not None:
            self.scene.removeItem(self._field_boundary_item)
            self._field_boundary_item = None

        if self._field_boundary is None or self._field_boundary_wh is None:
            return

        fw, fh = self._field_boundary_wh
        # Native Video-Größe bestimmen
        native = self.video_item.nativeSize()
        if native.isEmpty():
            # Noch kein Frame → versuche mit dem Kalibrierungswert
            vw, vh = fw, fh
        else:
            vw, vh = native.width(), native.height()

        # Polygon-Punkte in Video-Koordinaten umrechnen
        scale_x = vw / fw
        scale_y = vh / fh
        points = []
        for pt in self._field_boundary.reshape(-1, 2):
            points.append(QPointF(pt[0] * scale_x, pt[1] * scale_y))
        polygon = QPolygonF(points)

        from PySide6.QtWidgets import QGraphicsPolygonItem
        item = QGraphicsPolygonItem(polygon, self.video_item)
        pen = QPen(QColor(0, 255, 0, 180), 3)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)  # Breite bleibt bei Zoom konstant
        item.setPen(pen)
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        item.setZValue(5)  # Über dem Video, unter den Markern
        self._field_boundary_item = item

    def _on_ball_detected_impl(self, result, frame_idx, video_id):
        """Wird im Main-Thread aufgerufen nach YOLO-Erkennung."""
        if not self.has_video or self.player.source().toString() != video_id:
            return
        if result is None:
            self.status_message.emit("Kein Ball erkannt")
            return
        cx, cy, radius = result
        # Ausschlusszone prüfen
        if self._is_in_exclusion_zone(cx, cy, frame_idx, video_id):
            self.status_message.emit(f"Detektion in Ausschlusszone verworfen (Frame {frame_idx})")
            return
        # Feldgrenze prüfen
        if not self.is_inside_field(cx, cy):
            self.status_message.emit(
                f"Detektion außerhalb Feldgrenze verworfen (Frame {frame_idx}, "
                f"pos=({cx:.3f},{cy:.3f}))")
            return
        from model.marker import Marker
        # Prüfe ob auf diesem Frame bereits ein Ball-Marker existiert (Ausschluss ignorieren)
        existing = [m for m in self.markers if m.frame_index == frame_idx and m.type != "exclusion"]
        if existing:
            # Vorhandenen Marker aktualisieren
            m = existing[0]
            m.position = (cx, cy)
            m.radius = radius
            if m in self.marker_items:
                self._update_marker_item(m, self.marker_items[m])
            self.status_message.emit(f"Ball erkannt – Marker aktualisiert (Frame {frame_idx})")
        else:
            marker = Marker(
                video_id, frame_idx,
                round(frame_idx * self.ms_per_frame),
                (cx, cy), radius, "yolo",
            )
            item = self._create_marker_item(marker)
            self.markers.append(marker)
            self.marker_items[marker] = item
            if hasattr(self.session, 'add_marker'):
                self.session.add_marker(marker)
            self.status_message.emit(f"Ball erkannt – Marker gesetzt (Frame {frame_idx})")
        self._update_marker_visibility(self.current_frame())
        self.markers_changed.emit()
        main_window = self._find_main_window()
        if main_window:
            main_window.update_undo_redo_actions()

    # QMetaObject.invokeMethod kann keine beliebigen Argumente übergeben –
    # deshalb speichern wir das Ergebnis und rufen den Slot ohne Args auf.
    _ball_detect_result = None

    def _detect_ball_bg_safe(self, path, frame_idx, fps, video_id):
        import traceback
        try:
            print(f"[YOLO] Starte Erkennung: frame={frame_idx}, fps={fps}, path={path!r}")
            from detection.ball_detector import detect_ball_in_frame
            result = detect_ball_in_frame(
                path, frame_idx, fps,
                field_boundary=self._field_boundary,
                field_boundary_wh=self._field_boundary_wh,
                field_margin_px=self._field_margin_px,
            )
            print(f"[YOLO] Ergebnis: {result}")
            self._ball_detect_result = (result, frame_idx, video_id)
            from PySide6.QtCore import QMetaObject, Qt as _Qt
            QMetaObject.invokeMethod(self, '_on_ball_detected_slot', _Qt.ConnectionType.QueuedConnection)
        except Exception as exc:
            traceback.print_exc()
            self._ball_detect_result = (None, frame_idx, video_id)
            from PySide6.QtCore import QMetaObject, Qt as _Qt
            QMetaObject.invokeMethod(self, '_on_ball_detected_slot', _Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _on_ball_detected_slot(self):
        if self._ball_detect_result is not None:
            result, frame_idx, video_id = self._ball_detect_result
            self._ball_detect_result = None
            self._on_ball_detected_impl(result, frame_idx, video_id)

    # ── Batch-YOLO-Erkennung (alle Frames) ─────────────────────

    # Abbruch-Flag für laufende Batch-Erkennung
    _batch_cancel = False
    _batch_running = False
    # Queue für Batch-Ergebnisse (Thread → Main)
    _batch_queue: list = []
    # Thread-sichere Menge der bereits markierten Frames (wird auch zur Laufzeit aktualisiert)
    _batch_existing_frames: set = set()

    def _init_batch_lock(self):
        import threading
        if not hasattr(self, '_batch_lock'):
            self._batch_lock = threading.Lock()

    def detect_all_frames(self, skip_types: set | None = None):
        """Startet YOLO-Erkennung auf allen Frames des Videos im Hintergrund.

        skip_types: Menge von Marker-Typen ('manual', 'yolo', 'interpolated'),
                    deren Frames übersprungen werden sollen.
                    None = alle Typen überspringen.
        """
        if not self.has_video:
            return
        if self._batch_running:
            self.status_message.emit("Batch-Erkennung läuft bereits")
            return
        self.pause()
        from PySide6.QtCore import QUrl
        path = QUrl(self.player.source().toString()).toLocalFile()
        video_id = self.player.source().toString()
        total = self.total_frames()
        fps = self.fps

        # Frames ermitteln, die übersprungen werden sollen
        # Thread-sichere Menge – wird auch bei manueller Marker-Platzierung aktualisiert
        # Ausschluss-Marker zählen nicht als "bereits markiert"
        self._init_batch_lock()
        if skip_types is None:
            # Alle Ball-Marker-Typen überspringen (nicht Ausschluss)
            self._batch_existing_frames = {m.frame_index for m in self.session.markers
                              if m.video_file == video_id and m.type != "exclusion"}
        else:
            self._batch_existing_frames = {m.frame_index for m in self.session.markers
                              if m.video_file == video_id and m.type in skip_types}

        self._batch_cancel = False
        self._batch_running = True
        with self._batch_lock:
            self._batch_queue = []
        self.task_started.emit(self._batch_task_id(), "YOLO-Erkennung", total, True)

        import threading
        threading.Thread(
            target=self._detect_all_bg,
            args=(path, video_id, total, fps),
            daemon=True,
        ).start()

    def cancel_batch_detection(self):
        """Bricht die laufende Batch-Erkennung ab."""
        if self._batch_running:
            self._batch_cancel = True

    def _detect_all_bg(self, path, video_id, total, fps):
        """Hintergrund-Thread: YOLO auf alle Frames."""
        from detection.ball_detector import detect_ball_in_frame
        from PySide6.QtCore import QMetaObject, Qt as _Qt

        detected = 0
        skipped = 0

        # Positions-Anker aus allen vorhandenen Ball-Markern aufbauen
        # (manuell gesetzte Marker dienen als besonders starke Referenz)
        anchor_map: dict[int, tuple[float, float]] = {}
        for m in self.session.markers:
            if m.video_file == video_id and m.type != "exclusion":
                anchor_map[m.frame_index] = (m.position[0], m.position[1])

        # Ausschluss-Marker vorberechnen (Thread-sicher)
        exclusion_list = [
            (m.frame_index, m.position[0], m.position[1], m.radius)
            for m in self.session.markers
            if m.video_file == video_id and m.type == "exclusion"
        ]

        # Feldgrenze-Daten kopieren (Thread-sicher)
        field_boundary = self._field_boundary
        field_boundary_wh = self._field_boundary_wh
        field_margin_px = self._field_margin_px

        last_anchor: tuple[float, float] | None = None
        suppressed = 0

        for frame_idx in range(total):
            if self._batch_cancel:
                break

            # Anker aktualisieren: bekannte Position für diesen oder vorherigen Frame
            if frame_idx in anchor_map:
                last_anchor = anchor_map[frame_idx]

            if frame_idx in self._batch_existing_frames:
                skipped += 1
                continue

            try:
                result = detect_ball_in_frame(
                    path, frame_idx, fps, anchor=last_anchor,
                    field_boundary=field_boundary,
                    field_boundary_wh=field_boundary_wh,
                    field_margin_px=field_margin_px,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result = None

            # Ausschlusszone prüfen
            if result is not None and exclusion_list:
                cx, cy, _r = result
                if self._check_exclusion_list(cx, cy, frame_idx, exclusion_list):
                    result = None
                    suppressed += 1

            # Ergebnis in die Queue legen und im Main-Thread verarbeiten
            with self._batch_lock:
                self._batch_queue.append((result, frame_idx, video_id))
            if result is not None:
                detected += 1
                # Neue Detektion wird selbst zum Anker für Folge-Frames
                last_anchor = (result[0], result[1])
                anchor_map[frame_idx] = last_anchor

            # Alle 5 Frames UI aktualisieren (nicht jeden Frame → Performance)
            if frame_idx % 5 == 0 or frame_idx == total - 1:
                QMetaObject.invokeMethod(
                    self, '_process_batch_queue', _Qt.ConnectionType.QueuedConnection
                )

            # Status-Update alle 10 Frames
            if frame_idx % 10 == 0:
                progress = frame_idx + 1
                excl_info = f", {suppressed} unterdrückt" if suppressed else ""
                self._batch_progress_data = (progress, total, f"{detected} erkannt, {skipped} übersprungen{excl_info}")
                QMetaObject.invokeMethod(
                    self, '_update_batch_status', _Qt.ConnectionType.QueuedConnection
                )

        self._batch_running = False
        cancelled = " (abgebrochen)" if self._batch_cancel else ""
        excl_info = f", {suppressed} unterdrückt" if suppressed else ""
        self._batch_finish_text = f"{detected} Bälle erkannt, {skipped} übersprungen{excl_info}{cancelled}"
        self._batch_progress_data = (total, total, self._batch_finish_text)
        QMetaObject.invokeMethod(
            self, '_update_batch_status', _Qt.ConnectionType.QueuedConnection
        )
        QMetaObject.invokeMethod(
            self, '_on_batch_done', _Qt.ConnectionType.QueuedConnection
        )

    _batch_progress_data = (0, 0, "")
    _batch_finish_text = ""

    def _batch_task_id(self) -> str:
        return f"batch_yolo_{id(self)}"

    @Slot()
    def _update_batch_status(self):
        cur, tot, detail = self._batch_progress_data
        self.batch_progress.emit(self._batch_task_id(), cur, tot, detail)
        if not self._batch_running:
            self.task_finished.emit(self._batch_task_id(), self._batch_finish_text)

    @Slot()
    def _process_batch_queue(self):
        """Verarbeitet angesammelte Batch-Ergebnisse im Main-Thread."""
        from model.marker import Marker
        with self._batch_lock:
            queue = self._batch_queue
            self._batch_queue = []

        video_id_check = self.player.source().toString() if self.has_video else ''
        for result, frame_idx, video_id in queue:
            if video_id != video_id_check:
                continue
            if result is None:
                continue
            cx, cy, radius = result
            # Nochmal prüfen ob inzwischen ein Ball-Marker existiert (z.B. manuell gesetzt)
            # Ausschluss-Marker ignorieren – sie blockieren keine Ball-Marker
            existing = [m for m in self.markers if m.frame_index == frame_idx and m.type != "exclusion"]
            if existing:
                continue
            marker = Marker(
                video_id, frame_idx,
                round(frame_idx * self.ms_per_frame),
                (cx, cy), radius, "yolo",
            )
            item = self._create_marker_item(marker)
            self.markers.append(marker)
            self.marker_items[marker] = item
            self.session.markers.append(marker)
            self._batch_existing_frames.add(frame_idx)

        self._update_marker_visibility(self.current_frame())
        self.markers_changed.emit()

    @Slot()
    def _on_batch_done(self):
        """Wird aufgerufen wenn die Batch-Erkennung abgeschlossen ist.
        Führt einen temporalen Konsistenz-Filter durch, um
        räumlich isolierte Fehlerkennungen zu entfernen."""
        if not self._batch_cancel:
            self._apply_temporal_filter()
        self.session.undo_stack.clear()
        self.session.redo_stack.clear()
        main_window = self._find_main_window()
        if main_window:
            main_window.update_undo_redo_actions()
            main_window.timeline.update()

    def _apply_temporal_filter(self):
        """Entfernt YOLO-Marker, die temporale Ausreißer sind.
        Manuelle Marker dienen als vertrauenswürdige Referenzpunkte."""
        from detection.ball_detector import filter_temporal_outliers
        if not self.has_video:
            return
        video_id = self.player.source().toString()
        # Nur YOLO-Marker dieses Videos sammeln
        yolo_markers = {
            m.frame_index: m
            for m in self.markers
            if m.video_file == video_id and m.type == "yolo"
        }
        if len(yolo_markers) <= 2:
            return

        detections = {
            f: (m.position[0], m.position[1], m.radius)
            for f, m in yolo_markers.items()
        }

        # Manuelle (und interpolierte) Marker als Anker-Referenzen
        anchors = {
            m.frame_index: (m.position[0], m.position[1])
            for m in self.markers
            if m.video_file == video_id and m.type in ("manual", "interpolated")
        }

        clean = filter_temporal_outliers(detections, anchors=anchors)
        removed_frames = set(detections.keys()) - set(clean.keys())

        if not removed_frames:
            return

        # Ausreißer-Marker entfernen
        for f in removed_frames:
            m = yolo_markers[f]
            if m in self.marker_items:
                item = self.marker_items.pop(m)
                self.scene.removeItem(item)
            if m in self.markers:
                self.markers.remove(m)
            if m in self.session.markers:
                self.session.markers.remove(m)

        self._update_marker_visibility(self.current_frame())
        self.markers_changed.emit()
        n = len(removed_frames)
        self.status_message.emit(f"Temporaler Filter: {n} Ausreißer entfernt")

    # ── Interpolation ──────────────────────────────────────────

    def interpolate_markers(self):
        """Interpoliert Marker zwischen allen vorhandenen Marker-Paaren dieses Videos.

        Ball-Marker (manual/yolo/interpolated) und Ausschluss-Marker werden
        getrennt interpoliert, damit sie sich nicht gegenseitig beeinflussen.
        """
        if not self.has_video:
            return 0
        video_id = self.player.source().toString()
        all_markers = [m for m in self.session.markers if m.video_file == video_id]

        # Ball-Marker und Ausschluss-Marker getrennt sammeln
        ball_markers = sorted(
            [m for m in all_markers if m.type != "exclusion"],
            key=lambda m: m.frame_index,
        )
        exclusion_markers = sorted(
            [m for m in all_markers if m.type == "exclusion"],
            key=lambda m: m.frame_index,
        )

        if len(ball_markers) < 2 and len(exclusion_markers) < 2:
            self.status_message.emit("Mindestens 2 Marker für Interpolation nötig")
            return 0

        from interpolation.linear import LinearInterpolation
        from model.marker import Marker
        interp = LinearInterpolation()
        count = 0

        existing_frames = {m.frame_index for m in all_markers}

        # Ball-Marker interpolieren → Typ "interpolated"
        count += self._interpolate_chain(
            ball_markers, "interpolated", existing_frames,
            video_id, interp, Marker
        )

        # Ausschluss-Marker interpolieren → Typ "exclusion"
        count += self._interpolate_chain(
            exclusion_markers, "exclusion", existing_frames,
            video_id, interp, Marker
        )

        if count > 0:
            self.sync_markers_with_session()
            self.markers_changed.emit()
        self.status_message.emit(f"{count} Marker interpoliert")
        return count

    def _interpolate_chain(self, markers, target_type, existing_frames, video_id, interp, Marker):
        """Interpoliert zwischen aufeinanderfolgenden Markern einer Kette.

        Returns:
            Anzahl neu erzeugter Marker.
        """
        count = 0
        for i in range(len(markers) - 1):
            m1 = markers[i]
            m2 = markers[i + 1]
            gap = m2.frame_index - m1.frame_index
            if gap <= 1:
                continue
            for f in range(m1.frame_index + 1, m2.frame_index):
                if f in existing_frames:
                    continue
                x, y, radius = interp.interpolate(m1, m2, f)
                marker = Marker(
                    video_id, f,
                    round(f * self.ms_per_frame),
                    (x, y), radius, target_type,
                )
                self.session.markers.append(marker)
                existing_frames.add(f)
                count += 1
        return count

    # ── Zentrale Marker-Berechnungen ──────────────────────────────────
    # Marker-Items sind Kinder des video_item und arbeiten in dessen
    # lokalem Koordinatensystem (0..width, 0..height).  Dadurch bewegen
    # sie sich bei Zoom/Pan/Resize immer exakt mit dem Video mit.

    def _video_rect(self):
        """Gibt das aktuelle sceneBoundingRect des Video-Items zurück."""
        return self.video_item.sceneBoundingRect()

    def _video_local_size(self):
        """Gibt (width, height) des Video-Items in lokalen Koordinaten zurück."""
        s = self.video_item.size()
        w = s.width() if s.width() > 0 else 1.0
        h = s.height() if s.height() > 0 else 1.0
        return (w, h)

    def _min_side(self):
        """Kürzere Seite des Video-Items (Basis für Radius-Normierung)."""
        w, h = self._video_local_size()
        return min(w, h)

    def _marker_local_center(self, marker):
        """Berechnet die Position eines Markers in lokalen Video-Koordinaten."""
        w, h = self._video_local_size()
        nx = max(0.0, min(1.0, marker.position[0]))
        ny = max(0.0, min(1.0, marker.position[1]))
        return (nx * w, ny * h)

    def _marker_pixel_radius(self, marker):
        """Berechnet den Pixel-Radius eines Markers aus dem normierten Radius."""
        return marker.radius * self._min_side()

    def _scene_to_norm(self, scene_pos):
        """Konvertiert eine Scene-Position in normierte (0..1) Video-Koordinaten."""
        local_pos = self.video_item.mapFromScene(scene_pos)
        w, h = self._video_local_size()
        nx = local_pos.x() / w
        ny = local_pos.y() / h
        return (max(0.0, min(1.0, nx)),
                max(0.0, min(1.0, ny)))

    # ── Marker-Farben nach Typ ─────────────────────────────────────────

    @staticmethod
    def _marker_color_for_type(marker_type: str) -> QColor:
        """Gibt die Farbe für einen Marker-Typ zurück."""
        if marker_type == "yolo":
            return QColor(0, 120, 255, 160)      # Blau
        elif marker_type == "interpolated":
            return QColor(255, 165, 0, 160)      # Orange
        elif marker_type == "exclusion":
            return QColor(85, 85, 85, 140)       # Grau (Ausschlusszone)
        else:  # "manual"
            return QColor(255, 0, 0, 160)        # Rot

    # ── Zentrales Marker-Zeichnen ─────────────────────────────────────

    def _create_marker_item(self, marker):
        """Erstellt ein QGraphicsEllipseItem als Kind des video_item."""
        item = QGraphicsEllipseItem(self.video_item)
        item.setBrush(self._marker_color_for_type(marker.type))
        self._update_marker_item(marker, item)
        return item

    def _update_marker_item(self, marker, item):
        """Aktualisiert Position und Größe eines Marker-Items (lokale Video-Koordinaten)."""
        cx, cy = self._marker_local_center(marker)
        r = self._marker_pixel_radius(marker)
        item.setRect(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        # Farbe bei Typ-Änderung aktualisieren
        item.setBrush(self._marker_color_for_type(marker.type))
        # Ausschluss-Marker: gestrichelte rote Umrandung
        if marker.type == "exclusion":
            item.setPen(QPen(QColor(220, 50, 50, 200), 2.5, Qt.PenStyle.DashLine))
        else:
            item.setPen(QPen(Qt.PenStyle.NoPen))

    # ── FPS (delegiert an zentralen FpsDetector) ─────────────────────

    @property
    def fps(self):
        return self._fps_detector.fps

    @property
    def ms_per_frame(self):
        return self._fps_detector.ms_per_frame

    @property
    def has_video(self):
        """True wenn ein Video geladen ist."""
        source = self.player.source()
        return bool(source and source.toString())

    @property
    def is_playing(self):
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def play(self):
        if self.has_video:
            self.player.play()

    def pause(self):
        if self.has_video:
            self.player.pause()

    def toggle_play(self):
        if not self.has_video:
            return
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def step_forward(self, n=1):
        """Springt n Frames vorwärts."""
        if not self.has_video:
            return
        self.pause()
        new_pos = self.player.position() + int(n * self.ms_per_frame)
        new_pos = min(new_pos, self.player.duration())
        self.player.setPosition(new_pos)

    def step_backward(self, n=1):
        """Springt n Frames rückwärts."""
        if not self.has_video:
            return
        self.pause()
        new_pos = self.player.position() - int(n * self.ms_per_frame)
        new_pos = max(0, new_pos)
        self.player.setPosition(new_pos)

    def current_frame(self):
        """Aktueller Frame-Index."""
        if not self.has_video:
            return 0
        return round(self.player.position() / self.ms_per_frame)

    # ── Marker-Frame-Navigation ───────────────────────────────────

    def _marker_frames_for_video(self, marker_types=None):
        """Gibt eine sortierte Liste aller Frame-Indizes zurück, die Marker haben.

        marker_types: None = alle, oder set/list von Typen z.B. {"manual", "yolo"}
        """
        video = self.player.source().toString() if self.has_video else ''
        frames = sorted(set(
            m.frame_index for m in self.session.markers
            if m.video_file == video
            and (marker_types is None or m.type in marker_types)
        ))
        return frames

    def _unmarked_frames(self):
        """Gibt Frames zurück, die KEINEN Ball-Marker haben (zwischen erstem und letztem Marker).
        Ausschluss-Marker zählen nicht als Ball-Marker."""
        ball_types = {"manual", "yolo", "interpolated"}
        all_frames = self._marker_frames_for_video(ball_types)
        if len(all_frames) < 2:
            return []
        first, last = all_frames[0], all_frames[-1]
        marked = set(all_frames)
        return [f for f in range(first, last + 1) if f not in marked]

    def jump_to_next_marker_frame(self):
        """Springt zum nächsten Frame, der einen Marker hat."""
        if not self.has_video:
            return
        self.pause()
        cur = self.current_frame()
        frames = self._marker_frames_for_video()
        for f in frames:
            if f > cur:
                self.set_frame(f)
                return

    def jump_to_prev_marker_frame(self):
        """Springt zum vorherigen Frame, der einen Marker hat."""
        if not self.has_video:
            return
        self.pause()
        cur = self.current_frame()
        frames = self._marker_frames_for_video()
        for f in reversed(frames):
            if f < cur:
                self.set_frame(f)
                return

    def jump_to_next_manual_frame(self):
        """Springt zum nächsten Frame mit manuellem Marker."""
        self._jump_to_typed_frame(1, {"manual"})

    def jump_to_prev_manual_frame(self):
        """Springt zum vorherigen Frame mit manuellem Marker."""
        self._jump_to_typed_frame(-1, {"manual"})

    def jump_to_next_interpolated_frame(self):
        """Springt zum nächsten Frame mit nur interpoliertem Marker."""
        self._jump_to_typed_frame(1, {"interpolated"})

    def jump_to_prev_interpolated_frame(self):
        """Springt zum vorherigen Frame mit nur interpoliertem Marker."""
        self._jump_to_typed_frame(-1, {"interpolated"})

    def jump_to_next_yolo_frame(self):
        """Springt zum nächsten Frame mit YOLO-Marker."""
        self._jump_to_typed_frame(1, {"yolo"})

    def jump_to_prev_yolo_frame(self):
        """Springt zum vorherigen Frame mit YOLO-Marker."""
        self._jump_to_typed_frame(-1, {"yolo"})

    def jump_to_next_exclusion_frame(self):
        """Springt zum nächsten Frame mit Ausschluss-Marker."""
        self._jump_to_typed_frame(1, {"exclusion"})

    def jump_to_prev_exclusion_frame(self):
        """Springt zum vorherigen Frame mit Ausschluss-Marker."""
        self._jump_to_typed_frame(-1, {"exclusion"})

    def jump_to_next_unmarked_frame(self):
        """Springt zum nächsten Frame ohne Marker (Lücke)."""
        if not self.has_video:
            return
        self.pause()
        cur = self.current_frame()
        for f in self._unmarked_frames():
            if f > cur:
                self.set_frame(f)
                return

    def jump_to_prev_unmarked_frame(self):
        """Springt zum vorherigen Frame ohne Marker (Lücke)."""
        if not self.has_video:
            return
        self.pause()
        cur = self.current_frame()
        for f in reversed(self._unmarked_frames()):
            if f < cur:
                self.set_frame(f)
                return

    def _jump_to_typed_frame(self, direction, types):
        """Springt zum nächsten/vorherigen Frame mit bestimmtem Marker-Typ."""
        if not self.has_video:
            return
        self.pause()
        cur = self.current_frame()
        frames = self._marker_frames_for_video(types)
        if direction > 0:
            for f in frames:
                if f > cur:
                    self.set_frame(f)
                    return
        else:
            for f in reversed(frames):
                if f < cur:
                    self.set_frame(f)
                    return

    # ── Keyframe-Navigation (I-Frames) ─────────────────────────

    def _ensure_keyframes(self):
        """Baut den Keyframe-Index auf (einmalig pro Video, lazy).

        Sofort wird eine Heuristik (alle ~2s) verwendet, damit die UI
        nicht blockiert.  Im Hintergrund läuft ffprobe und ersetzt die
        Heuristik-Liste, sobald die echten Keyframes bekannt sind.
        """
        vid = self.player.source().toString() if self.has_video else ''
        if hasattr(self, '_keyframe_video') and self._keyframe_video == vid:
            return                       # bereits initialisiert
        self._keyframe_video = vid
        self._keyframe_list: list[int] = []
        self._keyframes_loaded = False
        if not vid:
            return

        from PySide6.QtCore import QUrl
        path = QUrl(vid).toLocalFile()

        # ── Statusmeldung ────────────────────────────────────────
        self.keyframes_status.emit("Keyframes werden analysiert …")
        self.task_started.emit(self._kf_task_id(), "Keyframe-Analyse", 0, False)

        # ── Hintergrund: echte Keyframes via ffprobe ─────────────
        import threading
        self._kf_thread = threading.Thread(
            target=self._load_keyframes_bg,
            args=(path, self.fps, vid),
            daemon=True,
        )
        self._kf_thread.start()

    def _kf_task_id(self) -> str:
        return f"keyframes_{id(self)}"

    def _load_keyframes_bg(self, path: str, fps: float, vid_id: str):
        """Wird in einem Daemon-Thread ausgeführt – liest echte Keyframes."""
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                 '-show_entries', 'packet=pts_time,flags',
                 '-of', 'csv', path],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                self._emit_kf_done(vid_id)
                return
            kf_frames: list[int] = []
            for line in result.stdout.splitlines():
                # Format: "packet,<pts_time>,<flags>"  – K__ = Keyframe
                parts = line.split(',')
                if len(parts) >= 3 and 'K' in parts[2]:
                    try:
                        kf_frames.append(round(float(parts[1]) * fps))
                    except (ValueError, IndexError):
                        pass
            if kf_frames:
                kf_frames = sorted(set(kf_frames))
                # Nur aktualisieren, wenn noch dasselbe Video geladen ist
                if getattr(self, '_keyframe_video', '') == vid_id:
                    self._keyframe_list = kf_frames
        except Exception:
            pass          # Heuristik bleibt bestehen
        self._emit_kf_done(vid_id)

    def _emit_kf_done(self, vid_id: str):
        """Thread-sicher: signalisiert dem UI-Thread, dass Keyframes bereit sind."""
        from PySide6.QtCore import QMetaObject, Qt as _Qt, Q_ARG
        # Nur wenn noch dasselbe Video geladen ist
        if getattr(self, '_keyframe_video', '') != vid_id:
            return
        self._keyframes_loaded = True
        # Signals müssen aus dem Main-Thread emittiert werden
        QMetaObject.invokeMethod(
            self, '_on_keyframes_ready', _Qt.ConnectionType.QueuedConnection
        )

    @Slot()
    def _on_keyframes_ready(self):
        """Slot – wird im Main-Thread aufgerufen wenn ffprobe fertig ist."""
        n = len(self._keyframe_list)
        self.keyframes_status.emit(f"{n} Keyframes geladen")
        self.keyframes_ready.emit()
        self.task_finished.emit(self._kf_task_id(), f"{n} Keyframes geladen")

    def jump_to_next_keyframe(self):
        """Springt zum nächsten Keyframe (I-Frame)."""
        if not self.has_video:
            return
        if not getattr(self, '_keyframes_loaded', False):
            return          # Noch nicht bereit
        self.pause()
        cur = self.current_frame()
        for f in self._keyframe_list:
            if f > cur:
                self.set_frame(f)
                return

    def jump_to_prev_keyframe(self):
        """Springt zum vorherigen Keyframe (I-Frame)."""
        if not self.has_video:
            return
        if not getattr(self, '_keyframes_loaded', False):
            return          # Noch nicht bereit
        self.pause()
        cur = self.current_frame()
        for f in reversed(self._keyframe_list):
            if f < cur:
                self.set_frame(f)
                return

    def _on_position_changed(self, position):
        """Update Frame-Label und Marker-Sichtbarkeit bei Positionsänderung."""
        frame = round(position / self.ms_per_frame) if self.has_video else 0
        self._frame_label.setText(f"Frame: {frame}")
        # Während eines Seeks keine Marker anzeigen – das macht der Timer
        if not self._seeking:
            self._update_marker_visibility(frame)

    def _on_native_size_changed(self, size):
        """Wird aufgerufen wenn das Video seine native Größe meldet."""
        if size.isValid() and not size.isEmpty():
            self.video_item.setSize(size)
            self.scene.setSceneRect(self._video_rect())
            self.view.fitInView(self.video_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_level = 1.0
            self.sync_markers_with_session()

    def _detect_video_resolution(self, filepath):
        """Erkennt die native Auflösung der Videodatei per cv2."""
        try:
            import cv2
            cap = cv2.VideoCapture(filepath)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return QSize(w, h)
        except Exception:
            pass
        return None

    # ── Video ─────────────────────────────────────────────────────────

    def set_frame(self, frame):
        if self.player is not None:
            # Marker verstecken – _on_video_frame_changed zeigt sie nach Render
            self._seeking = True
            self._hide_all_markers()
            self.player.setPosition(round(frame * self.ms_per_frame))

    def _hide_all_markers(self):
        """Versteckt alle Marker-Items sofort."""
        for item in self.marker_items.values():
            item.setVisible(False)

    def _on_video_frame_changed(self):
        """Wird aufgerufen wenn ein neues Videobild dekodiert wurde."""
        if self._seeking:
            self._seeking = False
            self._update_marker_visibility(self.current_frame())

    def total_frames(self):
        if self.player is not None and self.player.duration() > 0:
            return int(self.player.duration() / self.ms_per_frame)
        return 1

    def open_video(self):
        from PySide6.QtWidgets import QFileDialog
        from PySide6.QtCore import QUrl
        filename, _ = QFileDialog.getOpenFileName(self, "Video öffnen", "", "Video Files (*.mp4 *.avi *.mov)")
        if filename:
            self.player.setSource(QUrl.fromLocalFile(filename))
            self.player.setPosition(0)
            self.player.pause()
            self._zoom_level = 1.0
            self._fps_detector.detect_from_file(filename)
            resolution = self._detect_video_resolution(filename)
            if resolution:
                self.video_loaded.emit(resolution)
            # Keyframe-Analyse im Hintergrund starten
            self._ensure_keyframes()
            main_window = self._find_main_window()
            if main_window:
                main_window.update_undo_redo_actions()

    # ── Hilfsfunktionen ───────────────────────────────────────────────

    def _find_main_window(self):
        """Sucht das MainWindow in der Widget-Hierarchie."""
        w = self.parent()
        while w and not hasattr(w, 'update_undo_redo_actions'):
            w = w.parent()
        return w

    # ── Events ────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Scroll-Position relativ merken, damit der Ausschnitt nicht springt
        h_bar = self.view.horizontalScrollBar()
        v_bar = self.view.verticalScrollBar()
        h_ratio = h_bar.value() / max(1, h_bar.maximum()) if h_bar.maximum() > 0 else 0.5
        v_ratio = v_bar.value() / max(1, v_bar.maximum()) if v_bar.maximum() > 0 else 0.5

        # fitInView + Zoom wiederherstellen
        self.view.fitInView(self.video_item, Qt.AspectRatioMode.KeepAspectRatio)
        if self._zoom_level > 1.01:
            self.view.scale(self._zoom_level, self._zoom_level)
        self.scene.setSceneRect(self._video_rect())

        # Scroll-Position wiederherstellen
        if self._zoom_level > 1.01:
            h_bar.setValue(int(h_ratio * h_bar.maximum()))
            v_bar.setValue(int(v_ratio * v_bar.maximum()))

    def eventFilter(self, obj, event):
        from model.marker import Marker
        if obj is not self.view.viewport():
            return False

        # ── Mausrad: Marker-Resize oder Zoom (VOR der View verarbeiten!) ──
        if event.type() == QEvent.Type.Wheel:
            # Ctrl+Wheel → Zoom
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if event.angleDelta().y() > 0:
                    self._apply_zoom(ZOOM_FACTOR_STEP)
                elif event.angleDelta().y() < 0:
                    self._apply_zoom(1.0 / ZOOM_FACTOR_STEP)
                return True
            # Mausrad über Marker → Marker-Resize
            scene_pos = self.view.mapToScene(event.position().toPoint())
            for marker, item in self.marker_items.items():
                if item.isVisible() and item.contains(item.mapFromScene(scene_pos)):
                    delta = event.angleDelta().y() / 1200.0
                    new_radius = marker.radius * (1 + delta)
                    new_radius = max(MIN_MARKER_RADIUS, min(MAX_MARKER_RADIUS, new_radius))
                    marker.radius = new_radius
                    # Manuell angepasst → Typ auf "manual" setzen (außer Ausschluss)
                    if marker.type not in ("manual", "exclusion"):
                        marker.type = "manual"
                        item.setBrush(self._marker_color_for_type("manual"))
                    self._update_marker_item(marker, item)
                    if hasattr(self.session, 'resize_marker'):
                        self.session.resize_marker(marker, new_radius)
                    return True
            # Kein Marker getroffen → Event normal weiterleiten
            return False

        # ── Mittlere Maustaste → Pan ──
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.MiddleButton and self._zoom_level > 1.01:
                self._panning = True
                self._pan_start = event.position()
                self.view.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
        if event.type() == QEvent.Type.MouseMove and self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.view.horizontalScrollBar().setValue(
                self.view.horizontalScrollBar().value() - int(delta.x()))
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().value() - int(delta.y()))
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.MiddleButton and self._panning:
                self._panning = False
                self.view.setCursor(Qt.CursorShape.ArrowCursor)
                return True

        if event.type() == QEvent.Type.MouseButtonPress:
            # Rechtsklick → Marker löschen
            if isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.RightButton:
                scene_pos = self.view.mapToScene(event.pos())
                for m, item in list(self.marker_items.items()):
                    if item.isVisible() and item.contains(item.mapFromScene(scene_pos)):
                        # Marker aus Scene, lokaler Liste und Session entfernen
                        if item.scene():
                            item.scene().removeItem(item)
                        del self.marker_items[m]
                        if m in self.markers:
                            self.markers.remove(m)
                        if hasattr(self.session, 'remove_marker'):
                            self.session.remove_marker(m)
                        main_window = self._find_main_window()
                        if main_window:
                            main_window.update_undo_redo_actions()
                            main_window.timeline.update()
                        self.status_message.emit("Marker gelöscht")
                        self.markers_changed.emit()
                        return True
                return False
            if isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.view.mapToScene(event.pos())
                local_pos = self.video_item.mapFromScene(scene_pos)
                # Prüfe ob ein existierender Marker angeklickt wurde
                for m, item in self.marker_items.items():
                    if item.isVisible() and item.contains(item.mapFromScene(scene_pos)):
                        self._dragged_marker = m
                        self._drag_local_offset = local_pos - item.rect().center()
                        self._prev_marker_pos = tuple(m.position)
                        return True
                # Neuen Marker anlegen wenn innerhalb des Videos
                # Shift+Klick → Ausschluss-Marker, sonst → manuell
                is_exclusion = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                marker_type = "exclusion" if is_exclusion else "manual"
                vid_rect = self.video_item.boundingRect()
                if vid_rect.contains(local_pos):
                    frame = round(self.player.position() / self.ms_per_frame)
                    timestamp = self.player.position()
                    video_file = self.player.source().toString() if hasattr(self.player, 'source') else ''
                    norm_x, norm_y = self._scene_to_norm(scene_pos)
                    marker = Marker(video_file, frame, timestamp,
                                    (norm_x, norm_y), DEFAULT_MARKER_RADIUS, marker_type)
                    item = self._create_marker_item(marker)
                    self.markers.append(marker)
                    self.marker_items[marker] = item
                    if hasattr(self.session, 'add_marker'):
                        self.session.add_marker(marker)
                    # Batch-Erkennung: Frame als bereits markiert registrieren
                    if self._batch_running:
                        self._batch_existing_frames.add(frame)
                    main_window = self._find_main_window()
                    if main_window:
                        main_window.update_undo_redo_actions()
                    self.markers_changed.emit()
                    self._dragged_marker = marker
                    self._drag_local_offset = local_pos - item.rect().center()
                    return True

        elif event.type() == QEvent.Type.MouseMove:
            if hasattr(self, '_dragged_marker') and self._dragged_marker:
                scene_pos = self.view.mapToScene(event.pos())
                local_pos = self.video_item.mapFromScene(scene_pos)
                item = self.marker_items[self._dragged_marker]
                rect = item.rect()
                new_center = local_pos - self._drag_local_offset
                item.setRect(new_center.x() - rect.width() / 2,
                             new_center.y() - rect.height() / 2,
                             rect.width(), rect.height())
                return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if hasattr(self, '_dragged_marker') and self._dragged_marker:
                if hasattr(self.session, 'move_marker') and hasattr(self, '_prev_marker_pos'):
                    scene_pos = self.view.mapToScene(event.position().toPoint())
                    norm_x, norm_y = self._scene_to_norm(scene_pos)
                    self.session.move_marker(self._dragged_marker, (norm_x, norm_y))
                    # Manuell verschoben → Typ auf "manual" setzen (außer Ausschluss)
                    if self._dragged_marker.type not in ("manual", "exclusion"):
                        self._dragged_marker.type = "manual"
                        item = self.marker_items.get(self._dragged_marker)
                        if item:
                            item.setBrush(self._marker_color_for_type("manual"))
                    self._prev_marker_pos = None
                self._dragged_marker = None
                self._drag_local_offset = None
                return True

        return False

    def keyPressEvent(self, event):
        video_loaded = self.has_video
        main_window = self._find_main_window()
        # Undo: Ctrl+Z or Z
        if (event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier)) or event.key() == Qt.Key_Z:
            if video_loaded and hasattr(self.session, 'undo') and hasattr(self.session, 'undo_stack'):
                if self.session.undo_stack:
                    self.session.undo()
                    self.sync_markers_with_session()
                    if main_window:
                        main_window.update_undo_redo_actions()
                    event.accept()
                    return
                else:
                    event.ignore()
                    return
            else:
                event.ignore()
                return
        # Redo: Ctrl+Y or Y
        if (event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier)) or event.key() == Qt.Key_Y:
            if video_loaded and hasattr(self.session, 'redo') and hasattr(self.session, 'redo_stack'):
                if self.session.redo_stack:
                    self.session.redo()
                    self.sync_markers_with_session()
                    if main_window:
                        main_window.update_undo_redo_actions()
                    event.accept()
                    return
                else:
                    event.ignore()
                    return
            else:
                event.ignore()
                return
        super().keyPressEvent(event)

    def sync_markers_with_session(self):
        """Synchronisiert alle Marker-Items mit dem Session-Zustand (zentral)."""
        # Alte Marker-Items entfernen (Kinder des video_item)
        for item in self.marker_items.values():
            # parentItem entfernen; scene.removeItem reicht
            if item.scene():
                item.scene().removeItem(item)
        self.marker_items.clear()
        current_video = self.player.source().toString() if hasattr(self.player, 'source') and self.player.source() else ''
        self.markers = [m for m in self.session.markers if m.video_file == current_video]
        for marker in self.markers:
            item = self._create_marker_item(marker)
            self.marker_items[marker] = item
        # Sichtbarkeit nach aktuellem Frame aktualisieren
        self._update_marker_visibility(self.current_frame())
        self.markers_changed.emit()

    def _update_marker_visibility(self, frame):
        """Zeigt nur Marker an, die zum aktuellen Frame gehören."""
        for marker, item in self.marker_items.items():
            item.setVisible(marker.frame_index == frame)
        self._update_stats_label(frame)

    # ── Marker auf aktuellem Frame hervorheben / anspringen ─────────

    def _current_frame_markers(self):
        """Gibt alle Marker auf dem aktuellen Frame des aktuellen Videos zurück."""
        if not self.has_video:
            return []
        video = self.player.source().toString()
        frame = self.current_frame()
        return [m for m in self.session.markers
                if m.video_file == video and m.frame_index == frame]

    def focus_next_marker(self):
        """Zykliert zum nächsten Marker auf dem aktuellen Frame,
        zoomt heran und hebt ihn hervor."""
        frame_markers = self._current_frame_markers()
        if not frame_markers:
            self.status_message.emit("Keine Marker auf diesem Frame")
            return
        # Nächsten Index berechnen (zyklisch)
        self._highlight_index = (self._highlight_index + 1) % len(frame_markers)
        marker = frame_markers[self._highlight_index]
        self._highlight_marker(marker)
        # Statusmeldung
        idx = self._highlight_index + 1
        total = len(frame_markers)
        pos = marker.position
        self.status_message.emit(
            f"Marker {idx}/{total}  –  Typ: {marker.type}  "
            f"Pos: ({pos[0]:.3f}, {pos[1]:.3f})"
        )

    def focus_prev_marker(self):
        """Zykliert zum vorherigen Marker auf dem aktuellen Frame."""
        frame_markers = self._current_frame_markers()
        if not frame_markers:
            self.status_message.emit("Keine Marker auf diesem Frame")
            return
        self._highlight_index = (self._highlight_index - 1) % len(frame_markers)
        marker = frame_markers[self._highlight_index]
        self._highlight_marker(marker)
        idx = self._highlight_index + 1
        total = len(frame_markers)
        pos = marker.position
        self.status_message.emit(
            f"Marker {idx}/{total}  –  Typ: {marker.type}  "
            f"Pos: ({pos[0]:.3f}, {pos[1]:.3f})"
        )

    def _highlight_marker(self, marker):
        """Zoomt zum Marker, zentriert die View und zeigt einen pulsierenden Highlight-Ring."""
        self._remove_highlight()
        # Marker-Position in lokalen Video-Koordinaten
        cx, cy = self._marker_local_center(marker)
        r = self._marker_pixel_radius(marker)
        # Highlight-Ring (2.5× Marker-Radius, leuchtend gelb, dick)
        ring_r = max(r * 2.5, 30)  # mindestens 30px damit man es sieht
        ring = QGraphicsEllipseItem(
            cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r,
            self.video_item
        )
        ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        ring.setPen(QPen(QColor(255, 255, 0, 220), 3.0, Qt.PenStyle.SolidLine))
        ring.setZValue(100)  # über allem
        self._highlight_items.append(ring)
        # Zweiter Ring (Fadenkreuz-Linie, etwas größer)
        cross_r = ring_r * 1.5
        ring2 = QGraphicsEllipseItem(
            cx - cross_r, cy - cross_r, 2 * cross_r, 2 * cross_r,
            self.video_item
        )
        ring2.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        ring2.setPen(QPen(QColor(255, 255, 0, 120), 1.5, Qt.PenStyle.DashLine))
        ring2.setZValue(100)
        self._highlight_items.append(ring2)
        # Zoom auf 4× und View auf Marker zentrieren
        target_zoom = 4.0
        if self._zoom_level < target_zoom:
            factor = target_zoom / self._zoom_level
            self._zoom_level = target_zoom
            self.view.resetTransform()
            self.view.fitInView(self.video_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.view.scale(target_zoom, target_zoom)
            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Zentriere auf Marker-Position (szene-Koordinaten)
        scene_pos = self.video_item.mapToScene(QPointF(cx, cy))
        self.view.centerOn(scene_pos)
        # Highlight nach 3 Sekunden entfernen
        self._highlight_timer.start(3000)

    def _remove_highlight(self):
        """Entfernt alle temporären Highlight-Items."""
        self._highlight_timer.stop()
        for item in self._highlight_items:
            if item.scene():
                item.scene().removeItem(item)
        self._highlight_items.clear()

    def _update_stats_label(self, frame):
        """Aktualisiert die Marker-Statistik unter dem Frame-Label."""
        if not self.has_video:
            self._stats_label.setText("")
            return
        video_id = self.player.source().toString()
        video_markers = [m for m in self.session.markers if m.video_file == video_id]
        total = len(video_markers)
        if total == 0:
            self._stats_label.setText("Keine Marker")
            return
        n_manual = sum(1 for m in video_markers if m.type == "manual")
        n_yolo = sum(1 for m in video_markers if m.type == "yolo")
        n_interp = sum(1 for m in video_markers if m.type == "interpolated")
        n_excl = sum(1 for m in video_markers if m.type == "exclusion")
        # Marker auf aktuellem Frame
        frame_markers = [m for m in video_markers if m.frame_index == frame]
        frame_info = ""
        if frame_markers:
            types = ", ".join(m.type for m in frame_markers)
            frame_info = f"  |  Frame: {len(frame_markers)} ({types})"
        excl_info = f'&nbsp;&nbsp;<span style="color:#555">\u25cf</span>&nbsp;{n_excl} Ausschluss' if n_excl else ''
        self._stats_label.setText(
            f'<span style="color:#d00">\u25cf</span>&nbsp;{n_manual} manuell'
            f'&nbsp;&nbsp;<span style="color:#0078ff">\u25cf</span>&nbsp;{n_yolo} YOLO'
            f'&nbsp;&nbsp;<span style="color:#ffa500">\u25cf</span>&nbsp;{n_interp} interpoliert'
            f'{excl_info}'
            f'&nbsp;&nbsp;= {total} gesamt{frame_info}'
        )

    # ── Zoom & Pan ────────────────────────────────────────────────────

    def _apply_zoom(self, factor, anchor_pos=None):
        """Wendet einen Zoom-Faktor an, begrenzt auf MIN_ZOOM..MAX_ZOOM."""
        new_zoom = self._zoom_level * factor
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, new_zoom))
        real_factor = new_zoom / self._zoom_level
        if abs(real_factor - 1.0) < 1e-6:
            return
        self._zoom_level = new_zoom
        self.view.scale(real_factor, real_factor)
        # Scrollbars nur anzeigen wenn reingezoomt
        if self._zoom_level > 1.01:
            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def reset_zoom(self):
        """Zoom auf 1:1 zurücksetzen."""
        self.view.resetTransform()
        self._zoom_level = 1.0
        self.view.fitInView(self.video_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        """Fallback – wird nur erreicht wenn eventFilter das Event nicht konsumiert hat."""
        super().wheelEvent(event)
