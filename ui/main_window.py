import os

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QMenuBar, QFileDialog,
                               QMessageBox, QPushButton, QStatusBar, QLabel, QDialog, QCheckBox,
                               QRadioButton, QGroupBox, QDialogButtonBox, QButtonGroup, QComboBox,
                               QSpinBox)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import QSize, Qt, QUrl, QTimer
from ui.video_panel import VideoPanel
from ui.timeline_widget import TimelineWidget
from ui.progress_widget import ProgressWidget
from model.session import Session
from autosave.autosave import Autosave
from export.exporter import export_markers, import_markers

from shared.kaderblick_qt_theme import BrandHeaderWidget

class MainWindow(QMainWindow):
    def resizeEvent(self, event):
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        # Prüfe, ob Maus am Fensterrand (Resize-Handle) gedrückt wurde
        if event.button() == Qt.MouseButton.LeftButton:
            # Qt gibt keine direkte Info, ob am Fensterrand, aber wir können prüfen, ob Cursor-Shape ein Resize ist
            cursor_shape = self.cursor().shape()
            if cursor_shape in [
                Qt.CursorShape.SizeHorCursor,
                Qt.CursorShape.SizeVerCursor,
                Qt.CursorShape.SizeFDiagCursor,
                Qt.CursorShape.SizeBDiagCursor,
                Qt.CursorShape.SizeAllCursor
            ]:
                self._resize_active = True
                self._debug_resize("RESIZE START")
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, '_resize_active') and self._resize_active:
            self._resize_active = False
            self._debug_resize("RESIZE END")
        super().mouseReleaseEvent(event)

    def _debug_resize(self, label):
        left_panel = self.left_panel
        right_panel = self.right_panel
        print(f"[DEBUG] {label}")
        for panel, name in [(left_panel, "LeftPanel"), (right_panel, "RightPanel")]:
            video_rect = panel.video_item.boundingRect()
            video_scene_rect = panel.video_item.sceneBoundingRect()
            overlay_rect = panel.view.viewport().rect()
            scroll_x_visible = panel.view.horizontalScrollBar().isVisible()
            scroll_y_visible = panel.view.verticalScrollBar().isVisible()
            marker_info = []
            for marker in panel.markers:
                norm_pos = marker.position
                pixel_x = video_scene_rect.x() + norm_pos[0] * video_scene_rect.width()
                pixel_y = video_scene_rect.y() + norm_pos[1] * video_scene_rect.height()
                r = marker.radius * video_scene_rect.width()
                marker_info.append(f"Marker: norm=({norm_pos[0]:.4f},{norm_pos[1]:.4f}), pixel=({pixel_x:.1f},{pixel_y:.1f}), radius={r:.1f}")
            print(f"[DEBUG] {name} Video: {video_rect}, Scene: {video_scene_rect}, Overlay: {overlay_rect}")
            print(f"[DEBUG] {name} Scrollbars: horizontal={scroll_x_visible}, vertical={scroll_y_visible}")
            for info in marker_info:
                print(f"[DEBUG] {name} {info}")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaderblick - Ballmarker")
        self.session = Session()
        self.autosave = Autosave(self.session, get_video_paths=self._get_loaded_video_paths, get_sync_offset=lambda: self._sync_offset_frames)
        from ui.video_graphics_panel import VideoGraphicsPanel
        self.left_panel = VideoGraphicsPanel(self.session)
        self.right_panel = VideoGraphicsPanel(self.session)
        self.timeline = TimelineWidget(self.session, self.left_panel, self.right_panel)

        # Zentraler Play/Pause-Button
        self.play_btn = QPushButton("▶  Play / Pause")
        self.play_btn.setMinimumHeight(36)
        self.play_btn.clicked.connect(self._toggle_play_all)

        # Frame-Navigation Buttons
        self.step_back_btn = QPushButton("◀◀ -25")
        self.step_back_btn.clicked.connect(lambda: self._step_all(-25))
        self.prev_frame_btn = QPushButton("◀ -1")
        self.prev_frame_btn.clicked.connect(lambda: self._step_all(-1))
        self.next_frame_btn = QPushButton("+1 ▶")
        self.next_frame_btn.clicked.connect(lambda: self._step_all(1))
        self.step_fwd_btn = QPushButton("+25 ▶▶")
        self.step_fwd_btn.clicked.connect(lambda: self._step_all(25))

        # Sprungziel-Dropdown + Sprung-Buttons
        self.jump_type_combo = QComboBox()
        self.jump_type_combo.addItem("◆ Beliebiger Marker", "any")
        self.jump_type_combo.addItem("🔴 Manuell", "manual")
        self.jump_type_combo.addItem("🔵 YOLO", "yolo")
        self.jump_type_combo.addItem("🟠 Interpoliert", "interpolated")
        self.jump_type_combo.addItem("⬛ Ausschluss", "exclusion")
        self.jump_type_combo.addItem("⚪ Lücke (kein Marker)", "gap")
        self.jump_type_combo.addItem("◐ Keyframe", "keyframe")
        self.jump_type_combo.setToolTip("Sprungziel wählen (Ctrl+Links/Rechts zum Springen)")
        self.jump_type_combo.setMinimumWidth(160)

        self.jump_prev_btn = QPushButton("◀")
        self.jump_prev_btn.setToolTip("Zum vorherigen Sprungziel (Ctrl+Links)")
        self.jump_prev_btn.setFixedWidth(36)
        self.jump_prev_btn.clicked.connect(lambda: self._jump_selected(-1))
        self.jump_next_btn = QPushButton("▶")
        self.jump_next_btn.setToolTip("Zum nächsten Sprungziel (Ctrl+Rechts)")
        self.jump_next_btn.setFixedWidth(36)
        self.jump_next_btn.clicked.connect(lambda: self._jump_selected(1))

        # Control-Leiste
        controls = QHBoxLayout()
        controls.addWidget(self.step_back_btn)
        controls.addWidget(self.prev_frame_btn)
        controls.addWidget(self.play_btn, stretch=1)
        controls.addWidget(self.next_frame_btn)
        controls.addWidget(self.step_fwd_btn)
        controls.addSpacing(16)
        controls.addWidget(self.jump_prev_btn)
        controls.addWidget(self.jump_type_combo)
        controls.addWidget(self.jump_next_btn)

        # Sync-Offset Steuerung
        controls.addSpacing(16)
        self._sync_offset_frames = 0
        sync_label = QLabel("Sync-Offset (Frames):")
        sync_label.setToolTip("Verschiebt das rechte Video relativ zum linken.\n"
                              "Positiv = rechtes Video wird vorgeschoben,\n"
                              "Negativ = rechtes Video wird zurückgeschoben.")
        controls.addWidget(sync_label)
        self._sync_offset_spin = QSpinBox()
        self._sync_offset_spin.setRange(-10000, 10000)
        self._sync_offset_spin.setValue(0)
        self._sync_offset_spin.setSuffix(" F")
        self._sync_offset_spin.setToolTip("Frame-Offset für das rechte Video")
        self._sync_offset_spin.setFixedWidth(100)
        self._sync_offset_spin.valueChanged.connect(self._on_sync_offset_changed)
        controls.addWidget(self._sync_offset_spin)

        # Werkzeug-Leiste: YOLO + Interpolation
        tools = QHBoxLayout()
        self.detect_btn = QPushButton("⚽ Ball erkennen (YOLO)")
        self.detect_btn.setToolTip("YOLO-Erkennung auf dem aktuellen Frame beider Videos (Ctrl+D)")
        self.detect_btn.clicked.connect(self._detect_ball_all)
        self.detect_all_btn = QPushButton("🎥 Alle Frames erkennen")
        self.detect_all_btn.setToolTip("YOLO-Erkennung auf allen Frames beider Videos (Ctrl+Shift+D)")
        self.detect_all_btn.clicked.connect(self._detect_all_frames)
        self.cancel_batch_btn = QPushButton("❌ Abbrechen")
        self.cancel_batch_btn.setToolTip("Laufende Batch-Erkennung abbrechen")
        self.cancel_batch_btn.clicked.connect(self._cancel_batch)
        self.cancel_batch_btn.setEnabled(False)
        self.interpolate_btn = QPushButton("↔ Interpolieren")
        self.interpolate_btn.setToolTip("Marker zwischen vorhandenen Markern linear interpolieren (Ctrl+I)")
        self.interpolate_btn.clicked.connect(self._interpolate_all)
        tools.addWidget(self.detect_btn, stretch=1)
        tools.addWidget(self.detect_all_btn, stretch=1)
        tools.addWidget(self.cancel_batch_btn)
        tools.addWidget(self.interpolate_btn, stretch=1)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._brand_header = BrandHeaderWidget(subtitle="Ball Marker", tone="brand")
        layout.addWidget(self._brand_header)
        self._video_layout = QHBoxLayout()
        self._video_layout.addWidget(self.left_panel, stretch=1)
        self._video_layout.addWidget(self.right_panel, stretch=1)
        layout.addLayout(self._video_layout)
        layout.addLayout(controls)
        layout.addLayout(tools)
        layout.addWidget(self.timeline)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self._setup_menu()

        # ── Statusbar ─────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._kf_status_label = QLabel("")
        self._statusbar.addPermanentWidget(self._kf_status_label)
        self._kf_status_timer = QTimer(self)
        self._kf_status_timer.setSingleShot(True)
        self._kf_status_timer.timeout.connect(lambda: self._kf_status_label.setText(""))

        # ── Fortschrittsanzeige ────────────────────────────────────
        self._progress = ProgressWidget()
        self._statusbar.addPermanentWidget(self._progress)
        self._progress.cancel_requested.connect(self._on_progress_cancel)

        # Auto-Resize wenn ein Video geladen wird
        self.left_panel.video_loaded.connect(self._on_video_loaded)
        self.right_panel.video_loaded.connect(self._on_video_loaded)

        # Keyframe-Status Signale
        self.left_panel.keyframes_status.connect(self._on_kf_status)
        self.right_panel.keyframes_status.connect(self._on_kf_status)
        self.left_panel.keyframes_ready.connect(self._on_kf_ready)
        self.right_panel.keyframes_ready.connect(self._on_kf_ready)

        # Status-Signale von Panels
        self.left_panel.status_message.connect(self._on_panel_status)
        self.right_panel.status_message.connect(self._on_panel_status)

        # Fortschritts-Signale von Panels
        for panel in (self.left_panel, self.right_panel):
            panel.task_started.connect(self._on_task_started)
            panel.batch_progress.connect(self._on_batch_progress)
            panel.task_finished.connect(self._on_task_finished)
            panel.markers_changed.connect(self._on_markers_changed)

        # Autosave-Recovery beim Start prüfen
        self._check_autosave_recovery()
        self.autosave.start()

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Datei")

        import_action = QAction("Importieren...", self)
        import_action.setShortcut(QKeySequence("Ctrl+O"))
        import_action.triggered.connect(self.import_session)
        file_menu.addAction(import_action)

        export_action = QAction("Exportieren...", self)
        export_action.setShortcut(QKeySequence("Ctrl+S"))
        export_action.triggered.connect(self.export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_action.triggered.connect(self.undo)
        file_menu.addAction(self.undo_action)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        self.redo_action.triggered.connect(self.redo)
        file_menu.addAction(self.redo_action)
        self.update_undo_redo_actions()

        # ── Werkzeuge-Menü ───────────────────────────────────────
        tools_menu = menubar.addMenu("Werkzeuge")

        detect_action = QAction("Ball erkennen (YOLO)", self)
        detect_action.setShortcut(QKeySequence("Ctrl+D"))
        detect_action.triggered.connect(self._detect_ball_all)
        tools_menu.addAction(detect_action)

        detect_all_action = QAction("Alle Frames erkennen (YOLO)", self)
        detect_all_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
        detect_all_action.triggered.connect(self._detect_all_frames)
        tools_menu.addAction(detect_all_action)

        interpolate_action = QAction("Marker interpolieren", self)
        interpolate_action.setShortcut(QKeySequence("Ctrl+I"))
        interpolate_action.triggered.connect(self._interpolate_all)
        tools_menu.addAction(interpolate_action)

        tools_menu.addSeparator()

        reset_action = QAction("Marker zurücksetzen…", self)
        reset_action.triggered.connect(self._reset_markers)
        tools_menu.addAction(reset_action)

        tools_menu.addSeparator()

        field_cal_action = QAction("Feldkalibrierung laden…", self)
        field_cal_action.triggered.connect(self._load_field_calibration)
        tools_menu.addAction(field_cal_action)

        field_create_action = QAction("Feldkalibrierung erstellen/bearbeiten…", self)
        field_create_action.triggered.connect(self._create_field_calibration)
        tools_menu.addAction(field_create_action)

        field_export_action = QAction("Feldkalibrierung exportieren…", self)
        field_export_action.triggered.connect(self._export_field_calibration)
        tools_menu.addAction(field_export_action)

        field_clear_action = QAction("Feldgrenze entfernen", self)
        field_clear_action.triggered.connect(self._clear_field_boundary)
        tools_menu.addAction(field_clear_action)

        tools_menu.addSeparator()

        export_training_action = QAction("Trainingsdaten exportieren…", self)
        export_training_action.triggered.connect(self._export_training_data)
        tools_menu.addAction(export_training_action)

        load_model_action = QAction("Eigenes Modell laden…", self)
        load_model_action.triggered.connect(self._load_custom_model)
        tools_menu.addAction(load_model_action)

        # ── Navigation-Menü ──────────────────────────────────────
        nav_menu = menubar.addMenu("Navigation")

        nav_menu.addAction(self._make_action("Nächster Marker", "Ctrl+Right", lambda: self._jump_marker_all(1)))
        nav_menu.addAction(self._make_action("Vorheriger Marker", "Ctrl+Left", lambda: self._jump_marker_all(-1)))
        nav_menu.addSeparator()
        nav_menu.addAction(self._make_action("Nächster manueller Marker", "Ctrl+Shift+Right", lambda: self._jump_typed_all(1, "manual")))
        nav_menu.addAction(self._make_action("Vorheriger manueller Marker", "Ctrl+Shift+Left", lambda: self._jump_typed_all(-1, "manual")))
        nav_menu.addSeparator()
        nav_menu.addAction(self._make_action("Nächster interpolierter Marker", "", lambda: self._jump_typed_all(1, "interpolated")))
        nav_menu.addAction(self._make_action("Vorheriger interpolierter Marker", "", lambda: self._jump_typed_all(-1, "interpolated")))
        nav_menu.addSeparator()
        nav_menu.addAction(self._make_action("Nächster YOLO-Marker", "", lambda: self._jump_typed_all(1, "yolo")))
        nav_menu.addAction(self._make_action("Vorheriger YOLO-Marker", "", lambda: self._jump_typed_all(-1, "yolo")))
        nav_menu.addSeparator()
        nav_menu.addAction(self._make_action("Nächste Lücke (kein Marker)", "", lambda: self._jump_unmarked_all(1)))
        nav_menu.addAction(self._make_action("Vorherige Lücke (kein Marker)", "", lambda: self._jump_unmarked_all(-1)))

        nav_menu.addSeparator()
        nav_menu.addAction(self._make_action(
            "Nächsten Marker auf Frame hervorheben", "Tab",
            lambda: self._focus_marker_on_frame(1)))
        nav_menu.addAction(self._make_action(
            "Vorherigen Marker auf Frame hervorheben", "Shift+Tab",
            lambda: self._focus_marker_on_frame(-1)))
        nav_menu.addAction(self._make_action(
            "Marker-Hervorhebung entfernen / Zoom zurück", "Escape",
            self._clear_marker_highlight))

    def _make_action(self, text, shortcut, callback):
        """Hilfsmethode: Erstellt eine QAction mit optionalem Shortcut."""
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        return action

    def update_undo_redo_actions(self):
        video_loaded = self.left_panel.has_video or self.right_panel.has_video
        undo_stack_len = len(self.session.undo_stack)
        redo_stack_len = len(self.session.redo_stack)
        undo_enabled = video_loaded and undo_stack_len > 0
        redo_enabled = video_loaded and redo_stack_len > 0
        self.undo_action.setEnabled(undo_enabled)
        self.redo_action.setEnabled(redo_enabled)
        self.menuBar().update()
        self.menuBar().repaint()

    # ── Video-Pfade für Autosave ────────────────────────────────────

    def _get_loaded_video_paths(self) -> list[str]:
        """Gibt die Pfade der aktuell geladenen Videos zurück (für Autosave)."""
        paths = []
        for panel in (self.left_panel, self.right_panel):
            if panel.has_video:
                paths.append(panel.player.source().toString())
            else:
                paths.append("")
        return paths

    # ── Auto-Resize bei Video-Laden ───────────────────────────────────

    def _on_video_loaded(self, resolution: QSize):
        """Passt die Fenstergröße automatisch an die Video-Auflösung an."""
        # Keyframes noch nicht verfügbar
        self._keyframes_available = False

        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if not screen:
            return
        available = screen.availableGeometry()
        max_w = int(available.width() * 0.9)
        max_h = int(available.height() * 0.9)

        # Zwei Panels nebeneinander: doppelte Breite
        num_panels = 2
        video_w = resolution.width() * num_panels
        video_h = resolution.height()

        # Platz für UI-Elemente (Menübar, Buttons, Timeline, Margins)
        ui_extra_h = 200
        ui_extra_w = 60

        target_w = video_w + ui_extra_w
        target_h = video_h + ui_extra_h

        # Herunterskalieren wenn nötig, Seitenverhältnis beibehalten
        if target_w > max_w or target_h > max_h:
            scale = min(max_w / target_w, max_h / target_h)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)

        # Mindestgröße
        target_w = max(target_w, 800)
        target_h = max(target_h, 500)

        self.resize(target_w, target_h)

        # Beide Panels gleichmäßig verteilen (wichtig wenn 2. Video
        # nachträglich geladen wird und resize ein No-Op ist)
        self._video_layout.invalidate()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._equalize_panels)

        # Ggf. gespeicherte Feldkalibrierung auf das neue Video anwenden
        from autosave.autosave import DEFAULT_FIELD_CALIBRATION_PATH
        if os.path.isfile(DEFAULT_FIELD_CALIBRATION_PATH):
            sender_panel = self.sender()
            if sender_panel and hasattr(sender_panel, 'load_field_calibration'):
                if sender_panel._field_boundary is None:
                    sender_panel.load_field_calibration(DEFAULT_FIELD_CALIBRATION_PATH)

        # Sofort speichern wenn sich der Video-Zustand ändert
        self.autosave.save()

    def _equalize_panels(self):
        """Erzwingt gleichmäßige Breitenverteilung beider Video-Panels."""
        half = self.left_panel.parent().width() // 2
        self.left_panel.setMinimumWidth(half)
        self.right_panel.setMinimumWidth(half)
        # Einschränkung sofort wieder aufheben, damit Resize frei bleibt
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._release_panel_constraints)

    def _release_panel_constraints(self):
        """Hebt die temporäre Mindestbreite beider Panels wieder auf."""
        self.left_panel.setMinimumWidth(0)
        self.right_panel.setMinimumWidth(0)

    def _on_sync_offset_changed(self, value: int):
        """Wird aufgerufen wenn der Sync-Offset geändert wird."""
        self._sync_offset_frames = value
        self.timeline.sync_offset = value
        # Rechtes Video sofort an neue Position anpassen
        if self.right_panel.has_video and self.left_panel.has_video:
            left_frame = self.left_panel.current_frame()
            self.right_panel.set_frame(left_frame + self._sync_offset_frames)
        self.autosave.save()

    # ── Zentrales Play/Pause ──────────────────────────────────────────

    def _toggle_play_all(self):
        """Startet/pausiert beide Videos gleichzeitig (robust bei nur einem Video)."""
        any_playing = self.left_panel.is_playing or self.right_panel.is_playing
        if any_playing:
            self.left_panel.pause()
            self.right_panel.pause()
        else:
            self.left_panel.play()
            self.right_panel.play()

    def _step_all(self, n):
        """Springt beide Videos um n Frames (positiv=vorwärts, negativ=rückwärts)."""
        if n > 0:
            self.left_panel.step_forward(n)
            self.right_panel.step_forward(n)
        else:
            self.left_panel.step_backward(-n)
            self.right_panel.step_backward(-n)
        # Sync-Offset auf rechtes Panel anwenden
        if self._sync_offset_frames != 0 and self.right_panel.has_video:
            left_frame = self.left_panel.current_frame()
            self.right_panel.set_frame(left_frame + self._sync_offset_frames)
        self.timeline.selected_frame = self.left_panel.current_frame()
        self.timeline.update()

    def _jump_selected(self, direction):
        """Springt gemäß dem aktuell im Dropdown gewählten Typ."""
        jump_type = self.jump_type_combo.currentData()
        if jump_type == "any":
            self._jump_marker_all(direction)
        elif jump_type in ("manual", "yolo", "interpolated", "exclusion"):
            self._jump_typed_all(direction, jump_type)
        elif jump_type == "gap":
            self._jump_unmarked_all(direction)
        elif jump_type == "keyframe":
            self._jump_keyframe_all(direction)

    def _jump_marker_all(self, direction):
        """Springt beide Videos zum nächsten/vorherigen Marker-Frame."""
        if direction > 0:
            self.left_panel.jump_to_next_marker_frame()
            self.right_panel.jump_to_next_marker_frame()
        else:
            self.left_panel.jump_to_prev_marker_frame()
            self.right_panel.jump_to_prev_marker_frame()
        self.timeline.selected_frame = self.left_panel.current_frame()
        self.timeline.update()

    def _jump_keyframe_all(self, direction):
        """Springt beide Videos zum nächsten/vorherigen Keyframe."""
        if not self._kf_buttons_enabled():
            return
        if direction > 0:
            self.left_panel.jump_to_next_keyframe()
            self.right_panel.jump_to_next_keyframe()
        else:
            self.left_panel.jump_to_prev_keyframe()
            self.right_panel.jump_to_prev_keyframe()
        self.timeline.selected_frame = self.left_panel.current_frame()

    def _jump_typed_all(self, direction, marker_type):
        """Springt beide Videos zum nächsten/vorherigen Frame mit bestimmtem Marker-Typ."""
        method_map = {
            ("manual", 1): "jump_to_next_manual_frame",
            ("manual", -1): "jump_to_prev_manual_frame",
            ("interpolated", 1): "jump_to_next_interpolated_frame",
            ("interpolated", -1): "jump_to_prev_interpolated_frame",
            ("yolo", 1): "jump_to_next_yolo_frame",
            ("yolo", -1): "jump_to_prev_yolo_frame",
            ("exclusion", 1): "jump_to_next_exclusion_frame",
            ("exclusion", -1): "jump_to_prev_exclusion_frame",
        }
        method_name = method_map.get((marker_type, direction))
        if method_name:
            getattr(self.left_panel, method_name)()
            getattr(self.right_panel, method_name)()
        self.timeline.selected_frame = self.left_panel.current_frame()
        self.timeline.update()

    def _jump_unmarked_all(self, direction):
        """Springt beide Videos zum nächsten/vorherigen Frame ohne Marker."""
        if direction > 0:
            self.left_panel.jump_to_next_unmarked_frame()
            self.right_panel.jump_to_next_unmarked_frame()
        else:
            self.left_panel.jump_to_prev_unmarked_frame()
            self.right_panel.jump_to_prev_unmarked_frame()
        self.timeline.selected_frame = self.left_panel.current_frame()
        self.timeline.update()

    # ── Marker auf Frame hervorheben ──────────────────────────────

    def _focus_marker_on_frame(self, direction):
        """Hebt den nächsten/vorherigen Marker auf dem aktuellen Frame hervor."""
        if direction > 0:
            self.left_panel.focus_next_marker()
            self.right_panel.focus_next_marker()
        else:
            self.left_panel.focus_prev_marker()
            self.right_panel.focus_prev_marker()

    def _clear_marker_highlight(self):
        """Entfernt Highlight und setzt Zoom zurück."""
        self.left_panel._remove_highlight()
        self.right_panel._remove_highlight()
        self.left_panel.reset_zoom()
        self.right_panel.reset_zoom()

    # ── Keyframe-Status ───────────────────────────────────────────

    def _kf_buttons_enabled(self):
        return getattr(self, '_keyframes_available', False)

    def _on_kf_status(self, text: str):
        """Wird aufgerufen wenn sich der Keyframe-Ladestatus ändert."""
        self._kf_status_label.setText(text)
        # Timer stoppen falls aktiv
        self._kf_status_timer.stop()

    def _on_kf_ready(self):
        """Wird aufgerufen wenn Keyframes fertig geladen sind."""
        self._keyframes_available = True
        # Statustext nach 3 Sekunden ausblenden
        self._kf_status_timer.start(3000)

    def _on_panel_status(self, text: str):
        """Status-Meldung von einem Panel (YOLO, Interpolation, etc.)."""
        self._statusbar.showMessage(text, 5000)

    # ── Fortschrittsanzeige ───────────────────────────────────────────

    def _on_task_started(self, task_id: str, label: str, total: int, cancellable: bool):
        self._progress.start_task(task_id, label, total, cancellable)

    def _on_batch_progress(self, task_id: str, current: int, total: int, detail: str):
        self._progress.update_task(task_id, current, detail)

    def _on_task_finished(self, task_id: str, message: str):
        self._progress.finish_task(task_id, message)

    def _on_progress_cancel(self, task_id: str):
        """Wird aufgerufen wenn der Benutzer einen Task über die Fortschrittsanzeige abbricht."""
        self.left_panel.cancel_batch_detection()
        self.right_panel.cancel_batch_detection()

    def _on_markers_changed(self):
        """Wird aufgerufen wenn sich Marker in einem Panel geändert haben."""
        # Timeline aktuell halten (Frame-Position aus dem linken Panel)
        if self.left_panel.has_video:
            self.timeline.selected_frame = self.left_panel.current_frame()
        self.timeline.update()
        # Sofort speichern bei Marker-Änderung
        self.autosave.save()

    # ── YOLO + Interpolation ──────────────────────────────────────────

    def _detect_ball_all(self):
        """Startet YOLO-Erkennung auf dem aktuellen Frame beider Panels."""
        self.left_panel.detect_ball()
        self.right_panel.detect_ball()

    def _detect_all_frames(self):
        """Startet YOLO-Erkennung auf allen Frames beider Videos."""
        if not self.left_panel.has_video and not self.right_panel.has_video:
            self._statusbar.showMessage("Kein Video geladen.", 3000)
            return

        # ── Dialog: welche Marker-Typen überspringen? ─────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("YOLO-Erkennung – alle Frames")
        dlg_layout = QVBoxLayout(dlg)

        info = QLabel(
            "YOLO-Erkennung auf allen Frames beider Videos starten.\n"
            "Dies kann je nach Videolänge mehrere Minuten dauern.\n"
        )
        info.setWordWrap(True)
        dlg_layout.addWidget(info)

        skip_group = QGroupBox("Frames mit diesen Marker-Typen überspringen:")
        skip_layout = QVBoxLayout(skip_group)
        cb_manual = QCheckBox("Manuell gesetzte Marker")
        cb_yolo = QCheckBox("YOLO-erkannte Marker")
        cb_interp = QCheckBox("Interpolierte Marker")
        cb_manual.setChecked(True)
        cb_yolo.setChecked(True)
        cb_interp.setChecked(True)
        skip_layout.addWidget(cb_manual)
        skip_layout.addWidget(cb_yolo)
        skip_layout.addWidget(cb_interp)
        dlg_layout.addWidget(skip_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        skip_types: set[str] = set()
        if cb_manual.isChecked():
            skip_types.add("manual")
        if cb_yolo.isChecked():
            skip_types.add("yolo")
        if cb_interp.isChecked():
            skip_types.add("interpolated")

        self.cancel_batch_btn.setEnabled(True)
        self.detect_all_btn.setEnabled(False)
        self.left_panel.detect_all_frames(skip_types)
        self.right_panel.detect_all_frames(skip_types)
        # Timer um Button-Status zu überwachen
        self._batch_check_timer = QTimer(self)
        self._batch_check_timer.timeout.connect(self._check_batch_running)
        self._batch_check_timer.start(500)

    def _cancel_batch(self):
        """Bricht die laufende Batch-Erkennung ab."""
        self.left_panel.cancel_batch_detection()
        self.right_panel.cancel_batch_detection()

    def _check_batch_running(self):
        """Prüft ob die Batch-Erkennung noch läuft."""
        still_running = self.left_panel._batch_running or self.right_panel._batch_running
        if not still_running:
            self.cancel_batch_btn.setEnabled(False)
            self.detect_all_btn.setEnabled(True)
            self._batch_check_timer.stop()
            self.timeline.update()

    def _interpolate_all(self):
        """Interpoliert Marker zwischen vorhandenen Markern in beiden Panels."""
        n1 = self.left_panel.interpolate_markers()
        n2 = self.right_panel.interpolate_markers()
        total = n1 + n2
        if total > 0:
            self.timeline.update()
            self.update_undo_redo_actions()

    def closeEvent(self, event):
        """Beim Schließen den aktuellen Zustand speichern."""
        self.autosave.save()
        self.autosave.stop()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        """Globale Tastatursteuerung für Wiedergabe und Frame-Navigation."""
        key = event.key()
        mod = event.modifiers()
        # Space → Play/Pause
        if key == Qt.Key.Key_Space:
            self._toggle_play_all()
            event.accept()
            return
        # Pfeiltaste rechts
        if key == Qt.Key.Key_Right:
            if mod & Qt.KeyboardModifier.ControlModifier:
                self._jump_selected(1)
            else:
                n = 25 if mod & Qt.KeyboardModifier.ShiftModifier else 1
                self._step_all(n)
            event.accept()
            return
        # Pfeiltaste links
        if key == Qt.Key.Key_Left:
            if mod & Qt.KeyboardModifier.ControlModifier:
                self._jump_selected(-1)
            else:
                n = 25 if mod & Qt.KeyboardModifier.ShiftModifier else 1
                self._step_all(-n)
            event.accept()
            return
        # Ctrl+0 → Zoom zurücksetzen
        if key == Qt.Key.Key_0 and mod & Qt.KeyboardModifier.ControlModifier:
            self.left_panel.reset_zoom()
            self.right_panel.reset_zoom()
            event.accept()
            return
        # Escape → Highlight entfernen + Zoom zurück
        if key == Qt.Key.Key_Escape:
            self._clear_marker_highlight()
            event.accept()
            return
        super().keyPressEvent(event)

    def export(self):
        from autosave.autosave import DEFAULT_EXPORT_DIR
        os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
        filename, _ = QFileDialog.getSaveFileName(self, "Exportieren", os.path.join(DEFAULT_EXPORT_DIR, "ballmarker.json"), "JSON Files (*.json)")
        if filename:
            saved_path = export_markers(self.session.markers, filename, sync_offset_frames=self._sync_offset_frames)
            QMessageBox.information(self, "Export", f"Export erfolgreich: {saved_path}")
        self.update_undo_redo_actions()

    def import_session(self):
        """Importiert eine zuvor exportierte JSON-Datei und stellt die Session wieder her."""
        from autosave.autosave import DEFAULT_EXPORT_DIR
        filename, _ = QFileDialog.getOpenFileName(self, "Importieren", DEFAULT_EXPORT_DIR, "JSON Files (*.json)")
        if not filename:
            return
        self._load_markers_from_file(filename)

    def _load_markers_from_file(self, filename, silent=False):
        """Lädt Marker aus einer JSON-Datei und stellt Videos + Marker wieder her."""
        try:
            markers = import_markers(filename)
        except Exception as e:
            QMessageBox.warning(self, "Import-Fehler", f"Datei konnte nicht geladen werden:\n{e}")
            return
        if not markers:
            if not silent:
                QMessageBox.information(self, "Import", "Keine Marker in der Datei gefunden.")
            return

        # Bestehende Marker ersetzen
        self.session.markers.clear()
        self.session.undo_stack.clear()
        self.session.redo_stack.clear()
        self.session.markers.extend(markers)

        # Alle referenzierten Videos ermitteln
        video_files = sorted(set(m.video_file for m in markers))

        # Videos in die Panels laden  (erstes → links, zweites → rechts)
        panels = [self.left_panel, self.right_panel]
        for i, panel in enumerate(panels):
            if i < len(video_files):
                vf = video_files[i]
                # file:// URI → lokaler Pfad
                local_path = QUrl(vf).toLocalFile() if vf.startswith("file://") else vf
                if local_path and os.path.isfile(local_path):
                    panel.load_video(local_path)

        # KF-Status zurücksetzen (werden nach Analyse verfügbar)
        self._keyframes_available = False

        # Marker-Darstellung synchronisieren
        self.left_panel.sync_markers_with_session()
        self.right_panel.sync_markers_with_session()
        self.timeline.update()
        self.update_undo_redo_actions()
        if not silent:
            QMessageBox.information(self, "Import", f"{len(markers)} Marker aus {len(video_files)} Video(s) geladen.")

    def _check_autosave_recovery(self):
        """Prüft beim Start ob eine Autosave-Datei vorhanden ist und bietet Wiederherstellung an."""
        if not self.autosave.has_recovery():
            return
        reply = QMessageBox.question(
            self,
            "Sitzung wiederherstellen",
            "Es wurde eine automatisch gespeicherte Sitzung gefunden.\n"
            "Möchten Sie die letzte Sitzung wiederherstellen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._restore_full_session()
        else:
            self.autosave.clear()

    def _restore_full_session(self):
        """Stellt die komplette Sitzung wieder her (Videos, Marker, Kalibrierung)."""
        session_data = self.autosave.load_session_data()

        # 1. Videos laden
        loaded_videos = session_data.get("loaded_videos", [])
        panels = [self.left_panel, self.right_panel]
        for i, panel in enumerate(panels):
            if i < len(loaded_videos) and loaded_videos[i]:
                vf = loaded_videos[i]
                local_path = QUrl(vf).toLocalFile() if vf.startswith("file://") else vf
                if local_path and os.path.isfile(local_path):
                    panel.load_video(local_path)

        # 2. Marker laden
        markers = self.autosave.recover()
        if markers:
            self.session.markers.clear()
            self.session.undo_stack.clear()
            self.session.redo_stack.clear()
            self.session.markers.extend(markers)

            # Falls Videos nicht aus loaded_videos kamen, aus Markern ableiten
            if not loaded_videos:
                video_files = sorted(set(m.video_file for m in markers))
                for i, panel in enumerate(panels):
                    if i < len(video_files) and not panel.has_video:
                        vf = video_files[i]
                        local_path = QUrl(vf).toLocalFile() if vf.startswith("file://") else vf
                        if local_path and os.path.isfile(local_path):
                            panel.load_video(local_path)

            self.left_panel.sync_markers_with_session()
            self.right_panel.sync_markers_with_session()
            self.timeline.update()
            self.update_undo_redo_actions()

        # 3. Sync-Offset wiederherstellen
        sync_offset = session_data.get("sync_offset_frames", 0)
        if sync_offset:
            self._sync_offset_spin.setValue(sync_offset)

        # 4. Feldkalibrierung wiederherstellen
        from autosave.autosave import DEFAULT_FIELD_CALIBRATION_PATH
        if os.path.isfile(DEFAULT_FIELD_CALIBRATION_PATH):
            self._restore_field_calibration(DEFAULT_FIELD_CALIBRATION_PATH)

    # ── Feldkalibrierung ──────────────────────────────────────────

    def _load_field_calibration(self):
        """Öffnet einen Dialog zum Laden einer field_calibration.json."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Feldkalibrierung laden", "",
            "JSON-Dateien (*.json);;Alle Dateien (*)")
        if not path:
            return
        loaded = 0
        for panel in (self.left_panel, self.right_panel):
            if panel.has_video:
                if panel.load_field_calibration(path):
                    loaded += 1
        if loaded == 0:
            self.statusBar().showMessage("Keine passende Kamera zugeordnet – bitte Videos zuerst laden", 5000)
        else:
            # Externe Kalibrierung in den Standard-Pfad kopieren
            from autosave.autosave import DEFAULT_FIELD_CALIBRATION_PATH
            if os.path.normpath(path) != os.path.normpath(DEFAULT_FIELD_CALIBRATION_PATH):
                import shutil
                os.makedirs(os.path.dirname(DEFAULT_FIELD_CALIBRATION_PATH), exist_ok=True)
                shutil.copy2(path, DEFAULT_FIELD_CALIBRATION_PATH)

    def _clear_field_boundary(self):
        """Entfernt die Feldgrenze von beiden Panels."""
        for panel in (self.left_panel, self.right_panel):
            panel.clear_field_boundary()

    def _restore_field_calibration(self, cal_path: str):
        """Stellt die Feldkalibrierung für alle geladenen Videos wieder her."""
        loaded = 0
        for panel in (self.left_panel, self.right_panel):
            if panel.has_video:
                if panel.load_field_calibration(cal_path):
                    loaded += 1
        if loaded:
            self.statusBar().showMessage(f"Feldkalibrierung wiederhergestellt ({loaded} Kamera(s))", 3000)

    # ── Feldkalibrierung erstellen ────────────────────────────────

    def _create_field_calibration(self):
        """Öffnet den interaktiven Kalibrierungsdialog für die aktive Kameraseite."""
        from calibration.calibration_dialog import FieldCalibrationDialog
        from PySide6.QtCore import QUrl

        # Video-Pfade beider Kameras sammeln
        video_paths = {}
        frame_index = 0
        camera_id = 0

        if self.left_panel.has_video:
            video_paths[0] = QUrl(self.left_panel.player.source().toString()).toLocalFile()
        if self.right_panel.has_video:
            video_paths[1] = QUrl(self.right_panel.player.source().toString()).toLocalFile()

        if not video_paths:
            QMessageBox.warning(self, "Feldkalibrierung",
                                "Bitte zuerst ein Video laden.")
            return

        # Initiale Kamera: bevorzugt links, sonst rechts
        if 0 in video_paths:
            camera_id = 0
            frame_index = self.left_panel.current_frame()
        else:
            camera_id = 1
            frame_index = self.right_panel.current_frame()

        from autosave.autosave import DEFAULT_FIELD_CALIBRATION_PATH
        cal_path = DEFAULT_FIELD_CALIBRATION_PATH

        dlg = FieldCalibrationDialog(
            parent=self,
            video_path=video_paths.get(camera_id, ""),
            camera_id=camera_id,
            frame_index=frame_index,
            calibration_path=cal_path,
            video_paths=video_paths,
        )
        dlg.calibration_saved.connect(self._on_calibration_saved)
        dlg.exec()

    def _on_calibration_saved(self, path: str):
        """Wird aufgerufen wenn der Kalibrierungsdialog gespeichert hat."""
        # Kalibrierung in beiden Panels neu laden
        loaded = 0
        for panel in (self.left_panel, self.right_panel):
            if panel.has_video:
                if panel.load_field_calibration(path):
                    loaded += 1
        if loaded:
            self.statusBar().showMessage(
                f"Feldkalibrierung gespeichert und angewendet ({loaded} Kamera(s))", 5000)
        else:
            self.statusBar().showMessage("Feldkalibrierung gespeichert", 3000)

    def _export_field_calibration(self):
        """Exportiert die aktuelle Feldkalibrierung."""
        # Bestehende Kalibrierung finden
        from autosave.autosave import DEFAULT_FIELD_CALIBRATION_PATH
        cal_path = DEFAULT_FIELD_CALIBRATION_PATH
        if not os.path.isfile(cal_path):
            QMessageBox.information(
                self, "Export",
                "Keine Feldkalibrierung vorhanden.\n"
                "Bitte zuerst eine Kalibrierung erstellen oder laden.")
            return

        from autosave.autosave import DEFAULT_EXPORT_DIR
        dest, _ = QFileDialog.getSaveFileName(
            self, "Feldkalibrierung exportieren",
            os.path.join(DEFAULT_EXPORT_DIR, "field_calibration.json"),
            "JSON-Dateien (*.json)")
        if not dest:
            return

        import shutil
        try:
            shutil.copy2(cal_path, dest)
            QMessageBox.information(self, "Export",
                                   f"Feldkalibrierung exportiert:\n{dest}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Export fehlgeschlagen:\n{e}")

    # ── Marker zurücksetzen ────────────────────────────────────────

    def _export_training_data(self):
        """Exportiert aktuelle Marker als YOLO-Trainingsdaten."""
        from autosave.autosave import DEFAULT_EXPORT_DIR
        from training.export_training_data import export_yolo_dataset

        if not self.session.markers:
            QMessageBox.warning(self, "Trainingsdaten", "Keine Marker vorhanden.\nBitte zuerst Bälle markieren.")
            return

        # Erst Marker als JSON exportieren (temporär)
        import tempfile
        tmp_json = os.path.join(tempfile.gettempdir(), "ballmarker_training_export.json")
        from export.exporter import export_markers as _export
        _export(self.session.markers, tmp_json)

        # Zielverzeichnis wählen
        output_dir = QFileDialog.getExistingDirectory(
            self, "Trainingsdaten-Verzeichnis wählen",
            os.path.join(DEFAULT_EXPORT_DIR, "yolo_dataset"))
        if not output_dir:
            return

        try:
            stats = export_yolo_dataset(tmp_json, output_dir)
            QMessageBox.information(
                self, "Trainingsdaten",
                f"Export erfolgreich!\n\n"
                f"Frames: {stats['total_frames']}\n"
                f"Marker: {stats['total_markers']}\n"
                f"Train: {stats['train']} Frames\n"
                f"Val: {stats['val']} Frames\n\n"
                f"Zum Trainieren:\n"
                f"  python -m training.train_model {os.path.join(output_dir, 'dataset.yaml')}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Export fehlgeschlagen:\n{e}")
        finally:
            if os.path.isfile(tmp_json):
                os.remove(tmp_json)

    def _load_custom_model(self):
        """Lädt ein benutzerdefiniertes YOLO-Modell (.pt)."""
        from detection.ball_detector import load_custom_model, CUSTOM_MODEL_PATH
        models_dir = os.path.dirname(CUSTOM_MODEL_PATH)
        path, _ = QFileDialog.getOpenFileName(
            self, "YOLO-Modell laden", models_dir, "PyTorch Modell (*.pt)")
        if not path:
            return
        try:
            load_custom_model(path)
            QMessageBox.information(
                self, "Modell geladen",
                f"Custom-Modell geladen:\n{os.path.basename(path)}\n\n"
                f"YOLO-Erkennung nutzt jetzt dieses Modell.")
            self._statusbar.showMessage(f"Custom-Modell: {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden:\n{e}")

    def _reset_markers(self):
        """Öffnet einen Dialog zum gezielten Löschen von Markern."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Marker zurücksetzen")
        dlg.setMinimumWidth(340)
        layout = QVBoxLayout(dlg)

        # ── Geltungsbereich ───────────────────────────────────────
        scope_group = QGroupBox("Geltungsbereich")
        scope_layout = QVBoxLayout(scope_group)
        scope_btn_group = QButtonGroup(dlg)

        rb_all_videos = QRadioButton("Alle Videos")
        rb_all_videos.setChecked(True)
        scope_btn_group.addButton(rb_all_videos, 0)
        scope_layout.addWidget(rb_all_videos)

        # Radio-Buttons für jedes geladene Video
        video_ids = []
        for panel, label in [(self.left_panel, "Video links"), (self.right_panel, "Video rechts")]:
            if panel.has_video:
                vid = panel.player.source().toString()
                video_ids.append(vid)
                # Kurzen Dateinamen extrahieren
                short = QUrl(vid).fileName() if vid.startswith("file://") else vid.rsplit("/", 1)[-1]
                rb = QRadioButton(f"Nur {label} ({short})")
                scope_btn_group.addButton(rb, len(video_ids))  # 1-basiert
                scope_layout.addWidget(rb)

        layout.addWidget(scope_group)

        # ── Marker-Typen ─────────────────────────────────────────
        type_group = QGroupBox("Welche Marker löschen?")
        type_layout = QVBoxLayout(type_group)

        cb_manual = QCheckBox("🔴 Manuelle Marker")
        cb_manual.setChecked(True)
        cb_yolo = QCheckBox("🔵 YOLO-Marker")
        cb_yolo.setChecked(True)
        cb_interp = QCheckBox("🟠 Interpolierte Marker")
        cb_interp.setChecked(True)

        type_layout.addWidget(cb_manual)
        type_layout.addWidget(cb_yolo)
        type_layout.addWidget(cb_interp)
        layout.addWidget(type_group)

        # ── Buttons ──────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Ausgewählte Typen sammeln
        types_to_delete = set()
        if cb_manual.isChecked():
            types_to_delete.add("manual")
        if cb_yolo.isChecked():
            types_to_delete.add("yolo")
        if cb_interp.isChecked():
            types_to_delete.add("interpolated")

        if not types_to_delete:
            return

        # Geltungsbereich bestimmen
        scope_id = scope_btn_group.checkedId()
        if scope_id == 0:
            scope_video = None  # alle
        else:
            scope_video = video_ids[scope_id - 1]

        # Marker filtern und löschen
        to_remove = [
            m for m in self.session.markers
            if m.type in types_to_delete
            and (scope_video is None or m.video_file == scope_video)
        ]

        if not to_remove:
            self._statusbar.showMessage("Keine passenden Marker gefunden.", 3000)
            return

        # Bestätigung
        type_names = []
        if "manual" in types_to_delete:
            type_names.append("manuelle")
        if "yolo" in types_to_delete:
            type_names.append("YOLO")
        if "interpolated" in types_to_delete:
            type_names.append("interpolierte")
        scope_text = "allen Videos" if scope_video is None else QUrl(scope_video).fileName()
        msg = f"{len(to_remove)} Marker ({', '.join(type_names)}) aus {scope_text} löschen?"
        reply = QMessageBox.question(
            self, "Marker zurücksetzen", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for m in to_remove:
            self.session.markers.remove(m)
        self.session.undo_stack.clear()
        self.session.redo_stack.clear()

        self.left_panel.sync_markers_with_session()
        self.right_panel.sync_markers_with_session()
        self.timeline.update()
        self.update_undo_redo_actions()
        self._statusbar.showMessage(f"{len(to_remove)} Marker gelöscht.", 5000)

    def undo(self):
        self.session.undo()
        self.left_panel.sync_markers_with_session()
        self.right_panel.sync_markers_with_session()
        self.timeline.update()
        self.update_undo_redo_actions()

    def redo(self):
        self.session.redo()
        self.left_panel.sync_markers_with_session()
        self.right_panel.sync_markers_with_session()
        self.timeline.update()
        self.update_undo_redo_actions()
