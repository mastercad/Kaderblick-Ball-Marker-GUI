from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from video.video_player import VideoPlayer
from ui.marker_overlay import MarkerOverlay

class VideoPanel(QWidget):
    def __init__(self, side, session):
        super().__init__()
        self.side = side
        self.session = session
        self.player = VideoPlayer(self.side)
        self.info_label = QLabel(f"{side.capitalize()} Video")
        self.frame_label = QLabel("Frame: 0")
        self.play_btn = QPushButton("Play/Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn = QPushButton("<")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn = QPushButton(">")
        self.next_btn.clicked.connect(self.next_frame)
        self.jump_btn = QPushButton("Jump Keyframe")
        self.jump_btn.clicked.connect(self.jump_keyframe)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.info_label)
        control_layout.addWidget(self.frame_label)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.jump_btn)
        layout = QVBoxLayout()
        layout.addLayout(control_layout)
        layout.addWidget(self.player.open_btn)
        layout.addWidget(self.player.video_widget)
        self.setLayout(layout)
        # Overlay direkt auf QVideoWidget
        self.overlay = MarkerOverlay(self.session, self.player)
        self.overlay.setParent(self.player.video_widget)
        self.overlay.setGeometry(self.player.video_widget.rect())
        self.overlay.show()
        self.overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.player.video_widget.rect())

    def toggle_play(self):
        self.player.toggle_play()
        self.update_frame_label()

    def prev_frame(self):
        self.player.set_frame(max(0, self.player.current_frame() - 1))
        self.update_frame_label()

    def next_frame(self):
        self.player.set_frame(self.player.current_frame() + 1)
        self.update_frame_label()

    def jump_keyframe(self):
        # Springe zum nächsten Keyframe
        current = self.player.current_frame()
        keyframes = sorted([m.frame_index for m in self.session.markers if m.video_file == self.player.video_file and m.type == "manual"])
        for kf in keyframes:
            if kf > current:
                self.player.set_frame(kf)
                self.update_frame_label()
                return
        if keyframes:
            self.player.set_frame(keyframes[0])
            self.update_frame_label()

    def update_frame_label(self):
        self.frame_label.setText(f"Frame: {self.player.current_frame()}")
