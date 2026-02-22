
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QSpinBox
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QUrl, QTimer

class VideoPlayer(QWidget):
    def __init__(self, side):
        super().__init__()
        self.side = side
        self.video_file = ""
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        self.open_btn = QPushButton("Video öffnen")
        self.open_btn.clicked.connect(self.open_video)
        self.info_label = QLabel("Kein Video geladen")
        self.offset_label = QLabel("Startoffset (Frames):")
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(0, 10000)
        self.offset_spin.setValue(0)
        self.offset_spin.valueChanged.connect(self.set_offset)
        self.offset = 0
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        layout = QVBoxLayout()
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(self.offset_label)
        offset_layout.addWidget(self.offset_spin)
        layout.addWidget(self.open_btn)
        layout.addLayout(offset_layout)
        layout.addWidget(self.video_widget)
        layout.addWidget(self.info_label)
        self.setLayout(layout)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Video öffnen", "", "Video Files (*.mp4 *.avi *.mov)")
        if filename:
            self.video_file = filename
            self.player.setSource(QUrl.fromLocalFile(filename))
            self.info_label.setText(filename)
            self.video_widget.show()
            self.player.setPosition(0)

    def current_frame(self):
        return int(self.player.position() / 40) - self.offset

    def current_timestamp(self):
        return self.player.position()

    def total_frames(self):
        return max(1, int(self.player.duration() / 40))

    def set_frame(self, frame):
        self.player.setPosition((frame + self.offset) * 40)

    def set_offset(self, value):
        self.offset = value

    def toggle_play(self):
        if self.playing:
            self.player.pause()
            self.timer.stop()
        else:
            self.player.play()
            self.timer.start(40)
        self.playing = not self.playing

    def _update_frame(self):
        # Synchronisiere Timeline
        pass
