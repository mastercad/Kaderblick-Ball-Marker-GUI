import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class _FakePlayer:
    def __init__(self):
        self._position = 0
        self._duration = 40

    def position(self):
        return self._position

    def duration(self):
        return self._duration

    def setPosition(self, value):
        self._position = value


class _FakeFpsDetector:
    ms_per_frame = 40
    fps = 25


def test_video_player_methods():
    from video.video_player import VideoPlayer

    vp = VideoPlayer.__new__(VideoPlayer)
    vp.player = _FakePlayer()
    vp._fps_detector = _FakeFpsDetector()
    vp.offset = 0

    assert vp.current_frame() == 0
    assert vp.current_timestamp() == 0
    assert vp.total_frames() == 1  # Da kein Video geladen ist, sollte total_frames mindestens 1 sein
    vp.set_frame(10)
    assert vp.current_frame() == 10
