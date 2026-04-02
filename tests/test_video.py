import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    """Stellt sicher, dass eine QApplication existiert (nötig für QWidget-basierte Klassen)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_video_player_methods(qapp):
    from video.video_player import VideoPlayer
    vp = VideoPlayer("left")
    assert vp.current_frame() == 0
    assert vp.current_timestamp() == 0
    assert vp.total_frames() == 1  # Da kein Video geladen ist, sollte total_frames mindestens 1 sein
    vp.set_frame(10)
    # current_frame bleibt 0, da kein Video geladen ist
    assert vp.current_frame() == 0
