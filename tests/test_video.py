import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from video.video_player import VideoPlayer

def test_video_player_methods():
    vp = VideoPlayer("left")
    assert vp.current_frame() == 0
    assert vp.current_timestamp() == 0
    assert vp.total_frames() == 1  # Da kein Video geladen ist, sollte total_frames mindestens 1 sein
    vp.set_frame(10)
    # current_frame bleibt 0, da kein Video geladen ist
    assert vp.current_frame() == 0
