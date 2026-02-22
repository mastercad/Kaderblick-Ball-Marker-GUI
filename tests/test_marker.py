import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker

def test_marker_creation():
    m = Marker("video.mp4", 10, 333, (0.5, 0.5), 0.05, "manual")
    assert m.video_file == "video.mp4"
    assert m.frame_index == 10
    assert m.position == (0.5, 0.5)
    assert m.radius == 0.05
    assert m.type == "manual"
