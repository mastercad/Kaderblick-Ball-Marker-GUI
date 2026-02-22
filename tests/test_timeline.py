import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker
from model.session import Session

def test_timeline_marker():
    s = Session()
    m = Marker("video.mp4", 5, 200, (0.2, 0.2), 0.05, "manual")
    s.add_marker(m)
    assert s.markers[0].frame_index == 5
