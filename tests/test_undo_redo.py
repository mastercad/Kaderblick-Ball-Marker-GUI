import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker
from model.session import Session

def test_undo_redo():
    s = Session()
    m = Marker("video.mp4", 1, 40, (0.1, 0.1), 0.05, "manual")
    s.add_marker(m)
    assert m in s.markers
    s.undo()
    assert m not in s.markers
    s.redo()
    assert m in s.markers
