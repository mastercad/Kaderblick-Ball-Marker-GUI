import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker
from model.session import Session
from autosave.autosave import Autosave

def test_autosave(tmp_path):
    s = Session()
    s.autosave_path = tmp_path / "autosave.json"
    m = Marker("video.mp4", 1, 40, (0.1, 0.1), 0.05, "manual")
    s.add_marker(m)
    a = Autosave(s)
    a.save()
    assert os.path.exists(s.autosave_path)
