import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker
from model.session import Session
from export.exporter import export_markers
from autosave.autosave import Autosave

def test_full_flow(tmp_path):
    s = Session()
    s.autosave_path = tmp_path / "autosave.json"
    m1 = Marker("video.mp4", 1, 40, (0.1, 0.1), 0.05, "manual")
    m2 = Marker("video.mp4", 2, 80, (0.2, 0.2), 0.06, "interpolated")
    s.add_marker(m1)
    s.add_marker(m2)
    Autosave(s).save()
    export_markers(s.markers, tmp_path / "markers.json")
    assert len(s.markers) == 2
