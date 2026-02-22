import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker
from export.exporter import export_markers

def test_export(tmp_path):
    m = Marker("video.mp4", 10, 333, (0.5, 0.5), 0.05, "manual")
    filename = tmp_path / "markers.json"
    export_markers([m], filename)
    assert os.path.exists(filename)
