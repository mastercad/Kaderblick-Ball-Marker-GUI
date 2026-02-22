import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.marker import Marker
from interpolation.linear import LinearInterpolation
from interpolation.quadratic import QuadraticInterpolation

def test_linear_interpolation():
    m1 = Marker("video.mp4", 0, 0, (0.0, 0.0), 0.05, "manual")
    m2 = Marker("video.mp4", 10, 333, (1.0, 1.0), 0.10, "manual")
        x, y, r = interp.interpolate(m1, m2, 5)
    assert abs(x - 0.5) < 1e-6
    assert abs(y - 0.5) < 1e-6
    assert abs(r - 0.075) < 1e-6
