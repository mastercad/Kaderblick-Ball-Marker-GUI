from interpolation.strategy import InterpolationStrategy

class QuadraticInterpolation(InterpolationStrategy):
    def interpolate(self, marker1, marker2, frame):
        f1, f2 = marker1.frame_index, marker2.frame_index
        t = (frame - f1) / (f2 - f1)
        x = marker1.position[0] + t * (marker2.position[0] - marker1.position[0])
        y = marker1.position[1] + t * (marker2.position[1] - marker1.position[1]) + 0.1 * t * (1 - t)
        radius = marker1.radius + t * (marker2.radius - marker1.radius)
        return (x, y, radius)
    # Additional functionality for ballistic marker interpolation can be added here
