
class Marker:
    def __init__(self, video_file: str, frame_index: int, timestamp_ms: int, position: tuple[float, float], radius: float, marker_type: str):
        self.video_file = video_file
        self.frame_index = frame_index
        self.timestamp_ms = timestamp_ms
        self.position = position  # (x, y), normalized
        self.radius = radius      # normalized
        self.type = marker_type   # "manual" or "interpolated"
