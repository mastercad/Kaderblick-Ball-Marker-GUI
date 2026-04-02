
class Marker:
    def __init__(self, video_file: str, frame_index: int, timestamp_ms: int, position: tuple[float, float], radius: float, marker_type: str):
        self.video_file = video_file
        self.frame_index = frame_index
        self.timestamp_ms = timestamp_ms
        self.position = position  # (x, y), normalized
        self.radius = radius      # normalized
        self.type = marker_type   # "manual", "yolo", "interpolated", or "exclusion"

    def to_dict(self) -> dict:
        """Serialisiert den Marker in ein JSON-kompatibles dict."""
        return {
            "video_file": self.video_file,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "position": {"x": self.position[0], "y": self.position[1]},
            "radius": self.radius,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Marker":
        """Erzeugt einen Marker aus einem dict (z.B. aus JSON)."""
        pos = d["position"]
        return cls(
            video_file=d["video_file"],
            frame_index=int(d["frame_index"]),
            timestamp_ms=int(d["timestamp_ms"]),
            position=(float(pos["x"]), float(pos["y"])),
            radius=float(d["radius"]),
            marker_type=d.get("type", "manual"),
        )
