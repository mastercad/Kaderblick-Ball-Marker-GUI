
import json
import threading
import time

class Autosave:
    def __init__(self, session):
        self.session = session
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._autosave_loop, daemon=True).start()

    def _autosave_loop(self):
        while self.running:
            self.save()
            time.sleep(30)

    def save(self):
        data = []
        for m in self.session.markers:
            data.append({
                "video_file": m.video_file,
                "frame_index": m.frame_index,
                "timestamp_ms": m.timestamp_ms,
                "position": {
                    "x": m.position[0],
                    "y": m.position[1]
                },
                "radius": m.radius,
                "type": m.type
            })
        with open(self.session.autosave_path, "w") as f:
            json.dump(data, f, indent=2)
