
import json
import os
import threading
import time

from export.exporter import export_markers, import_markers

# Autosave alle 15 Sekunden
_AUTOSAVE_INTERVAL = 15

# ── Zentrale Pfade für alle Laufzeitdaten ─────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AUTOSAVE_SESSION_PATH = os.path.join(_DATA_DIR, "session.json")
DEFAULT_FIELD_CALIBRATION_PATH = os.path.join(_DATA_DIR, "field_calibration.json")
DEFAULT_EXPORT_DIR = _DATA_DIR


class Autosave:
    def __init__(self, session, get_video_paths=None, get_sync_offset=None):
        self.session = session
        self.running = False
        self._last_marker_count = 0
        self._get_video_paths = get_video_paths  # Callback: () -> [str, str]
        self._get_sync_offset = get_sync_offset  # Callback: () -> int

    def start(self):
        self.running = True
        threading.Thread(target=self._autosave_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _autosave_loop(self):
        while self.running:
            time.sleep(_AUTOSAVE_INTERVAL)
            self.save()

    def save(self):
        """Speichert die aktuelle Session atomar (Marker + Videopfade)."""
        try:
            path = str(self.session.autosave_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Marker-Daten aufbauen
            from export.exporter import _build_export_data
            data = _build_export_data(self.session.markers)

            # Videopfade separat speichern (unabhängig von Markern)
            if self._get_video_paths:
                data["loaded_videos"] = self._get_video_paths()

            # Sync-Offset speichern
            if self._get_sync_offset:
                offset = self._get_sync_offset()
                if offset:
                    data["sync_offset_frames"] = offset

            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            self._last_marker_count = len(self.session.markers)
        except Exception as e:
            print(f"[Autosave] Fehler beim Speichern: {e}")

    def has_recovery(self) -> bool:
        """Prüft ob eine Autosave-Datei vorhanden ist und Daten enthält."""
        if not os.path.isfile(self.session.autosave_path):
            return False
        try:
            with open(self.session.autosave_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Recovery anbieten wenn Videos oder Marker vorhanden
            has_videos = bool(data.get("loaded_videos", []))
            has_markers = bool(data.get("videos", []))
            return has_videos or has_markers
        except Exception:
            return False

    def load_session_data(self) -> dict:
        """Lädt die komplette Session-Datei als dict."""
        try:
            with open(self.session.autosave_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Autosave] Fehler beim Lesen: {e}")
            return {}

    def recover(self):
        """Stellt Marker aus der Autosave-Datei wieder her. Gibt die geladenen Marker zurück."""
        try:
            markers = import_markers(self.session.autosave_path)
            return markers
        except Exception as e:
            print(f"[Autosave] Fehler beim Wiederherstellen: {e}")
            return []

    def clear(self):
        """Löscht die Autosave-Datei."""
        try:
            if os.path.isfile(self.session.autosave_path):
                os.remove(self.session.autosave_path)
        except Exception:
            pass
