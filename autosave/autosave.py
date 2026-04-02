
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
AUTOSAVE_FIELD_CAL_PATH = os.path.join(_DATA_DIR, "field_calibration_state.json")
DEFAULT_EXPORT_DIR = _DATA_DIR


class Autosave:
    def __init__(self, session):
        self.session = session
        self.running = False
        self._last_marker_count = 0

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
        """Speichert die aktuelle Session atomar (tmp + rename)."""
        try:
            path = self.session.autosave_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            export_markers(self.session.markers, path)
            self._last_marker_count = len(self.session.markers)
        except Exception as e:
            print(f"[Autosave] Fehler beim Speichern: {e}")

    # ── Feldkalibrierung persistieren ─────────────────────────────

    def save_field_calibration_path(self, cal_path: str | None):
        """Speichert den Pfad zur Feldkalibrierungsdatei."""
        try:
            os.makedirs(os.path.dirname(AUTOSAVE_FIELD_CAL_PATH), exist_ok=True)
            if cal_path:
                data = {"field_calibration_path": cal_path}
                tmp = AUTOSAVE_FIELD_CAL_PATH + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp, AUTOSAVE_FIELD_CAL_PATH)
            else:
                if os.path.isfile(AUTOSAVE_FIELD_CAL_PATH):
                    os.remove(AUTOSAVE_FIELD_CAL_PATH)
        except Exception as e:
            print(f"[Autosave] Feldkalibrierung-State Fehler: {e}")

    def load_field_calibration_path(self) -> str | None:
        """Lädt den gespeicherten Pfad zur Feldkalibrierungsdatei."""
        try:
            if not os.path.isfile(AUTOSAVE_FIELD_CAL_PATH):
                return None
            with open(AUTOSAVE_FIELD_CAL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("field_calibration_path")
        except Exception as e:
            print(f"[Autosave] Feldkalibrierung-State lesen Fehler: {e}")
            return None

    def has_recovery(self) -> bool:
        """Prüft ob eine Autosave-Datei vorhanden ist."""
        return os.path.isfile(self.session.autosave_path)

    def recover(self):
        """Stellt Marker aus der Autosave-Datei wieder her. Gibt die geladenen Marker zurück."""
        try:
            markers = import_markers(self.session.autosave_path)
            return markers
        except Exception as e:
            print(f"[Autosave] Fehler beim Wiederherstellen: {e}")
            return []

    def clear(self):
        """Löscht die Autosave-Datei und den Feldkalibrierungs-State."""
        try:
            if os.path.isfile(self.session.autosave_path):
                os.remove(self.session.autosave_path)
        except Exception:
            pass
        try:
            if os.path.isfile(AUTOSAVE_FIELD_CAL_PATH):
                os.remove(AUTOSAVE_FIELD_CAL_PATH)
        except Exception:
            pass
