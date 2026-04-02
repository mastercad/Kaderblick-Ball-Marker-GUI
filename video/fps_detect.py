"""Zentrale FPS-Erkennung für Videodateien.

Nutzt QMediaMetaData (primär) und OpenCV (Fallback).
Wird von VideoPlayer und VideoGraphicsPanel gemeinsam verwendet.
"""

from PySide6.QtMultimedia import QMediaPlayer, QMediaMetaData

FALLBACK_FPS = 30.0


def detect_fps_cv2(filepath: str) -> float | None:
    """FPS per OpenCV aus der Videodatei lesen. Gibt None zurück bei Fehler."""
    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    except Exception:
        pass
    return None


def detect_fps_metadata(player: QMediaPlayer) -> float | None:
    """FPS aus den QMediaPlayer-Metadaten lesen. Gibt None zurück wenn nicht verfügbar."""
    meta = player.metaData()
    fps = meta.value(QMediaMetaData.Key.VideoFrameRate)
    if fps and float(fps) > 0:
        return float(fps)
    return None


class FpsDetector:
    """Mixin/Helper für Klassen die FPS-Erkennung benötigen.

    Erwartet:
      - self.player: QMediaPlayer
      - self._fps: float | None (wird vom Aufrufer initialisiert)
      - Zugriff auf den Dateipfad des Videos
    """

    def __init__(self):
        self._fps: float | None = None

    def on_metadata_changed(self):
        """Als Slot an player.metaDataChanged anbinden."""
        detected = detect_fps_metadata(self.player)
        if detected:
            self._fps = detected

    def detect_from_file(self, filepath: str):
        """FPS sofort per cv2 erkennen (synchron)."""
        self._fps = detect_fps_cv2(filepath)

    def get_video_filepath(self) -> str:
        """Dateipfad des aktuellen Videos ermitteln. Kann überschrieben werden."""
        source = self.player.source()
        if source and source.toLocalFile():
            return source.toLocalFile()
        return ""

    @property
    def fps(self) -> float:
        """Erkannte FPS, mit Fallback-Kette: Metadaten → cv2 → 30.0."""
        if self._fps and self._fps > 0:
            return self._fps
        filepath = self.get_video_filepath()
        if filepath:
            detected = detect_fps_cv2(filepath)
            if detected:
                self._fps = detected
                return self._fps
        return FALLBACK_FPS

    @property
    def ms_per_frame(self) -> float:
        """Millisekunden pro Frame."""
        return 1000.0 / self.fps
