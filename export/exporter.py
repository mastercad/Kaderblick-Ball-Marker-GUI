
import json
import os
from collections import defaultdict
from model.marker import Marker


def _build_export_data(markers, sync_offset_frames=0):
    """Baut die hierarchische Export-Struktur: gruppiert nach Video → Frame."""
    by_video = defaultdict(lambda: defaultdict(list))
    for m in markers:
        by_video[m.video_file][m.frame_index].append(m)

    videos = []
    for video_file, frames in sorted(by_video.items()):
        frame_list = []
        for frame_idx in sorted(frames.keys()):
            frame_markers = frames[frame_idx]
            ts = frame_markers[0].timestamp_ms if frame_markers else 0
            marker_dicts = []
            for m in frame_markers:
                marker_dicts.append({
                    "position": {"x": m.position[0], "y": m.position[1]},
                    "radius": m.radius,
                    "type": m.type,
                })
            frame_list.append({
                "frame_index": frame_idx,
                "timestamp_ms": ts,
                "markers": marker_dicts,
            })
        videos.append({
            "video_file": video_file,
            "frames": frame_list,
        })
    result = {"version": 1, "videos": videos}
    if sync_offset_frames:
        result["sync_offset_frames"] = sync_offset_frames
    return result


def export_markers(markers, filename, sync_offset_frames=0):
    """Exportiert alle Marker als strukturierte JSON-Datei. Gibt den finalen Dateipfad zurück."""
    data = _build_export_data(markers, sync_offset_frames=sync_offset_frames)
    filename = str(filename)
    if not os.path.splitext(filename)[1]:
        filename += ".json"
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filename)
    return filename


def import_markers(filename):
    """Importiert Marker aus einer JSON-Datei. Gibt eine Liste von Marker-Objekten zurück."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    markers = []
    # Neues hierarchisches Format (version >= 1)
    if isinstance(data, dict) and "videos" in data:
        for video_entry in data["videos"]:
            video_file = video_entry["video_file"]
            for frame_entry in video_entry["frames"]:
                frame_idx = int(frame_entry["frame_index"])
                ts = int(frame_entry["timestamp_ms"])
                for md in frame_entry["markers"]:
                    markers.append(Marker(
                        video_file=video_file,
                        frame_index=frame_idx,
                        timestamp_ms=ts,
                        position=(float(md["position"]["x"]), float(md["position"]["y"])),
                        radius=float(md["radius"]),
                        marker_type=md.get("type", "manual"),
                    ))
    # Altes flaches Format (Liste von Marker-Dicts) – Rückwärtskompatibilität
    elif isinstance(data, list):
        for d in data:
            markers.append(Marker.from_dict(d))
    return markers
