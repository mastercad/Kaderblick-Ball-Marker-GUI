#!/usr/bin/env python3
"""Konvertiert MJPEG-Aufnahmen (.mjpg) + WAV-Audio in eine saubere Videodatei.

Erledigt alles in einem Schritt:
  - Liest die MJPEG-Datei mit korrekter Framerate (25 FPS)
  - Fügt die dazugehörige WAV-Audiodatei hinzu
  - Gleicht Längenunterschiede zwischen Video und Audio aus
  - Erzeugt ein MP4 mit H.264 (oder wahlweise AVI mit MJPEG)

Aufruf:
    python convert_mjpeg.py /pfad/zum/aufnahme.mjpg
    python convert_mjpeg.py /pfad/zum/verzeichnis/     # alle .mjpg im Ordner

Optionen:
    --fps 25          Framerate (Standard: 25)
    --format mp4      Ausgabeformat: mp4 oder avi (Standard: mp4)
    --crf 18          Qualität bei mp4 (0=verlustfrei, 18=sehr gut, 23=Standard)
    --no-audio        Audio nicht hinzufügen
    --dry-run         Nur anzeigen, was passieren würde
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_audio(mjpg_path: Path) -> Path | None:
    """Sucht die passende WAV-Datei zur MJPEG-Aufnahme."""
    stem = mjpg_path.stem
    parent = mjpg_path.parent
    # Exakter Name zuerst
    wav = parent / f"{stem}.wav"
    if wav.exists():
        return wav
    # Varianten (normalisiert etc.)
    for candidate in sorted(parent.glob(f"{stem}*.wav")):
        return candidate
    return None


def get_duration(filepath: Path) -> float | None:
    """Dauer in Sekunden per ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(filepath)],
            capture_output=True, text=True, timeout=30,
        )
        val = result.stdout.strip()
        return float(val) if val and val != "N/A" else None
    except Exception:
        return None


def convert_one(mjpg_path: Path, fps: int, fmt: str, crf: int,
                include_audio: bool, dry_run: bool) -> bool:
    """Konvertiert eine einzelne MJPEG-Datei."""
    if not mjpg_path.exists():
        print(f"  FEHLER: {mjpg_path} existiert nicht!")
        return False

    ext = "mp4" if fmt == "mp4" else "avi"
    out_path = mjpg_path.with_suffix(f".{ext}")

    if out_path.exists():
        print(f"  ÜBERSPRUNGEN: {out_path.name} existiert bereits")
        return True

    wav_path = find_audio(mjpg_path) if include_audio else None

    print(f"  Eingabe:  {mjpg_path.name}")
    print(f"  Ausgabe:  {out_path.name}")
    print(f"  FPS:      {fps}")
    print(f"  Format:   {fmt} (CRF={crf})" if fmt == "mp4" else f"  Format:   {fmt}")
    if wav_path:
        print(f"  Audio:    {wav_path.name}")
    else:
        print(f"  Audio:    {'(deaktiviert)' if not include_audio else '(keine WAV gefunden)'}")

    if dry_run:
        print("  → DRY RUN, nichts passiert\n")
        return True

    # ── ffmpeg-Kommando bauen ─────────────────────────────────────
    cmd = ["ffmpeg", "-hide_banner", "-y"]

    # Video-Input: MJPEG mit korrekter Framerate
    cmd += ["-framerate", str(fps), "-f", "mjpeg", "-i", str(mjpg_path)]

    # Audio-Input (falls vorhanden)
    if wav_path:
        cmd += ["-i", str(wav_path)]

    if fmt == "mp4":
        # H.264 Encoding — gute Qualität, kleine Dateigröße
        cmd += [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-r", str(fps),         # Output-Framerate explizit setzen
        ]
    else:
        # AVI mit MJPEG — quasi verlustfrei (copy reicht nicht, da FPS kaputt)
        cmd += [
            "-c:v", "mjpeg",
            "-q:v", "2",            # Qualität 2 = sehr gut
            "-r", str(fps),
        ]

    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        # Kürze auf die kürzere Spur (Video/Audio Längenunterschied)
        cmd += ["-shortest"]

    cmd += [str(out_path)]

    print(f"  Starte: {' '.join(cmd[:6])} ...")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        if proc.returncode != 0:
            print(f"  FEHLER (exit {proc.returncode}):")
            # Letzte Zeilen der Fehlermeldung
            for line in proc.stderr.strip().split("\n")[-10:]:
                print(f"    {line}")
            return False
    except KeyboardInterrupt:
        print("\n  Abgebrochen!")
        # Unvollständige Datei löschen
        if out_path.exists():
            out_path.unlink()
        return False

    # Ergebnis prüfen
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        duration = get_duration(out_path)
        dur_str = f", {duration:.0f}s" if duration else ""
        print(f"  ✓ Fertig: {out_path.name} ({size_mb:.0f} MB{dur_str})\n")
        return True
    else:
        print(f"  FEHLER: Ausgabedatei wurde nicht erstellt!\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="MJPEG + WAV → Video konvertieren",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="+",
                        help="MJPEG-Dateien oder Verzeichnisse")
    parser.add_argument("--fps", type=int, default=25,
                        help="Framerate (Standard: 25)")
    parser.add_argument("--format", choices=["mp4", "avi"], default="mp4",
                        help="Ausgabeformat (Standard: mp4)")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264-Qualität, nur bei mp4 (0=verlustfrei, 18=sehr gut)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Kein Audio hinzufügen")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur anzeigen, nicht konvertieren")
    args = parser.parse_args()

    # Alle MJPEG-Dateien sammeln
    files: list[Path] = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.glob("*.mjpg")) + sorted(p.glob("*.mjpeg"))
            if not found:
                print(f"Keine .mjpg-Dateien in {p}")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Nicht gefunden: {p}")

    if not files:
        print("Keine Dateien zum Konvertieren gefunden.")
        sys.exit(1)

    print(f"{'DRY RUN — ' if args.dry_run else ''}"
          f"{len(files)} Datei(en) zu konvertieren\n")

    ok, fail = 0, 0
    for f in files:
        print(f"[{ok + fail + 1}/{len(files)}] {f.name}")
        if convert_one(f, args.fps, args.format, args.crf,
                       not args.no_audio, args.dry_run):
            ok += 1
        else:
            fail += 1

    print(f"Fertig: {ok} erfolgreich, {fail} fehlgeschlagen")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
