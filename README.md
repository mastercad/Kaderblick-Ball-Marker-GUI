# ⚽ BallMarker GUI

Markiere die Ballposition in Fußball-Videoaufnahmen — Frame für Frame, unterstützt durch automatische Erkennung und Interpolation.

---

## Was kann die App?

- **Zwei Videos nebeneinander** laden (z. B. linke und rechte Spielfeldhälfte) — sie laufen synchron
- **Ball markieren** per Klick ins Bild — die Position wird pro Frame gespeichert
- **Automatische Ballerkennung (YOLO)** — erkennt den Ball automatisch, entweder im aktuellen Frame oder auf allen Frames gleichzeitig
- **Interpolation** — füllt die Lücken zwischen zwei gesetzten Markern automatisch auf
- **Feldkalibrierung** — interaktives Erstellen und Verwalten von Spielfeldmarkierungen pro Kameraseite (Feldrand, Mittellinie, Mittelkreis, Strafraum), inkl. Import/Export
- **Export & Import** — alle Markierungen als JSON speichern und später wieder laden
- **Autosave** — die Arbeit wird laufend im Hintergrund gesichert, damit nichts verloren geht

---

## Bedienung

### Video laden

Öffne die App (`python main.py`). Lade über Drag & Drop oder per Rechtsklick auf eines der beiden Videofenster dein Video.

### Marker setzen

| Aktion | Bedienung |
|---|---|
| **Ball markieren** | Linksklick ins Videobild |
| **Ausschluss-Marker setzen** | Shift + Linksklick ins Videobild |
| **Marker verschieben** | Marker anklicken und ziehen |
| **Markergröße ändern** | Mausrad über dem Marker |
| **Marker löschen** | Rechtsklick auf den Marker |

### Wiedergabe & Frame-Navigation

| Aktion | Tastatur | Button |
|---|---|---|
| Play / Pause | `Leertaste` | ▶ / ⏸ |
| 1 Frame vor / zurück | `→` / `←` | +1 ▶ / ◀ -1 |
| 25 Frames vor / zurück | `Shift + →` / `Shift + ←` | +25 ▶▶ / ◀◀ -25 |

### Zum nächsten Marker springen

Wähle im Dropdown neben den Abspieltasten, wohin du springen willst:

| Sprungziel | Beschreibung |
|---|---|
| ◆ Beliebiger Marker | Nächster Frame mit irgendeinem Marker |
| 🔴 Manuell | Nächster manuell gesetzter Marker |
| 🔵 YOLO | Nächster automatisch erkannter Marker |
| 🟠 Interpoliert | Nächster interpolierter Marker |
| ⚫ Ausschluss | Nächster Ausschluss-Marker |
| ⚪ Lücke | Nächster Frame **ohne** Marker |
| ◐ Keyframe | Nächster Keyframe im Video |

Springen: **Ctrl + →** (vorwärts) / **Ctrl + ←** (rückwärts), oder mit den ◀ / ▶ Buttons neben dem Dropdown.

### Zoom & Schwenken

| Aktion | Bedienung |
|---|---|
| Rein- / Rauszoomen | Mausrad (außerhalb eines Markers) |
| Bild verschieben | Mittlere Maustaste gedrückt halten und ziehen |
| Zoom zurücksetzen | `Ctrl + 0` |

### Werkzeuge

| Funktion | Tastenkürzel | Menü |
|---|---|---|
| Ball erkennen (aktueller Frame) | `Ctrl + D` | Werkzeuge → Ball erkennen (YOLO) |
| Alle Frames erkennen | `Ctrl + Shift + D` | Werkzeuge → Alle Frames erkennen |
| Marker interpolieren | `Ctrl + I` | Werkzeuge → Marker interpolieren |
| Marker zurücksetzen | — | Werkzeuge → Marker zurücksetzen… |
| Feldkalibrierung erstellen/bearbeiten | — | Werkzeuge → Feldkalibrierung erstellen/bearbeiten… |
| Feldkalibrierung laden | — | Werkzeuge → Feldkalibrierung laden… |
| Feldkalibrierung exportieren | — | Werkzeuge → Feldkalibrierung exportieren… |
| Feldgrenze entfernen | — | Werkzeuge → Feldgrenze entfernen |
| Rückgängig / Wiederholen | `Ctrl + Z` / `Ctrl + Y` | Datei → Undo / Redo |
| Exportieren | `Ctrl + S` | Datei → Exportieren… |
| Importieren | `Ctrl + O` | Datei → Importieren… |

### Tipps für die YOLO-Erkennung

- **Erst manuell, dann automatisch**: Setze auf einigen Schlüssel-Frames den Ball von Hand. Diese dienen als Referenzpunkte — YOLO springt dann nicht auf Wassertropfen oder Spiegelungen an.
- **Batch-Erkennung abbrechen**: Während „Alle Frames erkennen" läuft, kannst du jederzeit mit dem ❌-Button oder `Escape` abbrechen.
- **Überspringen wählen**: Beim Start der Batch-Erkennung kannst du festlegen, welche bereits markierten Frames übersprungen werden sollen (manuell, YOLO, interpoliert).
- **Nachkontrolle**: Nutze das Dropdown auf „🔵 YOLO" und springe mit Ctrl + → durch die erkannten Frames. Falsche Marker per Rechtsklick löschen.

---

## Markertypen

| Farbe | Typ | Bedeutung |
|---|---|---|
| 🔴 Rot | Manuell | Von dir per Hand gesetzt |
| 🔵 Blau | YOLO | Automatisch erkannt |
| 🟠 Orange | Interpoliert | Zwischen zwei Markern berechnet |
| ⚫ Grau (gestrichelt rot) | Ausschluss | Markiert eine Zone, in der YOLO keinen Ball erkennen soll |

Die Statistik unter dem Videobild zeigt dir jederzeit, wie viele Marker welchen Typs vorhanden sind.

---

## Ausschluss-Marker (Exclusion)

Manchmal erkennt YOLO fälschlicherweise Objekte als Ball, die keiner sind — z. B. Wassertropfen auf der Linse, Spiegelungen oder Eckfahnen.
Mit **Ausschluss-Markern** kannst du diese Bereiche sperren.

### So funktioniert es

1. **Setzen**: Halte `Shift` gedrückt und klicke ins Videobild. Es erscheint ein grauer Marker mit gestricheltem rotem Rand.
2. **Größe anpassen**: Scrolle über dem Marker, um seinen Radius zu vergrößern oder zu verkleinern. Der Radius bestimmt die Sperrzone.
3. **Wirkung**: Bei der YOLO-Erkennung (Einzelframe und Batch) wird jede Detektion, deren Mittelpunkt innerhalb eines Ausschluss-Markers auf dem gleichen oder einem nahen Frame (± 5 Frames) liegt, unterdrückt.
4. **Interpolation**: Ausschluss-Marker können genauso wie Ball-Marker interpoliert werden. So lässt sich eine wandernde Störquelle (z. B. ein sich bewegender Wasserfleck) über mehrere Frames abdecken.

### Tipps

- Setze Ausschluss-Marker **vor** der Batch-Erkennung, damit YOLO die Störquelle von Anfang an ignoriert.
- Im Dropdown kannst du „⚫ Ausschluss" wählen und mit Ctrl + → durch alle Ausschluss-Marker springen.
- Ausschluss-Marker werden beim Export/Import genauso gespeichert wie andere Marker.

---

## Interpolation

### Was ist Interpolation?

Interpolation füllt die **Lücken zwischen zwei vorhandenen Markern** automatisch auf.
Statt jeden einzelnen Frame von Hand zu markieren, setzt du Marker nur auf einigen Schlüssel-Frames — die Frames dazwischen werden berechnet.

Die App verwendet dafür **lineare Interpolation**: Position (x, y) und Radius werden gleichmäßig (proportional) zwischen den beiden Stützpunkten verteilt.

### Wie wende ich Interpolation an?

1. **Stützpunkte setzen**: Markiere den Ball auf mindestens zwei Frames. Je mehr Stützpunkte du setzt (z. B. alle 20–50 Frames), desto genauer wird das Ergebnis.
2. **Interpolation starten**: Drücke `Ctrl + I` oder wähle im Menü _Werkzeuge → Marker interpolieren_.
3. Fertig — alle Lücken zwischen aufeinander folgenden Markern werden jetzt mit orangenen Interpolations-Markern aufgefüllt.

### Welches Ergebnis ist zu erwarten?

- Zwischen jedem Marker-Paar entstehen neue **orangene Marker** (Typ „interpolated") — einer pro Frame.
- Position und Größe ändern sich **gleichmäßig** von einem Stützpunkt zum nächsten (geradlinige Bewegung).
- Frames, auf denen bereits ein Marker existiert (egal welchen Typs), werden **nicht überschrieben**.
- Bei **geraden Ballbewegungen** (Pässe, Rollen) liefert die Interpolation sehr gute Ergebnisse.
- Bei **Richtungswechseln** (Abpraller, hohe Bälle) solltest du zusätzliche Stützpunkte an den Wendepunkten setzen, damit die Kurve genauer wird.

### Ball- und Ausschluss-Interpolation

Ball-Marker (`manual`, `yolo`, `interpolated`) und Ausschluss-Marker werden **getrennt** interpoliert, damit sie sich nicht gegenseitig beeinflussen:

| Kette | Stützpunkte | Ergebnis-Typ |
|---|---|---|
| Ball | Manuell / YOLO / Interpoliert | `interpolated` (🟠 orange) |
| Ausschluss | Ausschluss | `exclusion` (⚫ grau) |

### Empfohlener Workflow

1. Video laden und abspielen.
2. Auf markanten Stellen den Ball per Klick markieren (alle 20–50 Frames).
3. Optional: YOLO-Erkennung starten, um automatisch weitere Punkte zu setzen.
4. `Ctrl + I` drücken — die Lücken werden aufgefüllt.
5. Im Dropdown „⚪ Lücke" wählen und mit Ctrl + → durch verbleibende Lücken springen.
6. Fehlende Stellen nachmarkieren und ggf. erneut interpolieren.

---

## Feldkalibrierung

Die Feldkalibrierung definiert die sichtbaren Spielfeldgrenzen und -markierungen für jede Kameraseite. Sie wird von der YOLO-Erkennung genutzt, um Detektionen außerhalb des Spielfelds automatisch zu verwerfen — das reduziert Fehlerkennungen deutlich.

### Dual-Kamera-Setup

Jede Kamera sieht nur **eine Hälfte** des Spielfelds:

| Kamera | Sichtbereich | Sichtbare Elemente |
|--------|--------------|---------------------|
| Kamera 0 (Links) | Linke Spielfeldhälfte | Linkes Tor, linker Strafraum, linke Hälfte Mittelkreis, Mittellinie am rechten Bildrand |
| Kamera 1 (Rechts) | Rechte Spielfeldhälfte | Rechtes Tor, rechter Strafraum, rechte Hälfte Mittelkreis, Mittellinie am linken Bildrand |

Die Kalibrierungsdaten beider Kameras werden gemeinsam in einer JSON-Datei gespeichert (`data/field_calibration.json`) unter den Schlüsseln `cam0` und `cam1`.

### Kalibrierung erstellen

1. Video laden (mindestens ein Panel)
2. **Werkzeuge → Feldkalibrierung erstellen/bearbeiten…** wählen
3. Im Dialog öffnet sich der aktuelle Videoframe als Kalibrierungsbild
4. Je nach Kameraseite werden die passenden Modi angeboten:

| Schritt | Modus | Was markieren? |
|---------|-------|----------------|
| 1 | **Spielfeldrand** | Entlang der sichtbaren Außenlinie klicken (mind. 4 Punkte, mehr für Fischauge) |
| 2 | **Mittellinie** | Entlang der Mittellinie klicken (mind. 2 Punkte) |
| 3 | **Mittelkreis** | Punkte entlang des sichtbaren Halbkreises setzen |
| 4 | **Strafraum** | 4 Ecken des Strafraums im Uhrzeigersinn |

### Bedienung im Kalibrierungsdialog

| Aktion | Bedienung |
|--------|----------|
| Punkt setzen | Linksklick |
| Punkt auf Linie einfügen | Doppelklick nahe einer Verbindungslinie |
| Punkt entfernen | Rechtsklick auf den Punkt |
| Punkt verschieben | Punkt anklicken und ziehen |
| Letzten Punkt entfernen | `Z` oder Button |
| Nächster Modus | `Enter` / `N` oder Button |
| Vorheriger Modus | `B` oder Button |
| Zoom | Mausrad |
| Bild verschieben | Mittlere Maustaste |
| Bild einpassen | `0` oder ⊞ Button |
| Kameraseite wechseln | Dropdown oben links |

### Tipps für die Kalibrierung

- **Fischauge-Korrektur**: Bei Weitwinkelkameras (z. B. 155°) sind die Spielfeldlinien gekrümmt. Setze **viele Punkte** entlang der Kurven, damit die Grenze genauer abgebildet wird.
- **Punkt einfügen**: Per Doppelklick nahe einer Verbindungslinie wird ein neuer Punkt zwischen den beiden nächsten Eckpunkten eingefügt – ideal zum Nachbessern von Kurven.
- **Punkt entfernen**: Rechtsklick auf einen bestehenden Punkt entfernt diesen sofort.
- **Punkte verschieben**: Alle gesetzten Punkte können nachträglich per Drag & Drop verschoben werden.
- **Modus leeren**: Falls du einen Modus komplett neu machen willst, nutze den „Modus leeren" Button.
- Die Kalibrierung wird automatisch gespeichert und beim nächsten Start wiederhergestellt.

### Import / Export

- **Kalibrierung importieren**: Im Dialog über „📂 Kalibrierung importieren…" eine bestehende `field_calibration.json` laden.
- **Kalibrierung exportieren**: Im Dialog über „💾 Kalibrierung exportieren…" oder im Hauptmenü über „Werkzeuge → Feldkalibrierung exportieren…".
- Das Dateiformat ist kompatibel mit dem externen `manual_field_calibration.py`-Tool — bestehende Kalibrierungen können direkt übernommen werden.

### Datenformat

Die JSON-Datei enthält pro Kamera einen Eintrag:

```json
{
  "cam0": {
    "field_boundary": [[x1, y1], [x2, y2], ...],
    "center_line": [[x1, y1], [x2, y2], ...],
    "center_half_ellipse_points": [[x1, y1], ...],
    "penalty_area_left": [[x1, y1], ...],
    "frame_width": 3840,
    "frame_height": 2160,
    "video_path": "/pfad/zum/video.mp4",
    "camera_id": 0
  },
  "cam1": {
    "field_boundary": [...],
    "center_line": [...],
    "penalty_area_right": [...],
    ...
  }
}
```

---

## Installation

1. **Python 3.10** oder neuer installieren
2. Abhängigkeiten installieren:
   ```
   pip install -r requirements.txt
   ```
3. App starten:
   ```
   python main.py
   ```
