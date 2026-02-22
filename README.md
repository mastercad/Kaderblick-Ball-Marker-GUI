
BallmarkerGui: Plattformübergreifende Desktop-App zur manuellen und interpolierten Ballpositions-Annotation für Fußballvideos.

Features:
- Zwei synchronisierte Videos (linke/rechte Spielfeldhälfte)
- Marker-Logik (manuell/interpoliert, Kreisform, Größe, Drag, Delete)
- Timeline mit Keyframe-Navigation und Marker-Darstellung
- Undo/Redo für alle Marker-Operationen
- Autosave (automatisches Speichern im Hintergrund)
- Export (JSON, alle Markerfelder)
- 100% Testabdeckung (Unit- und Integrationstests)
- Optimale Usability für Nicht-Programmierer

Installation:
1. Python 3.10+ installieren
2. Abhängigkeiten installieren:
	pip install -r requirements.txt

Ausführung:
	python main.py

Tests:
	pytest

Dateistruktur:
- main.py: Einstiegspunkt
- ui/: UI-Komponenten
- video/: Video-Handling
- model/: Datenmodell
- interpolation/: Interpolationslogik
- export/: Export
- autosave/: Autosave
- tests/: Tests

Weitere Hinweise:
- Markergröße bleibt unabhängig vom Zoom
- Cursor-Feedback je nach Aktion
- Timeline und Marker sind farblich und visuell unterscheidbar
