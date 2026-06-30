# Workflow für saubere Ballerkennung

Diese Anleitung beschreibt den empfohlenen Ablauf, um mit der Anwendung eine möglichst zuverlässige Ballerkennung aufzubauen. Der wichtigste Gedanke dabei ist: Die Erkennung wird nicht durch einen einzelnen Knopfdruck gut, sondern durch saubere Beispiele, klare Feldgrenzen und wiederholtes Prüfen.

Der empfohlene Weg ist der **Heatmap-Detector**. Er ist für kleine Bälle in 4K-Übersichtsaufnahmen gedacht und arbeitet besser zu dieser Kamerasituation als eine reine Bounding-Box-Erkennung.

## Ziel des Workflows

Am Ende soll die Anwendung in neuen Spielszenen möglichst zuverlässig erkennen, wo sich der Ball befindet. Dafür braucht sie Beispiele aus genau der Perspektive, in der später erkannt werden soll:

- Ball nah an der Kamera
- Ball weit weg auf dem Spielfeld
- Ball am Boden
- Ball in der Luft
- Ball vor Spielern
- Ball in Schattenbereichen
- Ball auf hellen und dunklen Rasenstellen
- Situationen mit ähnlichen Objekten wie Stutzen, Hütchen, Linien, Köpfen, Schuhen oder hellen Flecken

Je besser diese Situationen in den Trainingsdaten vorkommen, desto stabiler wird die Erkennung.

## Grundregel für gute Ergebnisse

Nicht nur einfache Beispiele markieren. Gerade die schwierigen Fälle sind wichtig.

Ein Modell, das nur große, klare Bälle nahe an der Kamera gesehen hat, wird kleine Bälle in der Spielfeldmitte kaum zuverlässig erkennen. Ein Modell, das viele echte Spielsituationen gesehen hat, wird deutlich robuster.

## Empfohlener Ablauf

1. Video laden.
2. Feldgrenze kalibrieren.
3. Bälle in vielen unterschiedlichen Situationen markieren.
4. Heatmap-Trainingsdaten exportieren.
5. Heatmap-Modell intern trainieren.
6. Heatmap-Erkennung testen.
7. Fehler korrigieren und gezielt weitere Beispiele hinzufügen.
8. Erneut exportieren und trainieren.

Dieser Kreislauf ist normal. Das erste Modell ist selten das beste Modell.

## 1. Geeignetes Videomaterial auswählen

Für den Aufbau eines guten Modells sollten die Videos zur späteren Nutzung passen.

Gute Trainingsvideos haben:

- dieselbe Kamera oder denselben Kameratyp
- dieselbe Montageposition
- ähnliche Brennweite bzw. denselben Bildausschnitt
- echte Spiel- oder Trainingssituationen
- verschiedene Lichtverhältnisse
- genügend Szenen, in denen der Ball sichtbar ist

Weniger geeignet sind:

- Videos aus völlig anderer Perspektive
- stark verwackelte Aufnahmen
- extrem unscharfe oder überbelichtete Aufnahmen
- Szenen, in denen der Ball kaum oder gar nicht sichtbar ist
- nur sehr kurze Tests mit wenigen Einzelbildern

Wichtig: Wenn die Kamera dauerhaft auf 4K-Halbfeldansicht arbeitet, sollten auch die Trainingsdaten aus dieser 4K-Halbfeldansicht kommen.

## 2. Video in der Anwendung laden

1. Anwendung öffnen.
2. Linkes und/oder rechtes Video laden.
3. Prüfen, ob das Bild korrekt angezeigt wird.
4. Mit der Zeitleiste an mehrere Stellen springen und schauen, ob Bild und Spielverlauf plausibel sind.

Wenn zwei Kameras verwendet werden, sollten beide Videos geladen und sauber synchronisiert werden, bevor viele Marker gesetzt werden.

## 3. Feldgrenze kalibrieren

Die Feldgrenze ist wichtig, weil sie der Erkennung hilft, unwahrscheinliche Bereiche auszublenden. Alles außerhalb des Spielfelds soll möglichst nicht als Ball gewertet werden.

So wird die Feldgrenze gesetzt:

1. Im Menü **Werkzeuge** die Feldkalibrierung öffnen.
2. Die sichtbare Spielfeldgrenze möglichst genau nachzeichnen.
3. Bei Weitwinkelbildern lieber mehr Punkte setzen als zu wenige.
4. Ecken, Linienverläufe und gebogene Bildbereiche sorgfältig erfassen.
5. Speichern und schließen.

Worauf achten:

- Die Linie soll an der echten Spielfeldgrenze liegen, nicht grob daneben.
- Bei stark verzerrten Kamerabildern dürfen mehr Punkte gesetzt werden.
- Spieler, Trainerbank, Zuschauerbereich und Wege außerhalb des Felds sollen nicht versehentlich Teil des Feldes sein.

Die Kalibrierung muss normalerweise nur geändert werden, wenn sich Kamera, Bildausschnitt oder Montageposition ändern.

## 4. Bälle sauber markieren

Die Markierungen sind die wichtigste Grundlage für das Modell.

So sollte markiert werden:

- Der Marker sitzt auf dem Mittelpunkt des Balls.
- Der Marker ist so klein wie sinnvoll, aber groß genug, um den Ball sichtbar abzudecken.
- Nur echte Bälle markieren.
- Hütchen, Schuhe, Stutzen, Köpfe, Linienflecken oder Schatten nicht als Ball markieren.
- Wenn der Ball nicht sicher erkennbar ist, lieber diesen Frame überspringen.

Bei kleinen Bällen zählt Genauigkeit. Wenn der Ball nur wenige Pixel groß ist, macht ein daneben gesetzter Marker einen großen Unterschied.

## 5. Welche Situationen markiert werden sollten

Für gute Ergebnisse nicht nur die einfachen Frames markieren.

Unbedingt markieren:

- Ball weit entfernt in der Spielfeldmitte
- Ball nahe an der Kamera
- Ball am Rand des Spielfelds
- Ball vor Spielern
- Ball neben weißen Stutzen oder Schuhen
- Ball in der Luft
- Ball bei Sonne
- Ball im Schatten
- ruhender Ball
- schneller Ball
- Ball teilweise verdeckt
- mehrere Bälle beim Aufwärmen

Gezielt schwierige Beispiele sind wertvoller als viele fast gleiche leichte Beispiele.

## 6. Wie viele Markierungen sinnvoll sind

Für einen ersten Test reichen wenige Markierungen, aber für eine brauchbare Erkennung ist mehr nötig.

Richtwerte:

| Markierungen | Erwartung |
| --- | --- |
| 20-50 | Nur Funktionsprüfung, noch keine zuverlässige Erkennung |
| 100-300 | Erste erkennbare Verbesserung, noch instabil |
| 500-1000 | Deutlich brauchbarer, wenn die Beispiele vielfältig sind |
| 1000+ | Solider Bereich für echte Spielsituationen |

Wichtiger als die reine Anzahl ist die Mischung. 300 gut verteilte, schwierige Beispiele können nützlicher sein als 1000 fast gleiche einfache Beispiele.

## 7. Harte Negativbeispiele bewusst mitnehmen

Das Modell muss nicht nur lernen, was ein Ball ist. Es muss auch lernen, was **kein** Ball ist.

Typische Verwechslungen:

- weiße Stutzen
- helle Schuhe
- Hütchen
- Kreidelinien
- helle Flecken im Rasen
- Köpfe
- Reflexionen
- Schattenkanten
- kleine helle Objekte außerhalb des Spielfelds

Der Heatmap-Export erzeugt automatisch Negativsamples aus den markierten Szenen. Trotzdem hilft es, Videos und Frames zu verwenden, in denen solche Verwechslungsobjekte vorkommen. Dadurch sieht das Training genau die Fehlerfälle, die später vermieden werden sollen.

## 8. Heatmap-Trainingsdaten exportieren

Wenn genügend Marker gesetzt wurden:

1. Menü **Werkzeuge** öffnen.
2. **Heatmap-Trainingsdaten exportieren...** wählen.
3. Zielordner auswählen, zum Beispiel `heatmap_dataset`.
4. Export starten.
5. Den Fortschritt im Dialog beobachten.

Der Export erzeugt keine einfachen Gesamtbilder für eine Box-Erkennung. Stattdessen werden kleine Bildsequenzen, Zielpunkte und Negativbeispiele erzeugt. Das passt besser zu kleinen Bällen, weil der Detector später nach dem wahrscheinlichen Ballzentrum sucht.

Nach dem Export zeigt die Anwendung eine Zusammenfassung:

- Anzahl der Ball-Samples
- Anzahl der Negativsamples
- Anzahl der Quell-Frames
- Aufteilung in Training und Prüfung
- Bildgröße der Trainingsausschnitte

Wenn hier nur sehr wenige Samples stehen, ist das Ergebnis später entsprechend schwach.

## 9. Heatmap-Modell intern trainieren

Nach dem Export fragt die Anwendung, ob direkt trainiert werden soll.

Empfohlen:

1. Frage mit **Ja** bestätigen.
2. Einstellungen zunächst auf den Standardwerten lassen.
3. **Training starten** drücken.
4. Fortschritt im Dialog verfolgen.
5. Warten, bis das Training abgeschlossen ist.

Die wichtigsten Einstellungen:

| Einstellung | Bedeutung | Empfehlung |
| --- | --- | --- |
| Epochen | Wie oft alle Beispiele durchlaufen werden | Erst einmal Standardwert verwenden |
| Batch-Größe | Wie viele Beispiele gleichzeitig verarbeitet werden | Bei Speicherproblemen kleiner stellen |
| Rechnen auf | Automatisch, Grafikkarte oder Prozessor | Automatisch verwenden |

Wenn das Training wegen Speicherproblemen abbricht, die Batch-Größe reduzieren und erneut starten.

Nach erfolgreichem Training wird das Heatmap-Modell gespeichert. Der Button **Ball erkennen (Heatmap)** verwendet dieses Modell automatisch.

## 10. Erkennung testen

Nach dem Training nicht sofort einem kompletten Spiel blind vertrauen. Erst gezielt prüfen.

Guter Testablauf:

1. Mehrere Stellen im Video auswählen.
2. Einfache Situationen testen.
3. Schwierige Situationen testen.
4. Weit entfernte Bälle testen.
5. Nahe Bälle testen.
6. Szenen mit Hütchen, Stutzen und Linien testen.
7. Treffer und Fehler notieren.

Für den Test den Button **Ball erkennen (Heatmap)** verwenden.

Wichtig ist nicht, ob ein einzelner Frame zufällig passt. Wichtig ist, ob die Erkennung über viele unterschiedliche Szenen stabil wirkt.

## 11. Fehler richtig verbessern

Wenn die Erkennung falsche oder fehlende Ergebnisse liefert, nicht wahllos weitertrainieren. Besser gezielt verbessern.

### Ball wird nicht erkannt

Mögliche Ursachen:

- Zu wenige ähnliche Beispiele im Training.
- Ball ist in diesem Bildbereich besonders klein.
- Ball liegt in Schatten oder vor unruhigem Hintergrund.
- Ball ist teilweise verdeckt.
- Marker waren zu ungenau.

Was tun:

- Genau solche Frames nachmarkieren.
- Ähnliche Szenen aus anderen Stellen hinzufügen.
- Darauf achten, dass der Marker wirklich auf dem Ballzentrum sitzt.
- Danach Heatmap-Daten erneut exportieren und Modell neu trainieren.

### Falsches Objekt wird als Ball erkannt

Mögliche Ursachen:

- Das Objekt sieht dem Ball im Bild ähnlich.
- Zu wenige Negativbeispiele mit diesem Objekt.
- Zu wenige echte Bälle in ähnlicher Umgebung.

Was tun:

- Szenen mit diesem Verwechslungsobjekt in den Trainingsbestand aufnehmen.
- Echte Bälle in ähnlichen Bildbereichen markieren.
- Falls dort kein Ball ist, keinen Marker setzen.
- Neu exportieren und trainieren.

### Erkennung ist nur nahe an der Kamera gut

Mögliche Ursache:

- Im Training fehlen kleine, weit entfernte Bälle.

Was tun:

- Mehr entfernte Bälle markieren.
- Besonders Spielfeldmitte und gegenüberliegende Seite berücksichtigen.
- Auch kleine Bälle in der Luft markieren, wenn sie sichtbar sind.

### Erkennung ist nur bei bestimmten Lichtverhältnissen gut

Mögliche Ursache:

- Training enthält zu wenig Abwechslung bei Sonne, Schatten und Bewölkung.

Was tun:

- Markierungen aus verschiedenen Tageszeiten und Lichtlagen ergänzen.
- Szenen mit Schattenkanten gezielt aufnehmen.

## 12. Sinnvoller Verbesserungszyklus

Ein guter Arbeitsrhythmus sieht so aus:

1. 200-500 gemischte Ballbeispiele markieren.
2. Heatmap-Daten exportieren.
3. Heatmap-Modell trainieren.
4. 20-50 Teststellen prüfen.
5. Fehlerfälle sammeln.
6. Genau diese Fehlerfälle nachmarkieren oder ergänzen.
7. Erneut exportieren und trainieren.

Nach jeder Runde sollte die Erkennung in den zuvor schwierigen Situationen besser werden. Wenn sie schlechter wird, prüfen:

- Sind Marker ungenau gesetzt?
- Wurden falsche Objekte als Ball markiert?
- Sind zu viele fast gleiche Beispiele und zu wenige schwierige Beispiele enthalten?
- Wurde mit zu wenigen Beispielen trainiert?

## 13. Wann YOLO verwenden?

YOLO kann weiterhin als Nebenweg nützlich sein, vor allem bei großen oder nahen Bällen. Für kleine Bälle in einer 4K-Halbfeldkamera ist der Heatmap-Detector aber der empfohlene Hauptweg.

Empfehlung:

- **Heatmap** für kleine und weit entfernte Bälle.
- **YOLO** höchstens als Zusatztest bei großen, nahen Bällen.
- Nicht versuchen, schlechte kleine-Ball-Erkennung nur durch schönere YOLO-Anzeigen zu retten.

## 14. Woran man ein brauchbares Modell erkennt

Ein brauchbares Modell erkennt nicht jeden Ball perfekt, aber es zeigt ein stabiles Verhalten.

Gute Zeichen:

- Der Ball wird in verschiedenen Bildbereichen gefunden.
- Nahe und entfernte Bälle werden erkannt.
- Hütchen und Stutzen werden seltener verwechselt.
- Schatten und Linien verursachen weniger falsche Treffer.
- Die Erkennung wirkt über mehrere Spielszenen hinweg nachvollziehbar.

Warnzeichen:

- Es wird fast nie etwas erkannt.
- Immer dieselben falschen Objekte werden erkannt.
- Nur nahe Bälle funktionieren.
- Entfernte Bälle werden grundsätzlich ignoriert.
- Nach neuem Training ist das Ergebnis schlechter als vorher.

Bei Warnzeichen nicht weiter auf demselben Stand testen, sondern gezielt Trainingsdaten verbessern.

## 15. Praktische Checkliste vor dem Training

Vor dem Export prüfen:

- Feldgrenze ist gesetzt und passt zum Bild.
- Marker sitzen auf dem Ballzentrum.
- Es gibt nahe und entfernte Ballbeispiele.
- Es gibt Beispiele aus verschiedenen Bildbereichen.
- Es gibt Beispiele mit Sonne und Schatten.
- Es gibt schwierige Szenen mit Spielern in Ballnähe.
- Verwechslungsobjekte kommen im Material vor.
- Offensichtlich falsche Marker wurden entfernt.

Vor dem Verwenden des Modells prüfen:

- Training ist vollständig abgeschlossen.
- Heatmap-Erkennung wurde an mehreren Stellen getestet.
- Fehler wurden nicht nur angeschaut, sondern als neue Trainingsbeispiele genutzt.
- Das Ergebnis ist in echten Spielsituationen stabil genug.

## 16. Häufige Missverständnisse

### "Ich habe drei Bilder trainiert, warum erkennt das Modell nichts?"

Drei Bilder reichen nur für einen technischen Funktionstest. Daraus entsteht keine zuverlässige Erkennung für ein ganzes Spielfeld.

### "Der Ball ist für das Auge sichtbar, warum erkennt ihn die Anwendung nicht?"

Das menschliche Auge nutzt Bewegung, Kontext und Erfahrung. Ein Modell braucht dafür passende Beispiele. Gerade bei sehr kleinen Bällen muss es viele ähnliche Situationen gesehen haben.

### "Warum nicht einfach das ganze 4K-Bild verkleinern?"

Wenn ein sehr kleiner Ball im Gesamtbild weiter verkleinert wird, bleiben kaum erkennbare Informationen übrig. Deshalb arbeitet der Heatmap-Workflow mit Ausschnitten, Sequenzen und Ballzentren statt mit einer einfachen Verkleinerung des kompletten Bildes.

### "Warum sind Negativbeispiele wichtig?"

Weil viele Dinge im Bild ähnlich klein und hell sein können. Ohne Negativbeispiele lernt das Modell schlechter, Stutzen, Hütchen, Linien und helle Flecken zu ignorieren.

## 17. Kurzfassung

Für gute Ergebnisse:

1. Feldgrenze sauber setzen.
2. Viele unterschiedliche Bälle präzise markieren.
3. Schwierige Fälle nicht auslassen.
4. Heatmap-Trainingsdaten exportieren.
5. Heatmap-Modell intern trainieren.
6. Heatmap-Erkennung testen.
7. Fehler gezielt nachmarkieren.
8. Wiederholen, bis die Erkennung stabil ist.

Der wichtigste Erfolgsfaktor ist die Qualität und Vielfalt der Markierungen.
