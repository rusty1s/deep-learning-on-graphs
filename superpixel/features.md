# Feature Extraction

* Anzahl an Pixeln (bzw. Flaecheninhalt)
* Schwerpunkt
* Perimeter: Summe der Laenge aller Polygonlinien
* Axis-Aligned Bounding Box (AABB) => Hoehe und Breite (und Flaeche)
* Oriented Bounding Box (OBB) => minimale Bounding Box unabhaengig der Achsen
  => Hoehe und Breite (und Flaeche)
* Aspect Ratio von OBB (groessere durch kleinere)
* Der kleinere Winkel zwischen der groesseren Linie des OBBs zur x-Achse
* Rectangularity: Flaecheninhalt des Polygons durch die Flaeche des OBBS
  * wenn 1, dann ist das Polygon voellig rechteckig
* Circularity: Aehnlichkeit des Polygons zu einem Kreis mit Hilfe der Flaeche
  und der Summe der Laenge aller Polygonlinien
  * Formel laesst sich einfach beweisen, indem man Kreisumfang und Kreisflaeche
    durch Formel mit Radius ersetzt
* Compactness: 1 falls Kreis, verringert sich falls das Polygon laenger wird =>
  Durchmesser durch Laenge der groesseren Linie von OBB
* Central Moment ist Intensity-Weighted Feature eines Polygons
  * Die Koordinaten (x, y) aller Pixel in dem Polygon werden gewichtet nach
    ihrem relativen Anteil zu der totalen Summe an Intensitaeten im Polygon

## Umwandlung fuer Pixelpolygone

* Perimeter-Umwandlung in Anzahl der Pixel am Rand?
* Circularity?
