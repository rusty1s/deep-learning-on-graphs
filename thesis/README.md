# Master-Thesis

* **Universität:** TU Dortmund
* **Fakultät:** Lehrstuhl für Graphische Systeme

# Convolutional Neural Networks auf Graphrepräsentationen von Bildern

## Inhalt

* Motivation
* Relevante Arbeiten
* verschiedene **Graphrepräsentationen von Bildern**
  * Segmentierungsgraphen
  * Superpixel
    * verschiedene Superpixelberechnungsverfahren
  * Vektorgraphiken als Graph- bzw. Baumrepräsentation
* **Convolutional Neural Networks auf Graphen**
  * Verfahren
    * Multi-Scale Feature Maps
* **Convolutional Neural Networks auf Graphrepräsentationen von Bildern**
  * *Problem:* Klassifikation (bessere Evaluationsmethoden)
  * *Problem:* Segmentierung
* **Evaluation**
  * Performance unterschiedlicher Graphrepräsentationsverfahren
  * Performance CNN auf Graphen gegenüber CNN auf Bildern
  * Laufzeit CNN auf Graphen gegenüber CNN auf Bildern
  * gegenüber Graph-Kernels

## Motivation

* Potential sehr gute Resultate zu liefern
* CNNs auf Graphen sind im Allgemein kleiner als CNNs auf Bildern
* weniger Trainingsdaten benötigt?
* Rauschentfernung (Superpixelrepräsentation zwischen komprimiertem JPEG und
  PNG)
* keine strikte Größe der Bilder notwendig
* ggf. mehr Augmentierungsmöglichkeiten zusätzlich zur Bildaugmentierung

### 3 Szenarien

* CNN auf den normalen Bildern
* CNN auf downgescalten Bildern zur Informationsreduktion
* CNN auf Segmentierungen bzw. Graphen zur Informationsreduktion
  * Vergleich mit VGG 16/19 Referenznetz

## Schwächen

### Augmentierung

* Augmentierung der Daten ist im Bereich Deep Learning sehr essenziell
* Augmentierung auf den Bildern oder auf der Superpixelrepräsentation oder auf
  den Graphen?
* Augmentierung ist eine Möglichkeit, dem Netz zu zeigen, welche Sachen
  irrelevant für die Klassifikation sind (z.B. unabhängig von Brightness oder
  Normalisieren auf zero mean/one stddev)
* auf den **Graphen:**
  * Distorted Inputs sind nicht ohne weiteres möglich, da z.B. Farbänderungen 
    die Graphstruktur verändern können
* auf der **Superpixelrepraesentation:**
  * Cropping der Superpixelrepraesentation ist nicht moeglich, da ein Crop auf 
    dem Bild mit anschliessender Superpixelberechnung sowie der gleiche Crop 
    auf dem Bild und der vorher berechneten Superpixelrepraesentation zu 
    unterschiedlichen Ergebnissen fuehrt.
    * **TODO: Beweis**
    * evtl. ist dies garnicht so schlecht

## Offene Fragen

* Wie verarbeiten andere CNNs Superpixel?
* Welche Informationen werden bei den Segmentierungsgraphen zur Klassifizierung
  verwendet?
* Wie können SVGs über Graphen repräsentiert werden?
