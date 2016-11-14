# Master-Thesis

* **Universität:** TU Dortmund
* **Fakultät:** Lehrstuhl für Graphische Systeme

# Anwendbarkeit von Convolutional Neural Networks auf Graph-Repräsentationen von Bildern

## Inhalt

* verschiedene **Graph-Repräsentationen von Bildern**
  * Segmentierungsgraphen
  * Superpixel
    * verschiedene Superpixelberechnungsverfahren
  * Vektorgraphiken als Graph- bzw. Baum-Repräsentation
* **Convolutional Neural Networks auf Graphen**
  * Verfahren
* **Convolutional Neural Networks auf Graph-Repräsentationen von Bildern**
  * *Problem:* Klassifikation (bessere Evaluationsmethoden)
  * *Problem:* Segmentierung
* **Evaluation**
  * Performance unterschiedlicher Graph-Repräsentationsverfahren
  * Performance CNN auf Graphen gegenüber CNN auf Bildern
  * Laufzeit CNN auf Graphen gegenüber CNN auf Bildern
  * gegenüber Graph-Kernels

## Motivation

* Potential sehr gute Resultate zu liefern
* CNNs auf Graphen sind im Allgemein kleiner als CNNs auf Bildern

## Offene Fragen

* Python 2 vs. Python 3?
* Patchy-SAN: eigene Implementation oder nachfragen?
* Wie verarbeiten andere CNNs Superpixel?
* Welche Informationen werden bei den Segmentierungsgraphen zur Klassifizierung verwendet?
* Wie kann die Form von Flächen vektorisiert werden?
* Wie können SVGs über Graphen repräsentiert werden?
