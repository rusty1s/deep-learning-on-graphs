# Master-Thesis

* **Universität:** TU Dortmund
* **Fakultät:** Lehrstuhl für Graphische Systeme

# Convolutional Neural Networks auf Graphrepräsentationen von Bildern

## Inhalt

* Motivation
* Relevante Arbeiten
* verschiedene **Graph-Repräsentationen von Bildern**
  * Segmentierungsgraphen
  * Superpixel
    * verschiedene Superpixelberechnungsverfahren
  * Vektorgraphiken als Graph- bzw. Baum-Repräsentation
* **Convolutional Neural Networks auf Graphen**
  * Verfahren
    * Multi-Scale Feature Maps
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
* weniger Trainingsdaten benötigt?
* Rauschentfernung (Superpixelrepräsentation zwischen komprimiertem JPEG und
  PNG)
* keine strikte Größe der Bilder notwendig

## Schwächen

* Distorted Inputs sind nicht ohne weiteres möglich, da z.B. Farbänderungen die
  Graphstruktur verändern können.

## Offene Fragen

* Wie verarbeiten andere CNNs Superpixel?
* Welche Informationen werden bei den Segmentierungsgraphen zur Klassifizierung
  verwendet?
* Wie kann die Form von Flächen vektorisiert werden?
* Wie können SVGs über Graphen repräsentiert werden?
