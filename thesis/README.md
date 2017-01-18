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
* ggf. mehr Augmentierungsmöglichkeiten zusätzlich zur Bildaugmentierung

## Schwächen

### Augmentierung

* Augmentierung der Daten ist im Bereich Deep Learning sehr essenziell
* Augmentierung auf den Bildern oder auf der Superpixel Repräsentation oder auf
  den Graphen?
* Augmentierung ist eine Möglichkeit, dem Netz zu zeigen, welche Sachen
  irrelevant für die Klassifikation sind (z.B. unabhängig von Brightness oder
  Normalisieren auf zero mean/one stddev)
* Distorted Inputs sind nicht ohne weiteres möglich, da z.B. Farbänderungen die
  Graphstruktur verändern können

## Offene Fragen

* Wie verarbeiten andere CNNs Superpixel?
* Welche Informationen werden bei den Segmentierungsgraphen zur Klassifizierung
  verwendet?
* Wie können SVGs über Graphen repräsentiert werden?
