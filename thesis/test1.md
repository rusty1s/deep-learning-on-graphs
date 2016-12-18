# Test 1 16.12.2016

Ich habe den Code soweit fertiggestellt, dass man nun einfach aus einer
beliebigen Superpixelrepräsentation einen Graphen generieren kann und dieser
dann in eine 1D Matrix umgewandelt wird der Form

```
[Neighborhood1, Neighborhood2, ...]
```

Die einzelnen Knoten der Nachbarschaften fassen dann noch die Features, sodass
wir eigentlich von einer 2D Matrix sprechen können der Form

```
[Nachbarschaftsgröße * Größe der Knotenauswahl, Channels]
```

Features sind:
* Position (x,y)
* Anzahl Pixel
* RGB

Die Auswahl der Knoten und der Nachbarschaften verfolgt in Scan-Line-Order.

* Nachbarschaftsgröße: 9
* Knoten: 100
* Stride der Knotenauswahl: 1

## SLIC

Slic auf CIFAR-100 angewendet mit folgenden Parametern:
* Segmente: 100
* Sigma: 0.0
* Compactness: 1
* Max Iterations: 10

## Netz

Das Netz besteht aus 2 Convolutions + Max Pooling und einem Fully-Connected
Layer.

* Eingabe [1, 1, 900, 6]
* Conv1: 32 Features, Größe und Stride wie Nachbarschaftsgröße 9, das heißt
  eine Nachbarschaft wird umgewandelt in 32 Features, max pool 2
  Conv2: 64 Features, patch 3, stride 1, max pool 2
* Fully: ein layer mit 1024
* Output: 10

Das Netz läuft unfassbar schnell, das ist mit Sicherheit zum Großteil der 1d 
Convolution geschuldet, es ist trotzdem bemerkenswert schnell.

## Probleme

Das Netz lernt nicht. Wir erreichen 30% Accuracy bei den Trainingsdaten was
schlecht ist.
Es ist anzumerken, dass das klassische Convoultional Netz auf dem Cifar-10
Datensatz auch nicht lernt. Ich glaube da mache ich einfach irgendwas falsch.
Ich weiß nur nicht was.
Eventuell ist dies der zweiten Conv Ebene geschuldet. Da wir auf 1D Daten
arbeiten macht die eventuell keinen Sinn. Benachbarte Knoten sind nicht
unbedingt benachbart im Bild, dann macht eine Verbindung dieser eventuell
garkeinen Sinn. Beispiel: Wenn wir ganz rechts sind, ist der nächste Knoten
eine Ebene drunter ganz links.

## Codeprobleme

Numpy Beispiele

## Weiteres Vorgehen

* Nach der ersten Conv das ganze in eine 2 dimensionale Form bringen
* Code von Conv1 sollte zu Conv2 ausgeweitet werden
* Trainieren auf 1D Cifar und 2D Cifar Bildern anstatt auf Graphen (zum 
  Vergleich)
* Gucken was Graph Kernels lernen bzw welche Features sie generieren zur 
  Bildklassifierung
* Vielleicht ist die Positionsangabe schuld
* Zweite Conv-Ebene mal rauslassen
* größere Nachbarschaften betrachten (bisher unsere einzige Möglichkeit 
  Lokalität auszudrücken)
* Cifar10 Klasse umbauen, sodass sie uns 1d und 2d Eingaben liefert um darauf 
  zu lernen
* Alle Datensätze in der Zukunft sollten dem Prinzip `data` und `labels` 
  folgen. Damit können wir eine universelle Klasse schaffen, um `next_batch` zu 
  realisieren, die die Pfade zu den Batches nehmen kann, und diese verwaltet.
