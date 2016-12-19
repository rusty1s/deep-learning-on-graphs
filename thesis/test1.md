# Test 1 16.12.2016

Ich habe den Code soweit fertiggestellt, dass man nun aus einer beliebigen
Superpixelrepräsentation einen Graphen generieren kann und dieser dann in eine
1D Matrix umgewandelt wird der Form

```
[Neighborhood1, Neighborhood2, ...]
```

Die einzelnen Knoten der Nachbarschaften fassen die Features, sodass wir
eigentlich von einer 2D Matrix sprechen können der Form

```
[Nachbarschaftsgröße * Größe der Knotenauswahl, Channels]
```

Features sind:
* Position (x,y)
* Anzahl Pixel
* RGB

Die Auswahl der Knoten und der Nachbarschaften verfolgt in Scan-Line-Order:

* Nachbarschaftsgröße: 9
* Knoten: 100
* Stride der Knotenauswahl: 1

```python
rep = image_to_slic_zero(image, NUM_SEGMENTS,
    compactness=COMPACTNESS,
    max_iterations=MAX_ITERATIONS,
    sigma=SIGMA)

superpixels = extract_superpixels(image, rep)

graph = create_superpixel_graph(superpixels, node_mapping,
    edge_mapping)

fields = receptive_fields(graph, order, STRIDE, WIDTH, SIZE,
    node_features, NODE_FEATURE_SIZE)
```

## SLIC

SLIC-Zero auf Cifar-100 angewendet mit folgenden Parametern:
* Segmente: `100`
* Sigma: `0.0` (Width of gaussian smoothing kernel for pre-processing)
* Compactness: `1.0` (Balances color proximity and space proximity. A higher
  value gives more weight to space proximity making superpixel shapes more
  square/cubic)
* Max Iterations: `10` (Iterationen of k-means)

## Netz

Das Netz besteht aus 2 mal Convolution + Max Pooling und einem Fully-Connected
Layer.

* Eingabe: `[1, 1, 900, 6]`
* Conv1: 32 Features, Patch und Stride wie Nachbarschaftsgröße 9 (das heißt
  eine Nachbarschaft wird in 32 Feature umgewandelt), Max pool 2
  Conv2: 64 Features, Patch 3, Stride 1, Max pool 2
* Fully: ein Layer mit 1024 Neuronen
* Output: 10

## Probleme

Das Netz lernt nicht. Wir erreichen 30% Accuracy bei den Trainingsdaten was
schlecht ist.
Eventuell ist dies der zweiten Conv Ebene geschuldet. Da wir auf 1D Daten 
arbeiten macht die eventuell keinen Sinn. Benachbarte Knoten sind nicht 
unbedingt benachbart im Bild, dann macht eine Verbindung dieser eventuell 
garkeinen Sinn. Beispiel: Wenn wir ganz rechts sind, ist der nächste Knoten 
eine Ebene drunter ganz links.

## Codeprobleme

Numpy Beispiele

## Weiteres Vorgehen

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
