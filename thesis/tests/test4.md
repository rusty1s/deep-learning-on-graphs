# Test 4 und Auswertung: 01.02.2017

## Feature Extraction

Wir entbändigen uns unnötigen Features, die bereits aus existierenden Features
berechnet werden und vertrauen darauf, dass das Netz diese für uns ermittelt.
So kommen wir von 83 Features auf 45 Features.
So fliegen unteranderem die zentralen, skalierungsinvarianten, und
rotationsinvarianten Momente raus, da diese aus den eigentlichen Momenten
berechnet werden können.

So gilt zum Beispiel:
```
mu_00 = M_00
mu_11 = M_11 - xM_01 wobei x = M_10/M_00
...
```

Ebenso fliegen Sachen raus wie `extent = M_00 / bbox_area`.
Es bleiben folgende Features:

```
M_00 bis M_33 (16)
Bbox Width/Height (2)
Convex Area (1)
Perimter (1)
Weighted M_00 bis M_33 (16)
Mean Color (3)
Min Color (3)
Max Color (3)
==========================
45
```

**Es fehlt noch die Oriented Bounding Box.**

## Feature Selection

Ich habe eine Kovarianzmatrix aufgebaut über die Features des
Trainingdatensatzes.

<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/cov.svg"
width="800" />

## Neighborhood Assembly (Grid Spiral)

Für die Neighborhood Assembly eines Knotens wurde ein spezieller Algorithmus
implementiert, der die nächsten Knoten um den Rootknoten ähnlich wie bei einer
Spirale einsammelt.

Dies wurde implementiert, da der eigentliche Gedanke, eine Convolution auf
Basis des Grids, das SLIC erzeugt, nicht möglich ist.
Es ist daher nicht möglich, da SLIC kein vollkommenes Grid erzeugt.
Es werden teilweise Knoten hinzugefügt oder entfernt, wenn dies sinnvoll
erscheint.
Damit spuckt SLIC auch immer nur eine approximierte Anzahl an Segmenten aus,
die gewünscht waren.

Der Grid-Spiral-Algorithmus funktioniert wie folgt:

1. Der Root Knoten ist immer an Index 0.
2. Es wird der nächstgelegene Nachbar zum Rootknoten gesucht und der
   Neighborhood angehängt.
3. Es wird wiederholt ein Nachbar `y` zum letzten hinzugefügten Knoten `x`
   gesucht, sodass `w(x, y) + w(root, y)` minimal.

Die Betrachtung von `w(root, y)` hilft, das wir zentral beim Root Knoten
bleiben und liefert uns das Spirallayout.

```python
neighborhoods[0] = root
x = root

for i in range(1, size):
  finde Knoten y, sodass n(x, y) und w(x, y) + w(y, root) minimal
  neighborhoods[i] = y
  x = y
```

<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/1.png"
width="400" />
<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/1_slic.png"
width="400" />

<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/2.png"
width="400" />
<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/2_slic.png"
width="400" />

<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/3.png"
width="400" />
<img
src="https://github.com/rusty1s/deep-learning/raw/master/thesis/tests/images/test4/3_slic.png"
width="400" />

## Network

Für den Trainingsdatensatz von PatchySan wird jedes Example aus PascalVoc fünf
mal durchlaufen und dabei zufällig augmentiert.

```python
image = tf.random_crop(image, shape)  # 0.75 der Shape des Bildes
image = tf.image.random_flip_left_right(image)
image = tf.image.random_saturation(image, lower=0.8, upper=1.0)
image = tf.image.random_contrast(image, lower=0.8, upper=1.0)
```

Ergebnis ist ein Bild mit Shape `[168, 168, 3]`.
Dieses wird umgewandelt in eine normalisierte Graphdarstellung der Form `[150,
18, 45]`.

Die entgültigen Daten von PatchySan werden nicht augmentiert, da ich noch nicht
weiß wie.
Sie werden jedoch linear skaliert auf Zero Mean / Unit Norm.
**Wie sinnvoll ist dies bei Nicht-Bilder?**

Die Netzwerkstruktur ist dann wie folgt:

```json
{
  "dataset": {
    "name": "patchy_san",
    "dataset": {
      "name": "pascal_voc",
    },
    "grapher": {
      "name": "segmentation",
      "segmentation": {
        "name": "slic",
        "num_segments": 300,
        "compactness": 30.0,
        "max_iterations": 10,
        "sigma": 0.0
      },
      "adjacencies_from_segmentation": [
        "euclidean_distance"
      ]
    },
    "write_num_epochs": 5,
    "distort_inputs": true,
    "node_labeling": "scanline",
    "num_nodes": 150,
    "node_stride": 2,
    "neighborhood_assembly": "grid_spiral",
    "neighborhood_size": 18
  },
  "batch_size": 128,
  "last_step": 20000,
  "learning_rate": 0.1,
  "epsilon": 1,
  "beta1": 0.9,
  "beta2": 0.999,
  "distort_inputs": false,
  "zero_mean_inputs": true,
  "network": {
    "conv": [
      {
        "comment": "Convolution with size 1 to learn new features",
        "output_channels": 64,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": { "constant": 0.0 },
        "fields": { "size": [1, 1], "strides": [1, 1] }
      },
      {
        "comment": "Reduces input to [150, 9, 128]",
        "output_channels": 128,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": { "constant": 0.0 },
        "fields": { "size": [1, 3], "strides": [1, 1] },
        "max_pool": { "size": [1, 2], "strides": [1, 2] }
      },
      {
        "comment": "Reduces input to [150, 5, 256]",
        "output_channels": 256,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": { "constant": 0.0 },
        "fields": { "size": [1, 3], "strides": [1, 1] },
        "max_pool": { "size": [1, 2], "strides": [1, 2] }
      },
      {
        "comment": "Reduces input to [150, 3, 256]",
        "output_channels": 256,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": { "constant": 0.0 },
        "fields": { "size": [1, 3], "strides": [1, 1] },
        "max_pool": { "size": [1, 2], "strides": [1, 2] }
      },
      {
        "comment": "Reduces input to [150, 1, 512]",
        "output_channels": 512,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": { "constant": 0.0 },
        "fields": { "size": [1, 3], "strides": [1, 1] },
        "max_pool": { "size": [1, 3], "strides": [1, 3] }
      }
    ],
    "fully_connected": [
      {
        "output_channels": 2048,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": {"constant": 1.0 }
      },
      {
        "output_channels": 1024,
        "weights": { "stddev": 1e-1, "decay": 0.0 },
        "biases": {"constant": 1.0 }
      }
    ],
    "softmax_linear": {
      "output_channels": 20,
      "weights": { "stddev": 1e-1, "decay": 0.0 },
      "biases": { "constant": 1.0 }
    }
  }
}
```

Es gibt demnach 5 Convolutional Layer, mit zwei darauf folgenden
Fully-Connected-Layern und einem Softmax.

```
[150, 18, 45] =>
[150, 18, 64] =>
[150, 9, 128] =>
[150, 5, 256] =>
[150, 3, 256] =>
[150, 1, 256] =>
[2048] =>
[1024] =>
[10]
```

## Evaluation

* `3.5 examples/sec`
* `30 sec/batch`
* `10.000 min => 7 days`

| Step  | Loss | Accuracy |
| -----:| ----:| --------:|
| 2500  | 2.4  | 0.27     |

Die Loss wandert nur sehr langsam nach unten, aber sie fällt.

## Future Work

### Netzwerkstruktur

Die Netzwerkstruktur von PatchySan wird nicht ganz deutlich.
Sie berücksichtigen jedoch auch die Kantenattribute des Graphen.
Das habe ich bisher noch nicht gemacht.
Damit geht bei mir die eigentliche Graphstruktur flöten und ich betrachte
lediglich Sequenzen von Knoten, die *irgendwie* zusammenhängen.

Dabei erhalte ich einen Tensor mit Shape `[Nodes, Neighborhood, Channels]`.
Ich kann mir aber weiterhin einen Tensor für die Kantenattribute generieren.
Dieser würde dann eine Shape von `[Nodes, Neighborhood, Neighborhood,
Channels]` haben und müsste demnach reshaped werden.
PatchySan reshaped alles zu einem zweidimensionalen Tensor, ich betrachte
jedoch dreidimensionale.
Das ist eigentlich gehüpft wie gesprungen, solange ich nicht unterschiedliche
Nachbarschaften convolve.
Die Kantenattribute müssten demnach reshaped werden zu
`[Nodes, Neighborhood * Neighborhood, Channels]`.
PatchySan erwähnt die Verwendung von **Merge Layern**.
Die Evaluierung und Verwendung von diesen sogenannten Merge Layern, bei denen
die beiden Tensoren im späteren Verlauf des Netzes zusammengefügt werden, soll
weiter ausgearbeitet werden, um dem Netz damit mehr Informationen zu geben.

### Datensatz

Desweiteren ist der Datensatz PascalVOC wahrscheinlich nicht so gut für
Bildklassifierung geeignet.
Der Datensatz gibt einige Bilder, auf denen mehrere Klassen vorhanden sind.
Diese werden aber teilweise schon herausgefiltert, sodass ich lediglich 3500
Trainingsbilder erzeuge für 20 Klassen.
Das ist meiner Einschätzung nach extrem wenig.
Zudem gibt es einige sehr schwierige Klassen wie `DiningTable` und `Chair`,
an/auf denen dann Personen sitzen oder sich Gläser befinden, die ebenfalls
klassifiziert werden können.
Ebenso nehmen Objekte teilweise das ganze Bild ein oder sind abgeschnitten.
Ein Random-Cropping auf diesen Bildern hat daher eher negative als positive
Auswirkungen.

Ziel ist es daher, den ImageNet Datensatz zu verwenden (PascalVOC bildet eine
Untermenge von ImageNet).
Da der Grunddatensatz `2.5 TB` Daten besitzt mit mehr als 2000 Klassen, muss
dieser erst aufbereitet werden.

### Motiviation

Die Motivation, ein Bild erst in einen Graphen umzustrukturieren, ist unter
anderem die Datenreduktion.
Es liegt aber für den Input des Netzes garkeine Datenreduktion vor (zumindest
nicht bei eher quadratischen Regionen wie SLIC).
Statt eines Inputs von `[168, 168, 3] = 84.672` haben wir einen Input von
`[150, 18, 45] = 121.500`, was deutlich höher ist.
Werden weiterhin Kantenattribute betrachtet, so steigt diese Anzahl.

### Laufzeit

Die Laufzeit der Netze ist sehr frustierend.
Ich durchlaufe `3.5 examples/sec` und brauche damit für einen Batch mit Size
`128` ungefähr 30 Sekunden.
Hochgerechnet auf `20.000` Steps sind das `10.000` Minuten, was ungefähr 7
Tagen entspricht.

### TensorFlow Update

TensorFlow hat seit kurzem die erste `1.0` Version veröffentlicht und
`tensorflow` nun offiziell als `pip install` hinzugefügt.
Ein Update auf `1.0` wird vorgesehen.
