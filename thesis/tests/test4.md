# Test 4 und Auswertung: 01.02.2017

## Feature Extraction

## Feature Selection

## Neighborhood Assembly (Grid Spiral)

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
die beiden Tensoren zusammengefügt wird, soll weiter ausgearbeitet werden, um
dem Netz damit mehr Informationen zu geben.

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

### TensorFlow Update

TensorFlow hat seit kurzem die erste `1.0` Version veröffentlicht und
`tensorflow` nun offiziell als `pip install` hinzugefügt.
Ein Update auf `1.0` wird vorgesehen.
