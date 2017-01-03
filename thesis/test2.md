# Test 03.01.2017

Ich frage mich gerade, wie ich den Graphen aufbauen kann, damit er gescheite
Ergebnisse mit Patchy-San liefern kann.

Slic(o) auf dem Cifar-10 Datensatz liefert bescheidene Ergebnisse.
Wir betrachten beim Traininieren ein Bild von 24x24 Pixeln.
Eine Anwendung von Slic(o) mit einer beliebigen Anzahl an gewünschten Segmenten 
liefern stets Quadrate.
Das ist enttäuschend.
Wir reduzieren damit unser Bild auf ein kleineres und haben lediglich die
Meanfarbe.
Jedes weitere Feature, dass wir hinzufügen, ist bereits in den beiden Features
Farbe und Quadratgröße enhalten.
Darauf werden wir niemals bessere Ergebnisse erzielen können, als auf dem
eigentlichen Bild.
Das heißt, dass wir entweder einen besseren Superpixelalgorithmus brauchen, der
dann entsprechend auch zeitintensiver ist als Slic(o) oder diese Tatsache ganz
einfach tolerieren und uns nur mit größeren Bildern beschäftigen.

Cifar-10 rechnet auf Batches von [24, 24, 3].
Wir rechnen auf Batches von [Knotengröße, Nachbarschaftsgröße, Channels].
Damit erhalten wir definitiv einen größeren Batch.
Der Vorteil der sich einstellt ist, dass dieser Batch bereits eine
vordefinierte Convolution ist.
Das heißt wir rechnen nicht wie klassisch üblich auf 2 Convolution-Layern,
sondern nur noch einem.

Angenommen wir haben einen Superpixelalgorithmus, der traumhafte Ergebnisse
liefert.
Die Knoten bilden jeweils ein Segment mit darin enthaltenen Features:
* Farbe (Mean, Absolute difference)
* Schwerpunkt (Formattribut)
* Anzahl Pixel (Formattribut)
* Ausdehnung, also Höhe und Breite (Formattribut)
* ...

Dann können wir unseren Graphen aufbauen, in dem wir Kantenattribute
definieren.
Jedes Kantenattribut spiegelt eine Adjazenzmatrix wieder
* Distanz
* Farbunterschied
* lokale Nachbarn (mit Distanz (mit Treshold oder ohne) oder ohne)
* ...

Unterschiedliche Kantenattribute und Knotenattribute können auch kombiniert
werden, so dass wir mehrere Graphen erhalten mit unterschiedlichen Attributen,
wobei jeder Knoten jeweils das gleiche Segment beschreibt.

## Auswertung

* 20.000 Steps a 128 Batch-Size: 150 Minuten
* Beginn 21 Loss (kommt mir sehr hoch vor)
* 1400. Step: Loss 10, Acc 0.3-0.4
* 2500. Step: Loss 5.5, Acc unverändert
* 4000. Step: Loss 3, Acc unverändert
* Slico(50)
* [25, 10, 8] Input mit 50 Width und Stride-Size 2
* Features: rgb, relative center, count, height, width
* Konvertierung von 50000 Bildern 24x24 Pixeln (distorted) => ungefähr 1 Stunde
* Node labeling: Order, Neighborhood Labeling: Betweenness centrality
* Learning ungefähr doppelt so schnell wie auf normalem CIFAR-10 Datensatz

## Evaluation

Viele Features sind unnötig, weil sie immer das gleiche beinhalten, was durch
die gleiche Form der Superpixel zu erklären ist.
Darunter fallen center, count, width, height.
Also 5 von 8 Features sind unnötig (zumindest für das kleine CIFAR-10 Netz).

## ToDo

* Endscreenshot
* Example Cifar-10 mit SLIC
