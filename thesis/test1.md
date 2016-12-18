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

## Was mache ich beim Lernen falsch?

2D Conv auf Cifar-10 erreicht schlechte Ergebnisse. Warum?

```python
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

flat_size = 8 * 8 * 64
W_fc1 = weight_variable([flat_size, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, flat_size])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,
                                                                       y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
    images, labels = train_set.next_batch(50)
    train_step.run(feed_dict={
        x: images, y_: labels, keep_prob: 0.5,
    })

    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: images, y_: labels, keep_prob: 1.0,
        })
        print('Step', i, 'Training accuracy', train_accuracy)
```

**Ergebnisse:** Training accuracy nach 4000 Steps: `0.1`, also random????? Learning Rate schuld, weil zu klein?

## Codeprobleme

Cifar10 Konvertierung:
Wir haben ein Batch von 10000 Bildern kodiert als `[1024 *red, 1024*green, 1024*blue]`, also eine Shape von `[10000, 3072]`. Diese wollen wir in eine Shape von `[10000, 32 32, 3]` bringen.

```python
data = np.zeros((len(batch['data']), 32, 32, 3))

for i in range(len(batch['data'])):
    labels[i][batch['labels'][i]] = 1.0

    image = batch['data'][i]

    red = image[0:1024].reshape(32, 32)
    green = image[1024:2048].reshape(32, 32)
    blue = image[2048:3072].reshape(32, 32)

    data[i] = np.dstack((red, green, blue))
```

Es kommt mir unelegant vor, das so zu machen.

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
