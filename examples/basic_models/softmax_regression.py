# A softmax regression algoritm example. This example is using the MNIST 
# database of handwritten digits: http://yann.lecun.com/exdb/mnist.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)

img_size = 28
img_size_flat = img_size * img_size

x = tf.placeholder(tf.float32, [None, img_size_flat])

# Softmax:
#
# We want to be able to look at an image and give the probabilities for it 
# being each digit, e.g. our model might look at a picture of a nine and be 80% 
# sure it's a nine, but give a 5% change to it being an eight and a bit of 
# probability to all the others because it isn't 100% sure.
# This is a classic case where a softmax regression is the thing to do, because 
# softmax gives us a list of values between 0 and 1 that add up to 1.
weight = tf.Variable(tf.zeros([img_size_flat, 10]))
bias = tf.Variable(tf.zeros([10]))

evidence = tf.matmul(x, weight) + bias

# Training:
#
# The network is trained by minimizing a cost functin called cross-entropy.
y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(evidence, y))

# We minimize the cross-entropy by gradient descent with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Testing:
#
# We evaluate the network by comparing the real label with the computed highest 
# label and computing the accuracy over all input images x.
correct_prediction = tf.equal(tf.argmax(evidence, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# train the network 1000 times
for _ in range(1000):
    batch_images, batch_labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={ x: batch_images, y: batch_labels })

# run the test
print('Accuracy on train images:', sess.run(accuracy, feed_dict= { x: mnist.train.images, y: mnist.train.labels }))
print('Accuracy on test images:', sess.run(accuracy, feed_dict= { x: mnist.test.images, y: mnist.test.labels }))
