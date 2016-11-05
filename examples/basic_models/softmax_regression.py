# A softmax regression algoritm example. This example is using the MNIST 
# database of handwritten digits: http://yann.lecun.com/exdb/mnist.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)

train_images, train_labels = mnist.train.next_batch(5000)
test_images, test_labels = mnist.test.next_batch(200)

img_size = 28
img_size_flat = img_size * img_size

# placeholders for training and test images, that we can input into the network
train_x = tf.placeholder(tf.float32, [None, img_size_flat])
test_x = tf.placeholder(tf.float32, [img_size_flat])

# Softmax Regressions:
#
# We want to be able to look at an image and give the probabilities for it 
# being each digit, e.g. our model might look at a picture of a nine and be 80% 
# sure it's a nine, but give a 5% change to it being an eight and a bit of 
# probability to all the others because it isn't 100% sure.
# This is a classic case where a softmax regression is the thing to do, because 
# softmax gives us a list of values between 0 and 1 that add up to 1.
#
# A softmax regression has two steps:
# 1. We add up the evidence of our input being in certain classes. To tally up 
#    the evidence that a given image is in a particular class, we do a weighted 
#    sum of the pixel intensities over all training images over all classes.  
#    The weight is negative if that pixel having is negative if that pixel 
#    having a high intensity is evidence against the image being in that class, 
#    and positive if it is evidence in favor.
#    We also add some extra evidence called a bias, to be able to say that some 
#    things are more likely independent of the input.
# 2. We convert that evidence into probabilities.

# variables for weights and bias, that TensorFlow can modify
weight = tf.Variable(tf.zeros([img_size_flat, 10]))
bias = tf.Variable(tf.zeros([10]))

softmax = tf.nn.softmax(tf.matmul(train_x, weight) + bias)
