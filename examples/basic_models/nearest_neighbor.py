# A nearest neighbor learning algoritm example. This example is using the MNIST 
# database of handwritten digits: http://yann.lecun.com/exdb/mnist.

import numpy as np
import tensorflow as tf

# load mnist data:
# => 55,000 data points of training data (mnist.train)
# => 10,000 data points of test data (mnist.test)
#
# Every mnist data point has two parts: an image of a handwritten digit and a 
# corresponding label.
# 'One-hot-vector': a vector which is 0 in most dimensions, and 1 in a single 
# dimension => 3 would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)

# limit mnist data
train_images, train_labels = mnist.train.next_batch(5000)
test_images, test_labels = mnist.test.next_batch(200)

# Create placeholders, we'll input when we ask TensorFlow to run a computation.
# Mnist images are 28 pixels by 28 pixels, so we can flatten it into a vector 
# of 28x28 = 784 numbers.
# The first dimension is an index into the list of images.
# => e.g. train_images is a [5000, 784] and train_labels is a [5000, 10] array 
# of floats
img_size = 28
img_size_flat = img_size * img_size

train_x = tf.placeholder(tf.float32, [None, img_size_flat]) # variable indices
test_x = tf.placeholder(tf.float32, [img_size_flat]) # one index

#
# Nearest neighbor calculation using L1 distance:
#

# calculate L1 distances
distance = tf.reduce_sum(tf.abs(tf.sub(train_x, test_x)), reduction_indices=1)
# get min distance index (nearest neighboar)
prediction = tf.argmin(distance, 0)

# initialize the variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # initialize accuracy variable
    accuracy = 0.

    # loop over test data
    for i in range(len(test_images)):
        # get the nearest neighbor
        # n_index points to the index in train_labels
        nn_index = sess.run(prediction, feed_dict={ train_x: train_images, test_x: test_images[i] })

        # get the calculated label and the true label
        calculated_label = np.argmax(train_labels[nn_index])
        true_label = np.argmax(test_labels[i])

        # get nearest neighbor class label and compare it to its true label
        print ('Test', i+1, 'Prediction:', calculated_label, 'True Class:', true_label)

        # calculate new accuracy
        if calculated_label == true_label:
            accuracy += 1./len(test_images)

    print ('Done!')
    print ('Accuracy:', accuracy)
