# A convolutional network example. This example is using the MNIST database of 
# handwritten digits: http://yann.lecun.com/exdb/mnist.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)
img_size = 28

# parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
model_path = 'convolutional.ckpt'

# network parameters
n_input = img_size * img_size
n_output = 10
dropout = 0.75 # probability to keep units
