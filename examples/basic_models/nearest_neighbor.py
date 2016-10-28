# A nearest neighbor learning algoritm example. This example is using the MNIST 
# database of handwritten digits: http://yann.lecun.com/exdb/mnist

import numpy as np
import tensorflow as tf

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)
