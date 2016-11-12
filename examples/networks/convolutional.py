# A convolutional network example. This example is using the MNIST database of 
# handwritten digits: http://yann.lecun.com/exdb/mnist.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)
img_size = 28

# parameters
learning_rate = 0.0001
training_epochs = 20000
batch_size = 50
display_step = 10

# network parameters
n_input = img_size * img_size
n_output = 10
dropout = 0.5 # probability to keep units

# helper functions
def variable(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# model creation function
def convolutional(x, weights, biases, dropout):
    # reshape the input to a 4d matrix => last dimension = color channels
    x = tf.reshape(x, shape=[-1, img_size, img_size, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['1']) + biases['1'])
    pool1 = maxpool_2x2(conv1)

    conv2 = tf.nn.relu(conv2d(pool1, weights['2']) + biases['2'])
    pool2 = maxpool_2x2(conv2)

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(pool2_flat, weights['flat']) + biases['flat'])

    # apply dropout to reduce overfitting
    fc = tf.nn.dropout(fc, dropout)

    # output, class prediction
    return tf.matmul(fc, weights['out']) + biases['out']

# graph variables
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights = {
    # 5x5 patches, one input channel and 32 output channels
    '1': variable([5, 5, 1, 32]),
    # 5x5 patches, 32 input channels and 64 output channels
    '2': variable([5, 5, 32, 64]),
    # image sizes has been reduced to 7x7 with 64 features each
    # add fully-connected layer with 1024 neurons
    'flat': variable([7*7*64, 1024]),
    # 1024 inputs, 10 outputs, class prediction
    'out': variable([1024, n_output]),
}

biases = {
    # bias for each of the 32 output channels
    '1': variable([32]),
    # bias for each of the 64 output channels
    '2': variable([64]),
    # bias for each of the 1024 output channels
    'flat': variable([1024]),
    # bias for each of the 10 output channels
    'out': variable([n_output]),
}

# create the model
pred = convolutional(x, weights, biases, keep_prob)

# define cost function and training step
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict= { x: batch_x, y: batch_y, keep_prob: dropout })

        # display information
        if epoch % display_step == 9:
            acc = sess.run(accuracy, feed_dict = { x: batch_x, y: batch_y, keep_prob: 1.0 })
            print('Epoch:', epoch+1, 'Accuracy:', acc)

    print('Finished optimizing!')

    # run tests
    print('Accuracy on test images:', sess.run(accuracy, feed_dict= { x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0 }))
