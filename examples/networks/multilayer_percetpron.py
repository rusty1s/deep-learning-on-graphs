# A multilayer perceptron network example. This example is using the MNIST 
# database of handwritten digits: http://yann.lecun.com/exdb/mnist.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)
img_size = 28

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
model_path = 'multilayer_perceptron.ckpt'

# network parameters
n_input = img_size * img_size
n_hidden_1 = 256
n_hidden_2 = 256
n_output = 10

# model creation function
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # activation function is relu for perceptrons
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# graph variables
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output])),
}

# create the model
pred = multilayer_perceptron(x, weights, biases)

# define cost function and training step
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize variables
init = tf.initialize_all_variables()

# 'Saver'  op to save and restore all the variables
saver = tf.train.Saver()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batches = int(mnist.train.num_examples/batch_size)

        for i in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict= { x: batch_x, y: batch_y })
            cost = sess.run(cross_entropy, feed_dict = { x: batch_x, y: batch_y })

            avg_cost += cost / total_batches

        # display information
        if epoch % display_step == 0:
            print('Epoch:', epoch+1, 'Cost:', avg_cost)

    print('Finished optimizing!')
    save_path = saver.save(sess, model_path)
    print('Model saved in file:', save_path)

    # run tests
    print('Accuracy on train images:', sess.run(accuracy, feed_dict= { x: mnist.train.images, y: mnist.train.labels }))
    print('Accuracy on test images:', sess.run(accuracy, feed_dict= { x: mnist.test.images, y: mnist.test.labels }))
