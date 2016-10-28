import tensorflow as tf

input_value = tf.constant(1.0, name='input')
weight = tf.Variable(0.8, name='weight')
output_value = tf.mul(weight, input_value, name='output')

# initialize variables
init = tf.initialize_all_variables()

summary_writer = tf.train.SummaryWriter('log_graph', tf.get_default_graph())

# 1. run this script to create the log files
# 2. run `tensorboard --logdir=log_graph` on the command line
