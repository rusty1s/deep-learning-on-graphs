import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print('a=2, b=3')
    print('Addtion with constants: %i' % sess.run(a+b))
    print('Multiplication with constants: %i' % sess.run(a*b))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print('Addition with variables: %i' % sess.run(add, feed_dict={ a: 2, b: 3 }))
    print('Multiplication with variables: %i' % sess.run(mul, feed_dict={ a: 2, b: 3 }))

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print('a = [[3, 3]], b=[[2],[2]]')
    print('Matrix multiplication:')
    print(sess.run(product))
