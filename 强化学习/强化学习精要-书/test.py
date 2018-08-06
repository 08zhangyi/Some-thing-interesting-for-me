import tensorflow as tf

with tf.name_scope('123'):
    with tf.name_scope(None):
        c = tf.Variable(1, name='f')
        print(c.name)