import tensorflow as tf

b = tf.placeholder(tf.float32, [None, 100])
print(dir(b))