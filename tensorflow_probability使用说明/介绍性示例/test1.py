import tensorflow_probability.python.edward2 as ed
import tensorflow as tf


def model():
    p = ed.Beta(1., 1., name="p")  # RandomVariable对象，实际上是对distribution中的对象进行了包装
    x = ed.Bernoulli(probs=p, sample_shape=50, name="x")
    return x


x = model()  # shape=(50,)
print(x)
with tf.Session() as sess:
    print(sess.run(x))