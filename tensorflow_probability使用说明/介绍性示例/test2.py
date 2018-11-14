import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import edward2 as ed


def deep_exponential_family(data_size, feature_size, units, shape):
    # units表示每层维数的大小，从unit[2]开始到unit[0]，最后输出feature_size
    # data_size为batch_size的意思
    w2 = ed.Gamma(0.1, 0.3, sample_shape=[units[2], units[1]], name="w2")  # 前两个位置为参数
    w1 = ed.Gamma(0.1, 0.3, sample_shape=[units[1], units[0]], name="w1")
    w0 = ed.Gamma(0.1, 0.3, sample_shape=[units[0], feature_size], name="w0")

    z2 = ed.Gamma(0.1, 0.1, sample_shape=[data_size, units[2]], name="z2")
    print(z2)
    z1 = ed.Gamma(shape, shape / tf.matmul(z2, w2), name="z1")  # z1的形状跟着两个参数的形状跑，此处相当于对rate建模，concentration不建模
    print(z1)
    z0 = ed.Gamma(shape, shape / tf.matmul(z1, w1), name="z0")
    print(z0)
    x = ed.Poisson(tf.matmul(z0, w0), name="x")
    return x


x = deep_exponential_family(100, 2, [65, 25, 17], 0.1)
# z2的shape=(100, 17)
# z1的shape=(100, 25)
# z0的shape=(100, 65)
with tf.Session() as sess:
    print(sess.run(x))