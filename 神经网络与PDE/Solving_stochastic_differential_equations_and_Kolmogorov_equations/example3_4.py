import numpy as np
import tensorflow as tf
from common import kolmogorov_train_and_test

tf.reset_default_graph()
dtype = tf.float32
T, N, d = 1., 1, 100
r, c, K = 0.05, 0.1, 100.
Q = np.ones([d, d]) * 0.5
np.fill_diagonal(Q, 1.)
L = np.linalg.cholesky(Q).transpose()
sigma_norms = tf.constant(np.linalg.norm(L, axis=0), dtype=dtype)
sigma = tf.constant(L, dtype=dtype)
beta = tf.constant(0.1 + 0.5 * np.linspace(start=1. / d, stop=1., num=d, endpoint=True), dtype=dtype)

batch_size = 8192
neurons = [d + 100, d + 100, 1]
train_steps = 750000
mc_rounds, mc_freq = 10, 25000
mc_samples_ref, mc_rounds_ref = 1024, 1024
lr_boundaries = [250001, 500001]
lr_values = [0.001, 0.0001, 0.00001]
xi = tf.random_uniform(shape=(batch_size, d), minval=90., maxval=110., dtype=dtype)


def phi(x, axis=1):
    return np.exp(-r * T) * tf.maximum(K - tf.reduce_min(x, axis=axis, keep_dims=True) - K, 0.)


def mc_body(idx, p):
    _w = tf.matmul(tf.random_normal((mc_samples_ref * batch_size, d), stddev=np.sqrt(T / N), dtype=dtype), sigma)
    _w = tf.reshape(_w, (mc_samples_ref, batch_size, d))
    _x = xi * tf.exp((r - c - 0.5 * (beta * sigma_norms) ** 2) * T + beta * _w)
    return idx + 1, p + tf.reduce_mean(phi(_x, 2), axis=0)


x_sde = xi * tf.exp((r - c - 0.5 * (beta * sigma_norms) ** 2) * T + beta * tf.matmul(tf.random_normal((mc_samples_ref, batch_size, d), stddev=np.sqrt(T / N), dtype=dtype), sigma))
_, u = tf.while_loop(lambda idx, p: idx < mc_rounds_ref, mc_body, (tf.constant(0), tf.zeros((batch_size, 1), dtype)))
u_reference = u / tf.cast(mc_rounds_ref, tf.float32)
kolmogorov_train_and_test(xi, x_sde, phi, u_reference, neurons, lr_boundaries, lr_values, train_steps, mc_rounds, mc_freq, 'example3_4.csv', dtype)