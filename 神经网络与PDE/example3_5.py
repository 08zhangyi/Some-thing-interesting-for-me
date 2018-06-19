import numpy as np
import tensorflow as tf
from common import kolmogorov_train_and_test

tf.reset_default_graph()
dtype = tf.float32
batch_size = 1024

T, N, d = 1., 100, 3
alpha_1, alpha_2, alpha_3 = 10., 14., 8./3.
beta = tf.constant([0.15, 0.15, 0.15], dtype=dtype)
h = T / N
neurons = [d + 20, d + 20, 1]
train_steps = 750000
mc_rounds, mc_freq = 20, 25000
mc_samples_ref, mc_rounds_ref = 1024, 1024
lr_boundaries = [250001, 500001]
lr_values = [0.001, 0.0001, 0.00001]
xi = tf.stack([tf.random_uniform((batch_size,), minval=0.5, maxval=2.5, dtype=dtype), tf.random_uniform((batch_size,), minval=8., maxval=10., dtype=dtype), tf.random_uniform((batch_size,), minval=10., maxval=12., dtype=dtype)], axis=1)


def phi(x, axis=1):
    return tf.reduce_sum(x**2, axis=axis, keepdims=True)


def mu(x):
    x_1 = tf.expand_dims(x[:, :, 0], axis=2)
    x_2 = tf.expand_dims(x[:, :, 1], axis=2)
    x_3 = tf.expand_dims(x[:, :, 2], axis=2)
    return tf.concat([alpha_1 * (x_2 - x_1), alpha_2 * x_1 - x_2 - x_1 * x_3, x_1 * x_2 - alpha_3 * x_3], axis=2)


def sde_body(idx, s, samples):
    return tf.add(idx, 1), s + tf.cast(T / N * tf.sqrt(phi(mu(s), 2 if samples > 1 else 1)) <= 1.0, dtype) * mu(s) * T / N + beta * tf.random_normal((samples, batch_size, d), stddev=np.sqrt(T / N), dtype=dtype)


def mc_body(idx, p):
    _, _x = tf.while_loop(lambda _idx, s: _idx < N, lambda _idx, s: sde_body(_idx, s, mc_samples_ref), loop_var_mc)
    return idx + 1, p + tf.reduce_mean(phi(_x, 2), axis=0)


loop_var_mc = (tf.constant(0), tf.ones((mc_samples_ref, batch_size, d), dtype) * xi)
loop_var = (tf.constant(0), tf.ones((1, batch_size, d), dtype) * xi)
_, x_sde = tf.while_loop(lambda idx, s: idx < N, lambda idx, s: sde_body(idx, s, 1), loop_var)
_, u = tf.while_loop(lambda idx, p: idx < mc_rounds_ref, mc_body, (tf.constant(0), tf.zeros((batch_size, 1), dtype)))
u_reference = u / tf.cast(mc_rounds_ref, tf.float32)
kolmogorov_train_and_test(xi, x_sde, phi, u_reference, neurons, lr_boundaries, lr_values, train_steps, mc_rounds, mc_freq, 'example3_2.csv', dtype)