import numpy as np
import tensorflow as tf
from common import kolmogorov_train_and_test

tf.reset_default_graph()
dtype = tf.float32
batch_size = 1024

T, N, d = 1., 100, 3
alpha, K = 0.05, 110.
kappa, sigma = 0.6, 0.2
theta, rho = 0.04, -0.8
S_0 = tf.random_uniform((batch_size, d/2), minval=90., maxval=110., dtype=dtype)
V_0 = tf.random_uniform((batch_size, d/2), minval=0.02, maxval=0.2, dtype=dtype)
h = T / N
neurons = [d + 50, d + 50, 1]
train_steps = 750000
mc_rounds, mc_freq = 20, 25000
mc_samples_ref, mc_rounds_ref = 256, 4096
lr_boundaries = [250001, 500001]
lr_values = [0.001, 0.0001, 0.00001]
xi = tf.reshape(tf.stack([S_0, V_0], asix=2), (batch_size, d))


def phi(x, axis=1):
    return np.exp(-alpha * T) * tf.maximum(K - tf.reduce_mean(tf.exp(x), axis=axis, keepdims=True), 0.)


def sde_body(idx, s, v, samples):
    _sqrt_v = tf.sqrt(v)
    dw_1 = tf.random_normal(shape=(samples, batch_size, d / 2), stddev=np.sqrt(h), dtype=dtype)
    dw_2 = rho * dw_1 + np.sqrt(1. - rho ** 2) * tf.random_normal(shape=(samples, batch_size, d / 2), stddev=np.sqrt(h), dtype=dtype)
    return tf.add(idx, 1), s + (alpha - v / 2.) * h + _sqrt_v * dw_1, tf.maximum(tf.maximum(np.float32(sigma / 2. * np.sqrt(h)), tf.maximum(np.float32(sigma / 2. * np.sqrt(h)), _sqrt_v) + sigma / 2. * dw_2) ** 2 + (kappa * theta - sigma ** 2 / 4. - kappa * v) * h, 0.)


def mc_body(idx, p):
    _, _x, _v = tf.while_loop(lambda _idx, s, v: _idx < N, lambda _idx, s, v: sde_body(_idx, s, v, mc_samples_ref), loop_var_mc)
    return idx + 1, p + tf.reduce_mean(phi(_x, 2), axis=0)


loop_var_mc = (tf.constant(0), tf.ones((mc_samples_ref, batch_size, d / 2), dtype) * tf.log(S_0))
loop_var = (tf.constant(0), tf.ones((1, batch_size, d / 2), dtype) * tf.log(S_0))
_, x_sde, _v = tf.while_loop(lambda idx, s, v: idx < N, lambda idx, s, v: sde_body(idx, s, v, 1), loop_var)
_, u = tf.while_loop(lambda idx, p: idx < mc_rounds_ref, mc_body, (tf.constant(0), tf.zeros((batch_size, 1), dtype)))
u_reference = u / tf.cast(mc_rounds_ref, tf.float32)
kolmogorov_train_and_test(xi, x_sde, phi, u_reference, neurons, lr_boundaries, lr_values, train_steps, mc_rounds, mc_freq, 'example3_2.csv', dtype)