import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np

w2, w1, w0, z2, z1, z0 = None, None, None, None, None, None


def deep_exponential_family(data_size, feature_size, units, shape):
    global w2, w1, w0, z2, z1, z0  # 向外传递模型先验参数
    # units表示每层维数的大小，从unit[2]开始到unit[0]，最后输出feature_size
    # data_size为batch_size的意思
    w2 = ed.Gamma(0.1, 0.3, sample_shape=[units[2], units[1]], name="w2")  # 前两个位置为参数
    w1 = ed.Gamma(0.1, 0.3, sample_shape=[units[1], units[0]], name="w1")
    w0 = ed.Gamma(0.1, 0.3, sample_shape=[units[0], feature_size], name="w0")
    # z2相当于是原始分布，用z2结合参数传到生成x
    z2 = ed.Gamma(0.1, 0.1, sample_shape=[data_size, units[2]], name="z2")
    z1 = ed.Gamma(shape, shape / tf.matmul(z2, w2), name="z1")  # z1的形状跟着两个参数的形状跑，此处相当于对rate建模，concentration不建模
    z0 = ed.Gamma(shape, shape / tf.matmul(z1, w1), name="z0")
    x = ed.Poisson(tf.matmul(z0, w0), name="x")
    return x


def deep_exponential_family_variational(w2, w1, w0, z2, z1, z0, data_size, feature_size, units):
    num_samples = 10000  # 后验分布的采样个数
    qw2 = tf.nn.softplus(tf.random_normal([units[2], units[1]]))  # 初始化
    qw1 = tf.nn.softplus(tf.random_normal([units[1], units[0]]))
    qw0 = tf.nn.softplus(tf.random_normal([units[0], feature_size]))
    qz2 = tf.nn.softplus(tf.random_normal([data_size, units[2]]))
    qz1 = tf.nn.softplus(tf.random_normal([data_size, units[1]]))
    qz0 = tf.nn.softplus(tf.random_normal([data_size, units[0]]))
    return num_samples, qw2, qw1, qw0, qz2, qz1, qz0


def make_value_setter(**model_kwargs):
    def set_values(f, *args, **kwargs):
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        return ed.interceptable(f)(*args, **kwargs)
    return set_values


DATA_SIZE = 100
FEATURE_SIZE = 41
UNITS = [23, 7, 2]
SHAPE = 0.1
x = deep_exponential_family(DATA_SIZE, FEATURE_SIZE, UNITS, SHAPE)
num_samples, qw2, qw1, qw0, qz2, qz1, qz0 = deep_exponential_family_variational(w2, w1, w0, z2, z1, z0, DATA_SIZE, FEATURE_SIZE, UNITS)
log_joint = ed.make_log_joint_fn(deep_exponential_family) # 对model进行包装，生成其对应的log参数
x_sample = tf.placeholder(tf.float32, shape=[DATA_SIZE, FEATURE_SIZE])
def target_log_prob_fn(w2, w1, w0, z2, z1, z0):  # 用此包装后，可以做到后验分布的概率
    return log_joint(DATA_SIZE, FEATURE_SIZE, UNITS, SHAPE, w2=w2, w1=w1, w0=w0, z2=z2, z1=z1, z0=z0, x=x_sample)
hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob_fn, step_size=0.01, num_leapfrog_steps=5)
BURNING_STEPS = 1000  # MCMC的燃烧次数
states, kernels_results = tfp.mcmc.sample_chain(num_results=num_samples, current_state=[qw2, qw1, qw0, qz2, qz1, qz0], kernel=hmc_kernel, num_burnin_steps=BURNING_STEPS)
# states为w2, w1, w0, z2, z1, z0的采样后验分布
print(states)
print(kernels_results)

X_SAMPLE_VALUES = np.random.poisson(5., size=[DATA_SIZE, FEATURE_SIZE])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    states_4_values = sess.run(states[4], feed_dict={x_sample: X_SAMPLE_VALUES})
    print(np.mean(states_4_values, axis=0))