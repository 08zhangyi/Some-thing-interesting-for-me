import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np


def deep_exponential_family(data_size, feature_size, units, shape):
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
    return x, w2, w1, w0, z2, z1, z0


def deep_exponential_family_variational(w2, w1, w0, z2, z1, z0):
    # 用于生成指定分布的包装
    def trainable_positive_pointmass(shape, name=None):
        with tf.variable_scope(None, default_name="trainable_positive_pointmass"):
            # 没有PointMass分布，用Dirichlet分布代替
            return ed.Dirichlet(tf.nn.softplus(tf.get_variable("mean", shape)), name=name)
    def trainable_gamma(shape, name=None):
        with tf.variable_scope(None, default_name="trainable_gamma"):
            return ed.Gamma(tf.nn.softplus(tf.get_variable("shape", shape)),
                            1.0 / tf.nn.softplus(tf.get_variable("scale", shape)), name=name)
    # 生成参数后验分布的函数
    # qw2 = trainable_positive_pointmass(w2.shape, name="qw2")
    # qw1 = trainable_positive_pointmass(w1.shape, name="qw1")
    # qw0 = trainable_positive_pointmass(w0.shape, name="qw0")
    qw2 = trainable_gamma(w2.shape, name="qw2")
    qw1 = trainable_gamma(w1.shape, name="qw1")
    qw0 = trainable_gamma(w0.shape, name="qw0")
    qz2 = trainable_gamma(z2.shape, name="qz2")
    qz1 = trainable_gamma(z1.shape, name="qz1")
    qz0 = trainable_gamma(z0.shape, name="qz0")
    return qw2, qw1, qw0, qz2, qz1, qz0


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
x, w2, w1, w0, z2, z1, z0 = deep_exponential_family(DATA_SIZE, FEATURE_SIZE, UNITS, SHAPE)
qw2, qw1, qw0, qz2, qz1, qz0 = deep_exponential_family_variational(w2, w1, w0, z2, z1, z0)

# x_sample = np.random.poisson(5., size=[DATA_SIZE, FEATURE_SIZE])  # 生成虚拟的训练数据，size与模型匹配
x_sample = tf.placeholder(tf.float32, shape=[DATA_SIZE, FEATURE_SIZE])  # 可以用placeholder占位符
with ed.tape() as model_tape:
    with ed.interception(make_value_setter(w2=qw2, w1=qw1, w0=qw0, z2=qz2, z1=qz1, z0=qz0)):
        # 对分布的参数用后验分布进行替换，生成后验分布
        posterior_predictive, _, _, _, _, _, _ = deep_exponential_family(DATA_SIZE, FEATURE_SIZE, UNITS, SHAPE)
log_likelihood = posterior_predictive.distribution.log_prob(x_sample)
print(log_likelihood)  # log_likelihood为根据x_sample计算的对数似然函数

# 损失函数的定义，用变分法
kl = 0.
for rv_name, variational_rv in [("z0", qz0), ("z1", qz1), ("z2", qz2), ("w0", qw0), ("w1", qw1), ("w2", qw2)]:
    # rv_name代表先验分布的name
    # variational_rv代表后验分布的名字
    kl += tf.reduce_sum(variational_rv.distribution.kl_divergence(model_tape[rv_name].distribution))  # q分布与p分布计算KL散度，q为后验，p为先验
elbo = tf.reduce_mean(log_likelihood - kl)  # 后验似然变分下界估计，其负值损失函数
tf.summary.scalar("elbo", elbo)
optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.minimize(-elbo)

# 训练
X_SAMPLE_VALUES = np.random.poisson(5., size=[DATA_SIZE, FEATURE_SIZE])
TRAINING_STEPS = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 去暖用代码
    while True:
        _, elbo_values = sess.run([train_op, elbo], feed_dict={x_sample: X_SAMPLE_VALUES})
    for i in range(TRAINING_STEPS):
        _, elbo_values = sess.run([train_op, elbo], feed_dict={x_sample: X_SAMPLE_VALUES})
        print('第' + str(i+1) + '轮训练的似然变分下界估计为%.6f' % elbo_values)

# 评估训练结果，还在Session内
    def tfidf(x_sample):  # 使用tfifd函数评估结果
        num_documents = x_sample.shape[0]
        idf = tf.log(tf.cast(num_documents, tf.float32)) - tf.log(tf.cast(tf.count_nonzero(x_sample, axis=0), tf.float32))
        return x_sample * idf
    log_likelihood = tf.reduce_mean(posterior_predictive.distribution.log_prob(x_sample))
    observed_statistic = sess.run(tfidf(x_sample), feed_dict={x_sample: X_SAMPLE_VALUES})
    replicated_statistic = tfidf(posterior_predictive)
    replicated_statistics = np.array([sess.run(replicated_statistic) for _ in range(100)])
print(np.mean(observed_statistic))
print(np.mean(replicated_statistics))