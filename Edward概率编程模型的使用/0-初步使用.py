import numpy as np

# 数据准备
x_train = np.linspace(-3, 3, num=50)
y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
x_train = x_train.astype(np.float32).reshape((50, 1))
y_train = y_train.astype(np.float32).reshape((50, 1))

# 定义神经网络
import tensorflow as tf
from edward.models import Normal

# 给定网络的参数，参数的先验分布
W_0 = Normal(loc=tf.zeros([1, 2]), scale=tf.ones([1, 2]))  # 生成具体参数的办法，随机采样
W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

x = x_train
y = Normal(tf.matmul(tf.tanh(tf.matmul(x, W_0) + b_0), W_1) + b_1, 0.1)  # mu用神经网络计算
# y为指定参数下的观测值

# 对参数的后验分布进行正态分布的逼近，正态分布的均值方差为可变的参数
# 用tf.Variable包装起来的，表示学习时可以不断的更新的张量
qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [1, 2]), scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [1, 2])))
qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [2, 1]), scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [2, 1])))
qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [2]), scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))

import edward as ed

inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_train})  # 计算两个分布的KL散度，先验和后验
inference.run(n_iter=2)