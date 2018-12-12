from edward.models import Normal, Exponential
import tensorflow as tf

# 都是RandomVariable衍生出来
# 一元分布
Normal(loc=tf.constant(0.0), scale=tf.constant(1.0))
Normal(loc=tf.zeros(5), scale=tf.ones(5))
Exponential(rate=tf.ones([2, 3]))

# 多元分布
from edward.models import Dirichlet, MultivariateNormalTriL
K = 3
Dirichlet(concentration=tf.constant([0.1] * K))  # K为Dirichlet分布
MultivariateNormalTriL(loc=tf.zeros([5, K]), scale_tril=tf.ones([5, K, K]))  # loc的最后一位表示维数
MultivariateNormalTriL(loc=tf.zeros([2, 5, K]), scale_tril=tf.ones([2, 5, K, K]))

# 每个RandomVariable有方法log_prob()，mean()，sample()，且与计算图上的一个张量对应
# 可以支持诸多运算
from edward.models import Normal

x = Normal(loc=tf.zeros(10), scale=tf.ones(10))
y = tf.constant(5.0)
x + y, x - y, x * y, x / y
tf.tanh(x * y)
tf.gather(x, 2)
print(x[2])

# 有向图模型
from edward.models import Bernoulli, Beta
theta = Beta(1.0, 1.0)
x = Bernoulli(probs=tf.ones(50)*theta)

# 需要优化的参数
from edward.models import Bernoulli
theta = tf.Variable(0.0)
x = Bernoulli(probs=tf.ones(50) * tf.sigmoid(theta))

# 神经网络，slim版本
from edward.models import Bernoulli, Normal
from tensorflow.contrib import slim
N = 12
d = 2
z = Normal(loc=tf.zeros([N, d]), scale=tf.ones([N, d]))
h = slim.fully_connected(z, 256)
x = Bernoulli(probs=tf.sigmoid(slim.fully_connected(h, 28*28)))
# 神经网络，Keras版本
from edward.models import Bernoulli, Normal
from keras.layers import Dense
N = 12
d = 2
z = Normal(loc=tf.zeros([N, d]), scale=tf.ones([N, d]))
h = Dense(256, activation='relu')(z.value())  # 用到z.value
x = Bernoulli(probs=Dense(28*28)(h))