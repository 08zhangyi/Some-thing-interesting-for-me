import edward as ed
import tensorflow as tf

# 第一个字典为先验：后验的对应，第二个字典为观测值张量：观测值data对应
# qz与qbeta的后验分布中有可学习的参数
inference = ed.Inference({z: qz, beta: qbeta}, {x: x_train})
# 进行参数学习
inference.run()

# 更加精细的学习调节
inference = ed.Inference({z: qz, beta: qbeta}, {x: x_train})
inference.initialize()
tf.global_variables_initializer().run()
for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
inference.finalize()

from edward.models import Normal

theta = tf.Variable(0.0)
x = Normal(loc=tf.ones(10)*theta, scale=1.0)
inference = ed.Inference({}, {x: x_train})

# 条件推断，放在后面字典的后验不进行学习
inference = ed.Inference({beta: qbeta}, {x: x_train, z: qz})
# 隐含先验分布，也就是后验分布就用先验分布，不出现两个字典中
inference = ed.Inference({beta: qbeta}, {x: x_train})

# 基本模型
# p(x, z, beta) = Normal(x | beta, I) Categorical(z | pi) Normal(beta | 0, I)

# 变分推断
# beta的后验用一组参数正态逼近，z的后验用参数Categorical逼近
from edward.models import Categorical, Normal
qbeta = Normal(loc=tf.Variable(tf.zeros([K, D])), scale=tf.exp(tf.Variable(tf.zeros[K, D])))  # 定义后验分布
qz = Categorical(logits=tf.Variable(tf.zeros[N, K]))
inference = ed.VariationalInference({beta: qbeta, z: qz}, data={x: x_train})
# 用MAP方法做推断，推断方法都是继承VariationalInference
from edward.models import PointMass
qbeta = PointMass(params=tf.Variable(tf.zeros([K, D])))
qz = PointMass(params=tf.Variable(tf.zeros(N)))
inference = ed.MAP({beta: qbeta, z: qz}, data={x: x_train})

# MonteCarlo推断
# 用beta和z的采样分布，作为后验的逼近
from edward.models import Empirical
T = 10000  # number of samples
qbeta = Empirical(params=tf.Variable(tf.zeros([T, K, D])))  # Empirical为经验分布
qz = Empirical(params=tf.Variable(tf.zeros([T, N])))
inference = ed.MonteCarlo({beta: qbeta, z: qz}, data={x: x_train})

# 精确推断
from edward.models import Bernoulli, Beta
pi = Beta(1.0, 1.0)
x = Bernoulli(probs=pi, sample_shape=10)
pi_cond = ed.complete_conditional(pi)  # 计算pi的后验分布的精确表达式
sess.run(pi_cond, {x: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])})
