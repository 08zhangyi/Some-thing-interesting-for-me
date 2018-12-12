import edward as ed
import tensorflow as tf
from edward.models import Categorical, PointMass

# 混合推断算法
qbeta = PointMass(params=tf.Variable(tf.zeros([K, D])))
qz = Categorical(logits=tf.Variable(tf.zeros[N, K]))
# z的推断用一个方法
# beta的推断用另一个办法
inference_e = ed.VariationalInference({z: qz}, data={x: x_data, beta: qbeta})
inference_m = ed.MAP({beta: qbeta}, data={x: x_data, z: qz})
# 不被推断的变量放在第二个字典
for _ in range(10000):
    inference_e.update()
    inference_m.update()

# 信息传递算法
from edward.models import Categorical, Normal
# 数据定义部分
N1 = 1000  # 第一个数据集的大小
N2 = 2000  # 第二个数据集的大小
D = 2  # 数据维数
K = 5  # 聚类数量
# 模型定义
beta = Normal(loc=tf.zeros([K, D]), scale=tf.ones([K, D]))
z1 = Categorical(logits=tf.zeros([N1, K]))
z2 = Categorical(logits=tf.zeros([N2, K]))
x1 = Normal(loc=tf.gather(beta, z1), scale=tf.ones([N1, D]))
x2 = Normal(loc=tf.gather(beta, z2), scale=tf.ones([N2, D]))

# 后验分布
qbeta = Normal(loc=tf.Variable(tf.zeros([K, D])), scale=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
qz1 = Categorical(logits=tf.Variable(tf.zeros[N1, K]))
qz2 = Categorical(logits=tf.Variable(tf.zeros[N2, K]))
# beta一样，z不一样，输入不同的数据集进行推断
# beta相当于全局变量，x1和x2相当于不同服务器的负载
inference_z1 = ed.KLpq({beta: qbeta, z1: qz1}, {x1: x1_train})
inference_z2 = ed.KLpq({beta: qbeta, z2: qz2}, {x2: x2_train})
for _ in range(10000):
    inference_z1.update()
    inference_z2.update()