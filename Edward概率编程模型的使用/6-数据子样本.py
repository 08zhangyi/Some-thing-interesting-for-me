# N个大数据中抽取M个作为子样本
# 模型p(x, z, beta) = p(beta) PI(0-N)p(zn | beta) p(xn | zn, beta)
# 子计算图p(x, z, beta) = p(beta) PI(0-M)p(zm | beta) p(xm | zm, beta)
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Normal
N = 10000000  # 数据集大小
M = 128  # 采样大小
D = 2  # 数据维数
K = 5  # 分类维数
# 模型，都用M作为参数
beta = Normal(loc=tf.zeros([K, D]), scale=tf.ones([K, D]))
z = Categorical(logits=tf.zeros([M, K]))
x = Normal(loc=tf.gather(beta, z), scale=tf.ones([M, D]))
# 后验
qbeta = Normal(loc=tf.Variable(tf.zeros([K, D])), scale=tf.nn.softplus(tf.Variable(tf.zeros[K, D])))
qz_variables = tf.Variable(tf.zeros([M, K]))
qz = Categorical(logits=qz_variables)

x_ph = tf.placeholder(tf.float32, [M])  # 用x_ph传递数据
inference_global = ed.KLqp({beta: qbeta}, data={x: x_ph, z: qz})
inference_local = ed.KLqp({z: qz}, data={x: x_ph, beta: qbeta})
# 用尺度放缩的办法将学习数据调整，此处用到N
inference_global.initialize(scale={x: float(N) / M, z: float(N) / M})
inference_local.initialize(scale={x: float(N) / M, z: float(N) / M})
qz_init = tf.initialize_variables([qz_variables])
for _ in range(1000):
    x_batch = next_batch(size=M)  # 数据提取函数，x数据
    for _ in range(10):
        inference_local.update(feed_dict={x_ph: x_batch})
    inference_global.update(feed_dict={x_ph: x_batch})
    qz_init.run()  # qz由于是采样，每次需要重新初始化

# 内存够用时，qz可以全放入内存中
qz_variables = tf.Variable(tf.zeros([N, K]))
idx_ph = tf.placeholder(tf.int32, [M])
qz = Categorical(logits=tf.gather(qz_variables, idx_ph))

# 推断模型结构
# 从Inference开始衍生，两族：VaruationalInference和MonteCarlo