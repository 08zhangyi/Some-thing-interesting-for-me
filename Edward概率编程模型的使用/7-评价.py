import edward as ed
import tensorflow as tf

# 点估计评估
x_post = ed.copy(x, {z: qz})  # qz为z学习到的后验分布，语句的意思为，x对z的依赖替换为x_post对qz的依赖
# y_post为按照参数生成的y的后验分布，y_train为真实的数据
ed.evaluate('categorical_accuracy', data={y_post: y_train, x: x_train})
ed.evaluate('mean_absolute_error', data={y_post: y_train, x: x_train})
# 似然估计
ed.evaluate('log_likelihood', data={x_post: x_train})
# 拆分训练集和验证集的方式
from edward.models import Categorical
qz_test = Categorical(logits=tf.Variable(tf.zeros[N_test, K]))
inference_test = ed.Inference({z: qz_test}, data={x: x_test, beta: qbeta})
inference_test.run()  # 模型训练完成
x_post = ed.copy(x, {z: qz_test, beta: qbeta})
ed.evaluate('log_likelihood', data={x_post: x_valid})  # x_valid为测试表现的数据

# 后验预测检验，PPC
x_post = ed.copy(x, {z: qz})
ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x_post]), data={x_post: x_train})
ed.ppc(lambda xs, zs: tf.maximum(zs[z]), data={y_post: y_train, x_ph: x_train}, latent_vars={z: qz, beta: qbeta})  # 明确指出后验
