# 用于开发卷积capsule网络的结构
# 卷积的意义在于W矩阵的权重共享
import tensorflow as tf


u = tf.placeholder(tf.float32, shape=(BATCH_SIZE, u_x, u_y, u_channel, u_capsule_num))