# 用于开发卷积capsule网络的结构
# 卷积的意义在于W矩阵的权重共享，卷积方法都是valid
import tensorflow as tf


def squashing(s):
    # s的结构为{b, x, y, channel, capsule]，x为capsule个数，y为capsule大小
    s_norm = tf.norm(s, axis=-1)
    s_norm = tf.square(s_norm) / ((1 + tf.square(s_norm)) * s_norm)
    v = s * tf.expand_dims(s_norm, axis=-1)
    return v


def get_weight(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


BATCH_SIZE = None
# 输入的参数设置
u_x, u_y = 6, 6
u_channel_num = 32
u_capsule_num = 8
u = tf.placeholder(tf.float32, shape=(BATCH_SIZE, u_x, u_y, u_channel_num, u_capsule_num))

# 卷积参数
stride_x = 1  # 暂时都是默认为补偿为1，stride不起作用
stride_y = 1
size_x = 3
size_y = 3

# 根据卷积参数计算的输出的大小
v_x, v_y = 4, 4  # 暂时是手动，以后需要自动化计算出来，VALID模式
v_channel_num = 16
v_capsule_num = 9

u_predict = []
# 创造对应的卷积变换权重
W = get_weight(shape=(size_x, size_y, u_channel_num, v_channel_num, u_capsule_num, v_capsule_num))
# 先用笨办法实现，有待用更好的concat方法实现的更加紧凑
for v_channel_num_i in range(v_channel_num):
    W_temp = W[:, :, :, v_channel_num_i, :, :]
    u_predict_y = []
    for v_y_i in range(v_y):
        u_predict_x = []
        for v_x_i in range(v_x):
            u_temp = u[:, v_x_i:v_x_i+size_x, v_y_i:v_y_i+size_y, :, :]
            u_temp = tf.expand_dims(u_temp, axis=5)
            u_predict_temp = tf.reduce_sum(u_temp * W_temp, axis=4)
            u_predict_temp = tf.expand_dims(u_predict_temp, axis=-1)
            u_predict_x.append(u_predict_temp)
        u_predict_x = tf.concat(u_predict_x, axis=-1)
        u_predict_x = tf.expand_dims(u_predict_x, axis=-1)
        u_predict_y.append(u_predict_x)
    u_predict_y = tf.concat(u_predict_y, axis=-1)
    u_predict_y = tf.expand_dims(u_predict_y, axis=-1)
    u_predict.append(u_predict_y)
u_predict = tf.concat(u_predict, axis=-1)
print(u_predict.get_shape())

# routing参数计算
b = tf.constant(0.0, shape=(size_x, size_y, u_channel_num, v_x, v_y, v_channel_num))
b = tf.exp(b)
c = b / (tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.reduce_sum(b, axis=(3, 4, 5)), axis=3), axis=4), axis=5))
c = tf.expand_dims(c, axis=3)
s = tf.reduce_sum(u_predict * c, axis=(1, 2, 3))
s = tf.transpose(s, perm=(0, 2, 3, 4, 1))
v = squashing(s)
# routing值的更新
v_temp = tf.transpose(v, perm=(0, 4, 1, 2, 3))
print(u_predict.get_shape(), v_temp.get_shape())