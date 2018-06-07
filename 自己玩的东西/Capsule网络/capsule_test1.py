# mnist的capsule网络，有待继续完成
import tensorflow as tf
from capsule_functions import CapsuleFlat, MarginLoss


# 权重变量获取工具
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def fc(x, output_size, acf_function=tf.nn.relu):
    input_szie = x.get_shape().as_list()[-1]
    W = weight_variable([input_szie, output_size])
    b = bias_variable([output_size])
    y = acf_function(tf.matmul(x, W) + b)
    return y


# 输入部分
BATCH_SIZE = None
channel_num_input = 1
image_size = 28
image_input = tf.placeholder(tf.float32, [BATCH_SIZE, image_size, image_size, channel_num_input])

# conv1
stride_conv1 = 1
kernel_size_conv1 = 9
channel_num_conv1 = 256
W_conv1 = weight_variable([kernel_size_conv1, kernel_size_conv1, channel_num_input, channel_num_conv1])
b_conv1 = bias_variable([channel_num_conv1])
h_conv1 = tf.nn.relu(conv2d(image_input, W_conv1, stride_conv1, 'VALID') + b_conv1)

# conv2，文中虽然说是capsule间的卷积，但实际上和之间用卷积再reshape是一样的
stride_conv2 = 2
kernel_size_conv2 = 9
channel_num_conv2 = 256
W_conv2 = weight_variable([kernel_size_conv2, kernel_size_conv2, channel_num_conv1, channel_num_conv2])
b_conv2 = bias_variable([channel_num_conv2])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, stride_conv2, 'VALID') + b_conv2)

# 构造primary capsule
h_conv2_shape = h_conv2.get_shape().as_list()
h_conv2 = tf.reshape(h_conv2, shape=(-1, 6, 6, 32, 8))  # 根据文章的意思做出第一步的primary capsule，指标是最后的先变动，256变为32*8
primary_capsule = tf.reshape(h_conv2, shape=(-1, 6*6*32, 8))  # 展平primary capsule

# 随时函数的构造
digit_capsule, _ = CapsuleFlat(10, 16, 3)(primary_capsule)
y_onehot = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10])
loss = MarginLoss()(digit_capsule, y_onehot)

# 正则化损失函数的构造
# 先用masking取出相关的capsule
y_masking = tf.expand_dims(y_onehot, axis=-1)
capsule_masking = tf.reduce_sum(digit_capsule * y_masking, axis=1)

# fc1
fc1 = fc(capsule_masking, 512)

# fc2
fc2 = fc(fc1, 1024)

# fc3
fc3 = fc(fc2, 784, tf.nn.sigmoid)

# 正则化损失函数
loss_reg = tf.reduce_mean(tf.reduce_sum(tf.square(fc3 - tf.reshape(image_input, shape=(-1, 784))), axis=-1))
