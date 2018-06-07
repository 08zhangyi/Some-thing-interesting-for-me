# mnist的capsule网络，有待继续完成
import tensorflow as tf


# 权重变量获取工具
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


# 输入部分
BATCH_SIZE = None
channel_num_input = 1
image_size = 28
image_input = tf.placeholder(tf.float32, [BATCH_SIZE, image_size, image_size, channel_num_input])

# 2onv1
stride_conv1 = 1
kernel_size_conv1 = 9
channel_num_conv1 = 256
W_conv1 = weight_variable([kernel_size_conv1, kernel_size_conv1, channel_num_input, channel_num_conv1])
b_conv1 = bias_variable([channel_num_conv1])
h_conv1 = tf.nn.relu(conv2d(image_input, W_conv1, stride_conv1, 'VALID') + b_conv1)

# conv2
stride_conv2 = 2
kernel_size_conv2 = 9
channel_num_conv2 = 256
W_conv2 = weight_variable([kernel_size_conv2, kernel_size_conv2, channel_num_conv1, channel_num_conv2])
b_conv2 = bias_variable([channel_num_conv2])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, stride_conv2, 'VALID') + b_conv2)

# primary capsule层整理
# 这么做是不对的，需要reshape
CAPSULE_SIZE = 8
channel_num_primy_capsule = channel_num_conv2 // 8
capsule_layer = [h_conv2[:, :, :, (CAPSULE_SIZE*i):(CAPSULE_SIZE*i+CAPSULE_SIZE)] for i in range(channel_num_primy_capsule)]