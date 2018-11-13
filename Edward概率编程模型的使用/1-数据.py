import numpy as np
import tensorflow as tf

# 用numpy数组或者tf的Tensor
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
x_data = tf.concat([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
# 内部数据用tf.Variable存储

# 中途添加数据，用tf.placeholder
x_data = tf.placeholder(tf.float32, [100, 25])
# inference.update()中需要给出feed_dict函数

# 从文件中提取数据
filename_queue = tf.train.string_input_producer()
reader = tf.WholeFileReader()