# import tensorflow as tf
#
# batch = None
# in_depth = 8
# in_height = 6
# in_width = 6
# in_channels = 32
# x = tf.placeholder(tf.float32, [batch, in_depth, in_height, in_width, in_channels])
#
# out_channels = 64
# filter_depth = 8
# filter_height = 3
# filter_width = 3
# filter = tf.Variable(tf.truncated_normal(([filter_depth, filter_height, filter_width, in_channels, out_channels])))
#
# y = tf.nn.conv3d(x, filter, strides=(1, 8, 1, 1, 1), padding='VALID')
# print(y.get_shape())

list = []
for i in range(4):
    list_temp = []
    for j in range(5):
        list_temp.append(j)
    list.append(list_temp)
print(list)