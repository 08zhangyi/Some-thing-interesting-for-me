import numpy as np
import tensorflow as tf
import time
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.training.moving_averages import assign_moving_average


def neural_net(x, neurons, is_training, name, mv_decay=0.9, dtype=tf.float32):
    pass