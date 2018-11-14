from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution


# 自定义RandomVariable的模板，双继承
class CustomRandomVariable(RandomVariable, Distribution):
    def __init__(self, *args, **kwargs):
        super(CustomRandomVariable, self).__init__(*args, **kwargs)

        def _log_prob(self, value):
            raise NotImplementedError("log_prob is not implemented")

        def _sample_n(self, n, seed=None):
            # shape为(n,)的Tensor
            raise NotImplementedError("sample_n is not implemented")


import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Poisson
from scipy.stats import poisson


def _sample_n(self, n=1, seed=None):
    def np_sample(rate, n):
        return poisson.rvs(mu=rate, size=n, random_state=seed).astype(np.float32)
    val = tf.py_func(np_sample, [self.rate, n], [tf.float32])[0]
    batch_event_shape = self.batch_shape.concatenate(self.event_shape)
    shape = tf.concat([tf.expand_dims(n, 0), tf.convert_to_tensor(batch_event_shape)], 0)
    val = tf.reshape(val, shape)
    return val


Poisson._sample_n = _sample_n
sess = ed.get_session()
x = Poisson(rate=1.0)
sess.run(x)
sess.run(x)