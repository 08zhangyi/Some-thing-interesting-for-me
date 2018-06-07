# capsule封装的函数集合
import tensorflow as tf


# 最普通的capsule层定义
class CapsuleFlat:
    def __init__(self, v_capsule_num, v_capsule_dim, ROUTING_NUMBER, mode='normal', name=None):
        # capsule的相关参数
        self.v_capsule_num = v_capsule_num
        self.v_capsule_dim = v_capsule_dim
        self.ROUTING_NUMBER = ROUTING_NUMBER
        self.mode = mode
        self.name = name

    def __call__(self, u):
        # 获取输入的capsule结构
        u_capsule_num = u.get_shape().as_list()[1]
        u_capsule_dim = u.get_shape().as_list()[2]
        # 定义预测用的权重值
        W = self.get_weight([u_capsule_num, u_capsule_dim, self.v_capsule_num, self.v_capsule_dim], self.name)
        # 权重的乘法，计算u的预测值
        u_predict = tf.expand_dims(tf.expand_dims(u, axis=-1), axis=-1) * W
        u_predict = tf.reduce_sum(u_predict, axis=2)
        # 计算权重
        b = tf.constant(0.0, shape=[u_capsule_num, self.v_capsule_num])  # b使用全0初始化
        for i in range(self.ROUTING_NUMBER):
            if self.mode == 'normal':  # mode参数决定不同的capsule分配方式
                c = tf.expand_dims(tf.nn.softmax(b), axis=-1)
            elif self.mode == 'reversed':
                c = tf.expand_dims(tf.nn.softmax(b), axis=0)
            else:
                c = tf.expand_dims(tf.nn.softmax(b), axis=-1)
            s = tf.reduce_sum(u_predict * c, axis=1)
            v = self.squashing(s)
            # 更新权重的公式
            v_expand = tf.expand_dims(v, axis=1)
            a = tf.reduce_sum(v_expand * u_predict, axis=-1)
            b = b + a
        return v, W

    def squashing(self, s):
        # s的结构为{b, x, y]，x为capsule个数，y为capsule大小
        s_norm = tf.norm(s, axis=2)
        s_norm = tf.square(s_norm) / ((1 + tf.square(s_norm)) * s_norm)
        v = s * tf.expand_dims(s_norm, axis=2)
        return v

    def get_weight(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)


if __name__ == '__main__':
    BATCH_SIZE = None
    u_capsule_num = 30  # capsule的数量
    u_capsule_dim = 40  # capsule的大小
    u = tf.placeholder(tf.float32, shape=[BATCH_SIZE, u_capsule_num, u_capsule_dim])  # u为输入的capsule集合

    v_capsule_num = 25
    v_capsule_dim = 35
    ROUTING_NUMBER = 3
    capsule = CapsuleFlat(v_capsule_num, v_capsule_dim, ROUTING_NUMBER, name='lf')
    v, W = capsule(u)