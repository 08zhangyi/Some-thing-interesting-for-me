'''
论文Dynamic Routing Between Capsules的层结构的实现
论文Matrix Capsules With EM Routing的层结构的实现
'''
import torch as t


class DunamicRoutingCapsule(t.nn.Module):
    def __init__(self, ci_num, ci_dim, co_num, co_dim, routing_times=3):
        # ci表示输入的capsule参数，co表示输出的capsule参数
        # routing_times表示dynamic routing的次数
        super().__init__()
        self.W = t.nn.Parameter(t.randn(1, 1, co_num, co_dim, ci_dim))
        self.routing_times = routing_times

    def forward(self, u):
        # u是输入层的capsule
        # u的形状是batch_size * capsule_num * capsule_dim
        u = u.unsqueeze(2).unsqueeze(3)
        uij = t.sum(u * self.W, dim=4)  # batch_size * ci_num * co_num * co_dim
        c = self._dynamic_routing(uij)
        s = t.sum(uij * c, dim=1)
        v = self._squash(s)
        return v

    def _squash(self, s):
        # s的形状为batch_size * capsule_num * capsule_dim
        # 对s进行squash操作
        s_norm_square = t.norm(s, p=2, dim=2) * t.norm(s, p=2, dim=2)
        s1 = (s_norm_square / ((1.0 + s_norm_square) * t.norm(s, p=2, dim=2))).unsqueeze(2)
        v = s1 * s
        return v

    def _dynamic_routing(self, uij):
        b = t.zeros(8, 3, 5, 1)
        for i in range(self.routing_times):
            c = t.nn.functional.softmax(b, dim=2)
            s = t.sum(uij * c, dim=1)
            v = self._squash(s).unsqueeze(1)
            uij_temp = t.sum(uij * v, dim=3).unsqueeze(3)
            b = b + uij_temp
        return c


class EMRoutingCapsule(t.nn.Module):
    def __init__(self, ci_num, ci_dim, co_num, co_dim, routing_times=3):
        # ci表示输入的capsule参数，co表示输出的capsule参数
        # routing_times表示EM routing的次数
        super().__init__()
        self.W = t.nn.Parameter(t.randn(1, 1, co_num, co_dim, ci_dim))
        self.beta_u = t.nn.Parameter(t.randn(1))
        self.beta_a = t.nn.Parameter(t.randn(1))
        self.routing_times = routing_times

    def forward(self, u, a, lambda_value):
        # u，a是输入层的capsule
        # u的形状是batch_size * capsule_num * capsule_dim
        # a的形状是batch_size * capsule_num
        # lambda_value是一个数值，逆温度参数
        u = u.unsqueeze(2).unsqueeze(3)
        uij = t.sum(u * self.W, dim=4)  # batch_size * ci_num * co_num * co_dim
        self._EM_routing(uij, a)

    def _EM_routing(self, uij, a):
        R = t.ones(uij.size()[0], uij.size()[1], uij.size()[2]) / uij.size()[2]
        for i in range(self.routing_times):
            # M步
            a_temp = a.unsqueeze(2)
            R = R * a_temp
            miu = t.sum(R.unsqueeze(3) * uij, dim=1) / t.sum(R, dim=1).unsqueeze(2)
            miu_temp = miu.unsqueeze(1)
            sigma = t.sum(R.unsqueeze(3) * (uij - miu_temp)**2, dim=1) / t.sum(R, dim=1).unsqueeze(2)
            cost = (self.beta_u + t.log(sigma)) * (t.sum(R, dim=1).unsqueeze(2))
            a_o = lambda_value * (self.beta_a - t.sum(cost, dim=2))
            a_o = self._logistics(a_o)
            # E步
            p = t.randn(8, 3, 5)
            a_o_temp = a_o.unsqueeze(1)
            R = a_o_temp * p
            R_sum =  t.sum(R, dim=2).unsqueeze(2)
            R = R / R_sum

    def _logistics(self, x):
        x = 1 / (1 + t.exp(-x))
        return x


if __name__ == '__main__':
    batch_size = 8
    ci_num = 3
    ci_dim = 4
    co_num = 5
    co_dim = 6
    lambda_value = 5
    model = EMRoutingCapsule(ci_num, ci_dim, co_num, co_dim)
    u = t.randn(batch_size, ci_num, ci_dim)
    a = t.nn.functional.softmax(t.randn(batch_size, ci_num), dim=1)
    model(u, a, lambda_value)
