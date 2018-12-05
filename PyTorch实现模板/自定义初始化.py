import torch as t


# 应用apply函数初始化
def init_weights(m):
    print(m)
    if type(m) == t.nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)


net = t.nn.Sequential(t.nn.Linear(2, 2, bias=False), t.nn.Linear(2, 2, bias=False))
net.apply(init_weights)
print('----------------')


# 手动参数初始化的办法
class Layer_init_test1(t.nn.Module):
    def __init__(self):
        super().__init__()
        # 取parameter的方式初始化
        self.layer1 = t.nn.Linear(2, 2, bias=False)
        self.layer1.weight = t.nn.Parameter(t.Tensor([[0.2, 0.4], [0.6, 0.5]]))
        # 取data的方式初始化
        self.layer2 = t.nn.Linear(2, 2, bias=False)
        self.layer2.weight.data = t.Tensor([[0.3, 0.4], [0.1, 0.5]])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model = Layer_init_test1()


# 通过共享外界的某个参数实现初始化
class Layer1(t.nn.Module):
    def __init__(self, param_weight):
        super().__init__()
        self.layer = t.nn.Linear(1, 1, bias=False)
        self.layer.weight = param_weight

    def forward(self, x):
        x = self.layer(x)
        return x


class Layer2(t.nn.Module):
    def __init__(self, param_weight):
        super().__init__()
        self.layer = t.nn.Linear(1, 1, bias=False)
        self.layer.weight = param_weight
        self.layer_p = t.nn.Linear(1, 1, bias=False)
        self.layer_p.weight = t.nn.Parameter(t.Tensor([[2.0]]))

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_p(x)
        return x


# 定义同一个共享的参数
param_weight = t.nn.Parameter(t.Tensor([[1.5]]))
layer1 = Layer1(param_weight)
layer2 = Layer2(param_weight)
x = t.Tensor([[1.0]])
y1 = layer1(x)
y2 = layer2(x)
print(y1)
print(y2)
print('----------------')
# 两个层的layer.weight参数共享，修改一个同时也修改另一个
layer1.layer.weight.data = t.Tensor([[1.6]])
y1 = layer1(x)
y2 = layer2(x)
print(y1)
print(y2)
print('----------------')