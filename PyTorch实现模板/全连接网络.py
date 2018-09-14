import torch as t
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


# 全连接网络
class FullConnectedNetV1(t.nn.Module):
    def __init__(self):  # 参数在此处定义
        super().__init__()  # 必须先执行父类的初始化
        self.fc1 = t.nn.Linear(256, 128)
        self.fc2 = t.nn.Linear(128, 16)
        self.fc3 = t.nn.Linear(16, 3)

    def forward(self, x):  # 激活函数在此处定义
        # x的形状应为(batch_size, data_size)
        x = t.nn.ReLU()(self.fc1(x))
        x = t.nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x


# 全连接网络，自定义初始化
class FullConnectedNetV2(t.nn.Module):
    def __init__(self):  # 参数在此处定义
        super().__init__()  # 必须先执行父类的初始化
        self.fc1 = t.nn.Linear(256, 128)
        # 调用init模块赋值
        init.xavier_normal_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0.0)
        # 赋值给指定值
        self.fc1.bias.data = t.tensor(np.ones((128, )), dtype=t.float32)  # dtype需要匹配
        # self.fc1.bias.data = t.tensor(np.zeros((128, )), dtype=t.float64)
        self.fc2 = t.nn.Linear(128, 16)  # 不初始化，直接使用内容中的随机数
        self.fc3 = t.nn.Linear(16, 3)

    def forward(self, x):  # 激活函数在此处定义
        # x的形状应为(batch_size, data_size)
        x = t.nn.ReLU()(self.fc1(x))
        x = t.nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x


# 全连接网络，BatchNorm层
class FullConnectedNetV3(t.nn.Module):
    def __init__(self):  # 参数在此处定义
        super().__init__()  # 必须先执行父类的初始化
        self.fc1 = t.nn.Linear(256, 128)
        self.bn1 = t.nn.BatchNorm1d(128)
        self.fc2 = t.nn.Linear(128, 16)
        self.fc3 = t.nn.Linear(16, 3)

    def forward(self, x):  # 激活函数在此处定义
        # x的形状应为(batch_size, data_size)
        x = self.fc1(x)
        x = self.bn1(x)
        x = t.nn.ReLU()(x)
        x = t.nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x


# 全连接网络，用ModuleList定义
class FullConnectedNetV4(t.nn.Module):
    def __init__(self):  # 参数在此处定义
        super().__init__()  # 必须先执行父类的初始化
        self.linears = t.nn.ModuleList([t.nn.Linear(256, 128), t.nn.Linear(128, 16), t.nn.Linear(16, 3)])

    def forward(self, x):  # 激活函数在此处定义
        # x的形状应为(batch_size, data_size)
        x = t.nn.ReLU()(self.linears[0](x))
        x = t.nn.Sigmoid()(self.linears[1](x))
        x = self.linears[2](x)
        return x


# 全连接网络，用ModuleDict定义，比较新的PyTorch才有的功能
class FullConnectedNetV5(t.nn.Module):
    def __init__(self):  # 参数在此处定义
        super().__init__()  # 必须先执行父类的初始化
        self.linears = t.nn.ModuleDict({'fc1': t.nn.Linear(256, 128), 'fc2': t.nn.Linear(128, 16), 'fc3': t.nn.Linear(16, 3)})
        self.activations = t.nn.ModuleDict({'ReLU': t.nn.ReLU(), 'Sigmoid': t.nn.Sigmoid()})

    def forward(self, x):  # 激活函数在此处定义
        # x的形状应为(batch_size, data_size)
        x = self.activations['ReLU'](self.linears['fc1'](x))
        x = self.activations['Sigmoid'](self.linears['fc2'](x))
        x = self.linears['fc3'](x)
        return x


# ParameterList，ParameterDict待开发
pass


# 获取Module中信息的函数
def print_module_information(net):
    for parameter in net.parameters():
        # Module对象中的Parameter全部打印出来，可以打印Sub-Module中的Parameter
        print(parameter)
        x = parameter.data  # Parameter中的张量，可以手动修改
        print(x)
    for name, parameter in net.named_parameters():
        print(name)  # name为__init__()中定义的变量
net = FullConnectedNetV2()
print_module_information(net)


# 用Sequential方式定义Module
def get_sequential_model():
    # 默认命名方法
    model = t.nn.Sequential(t.nn.Linear(256, 128), t.nn.ReLU(), t.nn.Linear(128, 16), t.nn.ReLU(), t.nn.Linear(16, 3))
    x = t.randn(18, 256)
    x = model(x)
    print(x.size())
    print(model)  # 默认用序号命名
    import collections
    model = t.nn.Sequential(collections.OrderedDict([('fc1', t.nn.Linear(256, 128)), ('ReLU1', t.nn.ReLU()), ('fc2', t.nn.Linear(128, 16)), ('ReLU2', t.nn.ReLU()), ('fc3', t.nn.Linear(16, 3))]))
    print(model)  # 用OrderedDict手动命名
# get_sequential_model()


if __name__ == '__main__':
    net = FullConnectedNetV4()
    input = t.randn(18, 256)
    out = net(input)
    print(out.size())
