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


# 获取Module中信息的函数
net = FullConnectedNetV2()
for parameter in net.parameters():
    # Module对象中的parameter全部打印出来，可以打印Sub-Module中的parameter
    print(parameter)
for name, parameter in net.named_parameters():
    print(name)  # name为__init__()中定义的变量

if __name__ == '__main__':
    net = FullConnectedNetV2()
    input = t.randn(18, 256)
    out = net(input)
    print(out.size())
