import torch as t
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


# 全连接网络
class FullConnectedNet(t.nn.Module):
    def __init__(self):
        super().__init__()  # 必须先执行父类的初始化
        self.fc1 = t.nn.Linear(256, 128)
        self.fc2 = t.nn.Linear(128, 16)
        self.fc3 = t.nn.Linear(16, 3)

    def forward(self, x):
        # x的形状应为(batch_size, channels, height, width)
        x = t.nn.ReLU()(self.fc1(x))
        x = t.nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = FullConnectedNet()
    input = t.randn(18, 256)
    out = net(input)
    print(out.size())
