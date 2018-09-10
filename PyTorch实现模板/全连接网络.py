import torch as t
import numpy as np


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()  # 必须先执行父类的初始化
        self.conv1 = t.nn.Conv2d(1, 6, 5, stride=1, padding=1, dilation=1)  # in_channels，out_channels，k_size

    def forward(self, x):
        # x的形状应为(batch_size, channels, height, width)
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    net = Net()
    input = t.randn(1, 1, 32, 32)
    out = net(input)
    print(out.size())
