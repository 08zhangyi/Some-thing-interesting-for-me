# 50行实现一个ResNet
import torch as t


class ResidualBlock(t.nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = t.nn.Sequential(t.nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                                    t.nn.BatchNorm2d(outchannel),
                                    t.nn.ReLU(inplace=True),
                                    t.nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                                    t.nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return t.nn.functional.relu(out)


class ResNet(t.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.pre = t.nn.Sequential(t.nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                   t.nn.BatchNorm2d(64),
                                   t.nn.ReLU(inplace=True),
                                   t.nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.fc = t.nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = t.nn.Sequential(t.nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                                   t.nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return t.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = t.nn.functional.avg_pool2d(x,7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    model = ResNet()
    input = t.randn(1, 3, 224, 224)
    output = model(input)