import torch as t
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

t.manual_seed(1)
np.random.seed(1)

# 训练数据的准备工作
DATA_SIZE = 2000
BATCH_SIZE = 64

x = np.linspace(-7, 10, DATA_SIZE)[:, np.newaxis]  # x的形状为batch_size * feature_num
y = np.square(x) - 5 + np.random.normal(0, 2, x.shape)
x, y = t.from_numpy(x).float(), t.from_numpy(y).float()

test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
test_y = np.square(test_x) - 5 + np.random.normal(0, 2, test_x.shape)
test_x = t.from_numpy(test_x).float()
test_y = t.from_numpy(test_y).float()

train_dataset = Data.TensorDataset(x, y)  # 直接注入数据
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 对数据进行画图展示
# plt.scatter(x.numpy(), y.numpy(), s=30, label='train')
# plt.legend(loc='upper left')
# plt.show()

# Net的一些参数
N_HIDDEN = 8
B_INIT = -0.2
ACTIVATION_FUNC = F.relu


class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        if self.do_bn:
            self.bn_input = nn.BatchNorm1d(1, momentum=0.5)

        for i in range(N_HIDDEN):
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % i, fc)  # 设置，这样Module才能自动识别Net中用到的Module，归根结底，必须是Net的对象才能被Module识别到
            self._set_init(fc)  # 初始化fc中参数
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)
        self._set_init(self.predict)

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, B_INIT)

    def forward(self, x):
        if self.do_bn:
            x = self.bn_input(x)
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            if self.do_bn:
                x = self.bns[i](x)
            x = ACTIVATION_FUNC(x)
            layer_input.append(x)
        out = self.predict(x)
        return out, layer_input


nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
# print(nets)

LR = 0.03
EPOCH = 12
optimizers = [t.optim.Adam(net.parameters(), lr=LR) for net in nets]
loss_func = t.nn.MSELoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):  # 从DataLoader中获取数据
        batch_x, batch_y = batch_x, batch_y
        for net, opt in zip(nets, optimizers):
            predict, _ = net(batch_x)
            loss = loss_func(predict, batch_y)
            # 训练步骤
            opt.zero_grad()
            loss.backward()
            opt.step()

[net.eval() for net in nets]    # 将net设置为eval模式，这样BN中的参数不会进一步的更新
# net.train()  # eval的反函数为train
predictions = [net(test_x)[0] for net in nets]
plt.figure(3)
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='b', label='test')
plt.plot(test_x.data.numpy(), predictions[0].data.numpy(), lw='5', c='r', label='not-BN')
plt.plot(test_x.data.numpy(), predictions[1].data.numpy(), lw='5', c='g', label='BN')
plt.legend(loc='best')
plt.show()