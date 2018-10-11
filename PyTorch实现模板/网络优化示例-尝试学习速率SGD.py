import numpy as np
import torch as t
import torch.nn.functional as F
t.manual_seed(100)  # 手动设置种子，一比较结果


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 6, 5)
        self.conv2 = t.nn.Conv2d(6, 16, 5)
        self.fc1 = t.nn.Linear(16*5*5, 120)
        self.fc2 = t.nn.Linear(120, 84)
        self.fc3 = t.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 是定好种子，指定不同的输入A1，A2
np.random.seed(100)
A1 = np.random.randn(1, 1, 32, 32)
A2 = np.random.randn(1, 1, 32, 32)
A = [A1, A2]
B1 = np.random.randn(1, 10)
B2 = np.random.randn(1, 10)
B = [B1, B2]

# 设定网络实例化
net = Net()  # 不断的调用net，就可以使用固定的模型并对其优化
print(net)
# 设定损失函数和优化器
LEARNING_RATE = 0.02
criterion = t.nn.MSELoss()
optimizer = t.optim.SGD(net.parameters(), lr=LEARNING_RATE)

TRAIN_STEP = 100  # 训练的步数
for i in range(TRAIN_STEP):
    print('train step: '+str(i+1))
    # 准备输入输出数据
    input = t.Tensor(A[i % 2])  # 输入数据Tensor化
    output = net(input)
    target = t.Tensor(B[i % 2])  # 目标数据Tensor化
    # 设定损失函数
    loss = criterion(output, target)
    # 设定优化器
    # 更新net中的参数
    optimizer.zero_grad()  # 每步训练前，对net中参数的梯度归零，以重新计算梯度
    # optimizer.zero_grad()与net.zero_grad()一致，都是让net中的参数梯度归零
    loss.backward()
    optimizer.step()
    print('loss is: ', loss.detach().numpy())
