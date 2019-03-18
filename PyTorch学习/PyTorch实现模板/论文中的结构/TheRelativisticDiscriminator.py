'''
论文The Relativistic Discriminator: A Key Element Missing From Standard GAN中的结构实现
'''
import torch as t


class Generator(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(16, 64)
        self.fc2 = t.nn.Linear(64, 256)

    def forward(self, x):
        x = t.nn.ReLU()(self.fc1(x))
        x = t.nn.Tanh()(self.fc2(x))
        return x


class Discriminator(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(256, 64)
        self.fc2 = t.nn.Linear(64, 1)

    def forward(self, x):
        x = t.nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# 第一种GAN写法，用detach()截断梯度求导
class RSGAN(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discirminator = Discriminator()

    def forward(self, x_r, mode):
        '''
        :param x_r:
        :param mode: 'd'为d模式，'g'为g模式
        :return:
        '''
        z = t.randn(x_r.size()[0], 16)  # 产生噪声
        x_f = self.generator(z)  # fake样本
        if mode == 'd':
            x_f = x_f.detach()  # 从x_f处断开，使得梯度无法继续求得
        c_r = self.discirminator(x_r)
        c_f = self.discirminator(x_f)
        # g的损失函数
        loss_g = -(t.log(t.nn.functional.sigmoid(c_r - c_f)))
        loss_g = t.sum(loss_g) / loss_g.size()[0]
        # d的损失函数
        loss_d = -(t.log(t.nn.functional.sigmoid(c_f - c_r)))
        loss_d = t.sum(loss_d) / loss_d.size()[0]
        if mode == 'd':
            return loss_d
        else:
            return loss_g


def main():
    batch_size = 8
    dim = 256
    model = RSGAN()
    LEARNING_RATE = 0.0001
    # 定义优化器
    optimizer_g = t.optim.SGD(model.generator.parameters(), lr=LEARNING_RATE)
    optimizer_d = t.optim.SGD(model.discirminator.parameters(), lr=LEARNING_RATE)
    for i in range(10000):
        x_r = t.abs(t.randn(batch_size, dim))
        # d的训练
        optimizer_d.zero_grad()
        loss_d = model(x_r, 'd')
        loss_d.backward()
        optimizer_d.step()
        # g的训练
        optimizer_g.zero_grad()
        loss_g = model(x_r, 'g')
        loss_g.backward()
        optimizer_g.step()
        print(i, loss_d, loss_g)


# 第二种GAN写法，用retain_graph=True的写法
class RSGANV1(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discirminator = Discriminator()

    def forward(self, x_r):
        z = t.randn(x_r.size()[0], 16)  # 产生噪声
        x_f = self.generator(z)  # fake样本
        c_r = self.discirminator(x_r)
        c_f = self.discirminator(x_f)
        # g的损失函数
        loss_g = -(t.log(t.nn.functional.sigmoid(c_r - c_f)))
        loss_g = t.sum(loss_g) / loss_g.size()[0]
        # d的损失函数
        loss_d = -(t.log(t.nn.functional.sigmoid(c_f - c_r)))
        loss_d = t.sum(loss_d) / loss_d.size()[0]
        return loss_g, loss_d


def mainV1():
    batch_size = 8
    dim = 256
    model = RSGANV1()
    LEARNING_RATE = 0.0001
    # 定义优化器
    optimizer_g = t.optim.SGD(model.generator.parameters(), lr=LEARNING_RATE)
    optimizer_d = t.optim.SGD(model.discirminator.parameters(), lr=LEARNING_RATE)
    for i in range(10000):
        x_r = t.abs(t.randn(batch_size, dim))
        loss_g, loss_d = model(x_r)
        # g的训练
        optimizer_g.zero_grad()
        loss_g.backward(retain_graph=True)  # 保留此步计算的前向计算图不动，以留给后续计算
        optimizer_g.step()
        # d的训练
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        print(i, loss_g, loss_d)


if __name__ == '__main__':
    main()
