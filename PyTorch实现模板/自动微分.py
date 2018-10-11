import torch as t

# torch的新写法，无Variable（以废弃）
# 默认的requires_grad=False，x的requires_grad设置为True，才能对x求导
x = t.ones(2, 2)
x.requires_grad_(True)  # 手动设置Tensor x需要计算梯度
x = t.ones(2, 2, requires_grad=True)  # 直接用内置方法设置
x = t.tensor(t.ones(2, 2), requires_grad=True)  # 用t.tensor也可
x = t.Tensor([[1, 1], [1, 1]])
x.requires_grad_(True)  # 用Tensor必须手动设定requires_grad_
# tensor与Tensor的区别：tensor为一个方法，Tensor为一个类
y = x.sum()
y.backward()  # 从y开始向回传递梯度
print(x.grad)  # 观察y关于各个x变量的梯度值
y.backward()  # 再次从y开始向回传递梯度，与之前的梯度值累加
print(x.grad)
x.grad.data.zero_()  # x的梯度归置为0，梯度重新计算，不累计
y.backward()
print(x.grad)

# 用Variable的写法（已废弃）
x = t.autograd.Variable(t.ones(4, 5), requires_grad=True)
y = t.cos(x.sum())
y.backward()
print(x.grad)