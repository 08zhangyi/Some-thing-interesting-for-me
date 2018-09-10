import torch as t
import numpy as np

# 生成Tensor
x = t.Tensor(5, 3)
x = t.tensor([3, 4, 5])
a = np.random.rand(2, 3)
x = t.tensor(a)  # 从numpy数组到Tensor，第一种方法
x = t.from_numpy(a)  # 从numpy数组到Tensor，第二种方法
a[0, 0] = 2.3  # a与x共享内存
b = x.numpy()  # 从Tensor到numpy数组
b[0, 0] = 2.3  # b与x共享内存

# Tensor中值的修改
x[0, 0] = 2.5
x[0:2, 0:2] = t.tensor(np.ones((2, 2)))  # 一组值的修改必须用Tensor包起来

# 生成随机元素
x = t.rand(5, 3)
# 查看元素大小
x.size()
# 内置生成函数
x = t.rand(5, 3)
x = t.randn(5, 3)
x = t.randperm(5)
x = t.ones(5, 3)
x = t.zeros(5, 3)
a = t.tensor(np.ones((5, 3)) * 0.5)
x = t.bernoulli(a)  # 伯努利分布的采样
print(x)

# Tensor操作
x = t.rand(5, 3, 4)
y = t.rand(5, 3, 4)
z = x + y
z = t.Tensor(5, 3, 4)
z = t.add(x, y, out=z)  # 指定赋值给z

y.add(x)  # y不改变
y.add_(x)  # y改变，原位加法