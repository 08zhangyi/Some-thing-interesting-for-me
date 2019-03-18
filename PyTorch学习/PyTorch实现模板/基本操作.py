import torch as t
import numpy as np

# 生成Tensor
x = t.Tensor(5, 3)
x.numel()  # Tensor中元素个数
x = t.tensor([3, 4, 5])
a = np.random.rand(2, 3)
x = t.tensor(a)  # 从numpy数组到Tensor，第一种方法
x = t.from_numpy(a)  # 从numpy数组到Tensor，第二种方法，生成的为DoubleTensor，非默认的FloatTensor
print(x.dtype)  # Double对应t.float64或t.double，其他类型依此类推
a[0, 0] = 2.3  # a与x共享内存
b = x.numpy()  # 从Tensor到numpy数组
# b = x.tolist()  # 从Tensor到List
# x.numel()  # 打印Tensor中元素个数
b[0, 0] = 2.3  # b与x共享内存
x.requires_grad_(True)  # 若x的requires_grad为True，则不能直接用numpy()获取x的值，需要用x.detach().numpy()
b = x.detach().numpy()  # detach方法将张量从计算图上分离，不能求导，但可以改变数值

# Tensor中值的修改
x[0, 0] = 2.5
x[0:2, 0:2] = t.tensor(np.ones((2, 2)))  # 一组值的修改必须用Tensor包起来，如此赋值不改变内存地址

# 生成随机元素
x = t.rand(5, 3)
# 查看元素大小
x.size()
x.shape
# 内置生成函数
x = t.rand(5, 3)
x = t.randn(5, 3)
x = t.randperm(5)
x = t.ones(5, 3)
x = t.zeros(5, 3)
x = t.eye(2, 3)
x = t.arange(1, 6, 2)
x = t.linspace(1, 10, 3)
a = t.tensor(np.ones((5, 3)) * 0.5)
x = t.bernoulli(a)  # 伯努利分布的采样
print(x)

# Tensor操作
x = t.rand(5, 3, 4)
y = t.rand(5, 3, 4)
z = t.cat([x, y], dim=1)  # 矩阵拼接
print(z.size())
z = x + y
z = t.Tensor(5, 3, 4)
z = t.add(x, y, out=z)  # 指定赋值给z
print(z.size())
z = z.unsqueeze(2)  # 增加一个维度的操作
print(z.size())
z = z.squeeze(2)  # 减少一个维度的操作，z.squeeze()为挤出所有的维度
print(z.size())
z_size = z.size()
z = z.view(z_size[2], 1, 1, z_size[1], 1, z_size[0])  # 改变Tensor的形状
print(z.size())
z = z.resize_(5, 4, 1)  # 小了舍弃，大了填充数据，resize_的效果
print(z.size())
a = t.randn(1, 3, 2)
b = t.randn(2, 3, 2)
c = a.expand(2, 3, 2) + b  # 用expand扩张
print(c)
z.type(t.float64)  # 更改Tensor的类型
print(t.clamp(z, -0.1, 0.5))  # 上下截断

# Tensor选择函数
x = t.randn(3, 4)
print(x[0])
print(x[:, 1])
print(x[2, 3])
print(x > 1)
print(x[x > 1])  # 按照x>1的位置选出数值，不共享内存
print(x.masked_select(x > 1))  # 与上一行一致

y.add(x)  # y不改变
y.add_(x)  # y改变，原位加法
print(t.cuda.is_available())
print(t.cuda.device_count())
# Module在CUDA上的实现，要求相关的Tensor和Module都要CUDA实现
x = t.rand(5, 4).cuda(t.device('cuda:1'))  # 指定CUDA设备
x = t.rand(5, 4).cuda()
fc = t.nn.Linear(4, 3).cuda()  # 模块中的参数会自动转移到CUDA中
y = fc(x)

# Tensor类型转换
# t.set_default_tensor_type('torch.DoubleTensor')  # 设置默认类型的Tensor
a = t.Tensor(2, 3)
b = a.float()
c = a.type_as(b)
d = a.new(2, 3)  # 与构造新的Tensor等价，类型与a一致
t.set_default_tensor_type('torch.FloatTensor')

# 运算操作
a - t.arange(0, 6).float().view(2, 3)
b = t.cos(a)
b = a % 3
b = t.fmod(a, 3)  # 求模运算
b = a ** 2
b = t.pow(a, 2)
b = t.clamp(a, min=3)  # 下界截取到3
b = t.ones(2, 3)
c = b.sum(dim=0, keepdim=True)  # 保留维数，不用squeeze处理
c = b.sum(dim=0, keepdim=False)
c = b.sum(dim=1)
a = t.arange(0, 6).view(2, 3)
c = a.cumsum(dim=1)
a = t.linspace(0, 15, 6).view(2, 3)
b = t.linspace(15, 0, 6).view(2, 3)
c = a > b
print(c)
c = a[a>b]
c = t.max(a)
c = t.max(b, dim=1)
c = t.max(a, b)

# 对张量的结果进行连续化处理
a = t.Tensor(2, 3)
a.contiguous()

# Tensor存储的结构
a = t.arange(0, 6)
print(a.storage())  # Tensor的存储空间，可以通过storage共享空间
print(a.data_ptr())

# 持久化
a = t.Tensor(2, 3)
t.save(a, 'data\\a.pth')  # 保存Tensor，其他对象类似
b = t.load('data\\a.pth')  # 载入Tensor，其他对象类似
linear_layer = t.nn.Linear(3, 4)
t.save(linear_layer, 'data\\linear.pth')
linear_layer1 = t.load('data\\linear.pth')
print(linear_layer)
print(linear_layer1)