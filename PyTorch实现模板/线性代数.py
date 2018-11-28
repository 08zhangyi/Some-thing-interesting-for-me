import torch as t

batch_size = 16
size_1 = 4
size_2 = 5
output_size = 10

# 用Linear层实现矩阵乘法
x = t.randn(batch_size, size_1, size_2)
linear_layer = t.nn.Linear(size_2, output_size)
y = linear_layer(x)
print(y.size())  # batch_size * size_1, output_size

# 手动实现矩阵乘法
weights = t.randn(size_2, output_size)
y = t.matmul(x, weights)
print(y.size())  # batch_size * size_1, output_size

# 更底层的实现矩阵乘法的办法
x = t.randn(batch_size, size_1, size_2)
y = t.randn(batch_size, size_2, output_size)  # 计划x和y在第2和第1个维度想乘
x = x.unsqueeze(3)
y = y.unsqueeze(1)
z = x * y
z = t.sum(z, 2)
print(z.size())


# 矩阵元素乘
x = t.randn(3, 4, 5)
y = t.randn(3, 4, 5)
z = x * y
print(z.size())