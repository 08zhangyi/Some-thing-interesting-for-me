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
# 另外一种实现办法，加b的都是batch下的矩阵运算
x = t.randn(batch_size, size_1, size_2)
y = t.randn(batch_size, size_2, output_size)  # 计划x和y在第2和第1个维度想乘
z = t.bmm(x, y)
print(z.size())

# 矩阵元素乘
x = t.randn(3, 4, 5)
y = t.randn(3, 4, 5)
z = x * y
print(z.size())

# 矩阵行列式
x = t.randn(3, 3)
y = t.det(x)
# y = t.logdet(x)
print(y)

# 矩阵逆
y = t.inverse(x)  # x的逆
y = t.matmul(x, y)
print(y)

# 矩阵求迹
x = t.randn(3, 3)
y = t.trace(x)
print(y)

# 矩阵对角元素
x = t.randn(3, 3)
y = t.diag(x)
print(y)

# 矩阵上下三角矩阵
x = t.randn(5, 5)
y = t.triu(x)  # 上三角
y = t.tril(x)  # 下三角
y = t.triu(x, diagonal=1)  # 偏移量参数
print(y)

# 矩阵转置
x = t.randn(3, 4)
y = x.t()
y.contiguous()  # 转置后存储空间不连续化，进行连续化处理
print(x)

# 矩阵奇异值分解
x = t.randn(5, 5)
u, s, v = t.svd(x)
print(u, s, v)