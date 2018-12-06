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

print('----------------------')
a = t.ones(3, 4, requires_grad=True)
b = t.zeros(3, 4)  # 默认requires_grad=False
c = a + b  # c的requires_grad自动设定为True，由于a要求导
d = c.sum()
d.backward()
print(a.grad)
print(a.requires_grad, b.requires_grad, c.requires_grad)
x = t.ones(3, 4, requires_grad=True)
y = x**2 * t.exp(x)
print(y)
y.backward(t.ones(y.size()))  # 标量可以用.backward()方法，而y是张量，必须给定每个求梯度的值
# y对x的导数为一个矩阵，因此需要给一个关于y的方向，才能求出x的对应导出
# 记D=dx/dy，则Dx=D*Dy，Dy为y的方向梯度，Dx为x对应的方向梯度
print(x.grad)

print('----------------------')
x = t.ones(1)
b = t.rand(1, requires_grad=True)
w = t.rand(1, requires_grad=True)
y = w * x
z = y + b
print(x.is_leaf, b.is_leaf, w.is_leaf, y.is_leaf, z.is_leaf)
print(z.grad_fn)  # grad_fn的获取
print(w.grad)  # 未求梯度前grad为None
z.backward(retain_graph=True)
print(w.grad)
# w.grad.data.zero_()  # 重复求梯度前可以手动清零梯度
z.backward()
print(w.grad)
# z.backward()  # 不用retain_graph=True回报错，backward()时，前向计算图以被清空，因此无法再次backward()，加入retain_graph=True后，前向计算图保留

print('----------------------')
# 钩子hook方法
def variable_hook(grad):
    # 输入为梯度grad，必须如此定义
    print('y的梯度：\r\n', grad)  # 对grad的操作
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = w * x
hook_handle = y.register_hook(variable_hook)  # 注册hook方法
z = y.sum()
z.backward()
hook_handle.remove()  # 可以移除hook


print('----------------------')
# 自己定义自己的Function模块
class MultiplyAdd(t.autograd.Function):
    # forward和backward都必须注册为静态方法
    @staticmethod
    def forward(ctx, w, x, b):
        # ctx表示为静态类符号
        print('type in forwad', type(x))
        ctx.save_for_backward(w, x)  # 反向传播时候需要记录的变量
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        print('type in backward', type(x))
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        # 对应forward部分的输入
        # 不求导的输出为None
        return grad_w, grad_x, grad_b


x = t.ones(1)
b = t.rand(1, requires_grad=True)
w = t.rand(1, requires_grad=True)
z = MultiplyAdd.apply(w, x, b)
z.backward()
print(x.grad, w.grad, b.grad)