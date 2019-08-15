import numpy as np
import torch


learning_rate = 0.001


def get_data():
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    dtype = torch.FloatTensor
    X = torch.from_numpy(train_X).type(dtype).requires_grad_(False).view(17, 1)
    y = torch.from_numpy(train_Y).type(dtype).requires_grad_(False).view(17, 1)
    return X, y


def get_weights():
    w = torch.randn(1).requires_grad_(True)
    b = torch.randn(1).requires_grad_(True)
    return w, b


def simple_network(x):
    y_pred = torch.matmul(x, w) + b
    return y_pred


def loss_fn(y, y_pred):
    loss = (y_pred - y).pow(2).sum()
    for param in [w, b]:
        if param.grad is not None:
            param.grad.zero_()
    loss.backward()
    return loss.data


def optimize(learning_rate):
    global w, b
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad


x, y = get_data()
w, b = get_weights()
print(w.__class__)
for i in range(500):
    y_pred = simple_network(x)
    loss = loss_fn(y, y_pred)
    if i % 50 == 0:
        print(loss)
    optimize(learning_rate)