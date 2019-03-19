import torch as t
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

input_size = 1
output_size = 1
learning_rate = 0.001

xtrain = np.array([[2.3], [4.4], [3.7], [6.1], [7.3], [2.1], [5.6], [7.7], [8.7], [4.1], [6.7], [6.1], [7.5], [2.1], [7.2],
                   [5.6], [5.7], [7.7], [3.1]], dtype=np.float32)
ytrain = np.array([[3.7], [4.76], [4.], [7.1], [8.6], [3.5], [5.4], [7.6], [7.9], [5.3], [7.3], [7.5], [8.5], [3.2], [8.7],
                   [6.4], [6.6], [7.9], [6.3]], dtype=np.float32)

plt.figure()
plt.scatter(xtrain, ytrain)
plt.xlabel('xtrain')
plt.ylabel('ytrain')
# plt.show()


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss()
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
for epoch in range(num_epochs):
    inputs = t.from_numpy(xtrain)
    targets = t.from_numpy(ytrain)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch +1) % 50 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.data))

model.eval()
predicted = model(t.from_numpy(xtrain)).detach().numpy()
plt.plot(xtrain, ytrain, 'ro')
plt.plot(xtrain, predicted, label='predict')
plt.legend()
plt.show()