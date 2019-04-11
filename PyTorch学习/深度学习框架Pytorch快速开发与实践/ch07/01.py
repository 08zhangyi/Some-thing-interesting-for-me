import torch as t
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 3
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)

print(train_data.train_data.size())
print(train_data.train_labels.size())

for i in range(1, 4):
    plt.imshow(train_data.train_data[i].numpy(), cmap='gray')
    plt.title('%i' % train_data.train_labels[i])
    plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
test_x = t.unsqueeze(test_data.test_data, dim=1).type(t.FloatTensor)
test_y = test_data.test_labels


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)
params = list(cnn.parameters())
print(len(params))
print(params[0].size())

optimizer = t.optim.Adam(cnn.parameters(), lr=LR)

loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        output = cnn(x)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100 == 0:
            test_output = cnn(test_x)
            pred_y = t.max(test_output, 1)[1].squeeze()
            accuracy = sum(pred_y.numpy()==test_y.numpy()) / test_y.size(0)
            print('Epoch:', epoch, '|Step:', step, '|train loss:%.4f' % loss.data, '|test accuracy:%.4f' % accuracy)

test_output = cnn(test_x[:20])
pred_y = t.argmax(test_output, 1)[1].data.numpy().squeeze()
print(pred_y[:20], 'prediction number')
print(test_y[:20].numpy(), 'rea; number')