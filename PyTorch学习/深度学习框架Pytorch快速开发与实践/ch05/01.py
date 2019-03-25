import torch as t
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))

correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28)
    outputs = net(images)
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted==labels).sum()

print('Accuracy of the model on the 10000 test images: %d %%' % (100*correct/total))

for i in range(1, 4):
    plt.imshow(train_dataset.train_data[i].numpy(), cmap='gray')
    plt.title('%i' % train_dataset.train_labels[i])
    plt.show()

test_output = net(images[:20])
pred_y = t.max(test_output, 1)[1].data.numpy().squeeze()
print('prediction number', pred_y)
t.save(net.state_dict(), 'data/model.pkl')