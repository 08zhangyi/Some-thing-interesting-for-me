import torch as t
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

input_size = 784
num_classes = 10
num_epochs = 10
batch_size = 50
learning_rate = 0.001
# train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = t.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = t.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


net = LogisticRegression(5, 1)
print(net)

model = LogisticRegression(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))

correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28)
    outputs = model(images)
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted==labels).sum()

print('Accuracy of the model on the 10000 test images: %d %%' % (100*correct/total))

t.save(model.state_dict(), 'data/model.pkl')