import torch
import torch.nn
import torch.optim


class MyFirstNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        out = self.layer1(input)
        out = torch.nn.ReLU(out)
        out = self.layer2(out)
        return out


def cross_entropy(true_label, prediction):
    if true_label == 1:
        return -torch.log(prediction)
    else:
        return -torch.log(1-prediction)


loss = torch.nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()

loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.LongTensor(3).random_(5)
output = loss(input, target)
output.backward()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()