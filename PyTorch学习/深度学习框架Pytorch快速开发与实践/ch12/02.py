import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F


df = pd.read_excel("data\\上证指数数据.xlsx")
df1 = df.iloc[:100, 3:6].values
xtrain_features = torch.FloatTensor(df1)
df3 = df.iloc[1:101, 6].values
xtrain_labels = torch.FloatTensor(df3)
xtrain = torch.unsqueeze(xtrain_features, dim=1)
ytrain = torch.unsqueeze(xtrain_labels, dim=1)
x, y = xtrain, ytrain


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = Net(input_size=3, hidden_size=100, num_classes=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for epoch in range(100000):
    inputs = x
    target = y
    out = model(inputs)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print('Epoch[{}], loss: {:.6f}'.format(epoch+1, loss.data))

model.eval()
predict = model(x)
predict = predict.data.numpy()
print(predict)