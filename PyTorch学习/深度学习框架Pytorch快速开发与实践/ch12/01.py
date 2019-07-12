import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F


df = pd.read_excel("data\\上证指数数据.xlsx")
df1 = df.iloc[:100, 3:6].values
xtrain_features = torch.FloatTensor(df1)
df2 = df.iloc[1:101, 7].values
xtrain_labels = torch.FloatTensor(df2)
xtrain = torch.unsqueeze(xtrain_features, dim=1)
ytrain = torch.unsqueeze(xtrain_labels, dim=1)
x, y = xtrain, ytrain


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


model = Net(n_feature=3, n_hidden=10, n_output=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

num_epochs = 1000000
for epoch in range(num_epochs):
    inputs = x
    target = y
    out = model(inputs)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data))

model.eval()
predict = model(x)
predict = predict.data.numpy()
print(predict)