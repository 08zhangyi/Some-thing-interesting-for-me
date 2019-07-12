import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F


input_size = 1
hidden_size = 100
num_layers = 10
num_classes = 1

df = pd.read_excel("data\\上证指数数据.xlsx")
df1 = df.iloc[:100, 3:6].values
xtrain_features = torch.FloatTensor(df1)
df3 = df["涨跌"].astype(float)
xtrain_labels = torch.FloatTensor(df3[:100])
xtrain = torch.unsqueeze(xtrain_features, dim=1)
ytrain = torch.unsqueeze(xtrain_labels, dim=0)
x1 = xtrain_features.view(100, 3, 1)
y = ytrain


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


rnn = RNN(input_size, hidden_size, num_layers, num_classes)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)

for epoch in range(100000):
    inputs = x1
    target = y
    out = rnn(inputs)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print('Epoch[{}], loss: {:.6f}'.format(epoch+1, loss.data))

rnn.eval()
predict = rnn(x)
predict = predict.data.numpy()
print(predict)