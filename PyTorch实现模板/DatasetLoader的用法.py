import numpy as np
import torch as t
import torch.utils.data as Data

t.manual_seed(1)
np.random.seed(1)
DATA_SIZE = 2000
BATCH_SIZE = 64

x = np.linspace(-7, 10, DATA_SIZE)[:, np.newaxis]  # x的形状为batch_size * feature_num
y = np.square(x) - 5 + np.random.normal(0, 2, x.shape)
x, y = t.from_numpy(x).float(), t.from_numpy(y).float()

test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
test_y = np.square(test_x) - 5 + np.random.normal(0, 2, test_x.shape)
test_x = t.from_numpy(test_x).float()
test_y = t.from_numpy(test_y).float()

train_dataset = Data.TensorDataset(x, y)  # 直接注入数据
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for step, (batch_x, batch_y) in enumerate(train_loader):
    print(step)