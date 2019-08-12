import torch
from sklearn.datasets import load_boston
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from glob import glob

x = torch.rand(10)
print(x.size())

temp = torch.FloatTensor([23, 24, 24.5, 26, 27.2, 23.0])
print(temp.size())

boston = load_boston()
boston_tensor = torch.from_numpy(boston.data)
print(boston_tensor.size())
print(boston_tensor[:2])

panda = np.array(Image.open('data\\panda.jpg').resize((224, 224)))
panda_tensor = torch.from_numpy(panda)
print(panda_tensor.size())
plt.imshow(panda)

sales = torch.FloatTensor([1000.0, 323.2, 333.4, 444.5, 1000.0, 323.2, 333.4, 444.5])
print(sales[:5])
print(sales[:-5])

plt.imshow(panda_tensor[:, :, 0].numpy())
plt.imshow(panda_tensor[25:175, 60:130, 0].numpy())

sales = torch.eye(3, 3)
print(sales[0, 1])

# cats = glob(data_path+'*.jpg')
# cat_imgs = np.array([np.array(Image.open(cat).resize((224, 224))) for cat in cats[:64]])
# cat_imgs = cat_imgs.reshape(-1, 224, 224, 3)
# cat_tensors = torch.from_numpy(cat_imgs)
# print(cat_tensors.size())

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = a + b
d = torch.add(a, b)
a += 5

print(a*b)

# a = torch.rand(10000, 10000)
# b = torch.rand(10000, 10000)
# a.matmul(b)
#
# a = a.cuda()
# b = b.cuda()
# a.matmul(b)

x = torch.ones(2, 2, requires_grad=True)
print(x.requires_grad)
y = x.mean()
y.backward()
print(x.grad)
print(x.grad_fn)
print(x.data)
print(y.grad_fn)