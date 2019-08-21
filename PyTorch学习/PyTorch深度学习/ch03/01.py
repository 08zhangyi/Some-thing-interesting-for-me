import torch
from torch.nn import Linear


inp = torch.randn(1, 10)
myLayer = Linear(in_features=10, out_features=5, bias=True)
myLayer(inp)
print(myLayer.weight)
print(myLayer.bias)

myLayer1 = Linear(10, 5)
myLayer2 = Linear(5, 2)
myLayer2(myLayer1(inp))

sample_data = torch.Tensor([[1, 2, -1, -1]])
myRelu = torch.nn.ReLU()
print(myRelu(sample_data))