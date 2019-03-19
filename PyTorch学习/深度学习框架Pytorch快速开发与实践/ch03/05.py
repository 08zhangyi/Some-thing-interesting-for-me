import torch as t

print(t.eq(t.Tensor([[1, 2], [3, 4]]), t.Tensor([[1, 1], [4, 4]])))

print(t.equal(t.Tensor([1, 2]), t.Tensor([1, 2])))

print(t.ge(t.Tensor([[1, 2], [3, 4]]), t.Tensor([[1, 1], [4, 4]])))

print(t.gt(t.Tensor([[1, 2], [3, 4]]), t.Tensor([[1, 1], [4, 4]])))