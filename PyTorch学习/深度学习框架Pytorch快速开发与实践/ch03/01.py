import torch as t

z = t.Tensor(4, 5)
print(z)


y = t.rand(4, 5)
print(z + y)

print(t.add(z, y))

b = z.numpy()
print(b)
