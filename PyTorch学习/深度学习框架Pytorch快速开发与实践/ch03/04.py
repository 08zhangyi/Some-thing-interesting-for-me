import torch as t

a = t.randn(1, 3)
print(a)

print(t.mean(a))

a = t.randn(4, 4)
print(a)
print(t.mean(a, 1))