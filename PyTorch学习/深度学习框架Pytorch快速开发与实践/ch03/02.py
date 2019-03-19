import numpy as np
import torch as t

a = np.ones(5)
b = t.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

x = t.zeros(2, 1, 2, 1, 2)
print(x.size())

y = t.squeeze(x)
print(y.size())

y = t.squeeze(x, 0)
print(y.size())

y = t.squeeze(x, 1)
print(y.size())