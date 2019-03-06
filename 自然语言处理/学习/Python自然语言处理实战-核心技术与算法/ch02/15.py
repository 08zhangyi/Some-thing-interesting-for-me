import numpy as np

vector = np.array([5, 10, 15, 20])
vector.sum()

matrix = np.array([[5, 10, 15],
                   [20, 25, 30],
                   [35, 40, 45]])
print(matrix.sum(axis=1))
print(matrix.sum(axis=0))