import numpy as np

matrix = np.array([[1, 2, 3], [20, 30, 40]])
print(matrix[0, 1])

matrix = np.array([[5, 10, 15],
                   [20, 25, 30],
                   [35, 40, 45]])
print(matrix[:, 1])
print(matrix[:, 0:2])
print(matrix[1:3, :])
print(matrix[1:3, 0:2])