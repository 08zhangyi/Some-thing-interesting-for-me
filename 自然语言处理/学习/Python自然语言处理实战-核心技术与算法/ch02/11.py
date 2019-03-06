import numpy as np

matrix = np.array([[5, 10, 15],
                   [20, 25, 30],
                   [35, 40, 45]])
m = (matrix == 25)
print(m)

second_column_25 = (matrix[:, 1] == 25)
print(second_column_25)
print(matrix[second_column_25, :])