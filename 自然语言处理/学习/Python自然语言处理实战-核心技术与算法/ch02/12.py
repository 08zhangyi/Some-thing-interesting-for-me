import numpy as np

vector = np.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
vector[equal_to_ten_or_five] = 50
print(vector)

matrix = np.array([[5, 10, 15],
                   [20, 25, 30],
                   [35, 40, 45]])
second_column_25 = (matrix[:, 1] == 25)
matrix[second_column_25, 1] = 10
print(matrix)