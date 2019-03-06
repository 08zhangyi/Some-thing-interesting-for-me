import numpy as np

matrix = np.array([['5', '10', '15'],
                   ['20', '25', '30'],
                   ['35', '40', '']])
second_column_25 = (matrix[:, 2] == '')
matrix[second_column_25, 2] = '0'
print(matrix)