import numpy as np

nfl = np.genfromtxt('data/price.csv', delimiter=',')
print(nfl)

nfl = np.genfromtxt('data/price.csv', dtype='U75', skip_header=1, delimiter=',')
print(nfl)