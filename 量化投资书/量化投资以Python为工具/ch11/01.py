import pandas as pd
import numpy as np

S1 = pd.Series()
S1
S2 = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
S2
S2.values
S2.index
S2['f'] = 11
S2
pd.Series({'a': 1, 'b': 3, 'c': 5, 'd': 7})
S3 = pd.Series([1, 3, -55, 7])
S3
np.random.seed(54321)
pd.Series(np.random.rand(5))
pd.Series(np.arange(2, 6))

S4 = pd.Series([0, np.NaN, 2, 4, 6, 8, True, 10, 12])
S4.head()
S4.head(3)
S4.tail()
S4.tail(6)
S4.take([2, 4, 0])

S5 = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
S5[2], S5['d']
S5[[1, 3, 4]]
S5[['b', 'e', 'd']]
S5[0:4]
S5['a':'d']
