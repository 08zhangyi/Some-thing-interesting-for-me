import numpy as np
import scipy.stats
import pandas as pd

np.random.binomial(100, 0.5, 20)
np.random.binomial(10, 0.3, 3)
scipy.stats.binom.pmf(20, 100, 0.5)
scipy.stats.binom.pmf(50, 100, 0.5)
dd = scipy.stats.binom.pmf(np.arange(0, 21, 1), 100, 0.5)
dd.sum()
scipy.stats.binom.cdf(20, 100, 0.5)
HSRet300 = pd.read_csv('return300.csv')
ret = HSRet300.iloc[:, 1]
print(ret.head(3))
p = len(ret[ret>0])/len(ret)
print(p)
prob = scipy.stats.binom.pmf(6, 10, p)
print(prob)