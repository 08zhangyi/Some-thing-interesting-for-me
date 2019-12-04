import numpy as np
import scipy.stats
import pandas as pd

Norm = np.random.normal(size=5)
print(Norm)
scipy.stats.norm.pdf(Norm)
scipy.stats.norm.cdf(Norm)
HSRet300 = pd.read_csv('return300.csv')
ret = HSRet300.iloc[:, 1]
HS300_RetMean = ret.mean()
print(HS300_RetMean)
HS300_RetVariance = ret.var()
print(HS300_RetVariance)
scipy.stats.norm.pdf(0.05, HS300_RetMean, HS300_RetVariance**0.5)
