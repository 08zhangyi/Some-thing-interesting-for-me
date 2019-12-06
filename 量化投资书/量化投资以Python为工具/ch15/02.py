import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHindex = pd.read_csv('TRD_Index.csv')
print(SHindex.head(3))
Retindex = SHindex.Retindex
Retindex.hist()

mu = Retindex.mean()
sigma = Retindex.std()
plt.plot(np.arange(-0.06, 0.062, 0.002), scipy.stats.norm.pdf(np.arange(-0.06, 0.062, 0.002), mu, sigma))
plt.show()

print(scipy.stats.t.interval(0.95, len(Retindex)-1, mu, scipy.stats.sem(Retindex)))