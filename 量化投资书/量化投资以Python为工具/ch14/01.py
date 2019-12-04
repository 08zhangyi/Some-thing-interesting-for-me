import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RandomNumber = np.random.choice([1, 2, 3, 4, 5], size=100, replace=True, p=[0.1, 0.1, 0.3, 0.3, 0.2])
pd.Series(RandomNumber).value_counts()
pd.Series(RandomNumber).value_counts()/100

HSRet300 = pd.read_csv('return300.csv')
HSRet300.head(n=2)
density = stats.kde.gaussian_kde(HSRet300.iloc[:, 1])
bins = np.arange(-5, 5, 0.02)
plt.subplot(211)
plt.plot(bins, density(bins))
plt.title('沪深300收益率序列的概率密度曲线图')
plt.subplot(212)
plt.plot(bins, density(bins).cumsum())
plt.title('沪深300收益率序列的累积分布函数图')

plt.show()
