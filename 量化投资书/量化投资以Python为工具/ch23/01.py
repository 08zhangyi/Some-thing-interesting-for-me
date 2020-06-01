from statsmodels.tsa import stattools
import pandas as pd
from statsmodels.graphics.tsaplots import *
import matplotlib.pyplot as plt
from arch.unitroot import ADF
import numpy as np

data = pd.read_table('TRD_index.txt', sep='\t', index_col='Trddt')
SHindex = data[data.Indexcd==1]
SHindex.index = pd.to_datetime(SHindex.index)
SHRet = SHindex.Retindex
print(type(SHRet))
print(SHRet.head())
print(SHRet.tail())
acfs = stattools.acf(SHRet)
print(acfs[:5])
pacfs = stattools.pacf(SHRet)
print(pacfs[:5])

plot_acf(SHRet, use_vlines=True, lags=30)

plot_pacf(SHRet, use_vlines=True, lags=30)

SHclose = SHindex.Clsindex
SHclose.plot()
plt.title('2014-2015年上证综指收盘指数时序图')

SHRet.plot()
plt.title('2014-2015年上证综指收益率指数时序图')

plot_acf(SHRet, use_vlines=True, lags=30)
plot_pacf(SHRet, use_vlines=True, lags=30)
plot_acf(SHclose, use_vlines=True, lags=30)

adfSHRet = ADF(SHRet)
print(adfSHRet.summary().as_text())

adfSHclose = ADF(SHclose)
print(adfSHclose.summary().as_text())

whiteNoise = np.random.standard_normal(size=500)
plt.plot(whiteNoise, c='b')
plt.title('White Noise')

LjungBox1 = stattools.q_stat(stattools.acf(SHRet)[1:13], len(SHRet))
print(LjungBox1)
print(LjungBox1[1][-1])
LjungBox2 = stattools.q_stat(stattools.acf(SHclose)[1:13], len(SHRet))
print(LjungBox2[1][-1])
