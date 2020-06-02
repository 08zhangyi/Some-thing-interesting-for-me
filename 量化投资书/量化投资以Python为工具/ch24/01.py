import pandas as pd
from arch.unitroot import ADF
from statsmodels.tsa import stattools, arima_model
from statsmodels.graphics.tsaplots import *
import matplotlib.pyplot as plt
import math

CPI = pd.read_csv('CPI.csv', index_col='time')
CPI.index = pd.to_datetime(CPI.index)
print(CPI.head(n=3))
print(CPI.tail(n=3))
print(CPI.shape)

CPItrain = CPI[3:]
print(CPItrain.head(n=3))
CPI.sort().plot(title='CPI 2005-2014')

CPItrain = CPItrain.dropna().CPI
print(ADF(CPItrain, max_lags=10).summary().as_text())

LjungBox = stattools.q_stat(stattools.acf(CPItrain)[1:12], len(CPItrain))
print(LjungBox[1][-1])

axe1 = plt.subplot(121)
axe2 = plt.subplot(122)
plot1 = plot_acf(CPItrain, lags=30, ax=axe1)
plot2 = plot_pacf(CPItrain, lags=30, ax=axe2)

model1 = arima_model.ARIMA(CPItrain, order=(1, 0, 1)).fit()
print(model1.summary())
model2 = arima_model.ARIMA(CPItrain, order=(1, 0, 2)).fit()
print(model2.summary())
model3 = arima_model.ARIMA(CPItrain, order=(2, 0, 1)).fit()
model4 = arima_model.ARIMA(CPItrain, order=(2, 0, 2)).fit()
model5 = arima_model.ARIMA(CPItrain, order=(3, 0, 1)).fit()
model6 = arima_model.ARIMA(CPItrain, order=(3, 0, 2)).fit()

print(model6.conf_int())

stdresid = model6.resid / math.sqrt(model6.sigma)
plt.plot(stdresid)
plot_acf(stdresid, lags=20)
LjungBox = stattools.q_stat(stattools.acf(stdresid)[1:13], len(stdresid))
print(LjungBox[1][-1])
LjungBox = stattools.q_stat(stattools.acf(stdresid)[1:20], len(stdresid))
print(LjungBox[1][-1])
plot_acf(stdresid, lags=40)

print(model6.forecast(3)[0])

print(CPI.head(3))

Datang = pd.read_csv('Datang.csv', index_col='time')
Datang.index = pd.to_datetime(Datang.index)
returns = Datang['2014-01-01':'2016-01-01']
print(returns.head(n=3))
print(returns.tail(n=3))
print(ADF(returns).summary())
print(stattools.q_stat(stattools.acf(returns)[1:12], len(returns))[1])

print(stattools.arma_order_select_ic(returns, max_ma=4))
model = arima_model.ARIMA(returns, order=(1, 0, 1)).fit()
print(model.summary())
print(model.conf_int())
stdresid = model.resid / math.sqrt(model.sigma2)
plt.plot(stdresid)
plot_acf(stdresid, lags=12)
LjungBox = stattools.q_stat(stattools.acf(stdresid)[1:12], len(stdresid))
print(LjungBox[1])