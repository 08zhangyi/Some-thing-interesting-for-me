import pandas as pd
import ffn
from scipy.stats import norm

SAPower = pd.read_csv('SAPower.csv', index_col='Date')
SAPower.index = pd.to_datetime(SAPower.index)
DalianRP = pd.read_csv('DalianRP.csv', index_col='Date')
DalianRP.index = pd.to_datetime(DalianRP.index)
returnS = ffn.to_returns(SAPower.Close).dropna()
returnD = ffn.to_returns(DalianRP.Close).dropna()
print(returnS.std())
print(returnD.std())


def cal_half_dev(returns):
    mu = returns.mean()
    temp = returns[returns<mu]
    half_deviation = (sum((mu-temp)**2)/len(returns))**0.5
    return half_deviation


print(cal_half_dev(returnS))
print(cal_half_dev(returnD))

print(returnS.quantile(0.05))
print(returnD.quantile(0.05))
print(norm.ppf(0.05, returnS.mean(), returnS.std()))
print(norm.ppf(0.05, returnD.mean(), returnD.std()))
print(returnS[returnS<=returnS.quantile(0.05)].mean())
print(returnD[returnD<=returnD.quantile(0.05)].mean())
