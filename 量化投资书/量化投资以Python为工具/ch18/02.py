import pandas as pd
import ffn

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