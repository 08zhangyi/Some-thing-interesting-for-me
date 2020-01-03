import pandas as pd
import ffn
import numpy as np

stock = pd.read_csv('stockszA.csv', index_col='Trddt')
Vanke = stock[stock.Stkcd==2]
close = Vanke.Clsprc
print(close.head())
close.index = pd.to_datetime(close.index)
close.index.name = 'Date'
print(close.head())
lagclose = close.shift(1)
print(lagclose.head())
Calclose = pd.DataFrame({'close': close, 'lagclose': lagclose})
print(Calclose.head())
simpleret = (close - lagclose) / lagclose
simpleret.name = 'simpleret'
print(simpleret.head())
calret = pd.merge(Calclose, pd.DataFrame(simpleret), left_index=True, right_index=True)
print(calret.head())
print(calret.iloc[5, :])

ffnSimpleret = ffn.to_returns(close)
ffnSimpleret.name = 'ffnSimpleret'
print(ffnSimpleret.head())

annualize = (1 + simpleret).cumprod()[-1]**(245/311)-1
print(annualize)


def annualize(returns, period):
    if period == 'day':
        return ((1 + returns).cumprod()[-1]**(245 / len(returns)) - 1)
    elif period == 'month':
        return ((1 + returns).cumprod()[-1]**(12 / len(returns)) - 1)
    elif period == 'quarter':
        return ((1 + returns).cumprod()[-1] ** (4 / len(returns)) - 1)
    elif period == 'year':
        return ((1 + returns).cumprod()[-1] ** (1 / len(returns)) - 1)
    else:
        raise Exception("Wrong period")


comporet = np.log(close/lagclose)
comporet.name = 'comporet'
print(comporet.head())
print(comporet[5])
ffnComporet = ffn.to_log_returns(close)
print(ffnComporet.head())

comporet2 = np.log(close/close.shift(2))
comporet2.name = 'comporet2'
print(comporet2.head())
print(comporet2[5])

comporet = comporet.dropna()
print(comporet.head())
sumcomporet = comporet + comporet.shift(1)
print(sumcomporet.head())

simpleret.plot()
((1+simpleret).cumpord()-1).plot()
