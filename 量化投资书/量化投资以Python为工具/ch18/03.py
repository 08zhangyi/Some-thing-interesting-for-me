import datetime
import pandas as pd
import ffn

r = pd.Series([0, 0.1, -0.1, -0.01, 0.01, 0.02], index=[datetime.date(2015, 7, x) for x in range(3, 9)])
print(r)
value = (1 + r).cumprod()
print(value)
D = value.cummax() - value
print(D)
d = D / (D + value)
print(d)
MDD = D.max()
print(MDD)
mdd = d.max()
print(mdd)

print(ffn.calc_max_drawdown(value))