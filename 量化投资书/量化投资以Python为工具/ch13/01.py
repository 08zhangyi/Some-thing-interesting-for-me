import pandas as pd
import matplotlib.pyplot as plt

returns = pd.read_csv('retdata.csv')
gsyh = returns.gsyh
plt.hist(gsyh)

returns.zglt.mean()
returns.pfyh.mean()
returns.zglt.median()
returns.pfyh.median()
returns.zglt.mode()
returns.pfyh.mode()
[returns.zglt.quantile(i) for i in [0.25, 0.75]]
[returns.pfyh.quantile(i) for i in [0.25, 0.75]]

returns.zglt.max() - returns.zglt.min()
returns.zglt.mad()
returns.zglt.var()
returns.zglt.std()
returns.pfyh.max() - returns.pfyh.min()
returns.pfyh.mad()
returns.pfyh.var()
returns.pfyh.std()

plt.show()