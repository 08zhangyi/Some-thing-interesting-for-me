import pandas as pd

ChinaBank = pd.read_csv('ChinaBank.csv', index_col='Date')
ChinaBank = ChinaBank.iloc[:, 1:]
ChinaBank.head()
ChinaBank.index = pd.to_datetime(ChinaBank.index)
Close = ChinaBank.Close

import matplotlib.pyplot as plt

plt.plot(Close['2014'])

plt.plot([1, 1, 0, 0, -1, 0, 1, 1, -1])
plt.ylim(-1.5, 1.5)
plt.xticks(range(9), ['2015-02-01', '2015-02-02', '2015-02-03', '2015-02-04', '2015-02-05', '2015-02-06',
                      '2015-02-07', '2015-02-08', '2015-02-09'])
plt.xticks(range(9), ['2015-02-01', '2015-02-02', '2015-02-03', '2015-02-04', '2015-02-05', '2015-02-06',
                      '2015-02-07', '2015-02-08', '2015-02-09'], rotation=45)

plt.plot(Close['2014'])
plt.title('中国银行2014年收盘价曲线')
plt.plot(Close['2014'])
plt.title('中国银行2014年收盘价曲线', loc='right')

plt.plot(Close['2014'])
plt.title('中国银行2014年收盘价曲线')
plt.xlabel('日期')
plt.ylabel('收盘价')

plt.plot(Close['2014'], label='收盘价')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.title('中国银行2014年收盘价曲线')
plt.grid(True, axis='y')

Open = ChinaBank.Open
plt.plot(Close['2014'], label='收盘价')
plt.plot(Open['2014'], label='开盘价')
plt.legend()

plt.plot(Close['2014'], label='收盘价', linestyle='solid')
plt.plot(Open['2014'], label='开盘价', ls='-.')
plt.legend()
plt.xlabel('日期')
plt.ylabel('价格')
plt.title('中国银行2014年开盘与收盘价曲线')
plt.grid(True, axis='y')

plt.plot(Close['2014'], c='r', label='收盘价')
plt.plot(Open['2014'], c='b', label='开盘价', ls='--')
plt.legend(loc='best')
plt.xlabel('日期')
plt.ylabel('价格')
plt.title('中国银行2014年开盘与收盘价曲线')
plt.grid(True, axis='both')

plt.show()