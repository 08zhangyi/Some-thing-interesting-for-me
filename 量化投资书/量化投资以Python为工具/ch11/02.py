from datetime import datetime
import pandas as pd

date = datetime(2016, 1, 1)
date = pd.Timestamp(date)
date
type(date)
ts = pd.Series(1, index=[date])
ts
ts.index
ts.index[0]
dates = ['2016-01-01', '2016-01-02', '2016-01-03']
ts = pd.Series([1, 2, 3], index=pd.to_datetime(dates))
ts
ts.index
ts.index[0]
dates = [datetime(2016, 1, 1), datetime(2016, 1, 2), datetime(2016, 1, 3)]
ts = pd.Series([1, 2, 3], index=dates)
ts.index[0]
ts['20160101']
ts['2016-01-01']
ts['01/01/2016']
ts
ts['2016']
ts['2016-01':'2016-02']
ts.truncate(after='2016-01-02')
ts.shift(1)
ts.shift(-1)
price = pd.Series([20.34, 20.56, 21.01, 20.65, 21.34],
                  index=pd.to_datetime(['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05']))
(price - price.shift(1))/price.shift(1)
ts.index.freq is None
rts = ts.resample('M', how='first')
rts
rts = ts.resample('MS', how='first')
rts