import pandas as pd
import numpy as np

dates = ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2106-01-06']
dates = pd.to_datetime(dates)
dates
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df

df
df.head(3)
df.tail(4)
df.columns
df.index
df.values
df.describe()

df[1:3]
df['A']
df[['A', 'C']]
df[df['A']>0]
df.loc[:, 'A']
df.loc[:, 'A':'C']
df.loc[dates[0:2], 'A':'C']
df.loc[dates[0], 'A']
df.at[dates[0], 'A']
df.loc[df.loc[:, 'A']>0]
df.iloc[2]
df.iloc[:, 2]
df.iloc[[1, 4], [2, 3]]
df.iloc[1:4, 2:4]
df.iloc[3, 3]
df.iat[3, 3]
df.loc[:, df.iloc[3]>0]
df.ix[2:5]
df.ix[[1, 3], 2]
df.ix[[1, 3], 'C']
df.ix[1:3, 'A':'C']
df.ix[1:3, df.iloc[3]>0]