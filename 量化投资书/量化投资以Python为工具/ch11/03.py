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

df.T
df.sort_index(axis=0, ascending=False)
df.sort_index(axis=1, ascending=False)
df.sort_values(by=['C'])
df
df.rank(axis=0)
df.rank(axis=1, ascending=False)
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20160102', periods=5))
s1
df['E'] = s1
df
df = df[list('ABCD')]
pd.concat([df, s1], axis=1)
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=pd.date_range('20160110', periods=3))
df1
df.append(df1)
pd.concat([df, df1], join='inner')
df.drop(dates[1:3])
df.drop('A', axis=1)
del df['A']
df
df.loc[dates[2], 'C'] = 0
df.iloc[0, 4] = 0
df.loc[:, 'B'] = np.arange(0, len(df))
df
new_index = pd.date_range('20160102', period=7)
df.reindex(new_index, column=list('ABCD'))