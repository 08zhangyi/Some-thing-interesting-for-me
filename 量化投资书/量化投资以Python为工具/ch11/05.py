import pandas as pd
import numpy as np

s1 = pd.Series([1, 2, 3], index=list('ABC'))
s2 = pd.Series([4, 5, 6], index=list('BCD'))
s1 + s2
df1 = pd.DataFrame(np.arange(1, 13).reshape(3, 4), index=list('abc'), columns=list('ABCD'))
df1 - s1
df2 = pd.DataFrame(np.arange(1, 13).reshape(4, 3), index=list('bcde'), columns=list('CDE'))
df1 * df2
df1.div(df2, fill_value=0)
df0 = pd.DataFrame(np.random.rand(6, 4), index=pd.date_range('20160101', periods=6), columns=list('ABCD'))
df0.apply(max, axis=0)
f = lambda x: x.max() - x.min()
df0.apply(f, axis=1)

print(df1)
print(df2)
df3 = df1.mul(df2, fill_value=0)
df3.isnull()
df3.notnull()
df3.B[df3.B.notnull()]

df4 = pd.DataFrame(np.random.rand(5, 4), index=list('abcde'), columns=list('ABCD'))
df4.loc['c', 'A'] = np.nan
df4.loc['b': 'd', 'C'] = np.nan
print(df4)
df4.fillna(0)
df4.fillna(method='ffill')
df4.fillna(method='bfill')
df4.fillna(method='backfill', axis=1)
df4.fillna(method='pad', limit=2)
df4.fillna('missing', inplace=True)
print(df4)

df4.loc['c', 'A'] = np.nan
df4.loc['b': 'd', 'C'] = np.nan
print(df4)
df4.dropna(axis=0)
df4.dropna(axis=1, thresh=3)
df4.dropna(axis=1, how='all')

df5 = pd.DataFrame({'c1': ['apple'] * 3 + ['banana'] * 3 + ['apple'],
                    'c2': ['a', 'a', 3, 3, 'b', 'b', 'a']})
print(df5)
df5.duplicated()
df5.drop_duplicates()
df5.duplicated(['c2'])
df5.drop_duplicates(['c2'])