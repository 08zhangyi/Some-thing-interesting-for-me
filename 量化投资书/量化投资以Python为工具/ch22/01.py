import pandas as pd

Index = pd.read_table('TRD_Index_22.txt', sep='\t', index_col='Trddt')
SHindex = Index[Index.Indexcd==1]
print(SHindex.head(n=3))

print(type(SHindex))
Clsindex = SHindex.Clsindex
print(Clsindex.head(n=3))
print(type(Clsindex))
print(type(Clsindex.index))

Clsindex.index = pd.to_datetime(Clsindex.index)
print(Clsindex.head())
print(type(Clsindex))
print(type(Clsindex.index))
Clsindex.plot()

SHindex.index = pd.to_datetime(SHindex.index)
SHindexPart = SHindex['2014-10-08':'2014-10-31']
print(SHindexPart.head(n=2))
print(SHindexPart.tail(n=2))

SHindex2015 = SHindex['2015']
print(SHindex2015.head(n=2))
print(SHindex2015.tail(n=2))

SHindexAfter2015 = SHindex['2015':]
print(SHindexAfter2015.head(n=2))
SHindexBefore2015 = SHindex[:'2014-12-31']
print(SHindexBefore2015.tail(n=2))

SHindex9End = SHindex['2014-09':'2014']
print(SHindex9End.head(n=2))
print(SHindex9End.tail(n=2))

print(Clsindex.head())
print(Clsindex.tail(n=1))
Clsindex.hist()
print(Clsindex.max())
print(Clsindex.min())
print(Clsindex.mean())
print(Clsindex.median())
print(Clsindex.std())
print(Clsindex.describe())
