import pandas as pd
import matplotlib.pyplot as plt

TRD_Index = pd.read_table('TRD_Index.txt', sep='\t')
SHindex = TRD_Index[TRD_Index.Indexcd==1]
print(SHindex.head(3))
SZindex = TRD_Index[TRD_Index.Indexcd==399106]
print(SZindex.head(3))

plt.scatter(SHindex.Retindex, SZindex.Retindex)
plt.title('上证综指与深圳成指收益率的散点图')
plt.xlabel('上证综指收益率')
plt.ylabel('深圳成指收益率')
plt.show()

SZindex.index = SHindex.index
SZindex.Retindex.corr(SHindex.Retindex)
