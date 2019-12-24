import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

TRD_Index = pd.read_table('TRD_Index.txt', sep='\t')
SHindex = TRD_Index[TRD_Index.Indexcd==1]
SZindex = TRD_Index[TRD_Index.Indexcd==399106]
SHRet = SHindex.Retindex
SZRet = SZindex.Retindex
SZRet.index = SHRet.index

model = sm.OLS(SHRet, sm.add_constant(SZRet)).fit()
print(model.summary())
print(model.fittedvalues[:5])

plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.show()

sm.qqplot(model.resid_pearson, stats.norm, line='45')
plt.show()

plt.scatter(model.fittedvalues, model.resid_pearson**0.5)
plt.xlabel('拟合值')
plt.ylabel('标准化残差的平方根')
plt.show()