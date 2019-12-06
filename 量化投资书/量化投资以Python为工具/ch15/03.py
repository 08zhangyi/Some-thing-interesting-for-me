import scipy.stats
import numpy as np
import pandas as pd

TRD_Index = pd.read_table('TRD_Index.txt', sep='\t')
SHindex = TRD_Index[TRD_Index.Indexcd==1]
SHRet = SHindex.Retindex
print(scipy.stats.ttest_1samp(SHRet, 0))

SZindex = TRD_Index[TRD_Index.Indexcd==399106]
SZRet = SZindex.Retindex
print(scipy.stats.ttest_ind(SHRet, SZRet))
print(scipy.stats.ttest_rel(SHRet, SZRet))