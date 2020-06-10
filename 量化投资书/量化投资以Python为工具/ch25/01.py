import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa import stattools
from arch import arch_model

SHret = pd.read_table('TRD_IndexSum.txt', index_col='Trddt', sep='\t')
SHret.index = pd.to_datetime(SHret.index)
SHret = SHret.sort_index()
plt.subplot(211)
plt.plot(SHret**2)
plt.xticks([])
plt.title('Squared Daily Return of SH Index')
plt.subplot(212)
plt.plot(np.abs(SHret))
plt.title('Absolute Daily Return of SH Index')

LjungBox = stattools.q_stat(stattools.acf(SHret**2)[1:13], len(SHret))
print(LjungBox[1][-1])

am = arch_model(SHret)
model = am.fit(update_freq=0)
print(model.summary())