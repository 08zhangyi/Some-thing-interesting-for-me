import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

penn = pd.read_excel('Penn World Table.xlsx', 2)
print(penn.head(3))

model = sm.OLS(np.log(penn.rgdpe), sm.add_constant(penn.iloc[:, -6:])).fit()
print(model.summary())

print(penn.iloc[:, -6:].corr())

model = sm.OLS(np.log(penn.rgdpe), sm.add_constant(penn.iloc[:, -5:-1])).fit()
print(model.summary())