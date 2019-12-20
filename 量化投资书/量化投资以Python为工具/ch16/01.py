import pandas as pd
import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols

year_return = pd.read_csv('TRD_Year.csv', encoding='gbk')
print(year_return.head())

model = ols('Return ~ C(Industry)', data=year_return.dropna()).fit()
tabel1 = anova.anova_lm(model)
print(tabel1)

PSID = pd.read_csv('PSID.csv')
print(PSID.head())

model = ols('earnings ~ C(married)+C(educatn)', data=PSID.dropna()).fit()
tabel2 = anova.anova_lm(model)
print(tabel2)

model = ols('earnings ~ C(married)*C(educatn)', data=PSID.dropna()).fit()
table3 = anova.anova_lm(model)
print(table3)