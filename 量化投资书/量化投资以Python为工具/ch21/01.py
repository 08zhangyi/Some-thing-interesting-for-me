import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

stock = pd.read_table('stock.txt', sep='\t', index_col='Trddt')
print(stock.head())
HXBank = stock[stock.Stkcd==600015]
print(HXBank.head(3))
HXBank.index = pd.to_datetime(HXBank.index)
HXRet = HXBank.Dretwd
HXRet.name = 'HXRet'
print(HXRet.head())
print(HXRet.tail())
HXRet.plot()

ThreeFactors = pd.read_table('ThreeFactors.txt', sep='\t', index_col='TradingDate')
print(ThreeFactors.head(n=3))
ThreeFactors.index = pd.to_datetime(ThreeFactors.index)
ThrFac = ThreeFactors['2014-01-02':]
ThrFac = ThrFac.iloc[:, [2, 4, 6]]
print(ThrFac.head())
HXThrFac = pd.merge(pd.DataFrame(HXRet), pd.DataFrame(ThrFac), left_index=True, right_index=True)
print(HXThrFac.head(n=3))
print(HXThrFac.tail(n=3))

plt.subplot(2, 2, 1)
plt.scatter(HXThrFac.HXRet, HXThrFac.RiskPremium2)
plt.subplot(2, 2, 2)
plt.scatter(HXThrFac.HXRet, HXThrFac.SMB2)
plt.subplot(2, 2, 3)
plt.scatter(HXThrFac.HXRet, HXThrFac.NML2)

regThrFac = sm.OLS(HXThrFac.HXRet, sm.add_constant(HXThrFac.iloc[:, 1:4]))
result = regThrFac.fit()
print(result.summary())

print(result.params)