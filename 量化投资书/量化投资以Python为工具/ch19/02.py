import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import ffn
import matplotlib
matplotlib.use('TkAgg')

stock = pd.read_table('stock.txt', sep='\t', index_col='Trddt')
stock.index = pd.to_datetime(stock.index)
fjgs = stock.ix[stock.Stkcd==600033, 'Dretwd']
fjgs.name = 'fjgs'
zndl = stock.ix[stock.Stkcd==600023, 'Dretwd']
zndl.name = 'zndl'
sykj = stock.ix[stock.Stkcd==600183, 'Dretwd']
sykj.name = 'sykj'
hxyh = stock.ix[stock.Stkcd==600015, 'Dretwd']
hxyh.name = 'hxyh'
byjc = stock.ix[stock.Stkcd==600004, 'Dretwd']
byjc.name = 'byjc'
sh_return = pd.concat([byjc, fjgs, hxyh, sykj, zndl], axis=1)
print(sh_return.head())

sh_return = sh_return.dropna()
cumreturn = (1+sh_return).cumprod()
sh_return.plot()
plt.title('Daily Return of 5 Stocks(2014-2015)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5, fancybox=True, shadow=True)
cumreturn.plot()
plt.title('Cumulative Return of 5 Stocks(2014-2015)')
plt.show()

print(sh_return.corr())


class MeanVariance:
    def __init__(self, returns):
        self.returns = returns

    def minVar(self, goalRet):
        covs = np.array(self.returns.cov())
        means = np.array(self.returns.mean())
        L1 = np.append(np.append(covs.swapaxes(0, 1), [means], 0), [np.ones(len(means))], 0).swapaxes(0, 1)
        L2 = list(np.ones(len(means)))
        L2.extend([0, 0])
        L3 = list(means)
        L3.extend([0, 0])
        L4 = np.array([L2, L3])
        L = np.append(L1, L4, 0)
        results = linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
        return np.array([list(self.returns.columns), results[:-2]])

    def frontierCurve(self):
        goals = [x/500000 for x in range(-100, 4000)]
        variances = list(map(lambda x: self.calVar(self.minVar(x)[1, :].astype(np.float)), goals))
        plt.plot(variances, goals)

    def meanRet(self, fracs):
        meanRisky = ffn.to_returns(self.returns).mean()
        assert len(meanRisky) == len(fracs), 'Length of fractions must be equal to number of assets'

    def calVar(self, fracs):
        return np.dot(np.dot(fracs, self.returns.cov()), fracs)


minVar = MeanVariance(sh_return)
minVar.frontierCurve()
plt.show()

train_set = sh_return['2014']
test_set = sh_return['2015']
varMinimizer = MeanVariance(train_set)
goal_return = 0.003
portfolio_weight = varMinimizer.minVar(goal_return)
print(portfolio_weight)
test_return = np.dot(test_set, np.array([portfolio_weight[1, :].astype(np.float)]).swapaxes(0, 1))
test_retrun = pd.DataFrame(test_return, index=test_set.index)
test_cum_return = (1+test_return).cumprod()
sim_weight = np.random.uniform(0, 1, (100, 5))
sim_weight = np.apply_along_axis(lambda x: x/sum(x), 1, sim_weight)
sim_return = np.dot(test_set, sim_weight.swapaxes(0, 1))
sim_return = pd.DataFrame(sim_return, index=test_cum_return.index)
sim_cum_return = (1 + sim_return).cumprod()
plt.plot(sim_cum_return.index, sim_cum_return, color='green')
plt.plot(test_cum_return.index, test_cum_return)
plt.show()