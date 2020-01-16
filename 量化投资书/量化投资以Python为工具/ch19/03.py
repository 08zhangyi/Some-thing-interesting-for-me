import numpy as np
import pandas as pd
import scipy.linalg

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


def blacklitterman(returns, tau, P, Q):
    mu = returns.mean()
    sigma = returns.cov()
    pi1 = mu
    ts = tau * sigma
    Omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
    middle = scipy.linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
    er = np.expand_dims(pi1, axis=0).T + np.dot(np.dot(np.dot(ts, P.T), middle), (Q-np.expand_dims(np.dot(P, pi1.T), axis=1)))
    posteriorSigma = sigma + ts - np.dot(ts.dot(P.T).dot(middle).dot(P), ts)
    return [er, posteriorSigma]


pick1 = np.array([1, 0, 1, 1, 1])
q1 = np.array([0.003*4])
pick2 = np.array([0.5, 0.5, 0, 0, -1])
q2 = np.array([0.001])
P = np.array([pick1, pick2])
Q = np.array([q1, q2])
print(P)
print(Q)
res = blacklitterman(sh_return, 0.1, P, Q)
p_mean = pd.DataFrame(res[0], index=sh_return.columns, columns=['posterior_mean'])
print(p_mean)
p_cov = res[1]
print(p_cov)


def blminVar(blres, goalRet):
    covs = np.array(blres[1])
    means = np.array(blres[0])
    L1 = np.append(np.append((covs.swapaxes(0, 1)), [means.flatten()], 0), [np.ones(len(means))], 0).swapaxes(0, 1)
    L2 = list(np.ones(len(means)))
    L2.extend([0, 0])
    L3 = list(means)
    L3.extend([0, 0])
    L4 = np.array([L2, L3])
    L = np.append(L1, L4, 0)
    results = scipy.linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
    return pd.DataFrame(results[:-2], index=blres[1].columns, columns=['p_weight'])


print(blminVar(res, 0.75/252))