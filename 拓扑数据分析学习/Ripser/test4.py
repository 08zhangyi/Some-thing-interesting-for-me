from ripser import ripser
from persim import plot_diagrams
import tadasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import time


def getGreedyPerm(D):
    """
    :param D: 点的距离矩阵
    :return:
    """
    N = D.shape[0]
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)  # 离0最远的点
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]


def getApproxSparseDM(lambdas, eps, D):
    """
    :param lambdas: 点的插入距离
    :param eps:
    :param D:
    :return:
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps
    nBounds = ((eps**2 + 3*eps +2)/eps)*lambdas
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])
    M = np.minimum((E0+E1)*minlam, E0*(minlam+maxlam))
    t = np.arange(len(I))
    t = t[D<=M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]
    t = np.ones(len(I))
    t[D <= 2 * minlam * E0] = 0
    D[t == 1] = 2.0 * (D[t == 1] - minlam[t == 1] * E0)
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()


# 原始版本
X = tadasets.infty_sign(n=2000, noise=0.1)
plt.scatter(X[:, 0], X[:, 1])
tic = time.time()
resultfull = ripser(X)
toc = time.time()
timefull = toc-tic
print("Elapsed Time: %.3g seconds, %i Edges added" % (timefull, resultfull['num_edges']))

# 逼近提速版本
eps = 0.1
tic = time.time()
D = pairwise_distances(X, metric='euclidean')
lambdas = getGreedyPerm(D)
DSparse = getApproxSparseDM(lambdas, eps, D)
resultsparse = ripser(DSparse, distance_matrix=True)  # 直接输入距离矩阵
toc = time.time()
timesparse = toc-tic
percent_added = 100.0*float(resultsparse['num_edges'])/resultfull['num_edges']
print("Elapsed Time: %.3g seconds, %i Edges added"%(timesparse, resultsparse['num_edges']))

# 两个距离矩阵画图
plt.figure(figsize=(10, 5))
plt.subplot(121)
D = pairwise_distances(X, metric='euclidean')
plt.imshow(D)
plt.title("Original Distance Matrix: %i Edges"%resultfull['num_edges'])
plt.subplot(122)
DSparse = DSparse.toarray()
DSparse = DSparse + DSparse.T
plt.imshow(DSparse)
plt.title("Sparse Distance Matrix: %i Edges"%resultsparse['num_edges'])
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_diagrams(resultfull['dgms'], show=False)
plt.title("Full Filtration: Elapsed Time %g Seconds"%timefull)
plt.subplot(122)
plt.title("Sparse Filtration (%.3g%% Added)\nElapsed Time %g Seconds"%(percent_added, timesparse))
plot_diagrams(resultsparse['dgms'], show=False)
plt.show()

eps = 0.4
tic = time.time()
D = pairwise_distances(X, metric='euclidean')
lambdas = getGreedyPerm(D)
DSparse = getApproxSparseDM(lambdas, eps, D)
resultsparse = ripser(DSparse, distance_matrix=True)
toc = time.time()
timesparse = toc-tic
percent_added = 100.0*float(resultsparse['num_edges'])/resultfull['num_edges']
print("Elapsed Time: %.3g seconds, %i Edges added"%(timesparse, resultsparse['num_edges']))

plt.figure(figsize=(10, 5))
plt.subplot(121)
D = pairwise_distances(X, metric='euclidean')
plt.imshow(D)
plt.title("Original Distance Matrix: %i Edges"%resultfull['num_edges'])
plt.subplot(122)
DSparse = DSparse.toarray()
DSparse = DSparse + DSparse.T
plt.imshow(DSparse)
plt.title("Sparse Distance Matrix: %i Edges"%resultsparse['num_edges'])
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_diagrams(resultfull['dgms'], show=False)
plt.title("Full Filtration: Elapsed Time %g Seconds"%timefull)
plt.subplot(122)
plt.title("Sparse Filtration (%.3g%% Added)\nElapsed Time %g Seconds"%(percent_added, timesparse))
plot_diagrams(resultsparse['dgms'], show=False)
plt.show()