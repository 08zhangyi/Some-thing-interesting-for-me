import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import tadasets


def drawLineColored(X, C):
    for i in range(X.shape[0]-1):
        plt.plot(X[i:i+2, 0], X[i:i+2, 1], c=C[i, :], lineWidth=3)


def plotCocycle2D(D, X, cocycle, thresh):
    """
    :param D: 点的距离函数
    :param X: 2D点云
    :param cocycle:
    :param thresh:
    """
    N = X.shape[0]  # 点数量
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]
    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:  # 小于等于thresh的对都连上线
                Y = np.zeros((len(t), 2))  # Y为Xi到Xj的连线
                Y[:, 0] = X[i, 0] + t * (X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t * (X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
    for k in range(cocycle.shape[0]):  # 属于cocycle的给连上
        [i, j, val] = cocycle[k, :]
        if D[i, j] < thresh:
            [i, j] = [min(i, j), max(i, j)]
            a = 0.5 * (X[i, :] + X[j, :])  # Xi和Xj的中点
            plt.text(a[0], a[1], '%g'%val, color='b')
    for i in range(N):  # 标记点
        plt.text(X[i, 0], X[i, 1], '%i'%i, color='r')
    plt.axis('equal')


np.random.seed(9)
x = tadasets.dsphere(n=12, d=1, noise=0.1)
plt.scatter(x[:, 0], x[:, 1])
plt.axis('equal')
plt.show()

result = ripser(x, coeff=17, do_cocycles=True)
diagrams = result['dgms']
cocycles = result['cocycles']
D = result['dperm2all']

dgm1 = diagrams[1]
idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])
plot_diagrams(diagrams, show=False)
plt.scatter(dgm1[idx, 0], dgm1[idx, 1], 20, 'k', 'x')
plt.title("Max 1D birth = %.3g, death = %.3g" % (dgm1[idx, 0], dgm1[idx, 1]))
plt.show()

cocycle = cocycles[1][idx]  # 1维，第idx个cocycle
print(cocycle)
thresh = dgm1[idx, 1]
plotCocycle2D(D, x, cocycle, thresh)
plt.title("1-Form Thresh=%g" % thresh)
plt.show()

thresh = dgm1[idx, 1] - 0.00001  # 需要更小一点的阈值
plotCocycle2D(D, x, cocycle, thresh)
plt.title("1-Form Thresh=%g" % thresh)
plt.show()


thresh = dgm1[idx, 0]
plotCocycle2D(D, x, cocycle, thresh)
plt.title("1-Form Thresh=%g"%thresh)
plt.show()