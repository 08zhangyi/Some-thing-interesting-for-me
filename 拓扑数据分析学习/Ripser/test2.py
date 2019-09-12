from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# 数据导入
data = datasets.make_circles(n_samples=100)[0] + 5 * datasets.make_circles(n_samples=100)[0]

# 数据计算与画图
dgms = ripser(data)['dgms']
print(dgms)  # dgms的结构
plot_diagrams(dgms, show=True)
plot_diagrams(dgms, plot_only=[0], ax=plt.subplot(121))
plot_diagrams(dgms, plot_only=[1], ax=plt.subplot(122))
plt.show()

# 不同系数的同调群
dgms = ripser(data, coeff=3)['dgms']
plot_diagrams(dgms, plot_only=[1], title="Homology of Z/3Z", show=True)
dgms = ripser(data, coeff=7)['dgms']
plot_diagrams(dgms, plot_only=[1], title="Homology of Z/7Z", show=True)

# 高阶同调群计算
dgms = ripser(data, maxdim=2)['dgms']
plot_diagrams(dgms, show=True)

# 限制计算用的最大半径值
dgms = ripser(data, thresh=0.2)['dgms']
plot_diagrams(dgms, show=True)
dgms = ripser(data, thresh=1)['dgms']
plot_diagrams(dgms, show=True)
dgms = ripser(data, thresh=2)['dgms']
plot_diagrams(dgms, show=True)
dgms = ripser(data, thresh=999)['dgms']
plot_diagrams(dgms, show=True)

# 画图设置
plot_diagrams(dgms, xy_range=[-2, 10, -1, 20])  # xy轴范围
import matplotlib.style
print(matplotlib.style.available)  # 画图的风格命令
plot_diagrams(dgms, colormap='seaborn')

# 画生命期长短图
plot_diagrams(dgms, lifetime=True, show=True)