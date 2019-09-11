import numpy as np
from ripser import ripser
from persim import plot_diagrams

data = np.random.random((100, 3))  # 数据形状为个数*维数
diagrams = ripser(data)['dgms']
plot_diagrams(diagrams, show=True)  # 画H0，H1图

import numpy as np
from ripser import Rips

rips = Rips()
data = np.random.random((100, 2))
diagrams = rips.fit_transform(data)
rips.plot(diagrams, show=True)
