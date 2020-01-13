import numpy as np
import math
import matplotlib.pyplot as plt


def cal_mean(frac):
    return 0.08*frac + 0.15*(1-frac)


mean = list(map(cal_mean, [x/50 for x in range(51)]))
sd_mat = np.array([list(map(lambda x: math.sqrt((x**2)*0.12**2+((1-x)**2)*0.25**2+2*x*(1-x)*(-1.5+i*0.5)*0.12*0.25), [x/50 for x in range(51)])) for i in range(1, 6)])
plt.plot(sd_mat[0, :], mean, label='-1')
plt.plot(sd_mat[1, :], mean, label='-0.5')
plt.plot(sd_mat[2, :], mean, label='0')
plt.plot(sd_mat[3, :], mean, label='0.5')
plt.plot(sd_mat[4, :], mean, label='1')
plt.legend(loc='upper left')
plt.show()