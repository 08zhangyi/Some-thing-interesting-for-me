import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.arange(0, 5, 0.002), scipy.stats.chi.pdf(np.arange(0, 5, 0.002), 3))
plt.title('Probability Density Plot of Chi-Square Distribution')

x = np.arange(-4, 4.004, 0.004)
plt.plot(x, scipy.stats.norm.pdf(x), label='Normal')
plt.plot(x, scipy.stats.t.pdf(x, 5), label='df=5')
plt.plot(x, scipy.stats.t.pdf(x, 30), label='df=30')
plt.legend()

plt.plot(np.arange(0, 5, 0.002), scipy.stats.f.pdf(np.arange(0, 5, 0.002), 4, 40))
plt.title('Probability Density Plot of F Distribution')
