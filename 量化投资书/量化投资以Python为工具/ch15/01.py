import scipy.stats
import numpy as np

x = [10.1, 10, 9.8, 10.5, 9.7, 10.1, 9.9, 10.2, 10.3, 9.9]
print(scipy.stats.t.interval(0.95, len(x)-1, np.mean(x), scipy.stats.sem(x)))