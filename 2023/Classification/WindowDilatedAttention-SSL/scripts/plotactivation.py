import os
import matplotlib.pyplot as plt
import numpy as np

# generates 20 random points whose value increases slowly, from 1e02 to 1e06

import random

# Sinh ngẫu nhiên 20 giá trị tăng dần trong khoảng từ 1e+02 đến 1e+07
random_values = sorted(random.sample(range(int(1e-01), int(1e+06) + 1), 20))* np.logspace(0, 1, 20, endpoint=True)
slow_increasing_values = sorted(random.sample(range(int(1e-2), int(1e+02) + 1), 20))
slow_increasing_values.sort()
# plot the 2 random values in log scale
plt.yscale('log') 
plt.plot(random_values, label='random values')
plt.plot(slow_increasing_values, label='slow increasing values')
plt.legend()
plt.show()

