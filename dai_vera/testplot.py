import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

times = np.arange(10)
values = np.random.rand(10) * 100

from drawlesioncurves import plot_sampled_curve  

plot_sampled_curve(ax, times, values, "pre")
plt.show()
