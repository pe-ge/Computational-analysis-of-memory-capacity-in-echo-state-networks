from library.aux import try_save_fig
from matplotlib import pyplot as plt
import numpy as np

means = np.load('means.npy')
stds = np.load('stds.npy')

xs = range(len(means[0]))
mean = np.average(means, axis=0)
std = np.mean(stds, axis=0)

plt.errorbar(
    xs,
    mean,
    yerr=std)

plt.grid(True)
# plt.xlim((0, 140))
# plt.ylim((-0.05, 1.2))
plt.title("error curves for ON method")
plt.xlabel("$k$", size=20)
plt.ylabel("$e(k)$", size=20)

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext='pdf')
plt.show()
