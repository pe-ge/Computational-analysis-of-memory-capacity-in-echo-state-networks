import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

ITERATIONS = 100
N = 100

mc = np.load('mc.npy')
le = np.load('le.npy')
sp = np.load('sp.npy')

xs = range(ITERATIONS + 1)
ax1 = plt.subplot(311)
#ax1.errorbar(xs, np.average(mc[:, :-1], axis=0), yerr=np.std(mc, axis=0))
ax1.errorbar(xs, np.average(mc, axis=0), yerr=np.std(mc, axis=0))
plt.grid(True)
plt.ylabel("memory capacity", size=20)

plt.setp(ax1.get_xticklabels(), visible=False)

# lyapunov
# ax2 = plt.subplot(312, sharex=ax1)
# ax2.errorbar(xs, np.average(le, axis=0), yerr=np.std(le, axis=0))
# plt.grid(True)
# plt.ylabel("lyapunov exp.", size=20)

# sparsity
# ax3 = plt.subplot(313, sharex=ax1)
# avg = np.average(sp, axis=0)
# std = np.std(sp, axis=0)
# ax3.errorbar(xs, [1 - a / (N * N) for a in avg], yerr=std)
# plt.grid(True)
# plt.ylabel("sparsity", size=20)

plt.xlabel("iteration", size=20)
try_save_fig("figures/")
try_save_fig("figures/", ext="pdf")
plt.show()
