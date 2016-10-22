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
plt.ylabel("MC", size=20)
plt.setp(ax1.get_xticklabels(), visible=False)

# lyapunov
ax2 = plt.subplot(312, sharex=ax1)
ax2.errorbar(xs, np.average(le, axis=0), yerr=np.std(le, axis=0))
plt.grid(True)
plt.ylabel("LE", size=20)
plt.setp(ax2.get_xticklabels(), visible=False)

# sparsity
ax3 = plt.subplot(313, sharex=ax1)
sp = 1 - sp / (N * N)
ax3.errorbar(xs, np.average(sp, axis=0), yerr=np.std(sp, axis=0))
plt.grid(True)
plt.ylabel("sparsity", size=20)

plt.xlabel("iteration", size=20)
try_save_fig("figures/")
try_save_fig("figures/", ext="pdf")
plt.show()
