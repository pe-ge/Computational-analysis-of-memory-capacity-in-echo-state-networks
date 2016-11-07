import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig


mc = np.load('mc.npy')
le = np.load('le.npy')
xs = range(mc.shape[0] + 1)


ax1 = plt.subplot(211)
ax1.errorbar(xs, np.average(mc, axis=0), yerr=np.std(mc, axis=0))
plt.ylabel('mc')
plt.title('ON method, N=300, tau=10**-20')

ax2 = plt.subplot(212)
ax2.errorbar(xs, np.average(le, axis=0), yerr=np.std(le, axis=0))
plt.ylabel('le')

plt.xlabel('iteration')


try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
