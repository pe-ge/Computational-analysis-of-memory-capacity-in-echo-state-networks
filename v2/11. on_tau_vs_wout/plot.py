import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

taus = [10**-i for i in range(20)]

iws = np.load('iws.npy')
ows_m = np.load('owsm.npy')
ows_s = np.load('owss.npy')

ax = plt.subplot(1, 1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.invert_xaxis()
# ax.errorbar(taus, np.mean(ows_m, axis=0))
ax.errorbar(taus, np.mean(ows_m, axis=0), yerr=np.std(ows_m, axis=0))
plt.xlabel(r'$\tau$', fontsize=20)
plt.ylabel(r'$\Vert\bf W^{out}\Vert}$', fontsize=14)
plt.title('ON Method')

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
