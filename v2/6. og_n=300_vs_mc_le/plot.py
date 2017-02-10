import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig


mc = np.load('mc.npy')
le = np.load('le.npy')
ORTHOPROCESS_ITERATIONS = 50
xs = range(ORTHOPROCESS_ITERATIONS + 1)


ax1 = plt.subplot(211)
ax1.errorbar(xs, np.average(mc, axis=0), yerr=np.std(mc, axis=0))
plt.ylabel('MC')
plt.ylim((100, 300))
plt.title('OG method')

ax2 = plt.subplot(212)
ax2.errorbar(xs, np.average(le, axis=0), yerr=np.std(le, axis=0))
plt.ylabel('LE')
plt.ylim((-0.25, 0.0))
plt.xlabel('iteration')

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
