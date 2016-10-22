import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

oths = np.load('oths.npy')
les = np.load('les.npy')
iterations = 100

xs = np.arange(0, iterations + 1, 1)

ax1 = plt.subplot(211)
plt.errorbar(xs,
             np.average(oths, axis=0),
             yerr=np.std(oths, axis=0))
# plt.xlim([0.922, 1])

plt.yticks(fontsize=12)
plt.ylabel("orthogonality", fontsize=16)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid(True)

ax2 = plt.subplot(212)
plt.errorbar(xs,
             np.average(les, axis=0),
             yerr=np.std(les, axis=0))
plt.xlabel("iterations", fontsize=16)
plt.grid(True)
plt.ylabel("Lyapunov exponent", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.xlim([0.922, 1])
try_save_fig("figures/")
try_save_fig("figures/", ext="pdf")
plt.show()
