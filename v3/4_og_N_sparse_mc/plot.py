import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from itertools import cycle

lines = ['-o', '-v', '-x', '-s', '-D', '-h', '-H', '-+', '-d', '-|']
linecycler = cycle(lines)
mcs = np.load('mcs.npy')

Ns = np.arange(100, 1001, 100)
sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]

for idx, sparsity in enumerate(sparsities):
    # print(len(reservoir_sizes), len(mc[:, i]))
    # print(np.mean(mcs[:, idx, :], axis=1))
    plt.errorbar(Ns, np.mean(mcs[:, idx, :], axis=1), yerr=np.std(mcs[:, idx, :], axis=1), label=sparsities[idx], fmt=next(linecycler))

plt.plot(Ns, Ns, 'k', linestyle=":")
plt.xlabel('reservoir size', fontsize=16)
plt.ylabel("memory capacity", fontsize=16)
plt.title('OG method')
plt.grid(True)
plt.legend(loc=2)
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.xlim([9, 101])
# plt.ylim([0, 700])
plt.savefig('OG_N_vs_spars_MC.png')
plt.show()
