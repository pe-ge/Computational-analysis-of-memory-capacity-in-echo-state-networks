import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from itertools import cycle

lines = ['-o', '-v', '-x', '-s', '-D', '-h', '-H', '-+', '-d', '-|']
linecycler = cycle(lines)
sps = np.load('sps200-5.npy')

sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]
etas = [i*10**-2 for i in range(1, 11)]

for idx, sparsity in enumerate(sparsities):
    # print(len(reservoir_sizes), len(mc[:, i]))
    # print(np.mean(mcs[:, idx, :], axis=1))
    plt.errorbar(etas, np.mean(sps[idx, :, :], axis=1), yerr=np.std(sps[idx, :, :], axis=1), label=sparsities[idx], fmt=next(linecycler))

# plt.plot(Ns, Ns, 'k', linestyle=":")
plt.xlabel('eta', fontsize=16)
plt.ylabel('sparsity', fontsize=16)
plt.title('ON method, N=200, 5 instances')
plt.grid(True)
plt.legend(loc=2)
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.xlim([9, 101])
# plt.ylim([0, 700])
plt.savefig('ON_N=200_eta_vs_sparsity.png')
plt.show()
