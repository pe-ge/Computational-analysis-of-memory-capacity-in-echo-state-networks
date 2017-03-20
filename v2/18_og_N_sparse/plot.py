import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from itertools import cycle

lines = ['-o', '-v', '-x', '-s', '-D', '-h', '-H', '-+', '-d', '-|']
linecycler = cycle(lines)
mc = np.load('mc.npy')
stds = np.load('stds.npy')

reservoir_sizes = list(range(10, 100 + 1, 10))  # [16, 25, 64, 100]
sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]

print(mc)
for i in range(mc.shape[1]):
    # print(len(reservoir_sizes), len(mc[:, i]))
    plt.errorbar(reservoir_sizes, mc[:, i], yerr=stds[:, i], label=sparsities[i], fmt=next(linecycler))

plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
plt.xlabel('reservoir size', fontsize=16)
plt.ylabel("memory capacity", fontsize=16)
plt.title('OG method')
plt.grid(True)
plt.legend(loc=2)
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.xlim([9, 101])
plt.savefig('OG_N_small_vs_spars_MC.png')
plt.show()
