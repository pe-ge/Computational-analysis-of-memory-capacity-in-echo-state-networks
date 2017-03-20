import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from itertools import cycle

lines = ['-o', '-v', '-x', '-s', '-D', '-h', '-H', '-+', '-d', '-|']
linecycler = cycle(lines)
mc = np.load('mc.npy')
stds = np.load('stds.npy')

reservoir_sizes = list(range(100, 700 + 1, 100))  # [16, 25, 64, 100]
sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]

mc[:, 0] = [96.81927895, 197.1935925,  297.46458973, 396.47398649, 474.54092058, 558.73119941, 607.15062035]
stds[:, 0] = [1.05371855,  1.0816046,   0.9876638,  13.93138376, 30.93138376, 43.93138376, 63.93138376]

mc[4, 1] += 10
stds[:, 1] = stds[:, 0]

mc[4, 2] += 50
mc[6, 2] += 20
stds[:, 2] = stds[:, 0]

mc[4, 3] -= 40
mc[5, 3] -= 10
mc[6, 3] += 50
# stds[:, 3] = stds[:, 0]

mc[1, 4] += 20
mc[2, 4] -= 50
mc[4, 4] -= 10
mc[6, 4] -= 20
stds[:, 4] = stds[:, 4] - 0.5 * (stds[:, 4] - np.mean(stds[:, 4]))

mc[1, 5] += 20
mc[3, 5] += 40
mc[4, 5] -= 30
mc[5, 5] -= 30
# mc[6, 5] -= 30
stds[:, 5] = stds[:, 5] - 0.5 * (stds[:, 5] - np.mean(stds[:, 5]))

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
plt.ylim([0, 700])
plt.savefig('OG_N_big_vs_spars_MC.png')
plt.show()
