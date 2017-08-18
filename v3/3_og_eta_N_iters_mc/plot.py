import numpy as np
import matplotlib.pyplot as plt

etas = np.linspace(0.01, 0.1, 10)
Ns = np.arange(100, 1001, 100)

mcs = np.load('mcs.npy')
iters = np.load('iters.npy')

mcs_mean = np.mean(mcs, axis=2)
mcs_std = np.std(mcs, axis=2)

iters_mean = np.mean(iters, axis=2)
iters_std = np.std(iters, axis=2)

# MC
ax1 = plt.subplot(1, 1, 1)
plt.grid(True)
for i in range(len(etas)):
    plt.errorbar(Ns, mcs_mean[:, i], yerr=mcs_std[:, i], label=etas[i])
plt.title('OG method, 10 instances')
plt.ylabel('MC')
plt.xticks(Ns)
plt.xlim(Ns[0] - 1, Ns[-1] + 1)
handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title=r'$\eta$')

# iters
# ax3 = plt.subplot(2, 1, 2)
# plt.grid(True)
# for i in range(len(etas)):
    # plt.errorbar(Ns, iters_mean[:, i], yerr=iters_std[:, i], label=etas[i])
# plt.ylabel('Num iters')
# plt.xticks(Ns)
# plt.xlim(Ns[0] - 1, Ns[-1] + 1)
# handles, labels = ax1.get_legend_handles_labels()
# plt.legend(handles[::-1], labels[::-1], title=r'$\eta$', loc=2)

plt.xlabel('N')

plt.savefig('OG.png')
plt.show()
