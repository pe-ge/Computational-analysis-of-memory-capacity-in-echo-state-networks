import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

oths = np.load('oths.npy')
mcs = np.load('mcs.npy')
les = np.load('les.npy')
tau = 0.01
eta = 0.04

# oths vs mcs

ax1 = plt.subplot(211)
for i in range(len(mcs)):
    ax1.plot(oths[i], mcs[i])

plt.xlim([0.918, 1])

plt.title("change of MC during orthogonalization process", fontsize=20)
plt.ylabel("memory capacity", fontsize=20)
plt.grid(True)

# oths vs lyapunov
ax2 = plt.subplot(212)
for i in range(len(les)):
    ax2.plot(oths[i], les[i])

plt.xlabel("orthogonality", fontsize=20, labelpad=-3)
plt.grid(True)
plt.ylabel("lyapunov exponent", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([0.918, 1])
try_save_fig("figures/e=" + str(eta) + "t=" + str(tau))
try_save_fig("figures/e=" + str(eta) + "t=" + str(tau), ext="pdf")
plt.show()
