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


plt.title("change of MC during orthogonalization process", fontsize=24)
plt.ylabel("memory capacity", fontsize=24)
plt.grid(True)

# oths vs lyapunov
ax2 = plt.subplot(212)
def plot_les(OTHs, LEs):
    for i in range(len(LEs)):
        nz_les = []
        nz_oths = []
        for j in range(len(LEs[i])):
            if LEs[i][j] != 0.0:
                nz_les.append(LEs[i][j])
                nz_oths.append(OTHs[i][j])
        ax2.plot(nz_oths, nz_les)

plot_les(oths, les)
plt.xlabel("orthogonality", fontsize=24, labelpad=-3)
plt.grid(True)
plt.ylabel("lyapunov exponent", fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
try_save_fig("figures/e=" + str(eta) + "t=" + str(tau))
try_save_fig("figures/e=" + str(eta) + "t=" + str(tau), ext="pdf")
plt.show()
