import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

oths = np.load('oths.npy')
mcs = np.load('mcs.npy')
les = np.load('les.npy')
eta = 0.03

# oths vs mcs
def fix_mcs(MCs):
    fixed_mc = []
    for i in range(len(mcs)):
        f = []
        mc = mcs[i]
        for a in mc:
            if type(a) == np.float64:
                f.append(a)
            else:
                f.append(a[0])
        fixed_mc.append(f)

    return fixed_mc

fixed_mcs = fix_mcs(mcs)

ax1 = plt.subplot(211)
for i in range(len(fixed_mcs)):
    ax1.plot(oths[i], fixed_mcs[i])


plt.title("change of MC during orthogonalization process", fontsize=24)
plt.ylabel("memory capacity", fontsize=24)
plt.grid(True)

# oths vs lyapunov
ax2 = plt.subplot(212)
def plot_les(OTHs, MCs, LEs):
    for i in range(len(LEs)):
        les = [LEs[i][0]]
        oths = [OTHs[i][0]]
        for j in range(len(MCs[i])):
            if type(MCs[i][j]) == tuple:
                l = MCs[i][j][1][0]
                if l != 0.0:
                    les.append(l)
                    oths.append(OTHs[i][j])
        ax2.plot(oths, les)

plot_les(oths, mcs, les)
plt.xlabel("orthogonality", fontsize=24, labelpad=-3)
plt.grid(True)
plt.ylabel("lyapunov exponent", fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
try_save_fig("figures/eta=" + str(eta))
try_save_fig("figures/eta=" + str(eta), ext="pdf")
plt.show()
