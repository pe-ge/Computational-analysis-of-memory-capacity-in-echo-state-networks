import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from library.aux import try_save_fig

oths = np.load('oths.npy')
mcs = np.load('mcs.npy')
les = np.load('les.npy')


def find_left_right_boundaries(oths):
    left_b = oths[0][0]
    right_b = oths[0][-1]
    for i in range(len(oths)):
        left_b = max(left_b, oths[i][0])
        right_b = min(right_b, oths[i][-1])
    return left_b, right_b


left_b, right_b = find_left_right_boundaries(oths)
xs = np.arange(left_b, right_b, 0.001)

mc = []
for i in range(len(oths)):
    f = interp1d(oths[i], mcs[i])
    mc.append(f(xs))

ax1 = plt.subplot(211)
plt.errorbar(xs,
             np.average(mc, axis=0),
             yerr=np.std(mc, axis=0))
plt.xlim([0.922, 1])

plt.title("change of MC during OG process", fontsize=16)
plt.yticks(fontsize=12)
plt.ylabel("memory capacity", fontsize=16)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid(True)

ax2 = plt.subplot(212)
le = []
for i in range(len(oths)):
    f = interp1d(oths[i], np.reshape(les[i], len(les[i])))
    le.append(f(xs))

plt.errorbar(xs,
             np.average(le, axis=0),
             yerr=np.std(le, axis=0))
plt.xlabel("orthogonality", fontsize=16)
plt.grid(True)
plt.ylabel("Lyapunov exponent", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([0.922, 1])
try_save_fig("figures/")
try_save_fig("figures/", ext="pdf")
plt.show()
