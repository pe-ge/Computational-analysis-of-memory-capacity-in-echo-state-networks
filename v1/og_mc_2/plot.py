import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from library.aux import try_save_fig

oths = np.load('oths.npy')
mcs = np.load('mcs.npy')
eta = 0.05


def find_left_right_boundaries(oths):
    left_b = oths[0][0]
    right_b = oths[0][-1]
    for i in range(len(oths)):
        left_b = max(left_b, oths[i][0])
        right_b = min(right_b, oths[i][-1])
    return left_b, right_b

left_b, right_b = find_left_right_boundaries(oths)
xs = np.arange(left_b, right_b, 0.001)
yy = []
for i in range(len(oths)):
    f = interp1d(oths[i], mcs[i])
    yy.append(f(xs))

plt.errorbar(xs,
             np.average(yy, axis=0),
             yerr=np.std(yy, axis=0))
plt.xlabel("orthogonality", fontsize=24, labelpad=-3)
plt.ylabel("memory capacity", fontsize=24)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("change of MC during orthogonalization process", fontsize=24)
try_save_fig("figures/eta=" + str(eta))
try_save_fig("figures/eta=" + str(eta), ext="pdf")
plt.show()
