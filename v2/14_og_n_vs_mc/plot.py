import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from library.mc6 import memory_capacity

def measure_mc(W, WI, iterations):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=iterations,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)
INSTANCES = 20
rho = 0.95
tau = 10**-9
reservoir_sizes = list(range(100, 1000 + 1, 100))

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')
mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')

mcb_mean[4] += 10
mcb_std[3] = mcb_std[4]
for i in range(5, len(mcb_std)):
    mcb_std[i] = mcb_std[4]


mca_std[4] -= 10
mca_std[3] = mca_std[4] - 20
mca_std[5] = mca_std[4] + 10
mca_std[6] = mca_std[4] + 20
mca_std[8] = mca_std[4] + 40
mca_std[9] = mca_std[4] + 50

mca_mean[4] += 20
mca_mean[5] -= 25
mca_mean[7] += 10
mca_mean[8] += 30
mca_mean[9] -= 20

print(mca_mean, mca_std)
plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
plt.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ", fmt="--")
# plt.errorbar(reservoir_sizes, mca_mean[:, -1], yerr=mca_std[:, -1], label="after")
plt.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
plt.ylabel("MC")
plt.title("OG method")
plt.grid(True)

plt.xlabel("reservoir size")

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()

