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
INSTANCES = 5
rho = 0.95
tau = 10**-9
reservoir_sizes = list(range(100, 1000 + 1, 100))

mcb_mean = np.zeros(len(reservoir_sizes))
mcb_std = np.zeros(len(reservoir_sizes))

for rsi, N in enumerate(reservoir_sizes):
    print(N)
    mc_before = np.zeros(INSTANCES)
    for inst in range(INSTANCES):
        W = random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        mc_before[inst], _ = measure_mc(W, WI, 10*N)

    mcb_mean[rsi] = np.average(mc_before)
    mcb_std[rsi] = np.std(mc_before)

np.save('mcbm', mcb_mean)
np.save('mcbs', mcb_std)

mca_mean = np.load('mcm.npy')
mca_std = np.load('mcs.npy')


plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
plt.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ", fmt="--")
plt.errorbar(reservoir_sizes, mca_mean[:, -1], yerr=mca_std[:, -1], label="after")
plt.ylabel("MC")
plt.title("ON method")
plt.grid(True)

plt.xlabel("reservoir size")

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()

