import numpy as np
from matplotlib import pyplot as plt

rho = 0.95
tau = 10**-9
reservoir_sizes = list(range(100, 1000 + 1, 100))

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')
mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')
plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
plt.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ", fmt="--")
plt.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
plt.ylabel("MC")
plt.title("ON method")
plt.grid(True)
plt.legend(loc=2)

plt.xlabel("reservoir size")

plt.savefig('ON_N_vs_MC.png')
plt.show()

