import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

TARGET_SP_RADIUS = 0.95  # before 0.9
INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100
rho = TARGET_SP_RADIUS
tau = 0.01  # previously 0.001
eta = 3*10**-2  # learning rate, 1*10**-1 je uz privela
reservoir_sizes = list(range(100, 1000 + 1, 100))  # [16, 25, 64, 100]
# reservoir_sizes = list(range(10, 100 + 1, 10))  # [16, 25, 64, 100]

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')
leb_mean = np.load('lbm.npy')
leb_std = np.load('lbs.npy')

mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')
lea_mean = np.load('lam.npy')
lea_std = np.load('las.npy')


def replot():
    # res_size vs mc
    ax1 = plt.subplot(211)
    ax1.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
    ax1.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ", fmt="--")
    ax1.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
    plt.ylabel("MC")
    plt.title("ON method")
    plt.grid(True)

    ax2 = plt.subplot(212)
    ax2.errorbar(reservoir_sizes, leb_mean, yerr=leb_std, label="before ", fmt="--")
    ax2.errorbar(reservoir_sizes, lea_mean, yerr=lea_std, label="after")
    plt.ylabel("LE")
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlabel("reservoir size")
    plt.yticks(np.arange(0.0, -0.11, -0.02))

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
