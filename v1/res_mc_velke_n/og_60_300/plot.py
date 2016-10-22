import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

TARGET_SP_RADIUS = 0.95  # before 0.9
INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100
rho = TARGET_SP_RADIUS
tau = 0.01  # previously 0.001
eta = 3*10**-2  # learning rate, 1*10**-1 je uz privela
# reservoir_sizes = list(range(200, 500 + 1, 100))  # [16, 25, 64, 100]
reservoir_sizes = list(range(60, 300 + 1, 20))  # [16, 25, 64, 100]

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')
mcb_l_mean = np.load('mcblm.npy')
mcb_l_std = np.load('mcbls.npy')

mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')
mca_l_mean = np.load('mcalm.npy')
mca_l_std = np.load('mcals.npy')


def replot():
    # res_size vs mc
    plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
    plt.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ", fmt="--")
    plt.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
    plt.ylabel("memory capacity")
    plt.title("impact of OG method on MC".format(tau, rho))
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlabel("reservoir size")
    # plt.xlim([9,101])

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
