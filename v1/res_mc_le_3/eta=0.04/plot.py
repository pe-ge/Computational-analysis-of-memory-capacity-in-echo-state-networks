import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

TARGET_SP_RADIUS = 0.95  # before 0.9
INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100
rho = TARGET_SP_RADIUS
tau = 0.01  # previously 0.001
eta = 3*10**-2  # learning rate, 1*10**-1 je uz privela
reservoir_sizes = list(range(10, 100 + 1, 10))  # [16, 25, 64, 100]

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')
mcb_lap = np.load('mcbl.npy')
mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')
mca_lap = np.load('mcal.npy')


def get_mean_std(a):
    ym = []
    ys = []
    for i in range(np.size(a) // 10):
        pos = i * 10
        le = a[pos:pos + 10]
        ym.append(np.average(le))
        ys.append(np.std(le))

    return ym, ys


def replot():
    # res_size vs mc
    ax1 = plt.subplot(211)
    ax1.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle="--")
    ax1.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ")
    ax1.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
    plt.ylabel("memory capacity")
    plt.title("impact of orthogonalization process on MC".format(tau, rho))
    plt.grid(True)
    plt.legend(loc=2)

    # res_size vs lyap
    ax2 = plt.subplot(212)

    mcb_le_mean, mcb_le_std = get_mean_std(mcb_lap)
    ax2.errorbar(reservoir_sizes, mcb_le_mean, yerr=mcb_le_std, label="before")
    mca_le_mean, mca_le_std = get_mean_std(mca_lap)
    ax2.errorbar(reservoir_sizes, mca_le_mean, yerr=mca_le_std, label="after")
    plt.xlabel("reservoir size")
    plt.ylabel("lyapunov exponent")
    plt.grid(True)
    ax2.legend(loc=2)

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
