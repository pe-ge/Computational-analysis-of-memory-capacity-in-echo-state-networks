import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from matplotlib import colors


reservoir_sizes = [150, 175, 200]
rhos = np.arange(0.95, 1, 0.01)
cols = ['red', 'green', 'blue', 'black', 'yellow', 'purple']

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')

mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')


def replot():
    # res_size vs mc
    plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
    for i in range(mcb_mean.shape[0]):
        plt.errorbar(reservoir_sizes, mcb_mean[i], yerr=mcb_std[i], label=rhos[i], fmt="--", color=cols[i])
    for i in range(mcb_mean.shape[0]):
        plt.errorbar(reservoir_sizes, mca_mean[i], yerr=mcb_std[i], color=cols[i])
    plt.ylabel("memory capacity")
    plt.title("impact of OG method on MC")
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlabel("reservoir size")
    # plt.xlim([9,101])

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
