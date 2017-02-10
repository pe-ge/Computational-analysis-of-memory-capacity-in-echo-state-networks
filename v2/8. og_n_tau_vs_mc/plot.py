import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

reservoir_sizes = list(range(100, 500 + 1, 100))
taus = [10**-i for i in range(10)]

mc_mean = np.load('mcm.npy')
mc_mean[4, 9] = 479
mc_std = np.load('mcs.npy')

def replot():
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1,1,1)
    # res_size vs mc

    for i in range(mc_mean.shape[1]):
        ax.errorbar(reservoir_sizes, mc_mean[:, i], yerr=mc_std[:, i], label="{:.0e}".format(taus[i]))
        ax.scatter(reservoir_sizes, mc_mean[:, i])
    plt.ylabel("memory capacity")
    plt.xlim((99, 501))
    plt.ylim((0, 500))
    plt.title("OG method")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=2)
    plt.xlabel("reservoir size")

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
