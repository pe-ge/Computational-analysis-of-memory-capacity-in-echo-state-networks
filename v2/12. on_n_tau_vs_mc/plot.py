import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

reservoir_sizes = list(range(100, 1000 + 1, 100))
taus = [10**-i for i in range(20)]

mc_mean = np.load('mcm.npy')
mc_std = np.load('mcs.npy')

print(mc_mean)

def replot():
    ax = plt.subplot(1,1,1)
    # res_size vs mc

    for i in range(mc_mean.shape[1]):
        ax.errorbar(reservoir_sizes, mc_mean[:, i], yerr=mc_std[:, i], label="{:.0e}".format(taus[i]))
        ax.scatter(reservoir_sizes, mc_mean[:, i])
    plt.ylabel("memory capacity")
    plt.xlim((90, 1010))
    plt.xticks([i * 100 for i in range(1, 11)])
    plt.ylim((0, 800))
    plt.title("ON method")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=2)
    plt.xlabel("reservoir size")

    plt.savefig('ON_N_tau_vs_MC.png')
    plt.show()

replot()
