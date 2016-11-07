import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
from matplotlib import colors


reservoir_sizes = np.arange(150, 501, 50)

mc = np.load('mc.npy')
le = np.load('le.npy')


def replot():
    # res_size vs mc
    sp1 = plt.subplot(211)
    for i in range(mc.shape[0]):
        sp1.plot(mc[i, :-1], label=reservoir_sizes[i])
    plt.xlabel("time")
    plt.ylabel("memory capacity")
    plt.title("impact of OG method on MC")
    plt.legend(loc=4)
    sp2 = plt.subplot(212)
    for i in range(mc.shape[0]):
        sp2.plot(le[i, :-1], label=reservoir_sizes[i])

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
