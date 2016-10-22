import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

ORTHO_ITERATIONS = 100
legend_location = 4
eta = 0.03

MCs = np.load('mc.npy')
EVs = np.load('ev.npy')
SVs = np.load('sv.npy')
LEs = np.load('le.npy')

def replot():
    xs = range(ORTHO_ITERATIONS + 1)
    ax1 = plt.subplot(411)
    ax1.errorbar(
        xs,
        np.average(MCs, axis=0),
        yerr=np.std(MCs, axis=0))
    plt.grid(True)
    plt.ylabel("memory capacity")

    plt.setp(ax1.get_xticklabels(), visible=False)

    # eigenvalues
    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(xs, np.average(EVs, axis=0))

    plt.grid(True)
    plt.ylabel("abs(eigenvalues)")
    plt.setp(ax2.get_xticklabels(), visible=False)

    # singular values
    plt.subplot(413, sharex=ax1)
    plt.plot(xs, np.average(SVs, axis=0))

    plt.grid(True)
    plt.ylabel("singular value")

    # lyapunov
    ax4 = plt.subplot(414, sharex=ax1)
    ax4.errorbar(
        xs,
        np.average(LEs, axis=0),
        yerr=np.std(LEs, axis=0))
    plt.grid(True)
    plt.ylabel("lyapunov exponent")

    plt.xlabel("iteration")
    try_save_fig("figures/eta=" + str(eta))
    try_save_fig("figures/eta=" + str(eta), ext="pdf")
    plt.show()

replot()
