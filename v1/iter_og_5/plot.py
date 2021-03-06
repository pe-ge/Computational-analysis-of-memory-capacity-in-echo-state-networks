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
    for mc in MCs:
        plt.plot(xs, mc)
    plt.grid(True)
    # plt.yticks([i for i in range(20, 102, 20)])
    plt.ylabel("MC", size=20)
    plt.ylim((20, 102))

    plt.setp(ax1.get_xticklabels(), visible=False)

    # eigenvalues
    ax2 = plt.subplot(412, sharex=ax1)
    for ev in EVs:
        plt.plot(xs, np.amax(ev, axis=1))

    plt.grid(True)
#    plt.ylabel("max(abs(eigenvalues))")
    plt.ylabel(r'$\rho$', size=26)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # singular values
    ax3 = plt.subplot(413, sharex=ax1)
    for sv in SVs:
        plt.plot(xs, np.amax(sv, axis=1))

    plt.grid(True)
    plt.ylabel(r'$s_{max}$', size=26)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # lyapunov
    ax4 = plt.subplot(414, sharex=ax1)
    for le in LEs:
        plt.plot(xs, le)

    plt.grid(True)
    plt.ylabel('LE', size=26)

    plt.xlim((0, 60))
    plt.xlabel("iteration", size=20)

    try_save_fig("figures/eta=" + str(eta))
    try_save_fig("figures/eta=" + str(eta), ext="pdf")
    plt.show()

replot()
