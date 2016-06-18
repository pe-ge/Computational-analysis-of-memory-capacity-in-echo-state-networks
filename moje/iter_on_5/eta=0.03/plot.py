import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

ORTHO_ITERATIONS = 100
legend_location = 4
eta = 0.03

MCs = np.load('mc.npy')
EVs = np.load('ev.npy')
SVs = np.load('sv.npy')

def replot():
    xs = range(ORTHO_ITERATIONS + 1)
    ax1 = plt.subplot(311)
    # for mc in MCs:
        # plt.plot(xs, mc)
    ax1.errorbar(xs,
                 np.average(MCs, axis=0),
                 yerr=np.std(MCs, axis=0))
    plt.grid(True)
    # plt.yticks([i for i in range(20, 102, 20)])
    plt.ylabel("MC", size=20)
    plt.ylim((30, 102))

    plt.setp(ax1.get_xticklabels(), visible=False)

    # eigenvalues
    ax2 = plt.subplot(312, sharex=ax1)
    e = []
    for ev in EVs:
        e.append(np.amax(ev, axis=1))
    # for ev in EVs:
        # plt.plot(xs, np.amax(ev, axis=1))

    ax2.errorbar(xs,
             np.average(e, axis=0),
             yerr=np.std(e, axis=0))

    plt.grid(True)
#    plt.ylabel("max(abs(eigenvalues))")
    plt.ylabel(r'$\rho$', size=26)
    plt.setp(ax2.get_xticklabels(), visible=False)
    # e = []
    # ax2 = plt.subplot(312, sharex=ax1)
    # for ev in EVs:
        # plt.plot(xs, np.amax(ev, axis=1))
    # for ev in EVs:
        # e.append(np.amax(ev, axis=1))
    # # ax2.errorbar(xs,
             # # np.average(e, axis=0),
             # # yerr=np.std(e, axis=0))
#
    # # plt.yticks(np.arange(0.8, 1.01, 0.05))
    # plt.grid(True)
    # plt.ylabel(r'$\rho$', size=26)
    # plt.yticks(np.arange(0.76, 0.94, 0.04))
    # plt.setp(ax2.get_xticklabels(), visible=False)

    # singular values
    ax3 = plt.subplot(313, sharex=ax2)
    s = []
    # for sv in SVs:
        # plt.plot(xs, np.amax(sv, axis=1))
    for sv in SVs:
        s.append(np.amax(sv, axis=1))
    ax3.errorbar(xs,
             np.average(s, axis=0),
             yerr=np.std(s, axis=0))

    plt.grid(True)
    plt.ylabel(r'$s_{max}$', size=26)
    plt.ylim((0.95, 1.8))
    # plt.yticks(np.arange(1.0, 2.0, 0.2))

    plt.xlim((0, 60))
    plt.xlabel("iteration", size=20)

    try_save_fig("figures/eta=" + str(eta))
    try_save_fig("figures/eta=" + str(eta), ext="pdf")
    plt.show()

replot()
