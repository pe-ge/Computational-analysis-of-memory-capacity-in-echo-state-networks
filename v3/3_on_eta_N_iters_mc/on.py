# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
import numpy as np
from scipy.linalg import pinv
from numpy import random
from math import inf

INSTANCES = 10

RHO = 0.95
TAU = 10**-10


def mc(WI, W, iters_skip, iters_train, iters_test):
    # num output neurons
    N = WI.shape[0]
    L = N
    total_its = iters_skip + iters_train + iters_test

    # input
    u = np.random.uniform(-1.0, 1.0, total_its)
    # desired outputs
    D = np.zeros([L, iters_train])
    # reservoir activations
    X = np.zeros(N)
    # history of reservoir activations
    Xs = np.zeros([N, iters_train])

    # skipping (getting rid of transients)
    for it in range(iters_skip):
        X = np.tanh(np.dot(W, X) + np.dot(WI, u[it]))

    # training
    for it in range(iters_train):
        X = np.tanh(np.dot(W, X) + np.dot(WI, u[iters_skip + it]))
        Xs[:, it] = X
        D[:, it] = u[iters_skip + it:it:-1]

    # calculate output weights
    Xs_pinv = pinv(Xs)
    WO = np.dot(D, Xs_pinv)

    # testing
    y = np.zeros([L, iters_test])
    for it in range(iters_test):
        X = np.tanh(np.dot(W, X) + np.dot(WI, u[iters_skip + iters_train + it]))
        y[:, it] = np.dot(WO, X)

    # memory capacity
    mc = np.zeros(L)
    for h in range(L):
        cc = np.corrcoef(u[iters_skip + iters_train - h: total_its - h], y[h, :])[0, 1]
        mc[h] = cc * cc

    return np.sum(mc)


def train_model(N, eta):
    W = np.random.normal(0, 1, [N, N])
    W = W * (RHO / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = random.uniform(-TAU, TAU, N)

    mc_max = -inf
    opt_iters = 0

    while True:
        # W = learn_orthogonal(W, eta)
        W = learn_orthonormal(W, eta)
        last_mc = mc(WI, W, iters_skip=N, iters_train=10*N, iters_test=1000)
        print('N={}, eta={}, mc={}'.format(N, eta, last_mc))
        if mc_max < last_mc:
            mc_max = last_mc

        if mc_max > last_mc:
            print()
            return mc_max, opt_iters

        opt_iters += 1


etas = np.linspace(0.01, 0.1, 10)
Ns = np.arange(100, 1001, 100)

mcs = np.zeros([len(Ns), len(etas), INSTANCES])
iters = np.zeros([len(Ns), len(etas), INSTANCES])

for N_idx, N in enumerate(Ns):
    for eta_idx, eta in enumerate(etas):
        for inst in range(INSTANCES):
            mc_max, opt_iters = train_model(N, eta)
            mcs[N_idx, eta_idx, inst] = mc_max
            iters[N_idx, eta_idx, inst] = opt_iters

np.save('mcs', mcs)
np.save('iters', iters)
