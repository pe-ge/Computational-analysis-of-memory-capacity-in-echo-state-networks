from library.ortho import learn_orthogonal
# from library.ortho import learn_orthonormal
import numpy as np
from scipy.linalg import pinv
from numpy import random

INSTANCES = 10
ORTHO_ITERS = 20

RHO = 0.95
TAU = 10**-10

N = 1000


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


smoothness = 20
etas = np.linspace(0.005, 0.1, smoothness)

MCs = np.zeros([smoothness, INSTANCES])
for eta_idx, eta in enumerate(etas):
    print(eta)
    for inst in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (RHO / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-TAU, TAU, N)

        for it in range(ORTHO_ITERS):
            W = learn_orthogonal(W, eta)

        MCs[eta_idx, inst] = mc(WI, W, iters_skip=N, iters_train=10*N, iters_test=1000)

np.save('mc', MCs)
