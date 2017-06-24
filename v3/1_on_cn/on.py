# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
import numpy as np
from numpy import random
from numpy.linalg import cond
from scipy.linalg import pinv
from math import inf

INSTANCES = 10
ORTHO_ITERS = 50

RHO = 0.95
TAU = 10**-10


ETA_0 = 7*10**-2
# ETA = 3*10**-2

RES_SIZES = list(range(100, 1000 + 1, 100))

cond_before = np.zeros((len(RES_SIZES), INSTANCES))
cond_after = np.zeros((len(RES_SIZES), INSTANCES))


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


def pinv_acts(WI, W, iters_skip, iters_pinv):
    # num reservoir neurons
    N = WI.shape[0]

    # reservoir activations
    X = np.zeros(N)
    # history of reservoir activations
    Xs = np.zeros([N, iters_pinv])

    # skipping (getting rid of transients)
    for it in range(iters_skip):
        X = np.tanh(np.dot(W, X) + np.dot(WI, random.uniform(-1.0, 1.0)))

    # training
    for it in range(iters_pinv):
        X = np.tanh(np.dot(W, X) + np.dot(WI, random.uniform(-1.0, 1.0)))
        Xs[:, it] = X

    # calculate output weights
    return pinv(Xs)


for N_idx, N in enumerate(RES_SIZES):
    print(N)
    for inst_idx in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (RHO / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-TAU, TAU, N)

        acts = pinv_acts(WI, W, iters_skip=N, iters_pinv=10*N)
        cond_before[N_idx, inst_idx] = cond(acts)

        eta = ETA_0
        # eta = ETA
        for it in range(ORTHO_ITERS):
            # W = learn_orthogonal(W, eta)
            W = learn_orthonormal(W, eta)
            eta = eta * 0.9

        acts = pinv_acts(WI, W, iters_skip=N, iters_pinv=10*N)
        cond_after[N_idx, inst_idx] = cond(acts)

    np.save('cond_before', cond_before)
    np.save('cond_after', cond_after)
