# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
import numpy as np
from scipy.linalg import pinv
from numpy import random
from math import inf

INSTANCES = 5

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


def train_model(N, sparsity, eta):
    W = np.random.normal(0, 1, [N, N])
    W = W * (RHO / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = random.uniform(-TAU, TAU, N)

    # make sparse
    sparsity = 0.99
    for i in range(N):
        for j in range(N):
            if i != j and random.random() < sparsity:
                W[i, j] = 0.0
    W = W * (RHO / np.max(np.abs(np.linalg.eig(W)[0])))
    print(1.0 - np.count_nonzero(W) / (N * N))

    mc_max = -inf

    W = learn_orthonormal(W, eta)
    while True:
        print(1.0 - np.count_nonzero(W) / (N * N))
        W = learn_orthonormal(W, eta)
        last_mc = mc(WI, W, iters_skip=N, iters_train=10*N, iters_test=1000)
        if mc_max < last_mc:
            mc_max = last_mc

        if mc_max > last_mc:
            return W


N = 200
sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]
etas = [i*10**-2 for i in range(1, 11)]

sps = np.zeros([len(sparsities), len(etas), INSTANCES])

for spars_idx, sparsity in enumerate(sparsities):
    for eta_idx, eta in enumerate(etas):
        for inst in range(INSTANCES):
            print(sparsity, eta, inst)
            W = train_model(N, sparsity, eta)
            sps[spars_idx, eta_idx, inst] = 1.0 - np.count_nonzero(W) / (N * N)
            np.save('sps', sps)
