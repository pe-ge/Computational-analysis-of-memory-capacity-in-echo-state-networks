#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random
ORTHOPROCESS_ITERATIONS = 75

etas = np.arange(0.001, 0.05, 0.002)
N = 500
tau = 0.001
rho = 0.99

def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)


mc = np.zeros([len(etas)])
for eta_idx, eta in enumerate(etas):
    print(eta)
    W = random.normal(0, 1, [N, N])
    W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = random.uniform(-tau, tau, N)

    mcs = np.zeros(ORTHOPROCESS_ITERATIONS)
    for mc_idx in range(ORTHOPROCESS_ITERATIONS):
        W = learn_orthogonal(W, eta)
        mcs[mc_idx], _ = memory_capacity(W, WI)
    mc[eta_idx] = max(mcs)

np.save('mc', mc)
