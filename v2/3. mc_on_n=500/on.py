#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random
INSTANCES = 1
ORTHOPROCESS_ITERATIONS = 100

rho = 0.99
tau = 0.01  # previously 0.001
eta_0 = 7*10**-2

reservoir_sizes = np.arange(150, 501, 50)


def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=True)

mc = np.zeros([len(reservoir_sizes), ORTHOPROCESS_ITERATIONS + 1])
le = np.zeros([len(reservoir_sizes), ORTHOPROCESS_ITERATIONS + 1])

for N_ix, N in enumerate(reservoir_sizes):
    print(N)
    for inst in range(INSTANCES):
        W = random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        mc[N_ix, 0], le[N_ix, 0] = measure_mc(W, WI)

        eta = eta_0
        for ix in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
            eta = eta * 0.9
            mc[N_ix, ix], le[N_ix, ix] = measure_mc(W, WI)

np.save('mc', mc)
np.save('le', le)
