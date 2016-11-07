#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random
ORTHOPROCESS_ITERATIONS = 75

rhos = np.arange(0.9, 1.01, 0.01)
taus = np.arange(0.001, 0.0101, 0.001)
eta = 3*10**-2
N = 500

def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)


mc = np.zeros([len(rhos), len(taus)])
for rho_idx, rho in enumerate(rhos):
    for tau_idx, tau in enumerate(taus):
        W = random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        mcs = np.zeros(ORTHOPROCESS_ITERATIONS)
        for mc_idx in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthogonal(W, eta)
            mcs[mc_idx], _ = memory_capacity(W, WI)
        print('rho={}, tau={}, mc={}'.format(rho, tau, max(mcs)))
        mc[rho_idx, tau_idx] = max(mcs)

np.save('mc', mc)
