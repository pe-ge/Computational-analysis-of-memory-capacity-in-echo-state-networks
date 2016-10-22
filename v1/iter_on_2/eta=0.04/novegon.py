#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
novegon.py - OrtNOrmalization vs. orthoGONalization
Created: 13.8.2015

Goal: Compare the effects of orthogonalization with the effects of orthonormalization
"""

from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity

import numpy as np

tau = 0.01
rho = 0.95
N = 100
memory_max = int(N*1.2)

ORTHO_ITERATIONS = 10
TOTAL_ITERATIONS = 100
eta = 4*10**-2

legend_location = 4

def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI, memory_max=memory_max, iterations=1200,
                           iterations_coef_measure=1000, use_input=False,
                           target_later=True, calc_lyapunov=calc_lyapunov)


MCs, EVs, SVs, LEs = [], [], [], []

for it_total in range(TOTAL_ITERATIONS):
    print(it_total,'/', TOTAL_ITERATIONS)
    W = np.random.normal(0, 1, [N, N])
    W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = np.random.uniform(-tau, tau, N)
    WN = W

    mc_gon = np.zeros(ORTHO_ITERATIONS + 1)
    les = []
    mc_gon[0], le = measure_mc(WN, WI, True)
    les.append(le)

    eigen_values = np.zeros([ORTHO_ITERATIONS + 1, N])
    singular_values = np.zeros([ORTHO_ITERATIONS + 1, N])

    eigen_values[0, :] = np.sort(np.abs(np.linalg.eig(W)[0]))
    singular_values[0, :] = np.linalg.svd(W, compute_uv=False)

    for it in range(ORTHO_ITERATIONS):
        WN = learn_orthonormal(WN, eta)

        calc_lyapunov = (it + 1) % 10 == 0
        mc_gon[it + 1], le = measure_mc(WN, WI, calc_lyapunov)
        les.append(le)

        eigen_values[it + 1, :] = np.sort(np.abs(np.linalg.eig(WN)[0]))
        singular_values[it + 1, :] = np.linalg.svd(WN, compute_uv=False)

    MCs.append(mc_gon)
    EVs.append(eigen_values)
    SVs.append(singular_values)
    LEs.append(les)

print()
np.save('mc', MCs)
np.save('ev', EVs)
np.save('sv', SVs)
np.save('le', LEs)

