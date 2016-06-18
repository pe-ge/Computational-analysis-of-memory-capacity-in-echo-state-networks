#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
novegon.py - OrtNOrmalization vs. orthoGONalization
Created: 13.8.2015

Goal: Compare the effects of orthogonalization with the effects of
        orthonormalization
"""

from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity

import numpy as np

tau = 0.01
sigma = 0.092
N = 100
memory_max = int(N*1.2)

ORTHO_ITERATIONS = 100
TOTAL_ITERATIONS = 100


def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI, memory_max=memory_max, iterations=1200,
                           iterations_coef_measure=1000, use_input=False,
                           target_later=True, calc_lyapunov=calc_lyapunov)


smoothness = 10
etas = np.linspace(0.01, 0.1, smoothness)
decays = np.linspace(0.91, 1, smoothness)

MCs = np.zeros([smoothness, smoothness, TOTAL_ITERATIONS])
for eta_idx, eta_0 in enumerate(etas):
    for decay_idx, decay in enumerate(decays):
        print(eta_0, decay)
        for it_total in range(TOTAL_ITERATIONS):
            WG = np.random.normal(0, sigma, [N, N])
            WI = np.random.uniform(-tau, tau, N)

            eta = eta_0
            for it in range(ORTHO_ITERATIONS):
                WG = learn_orthonormal(WG, eta)
                eta = eta * decay

            MCs[eta_idx, decay_idx, it_total], _ = measure_mc(WG, WI, False)

print()
np.save('mc', MCs)
