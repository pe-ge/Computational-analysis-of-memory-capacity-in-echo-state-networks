#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
novegon.py - OrtNOrmalization vs. orthoGONalization
Created: 13.8.2015

Goal: Compare the effects of orthogonalization with the effects of
        orthonormalization
"""

from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity

import numpy as np
import random

tau = 0.01
sigma = 0.092
N = 100
memory_max = int(N*1.2)

ORTHO_ITERATIONS = 100
TOTAL_ITERATIONS = 1
ACT_ITERATIONS = 1000
eta = 3*10**-2


def measure_mc(W, WI):
    return memory_capacity(W, WI, memory_max=memory_max, iterations=1200,
                           iterations_coef_measure=1000, use_input=False,
                           target_later=True, calc_lyapunov=False)


activations = np.zeros([N, ACT_ITERATIONS])

for it_total in range(TOTAL_ITERATIONS):
    print(it_total)
    W = np.random.normal(0, sigma, [N, N])
    WI = np.random.uniform(-tau, tau, N)
    WG = W

    for it in range(ORTHO_ITERATIONS):
        WG = learn_orthogonal(WG, eta)

    # reset reservoir
    X = np.zeros(N)
    for i in range(N):
        X = np.tanh(np.dot(W, X) + np.dot(WI, 2 * random.random() - 1))

    u = np.random.uniform(-1.0, 1.0, ACT_ITERATIONS)
    for act_it in range(ACT_ITERATIONS):
        X = np.tanh(np.dot(W, X) + np.dot(WI, u[act_it]))
        activations[:, act_it] = X

np.save('activations', activations)
