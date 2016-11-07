#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emceorto.py  -MC vs. orthogonality (second)
Created 9.4.2015
Based on mvo.py

"""

from library.ortho import learn_orthogonal, orthogonality
from library.mc6 import memory_capacity

import numpy as np
from numpy import random, save

sigma = 0.092
tau = 0.01
q = 100
eta = 3*10**-2

ITERATIONS = 100

LINES = 100


def measure_mc(W, WI):
    return memory_capacity(W, WI, memory_max=150, iterations=1000,
                           iterations_coef_measure=1000,
                           use_input=False, target_later=True,
                           calc_lyapunov=True)


def measure_og(W):
    return orthogonality(W)

OTHS = np.zeros([LINES, ITERATIONS + 1])
LES = np.zeros([LINES, ITERATIONS + 1])
for l in range(LINES):
    W = random.normal(0, sigma, [q, q])
    WI = random.uniform(-tau, tau, q)

    mc, le = measure_mc(W, WI)
    print(le)
    og = measure_og(W)

    OTHS[l, 0] = og
    LES[l, 0] = le

    for it in range(ITERATIONS):
        W = learn_orthogonal(W, eta)

        mc, le = measure_mc(W, WI)
        og = measure_og(W)

        OTHS[l, it + 1] = og
        LES[l, it + 1] = le

save('oths', OTHS)
save('les', LES)
