#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortoven.py - Orthogonalization vs. N
Created 5.8.2015

Goal: Measure the effect of orthogonalization process, depending on N.
"""

from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random


INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100

rho = 0.95
tau = 0.01  # previously 0.001
eta = 3*10**-2

reservoir_sizes = list(range(10, 100 + 1, 10))  # [16, 25, 64, 100]


def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=calc_lyapunov)


mcb_mean = np.zeros(len(reservoir_sizes))
mcb_std = np.zeros(len(reservoir_sizes))
mcb_les_mean = np.zeros(len(reservoir_sizes))
mcb_les_std = np.zeros(len(reservoir_sizes))

mca_mean = np.zeros(len(reservoir_sizes))
mca_std = np.zeros(len(reservoir_sizes))
mca_les_mean = np.zeros(len(reservoir_sizes))
mca_les_std = np.zeros(len(reservoir_sizes))

for rsi, N in enumerate(reservoir_sizes):
    print(N)
    mc_before = np.zeros(INSTANCES)
    mc_after = np.zeros(INSTANCES)
    le_before = np.zeros(INSTANCES)
    le_after = np.zeros(INSTANCES)
    for inst in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        mc_before[inst], le_before[inst] = measure_mc(W, WI, True)

        for _ in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthogonal(W, eta)

        mc_after[inst], le_after[inst] = measure_mc(W, WI, True)

    mcb_mean[rsi] = np.average(mc_before)
    mcb_std[rsi] = np.std(mc_before)
    mcb_les_mean[rsi] = np.average(le_before)
    mcb_les_std[rsi] = np.std(le_before)

    mca_mean[rsi] = np.average(mc_after)
    mca_std[rsi] = np.std(mc_after)
    mca_les_mean[rsi] = np.average(le_after)
    mca_les_std[rsi] = np.std(le_after)

np.save('mcbm', mcb_mean)
np.save('mcbs', mcb_std)
np.save('mcblm', mcb_les_mean)
np.save('mcbls', mcb_les_std)

np.save('mcam', mca_mean)
np.save('mcas', mca_std)
np.save('mcalm', mca_les_mean)
np.save('mcals', mca_les_std)
