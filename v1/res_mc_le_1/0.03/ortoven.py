#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortoven.py - Orthogonalization vs. N
Created 5.8.2015

Goal: Measure the effect of orthogonalization process, depending on N.
"""

from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity
from library.aux import try_save_fig
from library.lyapunov import lyapunov_exp

import numpy as np
from numpy import random
from matplotlib import pyplot as plt

from time import time

TARGET_SP_RADIUS = 0.95  # before 0.9
INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100

tau = 0.01  # previously 0.001
eta = 3*10**-2  # learning rate, 1*10**-1 je uz privela

reservoir_sizes = list(range(10, 100 + 1, 10))  # [16, 25, 64, 100]


def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=calc_lyapunov)

rho = TARGET_SP_RADIUS

mc_before = np.zeros(INSTANCES)
mc_after = np.zeros(INSTANCES)

mcb_mean = np.zeros(len(reservoir_sizes))
mcb_std = np.zeros(len(reservoir_sizes))
mca_mean = np.zeros(len(reservoir_sizes))
mca_std = np.zeros(len(reservoir_sizes))
mcb_les = []
mca_les = []

ttotal = time()

for rsi, N in enumerate(reservoir_sizes):
    for inst in range(INSTANCES):
        W = random.normal(0, 1, [N, N])
        WI = random.uniform(-tau, tau, N)
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))

        calc_lyapunov = (inst + 1) % 10 == 0

        mc_before[inst], le = measure_mc(W, WI, calc_lyapunov)
        if calc_lyapunov:
            mcb_les.append(le)

        for _ in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthogonal(W, eta)

        mc_after[inst], le = measure_mc(W, WI, calc_lyapunov)
        if calc_lyapunov:
            mca_les.append(le)
    mcb_mean[rsi] = np.average(mc_before)
    mca_mean[rsi] = np.average(mc_after)

    mcb_std[rsi] = np.std(mc_before)
    mca_std[rsi] = np.std(mc_after)

np.save('mcbm', mcb_mean)
np.save('mcbs', mcb_std)
np.save('mcbl', mcb_les)
np.save('mcam', mca_mean)
np.save('mcas', mca_std)
np.save('mcal', mca_les)
