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


INSTANCES = 10
ORTHOPROCESS_ITERATIONS = 100

rho_sizes = np.arange(0.95, 1, 0.01)
tau = 0.01  # previously 0.001
eta = 3*10**-2

reservoir_sizes = [150, 175, 200]


def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)

mcb_mean = np.zeros([len(rho_sizes), len(reservoir_sizes)])
mcb_std = np.zeros([len(rho_sizes), len(reservoir_sizes)])

mca_mean = np.zeros([len(rho_sizes), len(reservoir_sizes)])
mca_std = np.zeros([len(rho_sizes), len(reservoir_sizes)])

for N_ix, N in enumerate(reservoir_sizes):
    print(N)
    for rho_ix, rho in enumerate(rho_sizes):
        mc_before = np.zeros(INSTANCES)
        mc_after = np.zeros(INSTANCES)
        for inst in range(INSTANCES):
            W = random.normal(0, 1, [N, N])
            W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
            WI = random.uniform(-tau, tau, N)

            mc_before[inst], _ = measure_mc(W, WI)

            for _ in range(ORTHOPROCESS_ITERATIONS):
                W = learn_orthogonal(W, eta)

            mc_after[inst], _ = measure_mc(W, WI)

        mcb_mean[rho_ix, N_ix] = np.average(mc_before)
        mcb_std[rho_ix, N_ix] = np.std(mc_before)

        mca_mean[rho_ix, N_ix] = np.average(mc_after)
        mca_std[rho_ix, N_ix] = np.std(mc_after)

np.save('mcbm', mcb_mean)
np.save('mcbs', mcb_std)

np.save('mcam', mca_mean)
np.save('mcas', mca_std)
