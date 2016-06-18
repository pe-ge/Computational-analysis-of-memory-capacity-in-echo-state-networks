#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from library.mc6 import memory_capacity
from library.ortho import orthogonality

import numpy as np
from numpy import random


INSTANCES = 100

rho = 0.8  # spectral radius
sigma = 0.08
tau = 0.01  # previously 0.001
eta = 3*10**-2

reservoir_sizes = list(range(10, 101, 10))


def gs(W, WI, N):
    X = np.append(W, WI).reshape(N + 1, N)
    Q, R = np.linalg.qr(X)
    new_W = Q[0:N]
    new_WI = Q[N]
    return new_W, new_WI


def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)


mcb_mean = np.zeros(len(reservoir_sizes))
mcb_std = np.zeros(len(reservoir_sizes))
orthob_mean = np.zeros(len(reservoir_sizes))
orthob_std = np.zeros(len(reservoir_sizes))

mca_mean = np.zeros(len(reservoir_sizes))
mca_std = np.zeros(len(reservoir_sizes))
orthoa_mean = np.zeros(len(reservoir_sizes))
orthoa_std = np.zeros(len(reservoir_sizes))

for rsi, N in enumerate(reservoir_sizes):
    print(N)
    mc_before = np.zeros(INSTANCES)
    mc_after = np.zeros(INSTANCES)
    ob = np.zeros(INSTANCES)
    oa = np.zeros(INSTANCES)
    for inst in range(INSTANCES):
        WI = np.random.uniform(-tau, tau, N)
        # W = np.random.normal(0, 1, [N, N])
        # W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        W = random.normal(0, sigma, [N, N])

        mc_before[inst], _ = measure_mc(W, WI)
        ob[inst] = orthogonality(W)

        W, WI = gs(W, WI, N)

        mc_after[inst], _ = measure_mc(W, WI)
        oa[inst] = orthogonality(W)

    mcb_mean[rsi] = np.average(mc_before)
    mcb_std[rsi] = np.std(mc_before)
    orthob_mean[rsi] = np.average(ob)
    orthob_std[rsi] = np.std(ob)

    mca_mean[rsi] = np.average(mc_after)
    mca_std[rsi] = np.std(mc_after)
    orthoa_mean[rsi] = np.average(oa)
    orthoa_std[rsi] = np.std(oa)

np.save('mcbm', mcb_mean)
np.save('mcbs', mcb_std)
np.save('obm', orthob_mean)
np.save('obs', orthob_std)

np.save('mcam', mca_mean)
np.save('mcas', mca_std)
np.save('oam', orthoa_mean)
np.save('oas', orthoa_std)
