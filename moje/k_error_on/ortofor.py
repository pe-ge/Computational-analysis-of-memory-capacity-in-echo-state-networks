#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortofor.py - orthogonalization impact on forgetting curve
Created: 13.8.2015

Goal: Investigate how the orthogonalization process
    changes forgetting curves (FC).
    Compare FC before and after orthogonalization
"""

from library.ortho import learn_orthonormal
from library.mc6forget import memory_capacity
import numpy as np

sigma = 0.092
tau = 0.01
reservoir_sizes = np.arange(10, 101, 10)
INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100
eta_0 = 3*10**-2

means = []
stds = []


def measure_mc(W, WI, memory_max):
    return memory_capacity(W, WI,
                           memory_max,
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True)


# for rsi, N in enumerate(reservoir_sizes):
N = 100
memory_max = int(1.4*N)  # before 1.2

for inst in range(INSTANCES):
    WI = np.random.uniform(-tau, tau, N)
    W = np.random.normal(0, sigma, [N, N])

    eta = eta_0
    for _ in range(ORTHOPROCESS_ITERATIONS):
        W = learn_orthonormal(W, eta)
        eta = 0.95 * eta

    mean, std = measure_mc(W, WI, memory_max)
    means.append(mean)
    stds.append(std)

np.save('means', means)
np.save('stds', stds)
