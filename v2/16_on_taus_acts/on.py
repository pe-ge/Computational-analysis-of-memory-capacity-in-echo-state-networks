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
import random

rho = 0.95
taus = [10**-i for i in range(10)]
# eta = 3*10**-2
eta_0 = 7*10**-2
N = 100

INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100
NUM_ACT_DATA = 100


acts_mean = np.zeros(len(taus))
acts_std = np.zeros(len(taus))

for tau_idx, tau in enumerate(taus):
    tau_mean = np.zeros(INSTANCES)
    tau_std = np.zeros(INSTANCES)
    for inst in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = np.random.uniform(-tau, tau, N)

        eta = eta_0
        for it in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
            # W = learn_orthogonal(W, eta)
            eta = eta * 0.9

        # reset reservoir
        X = np.zeros(N)
        for i in range(N):
            X = np.tanh(np.dot(W, X) + np.dot(WI, 2 * random.random() - 1))

        u = np.random.uniform(-1.0, 1.0, NUM_ACT_DATA)
        activations = np.zeros(NUM_ACT_DATA)
        stds = np.zeros(NUM_ACT_DATA)
        for act_it in range(NUM_ACT_DATA):
            X = np.tanh(np.dot(W, X) + np.dot(WI, u[act_it]))
            activations[act_it] = np.mean(X)
            stds[act_it] = np.std(X)

        print(tau_idx, activations)

        tau_mean[inst] = np.mean(activations)
        tau_std[inst] = np.mean(stds)

    acts_mean[tau_idx] = np.mean(tau_mean)
    acts_std[tau_idx] = np.mean(tau_std)
    np.save('acts_mean', acts_mean)
    np.save('acts_std', acts_std)
