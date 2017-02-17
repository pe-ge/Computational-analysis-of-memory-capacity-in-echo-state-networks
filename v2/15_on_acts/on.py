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

tau = 10**-10
rho = 0.95
reservoir_sizes = list(range(100, 1000 + 1, 100))
eta_0 = 7*10**-2

ORTHOPROCESS_ITERATIONS = 50
NUM_ACT_DATA = 100
INSTANCES = 10


acts_mean = np.zeros(len(reservoir_sizes))
acts_std = np.zeros(len(reservoir_sizes))

for N_idx, N in enumerate(reservoir_sizes):
    print(N)
    instances = np.zeros(INSTANCES)
    instances_stds = np.zeros(INSTANCES)
    for inst in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = np.random.uniform(-tau, tau, N)

        eta = eta_0
        for it in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
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

        instances[inst] = np.mean(activations)
        instances_stds[inst] = np.mean(stds)

    acts_mean[N_idx] = np.mean(instances)
    acts_std[N_idx] = np.mean(instances_stds)

    np.save('acts_mean', acts_mean)
    np.save('acts_std', acts_std)
