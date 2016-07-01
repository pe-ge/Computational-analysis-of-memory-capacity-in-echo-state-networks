#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
from numpy.linalg import norm
from library.ortho import learn_orthogonal, learn_orthonormal, orthogonality
from library.mc6 import memory_capacity

sigma = 0.092
tau = 0.01
N = 100
eta = 7*10**-2

ITERATIONS = 100


def measure_spectral_angle(wi, wj):
    if (wi.shape[0] != 100 or wj.shape[0] != 100):
        raise ValueError('not columns')
    return np.dot(np.transpose(wi), wj) / (norm(wi) * norm(wj))

def measure_mc(W, WI):
    return memory_capacity(W, WI, memory_max=150, iterations=1000,
                           iterations_coef_measure=1000,
                           use_input=False, target_later=True,
                           )


def set_spectral_angles(spectral_angles, iteration, W):
    angle_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            spectral_angles[iteration, angle_idx] = measure_spectral_angle(W[:, i], W[:, j])
            angle_idx += 1


def measure_og(W):
        return orthogonality(W)


spectral_angles = np.zeros([ITERATIONS + 1, N * (N - 1) / 2])
W = random.normal(0, sigma, [N, N])
WI = random.uniform(-tau, tau, N)

# calculate spectral angles before orthogonalization
set_spectral_angles(spectral_angles, 0, W)

e = eta
for iteration in range(1, ITERATIONS + 1):
    print(iteration)
    # W = learn_orthogonal(W, eta)
    W = learn_orthonormal(W, e)
    set_spectral_angles(spectral_angles, iteration, W)
    e = e * 0.9

np.save('spectral_angles', spectral_angles)
