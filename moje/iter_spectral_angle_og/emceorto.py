#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
from numpy.linalg import norm
from library.ortho import learn_orthogonal

sigma = 0.092
tau = 0.01
N = 100
eta = 3*10**-2

ITERATIONS = 100


def measure_spectral_angle(wi, wj):
    if (wi.shape[0] != 100 or wj.shape[0] != 100):
        raise ValueError('not columns')
    return np.dot(np.transpose(wi), wj) / (norm(wi) * norm(wj))


def set_spectral_angles(spectral_angles, iteration, W):
    angle_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            spectral_angles[iteration, angle_idx] = measure_spectral_angle(W[:, i], W[:, j])
            angle_idx += 1

spectral_angles = np.zeros([ITERATIONS + 1, N * (N - 1) / 2])
W = random.normal(0, sigma, [N, N])
WI = random.uniform(-tau, tau, N)

# calculate spectral angles before orthogonalization
set_spectral_angles(spectral_angles, 0, W)

for iteration in range(1, ITERATIONS + 1):
    print(iteration)
    W = learn_orthogonal(W, eta)
    set_spectral_angles(spectral_angles, iteration, W)

np.save('spectral_angles', spectral_angles)
