# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
from library.ortho import orthogonality

import numpy as np
from numpy import random


INSTANCES = 2
ORTHOPROCESS_ITERATIONS = 50

rho = 0.95
tau = 10**-10
eta_0 = 7*10**-2
# eta = 3*10**-2

reservoir_sizes = list(range(100, 1000 + 1, 100))

ortog_mean = np.zeros((len(reservoir_sizes), ORTHOPROCESS_ITERATIONS + 1))
ortog_std = np.zeros((len(reservoir_sizes), ORTHOPROCESS_ITERATIONS + 1))

for rsi, N in enumerate(reservoir_sizes):
    print(N)
    instances = np.zeros((INSTANCES, ORTHOPROCESS_ITERATIONS + 1))
    for inst in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        instances[inst, 0] = orthogonality(W)

        eta = eta_0
        for it in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
            instances[inst, it + 1] = orthogonality(W)
            eta = eta * 0.9

    ortog_mean[rsi, :] = np.mean(instances, axis=0)
    ortog_std[rsi, :] = np.std(instances, axis=0)

    np.save('ortog_mean', ortog_mean)
    np.save('ortog_std', ortog_std)
