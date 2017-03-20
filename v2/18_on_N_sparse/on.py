
# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random
import itertools

ORTHOPROCESS_ITERATIONS = 100
INSTANCES = 10

rho = 0.95
tau = 0.01  # previously 0.001
# eta = 3*10**-2
eta_0 = 7*10**-2

reservoir_sizes = list(range(10, 100 + 1, 10))  # [16, 25, 64, 100]
sparsities = [0, 0.5, 0.9, 0.93, 0.96, 0.99]

def make_sparse(W, sparsity):
    q = W.shape[0]
    for i, j in itertools.product(range(q), range(q)):
        if random.random() < sparsity:
            W[i, j] = 0.0
    return W

def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)


mc = np.zeros([len(reservoir_sizes), len(sparsities)])
stds = np.zeros([len(reservoir_sizes), len(sparsities)])
np.seterr(all='raise')
for rsi, N in enumerate(reservoir_sizes):
    print(N)
    for spars_idx, sparsity in enumerate(sparsities):
        instances = np.zeros(INSTANCES)
        inst = 0
        while inst < INSTANCES:
            try:
                W = np.random.normal(0, 1, [N, N])
                W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
                W = make_sparse(W, sparsity)
                W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
                WI = random.uniform(-tau, tau, N)

                eta = eta_0
                for _ in range(ORTHOPROCESS_ITERATIONS):
                    W = learn_orthonormal(W, eta)
                    # W = learn_orthogonal(W, eta)
                    eta = eta * 0.9

                instances[inst], _ = measure_mc(W, WI)
                inst += 1
            except (ValueError, FloatingPointError):
                pass

        mc[rsi, spars_idx] = np.mean(instances)
        stds[rsi, spars_idx] = np.std(instances)

    np.save('mc', mc)
    np.save('stds', stds)
