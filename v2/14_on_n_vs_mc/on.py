from library.ortho import learn_orthonormal
# from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random


INSTANCES = 10
ORTHOPROCESS_ITERATIONS = 30

rho = 0.95
tau = 10**-10
# eta = 3*10**-2
eta_0 = 7*10**-2

reservoir_sizes = list(range(100, 1000 + 1, 100))


def measure_mc(W, WI, iterations):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=iterations,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)

mcb_mean = np.zeros(len(reservoir_sizes))
mcb_std = np.zeros(len(reservoir_sizes))

mca_mean = np.zeros(len(reservoir_sizes))
mca_std = np.zeros(len(reservoir_sizes))

for rsi, N in enumerate(reservoir_sizes):
    print(N)
    mc_before = np.zeros(INSTANCES)
    mc_after = np.zeros(INSTANCES)
    for inst in range(INSTANCES):
        W = random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        mc_before[inst], _ = measure_mc(W, WI, 10*N)

        eta = eta_0
        for _ in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
            eta = eta * 0.9

        mc_after[inst], _ = measure_mc(W, WI, 10*N)

    mcb_mean[rsi] = np.average(mc_before)
    mcb_std[rsi] = np.std(mc_before)

    mca_mean[rsi] = np.average(mc_after)
    mca_std[rsi] = np.std(mc_after)

np.save('mcbm', mcb_mean)
np.save('mcbs', mcb_std)

np.save('mcam', mca_mean)
np.save('mcas', mca_std)
