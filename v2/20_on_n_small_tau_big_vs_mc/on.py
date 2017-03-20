# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random


INSTANCES = 10
ORTHOPROCESS_ITERATIONS = 50

rho = 0.95
# eta = 3*10**-2
eta_0 = 7*10**-2

reservoir_sizes = list(range(10, 100 + 1, 10))
taus = [10, 100, 1000]

def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)


mc_mean = np.zeros([len(reservoir_sizes), len(taus)])
mc_std = np.zeros([len(reservoir_sizes), len(taus)])

for tau_idx, tau in enumerate(taus):
    print(tau_idx)
    for rsi, N in enumerate(reservoir_sizes):
        mc = np.zeros(INSTANCES)
        for inst in range(INSTANCES):
            W = random.normal(0, 1, [N, N])
            W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
            WI = random.uniform(-tau, tau, N)

            eta = eta_0
            for _ in range(ORTHOPROCESS_ITERATIONS):
                # W = learn_orthogonal(W, eta)
                W = learn_orthonormal(W, eta)
                eta = eta * 0.9

            mc[inst], _ = measure_mc(W, WI)
            print(N, mc[inst])

        mc_mean[rsi, tau_idx] = np.average(mc)
        mc_std[rsi, tau_idx] = np.std(mc)

        np.save('mcm', mc_mean)
        np.save('mcs', mc_std)
