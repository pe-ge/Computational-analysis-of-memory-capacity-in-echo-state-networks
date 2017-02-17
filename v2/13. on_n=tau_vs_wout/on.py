from library.ortho import learn_orthonormal
from library.mc6 import output_weights

import numpy as np
from numpy import random


ORTHOPROCESS_ITERATIONS = 50
INSTANCES = 10

rho = 0.99
taus = [10**-i for i in range(10)]
eta_0 = 7*10**-2

reservoir_sizes = list(range(100, 1000 + 1, 100))
ow = np.zeros([len(taus), len(taus)])


def OW(W, WI, iterations):
    return output_weights(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=iterations,
                           use_input=False,
                           target_later=True)

for N_idx, N in enumerate(reservoir_sizes):
    print('N: {}'.format(N))
    for tau_idx, tau in enumerate(taus):
        print('tau: {}'.format(tau))
        ows = np.zeros(INSTANCES)
        for inst_idx in range(INSTANCES):
            W = random.normal(0, 1, [N, N])
            W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
            WI = random.uniform(-tau, tau, N)

            eta = eta_0
            for ix in range(ORTHOPROCESS_ITERATIONS):
                W = learn_orthonormal(W, eta)
                eta = eta * 0.9

            ows[inst_idx] = np.linalg.norm(OW(W, WI, N * 10))

        ow[N_idx, tau_idx] = np.mean(ows)
        np.save('ow', ow)
