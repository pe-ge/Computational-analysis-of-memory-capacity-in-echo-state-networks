from library.ortho import learn_orthogonal
from library.mc6 import memory_capacity

import numpy as np
from numpy import random


INSTANCES = 5
ORTHOPROCESS_ITERATIONS = 100

rho = 0.99
tau = 10**-20
eta = 3*10**-2

N = 1000


def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)

mc = np.zeros([INSTANCES, ORTHOPROCESS_ITERATIONS + 1])

for inst in range(INSTANCES):
    print(inst)
    W = random.normal(0, 1, [N, N])
    W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = random.uniform(-tau, tau, N)

    mc[inst, 0], _ = measure_mc(W, WI)

    for ix in range(ORTHOPROCESS_ITERATIONS):
        W = learn_orthogonal(W, eta)
        mc[inst, ix + 1], _ = measure_mc(W, WI)
        np.save('mc', mc)
