from library.ortho import learn_orthonormal
from library.mc6 import output_weights

import numpy as np
from numpy import random


ORTHOPROCESS_ITERATIONS = 100
INSTANCES = 100

rho = 0.99
taus = [10**-i for i in range(20)]
eta_0 = 7*10**-2
tau = 10**-20

N = 100
IWs = np.zeros([INSTANCES, len(taus)])
OWs_m = np.zeros([INSTANCES, len(taus)])
OWs_s = np.zeros([INSTANCES, len(taus)])


def OW(W, WI):
    return output_weights(W, WI,
                           memory_max=int(1.1*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=False)

for inst_idx in range(INSTANCES):
    for tau_idx, tau in enumerate(taus):
        print(inst_idx, tau)
        W = random.normal(0, 1, [N, N])
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-tau, tau, N)

        IWs[inst_idx, tau_idx] = np.linalg.norm(WI)

        eta = eta_0
        for ix in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
            eta = eta * 0.9

        ow = OW(W, WI)
        norm = np.linalg.norm(ow, axis=1)
        OWs_m[inst_idx, tau_idx], OWs_s[inst_idx, tau_idx] = np.mean(norm), np.std(norm)

np.save('iws', IWs)
np.save('owsm', OWs_m)
np.save('owss', OWs_s)
