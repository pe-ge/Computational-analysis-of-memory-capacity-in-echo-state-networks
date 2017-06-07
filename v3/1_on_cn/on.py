# from library.ortho import learn_orthogonal
from library.ortho import learn_orthonormal
import numpy as np
from numpy import random
from numpy.linalg import cond

INSTANCES = 5
ORTHOPROCESS_ITERATIONS = 50
RHO = 0.95
TAU = 10**-10
ETA_0 = 7*10**-2
# ETA = 3*10**-2

reservoir_sizes = list(range(100, 1000 + 1, 100))
res_sizes = len(reservoir_sizes)

cond_before = np.zeros((res_sizes, INSTANCES))
cond_after = np.zeros((res_sizes, INSTANCES))

for N_idx, N in enumerate(reservoir_sizes):
    print(N)
    for inst_idx in range(INSTANCES):
        W = np.random.normal(0, 1, [N, N])
        W = W * (RHO / np.max(np.abs(np.linalg.eig(W)[0])))
        WI = random.uniform(-TAU, TAU, N)

        cond_before[N_idx, inst_idx] = cond(W)

        eta = ETA_0
        for it in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthonormal(W, eta)
            eta = eta * 0.9

        cond_after[N_idx, inst_idx] = cond(W)

    np.save('cond_before', cond_before)
    np.save('cond_after', cond_after)
