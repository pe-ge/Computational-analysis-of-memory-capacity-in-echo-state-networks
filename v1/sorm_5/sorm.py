import numpy as np
from math import pi, cos, sin

from library.mc6 import memory_capacity

N = 100
memory_max = int(N*1.2)

MODELS = 100
ITERATIONS = 100
NUM_ROTATIONS = 5


def measure_mc(W, WI):
    return memory_capacity(W, WI, memory_max=memory_max, iterations=1200,
                           iterations_coef_measure=1000, use_input=False,
                           target_later=True, calc_lyapunov=True)


def generate_Q(N):
    h, k = 0, 0
    while h >= k:
        h, k = [np.random.randint(N) for _ in range(2)]
    phi = np.random.uniform(-pi, pi)

    Q = np.eye(N)
    Q[h, h] = cos(phi)
    Q[h, k] = -sin(phi)
    Q[k, h] = sin(phi)
    Q[k, k] = cos(phi)

    return Q


mc = np.zeros([MODELS, ITERATIONS + 1])
le = np.zeros([MODELS, ITERATIONS + 1])
sp = np.zeros([MODELS, ITERATIONS + 1])

for it_m in range(MODELS):
    print(it_m)
    Q = np.eye(N)
    for _ in range(NUM_ROTATIONS):
        Q = np.dot(Q, generate_Q(N))

    WI = np.dot(Q.transpose(), np.eye(N)[0])
    W = np.eye(N)[np.random.permutation(N), :]

    m, l = measure_mc(W, WI)
    mc[it_m, 0] = m
    le[it_m, 0] = l
    sp[it_m, 0] = np.count_nonzero(W)

    for it in range(ITERATIONS):
        Q = generate_Q(N)
        W = np.dot(W, Q)
    #    left = np.random.randint(2)
    #    if left == 0:
    #        W = np.dot(Q, W)
    #    else:
    #        W = np.dot(W, Q)
    #    W = np.dot(np.dot(Q, W), np.transpose(Q))

        m, l = measure_mc(W, WI)
        mc[it_m, it + 1] = m
        le[it_m, it + 1] = l
        sp[it_m, it + 1] = np.count_nonzero(W)

np.save('mc', mc)
np.save('le', le)
np.save('sp', sp)
