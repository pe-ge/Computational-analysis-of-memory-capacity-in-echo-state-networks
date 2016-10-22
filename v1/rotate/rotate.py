import numpy as np
from numpy import random
from library.ortho import learn_orthogonal, orthogonality
from library.mc6 import memory_capacity


def rotate(vect, theta):
    N = len(vect)
    rot_matrix = np.eye(N)

    rot_matrix[0, 0] = np.cos(theta)
    rot_matrix[1, 1] = np.cos(theta)
    rot_matrix[1, 0] = np.sin(theta)
    rot_matrix[0, 1] = -np.sin(theta)

    return np.dot(vect, rot_matrix)


def measure_mc(W, WI):
    return memory_capacity(W, WI, memory_max=memory_max, iterations=1200,
                           iterations_coef_measure=1000, use_input=False,
                           target_later=True, calc_lyapunov=False)


def measure_og(W):
    return orthogonality(W)

sigma = 0.092
tau = 0.01
q = 100
eta = 3*10**-2
memory_max = int(q*1.2)

ITERATIONS = 100
TOTAL = 10
MC_before = np.zeros([TOTAL])
MC_after = np.zeros([TOTAL])

for t in range(TOTAL):
    print(t)
    W = random.normal(0, sigma, [q, q])
    WI = random.uniform(-tau, tau, q)

    for it in range(ITERATIONS):
        W = learn_orthogonal(W, eta)

        MC_before[t], _ = measure_mc(W, WI)
        og = measure_og(W)

    angle = np.random.uniform(0, np.pi)
    for i in range(q):
        W[:, i] = rotate(W[:, i], 0.1)

    MC_after[t], _ = measure_mc(W, WI)

np.save('mc_before', MC_before)
np.save('mc_after', MC_after)
