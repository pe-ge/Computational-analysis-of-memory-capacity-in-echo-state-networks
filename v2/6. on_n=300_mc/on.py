from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity
import numpy as np
from numpy import random

INSTANCES = 50
ORTHOPROCESS_ITERATIONS = 50

eta_0 = 7*10**-2
N = 300
rho = 0.95
tau = 10**-20

def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=int(1.2*WI.shape[0]),
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True,
                           calc_lyapunov=True)


mc = np.zeros([INSTANCES, ORTHOPROCESS_ITERATIONS + 1])
le = np.zeros([INSTANCES, ORTHOPROCESS_ITERATIONS + 1])

for instance in range(INSTANCES):
    W = random.normal(0, 1, [N, N])
    W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = random.uniform(-tau, tau, N)

    mc[instance, 0], le[instance, 0] = measure_mc(W, WI)
    print('{}, {}, {}'.format(instance, mc[instance, 0], le[instance, 0]))
    eta = eta_0
    for iteration in range(ORTHOPROCESS_ITERATIONS):
        W = learn_orthonormal(W, eta)
        eta = eta * 0.9
        mc[instance, iteration + 1], le[instance, iteration + 1] = measure_mc(W, WI)
        print('{}, {}'.format(mc[instance, iteration + 1], le[instance, iteration + 1]))

np.save('mc', mc)
np.save('le', le)
