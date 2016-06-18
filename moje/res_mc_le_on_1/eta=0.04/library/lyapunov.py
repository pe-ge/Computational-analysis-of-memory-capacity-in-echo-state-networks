import numpy as np
import math


def lyapunov_exp(W, WI, x):
    Gama0 = 10**-12
    LE = 0

    for n in range(np.size(W, axis=0)):
        Deltax = np.zeros(np.size(W, axis=0))
        Deltax[n] = Gama0

        x1 = x
        x2 = np.add(x, Deltax)
        Gaman = 0

        for k in range(500):
            u = -1 + 2 * np.random.random()
            x1 = np.tanh(np.dot(W, x1) + WI * u)
            x2 = np.tanh(np.dot(W, x2) + WI * u)
            Gamak = np.sqrt(np.sum((x1 - x2) ** 2))
            if (math.isnan(Gamak)):
                print('je zle')
            Gaman = Gaman + np.log(Gamak / Gama0)
            x2 = x1 + (Gama0 / Gamak) * (x2 - x1)

        Gaman = Gaman / 500

        LE = LE + Gaman
    return LE / np.size(W, axis=0)
