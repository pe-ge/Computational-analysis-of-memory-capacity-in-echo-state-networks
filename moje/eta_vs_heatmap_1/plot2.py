import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from library.ortho import learn_orthonormal
from library.mc6 import memory_capacity


tau = 0.01
sigma = 0.092
N = 100
memory_max = int(N*1.2)

ORTHO_ITERATIONS = 100
TOTAL_ITERATIONS = 100

eta_min = 0.005
eta_max = 0.1
xi_min = 0.85
xi_max = 1


def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI, memory_max=memory_max, iterations=1200,
                           iterations_coef_measure=1000, use_input=False,
                           target_later=True)


def load_data():
    return np.load('x.npy'), np.load('y.npy'), np.load('z.npy')
    mc = np.load('mc50.npy')
    smoothness = 100
    x = np.linspace(0.005, 0.1, smoothness)
    y = np.linspace(0.85, 1, smoothness)
    return x, y, np.mean(mc, axis=2)


def generate_data():
    numdata = 30000
    x = np.random.uniform(eta_min, eta_max, numdata)
    y = np.random.uniform(xi_min, xi_max, numdata)
    z = np.zeros(numdata)
    for i in range(numdata):
        eta = x[i]
        decay = y[i]
        WG = np.random.normal(0, sigma, [N, N])
        WI = np.random.uniform(-tau, tau, N)

        for it in range(ORTHO_ITERATIONS):
            WG = learn_orthonormal(WG, eta)
            eta = eta * decay

        z[i], _ = measure_mc(WG, WI, False)
    np.save('x', x)
    np.save('y', y)
    np.save('z', z)
    print(z)
    return x, y, z


def main():
    x, y, z = load_data()
    print(z.shape)

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x, y, z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot
    plt.imshow(zz,
               origin='lower',
               cmap=plt.get_cmap('hot'),
               extent=[eta_min, eta_max, xi_min, xi_max],
               aspect='auto')
    m = cm.ScalarMappable(cmap=plt.get_cmap('hot'))
    m.set_array(z)
    plt.colorbar(m, label='memory capacity', ticks=[30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel(r'$\eta$', size=24)
    plt.ylabel(r'$\xi$', size=24)
    plt.savefig('eta_xi_mc_sampled.png')
    plt.show()


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x**i * y**j
    return z

main()
