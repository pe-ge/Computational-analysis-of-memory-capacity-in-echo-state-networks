import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

etas = np.arange(0.001, 0.05, 0.002)
mc = np.load('mc.npy')
tau = 0.001
rho = 0.99

plt.plot(etas, mc)

plt.xlabel('eta')
plt.ylabel('mc')
plt.title('OG method N: {}, tau: {}, rho: {}'.format(500, tau, rho))
try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
