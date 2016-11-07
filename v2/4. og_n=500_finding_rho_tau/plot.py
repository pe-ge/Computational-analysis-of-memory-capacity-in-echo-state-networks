import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig
import matplotlib


rhos = np.arange(0.9, 1.01, 0.01)
taus = np.arange(0.001, 0.0101, 0.001)
mc = np.load('mc.npy')

plt.imshow(mc,
           origin='lower',
           cmap=plt.get_cmap('hot'),
           extent=[min(taus), max(taus), min(rhos), max(rhos)],
           aspect='auto',
           interpolation='none')

print(mc.shape)

plt.colorbar(label='memory capacity')
plt.xlabel('tau')
plt.ylabel('rho')
plt.title('OG method, eta=0.03, N=500')
matplotlib.rcParams.update({'font.size': 18})

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
