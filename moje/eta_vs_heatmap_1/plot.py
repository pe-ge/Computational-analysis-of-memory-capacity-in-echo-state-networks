import numpy as np
import matplotlib.pyplot as plt

data = np.load('mc.npy')
heatmap = np.mean(data, axis=2)

eta_min = 0.01
eta_max = 0.1
xi_min = 0.91
xi_max = 1

plt.xlabel(r'$\eta$', size=24)
plt.ylabel(r'$\xi$', size=24)

plt.imshow(heatmap,
           origin='lower',
           cmap=plt.get_cmap('hot'),
           extent=[eta_min, eta_max, xi_min, xi_max])

plt.colorbar(label='memory capacity', ticks=[40, 50, 60, 70, 80, 90, 100])

plt.savefig('eta_xi_mc.png')

plt.show()
