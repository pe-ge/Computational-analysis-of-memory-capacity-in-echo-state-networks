import numpy as np
import matplotlib.pyplot as plt

data = np.load('mc.npy')
heatmap = np.mean(data, axis=2)


smoothness = 10
etas = np.linspace(0.01, 0.1, smoothness)
decays = np.linspace(0.91, 1, smoothness)

plt.xticks(etas)
plt.yticks(decays)
plt.xlabel(r'$\eta$', size=24)
plt.ylabel(r'$\xi$', size=24)
plt.pcolormesh(etas, decays, heatmap, cmap=plt.get_cmap('hot'))
cbar = plt.colorbar(label='Memory Capacity')
# cbar.ax.tick_params(width=20)

plt.savefig('eta_xi_mc.png')

plt.show()
