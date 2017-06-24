import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from library.aux import try_save_fig

data = np.load('mc50.npy')
heatmap = np.mean(data, axis=2)

# eta_min = 0.1
# eta_max = 0.5
# xi_min = 0.
# xi_max = 1
eta_min = 0.005
eta_max = 0.1
xi_min = 0.85
xi_max = 1

fig = plt.figure(figsize=(8, 6.5))
plt.xlabel(r'$\eta_0$', size=24)
plt.ylabel(r'$\xi$', size=24)

plt.imshow(heatmap,
           origin='lower',
           cmap=plt.get_cmap('hot'),
           extent=[eta_min, eta_max, xi_min, xi_max],
           aspect='auto')

plt.colorbar(label='memory capacity', ticks=[40, 50, 60, 70, 80, 90, 100])
matplotlib.rcParams.update({'font.size': 18})
# plt.colorbar()

try_save_fig('eta_xi_mc.png', ext="pdf")

plt.show()
