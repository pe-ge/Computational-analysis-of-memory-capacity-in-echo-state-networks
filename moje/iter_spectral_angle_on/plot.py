import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from library.aux import try_save_fig

spectral_angles = np.load('spectral_angles.npy')

fig = plt.figure(figsize=(8.5, 6.5))
plt.xlabel('iteration', size=24)
plt.ylabel('spectral angle', size=24)
plt.title('ON method', size=24)
plt.grid(True)
plt.plot(spectral_angles)
matplotlib.rcParams.update({'font.size': 18})
try_save_fig('it_spectral_angle_on.png', ext="pdf")
plt.show()
