import numpy as np
from matplotlib import pyplot as plt
import matplotlib

spectral_angles = np.load('spectral_angles.npy')

plt.xlabel('iteration', size=24)
plt.ylabel('spectral angle', size=24)
plt.title('OG method', size=24)
plt.grid(True)
plt.plot(spectral_angles)
matplotlib.rcParams.update({'font.size': 18})
plt.savefig('it_spectral_angle_og.png')
plt.show()
