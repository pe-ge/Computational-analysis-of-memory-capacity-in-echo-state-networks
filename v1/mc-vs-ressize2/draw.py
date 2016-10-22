import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from library.aux import try_save_fig

data = np.array(pickle.load(open('collect-rhos/data0.pickle', 'rb')))
fig = plt.figure(figsize=(8, 6.5))

x = np.linspace(0.8, 1.10, 20)
lines = data[5, :, :].transpose()
for points in lines:
    plt.scatter(x, points, s=1)

plt.grid(True)
plt.xlim(0.79, 1.11)
matplotlib.rcParams.update({'font.size': 18})
plt.ylim(-2, 82)
plt.xlabel(r'$\rho$', size=24)
plt.ylabel('memory capacity', size=24)
try_save_fig('rho.png', ext="pdf")
plt.show()
