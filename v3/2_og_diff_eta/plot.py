import numpy as np
import matplotlib.pyplot as plt

smoothness = 20
etas = np.linspace(0.005, 0.1, smoothness)

data = np.load('mc.npy')
print(np.max(data, axis=0))
# print(data)
plt.xlabel(r'$\eta$')
plt.ylabel('MC')
plt.title('OG method, 10 instances, 20 iterations each')
plt.errorbar(etas, np.mean(data, axis=1), np.std(data, axis=1))

plt.savefig('OG.png')
plt.show()
