import numpy as np
import matplotlib.pyplot as plt

xs = list(range(100, 1000 + 1, 100))
cond_before = np.load('cond_before.npy')
cond_after = np.load('cond_after.npy')
plt.errorbar(xs, np.mean(cond_before, axis=1), np.std(cond_before, axis=1), label='before ON')
plt.yscale('log')
plt.errorbar(xs, np.mean(cond_after, axis=1), np.std(cond_after, axis=1), label='after ON')
plt.xlabel('N')
plt.ylabel('cond. num. of inv. res. acts')
plt.legend()
# plt.show()
plt.savefig('ON_log')

plt.yscale('linear')
plt.savefig('ON_linear')
