import numpy as np
import matplotlib.pyplot as plt

taus = [10**-i for i in range(10)]
acts_mean = np.abs(np.load('acts_mean.npy'))
acts_std = np.load('acts_std.npy')
print(taus)
print(acts_mean)
# print(activations.shape)
# activations = activations[:, 0:1]
# print(activations.shape)
plt.yscale('log')
plt.xscale('log')
# plt.errorbar(taus, acts_mean, yerr=acts_std)
num = 10
plt.scatter(taus[0:num],acts_mean[0:num])
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$|$' + 'mean ' + r'$\Vert\bf X\Vert}$' + r'$|$', fontsize=14)
plt.grid(True)
# plt.axis('tight')
# plt.ylim(ymin=10**-30)
# for idx, line in enumerate(activations):
    # plt.scatter([idx] * len(line), line, s=1, marker='.')
    # plt.pcolormesh([idx] * 100, line, [1] * 100)

plt.savefig('ON_activations.png')
plt.show()
