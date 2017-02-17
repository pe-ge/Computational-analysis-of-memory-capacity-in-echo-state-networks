import numpy as np
import matplotlib.pyplot as plt

reservoir_sizes = list(range(100, 1000 + 1, 100))
acts_mean = np.load('acts_mean.npy')
acts_std = np.load('acts_std.npy')
# print(activations.shape)
# activations = activations[:, 0:1]
# print(activations.shape)

plt.errorbar(reservoir_sizes, acts_mean, yerr=acts_std)
plt.xlabel('N')
plt.ylabel('mean X activations')
# for idx, line in enumerate(activations):
    # plt.scatter([idx] * len(line), line, s=1, marker='.')
    # plt.pcolormesh([idx] * 100, line, [1] * 100)

plt.savefig('OG_activations.png')
plt.show()
