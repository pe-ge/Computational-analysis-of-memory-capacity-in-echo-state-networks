import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

means = np.load('ortog_mean.npy')
stds = np.load('ortog_std.npy')

reservoir_sizes = list(range(100, 1000 + 1, 100))
ORTHOPROCESS_ITERATIONS = 50
iterations = range(ORTHOPROCESS_ITERATIONS + 1)

for i in range(means.shape[0]):
    plt.errorbar(iterations, means[i, :], yerr=stds[i, :], label=reservoir_sizes[i])

plt.xlabel('iterations')
plt.ylabel('orthogonality')
plt.title('ON method')
# plt.grid(True)
plt.legend(loc=4)
try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
