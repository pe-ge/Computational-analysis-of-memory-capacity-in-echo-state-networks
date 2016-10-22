import numpy as np
import matplotlib.pyplot as plt

activations = np.load('activations.npy')
print(activations.shape)
# activations = activations[:, 0:1]
print(activations.shape)

for idx, line in enumerate(activations):
    plt.scatter([idx] * len(line), line, s=1, marker='.')
    # plt.pcolormesh([idx] * 100, line, [1] * 100)

plt.savefig('neurons_activations.png')
plt.show()
