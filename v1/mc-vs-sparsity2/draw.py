import numpy as np
import pickle
import matplotlib.pyplot as plt

data = np.array(pickle.load(open('collect-rhos/data0.pickle', 'rb')))
print(data.shape)
# data = data.reshape([6, 13, 100])

# x = np.arange(0.9, 1.55, 0.05)
x = np.arange(0, 7, 1)
lines = data[5, :].transpose()
plt.xlim(0.89, 1.51)
for y in lines:
    # plt.scatter(x, y, s=5)
    plt.scatter(x, y)

plt.grid(True)
plt.show()
