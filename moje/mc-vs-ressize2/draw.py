import numpy as np
import pickle
import matplotlib.pyplot as plt

data = np.array(pickle.load(open('collect-rhos/data0.pickle', 'rb')))
print(data[0, :][0])
print(data.shape)
# print(data.shape)
# print(data)
# plt.plot(data[0, :][1])
# plt.plot(data[0].shape)
# print(data.shape)
# data = data[:, :, :, 0]
# model_225 = data[5, :, :]
# print(data.shape)

# x = np.arange(0.9, 1.55, 0.05)
# lines = data[5, :, :].transpose()
# plt.xlim(0.89, 1.51)
# for y in lines:
    # plt.scatter(x, y, s=5)
# plt.plot(np.mean(model_225, axis=1))

plt.grid(True)
plt.show()
