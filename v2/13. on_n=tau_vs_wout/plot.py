import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

taus = [10**-i for i in range(0, 10, 1)]
reservoir_sizes = list(range(100, 1000 + 1, 100))

ow = np.load('ow.npy')
print(ow.shape)

ax = plt.subplot(1, 1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.invert_xaxis()
# ax.errorbar(taus, np.mean(ows_m, axis=0))
for i in range(ow.shape[0]):
    ax.plot(taus, ow[i, :], label=reservoir_sizes[i])

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc=2, title='reservoir size')

plt.xlabel(r'$\tau$', fontsize=20)
plt.ylabel('mean ' + r'$\Vert\bf w_{i}^{out}\Vert}$', fontsize=14)
plt.grid(True)
plt.title('OG Method')

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
