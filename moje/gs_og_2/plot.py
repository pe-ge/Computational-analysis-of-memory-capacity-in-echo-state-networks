import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig

reservoir_sizes = list(range(10, 101, 10))

mcb_mean = np.load('mcbm.npy')
mcb_std = np.load('mcbs.npy')

mca_mean = np.load('mcam.npy')
mca_std = np.load('mcas.npy')

orthob_mean = np.load('obm.npy')
orthob_std = np.load('obs.npy')
orthoa_mean = np.load('oam.npy')
orthoa_std = np.load('oas.npy')

plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle=":")
plt.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ", fmt="--")
plt.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
plt.xlabel("reservoir size")
plt.ylabel("memory capacity")
plt.title("impact of Gram-Schmidt process on MC")
plt.grid(True)
plt.legend(loc=4)

# ax2 = plt.subplot(212)
# ax2.errorbar(reservoir_sizes, orthob_mean, yerr=orthob_std, label="before ", fmt="--")
# ax2.errorbar(reservoir_sizes, orthoa_mean, yerr=orthoa_std, label="after")
# plt.ylabel("orthogonality")
# plt.ylim((0.5, 1.01))
# plt.grid(True)
# plt.legend(loc=4)

try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
