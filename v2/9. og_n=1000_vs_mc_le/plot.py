import numpy as np
from matplotlib import pyplot as plt
from library.aux import try_save_fig


ORTHOPROCESS_ITERATIONS = 100
mc = np.load('mc.npy')
le = np.load('le.npy')

xs = range(ORTHOPROCESS_ITERATIONS + 1)

sp1 = plt.subplot(211)
sp1.errorbar(xs, np.average(mc, axis=0), yerr=np.std(mc, axis=0))
plt.ylabel("MC")
plt.title("OG Method")

sp2 = plt.subplot(212)
sp2.errorbar(xs, np.average(le, axis=0), yerr=np.std(le, axis=0))

plt.xlabel("iteration")
plt.ylabel('LE')
try_save_fig("figures/figure")
try_save_fig("figures/figure", ext="pdf")
plt.show()
