#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
novegon.py - OrtNOrmalization vs. orthoGONalization
Created: 13.8.2015

Goal: Compare the effects of orthogonalization with the effects of orthonormalization
"""

from library.ortho import orthogonality, \
	learn_orthogonal, energy_orthogonal, \
	learn_orthonormal, energy_orthonormal

from library.aux import try_save_fig
from library.mc6 import memory_capacity

import numpy as np
from matplotlib import pyplot as plt
from time import time

tau = 0.01
rho = 0.95
N 	= 100
memory_max = int(N*1.2)

ORTHO_ITERATIONS = 200
eta = 3*10**-2

legend_location = 4

def measure_mc(W, WI):
	return memory_capacity(W, WI, memory_max=memory_max, iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)

W = np.random.normal(0, 1, [N, N])
WI = np.random.uniform(-tau, tau, N)
W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
WG = WN = W

mc_gon = np.zeros(ORTHO_ITERATIONS + 1)
mc_nor = np.zeros(ORTHO_ITERATIONS + 1)

eigenvaluesG = np.zeros([ORTHO_ITERATIONS + 1, N])
singular_valuesG = np.zeros([ORTHO_ITERATIONS + 1, N])
eigenvaluesN = np.zeros([ORTHO_ITERATIONS + 1, N])
singular_valuesN = np.zeros([ORTHO_ITERATIONS + 1, N])

mc_gon[0] = measure_mc(WG, WI)
mc_nor[0] = measure_mc(WN, WI)
eigenvaluesG[0,:] = np.sort(np.abs(np.linalg.eig(W)[0]))
singular_valuesG[0,:] = np.linalg.svd(W, compute_uv=False)
eigenvaluesN[0,:] = np.sort(np.abs(np.linalg.eig(W)[0]))
singular_valuesN[0,:] = np.linalg.svd(W, compute_uv=False)


for it in range(ORTHO_ITERATIONS):
	print('\riteration', it, 'of', ORTHO_ITERATIONS, end='')
	WG = learn_orthogonal(WG, eta)
	WN = learn_orthonormal(WN, eta)

	mc_gon[it + 1] = measure_mc(WG, WI)
	mc_nor[it + 1] = measure_mc(WN, WI)

	eigenvaluesG[it + 1,:] = np.sort(np.abs(np.linalg.eig(WG)[0]))
	singular_valuesG[it + 1,:] = np.linalg.svd(WG, compute_uv=False)

	eigenvaluesN[it + 1,:] = np.sort(np.abs(np.linalg.eig(WN)[0]))
	singular_valuesN[it + 1,:] = np.linalg.svd(WN, compute_uv=False)

print()

def replot():
	xs = range(ORTHO_ITERATIONS + 1)
	ax1 = plt.subplot(311)
	plt.plot(xs, mc_gon, label="orthogonalization")
	plt.plot(xs, mc_nor, label="orthonormalization")
	plt.grid(True)
	plt.legend(loc=legend_location)
	#plt.xlabel("iteration")
	plt.ylabel("memory capacity")

	plt.setp(ax1.get_xticklabels(), visible=False)

	# eigenvalues
	ax2 = plt.subplot(312, sharex=ax1)
	plt.plot(xs, eigenvaluesG, c='blue', label="gon")
	plt.plot(xs, eigenvaluesN, c='green', label="norm")

	plt.grid(True)
	#plt.legend(loc=legend_location)
	#plt.xlabel("iteration")
	plt.ylabel("abs(eigenvalues)")
	plt.setp(ax2.get_xticklabels(), visible=False)

	# singular values
	ax3 = plt.subplot(313, sharex=ax1)
	plt.plot(xs, singular_valuesG, c='blue', label="gon")
	plt.plot(xs, singular_valuesN, c='green', label="norm")

	plt.grid(True)
	#plt.legend(loc=legend_location)
	plt.xlabel("iteration")
	plt.ylabel("singular values")

	try_save_fig("figures/figure")
	try_save_fig("figures/figure", ext="pdf")
	plt.show()

replot()




