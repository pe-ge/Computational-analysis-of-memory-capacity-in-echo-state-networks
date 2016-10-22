#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortofor.py - orthogonalization impact on forgetting curve
Created: 13.8.2015

Goal: Investigate how the orthogonalization process changes forgetting curves (FC).
	Compare FC before and after orthogonalization
"""

from library.ortho import orthogonality, learn_orthogonal
from library.mc6forget import memory_capacity
from library.aux import try_save_fig

import numpy as np
from matplotlib import pyplot as plt

from time import time

rho = 0.95 # spectral radius
tau = 0.01
N = 100

memory_max = int(1.2*N) # before 1.2
INSTANCES = 100

ORTHOPROCESS_ITERATIONS = 100
eta = 4*10**-2

def measure_mc(W, WI):
	return memory_capacity(W, WI, memory_max=memory_max, iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)


forget_before = np.zeros([INSTANCES, memory_max])
forget_after = np.zeros([INSTANCES, memory_max])

tstart = time()

for inst in range(INSTANCES):
	print('instance', inst, 'of', INSTANCES)
	W = np.random.normal(0, 1, [N, N])
	WI = np.random.uniform(-tau, tau, N)
	W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))

	forget_before[inst, :] = measure_mc(W, WI)

	for _ in range(ORTHOPROCESS_ITERATIONS):
		W = learn_orthogonal(W, eta)

	forget_after[inst, :] = measure_mc(W, WI)

print("took {:2f} seconds".format(time() - tstart))
print("plotting")

mcbefore = np.sum(np.average(forget_before, axis=0))
mcafter = np.sum(np.average(forget_after, axis=0))

def replot():
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.errorbar(
		range(memory_max),
		np.average(forget_before, axis=0),
		yerr=np.std(forget_before, axis=0),
		label="before")
	ax.errorbar(
		range(memory_max),
		np.average(forget_after, axis=0),
		yerr=np.std(forget_after, axis=0),
		label="after")

	ax.text(45, 0.8, "MC = {:.2f}".format(mcbefore), color='blue')
	ax.text(96, 0.8, "MC = {:.2f}".format(mcafter), color='green')

	ax.grid(True)
	ax.legend()
	plt.title("MC forgetting curves before and after orthogonalization")
	plt.xlabel("$k$")
	plt.ylabel("$MC_k$")

	try_save_fig("figures/figure")
	try_save_fig("figures/figure", ext='pdf')
	plt.show()

replot()
