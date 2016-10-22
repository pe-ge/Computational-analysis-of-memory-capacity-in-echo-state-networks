#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortoven.py - Orthogonalization vs. N
Created 5.8.2015

Goal: Measure the effect of orthogonalization process, depending on N.
"""

from library.ortho import orthogonality, learn_orthogonal
from library.mc6 import memory_capacity
from library.aux import try_save_fig

import numpy as np
from numpy import random, floor, average
from matplotlib import pyplot as plt

from pylab import figtext, gca

from time import time

TARGET_SP_RADIUS = 0.95 # before 0.9
INSTANCES = 100
ORTHOPROCESS_ITERATIONS = 100

tau = 0.01 # previously 0.001
eta = 3*10**-2 # learning rate, 1*10**-1 je uz privela

reservoir_sizes = list(range(10, 100 + 1, 10)) #[16, 25, 64, 100]

def measure_mc(W, WI):
    return memory_capacity(W, WI, memory_max=int(1.1*WI.shape[0]), iterations=1200, iterations_coef_measure=1000, use_input=False, target_later=True)

rho = TARGET_SP_RADIUS

mc_before = np.zeros(INSTANCES)
mc_after = np.zeros(INSTANCES)

mcb_mean = np.zeros(len(reservoir_sizes))
mcb_std = np.zeros(len(reservoir_sizes))
mca_mean = np.zeros(len(reservoir_sizes))
mca_std = np.zeros(len(reservoir_sizes))

ttotal = time()

for rsi, N in enumerate(reservoir_sizes):
    print("reservoir size", N)
    tstart = time()
    for inst in range(INSTANCES):
        W = random.normal(0, 1, [N, N])
        WI = random.uniform(-tau, tau, N)
        W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))

        mc_before[inst] = measure_mc(W, WI)

        for _ in range(ORTHOPROCESS_ITERATIONS):
            W = learn_orthogonal(W, eta)

        mc_after[inst] = measure_mc(W, WI)
        print(inst, 'of', INSTANCES)
    mcb_mean[rsi] = np.average(mc_before)
    mca_mean[rsi] = np.average(mc_after)

    mcb_std[rsi] = np.std(mc_before)
    mca_std[rsi] = np.std(mc_after)

    print("\t took {:.2f} seconds".format(time() - tstart))


print("total time: {:.2f} seconds".format(time() - ttotal))

print("done, plotting")

def replot():
    plt.plot(reservoir_sizes, reservoir_sizes, 'k', linestyle="--")
    plt.errorbar(reservoir_sizes, mcb_mean, yerr=mcb_std, label="before ")
    plt.errorbar(reservoir_sizes, mca_mean, yerr=mca_std, label="after")
    plt.xlabel("reservoir size")
    plt.ylabel("memory capacity")
    plt.title("impact of orthogonalization process on MC".format(tau, rho))
    plt.grid(True)
    plt.legend(loc=2)

    print("tau={tau}, rho={rho}, eta={eta}".format(tau=tau, rho=rho, eta=eta))
    print("instances={i}, ortho-iterations={oi}".format(i=INSTANCES, oi=ORTHOPROCESS_ITERATIONS))

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext="pdf")
    plt.show()

replot()
