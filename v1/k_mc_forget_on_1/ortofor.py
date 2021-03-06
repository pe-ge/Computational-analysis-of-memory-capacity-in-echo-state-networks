#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ortofor.py - orthogonalization impact on forgetting curve
Created: 13.8.2015

Goal: Investigate how the orthogonalization process changes forgetting curves (FC).
    Compare FC before and after orthogonalization
"""

from library.ortho import learn_orthonormal
from library.mc6forget_old import memory_capacity
from library.aux import try_save_fig
from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt

sigma = 0.092
tau = 0.01
N = 100

memory_max = int(1.6*N)  # before 1.2
INSTANCES = 100

ORTHOPROCESS_ITERATIONS = 100
eta_0 = 7*10**-2

lines = ["--", "-","--","-.",":"]
linecycler = cycle(lines)

def measure_mc(W, WI):
    return memory_capacity(W, WI,
                           memory_max=memory_max,
                           iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False,
                           target_later=True)


forget_before = np.zeros([INSTANCES, memory_max])
forget_after = np.zeros([INSTANCES, memory_max])

for inst in range(INSTANCES):
    WI = np.random.uniform(-tau, tau, N)
    W = np.random.normal(0, sigma, [N, N])

    forget_before[inst, :] = measure_mc(W, WI)

    eta = eta_0
    for _ in range(ORTHOPROCESS_ITERATIONS):
        W = learn_orthonormal(W, eta)
        eta = eta * 0.90

    forget_after[inst, :] = measure_mc(W, WI)


mcbefore = np.sum(np.average(forget_before, axis=0))
mcafter = np.sum(np.average(forget_after, axis=0))

def replot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.errorbar(
        range(memory_max),
        np.average(forget_before, axis=0),
        yerr=np.std(forget_before, axis=0),
        label="before",
        fmt=next(linecycler))
    ax.errorbar(
        range(memory_max),
        np.average(forget_after, axis=0),
        yerr=np.std(forget_after, axis=0),
        label="after",
        fmt=next(linecycler))

    ax.text(45, 0.8, "MC = {:.2f}".format(mcbefore), color='blue')
    ax.text(96, 0.8, "MC = {:.2f}".format(mcafter), color='green')

    ax.grid(True)
    ax.legend()
    plt.xlim((0, 140))
    plt.ylim((-0.05, 1.2))
    plt.title("forgetting curves for ON method")
    plt.xlabel("$k$")
    plt.ylabel("$MC_k$")

    try_save_fig("figures/figure")
    try_save_fig("figures/figure", ext='pdf')
    plt.show()

replot()
