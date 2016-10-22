#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emceorto.py  -MC vs. orthogonality (second)
Created 9.4.2015
Based on mvo.py

Goal: Measure how does the memory capacity correlate with orthogonality.
Use orthogonalization process to adjust orthogonality.
Changed: use mc6 memory capacity (boedecker's definition, instead of mine)
"""

from library.ortho import orthogonality, learn_orthogonal
from library.mc6 import memory_capacity
import numpy as np
from numpy import random, floor, save

tau = 0.01
rho = 0.95
N = 100
eta = 3*10**-2

ITERATIONS = 100

LINES = 100

""" a 100-neuron reservoir usually begins with orthogonality 0.92.
For every 1/100 of orthogonality we measure current MC
"""


def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI, memory_max=150, iterations=1200,
                           iterations_coef_measure=1000,
                           use_input=False, target_later=True,
                           calc_lyapunov=calc_lyapunov)


def measure_og(W):
    return orthogonality(W)


all_oths = []
all_mcs = []
all_les = []
for l in range(LINES):
    W = np.random.normal(0, 1, [N, N])
    W = W * (rho / np.max(np.abs(np.linalg.eig(W)[0])))
    WI = random.uniform(-tau, tau, N)

    Wzal = W
    WIzal = WI

    oths = []
    mcs = []
    les = []
    mc, le = measure_mc(W, WI, True)
    og = measure_og(W)

    mcs.append(mc)
    oths.append(og)
    les.append(le)

    last_oth = floor(og*100)
    last_oth2 = floor(og*1000)
    last_oth3 = floor(og*10000)

    for it in range(ITERATIONS):
        W = learn_orthogonal(W, eta)
        calc_lyapunov = (it + 1) % 10 == 0
        og = measure_og(W)
        if floor(og*100) != last_oth:
            last_oth = floor(og*100)
            mc, le = measure_mc(W, WI, calc_lyapunov)
            mcs.append(mc)
            oths.append(og)
            les.append(le)
        elif og > 0.97 and floor(og*1000) != last_oth2:
            last_oth2 = floor(og*1000)
            mc, le = measure_mc(W, WI, calc_lyapunov)
            mcs.append(mc)
            oths.append(og)
            les.append(le)
        elif og > 0.999 and floor(og*10000) != last_oth3:
            last_oth3 = floor(og*10000)
            mc, le = measure_mc(W, WI, calc_lyapunov)
            mcs.append(mc)
            oths.append(og)
            les.append(le)

    all_oths.append(oths)
    all_mcs.append(mcs)
    all_les.append(les)

save('oths', all_oths)
save('mcs', all_mcs)
save('les', all_les)
