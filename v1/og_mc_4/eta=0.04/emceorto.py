#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emceorto.py  -MC vs. orthogonality (second)
Created 9.4.2015
Based on mvo.py

"""

from library.ortho import orthogonality, learn_orthogonal
from library.mc6 import memory_capacity

from numpy import random, floor, save

sigma = 0.092
tau = 0.01
q = 100
eta = 4*10**-2

ITERATIONS = 100

LINES = 100


def measure_mc(W, WI, calc_lyapunov):
    return memory_capacity(W, WI, memory_max=150, iterations=1000,
                           iterations_coef_measure=1000,
                           use_input=False, target_later=True,
                           calc_lyapunov=calc_lyapunov)


def measure_og(W):
    return orthogonality(W)

all_oths = []
all_mcs = []
all_les = []
for l in range(LINES):
    W = random.normal(0, sigma, [q, q])
    WI = random.uniform(-tau, tau, q)

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
        og = measure_og(W)
        if floor(og*100) != last_oth:
            last_oth = floor(og*100)
            mc, le = measure_mc(W, WI, True)
            mcs.append(mc)
            oths.append(og)
            les.append(le)
        elif og > 0.97 and floor(og*1000) != last_oth2:
            last_oth2 = floor(og*1000)
            mc, le = measure_mc(W, WI, True)
            mcs.append(mc)
            oths.append(og)
            les.append(le)
        elif og > 0.999 and floor(og*10000) != last_oth3:
            last_oth3 = floor(og*10000)
            mc, le = measure_mc(W, WI, True)
            mcs.append(mc)
            oths.append(og)
            les.append(le)

    all_oths.append(oths)
    all_mcs.append(mcs)
    all_les.append(les)

save('oths', all_oths)
save('mcs', all_mcs)
save('les', all_les)
