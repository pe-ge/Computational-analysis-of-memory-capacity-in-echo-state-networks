#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emceresi.py - Memory capacity vs. reservoir size
Created: 8.4.2015

Goal: Measure MC for different reservoir sizes.
"""

import sys
import numpy as np
from mc6 import memory_capacity
from gerplot2 import compute, draw

q = 100
tau = 0.01

reservoir_sizes = [16, 36, 49, 64, 100, 225]

rhos = np.arange(0.9, 1.51, 0.05)

savedir2 = 'collect-rhos'

def generate_input_matrix(tau, q):
    return np.random.uniform(-tau, tau, q)


def generate_matrix_from_rho(rho, q):
    M = np.random.normal(0, 1, [q, q])
    return M * (rho / np.max(np.abs(np.linalg.eig(M)[0])))


measure2 = {
    'savedir': savedir2,
    'xticks': rhos,
    'xticks_desc_name': 'rhos',
    'xlabel': "$\\rho$",
    'rv': lambda xval, lineval: rvgen(xval, lineval, generate_matrix_from_rho),
}

measures = {
    'rho':     measure2,
}

for m in measures.values():
    m['linelabels'] = ["{}".format(rs) for rs in reservoir_sizes]
    m['ylabel'] = "Lyapunov exponent"


def rvgen(xval, lineval, Wgen):
    WI = generate_input_matrix(tau, lineval)
    W = Wgen(xval, lineval)

    mc, le = memory_capacity(W,
                             WI,
                             memory_max=int(lineval*3/2),
                             iterations=1200,
                             iterations_coef_measure=1000,
                             use_input=False,
                             target_later=True,
                             calc_lyapunov=True)
    return le


def main():
    if len(sys.argv) < 2:
        print("Usage: '{0:} sigma compute' or {0:} sigma draw".format(sys.argv[0]))
        return

    task = 'rho'
    action = sys.argv[1]

    try:
        m = measures[task]
        savedir, xticks = m['savedir'], m['xticks']

        if action == 'compute':
            basic_data = "q={}\ntau={}\n{}={}\nreservoir_sizes={}\n".format(q,tau,m['xticks_desc_name'],repr(xticks),repr(reservoir_sizes))
            compute(m['rv'], savedir, xticks, reservoir_sizes, basic_data)
        elif action == 'draw':
            draw(savedir, xticks, reservoir_sizes, m['xlabel'], m['ylabel'], m['linelabels'], loc=1, save=True)
    except KeyError:
        print("unknown task")
        return


if __name__ == '__main__':
    main()
