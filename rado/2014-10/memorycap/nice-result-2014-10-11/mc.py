#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import *

p = 1 	# one input node
q = 100 # 100 reservoir nodes
r = 200 # 200 output nodes

MEMORY_MAX = 200
ITERATIONS = 2000
ITERATIONS_SKIPPED = 1000
ITERATIONS_MEASURED = ITERATIONS - ITERATIONS_SKIPPED

ITERATIONS_COEF_MEASURE = 1000


dist_WI = lambda: random.uniform(-1.,1.,[q,p])
dist_W = lambda: random.normal(0., sigma, [q,q]) 

dist_input = lambda: random.uniform(-0.8, 0.8, ITERATIONS) # maybe [1,1] ?

def memory_capacity(W, WI):
	"""Calculates memory capacity of a NN 
		[given by its input weights WI and reservoir weights W].
	W  = q x q matrix storing hidden reservoir weights
	WI = q x p matrix storing input weights
	"""
	# vector initialization
	MC = zeros(MEMORY_MAX) # here we store memory capacity
	X = zeros([q,1])	# reservoir activations, @FIXME, maybe try only q, instead of [q, 1] (performance?)
	S = zeros([q,ITERATIONS_MEASURED])

	# generate random input
	u = dist_input() # all input; dimension: [ITERATIONS, 1]

	# run 2000 iterations and fill the matrices D and S
	for it in range(ITERATIONS):
		X = tanh(dot(W, X) + dot(WI, u[it]))

		if it >= ITERATIONS_SKIPPED:
			# record it to S
			S[:, it - ITERATIONS_SKIPPED] = X[:,0]

	# prepare matrix D of desired values
	assert MEMORY_MAX < ITERATIONS_SKIPPED
	D = zeros([MEMORY_MAX, ITERATIONS_MEASURED])

	for h in range(MEMORY_MAX): # fill each row
		#FIXME maybe should be: 'ITERATIONS - (h+1)', it depends, whether we measure 0th iteration as well
		D[h,:] = u[ITERATIONS_SKIPPED - h : ITERATIONS - h] 
	
	# calculate pseudoinverse S+ and with it, the matrix WO
	S_PINV = linalg.pinv(S)
	WO = dot(D, S_PINV)

	# do a new run for an unbiased test of quality of our newly trained WO
	# we skip MEMORY_MAX iterations to have large enough window
	u = dist_input()
	X = zeros([q,1])
	o = zeros([MEMORY_MAX, ITERATIONS_COEF_MEASURE]) # 200 x 1000
	for it in range(ITERATIONS_COEF_MEASURE + MEMORY_MAX):
		X = tanh(dot(W, X) + dot(WI, u[it]))
		if it >= MEMORY_MAX:
			# we calculate output nodes using WO  ( @FIXME maybe not a column, but a row?)
			o[:, it - MEMORY_MAX] = dot(WO, X)[:,0]

	# correlate outputs with inputs (shifted)
	for h in range(MEMORY_MAX):
		k = h + 1
		MC[h] = corrcoef(u[MEMORY_MAX - h : MEMORY_MAX + ITERATIONS_COEF_MEASURE - h], o[h, : ]) [0, 1]
		MC[h] = MC[h] * MC[h]

	# ???

	# profit

	return MC


sigma = 0.11

W = dist_W()
WI = dist_WI()

MC = memory_capacity(W, WI)

print("sigma = %f;\t capacity = %f." % (sigma, sum(MC)))

# plot it!

import matplotlib.pyplot as plt

#plt.set_title("ahoj")
plt.plot(range(MEMORY_MAX), MC)

plt.ylabel('correlation coefficient')
plt.xlabel('memory size')
plt.title('Memory capacity (sigma = %.3f)' % sigma)
plt.show()


#print( dist_W())
