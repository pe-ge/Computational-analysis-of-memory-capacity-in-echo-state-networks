from lyapunov import lyapunov_exp
import numpy as np

W = np.loadtxt('../lyapunov/W')
Win = np.loadtxt('../lyapunov/Win')
x = np.loadtxt('../lyapunov/x')
print(lyapunov_exp(W, Win, x))
