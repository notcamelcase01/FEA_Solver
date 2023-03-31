"""
All units in metric system plox
"""
import  numpy as np

E = 30*10**6
mu = 0.3
G = E/(2*(1 + mu))
Eb = E/(1 - mu**2)
f0 = 0, -0., 0
F = 0, -1., 0
L = 1
k = 5/6
q0 = 1
h1 = 1/1000
h0 = 5 * h1
b = 100 * h1
a = 100 * h1
Rx = .1
Ry = np.inf
alpha = a / 2 / Rx