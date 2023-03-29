"""
All units in metric system plox
"""
import  numpy as np

E = 0.45*10**6
mu = 0.3
G = E/(2*(1 + mu))
Eb = E/(1 - mu**2)
f0 = 0, -0., 0
F = 0, -1., 0
L = 1
k = 5/6
q0 = 0.04
h = 0.125/1000
alpha = 0.1
b = 20/1000
Rx = 100/1000
a = 2 * alpha * 100/1000
Ry = np.inf