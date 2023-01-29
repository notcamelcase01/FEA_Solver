"""
IGNORE THIS FILE ITS GONNA BE SPAGATTI CODE
"""

import numpy as np

def f1(e):
    return  -1 + e*6 + 36*e**2

def f2(e):
    return -e * 6 - 36 * e ** 2

def f3(e):
    return 1 + e/3

def f4(e):
    return 1/12 + e/72 + e**2 *.3426,10/3 -e/36 + 3.329*e**2

def f5(e):
    return -1 + -.5*e -3/4*e**2,1*e,1 - .5*e - 3/8*e**2

def f6(e):
    return 1 - e +  2*e**2 - 6*e**3, 2 + 2*e - 2*e**2 + 6*e**3

def f7(e):
    return -1 - e/(2*np.exp(1)) + 3/(8*np.exp(2))*e

def f8(e):
    return 1,-1/e - 1

def f9(e):
    return 1, 1/np.sqrt(e) - 1/2, - 1/np.sqrt(e) - 1/2
e = .0001
print(f7(e))

