import numpy as np


dNdrs = np.array([[0, -0.5, 0, 0, 0, 0.5, 0, 0, 0],
               [0, 0, 0, .5, 0, 0, 0, -0.5, 0]])

xy = np.array([[0,  0 , 0, 5, 10, 10, 10, 5, 5],
               [0, 5, 10, 10, 10, 5, 0, 1, 5]])


print(dNdrs @ xy.T)