import numpy as np
import matplotlib.pyplot as plt

j = np.array([[0, -0.5, 0, 0, 0, 0.5, 0, 0, 0],
              [0, 0, 0, 0.5, 0, 0, 0, -0.5, 0]])

x = np.array([[0, 0, 0, 5, 10, 10, 10, 5, 5],
              [0, 5, 10, 10, 10, 5, 0, 0, 5]])

def f1(e):
    return 10/e + 1.8 + 6/5, 6/5 - 15/e - 4.2

def f2(e):
    return 6/5 - 36*e/125, -5/e - 6/5 + 36*e/25

print(f1(.01))
print(f2(.01))
