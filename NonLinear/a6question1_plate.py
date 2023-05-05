import numpy as np
import solver_non_linear as sol
import matplotlib.pyplot as plt

def get_b(y):
    return 0.005 + 0.005 * y / 0.1
