import numpy as np
import solver2d as sol

def get_lagrange_function(x, y, xi = (-1, 0, 1, 1, 1, 0, -1, -1, 0),
                                  yi = (-1, -1, -1, 0, 1, 1, 1, 0, 0)):
    N = np.zeros(9)
    Nx = np.zeros(9)
    Ny = np.zeros(9)
    for i in range(len(xi)):
        N[i] = ((1.5 * xi[i]**2 - 1) * x**2 + 0.5 * xi[i] * x + 1 - xi[i]**2) * ((1.5 * yi[i]**2 - 1) * y**2 + 0.5 * yi[i] * y + 1 - yi[i]**2)
        Nx[i] = ((1.5 * xi[i]**2 - 1) * x * 2 + 0.5 * xi[i]) * ((1.5 * yi[i]**2 - 1) * y**2 + 0.5 * yi[i] * y + 1 - yi[i]**2)
        Ny[i] = ((1.5 * xi[i]**2 - 1) * x**2 + 0.5 * xi[i] * x + 1 - xi[i]**2) * ((1.5 * yi[i]**2 - 1) * y * 2 + 0.5 * yi[i])
    return N[:, None], Nx[:, None], Ny[:, None]

w, e = sol.init_gauss_points(3)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=False)
for i in range(3):
    for j in range(3):
        N, A, B = get_lagrange_function(e[i], e[j])
        print(np.sum(N), np.sum(B), np.sum(A))


