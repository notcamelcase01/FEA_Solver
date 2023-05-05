import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt


DIMENSION = 2
element_type = 2
Elasticity = 30 * 10 ** 6
mu = 0.3
b = 0.01
h = 0.001
a = 0.1
nx = 1
ny = 1
lx = a
ly = b
f0 = 100
C = np.array([[Elasticity / (1 - mu ** 2), mu * Elasticity / (1 - mu ** 2), 0],
              [mu * Elasticity / (1 - mu ** 2), Elasticity / (1 - mu ** 2), 0],
              0, 0, Elasticity / (2 * (1 + mu))])


icon, nodal_array, (X, Y) = sol.get_2d_connectivity(nx, ny, lx, ly)
numberOfElements = icon.shape[0]
DOF = 2
GAUSS_POINTS_REQ = 2
numberOfNodes = nodal_array.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
nodePerElement = element_type ** DIMENSION
u0 = np.ones((numberOfNodes * DOF, 1))
f_app = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 0.1 * f0
for iter__ in range(10):
    KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
    T, r = sol.init_stiffness_force(numberOfNodes, DOF)
    for iter_ in range(100):
        for elm in range(numberOfElements):
            n = icon[elm][1:]
            xloc = []
            yloc = []
            for i in range(nodePerElement):
                xloc.append(nodal_array[1][n[i]])
                yloc.append(nodal_array[2][n[i]])
            xloc = np.array(xloc)[:, None]
            yloc = np.array(yloc)[:, None]
            iv = np.array(sol.get_assembly_vector(DOF, n, [0]))
            uloc = u0[iv[:, None], 0]
            iv = np.array(sol.get_assembly_vector(DOF, n, [1]))
            vloc = u0[iv[:, None], 0]
            kloc, floc = sol.init_stiffness_force(nodePerElement, DOF)
            tloc, rloc = sol.init_stiffness_force(nodePerElement, DOF)
            for x_gp in range(len(weightOfGaussPts)):
                for y_gp in range(len(weightOfGaussPts)):
                    N, Bx, By = sol.get_lagrange_shape_function(gaussPts[x_gp], gaussPts[y_gp])
                    J = np.zeros((2, 2))
                    J[0, 0] = Bx.T @ xloc
                    J[0, 1] = Bx.T @ yloc
                    J[1, 0] = By.T @ xloc
                    J[1, 1] = By.T @ yloc
                    Jinv = np.linalg.inv(J)
                    Bx, By = J[0, 0] * Bx + J[0, 1] * By, J[1, 0] * Bx + J[1, 1] * By
                    du_x = Bx.T @ uloc
                    du_y = By.T @ uloc
                    dv_x = Bx.T @ vloc
                    dv_y = By.T @ uloc
                    E = np.zeros((3, 1))
                    E[0, 0] = 0.5 * (du_x + du_x + du_x * du_x + dv_x * dv_x)
                    E[1, 0] = 0.5 * (dv_y + dv_y + du_y * du_y + dv_y * dv_y)
                    E[2, 0] = 0.5 * (du_y + dv_x + du_x * du_y + dv_x * dv_y)
                    S = C @ E
                    L = np.zeros((3, 2 * element_type))
                    L[0, 0 : 2] = Bx.T
                    L[1, 2 : 4] = By.T
                    L[2, 0 : 2] = By.T
                    L[2, 2 : 4] = Bx.T
                    Lu = np.zeros_like(L)
                    Lu[0, 0 : 2] = du_x * Bx.T
                    Lu[0, 2 : 4] = dv_x * Bx.T
                    Lu[1, 0 : 2] = du_y * Bx.T
                    Lu[1, 2 : 4] = dv_y * By.T
                    Lu[2, 0 : 2] = du_x * By.T + du_y * Bx.T
                    Lu[2, 2 : 4] = dv_x * Bx.T + dv_y * By.T
                    Bl = L + Lu
                    Bnl = np.zeros((4, 2 * element_type))
                    Bnl[0, 0: 2] = Bx.T
                    Bnl[1, 0: 2] = By.T
                    Bnl[1, 2: 4] = dv_y * By.T
                    Bnl[2, 0: 2] = du_x * By.T + du_y * Bx.T
                    Bnl[2, 2: 4] = dv_x * Bx.T + dv_y * By.T