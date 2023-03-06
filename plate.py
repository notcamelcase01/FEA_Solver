import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt
from parameters import E, b, f0, L, F, k, G
import keywords as param
import gencon as gencon
import kirchoffplate as kirk

plt.style.use('dark_background')

H = L/100
DIMENSION = 2
nx = 2
ny = 2
connectivityMatrix, nodalArray = gencon.get_2d_connectivity(nx, ny, L, L)
numberOfElements = connectivityMatrix.shape[0]
DOF = 6
element_type = param.ElementType.LINEAR
OVERRIDE_REDUCED_INTEGRATION = False
GAUSS_POINTS_REQ = 1
numberOfNodes = nodalArray.shape[0]
weightOfGaussPts, gaussPts = sol.init_gauss_points(1)
reduced_wts, reduced_gpts = sol.init_gauss_points(1 if (not OVERRIDE_REDUCED_INTEGRATION and
                                                        element_type == param.ElementType.LINEAR) else GAUSS_POINTS_REQ)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
for elm in range(numberOfElements):
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(element_type**DIMENSION):
        xloc.append(nodalArray[n[i]][1])
        yloc.append(nodalArray[n[i]][2])
    kloc, floc = sol.init_stiffness_force(element_type**DIMENSION, DOF)
    Dmat = np.zeros((7, 7))
    for igp in range(len(weightOfGaussPts)):
        Zmat = kirk.get_z_matrix(gaussPts[igp])
        Dmat += Zmat.T @ kirk.get_elasticity() @ Zmat * H/2
    for x_igp in range(len(weightOfGaussPts)):
        for y_igp in range(len(weightOfGaussPts)):
            Lmat, Lmatx, Lmaty = kirk.get_lagrange_shape_function(gaussPts[x_igp], gaussPts[y_igp])
            Nmat, Nmat1, Nmat2, Nmat3 = kirk.get_hermite_shape_function(gaussPts[x_igp], gaussPts[y_igp])
            Nmatxx, Nmat1xx, Nmat2xx, Nmat3xx = kirk.get_hermite_shape_function_derivative_xx(gaussPts[x_igp], gaussPts[y_igp])
            Nmatyy, Nmat1yy, Nmat2yy, Nmat3yy = kirk.get_hermite_shape_function_derivative_yy(gaussPts[x_igp], gaussPts[y_igp])
            Nmatxy, Nmat1xy, Nmat2xy, Nmat3xy = kirk.get_hermite_shape_function_derivative_xy(gaussPts[x_igp], gaussPts[y_igp])
            J = np.zeros((2, 2))
            J[0, 0] = Lmatx.T @ xloc
            J[0, 1] = Lmatx.T @ yloc
            J[1, 0] = Lmaty.T @ xloc
            J[1, 1] = Lmaty.T @ yloc
            H = np.zeros((3, 3))
            H[0, 0] = Nmatxx.T @ xloc
