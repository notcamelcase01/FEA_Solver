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
nx = 1
ny = 1
connectivityMatrix, nodalArray = gencon.get_2d_connectivity(nx, ny, L, L)
numberOfElements = connectivityMatrix.shape[0]
DOF = 6
element_type = param.ElementType.LINEAR
OVERRIDE_REDUCED_INTEGRATION = False
GAUSS_POINTS_REQ = 1
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
reduced_wts, reduced_gpts = sol.init_gauss_points(1 if (not OVERRIDE_REDUCED_INTEGRATION and
                                                        element_type == param.ElementType.LINEAR) else GAUSS_POINTS_REQ)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
for elm in range(numberOfElements):
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(element_type**DIMENSION):
        xloc.append(nodalArray[1][n[i]])
        yloc.append(nodalArray[2][n[i]])
    kloc, floc = sol.init_stiffness_force(element_type**DIMENSION, DOF)
    Dmat = np.zeros((7, 7))
    for igp in range(len(weightOfGaussPts)):
        Zmat = kirk.get_z_matrix(gaussPts[igp])
        Dmat += Zmat.T @ kirk.get_elasticity() @ Zmat * H/2
    Jx = -0.5 * (xloc[0] - xloc[2])
    Jy = -0.5 * (yloc[0] - yloc[1])
    for x_igp in range(len(weightOfGaussPts)):
        for y_igp in range(len(weightOfGaussPts)):
            Lmat, Lmatx, Lmaty = kirk.get_lagrange_shape_function(gaussPts[x_igp], gaussPts[y_igp], Jx, Jy)
            Nmat, Nmat1, Nmat2, Nmat3 = kirk.get_hermite_shape_function(gaussPts[x_igp], gaussPts[y_igp])
            Nmatxx, Nmat1xx, Nmat2xx, Nmat3xx = kirk.get_hermite_shape_function_derivative_xx(gaussPts[x_igp], gaussPts[y_igp], Jx)
            Nmatyy, Nmat1yy, Nmat2yy, Nmat3yy = kirk.get_hermite_shape_function_derivative_yy(gaussPts[x_igp], gaussPts[y_igp], Jy)
            Nmatxy, Nmat1xy, Nmat2xy, Nmat3xy = kirk.get_hermite_shape_function_derivative_xy(gaussPts[x_igp], gaussPts[y_igp], Jx, Jy)
            B = kirk.get_b_matrix(Lmatx, Lmaty, Nmatxx, Nmatyy, Nmatxy, Nmat1xx, Nmat1yy, Nmat1xy, Nmat2xx, Nmat2yy, Nmat2xy, Nmat3xx, Nmat3yy, Nmat3xy)
            N = kirk.get_n_matrix(Lmat, Nmat, Nmat1, Nmat2, Nmat3)
            kloc += B.T @ Dmat @ B * Jx * Jy * weightOfGaussPts[x_igp] * weightOfGaussPts[y_igp]
            floc += N.T @ np.array([0, 0, 1])[:, None] * weightOfGaussPts[x_igp] * weightOfGaussPts[y_igp] * Jx * Jy
    iv = sol.get_assembly_vector(DOF, n)
    fg += sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG += sol.assemble_stiffness(kloc, iv, numberOfNodes * DOF)

encastrate = np.where((nodalArray[1] == 0.0) | (nodalArray[2] == L))[0]
for i in encastrate:
    associated_nodes = [6*i + j for j in range(6)]
    for ibc in associated_nodes:
        KG, fg = sol.impose_boundary_condition(KG, fg, ibc, 0)

u = sol.get_displacement_vector(KG, fg)

print(u)