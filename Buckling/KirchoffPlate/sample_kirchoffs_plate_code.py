import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt
from parameters import L
import keywords as param
import kirchoffplate as kirk
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('dark_background')

H = L/100
DIMENSION = 2
nx = 14
ny = 14
lx = L
ly = L
connectivityMatrix, nodalArray, (X0, Y0)  = kirk.get_2d_connectivity(nx, ny, L, L)
numberOfElements = connectivityMatrix.shape[0]
DOF = 6
element_type = param.ElementType.LINEAR
nodePerElement = element_type ** DIMENSION
OVERRIDE_REDUCED_INTEGRATION = False
GAUSS_POINTS_REQ = 3
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
Dmat = np.zeros((7, 7))
for igp in range(len(weightOfGaussPts)):
    Zmat = kirk.get_z_matrix(gaussPts[igp] * H / 2)
    Dmat += Zmat.T @ kirk.get_elasticity() @ Zmat * H/2 * weightOfGaussPts[igp]
for elm in range(numberOfElements):
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(element_type**DIMENSION):
        xloc.append(nodalArray[1][n[i]])
        yloc.append(nodalArray[2][n[i]])
    xloc = np.array(xloc)[:, None]
    yloc = np.array(yloc)[:, None]
    kloc, floc = sol.init_stiffness_force(element_type**DIMENSION, DOF)
    for x_igp in range(len(weightOfGaussPts)):
        for y_igp in range(len(weightOfGaussPts)):
            B, J = kirk.get_bmat(gaussPts[x_igp], gaussPts[y_igp], xloc, yloc)
            N = kirk.get_nmat(gaussPts[x_igp], gaussPts[y_igp], J)
            kloc += B.T @ Dmat @ B * weightOfGaussPts[x_igp] * weightOfGaussPts[y_igp] * np.linalg.det(J)
            floc += N.T @ np.array([[0, 0, -1000000]]).T * weightOfGaussPts[x_igp] * weightOfGaussPts[y_igp] * np.linalg.det(J)
    iv = np.array(sol.get_assembly_vector(DOF, n))
    fg[iv[:, None], 0] += floc
    KG[iv[:, None], iv] += kloc

encastrate = np.where((np.isclose(nodalArray[1], 0)) | (np.isclose(nodalArray[1], lx)) | (np.isclose(nodalArray[2], 0)) | (np.isclose(nodalArray[2], ly)))[0]
iv = sol.get_assembly_vector(DOF, encastrate)
print(np.sum(fg))
for ibc in iv:
    KG, fg = sol.impose_boundary_condition(KG, fg, ibc, 0)

u = sol.get_displacement_vector(KG, fg)

u0 = []
v0 = []
theta_x = []
theta_y = []
w0 = []
for i in range(numberOfNodes):
    u0.append(u[DOF * i][0])
    v0.append(u[DOF * i + 1][0])
    w0.append(u[DOF * i + 2][0])
reqN, zeta, eta = kirk.get_node_from_cord(connectivityMatrix, (0.5, 0.5), nodalArray, numberOfElements, nodePerElement)
if reqN is None:
    raise Exception("Chose a position inside plate plis")
# Nmat, Nmat1, Nmat2, Nmat3 = kirk.get_hermite_shape_fn_re(eta, zeta, justN = True)
# wt = np.array([u[DOF * i + 2][0] for i in reqN])[:, None]
# wxt = np.array([u[DOF * i + 3][0] for i in reqN])[:, None]
# wyt = np.array([u[DOF * i + 4][0] for i in reqN])[:, None]
# wxyt = np.array([u[DOF * i + 5][0] for i in reqN])[:, None]
#
# xxx = Nmat.T @ wt + Nmat1.T @ wxt + Nmat2.T @ wyt + Nmat3.T @ wxyt
xxx = max(w0, key=abs)
w0 = - np.array(w0)
w0 = w0.reshape((ny + 1, nx + 1))
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
#w0 = - np.array(w0) * 1 / np.min(w0) # Scaled w to make it look better
print(w0)

ax.plot_wireframe(X0, Y0, w0)
ax.set_title("w0 is scaled to make graph look prettier")
ax.set_axis_off()
fig2, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.contourf(X0, Y0, w0, 70, cmap='jet')
ax.set_title('Contour Plot, w_A = {x}'.format(x = xxx))
ax.set_xlabel('_x')
ax.set_ylabel('_y')
ax.set_aspect('equal')
plt.show()
