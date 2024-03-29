import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt
from parameters_q3 import alpha, Rx, Ry, q0, a, b, h1, h0, E, G, k, mu
import keywords as param
import gencon as gencon
import sanders as sand
import time
plt.style.use('dark_background')


H = h0
DIMENSION = 2
nx = 50
ny = 50
lx = a
ly = b
element_type = param.ElementType.Q4
OVERRIDE_REDUCED_INTEGRATION = False

connectivityMatrix, nodalArray, (X, Y) = gencon.get_2D_connectivity_Hybrid(nx, ny, lx, ly, element_type)
numberOfElements = connectivityMatrix.shape[0]
DOF = 5
GAUSS_POINTS_REQ = 2
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
reduced_wts, reduced_gpts = sol.init_gauss_points(1 if (not OVERRIDE_REDUCED_INTEGRATION and
                                                        element_type == param.ElementType.Q4) else GAUSS_POINTS_REQ)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
nodePerElement = element_type ** DIMENSION
nf, zeta, eta = sol.get_node_from_cord(connectivityMatrix, (lx/2, ly/2), nodalArray, numberOfElements, nodePerElement, element_type)

tik = time.time()
for elm in range(numberOfElements):
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(nodePerElement):
        xloc.append(nodalArray[1][n[i]])
        yloc.append(nodalArray[2][n[i]])
    xloc = np.array(xloc)[:, None]
    yloc = np.array(yloc)[:, None]
    kloc, floc = sol.init_stiffness_force(nodePerElement, DOF)
    if (nf == n).all():
        N, Nx, Ny = sand.get_lagrange_shape_function(zeta, eta, element_type)
        floc = sand.get_n_matrix(N).T @ np.array([[0, 0, q0, 0, 0]]).T
    for xgp in range(len(weightOfGaussPts)):
        for ygp in range(len(weightOfGaussPts)):
            N, Nx, Ny = sand.get_lagrange_shape_function(gaussPts[xgp], gaussPts[ygp], element_type)
            xx = N.T @ xloc
            H = h1 + (h0 - h1) * (1 - 2 * xx[0][0] / lx) ** 2
            D1mat = np.zeros((9, 9))
            for igp in range(len(weightOfGaussPts)):
                Z1mat = sand.get_z1_matrix(0.5 * H * gaussPts[igp], Rx, Ry)
                D1mat += Z1mat.T @ sand.get_C1_matrix(E, mu) @ Z1mat * 0.5 * H * weightOfGaussPts[igp]
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            T1 = np.zeros((9, 9))
            T1[0:2, 0:2] = Jinv
            T1[2:4, 2:4] = Jinv
            T1[4:6, 4:6] = Jinv
            T1[6:8, 6:8] = Jinv
            T1[8, 8] = 1
            B1 = T1 @ sand.get_b1_matrix(N, Nx, Ny)
            kloc += B1.T @ D1mat @ B1 * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
            floc += (sand.get_n_matrix(N).T @ np.array([[0, 0, 0, 0, 0]]).T) * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
    for xgp in range(len(reduced_wts)):
        for ygp in range(len(reduced_wts)):
            N, Nx, Ny = sand.get_lagrange_shape_function(reduced_gpts[xgp], reduced_gpts[ygp], element_type)
            xx = N.T @ xloc
            H = h0 + (h1 - h0) * np.sin(np.pi * xx[0][0] / lx)
            D2mat = np.zeros((6, 6))
            for igp in range(len(weightOfGaussPts)):
                Z2mat = sand.get_z2_matrix(Rx, Ry)
                D2mat += Z2mat.T @ sand.get_C2_matrix(G, k) @ Z2mat
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            T2 = np.eye(6)
            T2[2:4, 2:4] = Jinv
            B2 = T2 @ sand.get_b2_matrix(N, Nx, Ny)
            kloc += B2.T @ D2mat @ B2 * reduced_wts[xgp] * reduced_wts[ygp] * np.linalg.det(J)
    iv = np.array(sol.get_assembly_vector(DOF, n))
    fg[iv[:, None], 0] += floc
    KG[iv[:, None], iv] += kloc
encastrate = np.where((np.isclose(nodalArray[1], 0)) | (np.isclose(nodalArray[1], lx)))[0]
iv = sol.get_assembly_vector(DOF, encastrate)
for i in iv:
    KG, fg = sol.impose_boundary_condition(KG, fg, i, 0)
u = sol.get_displacement_vector(KG, fg)
tok = time.time()
print(tok - tik)
w0 = []
winit = (np.sqrt(Rx ** 2 - (nodalArray[1] - lx/2)**2) - Rx * np.cos(alpha))
for i in range(numberOfNodes):
    x = nodalArray[1][i]
    w0.append(u[DOF * i + 2][0])
reqN, zeta, eta = sol.get_node_from_cord(connectivityMatrix, (lx/2, ly/2), nodalArray, numberOfElements, nodePerElement, element_type)
if reqN is None:
    raise Exception("Chose a position inside plate plis")
N, Nx, Ny = sand.get_lagrange_shape_function(zeta, eta, element_type)
wt = np.array([w0[i] for i in reqN])[:, None]
xxx = N.T @ wt
print("In Curvilinear at mid point (mm)", xxx[0][0] * 1000)
wt = np.array([w0[i] + winit[i] for i in reqN])[:, None]
xxx = N.T @ wt
print("Adding curvature at mid point (mm)", xxx[0][0] * 1000)
w0 = np.array(w0)
#print(w0.reshape(((element_type - 1) * ny + 1, (element_type - 1) *  nx + 1)))
oi = abs(max(w0.min(), w0.max(), key=abs))
w0 = (w0 + winit)
w0 = w0.reshape(((element_type - 1) * ny + 1, (element_type - 1) *  nx + 1))
np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)
fig2 = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, w0)
ax.set_aspect('equal')
ax.set_title('w0 is scaled to make graph look prettier')
ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.contourf(X, Y, w0, 100, cmap='jet')
ax.set_title('Contour Plot')
ax.set_xlabel('_x')
ax.set_ylabel('_y')
ax.set_aspect('equal')

plt.show()
