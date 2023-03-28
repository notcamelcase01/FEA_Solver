import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt
from parameters import L
from parameters import L, Rx, Ry, h, q0, a, b, G
import keywords as param
from Shells import gencon as gencon
import mindlinplate as mind
plt.style.use('dark_background')


H = h
DIMENSION = 2
nx = 10
ny = 10
lx = a
ly = b
connectivityMatrix, nodalArray, (X, Y) = gencon.get_2D_connectivity_Q9(nx, ny, lx, ly)
numberOfElements = connectivityMatrix.shape[0]
DOF = 5
element_type = param.ElementType.QUADRATIC
GAUSS_POINTS_REQ = 3
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
reduced_wts, reduced_gpts = sol.init_gauss_points(3)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
nodePerElement = element_type**DIMENSION
D1mat = np.zeros((8, 8))
D2mat = np.zeros((4, 4))
for igp in range(len(weightOfGaussPts)):
    Z1mat = mind.get_z1_matrix(0.5 * H * gaussPts[igp])
    Z2mat = mind.get_z2_matrix(0.5 * H * gaussPts[igp])
    D1mat += Z1mat.T @ mind.get_C1_matrix() @ Z1mat * 0.5 * H * weightOfGaussPts[igp]
    D2mat += Z2mat.T @ mind.get_C2_matrix() @ Z2mat * 0.5 * H * weightOfGaussPts[igp]

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
    asd  = np.copy(kloc)
    for xgp in range(len(weightOfGaussPts)):
        xx = 0.5 * (xloc[3] + xloc[0]) + 0.5 * (xloc[3] - xloc[0]) * gaussPts[xgp]
        q = q0 * np.sin(xx * np.pi/lx)[0]
        for ygp in range(len(weightOfGaussPts)):
            N, Nx, Ny = mind.get_lagrange_shape_function(gaussPts[xgp], gaussPts[ygp])
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            T1 = np.zeros((8, 8))
            T1[0:2, 0:2] = Jinv
            T1[2:4, 2:4] = Jinv
            T1[4:6, 4:6] = Jinv
            T1[6:8, 6:8] = Jinv
            gSJDN = mind.get_B1_matrix(Nx, Ny)
            B1 = T1 @ mind.get_B1_matrix(Nx, Ny)
            kloc += B1.T @ D1mat @ B1 * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
            floc += (mind.get_N_matrix(N).T @ np.array([[0, 0, q, 0, 0]]).T) * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
    for xgp in range(len(reduced_wts)):
        for ygp in range(len(reduced_wts)):
            N, Nx, Ny = mind.get_lagrange_shape_function(reduced_gpts[xgp], reduced_gpts[ygp])
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            T2 = np.eye(4)
            T2[0:2, 0:2] = Jinv
            B2 = T2 @ mind.get_B2_matrix(N, Nx, Ny)
            asd +=  B2.T @ D2mat @ B2 * reduced_wts[xgp] * reduced_wts[ygp] * np.linalg.det(J)
            kloc += B2.T @ D2mat @ B2 * reduced_wts[xgp] * reduced_wts[ygp] * np.linalg.det(J)
    iv = sol.get_assembly_vector(DOF, n)
    fg += sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG += sol.assemble_2Dmat(kloc, iv, numberOfNodes * DOF)
print(sum(fg))
encastrate = np.where((np.isclose(nodalArray[1], 0)) | (np.isclose(nodalArray[1], lx)))[0]
iv = sol.get_assembly_vector(DOF, encastrate)
for i in iv:
    KG, fg = sol.impose_boundary_condition(KG, fg, i, 0)
print(sum(fg))

u = sol.get_displacement_vector(KG, fg)
u0 = []
v0 = []
theta_x = []
theta_y = []
w0 = []
for i in range(numberOfNodes):
    u0.append(u[DOF * i][0])
    v0.append(u[DOF * i + 1][0])
    theta_x.append(u[DOF * i + 3][0])
    theta_y.append(u[DOF * i + 4][0])
    w0.append(u[DOF * i + 2][0])
reqN, zeta, eta = sol.get_node_from_cord(connectivityMatrix, (0.5 * lx, 0.5 * ly), nodalArray, numberOfElements, nodePerElement)
if reqN is None:
    raise Exception("Chose a position inside plate plis")
N, Nx, Ny = mind.get_lagrange_shape_function(zeta, eta)
wt = np.array([u[DOF * i + 2][0] for i in reqN])[:, None]
xxx = N.T @ wt
print("jjj", xxx)
oi =  abs(max(w0, key=abs))
#print(np.array(w0).reshape(((element_type - 1) * ny + 1, (element_type - 1) * nx + 1)))
w0 = -np.array(w0) / oi * lx / 10 # Scaled w to make it look better
w0 = w0.reshape(((element_type - 1) * ny + 1, (element_type - 1) * nx + 1))
fig2 = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, w0)
ax.set_aspect('equal')
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.contourf(X, Y, w0, 100, cmap='jet')
ax.set_title('Contour Plot, w_A = {x}'.format(x = xxx))
ax.set_xlabel('_x')
ax.set_ylabel('_y')
ax.set_aspect('equal')
plt.show()
