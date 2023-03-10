import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt
from parameters import L
import keywords as param
from Plates import gencon as gencon
import mindlinplate as mind
plt.style.use('dark_background')
from mpl_toolkits.mplot3d import Axes3D


H = L/100
DIMENSION = 2
# DISCRITIZATION OF HOLE
"""
PLEASE THIS DISCRITIZATION IS A BIT DELICATE TRY TO KEEP NUMER OF ELEMENTS ON HOLE LESS THAN BIG PLATE
IT WOULD WORK REGARDLESS JUST IT ACTS UP SOMETIMES ALSO ALSO Hx,Hy > 1
"""
Hx = 2
Hy = 2
by_max = 0.3
by_min = 0.1
bx_max = 0.9
bx_min = 0.7
# DISCRITIZATION OF PLATE
nx = 8
ny = 8
lx = L
ly = L
connectivityMatrix, nodalArray, (X, Y), nodalArray1 = gencon.get_2d_connectivity_hole(nx, ny, lx, ly, Hx, Hy, by_max, by_min, bx_max, bx_min)
numberOfElements = connectivityMatrix.shape[0]
DOF = 5
element_type = param.ElementType.LINEAR
GAUSS_POINTS_REQ = 3
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
reduced_wts, reduced_gpts = sol.init_gauss_points(1)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
nodePerElement = element_type**DIMENSION
D1mat = np.zeros((8, 8))
D2mat = np.zeros((4, 4))
for igp in range(len(weightOfGaussPts)):
    Z1mat = mind.get_z1_matrix(0.5 * H * gaussPts[igp])
    Z2mat = mind.get_z2_matrix(0.5 * H * gaussPts[igp])
    D1mat += Z1mat.T @ mind.get_C1_matrix() @ Z1mat * 0.5 * H * weightOfGaussPts[igp]
    D2mat += Z2mat.T @ mind.get_C2_matrix() @ Z2mat * 0.5 * H * weightOfGaussPts[igp]
hole_elements = []
for elm in range(numberOfElements):
    KK = 1
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(nodePerElement):
        xloc.append(nodalArray1[1][n[i]])
        yloc.append(nodalArray1[2][n[i]])
    if np.isnan(np.sum(xloc)) or np.isnan(np.sum(yloc)):
        """
        CHECKING IF ELEMENT IS IN HOLE
        if THERE IS HOLE THEN STIFFNESS WILL BE 0 
        """
        KK = 0
        hole_elements.append(elm)
    xloc = []
    yloc = []
    for i in range(nodePerElement):
        xloc.append(nodalArray[1][n[i]])
        yloc.append(nodalArray[2][n[i]])
    xloc = np.array(xloc)[:, None]
    yloc = np.array(yloc)[:, None]
    kloc, floc = sol.init_stiffness_force(nodePerElement, DOF)
    for xgp in range(len(weightOfGaussPts)):
        for ygp in range(len(weightOfGaussPts)):
            N, Nx, Ny = mind.get_lagrange_shape_function(gaussPts[xgp], gaussPts[ygp])
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            T1 = np.zeros((8, 8))
            for pli in range(4):
                T1 += sol.assemble_2Dmat(Jinv, [2 * pli, 2 * pli + 1], 8)
            B1 = T1 @ mind.get_B1_matrix(Nx, Ny)
            kloc +=  B1.T @ D1mat @ B1 * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
            floc += (mind.get_N_matrix(N).T @ np.array([[0, 0, 0, 0, -1000000]]).T) * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
    for xgp in range(len(reduced_wts)):
        for ygp in range(len(reduced_wts)):
            N, Nx, Ny = mind.get_lagrange_shape_function(reduced_gpts[xgp], reduced_gpts[ygp])
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            T2 = np.zeros((4, 4))
            T2 += sol.assemble_2Dmat(Jinv, [2 * 1, 2 * 1 + 1], 4)
            T2[0, 0] = 1
            T2[1, 1] = 1
            B2 = T2 @ mind.get_B2_matrix(N, Nx, Ny)
            kloc += B2.T @ D2mat @ B2 * reduced_wts[xgp] * reduced_wts[ygp] * np.linalg.det(J)
    iv = mind.get_assembly_vector(DOF, n)
    fg += KK * sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG += KK * sol.assemble_2Dmat(kloc, iv, numberOfNodes * DOF)


print(hole_elements)
# for i in iv:
#     fg[i] = 0
encastrate = np.where((np.isclose(nodalArray[1], 0)) | (np.isclose(nodalArray[1], lx)) | (np.isclose(nodalArray[2], 0)) | (np.isclose(nodalArray[2], ly)))[0]
iv = sol.get_assembly_vector(DOF, encastrate)
for i in iv:
    KG, fg = sol.impose_boundary_condition(KG, fg, i, 0)

u = sol.get_displacement_vector(KG, fg)
# iv = sol.get_assembly_vector(DOF, [30])
# for i in iv:
#     u[i] = 0
u0 = []
v0 = []
theta_x = []
theta_y = []
w0 = []
for i in range(numberOfNodes):
    u0.append(u[DOF * i][0])
    v0.append(u[DOF * i + 1][0])
    theta_x.append(u[DOF * i + 2][0])
    theta_y.append(u[DOF * i + 3][0])
    w0.append(u[DOF * i + 4][0])

reqN, zeta, eta = mind.get_node_from_cord(connectivityMatrix, (0.7, 0.3), nodalArray1, numberOfElements, nodePerElement)
if reqN is None:
    raise Exception("Chose a position inside plate plis")
N, Nx, Ny = mind.get_lagrange_shape_function(zeta, eta)
wt = np.array([u[DOF * i + 4][0] for i in reqN])[:, None]
xxx = N.T @ wt
print(xxx)
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
w0 = - np.array(w0) * 0.3 / np.min(w0) # Scaled w to make it look better
w0 = w0.reshape((ny + Hy + 1, Hx + nx + 1))
ax.plot_wireframe(X, Y, w0)
ax.set_aspect('equal')
ax.set_title("w0 is scaled to make graph look prettier")
ax.set_axis_off()
plt.show()

