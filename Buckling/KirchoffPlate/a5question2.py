import numpy as np
import solver2d as sol
import matplotlib.pyplot as plt
import keywords as param
import kirchoffplate as kirk
import scipy as sc
from mpl_toolkits.mplot3d import Axes3D
import time
plt.style.use('dark_background')


H = 1/1000
L = H * 100
DIMENSION = 2
nx = 10
ny = 10
lx = L
ly = L

E = 30 * 10 ** 6
mu = 0.3
G = E * H ** 3 / (12 * (1 + mu))
E = E * H ** 3 / (12 * (1 - mu ** 2))
query_mode = 1
str_path = 'data_plate/'


connectivityMatrix, nodalArray, (X0, Y0)  = kirk.get_2d_connectivity(nx, ny, lx, ly)
numberOfElements = connectivityMatrix.shape[0]
DOF = 4
element_type = param.ElementType.LINEAR
nodePerElement = element_type ** DIMENSION
OVERRIDE_REDUCED_INTEGRATION = False
GAUSS_POINTS_REQ = 3
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINTS_REQ)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
GG = np.zeros_like(KG)
tik = time.time()
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
    gloc = np.zeros_like(kloc)
    for x_igp in range(len(weightOfGaussPts)):
        for y_igp in range(len(weightOfGaussPts)):
            Hmat, J = kirk.get_hermite_shapes(gaussPts[x_igp], gaussPts[y_igp], xloc, yloc)
            kloc += (-2 * G * (Hmat[[5], :].T @ Hmat[[5], :]) - E * (Hmat[[3], :].T @ Hmat[[3], :] + mu * Hmat[[3], :].T @ Hmat[[4], :]) - E * (Hmat[[4], :].T @ Hmat[[4], :] + mu * Hmat[[4], :].T @ Hmat[[3], :])) * weightOfGaussPts[x_igp] * weightOfGaussPts[y_igp] * np.linalg.det(J)
            gloc += -(Hmat[[2], :].T @ Hmat[[2], :]) * weightOfGaussPts[x_igp] * weightOfGaussPts[y_igp] * np.linalg.det(J)
    iv = np.array(sol.get_assembly_vector(DOF, n))
    GG[iv[:, None], iv] += gloc
    KG[iv[:, None], iv] += kloc
tok = time.time()
print(tok - tik)
# encastrate = np.where((np.isclose(nodalArray[2], ly)) | (np.isclose(nodalArray[2], 0)))[0]
counter = 0
encastrate = np.where((np.isclose(nodalArray[1], lx)) | (np.isclose(nodalArray[1], 0)))[0]
iv = sol.get_assembly_vector(DOF, encastrate, required_dof=[0, 2, 3])
for ibc in iv:
    counter += 1
    KG = sol.impose_boundary_condition(KG, ibc, 0)
    GG = sol.impose_boundary_condition(GG, ibc, 0)

encastrate = np.where((np.isclose(nodalArray[2], 0)))[0]
iv = sol.get_assembly_vector(DOF, encastrate, required_dof=[0, 1, 3])
for ibc in iv:
    counter += 1
    KG = sol.impose_boundary_condition(KG, ibc, 0)
    GG = sol.impose_boundary_condition(GG, ibc, 0)

eigenvalues, eigenvectors = sc.linalg.eig(KG, GG)
tok = time.time()
print(tok - tik)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real
idx = eigenvalues[:-counter].argsort()
eigenvalues[:-counter] = eigenvalues[idx]
eigenvectors[:, :-counter] = eigenvectors[:, idx]
np.save(str_path + 'eigval.npy', eigenvalues)
np.save(str_path + 'eigvec.npy', eigenvectors)
np.save(str_path + 'Y0.npy', Y0)
np.save(str_path + 'X0.npy', X0)
np.save(str_path + 'nxnydof.npy', np.array((nx, ny, DOF)))
w0 = eigenvectors[np.arange(0, len(eigenvalues), DOF), query_mode - 1]
w0 = w0.reshape((ny + 1, nx + 1))
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_wireframe(X0, Y0, w0)
ax.set_title("Buckling Mode - {y} at critical load of (N/m) : {x}".format(y = query_mode, x = eigenvalues[query_mode - 1]))
ax.set_axis_off()
ax.set_box_aspect(aspect = (1, 1, 1))
plt.show()




