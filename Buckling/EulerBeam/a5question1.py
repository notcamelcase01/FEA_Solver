import numpy as np
import scipy as sc
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, L
import keywords as param

plt.style.use('dark_background')


def get_height():
    """
    Return height as it vary
    :return: height
    """
    return L/100

'''
Define 1D FEA Model
'''
numberOfElements = 20
DOF = 2
element_type = 2
b = get_height()

numberOfNodes = numberOfElements + 1
x = sol.get_node_points_coords(numberOfNodes, L)
connectivityMatrix = sol.get_connectivity_matrix(numberOfElements, element_type)
weightOfGaussPts, gaussPts = sol.init_gauss_points(3)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
G = np.zeros_like(KG)
for elm in range(numberOfElements):
    n = sol.get_node_from_element(connectivityMatrix, elm, element_type)
    xloc = []
    for i in range(len(n)):
        xloc.append(x[n[i]])
    le = xloc[-1] - xloc[0]
    Jacobian = le / 2
    kloc, floc = sol.init_stiffness_force(element_type, DOF)
    gloc = np.zeros_like(kloc)
    for igp in range(len(weightOfGaussPts)):
        N, Nx, Nxx = sol.get_hermite_fn(gaussPts[igp], Jacobian)
        Moi = b * get_height() ** 3 / 12
        kloc += E * Moi * np.outer(Nxx, Nxx) * Jacobian * weightOfGaussPts[igp]
        gloc += np.outer(Nx, Nx) * Jacobian * weightOfGaussPts[igp]
    iv = np.array(sol.get_assembly_vector(DOF, n))
    KG[iv[:, None], iv] += kloc
    G[iv[:, None], iv] += gloc


KG = sol.impose_boundary_condition(KG, 0, 0)
KG = sol.impose_boundary_condition(KG, -2, 0)
G = sol.impose_boundary_condition(G, 0, 0)
G = sol.impose_boundary_condition(G, -2, 0)
eigenvalues, eigenvectors = sc.linalg.eig(KG, G)
eigenvalues = eigenvalues.real
idx = eigenvalues[:-2].argsort()
print(idx, idx.shape)
eigenvalues[:-2] = eigenvalues[idx]
eigenvectors[:, :-2] = eigenvectors[:, idx]
print(eigenvalues)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(x, eigenvectors[np.arange(0, len(eigenvalues), DOF), 0])
ax.set_title("Mode 1 buckling , at critical load of : {x}".format(x = eigenvalues[0]))
plt.show()
