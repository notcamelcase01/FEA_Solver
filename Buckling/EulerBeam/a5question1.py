import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, b, f0, L, F, Eb
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
        Moi = get_height() ** 3 / 12
        kloc += Eb * Moi * np.outer(Nxx, Nxx) * Jacobian * weightOfGaussPts[igp]
        gloc += np.outer(Nx, Nx) * Jacobian * weightOfGaussPts[igp]
    iv = np.array(sol.get_assembly_vector(DOF, n))
    KG[iv[:, None], iv] += kloc
    G[iv[:, None], iv] += gloc


KG = sol.impose_boundary_condition(KG, 0, 0)[0]
KG = sol.impose_boundary_condition(KG, -2, 0)[0]
G = sol.impose_boundary_condition(G, 0, 0)[0]
G = sol.impose_boundary_condition(G, -2, 0)[0]
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(G) @ KG)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
eigenvalues = np.flip(eigenvalues)[0:]
eigenvectors = np.flip(eigenvectors).T[0:]
print(eigenvalues)

