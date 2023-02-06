import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, b, L, h, rho

plt.style.use('dark_background')

'''
Define 1D FEA Model
'''
numberOfElements = 30
DOF = 2
element_type = 2
crack_point = L/3
crack_ratio = 0.0

numberOfNodes = numberOfElements + 1
x = sol.get_node_points_coords(numberOfNodes, L)
connectivityMatrix = sol.get_connectivity_matrix(numberOfElements, element_type)

# We don't need force vector for modal analysis
KG = sol.init_stiffness_force(numberOfNodes, DOF)[0]
MG = sol.init_stiffness_force(numberOfNodes, DOF)[0]
for elm in range(numberOfElements):
    n = sol.get_node_from_element(connectivityMatrix, elm, element_type)
    xloc = []
    for i in range(len(n)):
        xloc.append(x[n[i]])
    le = xloc[-1] - xloc[0]
    k = E * b * h**3 / (12 * le**3)
    m = rho * b * h * le / 420
    if xloc[0] <= crack_point <= xloc[-1]:
        k = E * (b - crack_ratio * b) * (h - crack_ratio * h)**3 / (12 * le ** 3)
    kloc = k * np.array([[12, 6 * le, -12, 6 * le],
                         [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                         [-12, -6 * le, 12, -6 * le],
                         [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    mloc = m * np.array([[156, 22 * le, 54, -13 * le],
                         [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                         [54, 13 * le, 156, -22 * le],
                         [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
    iv = [DOF * n[0], DOF * n[0] + 1, DOF * n[1], DOF * n[1] + 1]
    KG = KG + sol.assemble_stiffness(kloc, iv, numberOfNodes * DOF)
    MG = MG + sol.assemble_stiffness(mloc, iv, numberOfNodes * DOF)


KG = sol.impose_boundary_condition(KG, 0, 0)[0]
KG = sol.impose_boundary_condition(KG, -2, 0)[0]
MG = sol.impose_boundary_condition(MG, 0, 0)[0]
MG = sol.impose_boundary_condition(MG, -2, 0)[0]
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(MG) @ KG)

"""
POST PROCESSING
"""
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
eigenvalues = np.sqrt(eigenvalues)/(2 * np.pi)
eigenvalues = np.flip(eigenvalues)[2:]
eigenvectors = np.flip(eigenvectors).T[2:]
disp = np.zeros((4, numberOfNodes))
for i in range(4):
    for j in range(numberOfNodes):
        disp[i, j] = eigenvectors[i, 2*j+1]
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
ax[0][0].plot(x, disp[0])
ax[0][0].set_title("Mode Shape 1, Frequency : {f}".format(f=eigenvalues[0]))
ax[0][1].plot(x, disp[1])
ax[0][1].set_title("Mode Shape 2, Frequency : {f}".format(f=eigenvalues[1]))
ax[1][0].plot(x, disp[2])
ax[1][0].set_title("Mode Shape 3, Frequency : {f}".format(f=eigenvalues[2]))
ax[1][1].plot(x, disp[3])
ax[1][1].set_title("Mode Shape 1, Frequency : {f}".format(f=eigenvalues[3]))
fig.suptitle("Modes and Frequencies for crack ration : {c}".format(c=crack_ratio))
fig.subplots_adjust(wspace=0.1, hspace=0.45)
for axx in ax:
    for axxx in axx:
        axxx.set_yticks([])
plt.show()
