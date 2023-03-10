import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, b, f0, L, F
import keywords as param

plt.style.use('dark_background')


def get_height(xp):
    """
    Return height as it vary
    :param xp: x coord
    :return: height
    """
    return .01 * (1 - xp * 0.5 / L)

'''
Define 1D FEA Model
'''
numberOfElements = 20
DOF = 2
qx = L/2
element_type = 2

numberOfNodes = numberOfElements + 1
x = sol.get_node_points_coords(numberOfNodes, L)
connectivityMatrix = sol.get_connectivity_matrix(numberOfElements, element_type)
weightOfGaussPts, gaussPts = sol.init_gauss_points(3)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
fg[-2] = F[1]
for elm in range(numberOfElements):
    n = sol.get_node_from_element(connectivityMatrix, elm, element_type)
    xloc = []
    for i in range(len(n)):
        xloc.append(x[n[i]])
    le = xloc[-1] - xloc[0]
    Jacobian = le / 2
    kloc, floc = sol.init_stiffness_force(element_type, DOF)
    for igp in range(len(weightOfGaussPts)):
        xx = 0.5 * (xloc[-1] + xloc[0]) + 0.5 * (xloc[-1] - xloc[0]) * gaussPts[igp]
        Nmat, Bmat = sol.get_hermite_fn(gaussPts[igp], Jacobian)
        Moi = b * get_height(xx) ** 3 / 12
        kloc += E * Moi * np.outer(Bmat, Bmat) * Jacobian * weightOfGaussPts[igp]
        f1 = sol.get_body_force(.25 * L, .75 * L, xloc, f0)
        floc += Nmat * f1[1] * Jacobian * weightOfGaussPts[igp]

    iv = [DOF * n[0], DOF * n[0] + 1, DOF * n[1], DOF * n[1] + 1]
    fg = fg + sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG = KG + sol.assemble_stiffness(kloc, iv, numberOfNodes * DOF)

print(np.sum(fg))
KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
KG, fg = sol.impose_boundary_condition(KG, fg, 1, 0)
u = sol.get_displacement_vector(KG, fg)
print(u[-2])
'''
Post-processing
'''
disp = np.zeros(numberOfNodes)
deflection = np.zeros(numberOfNodes)
for i in range(numberOfNodes):
    disp[i] = u[2 * i]
    deflection[i] = u[2 * i + 1]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(x, disp, '-o')
ax.set_xlabel("x")
ax.set_ylabel("transverse displacement")
z_across = np.linspace(-get_height(qx)/2, get_height(qx)/2, 20)
strain_xx = np.zeros(len(z_across))
for i in range(numberOfElements):
    n = sol.get_node_from_element(connectivityMatrix, i, element_type)
    xloc = []
    for j in range(len(n)):
        xloc.append(x[n[j]])
    uloc = np.array([u[2 * n[0]], u[2 * n[0] + 1], u[2 * n[1]], u[2 * n[1] + 1]])
    if xloc[0] <= qx <= xloc[-1]:
        eta = -1 + 2 * (qx - xloc[0]) / (xloc[-1] - xloc[0])
        Nmat, Bmat = sol.get_hermite_fn(eta, 0.5 * (xloc[-1] - xloc[0]))
        disp = np.dot(Nmat.T, uloc)[0][0]
        for jj in range(len(z_across)):
            strain_xx[jj] = -z_across[jj] * Bmat.T @ uloc
        ax.set_title('Displacements at {qx}m: {dp}m'.format(qx=qx, dp=disp))
        ax.scatter(qx, disp, color="#000000")
        break
fig,  (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 5))
ax2.plot(strain_xx, z_across, marker="o")
ax2.set_xlabel("$\\epsilon_{xx}$")
ax2.set_ylabel("Z")
ax2.set_title("Strain at top fiber : {ttt}".format(ttt=strain_xx[-1]))
ax3.plot(strain_xx * E, z_across, marker="o")
ax3.set_xlabel("$\\sigma_{xx}$")
ax3.set_ylabel("Z")
ax3.set_title("Stress at top fiber : {ttt}".format(ttt=strain_xx[-1] * E))
plt.show()

