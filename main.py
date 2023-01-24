import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, b, f0, L
import keywords as param

'''
Define 1D FEA Model
'''
plt.style.use('dark_background')
numberOfElements = 29
DOF = 2
element_type = param.ElementType.LINEAR

numberOfNodes = numberOfElements + 1
x = sol.get_node_points_coords(numberOfNodes, L)
connectivityMatrix = sol.get_connectivity_matrix(numberOfElements, element_type)
weightOfGaussPts, gaussPts = sol.init_gauss_points(3)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)

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
        Moi = b * (.005 + .005 * xx) ** 3 / 12
        kloc += E * Moi * np.outer(Bmat, Bmat) * Jacobian * weightOfGaussPts[igp]
        Nmat = Nmat.reshape(Nmat.shape[0], 1)
        floc += -Nmat * f0 * Jacobian * weightOfGaussPts[igp]

    iv = [DOF * n[0], DOF * n[0] + 1, DOF * n[1], DOF * n[1] + 1]
    fg = fg + sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG = KG + sol.assemble_stiffness(kloc, iv, numberOfNodes * DOF)

KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
KG, fg = sol.impose_boundary_condition(KG, fg, 1, 0)
u = sol.get_displacement_vector(KG, fg)

'''
Post-processing
'''
disp = np.zeros(numberOfNodes)
deflection = np.zeros(numberOfNodes)
for i in range(numberOfNodes):
    disp[i] = u[2 * i]
    deflection[i] = u[2 * i + 1]
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
ax.plot(x, disp, '-o')
ax.set_xlabel("x")
ax.set_ylabel("transverse displacement")
plt.show()
