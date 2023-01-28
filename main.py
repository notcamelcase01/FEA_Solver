import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, b, f0, L, F, k, G
import keywords as param

'''
Define 1D FEA Model
'''
numberOfElements = 200
DOF = 3
qx = L
element_type = param.ElementType.QUAD

numberOfNodes = (element_type - 1) * numberOfElements + 1
x = sol.get_node_points_coords(numberOfNodes, L)
connectivityMatrix = sol.get_connectivity_matrix(numberOfElements, element_type)
weightOfGaussPts, gaussPts = sol.init_gauss_points(3)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
fg[-2] = F

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
        Nmat, Bmat = sol.get_lagrange_fn(gaussPts[igp], Jacobian, element_type)
        h = (.01 - .005 * xx)
        Moi = b * h ** 3 / 12
        A = b * h
        Kuu = E * A * np.outer(Bmat, Bmat) * Jacobian * weightOfGaussPts[igp]
        Kww = k * G * A * np.outer(Bmat, Bmat) * Jacobian * weightOfGaussPts[igp]
        Kwt = k * G * A * np.outer(Bmat, Nmat) * Jacobian * weightOfGaussPts[igp]
        Ktt = (E * Moi * np.outer(Bmat, Bmat) + k * G * A * np.outer(Nmat, Nmat)) * Jacobian * weightOfGaussPts[igp]
        kloc += sol.get_timoshenko_stiffness(Kuu, Kww, Kwt, Ktt, element_type)
        # print(Nmat.shape)
        # print(floc.shape)
        # floc += -Nmat * f0 * Jacobian * weightOfGaussPts[igp]
    iv = sol.get_assembly_vector(DOF, n)
    # fg = fg + sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG += sol.assemble_stiffness(kloc, iv, numberOfNodes * DOF)

print(np.linalg.det(KG))
KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
KG, fg = sol.impose_boundary_condition(KG, fg, 1, 0)
KG, fg = sol.impose_boundary_condition(KG, fg, 2, 0)
u = sol.get_displacement_vector(KG, fg)
print(u)
'''
Post-processing
'''
ad = []
td = []
theta = []
for i in range(numberOfNodes):
    ad.append(u[3*i][0])
    td.append(u[3*i + 1][0])
    theta.append(u[3*i + 2][0])
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.plot(x, td)
plt.show()
