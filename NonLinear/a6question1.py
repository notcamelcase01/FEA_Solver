import numpy as np
import solver_non_linear as sol
import matplotlib.pyplot as plt

def get_b(y):
    return 0.005 + 0.005 * y / 0.1

f0 = 100
Elasticity = 30 * 10 ** 6
b = 0.01
h = 0.001
a = 0.1
Area = b * h
element_type = 2
DOF = 1
DIMENSION = 1
numberOfElements = 400
numberOfNodes = numberOfElements + 1
nodePeElement = element_type ** DIMENSION
wgp, egp = sol.init_gauss_points(2)
x = sol.get_node_points_coords(numberOfNodes, a)
icon = sol.get_connectivity_matrix(numberOfElements, element_type)
u0 = np.ones((numberOfNodes, 1))
f_app = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 0.1 * f0 * b
for iter__ in range(1): #Force Increment
    u0 = np.ones((numberOfNodes, 1))
    KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)

    for iter_ in range(100):
        for elm in range(numberOfElements):
            n = sol.get_node_from_element(icon, elm, element_type)
            xloc = []
            iv = sol.get_assembly_vector(DOF, n)
            for i in range(len(n)):
                xloc.append(x[n[i]])
            le = xloc[-1] - xloc[0]
            Jacobian = le / 2
            xloc = np.array(xloc)[:, None]
            iv = np.array(sol.get_assembly_vector(DOF, n))
            uloc = u0[iv[:, None], 0]
            kloc, floc = sol.init_stiffness_force(nodePeElement, DOF)
            for igp in range(len(wgp)):
               N, B = sol.get_lagrange_interpolation_fn_1d(egp[igp], element_type)
               xx = N.T @ xloc
               B = B / Jacobian
               du = B.T @ uloc
               Bnl =  B
               Bl = (B + du * B)
               E =  du + du ** 2 * 0.5
               kloc += Elasticity * Area * (Bl @ Bl.T + E * Bnl @ Bnl.T) * wgp[igp] * Jacobian
               floc += (N *  f_app[iter__] * np.sin(np.pi * xx / a) - Bl * E * Elasticity * Area) * Jacobian * wgp[igp]
            fg[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc
        KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
        uN  =  np.linalg.solve(KG, fg)
        if np.linalg.norm(np.abs(uN - u0), np.inf) / np.linalg.norm(np.abs(uN), np.inf) < 10 ** (-6):
            print(r"{d} step convergence in {p} iteration".format(d = iter__ + 1, p = iter_))
            break
        u0 = uN

fig, yy  = plt.subplots(1, 1, figsize=(12, 8))
yy.plot(x, u0)
print("Displacement at the end of rod (x = a) where a = 0.1 is ", u0[-1][0])
plt.show()