import numpy as np
import solver_non_linear as sol

E = 30 * 10 ** 6
b = 0.01
h = 0.001
a = 0.1
Area = b * h
element_type = 2
DOF = 1
DIMENSION = 1
numberOfElements = 30
numberOfNodes = numberOfElements + 1
nodePeElement = element_type ** DIMENSION
wgp, egp = sol.init_gauss_points(2)
x = sol.get_node_points_coords(numberOfNodes, a)
icon = sol.get_connectivity_matrix(numberOfElements, element_type)
f_app = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 0.1
for iter_ in range(10):
    KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
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
        kloc, floc = sol.init_stiffness_force(nodePeElement, DOF)
        for igp in range(len(wgp)):
           N, B = sol.get_lagrange_interpolation_fn_1d(egp[igp], element_type)
           xx = N.T @ xloc
           B = B / Jacobian
           kloc += E * Area * (B @ B.T) * wgp[igp] * Jacobian
           floc += (N * np.sin(np.pi * xx / a)) * Jacobian * wgp[igp]
        fg[iv[:, None], 0] += floc
        KG[iv[:, None], iv] += kloc
    KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
    u0 = np.linalg.solve(KG, fg)
    print(u0[-1])
