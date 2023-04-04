import numpy as np
import solver_non_linear as sol

element_type = 2
DOF = 1
L = 1
numberOfElements = 2
numberOfNodes = numberOfElements + 1
wgp, egp = sol.init_gauss_points(3)
x = sol.get_node_points_coords(numberOfNodes, L)
icon = sol.get_connectivity_matrix(numberOfElements, element_type)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
T, r = sol.init_stiffness_force(numberOfNodes, DOF)
u0 = np.ones(numberOfNodes * DOF)[:, None]
while True:
    for elm in range(numberOfElements):
        n = sol.get_node_from_element(icon, elm, element_type)
        xloc = []
        iv = sol.get_assembly_vector(DOF, n)
        ulo = sol.get_nodal_displacement(u0, iv)
        for i in range(len(n)):
            xloc.append(x[n[i]])
        le = xloc[-1] - xloc[0]
        Jacobian = le / 2
        kloc, floc = sol.init_stiffness_force(element_type, DOF)
        tloc, rloc = sol.init_stiffness_force(element_type, DOF)
        for igp in range(len(wgp)):
            N, B = sol.get_lagrange_interpolation_fn_1d(egp[igp], Jacobian, element_type)
            ugp = N.T @ ulo
            dugp = B.T @ ulo
            kloc += B @ B.T * (ugp + np.sqrt(2)) * wgp[igp] * Jacobian
            tloc += (B @ B.T * (ugp + np.sqrt(2)) + B @ N.T + dugp) * wgp[igp] * Jacobian
            floc += N * wgp[igp] * Jacobian
        iv = np.array(iv)
        rloc = -kloc @ ulo  + floc
        fg[iv[:, None], 0] += floc
        KG[iv[:, None], iv] += kloc
        T[iv[:, None], iv] += tloc
        r[iv[:, None], 0] += rloc
    KG, fg = sol.impose_boundary_condition(KG, fg, 2, 0)
    u = np.linalg.solve(KG, fg)
    T, r = sol.impose_boundary_condition(T, r, 2, 0)
    deltau = np.linalg.solve(T, r)
    uN = u + deltau
    if np.linalg.norm(np.abs(uN - u0), np.inf)/np.linalg.norm(np.abs(uN), np.inf)  < 10**(-6):
        break
    u0 = uN
print(u0)