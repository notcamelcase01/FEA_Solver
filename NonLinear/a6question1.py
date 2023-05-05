import numpy as np
import solver_non_linear as sol
import matplotlib.pyplot as plt

tol = 1e-2
f0 = 100
Elasticity = 30 * 10 ** 6
b = 0.01
h = 0.001
a = 0.1
Area = b * h
element_type = 2
DOF = 1
DIMENSION = 1
numberOfElements = 10
numberOfNodes = numberOfElements + 1
nodePeElement = element_type ** DIMENSION
wgp, egp = sol.init_gauss_points(3)
x = sol.get_node_points_coords(numberOfNodes, a)
icon = sol.get_connectivity_matrix(numberOfElements, element_type)
u0 = np.zeros((numberOfNodes, 1))
f_app = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 0.1 * f0 * b
for iter__ in range(10): #Force Increment
    KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
    """
    I have done this tangent stiffness thing but this is not necessary since we can add increments directly in displacements 
    and thats what I am doing so just ignore these terms, I was trying to draw parallels between newton raphson and this
    """
    T, r = sol.init_stiffness_force(numberOfNodes, DOF)
    for _ in range(100):
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
            tloc, rloc = sol.init_stiffness_force(nodePeElement, DOF)
            for igp in range(len(wgp)):
               N, B = sol.get_lagrange_interpolation_fn_1d(egp[igp], element_type)
               xx = N.T @ xloc
               B = B / Jacobian
               du = B.T @ uloc
               Bnl =  B  *  0.5
               Bl = (B + du * B)
               E =  du + du ** 2 * 0.5
               kloc += Elasticity * Area * (Bl @ Bl.T) * wgp[igp] * Jacobian
               tloc += Elasticity * Area *  (E * Bnl @ Bnl.T + Bl @ Bl.T) * Jacobian * wgp[igp]
               rloc += (N *  f_app[iter__] * np.sin(np.pi * xx / a) - Bl * E * Elasticity * Area) * Jacobian * wgp[igp]
               floc += (N *  f_app[iter__] * np.sin(np.pi * xx / a)) * Jacobian * wgp[igp]
            fg[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc
            r[iv[:, None], 0] += rloc
            T[iv[:, None], iv] += tloc
        """
        I have done this tangent stiffness thing but this is not necessary since we can add increments directly in displacements 
        and thats what I am doing so just ignore these terms, I was trying to draw parallels between newton raphson and this
        """
        KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
        T, r = sol.impose_boundary_condition(T, r, 0, 0)
        u  =  np.linalg.solve(KG, fg)
        delta_u = np.linalg.solve(T, r)
        u0 = u0 + delta_u
        if (abs(np.linalg.norm(delta_u) / (np.linalg.norm(u0) + tol))) < tol:
            break

print("Displacement at the end of rod (x = a) where a = 0.1 is ", u0[-1][0])
