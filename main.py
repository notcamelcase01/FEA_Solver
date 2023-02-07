import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from parameters import E, b, f0, L, F, k, G
import keywords as param

plt.style.use('dark_background')


def get_height(xp):
    """
    Return height as it vary
    :param xp: x coord
    :return: height
    """
    return .01 * (1 - xp * 0.5 / L)


numberOfElements = 20
DOF = 3
qx = L/2
element_type = param.ElementType.LINEAR
OVERRIDE_REDUCED_INTEGRATION = False
GAUSS_POINTS_REQ = 3
REQUESTED_NODES = [0, 1]
numberOfNodes = (element_type - 1) * numberOfElements + 1
x = sol.get_node_points_coords(numberOfNodes, L, REQUESTED_NODES)
connectivityMatrix = sol.get_connectivity_matrix(numberOfElements, element_type)
weightOfGaussPts, gaussPts = sol.init_gauss_points(3)
reduced_wts, reduced_gpts = sol.init_gauss_points(1 if (not OVERRIDE_REDUCED_INTEGRATION and
                                                        element_type == param.ElementType.LINEAR) else GAUSS_POINTS_REQ)
KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
for elm in range(numberOfElements):
    n = sol.get_node_from_element(connectivityMatrix, elm, element_type)
    xloc = []
    for i in range(len(n)):
        xloc.append(x[n[i]])
    le = xloc[-1] - xloc[0]
    Jacobian = le / 2
    Kuu = np.zeros((element_type, element_type))
    Kww = np.zeros((element_type, element_type))
    Kwt = np.zeros((element_type, element_type))
    Ktt = np.zeros((element_type, element_type))
    kloc, floc = sol.init_stiffness_force(element_type, DOF)
    f1 = sol.get_body_force(.25 * L, .75 * L, xloc, f0)
    Fa, Ft, Mm = sol.get_point_force(L, xloc, F)
    floc += sol.get_timoshenko_force(Ft, Fa, Mm, element_type)
    """
    FULL INTEGRATION LOOP
    """
    for igp in range(len(weightOfGaussPts)):
        xx = 0.5 * (xloc[-1] + xloc[0]) + 0.5 * (xloc[-1] - xloc[0]) * gaussPts[igp]
        Nmat, Bmat = sol.get_lagrange_fn(gaussPts[igp], Jacobian, element_type)
        h = get_height(xx)
        Moi = b * h ** 3 / 12
        A = b * h
        Kuu += E * A * Bmat @ Bmat.T * Jacobian * weightOfGaussPts[igp]
        Ktt += E * Moi * Bmat @ Bmat.T * Jacobian * weightOfGaussPts[igp]
        ft = Nmat * f1[1] * Jacobian * weightOfGaussPts[igp]
        fa = Nmat * f1[0] * Jacobian * weightOfGaussPts[igp]
        m = Bmat * f1[2] * Jacobian * weightOfGaussPts[igp]
        floc += sol.get_timoshenko_force(ft, fa, m, element_type)
    """
    REDUCED INTEGRATION LOOP
    """
    for igp in range(len(reduced_wts)):
        xx = 0.5 * (xloc[-1] + xloc[0]) + 0.5 * (xloc[-1] - xloc[0]) * reduced_gpts[igp]
        Nmat, Bmat = sol.get_lagrange_fn(reduced_gpts[igp], Jacobian, element_type)
        h = get_height(xx)
        Moi = b * h ** 3 / 12
        A = b * h
        Kww += k * G * A * Bmat @ Bmat.T * Jacobian * reduced_wts[igp]
        Kwt += k * G * A * Bmat @ Nmat.T * Jacobian * reduced_wts[igp]
        Ktt += k * G * A * Nmat @ Nmat.T * Jacobian * reduced_wts[igp]
    kloc = sol.get_timoshenko_stiffness(Kuu, Kww, Kwt, Ktt, element_type)
    iv = sol.get_assembly_vector(DOF, n)
    fg += sol.assemble_force(floc, iv, numberOfNodes * DOF)
    KG += sol.assemble_stiffness(kloc, iv, numberOfNodes * DOF)

KG, fg = sol.impose_boundary_condition(KG, fg, 0, 0)
KG, fg = sol.impose_boundary_condition(KG, fg, 1, 0)
KG, fg = sol.impose_boundary_condition(KG, fg, 2, 0)
u = sol.get_displacement_vector(KG, fg)


'''
Post-processing
'''
ad = []
td = []
theta = []
for i in range(numberOfNodes):
    ad.append(u[3 * i][0])
    td.append(u[3 * i + 1][0])
    theta.append(u[3 * i + 2][0])
print(u[-2])
red_d = "Invalid query length"
z_across = np.linspace(-get_height(qx)/2, get_height(qx)/2, 20)
strain_xx = np.zeros(len(z_across))
for elm in range(numberOfElements):
    n = sol.get_node_from_element(connectivityMatrix, elm, element_type)
    xloc = []
    for i in range(len(n)):
        xloc.append(x[n[i]])
    le = xloc[-1] - xloc[0]
    uloc = sol.get_nodal_displacement(td, n)
    uloc_theta = sol.get_nodal_displacement(theta, n)
    uloc_axial = sol.get_nodal_displacement(ad, n)
    if xloc[0] <= qx <= xloc[-1]:
        gp = -1 + 2 * (qx - xloc[0]) / le
        Nmat, Bmat = sol.get_lagrange_fn(gp, le / 2, element_type)
        red_d = (Nmat.T @ uloc)[0][0]
        print(Bmat.T @ uloc_theta)
        for i in range(len(z_across)):
            strain_xx[i] = z_across[i] * (Bmat.T @ uloc_axial + Bmat.T @ uloc_theta)
        break
fig0, ax1 = plt.subplots(1, 1, figsize=(10, 5))
fig,  (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 5))
ax1.plot(x, td, marker="o", label="transverse displacement")
ax1.set_xlabel("Length")
ax1.set_ylabel("Transverse Displacement")
ax1.set_title("Displacement at x = {L}m : {ttt}m".format(L=qx, ttt=red_d))
ax1.legend()
ax2.plot(strain_xx, z_across, marker="o")
ax2.set_xlabel("$\\epsilon_{xx}$")
ax2.set_ylabel("Z")
ax2.set_title("Strain at top fiber : {ttt}".format(ttt=strain_xx[-1]))
ax3.plot(strain_xx * E, z_across, marker="o")
ax3.set_xlabel("$\\sigma_{xx}$")
ax3.set_ylabel("Z")
ax3.set_title("Stress at top fiber : {ttt}".format(ttt=strain_xx[-1] * E))
plt.show()