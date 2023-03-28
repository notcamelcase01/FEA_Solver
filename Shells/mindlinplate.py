import numpy as np
from parameters import  G, Eb, mu, k

def get_z1_matrix(z):
    return np.array([[1, 0, 0, 0, z, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, z],
                     [0, 1, 1, 0, 0, z, z, 0]])

def get_z2_matrix(z):
    return np.array([[0, 1, 0, 1],
                     [1, 0, 1, 0]])


def get_z_matrix(z):
    return np.array([[1, 0, z, 0, 0],
                     [0, 1, 0, z, 0],
                     [0, 0, 0, 0, 1]])

def get_B1_matrix(Nx, Ny):
    B1 = np.zeros((8, 5 * len(Nx)))
    for i in range(len(Nx)):
        B1[0, 5 * i] = Nx[i][0]
        B1[1, 5 * i] = Ny[i][0]
        B1[2, 5 * i + 1] = Nx[i][0]
        B1[3, 5 * i + 1] = Ny[i][0]
        B1[4, 5 * i + 3] = Nx[i][0]
        B1[5, 5 * i + 3] = Ny[i][0]
        B1[6, 5 * i + 4] = Nx[i][0]
        B1[7, 5 * i + 4] = Ny[i][0]
    return B1


def get_B2_matrix(N, Nx, Ny):
    B2 = np.zeros((4, 5 * len(N)))
    for i in range(len(N)):
        B2[0, 5 * i + 2] = Nx[i][0]
        B2[1, 5 * i + 2] = Ny[i][0]
        B2[2, 5 * i + 3] = N[i][0]
        B2[3, 5 * i + 4] = N[i][0]
    return B2

def get_N_matrix(N):
    N1 = np.zeros((5, 5 * len(N)))
    for i in range(len(N)):
        N1[0, 5 * i] = N[i][0]
        N1[1, 5 * i + 1] = N[i][0]
        N1[2, 5 * i + 2] = N[i][0]
        N1[3, 5 * i + 3] = N[i][0]
        N1[4, 5 * i + 4] = N[i][0]
    return N1

def get_C1_matrix():
    return Eb * np.array([[1, mu, 0],
                          [mu, 1, 0],
                          [0, 0, (1 - mu)/2]])

def get_C2_matrix():
    return k * np.array([[G, 0],
                     [0, G]])


# def get_lagrange_shape_function(x_gp, y_gp, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
#     Lmat = np.zeros(len(seq))
#     Lmatx = np.zeros(len(seq))
#     Lmaty = np.zeros(len(seq))
#     for i in range(len(seq)):
#         Lmat[i] = 0.25 * (1 + seq[i][0] * x_gp) * (1 + seq[i][1] * y_gp)
#         Lmatx[i] = 0.25  * (seq[i][0] * (1 + seq[i][1] * y_gp))
#         Lmaty[i] = 0.25  * (seq[i][1] * (1 + seq[i][0] * x_gp))
#     return Lmat[:, None], Lmatx[:, None], Lmaty[:, None]


def get_lagrange_shape_function(x, y, xi = (-1, 0, 1, 1, 1, 0, -1, -1, 0),
                                  yi = (-1, -1, -1, 0, 1, 1, 1, 0, 0)):
    N = np.zeros(9)
    Nx = np.zeros(9)
    Ny = np.zeros(9)
    for i in range(len(xi)):
        N[i] = ((1.5 * xi[i]**2 - 1) * x**2 + 0.5 * xi[i] * x + 1 - xi[i]**2) * ((1.5 * yi[i]**2 - 1) * y**2 + 0.5 * yi[i] * y + 1 - yi[i]**2)
        Nx[i] = ((1.5 * xi[i]**2 - 1) * x * 2 + 0.5 * xi[i]) * ((1.5 * yi[i]**2 - 1) * y**2 + 0.5 * yi[i] * y + 1 - yi[i]**2)
        Ny[i] = ((1.5 * xi[i]**2 - 1) * x**2 + 0.5 * xi[i] * x + 1 - xi[i]**2) * ((1.5 * yi[i]**2 - 1) * y * 2 + 0.5 * yi[i])
    return N[:, None], Nx[:, None], Ny[:, None]


def get_assembly_vector(DOF, n):
    """
    :param DOF: dof
    :param n: nodes
    :return: assembly points
    """
    iv = [0] * 20
    for i in range(len(n)):
        situ =  [0 + i, 4 + i, 8 + i, 12 + i, 16 + i]
        for j in range(len(situ)):
            iv[situ[j]] = n[i] * DOF + j
    return iv

def get_node_from_cord(icon, position, nodalArray, nelm, nodePerElement):
    for elm in range(nelm):
        n = icon[elm][1:]
        xloc = []
        yloc = []
        for i in range(nodePerElement):
            xloc.append(nodalArray[1][n[i]])
            yloc.append(nodalArray[2][n[i]])
        if xloc[0] <= position[0] <= xloc[4] and yloc[0] <= position[1] <= yloc[4]:
            eta = -1 + 2 * (position[1] - yloc[0]) / (yloc[4] - yloc[0])
            zeta = -1 + 2 * (position[0] - xloc[0]) / (xloc[4] - xloc[0])
            if np.isnan(np.sum(xloc)) or np.isnan(np.sum(yloc)):
                return None, None, None
            return n, zeta, eta
    return None, None, None