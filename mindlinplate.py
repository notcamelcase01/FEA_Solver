import numpy as np
from parameters import  G, Eb, mu

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
    return np.array([[Nx[0][0], Nx[1][0], Nx[2][0], Nx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [Ny[0][0], Ny[1][0], Ny[2][0], Ny[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, Nx[0][0], Nx[1][0], Nx[2][0], Nx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, Ny[0][0], Ny[1][0], Ny[2][0], Ny[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, Nx[0][0], Nx[1][0], Nx[2][0], Nx[3][0], 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, Ny[0][0], Ny[1][0], Ny[2][0], Ny[3][0], 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Nx[0][0], Nx[1][0], Nx[2][0], Nx[3][0], 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Ny[0][0], Ny[1][0], Ny[2][0], Ny[3][0], 0, 0, 0, 0]])



def get_B2_matrix(N, Nx, Ny):
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Nx[0][0], Nx[1][0], Nx[2][0], Nx[3][0]],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Ny[0][0], Ny[1][0], Ny[2][0], Ny[3][0]]])


def get_N_matrix(N):
    return np.array([[N[0][0], N[1][0], N[2][0], N[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0]]])

def get_C1_matrix():
    return Eb * np.array([[1, mu, 0],
                          [mu, 1, 0],
                          [0, 0, (1 - mu)/2]])

def get_C2_matrix():
    return np.array([[G, 0],
                     [0, G]])


def get_lagrange_shape_function(x_gp, y_gp, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
    Lmat = np.zeros(len(seq))
    Lmatx = np.zeros(len(seq))
    Lmaty = np.zeros(len(seq))
    for i in range(len(seq)):
        Lmat[i] = 0.25 * (1 + seq[i][0] * x_gp) * (1 + seq[i][1] * y_gp)
        Lmatx[i] = 0.25  * (seq[i][0] * (1 + seq[i][1] * y_gp))
        Lmaty[i] = 0.25  * (seq[i][1] * (1 + seq[i][0] * x_gp))
    return Lmat[:, None], Lmatx[:, None], Lmaty[:, None]


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