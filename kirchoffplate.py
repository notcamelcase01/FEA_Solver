import numpy as np
from parameters import  G, Eb, mu


def get_z_matrix(z):
    return  np.array([[1, 0, 0, 0, -z, 0, 0],
                  [0, 0, 0, 1, 0, -z, 0],
                  [0, 1, 1, 0, 0, 0, -2*z]])


def get_b_matrix(Lx, Ly, Nxx, Nyy, Nxy, N1xx, N1yy, N1xy, N2xx, N2yy, N2xy, N3xx, N3yy, N3xy):
    """
    Assuming is linear element for now, and stupid classical plate shenanigans
    :param N3xy: this is despair
    :param N3yy: this is despair
    :param N3xx: this is despair
    :param N2xy: this is despair
    :param N2yy: this is despair
    :param N2xx: this is despair
    :param N1xy: this is despair
    :param N1yy: this is despair
    :param N1xx: this is despair
    :param Lx: Lx
    :param Ly: Ly
    :param Nxx: Nxx
    :param Nyy: Nyy
    :param Nxy: Nxy
    :return: b matrix
    """
    # TODO: Make it general for n-noded element
    return np.array(
        [[Lx[0][0], Lx[1][0], Lx[2][0], Lx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [Ly[0][0], Ly[1][0], Ly[2][0], Ly[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, Lx[0][0], Lx[1][0], Lx[2][0], Lx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, Ly[0][0], Ly[1][0], Ly[2][0], Ly[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, Nxx[0][0], Nxx[1][0], Nxx[2][0], Nxx[3][0], N1xx[0][0], N1xx[1][0], N1xx[2][0], N1xx[3][0], N2xx[0][0], N2xx[1][0], N2xx[2][0], N2xx[3][0], N3xx[0][0], N3xx[1][0], N3xx[2][0], N3xx[3][0]],
         [0, 0, 0, 0, 0, 0, 0, 0, Nyy[0][0], Nyy[1][0], Nyy[2][0], Nyy[3][0], N1yy[0][0], N1yy[1][0], N1yy[2][0], N1yy[3][0], N2yy[0][0], N2yy[1][0], N2yy[2][0], N2yy[3][0], N3yy[0][0], N3yy[1][0], N3yy[2][0], N3yy[3][0]],
         [0, 0, 0, 0, 0, 0, 0, 0, Nxy[0][0], Nxy[1][0], Nxy[2][0], Nxy[3][0], N1xy[0][0], N1xy[1][0], N1xy[2][0], N1xy[3][0], N2xy[0][0], N2xy[1][0], N2xy[2][0], N2xy[3][0], N3xy[0][0], N3xy[1][0], N3xy[2][0], N3xy[3][0]]])


def get_n_matrix(L, N, N1, N2, N3):
    """
    Assuming q4 element
    :param L: L
    :param N: N
    :param N1: N1
    :param N2: N2
    :param N3: N3
    :return: n matrix
    """
    # TODO: Make it general for n-noded element
    return np.array([[L[0][0], L[1][0], L[2][0], L[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, L[0][0], L[1][0], L[2][0], L[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], N1[0][0], N1[1][0],
                      N1[2][0], N1[3][0], N2[0][0], N2[1][0], N2[2][0], N2[3][0], N3[0][0], N3[1][0], N3[2][0], N3[3][0]]])

def get_elasticity():
    """
    :return: C
    """
    return np.array([[Eb, mu * Eb, 0],
                     [mu * Eb, Eb, 0],
                     [0, 0, G]])



def get_lagrange_shape_function(x_gp, y_gp, Jx=1, Jy=1, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
    """
    :param Jx: d(x_gp)/dx
    :param Jy: d(y_gp)/dy
    :param seq: order in which nodes are picked, currently in "N" shape starting from bottom left
    :param x_gp: x coord
    :param y_gp: y coord
    :param element_type: element type (default linear)
    :return: lagrange fn for Q4 element
    """
    Lmat = np.zeros(len(seq))
    Lmatx = np.zeros(len(seq))
    Lmaty = np.zeros(len(seq))
    for i in range(len(seq)):
        Lmat[i] = 0.25 * (1 + seq[i][0] * x_gp) * (1 + seq[i][1] * y_gp)
        Lmatx[i] = 0.25 / Jx * (seq[i][0] * (1 + seq[i][1] * y_gp))
        Lmaty[i] = 0.25 / Jy * (seq[i][1] * (1 + seq[i][0] * x_gp))
    return Lmat[:, None], Lmatx[:, None], Lmaty[:, None]


def get_hermite_shape_function(x_gp, y_gp, Jx=1, Jy=1,  element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
    """
    :param Jy: Jy
    :param Jx: Jx
    :param seq: order in which nodes are picked, currently in "N" shape starting from bottom left
    :param x_gp: x coord
    :param y_gp: y coord
    :param element_type: element type (default linear)
    :return: lagrange fn for Q4 element
    """
    Nmat = np.zeros(len(seq))
    Nmat_1 = np.zeros(len(seq))
    Nmat_2 = np.zeros(len(seq))
    Nmat_3 = np.zeros(len(seq))
    for i in range(len(seq)):
        Nmat[i] = 1 / 16 * (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2)
        Nmat_1[i] = 1 / 16  * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1) * (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2)
        Nmat_2[i] = 1 / 16  * seq[i][1] * (y_gp + seq[i][1]) ** 2 * (y_gp * seq[i][1] - 1) * (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2)
        Nmat_3[i] = 1 / 16 * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1) * seq[i][1] * (y_gp + seq[i][1]) ** 2 * (y_gp * seq[i][1] - 1)
    return Nmat[:, None], Nmat_1[:, None], Nmat_2[:, None], Nmat_3[:, None]


def get_hermite_shape_function_derivative_xx(x_gp, y_gp, Jx, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
    """
    :param Jx: d(x_gp)/dx
    :param seq: order in which nodes are picked, currently in "N" shape starting from bottom left
    :param x_gp: x coord
    :param y_gp: y coord
    :param element_type: element type (default linear)
    :return: lagrange fn for Q4 element
    """
    Nmatxx = np.zeros(len(seq))
    Nmatxx_1 = np.zeros(len(seq))
    Nmatxx_2 = np.zeros(len(seq))
    Nmatxx_3 = np.zeros(len(seq))
    for i in range(len(seq)):
        Nmatxx[i] =   1 / (8 * Jx**2) * (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2) * ((x_gp * seq[i][0] - 2) + 2 * (x_gp + seq[i][0]) * seq[i][0])
        Nmatxx_1[i] = -1 / (8 * Jx**2) *  (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2) * (seq[i][0] * (x_gp * seq[i][0] - 1) + 2 * (x_gp + seq[i][0]) * seq[i][0] * seq[i][0])
        Nmatxx_2[i] = -1 / (8 * Jx**2) *  (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * (seq[i][1] * (y_gp * seq[i][1] - 1) + 2 * (y_gp + seq[i][1]) * seq[i][1] * seq[i][1])
        Nmatxx_3[i] = 1 / (8 * Jx**2) * (seq[i][0] * (x_gp * seq[i][0] - 1) + 2 * (x_gp + seq[i][0]) * seq[i][0] * seq[i][0]) * seq[i][1] * (y_gp + seq[i][1]) ** 2 * (y_gp * seq[i][1] - 1)
    return Nmatxx[:, None], Nmatxx_1[:, None], Nmatxx_2[:, None], Nmatxx_3[:, None]


def get_hermite_shape_function_derivative_yy(x_gp, y_gp, Jy, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
    """
    :param Jy: d(y_gp)/dy
    :param seq: order in which nodes are picked, currently in "N" shape starting from bottom left
    :param x_gp: x coord
    :param y_gp: y coord
    :param element_type: element type (default linear)
    :return: lagrange fn for Q4 element
    """
    Nmatyy = np.zeros(len(seq))
    Nmatyy_1 = np.zeros(len(seq))
    Nmatyy_2 = np.zeros(len(seq))
    Nmatyy_3 = np.zeros(len(seq))
    for i in range(len(seq)):
        Nmatyy[i] =  1 / (8 * Jy**2) * (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * ((y_gp * seq[i][1] - 2) + 2 * (y_gp + seq[i][1]) * seq[i][1])
        Nmatyy_1[i] = -1 / (8 * Jy**2) * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1) * ((y_gp * seq[i][1] - 2) + 2 * (y_gp + seq[i][1]) * seq[i][1])
        Nmatyy_2[i] = -1 / (8 * Jy**2) *  (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * (seq[i][1] * (y_gp * seq[i][1] - 1) + 2 * (y_gp + seq[i][1]) * seq[i][1] * seq[i][1])
        Nmatyy_3[i] = 1 / (8 * Jy**2) * (seq[i][1] * (y_gp * seq[i][1] - 1) + 2 * (y_gp + seq[i][1]) * seq[i][1] * seq[i][1]) * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1)
    return Nmatyy[:, None], Nmatyy_1[:, None], Nmatyy_2[:, None], Nmatyy_3[:, None]


def get_hermite_shape_function_derivative_xy(x_gp, y_gp, Jx, Jy, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
    """
    :param Jy: d(y_gp)/dy
    :param Jx: d(x_gp)/dx
    :param seq: order in which nodes are picked, currently in "N" shape starting from bottom left
    :param x_gp: x coord
    :param y_gp: y coord
    :param element_type: element type (default linear)
    :return: lagrange fn for Q4 element
    """
    Nmatxy = np.zeros(len(seq))
    Nmatxy_1 = np.zeros(len(seq))
    Nmatxy_2 = np.zeros(len(seq))
    Nmatxy_3 = np.zeros(len(seq))
    for i in range(len(seq)):
        Nmatxy[i] = 1 / (16 * Jx * Jy) * (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 2) + (x_gp + seq[i][0]) ** 2 * seq[i][0]) * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 2) + (y_gp + seq[i][1]) ** 2 * seq[i][1])
        Nmatxy_1[i] = -1 / (16 * Jx * Jy) * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 2) + (y_gp + seq[i][1]) ** 2 * seq[i][1]) * seq[i][0] * (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 1) + (x_gp + seq[i][0]) ** 2 * seq[i][0])
        Nmatxy_2[i] = -1 / (16 * Jx * Jy) * seq[i][1] * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 1) + (y_gp + seq[i][1]) ** 2 * seq[i][1]) *  (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 2) + (x_gp + seq[i][0]) ** 2 * seq[i][0])
        Nmatxy_3[i] = 1 / (16 * Jx * Jy) * seq[i][0] * seq[i][1] * (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 1) + (x_gp + seq[i][0]) ** 2 * seq[i][0]) * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 1) + (y_gp + seq[i][1]) ** 2 * seq[i][1])
    return Nmatxy[:, None], Nmatxy_1[:, None], Nmatxy_2[:, None], Nmatxy_3[:, None]


def get_assembly_vector(DOF, n):
    """
    :param DOF: dof
    :param n: nodes
    :return: assembly points
    """
    iv = [0] * 25
    for i in range(len(n)):
        situ =  [0 + i, 4 + i, 8 + i, 12 + i, 16 + i, 20 + i]
        for j in range(len(situ)):
            iv[situ[j]] = n[i] * DOF + j
    return iv