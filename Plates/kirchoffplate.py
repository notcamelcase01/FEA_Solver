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
                     [0, 0, 0, 0, 0, 0, 0, 0, N[0][0], N[1][0], N[2][0], N[3][0], N1[0][0], N1[1][0], N1[2][0], N1[3][0], N2[0][0], N2[1][0], N2[2][0], N2[3][0], N3[0][0], N3[1][0], N3[2][0], N3[3][0]]])

def get_elasticity():
    """
    :return: C
    """
    return np.array([[Eb, mu * Eb, 0],
                     [mu * Eb, Eb, 0],
                     [0, 0, G]])



def get_lagrange_shape_function(x_gp, y_gp, jx, jy, element_type=2, seq=((-1, -1), (1, -1), (1, 1), (-1, 1))):
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
        Lmatx[i] = 0.25 / jx * (seq[i][0] * (1 + seq[i][1] * y_gp))
        Lmaty[i] = 0.25 / jy * (seq[i][1] * (1 + seq[i][0] * x_gp))
    return Lmat[:, None], Lmatx[:, None], Lmaty[:, None]


def get_hermite_shape_function(x_gp, y_gp, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
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
        Nmat_1[i] = -1 / 16  * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1) * (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2)
        Nmat_2[i] = -1 / 16  * seq[i][1] * (y_gp + seq[i][1]) ** 2 * (y_gp * seq[i][1] - 1) * (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2)
        Nmat_3[i] = 1 / 16 * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1) * seq[i][1] * (y_gp + seq[i][1]) ** 2 * (y_gp * seq[i][1] - 1)
    return Nmat[:, None], Nmat_1[:, None], Nmat_2[:, None], Nmat_3[:, None]


def get_hermite_shape_function_derivative_xx(x_gp, y_gp, J, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
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
        Nmatxx[i] =   1 / 8 /J * (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2) * ((x_gp * seq[i][0] - 2) + 2 * (x_gp + seq[i][0]) * seq[i][0])
        Nmatxx_1[i] = -1 / 8/ J *  (y_gp + seq[i][1]) ** 2 * (seq[i][1] * y_gp - 2) * (seq[i][0] * (x_gp * seq[i][0] - 1) + 2 * (x_gp + seq[i][0]) * seq[i][0] * seq[i][0])
        Nmatxx_2[i] = -1 / 8 /J  *  (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * (seq[i][1] * (y_gp * seq[i][1] - 1) + 2 * (y_gp + seq[i][1]) * seq[i][1] * seq[i][1])
        Nmatxx_3[i] = 1 / 8 / J * (seq[i][0] * (x_gp * seq[i][0] - 1) + 2 * (x_gp + seq[i][0]) * seq[i][0] * seq[i][0]) * seq[i][1] * (y_gp + seq[i][1]) ** 2 * (y_gp * seq[i][1] - 1)
    return Nmatxx[:, None], Nmatxx_1[:, None], Nmatxx_2[:, None], Nmatxx_3[:, None]


def get_hermite_shape_function_derivative_yy(x_gp, y_gp,J ,element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
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
        Nmatyy[i] =  1 / 8 /J * (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * ((y_gp * seq[i][1] - 2) + 2 * (y_gp + seq[i][1]) * seq[i][1])
        Nmatyy_1[i] = -1 / 8 /J* seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1) * ((y_gp * seq[i][1] - 2) + 2 * (y_gp + seq[i][1]) * seq[i][1])
        Nmatyy_2[i] = -1 / 8 /J *  (x_gp + seq[i][0]) ** 2 * (seq[i][0] * x_gp - 2) * (seq[i][1] * (y_gp * seq[i][1] - 1) + 2 * (y_gp + seq[i][1]) * seq[i][1] * seq[i][1])
        Nmatyy_3[i] = 1 / 8 /J * (seq[i][1] * (y_gp * seq[i][1] - 1) + 2 * (y_gp + seq[i][1]) * seq[i][1] * seq[i][1]) * seq[i][0] * (x_gp + seq[i][0]) ** 2 * (x_gp * seq[i][0] - 1)
    return Nmatyy[:, None], Nmatyy_1[:, None], Nmatyy_2[:, None], Nmatyy_3[:, None]


def get_hermite_shape_function_derivative_xy(x_gp, y_gp, J, element_type=2, seq=((-1, -1), (-1, 1), (1, -1), (1, 1))):
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
        Nmatxy[i] = 1 / 16 / J * (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 2) + (x_gp + seq[i][0]) ** 2 * seq[i][0]) * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 2) + (y_gp + seq[i][1]) ** 2 * seq[i][1])
        Nmatxy_1[i] = -1 / 16/ J * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 2) + (y_gp + seq[i][1]) ** 2 * seq[i][1]) * seq[i][0] * (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 1) + (x_gp + seq[i][0]) ** 2 * seq[i][0])
        Nmatxy_2[i] = -1 / 16/ J * seq[i][1] * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 1) + (y_gp + seq[i][1]) ** 2 * seq[i][1]) *  (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 2) + (x_gp + seq[i][0]) ** 2 * seq[i][0])
        Nmatxy_3[i] = 1 / 16/ J * seq[i][0] * seq[i][1] * (2 * (x_gp + seq[i][0]) * (x_gp * seq[i][0] - 1) + (x_gp + seq[i][0]) ** 2 * seq[i][0]) * (2 * (y_gp + seq[i][1]) * (y_gp * seq[i][1] - 1) + (y_gp + seq[i][1]) ** 2 * seq[i][1])
    return Nmatxy[:, None], Nmatxy_1[:, None], Nmatxy_2[:, None], Nmatxy_3[:, None]


def get_assembly_vector(DOF, n):
    """
    :param DOF: dof
    :param n: nodes
    :return: assembly points
    """
    iv = [0] * 24
    for i in range(len(n)):
        situ =  [0 + i, 4 + i, 8 + i, 12 + i, 16 + i, 20 + i]
        for j in range(len(situ)):
            iv[situ[j]] = n[i] * DOF + j
    return iv


def get_2d_connectivity(nx, ny, lx, ly):
    """
    :param lx: total width
    :param ly: total height
    :param nx: no. of elements along width
    :param ny: no. of elements along height
    :return: icon, plot handle of mesh
    """
    nelm = nx * ny
    nnod = (nx + 1) * (ny + 1)
    height = ly
    width = lx
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    x_1, y_1 = np.meshgrid(x, y)
    x_1 = x_1.reshape(1, nnod)[0]
    y_1 = y_1.reshape(1, nnod)[0]
    node_array = np.array([np.arange(0, nnod, 1, dtype=np.int32), x_1, y_1])
    icon = np.zeros((5, nelm), dtype=np.int32)
    icon[0, :] = np.arange(0, nelm, 1)
    icon[1, :] = np.where((node_array[1] != width) & (node_array[2] != height))[0]
    icon[2, :] = icon[1, :] + 1
    icon[3, :] = icon[2, :] + nx + 1
    icon[4, :] = icon[3, :] - 1
    icon = icon.transpose()
    return icon, node_array, np.meshgrid(x, y)

def get_BorN_F(x, y, dx, dy, jx, jy, needN = False, justN = False):
    L, Lx, Ly = get_lagrange_shape_function(x, y, jx, jy)
    Nx = np.array([0.25 * (2 - 3 * x + x ** 3),
                   -(dx/8) * (1 - x - x**2 + x**3),
                   0.25 * (2 + 3 * x - x**3),
                   -(dx/8) * (-1-x+x**2+x**3)])[:, None]
    Ny = np.array([0.25 * (2 - 3 * y + y ** 3),
                   -(dy / 8) * (1 - y - y ** 2 + y ** 3),
                   0.25 * (2 + 3 * y - y ** 3),
                   -(dy / 8) * (-1 - y + y ** 2 + y ** 3)])[:, None]
    N1 = (Nx[0:2] @ Ny[0:2].T).T.reshape(4, 1)
    N2 = (Nx[2:4] @ Ny[0:2].T).T.reshape(4, 1)
    N3 = (Nx[2:4] @ Ny[2:4].T).T.reshape(4, 1)
    N4 = (Nx[0:2] @ Ny[2:4].T).T.reshape(4, 1)
    if justN:
        return N1, N2, N3, N4
    if needN:
        return np.array([[L[0][0], L[1][0], L[2][0], L[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, L[0][0], L[1][0], L[2][0], L[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, N1[0][0], N1[1][0], N1[2][0], N1[3][0], N2[0][0], N2[1][0], N2[2][0], N2[3][0], N3[0][0], N3[1][0], N3[2][0], N3[3][0], N4[0][0], N4[1][0], N4[2][0], N4[3][0]]])

    dNx = np.array([0.25 * (-3 + 3 * x ** 2),
                  -(dx / 8) * (-1 - 2 * x + 3 * x ** 2),
                    0.25 * (3 - 3 * x ** 2),
                    -(dx / 8) * (-1 + 2 * x + 3 * x ** 2)])[:, None]
    dNy = np.array([0.25 * (-3 + 3 * y ** 2),
                  -(dy / 8) * (-1 - 2 * y + 3 * y ** 2),
                    0.25 * (3 - 3 * y ** 2),
                    -(dy / 8) * (-1 + 2 * y + 3 * y ** 2)])[:, None]
    Nx1 = (dNx[0:2] @ Ny[0:2].T).T.reshape(4, 1) * (1 / jx)
    Nx2 = (dNx[2:4] @ Ny[0:2].T).T.reshape(4, 1) * (1 / jx)
    Nx3 = (dNx[2:4] @ Ny[2:4].T).T.reshape(4, 1) * (1 / jx)
    Nx4 = (dNx[0:2] @ Ny[2:4].T).T.reshape(4, 1) * (1 / jx)
    Ny1 = (Nx[0:2] @ dNy[0:2].T).T.reshape(4, 1) * (1 / jy)
    Ny2 = (Nx[2:4] @ dNy[0:2].T).T.reshape(4, 1) * (1 / jy)
    Ny3 = (Nx[2:4] @ dNy[2:4].T).T.reshape(4, 1) * (1 / jy)
    Ny4 = (Nx[0:2] @ dNy[2:4].T).T.reshape(4, 1) * (1 / jy)
    dNxx = np.array([0.25 * (6 * x),
                    -(dx / 8) * (-2 + 6 * x),
                     0.25 * (-6 * x),
                    -(dx / 8) * (2 + 6 * x)])[:, None]
    dNyy = np.array([0.25 * (6 * y),
                     -(dy / 8) * (-2 + 6 * y),
                     0.25 * (-6 * y),
                     -(dy / 8) * (2 + 6 * y)])[:, None]

    Nxx = (dNxx[0:2] @ Ny[0:2].T).T.reshape(4, 1) * (1 / jx**2)
    N1xx = (dNxx[2:4] @ Ny[0:2].T).T.reshape(4, 1) * (1 / jx**2)
    N2xx = (dNxx[2:4] @ Ny[2:4].T).T.reshape(4, 1) * (1 / jx**2)
    N3xx = (dNxx[0:2] @ Ny[2:4].T).T.reshape(4, 1) * (1 / jx**2)
    Nyy = (Nx[0:2] @ dNyy[0:2].T).T.reshape(4, 1) * (1 / jy**2)
    N1yy = (Nx[2:4] @ dNyy[0:2].T).T.reshape(4, 1) * (1 / jy**2)
    N2yy = (Nx[2:4] @ dNyy[2:4].T).T.reshape(4, 1) * (1 / jy**2)
    N3yy = (Nx[0:2] @ dNyy[2:4].T).T.reshape(4, 1) * (1 / jy**2)
    Nxy  = (dNx[0:2] @ dNy[0:2].T).T.reshape(4, 1) * (1 / jy * jx)
    N1xy = (dNx[2:4] @ dNy[0:2].T).T.reshape(4, 1) * (1 / jy * jx)
    N2xy = (dNx[2:4] @ dNy[2:4].T).T.reshape(4, 1) * (1 / jy * jx)
    N3xy = (dNx[0:2] @ dNy[2:4].T).T.reshape(4, 1) * (1 / jy * jx)
    return np.array(
        [[Lx[0][0], Lx[1][0], Lx[2][0], Lx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [Ly[0][0], Ly[1][0], Ly[2][0], Ly[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, Lx[0][0], Lx[1][0], Lx[2][0], Lx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, Ly[0][0], Ly[1][0], Ly[2][0], Ly[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, Nxx[0][0], Nxx[1][0], Nxx[2][0], Nxx[3][0], N1xx[0][0], N1xx[1][0], N1xx[2][0], N1xx[3][0], N2xx[0][0], N2xx[1][0], N2xx[2][0], N2xx[3][0], N3xx[0][0], N3xx[1][0], N3xx[2][0], N3xx[3][0]],
         [0, 0, 0, 0, 0, 0, 0, 0, Nyy[0][0], Nyy[1][0], Nyy[2][0], Nyy[3][0], N1yy[0][0], N1yy[1][0], N1yy[2][0], N1yy[3][0], N2yy[0][0], N2yy[1][0], N2yy[2][0], N2yy[3][0], N3yy[0][0], N3yy[1][0], N3yy[2][0], N3yy[3][0]],
         [0, 0, 0, 0, 0, 0, 0, 0, Nxy[0][0], Nxy[1][0], Nxy[2][0], Nxy[3][0], N1xy[0][0], N1xy[1][0], N1xy[2][0], N1xy[3][0], N2xy[0][0], N2xy[1][0], N2xy[2][0], N2xy[3][0], N3xy[0][0], N3xy[1][0], N3xy[2][0], N3xy[3][0]]])


if __name__ == "__main__":
    c, n, (i,j) = get_2d_connectivity(2, 1, 1, 0.5)
    print(c)
    print(n)