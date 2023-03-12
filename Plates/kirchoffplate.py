import numpy as np


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
    Eb = 30*10**6/(1-.09)
    G = 30*10**6/(2+2*.3)
    C = np.array([[Eb, .3 * Eb, 0],
                     [.3 * Eb, Eb, 0],
                     [0, 0, G]])
    return C



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

def get_BorN_F(x, y, jx, jy, needN = False, justN = False):
    L, Lx, Ly = get_lagrange_shape_function(x, y, jx, jy)
    Nx = np.array([0.25 * (2 - 3 * x + x ** 3),
                   -(jx / 4) * (1 - x - x**2 + x**3),
                   0.25 * (2 + 3 * x - x**3),
                   -(jx / 4) * (-1 - x + x**2 + x**3)])[:, None]
    Ny = np.array([0.25 * (2 - 3 * y + y ** 3),
                   -(jy / 4) * (1 - y - y ** 2 + y ** 3),
                   0.25 * (2 + 3 * y - y ** 3),
                   -(jy / 4) * (-1 - y + y ** 2 + y ** 3)])[:, None]
    N1 = (Nx[0:2] @ Ny[0:2].T).T.reshape(4,)
    N2 = (Nx[2:4] @ Ny[0:2].T).T.reshape(4,)
    N3 = (Nx[2:4] @ Ny[2:4].T).T.reshape(4,)
    N4 = (Nx[0:2] @ Ny[2:4].T).T.reshape(4,)
    H = [N1, N2, N3, N4]
    if justN:
        return N1[:, None], N2[:, None], N3[:, None], N4[:, None]
    if needN:
        B1 = np.zeros((3, 24))
        for k in range(4):
            B1[0, 6 * k] = L[k][0]
            B1[1, 6 * k + 1] = L[k][0]
            B1[2, 6 * k + 2:6 * k + 6] = H[k]
        return B1
        # return np.array([[L[0][0], L[1][0], L[2][0], L[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #              [0, 0, 0, 0, L[0][0], L[1][0], L[2][0], L[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #              [0, 0, 0, 0, 0, 0, 0, 0, N1[0][0], N1[1][0], N1[2][0], N1[3][0], N2[0][0], N2[1][0], N2[2][0], N2[3][0], N3[0][0], N3[1][0], N3[2][0], N3[3][0], N4[0][0], N4[1][0], N4[2][0], N4[3][0]]])

    dNx = np.array([0.25 * (-3 + 3 * x ** 2),
                  -(jx / 4) * (-1 - 2 * x + 3 * x ** 2),
                    0.25 * (3 - 3 * x ** 2),
                    -(jx / 4) * (-1 + 2 * x + 3 * x ** 2)])[:, None]
    dNy = np.array([0.25 * (-3 + 3 * y ** 2),
                  -(jy / 4) * (-1 - 2 * y + 3 * y ** 2),
                    0.25 * (3 - 3 * y ** 2),
                    -(jy / 4) * (-1 + 2 * y + 3 * y ** 2)])[:, None]
    Nx1 = (dNx[0:2] @ Ny[0:2].T).T.reshape(4, 1) * (1 / jx)
    Nx2 = (dNx[2:4] @ Ny[0:2].T).T.reshape(4, 1) * (1 / jx)
    Nx3 = (dNx[2:4] @ Ny[2:4].T).T.reshape(4, 1) * (1 / jx)
    Nx4 = (dNx[0:2] @ Ny[2:4].T).T.reshape(4, 1) * (1 / jx)
    Ny1 = (Nx[0:2] @ dNy[0:2].T).T.reshape(4, 1) * (1 / jy)
    Ny2 = (Nx[2:4] @ dNy[0:2].T).T.reshape(4, 1) * (1 / jy)
    Ny3 = (Nx[2:4] @ dNy[2:4].T).T.reshape(4, 1) * (1 / jy)
    Ny4 = (Nx[0:2] @ dNy[2:4].T).T.reshape(4, 1) * (1 / jy)
    dNxx = np.array([0.25 * (6 * x),
                    -(jx / 4) * (-2 + 6 * x),
                     0.25 * (-6 * x),
                    -(jx / 4) * (2 + 6 * x)])[:, None]
    dNyy = np.array([0.25 * (6 * y),
                     -(jy / 4) * (-2 + 6 * y),
                     0.25 * (-6 * y),
                     -(jy / 4) * (2 + 6 * y)])[:, None]

    Nxx = (dNxx[0:2] @ Ny[0:2].T).T.reshape(4, ) * (1 / jx**2)
    N1xx = (dNxx[2:4] @ Ny[0:2].T).T.reshape(4, ) * (1 / jx**2)
    N2xx = (dNxx[2:4] @ Ny[2:4].T).T.reshape(4, ) * (1 / jx**2)
    N3xx = (dNxx[0:2] @ Ny[2:4].T).T.reshape(4, ) * (1 / jx**2)
    Nyy = (Nx[0:2] @ dNyy[0:2].T).T.reshape(4, ) * (1 / jy**2)
    N1yy = (Nx[2:4] @ dNyy[0:2].T).T.reshape(4, ) * (1 / jy**2)
    N2yy = (Nx[2:4] @ dNyy[2:4].T).T.reshape(4, ) * (1 / jy**2)
    N3yy = (Nx[0:2] @ dNyy[2:4].T).T.reshape(4, ) * (1 / jy**2)
    Nxy  = (dNx[0:2] @ dNy[0:2].T).T.reshape(4, ) * (1 / jy / jx)
    N1xy = (dNx[2:4] @ dNy[0:2].T).T.reshape(4, ) * (1 / jy / jx)
    N2xy = (dNx[2:4] @ dNy[2:4].T).T.reshape(4, ) * (1 / jy / jx)
    N3xy = (dNx[0:2] @ dNy[2:4].T).T.reshape(4, ) * (1 / jy / jx)
    Hxx = [Nxx, N1xx, N2xx, N3xx]
    Hyy = [Nyy, N1yy, N2yy, N3yy]
    Hxy = [Nxy, N1xy, N2xy, N3xy]

    B1 = np.zeros((7, 24))
    for k in range(4):
        B1[0, 6 * k] = Lx[k][0]
        B1[1, 6 * k] = Ly[k][0]
        B1[2, 6 * k + 1] = Lx[k][0]
        B1[3, 6 * k + 1] = Ly[k][0]
        B1[4, 6 * k + 2:6 * k + 6] = Hxx[k]
        B1[5, 6 * k + 2:6 * k + 6] = Hyy[k]
        B1[6, 6 * k + 2:6 * k + 6] = Hxy[k]
    return B1

    # return np.array(
    #     [[Lx[0][0], Lx[1][0], Lx[2][0], Lx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [Ly[0][0], Ly[1][0], Ly[2][0], Ly[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, Lx[0][0], Lx[1][0], Lx[2][0], Lx[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, Ly[0][0], Ly[1][0], Ly[2][0], Ly[3][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, Nxx[0][0], Nxx[1][0], Nxx[2][0], Nxx[3][0], N1xx[0][0], N1xx[1][0], N1xx[2][0], N1xx[3][0], N2xx[0][0], N2xx[1][0], N2xx[2][0], N2xx[3][0], N3xx[0][0], N3xx[1][0], N3xx[2][0], N3xx[3][0]],
    #      [0, 0, 0, 0, 0, 0, 0, 0, Nyy[0][0], Nyy[1][0], Nyy[2][0], Nyy[3][0], N1yy[0][0], N1yy[1][0], N1yy[2][0], N1yy[3][0], N2yy[0][0], N2yy[1][0], N2yy[2][0], N2yy[3][0], N3yy[0][0], N3yy[1][0], N3yy[2][0], N3yy[3][0]],
    #      [0, 0, 0, 0, 0, 0, 0, 0, Nxy[0][0], Nxy[1][0], Nxy[2][0], Nxy[3][0], N1xy[0][0], N1xy[1][0], N1xy[2][0], N1xy[3][0], N2xy[0][0], N2xy[1][0], N2xy[2][0], N2xy[3][0], N3xy[0][0], N3xy[1][0], N3xy[2][0], N3xy[3][0]]])


if __name__ == "__main__":
    c, n, (i,j) = get_2d_connectivity(10, 10, 1, 1)
    print(c)
    print(n)