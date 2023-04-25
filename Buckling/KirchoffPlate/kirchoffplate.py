import numpy as np


def get_z_matrix(z):
    return  np.array([[1, 0, 0, 0, -z, 0, 0],
                  [0, 0, 0, 1, 0, -z, 0],
                  [0, 1, 1, 0, 0, 0, -2*z]])


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


def get_node_from_cord(icon, position, nodalArray, nelm, nodePerElement):
    for elm in range(nelm):
        n = icon[elm][1:]
        xloc = []
        yloc = []
        for i in range(nodePerElement):
            xloc.append(nodalArray[1][n[i]])
            yloc.append(nodalArray[2][n[i]])
        if xloc[0] <= position[0] <= xloc[2] and yloc[0] <= position[1] <= yloc[2]:
            eta = -1 + 2 * (position[1] - yloc[0]) / (yloc[2] - yloc[0])
            zeta = -1 + 2 * (position[0] - xloc[0]) / (xloc[2] - xloc[0])
            if np.isnan(np.sum(xloc)) or np.isnan(np.sum(yloc)):
                return None, None, None
            return n, zeta, eta
    return None, None, None


def get_hermite_shape_fn_re(x, y, J, justN = False, deriv = False):
    xi = np.array((-1, 1, 1, -1))
    yi = np.array((-1, -1, 1, 1))
    H1x = -0.25 * (x ** 3 * xi - 3 * x * xi - 2)
    H2x = 0.25 * (x ** 3 + xi * x ** 2 - xi - x)
    H1x_ = -0.75* xi * (x ** 2 - 1)
    H2x_ = 0.25 * (3 * x ** 2 + 2 * xi * x - 1)
    H1x__ =  -1.5 * x * xi
    H2x__ = 0.25 * (6 * x + 2 * xi)
    H1y = -0.25 * (y ** 3 * yi - 3 * y * yi - 2)
    H2y = 0.25 * (y ** 3 + yi * y ** 2 - yi - y)
    H1y_ = -0.75 * yi * (y ** 2 - 1)
    H2y_ = 0.25 * (3 * y ** 2 + 2 * yi * y - 1)
    H1y__ = -1.5 * y * yi
    H2y__ = 0.25 * (6 * y + 2 * yi)
    H2x, H2y =  H2x * J[0, 0] + H2y * J[0, 1], H1x * J[1, 0] + H2y * J[1, 1]
    H2x_, H2y_ = H2x_ * J[0, 0] + H2y_ * J[0, 1], H1x_ * J[1, 0] + H2y_ * J[1, 1]
    H2x__, H2y__ =  H2x__ * J[0, 0] + H2y__ * J[0, 1], H1x__ * J[1, 0] + H2y__ * J[1, 1]
    N = H1x * H1y
    Nb = H2x * H1y
    Nbb = H1x * H2y
    Nbbb = H2x * H2y
    if justN:
        return N[:, None], Nb[:, None], Nbb[:, None], Nbbb[:, None]
    if deriv:
        Nx = H1x_ * H1y
        Nbx = H2x_ * H1y
        Nbbx = H1x_ * H2y
        Nbbbx = H2x_ * H2y
        Ny = H1y_ * H1x
        Nby = H1y_ * H2x
        Nbby = H2y_ * H1x
        Nbbby = H2y_ * H2x
        return Nx, Nbx, Nbbx, Nbbbx, Ny, Nby, Nbby, Nbbby
    Nxx =  H1x__ * H1y
    Nbxx = H2x__ * H1y
    Nbbxx = H1x__ * H2y
    Nbbbxx = H2x__ * H2y
    Nyy = H1x * H1y__
    Nbyy = H2x * H1y__
    Nbbyy = H1x * H2y__
    Nbbbyy = H2x * H2y__
    Nxy = H1x_ * H1y_
    Nbxy = H2x_ * H1y_
    Nbbxy = H1x_ * H2y_
    Nbbbxy = H2x_ * H2y_
    return Nxx, Nbxx, Nbbxx, Nbbbxx, Nyy, Nbyy, Nbbyy, Nbbbyy, Nxy, Nbxy, Nbbxy, Nbbbxy


def get_lagrange_shape_function_re(x_gp, y_gp):
    """
    :param x_gp: x coord
    :param y_gp: y coord
    :return: lagrange fn for Q4 element
    """
    xi = np.array((-1, 1, 1, -1))
    yi = np.array((-1, -1, 1, 1))
    Lmat = 0.25 * (1 + xi * x_gp) * (1 + y_gp * yi)
    Lmatx = 0.25  * (xi * (1 + yi * y_gp))
    Lmaty = 0.25 * (yi * (1 + xi * x_gp))
    return Lmat[:, None], Lmatx[:, None], Lmaty[:, None]


def get_cartisian_shape_hermite(xloc, yloc, J, Jinv, Nxx, Nyy, Nxy, Nx, Ny):
    H = np.array([[J[0, 0] ** 2, J[0, 1] ** 2, 2 * J[0, 0] * J[0, 1]],
                  [J[1, 0] ** 2, J[1, 1] ** 2, 2 * J[1, 0] * J[1, 1]],
                  [J[0, 0] * J[1, 0], J[0, 1] * J[1, 1], J[0, 0] * J[1, 1] + J[1, 0] * J[1, 0]]])
    Hinv = np.linalg.inv(H)
    Lxy = 0.25 * np.array([1, -1, 1, -1])[:, None]
    P1 = np.zeros((3, 2))
    P1[2, 0] = Lxy.T @ xloc
    P1[2, 1] = Lxy.T @ yloc
    P = Hinv @ P1 @ Jinv
    Nxx, Nyy, Nxy = Hinv[0, 0] * Nxx + Hinv[0, 1] * Nyy + Hinv[0, 2] * Nxy - (P[0, 0] * Nx + P[0, 1] * Ny),  Hinv[1, 0] * Nxx + Hinv[1, 1] * Nyy + Hinv[1, 2] * Nxy - (P[1, 0] * Nx + P[1, 1] * Ny),  Hinv[2, 0] * Nxx + Hinv[2, 1] * Nyy + Hinv[2, 2] * Nxy - (P[2, 0] * Nx + P[2, 1] * Ny)
    return Nxx, Nyy, Nxy


def get_cartisian_shape_lagrange(xloc, yloc, Nx, Ny):
    J = np.zeros((2, 2))
    J[0, 0] = Nx.T @ xloc
    J[0, 1] = Nx.T @ yloc
    J[1, 0] = Ny.T @ xloc
    J[1, 1] = Ny.T @ yloc
    Jinv = np.linalg.inv(J)
    Nx, Ny =  Nx * Jinv[0, 0] + Ny * Jinv[0, 1], Nx * Jinv[1, 0] + Ny * Jinv[1, 1]
    return Nx, Ny, J, Jinv


def get_bmat(x, y, xloc, yloc):
    L, Lx, Ly = get_lagrange_shape_function_re(x, y)
    Lx, Ly, J, Jinv = get_cartisian_shape_lagrange(xloc, yloc, Lx, Ly)
    Nxx, N1xx, N2xx, N3xx, Nyy, N1yy, N2yy, N3yy, Nxy, N1xy, N2xy, N3xy = get_hermite_shape_fn_re(x, y, J)
    Nx, N1x, N2x, N3x, Ny, N1y, N2y, N3y = get_hermite_shape_fn_re(x, y, J, deriv=True)
    Nxx, Nyy, Nxy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, Nxx, Nyy, Nxy, Nx, Ny)
    N1xx, N1yy, N1xy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, N1xx, N1yy, N1xy, N1x, N1y)
    N2xx, N2yy, N2xy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, N2xx, N2yy, N2xy, N2x, N2y)
    N3xx, N3yy, N3xy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, N3xx, N3yy, N3xy, N3x, N3y)
    B1 = np.zeros((7, 24))
    for k in range(4):
        B1[0, 6 * k] = Lx[k][0]
        B1[1, 6 * k] = Ly[k][0]
        B1[2, 6 * k + 1] = Lx[k][0]
        B1[3, 6 * k + 1] = Ly[k][0]
        B1[4, 6 * k + 2:6 * k + 6] = [Nxx[k], N1xx[k], N2xx[k], N3xx[k]]
        B1[5, 6 * k + 2:6 * k + 6] = [Nyy[k], N1yy[k], N2yy[k], N3yy[k]]
        B1[6, 6 * k + 2:6 * k + 6] = [Nxy[k], N1xy[k], N2xy[k], N3xy[k]]
    return B1, J

def get_nmat(x, y, J):
    L, Lx, Ly = get_lagrange_shape_function_re(x, y)
    N, N1, N2, N3 = get_hermite_shape_fn_re(x, y, J, justN = True)
    B1 = np.zeros((3, 24))
    for k in range(4):
        B1[0, 6 * k] = L[k][0]
        B1[1, 6 * k + 1] = L[k][0]
        B1[2, 6 * k + 2:6 * k + 6] = [N[k], N1[k], N2[k], N3[k]]
    return B1


def get_hermite_shapes(x, y, xloc, yloc):
    L, Lx, Ly = get_lagrange_shape_function_re(x, y)
    Lx, Ly, J, Jinv = get_cartisian_shape_lagrange(xloc, yloc, Lx, Ly)
    N, N1, N2, N3 = get_hermite_shape_fn_re(x, y, J, justN = True)
    Nxx, N1xx, N2xx, N3xx, Nyy, N1yy, N2yy, N3yy, Nxy, N1xy, N2xy, N3xy = get_hermite_shape_fn_re(x, y, J)
    Nx, N1x, N2x, N3x, Ny, N1y, N2y, N3y = get_hermite_shape_fn_re(x, y, J, deriv=True)
    Nxx, Nyy, Nxy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, Nxx, Nyy, Nxy, Nx, Ny)
    N1xx, N1yy, N1xy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, N1xx, N1yy, N1xy, N1x, N1y)
    N2xx, N2yy, N2xy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, N2xx, N2yy, N2xy, N2x, N2y)
    N3xx, N3yy, N3xy = get_cartisian_shape_hermite(xloc, yloc, J, Jinv, N3xx, N3yy, N3xy, N3x, N3y)
    Nx, Ny =  Nx * Jinv[0, 0] + Ny * Jinv[0, 1], Nx * Jinv[1, 0] + Ny * Jinv[1, 1]
    N1x, N1y =  N1x * Jinv[0, 0] + N1y * Jinv[0, 1], N1x * Jinv[1, 0] + N1y * Jinv[1, 1]
    N2x, N2y =  N2x * Jinv[0, 0] + N2y * Jinv[0, 1], N2x * Jinv[1, 0] + N2y * Jinv[1, 1]
    N3x, N3y =  N3x * Jinv[0, 0] + N3y * Jinv[0, 1], N3x * Jinv[1, 0] + N3y * Jinv[1, 1]
    H = np.zeros((6, 16))
    for k in range(4):
        H[0, 4 * k: 4 * k + 4] = [N[k], N1[k], N2[k], N3[k]]
        H[1, 4 * k: 4 * k + 4] = [Nx[k], N1x[k], N2x[k], N3x[k]]
        H[2, 4 * k: 4 * k + 4] = [Ny[k], N1y[k], N2y[k], N3y[k]]
        H[3, 4 * k: 4 * k + 4] = [Nxx[k], N1xx[k], N2xx[k], N3xx[k]]
        H[4, 4 * k: 4 * k + 4] = [Nyy[k], N1yy[k], N2yy[k], N3yy[k]]
        H[5, 4 * k: 4 * k + 4] = [Nxy[k], N1xy[k], N2xy[k], N3xy[k]]
    return H, J


if __name__ == "__main__":
    c, n, (i,j) = get_2d_connectivity(10, 10, 1, 1)
    print(c)
    print(n)