import numpy as np



def get_node_from_element(icon, elm, elm_type):
    """
    :param icon: connectivity matrix
    :param elm: element number
    :param elm_type: element type 2 for 2 noded linear
    :return: nodes associated with that element
    """
    x = []
    for i in range(elm_type):
        x.append(icon[elm][i])
    return x


def get_connectivity_matrix(nelm, element_type):
    """
    :param nelm: number of elements
    :param element_type: 2 for linear
    :return: connectivity matrix
    """
    # TODO : Remove if statement and use loop
    icon = np.zeros((nelm, element_type), dtype=np.int32)
    if element_type == 2:
        for i in range(nelm):
            icon[i, 0] = i
            icon[i, 1] = 1 + i
    elif element_type == 3:
        for i in range(nelm):
            icon[i, 0] = 2 * i
            icon[i, 1] = 1 + 2 * i
            icon[i, 2] = 2 + 2 * i
    else:
        raise Exception("Uhm, This is wendy's, we don't, cubic element that here")
    return icon


def get_nodal_displacement(u, iv):
    """
    :param u: displacement vector
    :param iv: assembly vector
    :return: nodal displacement
    """
    uloc = np.zeros((len(iv), 1))
    for i in range(len(iv)):
        uloc[i] = u[iv[i]]
    return uloc

def get_node_points_coords(n, length, reqn=(0, 1)):
    """
    :param reqn: Requested lengths you want Nodes
    :param n: number of nodes
    :param length: length
    :return: node coordinates
    """
    x = np.array([])
    number_of_div = len(reqn) - 1
    rem = n % number_of_div
    exact_div = []
    for _ in range(number_of_div):
        exact_div.append(n // number_of_div)
    exact_div[-1] += rem - 1
    for i in range(number_of_div):
        x = np.hstack((x, np.linspace(length * reqn[i], length * reqn[i + 1], exact_div[i], endpoint=False)))
    x = np.append(x, reqn[-1] * length)
    return x


def get_lagrange_interpolation_fn_1d(x, element_type):
    """
    :param x: gp
    :param J: jaobian
    :param element_type: element type
    :return:
    """
    xi_ = np.array([-1, 1])[:, None]
    if element_type == 2:
        N = (1 + xi_ * x) / 2
        Nx = (xi_ / 2)
    else:
        raise Exception("Sir this is wendy's")
    return N, Nx

def init_gauss_points(n=3):
    """
    :param n: number of gauss points
    :return: (weights of gp,Gauss points)
    """
    if n == 1:
        wgp = np.array([2])
        egp = np.array([0])
    elif n == 2:
        wgp = np.array([1, 1])
        egp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif n == 3:
        wgp = np.array([5 / 9, 8 / 9, 5 / 9])
        egp = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    else:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 gauss points here")
    return wgp, egp


def init_stiffness_force(nnod, DOF):
    """
    :param nnod: number of nodes
    :param DOF: Dof
    :return: zero stiffness n force
    """
    return np.zeros((nnod * DOF, nnod * DOF)), np.zeros((nnod * DOF, 1))


def impose_boundary_condition(K, f, ibc, bc):
    """
    :param K: Stiffness matrix
    :param f: force vector
    :param ibc: node at with BC is prescribed
    :param bc: boundary condition
    :return: stiffness matrix and force vector after imposed bc
    """
    f = f - (K[:, ibc] * bc)[:, None]
    f[ibc] = bc
    K[:, ibc] = 0
    K[ibc, :] = 0
    K[ibc, ibc] = 1
    return K, f


def get_displacement_vector(K, f):
    """
    :param K: Non-singular stiffness matrix
    :param f: force vector
    :return: nodal displacement
    """
    return np.linalg.solve(K, f)



def get_assembly_vector(DOF, n):
    """
    :param DOF: dof
    :param n: nodes
    :return: assembly points
    """
    iv = []
    for i in n:
        for j in range(DOF):
            iv.append(DOF * i + j)
    return iv

