import numpy as np
import keywords as param


def assemble_stiffness(kloc, iv, n):
    """
    :param kloc: local stiffness matrix
    :param iv: nodes where stiffness are to be placed
    :param n: DOF*number of nodes
    :return: stiffness matrix to be added to global stiffness
    """
    K = np.zeros((n, n))
    for i in range(len(iv)):
        for j in range(len(iv)):
            K[iv[i]][iv[j]] = kloc[i][j]
    return K


def assemble_force(floc, iv, n):
    """
    :param floc: local force vector
    :param iv: nodes where forces are to be placed
    :param n: DOF*number of nodes
    :return: force vector to be added to global force vector
    """
    K = np.zeros((n, 1))
    for i in range(len(iv)):
        K[iv[i]] = floc[i]
    return K


def get_node_points_coords(n, l):
    """
    :param n: number of nodes
    :param l: length
    :return: node coordinates
    """
    return np.linspace(0, l, n)


def get_connectivity_matrix(nelm, element_type=2):
    """
    :param nelm: number of elements
    :param element_type: 2 for linear
    :return: connectivity matrix
    """
    icon = np.zeros((nelm, 2), dtype=np.int32)
    if element_type == param.ElementType.LINEAR:
        for i in range(nelm):
            icon[i, 0] = i
            icon[i, 1] = 1 + i
    else:
        raise Exception("Uhm, This is wendy's, we don't, quadratic element that here")
    return icon


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


def get_node_from_element(icon, elm, elm_type=2):
    """
    :param icon: connectivity matrix
    :param elm: element number
    :param elm_type: element type 2 for 2 noded linear
    :return: nodes associated with that element
    """
    x = []
    for i in range(int(elm_type)):
        x.append(icon[elm][i])
    return x


def get_hermite_fn(gp, J):
    """
    :param gp: eta or gauss points or natural coordinate
    :param J: jacobian
    :return: (H,H")
    """
    Nmat = np.array([.25 * (gp + 2) * (1 - gp) ** 2, J * .25 * (gp + 1) * (1 - gp) ** 2,
                     .25 * (-gp + 2) * (1 + gp) ** 2, J * .25 * (gp - 1) * (1 + gp) ** 2])
    Bmat = (1 / J ** 2) * np.array([1.5 * gp, (-.5 + 1.5 * gp) * J, -1.5 * gp, (.5 + 1.5 * gp) * J])
    return Nmat.reshape(Nmat.shape[0], 1), Bmat.reshape(Bmat.shape[0], 1)


def get_lagrange_fn(gp, J, element_type=2):
    """
    :param gp: gauss point
    :param J: jacobian
    :param element_type: element_type 2=LINEAR
    :return: (L,L")
    """
    if element_type != 2:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 gauss points here")
    Nmat = np.array([.5 * (-gp + 1), .5 * (1 + gp)])
    Bmat = np.array((1 / J) * [-.5, .5])
    return Nmat.reshape(Nmat.shape[0], 1), Bmat.reshape(Bmat.shape[0], 1)


def impose_boundary_condition(K, f, ibc, bc):
    """
    :param K: Stiffness matrix
    :param f: force vector
    :param ibc: node at with BC is prescribed
    :param bc: boundary condition
    :return: stiffness matrix and force vector after imposed bc
    """
    f = f - (K[:, ibc] * bc).reshape((len(f), 1))
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
    try:
        return np.linalg.inv(K).dot(f)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            return 0
        else:
            raise
