import numpy as np
import keywords as param


def assemble_stiffness(kloc, iv, n, v=None):
    """
    :param v: nodes where stiffness are to be places
    :param kloc: local stiffness matrix
    :param iv: nodes where stiffness are to be placed
    :param n: DOF*number of nodes
    :return: stiffness matrix to be added to global stiffness
    """
    # TODO: Remove if statement (make another function if this ain't resolved)
    if v is None:
        v = iv
    K = np.zeros((n, n))
    for i in range(len(iv)):
        for j in range(len(v)):
            K[iv[i]][v[j]] = kloc[i][j]
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


def get_hermite_fn(gp, J):
    """
    :param gp: eta or gauss points or natural coordinate
    :param J: jacobian
    :return: (H,H")
    """
    Nmat = np.array([.25 * (gp + 2) * (1 - gp) ** 2, J * .25 * (gp + 1) * (1 - gp) ** 2,
                     .25 * (-gp + 2) * (1 + gp) ** 2, J * .25 * (gp - 1) * (1 + gp) ** 2])
    Bmat = (1 / J ** 2) * np.array([1.5 * gp, (-.5 + 1.5 * gp) * J, -1.5 * gp, (.5 + 1.5 * gp) * J])
    return Nmat[:, None], Bmat[:, None]


def get_lagrange_fn(gp, J, element_type):
    """
    :param gp: gauss point
    :param J: jacobian
    :param element_type: element_type 2=LINEAR
    :return: (L,L')
    """
    # TODO: use loop instead of if statements
    if element_type == param.ElementType.LINEAR:
        Nmat = np.array([.5 * (1 - gp), .5 * (1 + gp)])
        Bmat = (1 / J) * np.array([-.5, .5])
    elif element_type == param.ElementType.QUAD:
        Nmat = np.array([0.5 * (-1 + gp) * gp, (-gp + 1) * (gp + 1), 0.5 * gp * (1 + gp)])
        Bmat = (1 / J) * np.array([0.5 * (-1 + 2 * gp), -2 * gp, 0.5 * (1 + 2 * gp)])
    else:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 node points here")

    return Nmat[:, None], Bmat[:, None]


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
    # TODO : Do with without taking inverse
    try:
        return np.linalg.inv(K) @ f
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("STOP INTING")
            return 0
        else:
            raise


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

