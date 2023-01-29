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


def get_node_points_coords(n, length):
    """
    :param n: number of nodes
    :param length: length
    :return: node coordinates
    """
    return np.linspace(0, length, n)


def get_connectivity_matrix(nelm, element_type):
    """
    :param nelm: number of elements
    :param element_type: 2 for linear
    :return: connectivity matrix
    """
    # TODO : Remove if statement and use loop
    icon = np.zeros((nelm, element_type), dtype=np.int32)
    if element_type == param.ElementType.LINEAR:
        for i in range(nelm):
            icon[i, 0] = i
            icon[i, 1] = 1 + i
    elif element_type == param.ElementType.QUAD:
        for i in range(nelm):
            icon[i, 0] = 2 * i
            icon[i, 1] = 1 + 2 * i
            icon[i, 2] = 2 + 2 * i
    else:
        raise Exception("Uhm, This is wendy's, we don't, cubic element that here")
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
    # TODO : Do with without taking inverse
    try:
        return np.linalg.inv(K).dot(f)
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


def get_timoshenko_stiffness(Kuu, Kww, Kwt, Ktt, element_type):
    """
    Its understood timoshenko model have DOF = 3
    :param element_type: element type
    :param Kuu: Axial
    :param Kww: Transverse
    :param Kwt: Coupling
    :param Ktt: Shear
    :return: timoshenko stiffness
    """
    st = np.zeros((element_type * 3, element_type * 3))
    for i in range(element_type):
        for j in range(element_type):
            kloc = np.array([[Kuu[i][j], 0, 0],
                             [0, Kww[i][j], Kwt[i][j]],
                             [0, Kwt[j][i], Ktt[i][j]]])
            iv = get_assembly_vector(1, np.arange(3 * i, 3 * i + 3, 1, dtype=np.int32))
            v = get_assembly_vector(1, np.arange(3 * j, 3 * j + 3, 1, dtype=np.int32))
            st += assemble_stiffness(kloc, iv, element_type * 3, v)
    return st


def get_timoshenko_force(ft, fa, m, element_type):
    """
    DOF is assumed to be 3 for timoshenko beam
    :param ft: transverse body force
    :param fa: axial body force
    :param m: body momentum
    :param element_type: element type
    :return: assembled force vector
    """
    stf = np.zeros((element_type*3, 1))
    for i in range(element_type):
        stf[3 * i, 0] = fa[i]
        stf[3 * i + 1, 0] = ft[i]
        stf[3 * i + 2, 0] = m[i]
    return stf


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
