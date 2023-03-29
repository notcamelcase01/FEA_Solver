import numpy as np
from parameters import  G, Eb, mu, k

def get_lagrange_shape_function(x, y, element_type=2):
    N = np.zeros(element_type ** 2)
    Nx = np.zeros(element_type ** 2)
    Ny = np.zeros(element_type ** 2)
    if element_type == 3:
        xi = (-1, 0, 1, 1, 1, 0, -1, -1, 0)
        yi = (-1, -1, -1, 0, 1, 1, 1, 0, 0)
        for i in range(len(xi)):
            N[i] = ((1.5 * xi[i]**2 - 1) * x**2 + 0.5 * xi[i] * x + 1 - xi[i]**2) * ((1.5 * yi[i]**2 - 1) * y**2 + 0.5 * yi[i] * y + 1 - yi[i]**2)
            Nx[i] = ((1.5 * xi[i]**2 - 1) * x * 2 + 0.5 * xi[i]) * ((1.5 * yi[i]**2 - 1) * y**2 + 0.5 * yi[i] * y + 1 - yi[i]**2)
            Ny[i] = ((1.5 * xi[i]**2 - 1) * x**2 + 0.5 * xi[i] * x + 1 - xi[i]**2) * ((1.5 * yi[i]**2 - 1) * y * 2 + 0.5 * yi[i])
    elif element_type == 2:
        seq = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        for i in range(len(seq)):
            N[i] = 0.25 * (1 + seq[i][0] * x) * (1 + seq[i][1] * y)
            Nx[i] = 0.25 * (seq[i][0] * (1 + seq[i][1] * y))
            Ny[i] = 0.25 * (seq[i][1] * (1 + seq[i][0] * x))
    return N[:, None], Nx[:, None], Ny[:, None]

def get_C1_matrix():
    return Eb * np.array([[1, mu, 0],
                          [mu, 1, 0],
                          [0, 0, (1 - mu)/2]])

def get_C2_matrix():
    return k * np.array([[G, 0],
                     [0, G]])


def get_z_matrix(z):
    return np.array([[1, 0, z, 0, 0],
                    [0, 1, 0, z, 0],
                    [0, 0, 0, 0, 1]])


def get_z1_matrix(z, Rx, Ry):
    return np.array([[1, 0, 0, 0, z, 0, 0, 0, 1/Rx],
                    [0, 0, 0, 1, 0, 0, 0, z, 1/Ry],
                    [0, 1, 1, 0, 0, z, z, 0, 0]])



def get_z2_matrix(Rx, Ry):
    return np.array([[-1/Rx, 0, 1, 0, 1, 0],
                     [0, -1/Ry, 0 ,1, 0, 1]])



def get_b1_matrix(N, Nx, Ny):
    B1 = np.zeros((9, 5 * len(N)))
    for i in range(len(N)):
        B1[0, 5 * i] = Nx[i][0]
        B1[1, 5 * i] = Ny[i][0]
        B1[2, 5 * i + 1] = Nx[i][0]
        B1[3, 5 * i + 1] = Ny[i][0]
        B1[4, 5 * i + 3] = Nx[i][0]
        B1[5, 5 * i + 3] = Ny[i][0]
        B1[6, 5 * i + 4] = Nx[i][0]
        B1[7, 5 * i + 4] = Ny[i][0]
        B1[8, 5 * i + 2] = N[i][0]
    return B1


def get_b2_matrix(N, Nx, Ny):
    B2 = np.zeros((6, 5 * len(N)))
    for i in range(len(N)):
        B2[0, 5 * i] = N[i][0]
        B2[1, 5 * i + 1] = N[i][0]
        B2[2, 5 * i + 2] = Nx[i][0]
        B2[3, 5 * i + 2] = Ny[i][0]
        B2[4, 5 * i + 3] = N[i][0]
        B2[5, 5 * i + 4] = N[i][0]
    return B2


def get_n_matrix(N):
    N1 = np.zeros((5, 5 * len(N)))
    for i in range(len(N)):
        N1[0, 5 * i] = N[i][0]
        N1[1, 5 * i + 1] = N[i][0]
        N1[2, 5 * i + 2] = N[i][0]
        N1[3, 5 * i + 3] = N[i][0]
        N1[4, 5 * i + 4] = N[i][0]
    return N1


if __name__ == "__main__":
    xi = [np.sqrt(1/3)]
    yi = [np.sqrt(1/3)]
    for i in range(len(xi)):
        N, Nx, Ny = get_lagrange_shape_function(xi[i], yi[i])
        print(N.T)
        print(Nx.T)
        print(Ny.T)