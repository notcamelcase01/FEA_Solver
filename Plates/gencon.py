import numpy as np
import matplotlib.pyplot as plt
import math as math

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
    icon[2, :] = icon[1, :] + nx + 1
    icon[3, :] = icon[1, :] + 1
    icon[4, :] = icon[2, :] + 1
    icon = icon.transpose()
    return icon, node_array, np.meshgrid(x, y)


def get_2d_connectivity_trap(nx, ny, lx, ly):
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
    icon[2, :] = icon[1, :] + nx + 1
    icon[3, :] = icon[1, :] + 1
    icon[4, :] = icon[2, :] + 1
    icon = icon.transpose()
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(-height/2, height/2, ny + 1)
    x_1, y_1 = np.meshgrid(x, y)
    y_1 *= 0.5/x.max() * x + 0.5
    x_2 = x_1.reshape(1, nnod)[0]
    y_2 = y_1.reshape(1, nnod)[0]
    node_array = np.array([np.arange(0, nnod, 1, dtype=np.int32), x_2, y_2])
    # fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    # ax1.scatter(x_1, y_1)
    # plt.show()
    return icon, node_array, (x_1, y_1)


def get_2d_connectivity_hole(nx, ny, lx, ly, Hx, Hy, by_max, by_min, bx_max, bx_min):
    """
    :param nx: plate number of elements along x
    :param ny: plates number of elements along y
    :param lx: lx length
    :param ly: ly length
    :param Hx: elements along x in hole
    :param Hy: elements along y in hole
    :param by_max: hole max y
    :param by_min: hole min y
    :param bx_max: hole max x
    :param bx_min: hole min x
    :return: mesh with a hole
    """
    hxc = (bx_min + bx_max) / 2
    hyc = (by_min + by_max) / 2
    nybelow = math.ceil(by_min * (ny+1) / ly)
    nyabove = ny - nybelow
    nxbelow = math.ceil(bx_min * (nx+1) / lx)
    nxabove = nx - nxbelow
    nelm = (nx + Hx) * (ny + Hy)
    nnod = (nx + Hx + 1) * (1 + ny + Hy)
    height = ly
    width = lx
    step_x = lx/nx/30
    step_y = ly/ny/30
    x = np.hstack((np.linspace(0, bx_min, nxbelow,endpoint=False), np.linspace(bx_min, bx_max, Hx,endpoint=False), np.linspace(bx_max, width, nxabove + 1)))
    y = np.hstack((np.linspace(0, by_min, nybelow,endpoint=False), np.linspace(by_min, by_max, Hy,endpoint=False), np.linspace(by_max, height, nyabove + 1)))
    x_1, y_1 = np.meshgrid(x, y)
    x_0, y_0 = np.meshgrid(x, y)


    x_2 = x_1.reshape(1, nnod)[0]
    y_2 = y_1.reshape(1, nnod)[0]
    node_array = np.array([np.arange(0, nnod, 1, dtype=np.int32), x_2, y_2])

    icon = np.zeros((5, nelm), dtype=np.int32)
    icon[0, :] = np.arange(0, nelm, 1)
    icon[1, :] = np.where((node_array[1] != width) & (node_array[2] != height))[0]
    icon[2, :] = icon[1, :] + nx + Hx + 1
    icon[3, :] = icon[1, :] + 1
    icon[4, :] = icon[2, :] + 1
    icon = icon.transpose()

    idx = ((x_1 <= bx_min + step_x) | (x_1 >= bx_max - step_x) | (y_1 <= by_min + step_y) | (y_1 >= by_max - step_y))

    x_0[~idx] = np.nan
    y_0[~idx] = np.nan
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    ax1.scatter(x_0, y_0)
    plt.show()
    # node_array = np.array([np.arange(0, nnod, 1, dtype=np.int32), x_2, y_2])

    return icon, node_array, (x_1, y_1)


if __name__ == "__main__":
    L = 1
    # DISCRITIZATION OF HOLE
    Hx = 5
    Hy = 5
    by_max = 0.3
    by_min = 0.1
    bx_max = 0.9
    bx_min = 0.7
    # DISCRITIZATION OF PLATE
    nx = 8
    ny = 8
    lx = L
    ly = L

    connectivityMatrix, nodalArray, (X, Y) = get_2d_connectivity_hole(nx, ny, lx, ly, Hx, Hy, by_max, by_min,
                                                                             bx_max, bx_min)
    print(nodalArray)
    print(connectivityMatrix)