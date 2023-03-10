import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    ic, na, mg = get_2d_connectivity(1, 1, 1, 1)
    print(na)
    print(ic)