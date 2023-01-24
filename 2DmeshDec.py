import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def get_2d_connectivity(nx, ny):
    """
    :param nx: no. of elements along width
    :param ny: no. of elements along height
    :return: icon, plot handle of mesh
    """
    nelm = nx * ny
    nnod = (nx + 1) * (ny + 1)
    height = 2 * ny
    width = 2 * nx
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    x_1, y_1 = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.scatter(x_1, y_1)
    x_1 = x_1.reshape(1, nnod)[0]
    y_1 = y_1.reshape(1, nnod)[0]
    node_array = [np.arange(0, nnod, 1, dtype=np.int32), x_1, y_1]
    icon = np.zeros((5, nelm), dtype=np.int32)
    icon[0, :] = np.arange(0, nelm, 1)
    icon[1, :] = np.where((node_array[1] != width) & (node_array[2] != height))[0]
    icon[2, :] = icon[1, :] + nx + 1
    icon[3, :] = icon[1, :] + 1
    icon[4, :] = icon[2, :] + 1
    icon = icon.transpose()
    for i in range(nnod):
        ax.text(node_array[1][i], node_array[2][i], str(node_array[0][i]))
    for i in range(nelm):
        ax.text(node_array[1][icon[i][1]] + 1, node_array[2][icon[i][1]] + 1, str(icon[i][0]),
                fontsize=12, color="blue")
    ax.axis("equal")
    ax.set_xlabel("Elements in blue , nodes in white")
    ax.set_title(str(nx)+"x"+str(ny) + " mesh with " + str(nnod) + " nodes")
    return icon, plt


x, p = get_2d_connectivity(5, 6)
print(x)
p.show()

