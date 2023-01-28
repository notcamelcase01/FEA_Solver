import numpy as np
import matplotlib.pyplot as plt


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.scatter(x_1, y_1)
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
        ax1.text(node_array[1][i], node_array[2][i], str(node_array[0][i]))
    for i in range(nelm):
        ax1.text(node_array[1][icon[i][1]] + 1, node_array[2][icon[i][1]] + 1, str(icon[i][0]),
                 fontsize=12, color="blue")
    ax1.axis("equal")
    ax1.set_xlabel("Elements in blue , nodes in white")
    ax1.set_title(str(nx) + "x" + str(ny) + " mesh with " + str(nnod) + " nodes")
    columns = ("Element", "#1", "#2", "#3", "#4")
    ax2.table(cellText=icon, colLabels=columns, loc="center")
    ax2.axis('tight')
    ax2.axis('off')
    ax2.set_title("icon matrix table")
    return icon, plt


ic, p = get_2d_connectivity(2, 3)
p.show()
