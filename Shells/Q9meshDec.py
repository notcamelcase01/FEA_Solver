import  numpy as np
from matplotlib import pyplot as plt

def get_2D_connectivity_Q9(nx, ny, lx, ly):
    """
    :param element_type: element type 2 is for Q9 and 1 is for Q4
    :param lx: total width
    :param ly: total height
    :param nx: no. of elements along width
    :param ny: no. of elements along height
    :return: icon, plot handle of mesh
    """
    nelm = nx * ny
    nnod = (2 * nx + 1) * (2 * ny + 1)
    height = ly
    width = lx
    x = np.linspace(0, width, (2 * nx + 1))
    y = np.linspace(0, height, (2 * ny + 1))
    x_1, y_1 = np.meshgrid(x, y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.scatter(x_1, y_1)
    x_1 = x_1.reshape(1, nnod)[0]
    y_1 = y_1.reshape(1, nnod)[0]
    node_array = np.array([np.arange(0, nnod, 1, dtype=np.int32), x_1, y_1])
    icon = np.zeros((10, nelm), dtype=np.int32)
    icon[0, :] = np.arange(0, nelm, 1)
    for i in range(nelm):
        icon[1, i]  =  2 * i + (2 * i) // (2 * nx)  * (2 + 2 * nx)
    icon[2, :] = icon[1, :] + 1
    icon[3, :] = icon[2, :] + 1
    icon[4, :] = icon[3, :] + (2 * nx + 1)
    icon[5, :] = icon[4, :] + (2 * nx + 1)
    icon[6, :] = icon[5, :] - 1
    icon[7, :] = icon[6, :] - 1
    icon[8, :] = icon[7, :] - (2 * nx + 1)
    icon[9, :] = icon[8, :] + 1
    icon = icon.transpose()
    for i in range(nnod):
        ax1.text(node_array[1][i], node_array[2][i], str(node_array[0][i]), fontsize=6)
    for i in range(nelm):
        ax1.plot([node_array[1][icon[i][1]], node_array[1][icon[i][3]]], [node_array[2][icon[i][1]], node_array[2][icon[i][3]]],'-k')
        ax1.plot([node_array[1][icon[i][1]], node_array[1][icon[i][7]]], [node_array[2][icon[i][1]], node_array[2][icon[i][7]]],'-k')
        ax1.plot([node_array[1][icon[i][5]], node_array[1][icon[i][7]]], [node_array[2][icon[i][5]], node_array[2][icon[i][7]]],'-k')
        ax1.plot([node_array[1][icon[i][5]], node_array[1][icon[i][3]]], [node_array[2][icon[i][5]], node_array[2][icon[i][3]]],'-k')
        ax1.text(0.4 * node_array[1][icon[i][1]] + 0.6 * node_array[1][icon[i][9]], 0.4 * node_array[2][icon[i][1]] + 0.6 * node_array[2][icon[i][9]], str(icon[i][0]),
                 fontsize=10, color="red")
    ax1.axis("equal")
    ax1.set_xlabel("Elements in red, nodes in blak")
    ax1.set_title(str(nx) + "x" + str(ny) + " mesh with " + str(nnod) + " nodes")
    columns = ("Element", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9")
    ax2.table(cellText=icon, colLabels=columns, loc="center")
    ax2.axis('tight')
    ax2.axis('off')
    ax2.set_title("icon matrix table")
    return icon, node_array, np.meshgrid(x, y), plt


if __name__ == "__main__":
    L = 1
    # DISCRITIZATION OF PLATE
    nx = 5
    ny = 4
    lx = L
    ly = L
    connectivityMatrix, nodalArray, (X, Y), p  = get_2D_connectivity_Q9(nx, ny, lx, ly)
    # connectivityMatrix, nodalArray, (X, Y), ii = get_2d_connectivity_hole(nx, ny, lx, ly, Hx, Hy, by_max, by_min,
    #                                                                          bx_max, bx_min)
    print(nodalArray)
    print(connectivityMatrix)
    p.show()


