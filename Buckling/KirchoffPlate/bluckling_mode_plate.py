import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')



nxnydof = np.load('data_plate/nxnydof.npy', allow_pickle=True)
nx = nxnydof[0]
ny = nxnydof[1]
DOF  = nxnydof[2]
query_mode = 1
X0 = np.load('data_plate/X0.npy', allow_pickle=True)
Y0 = np.load('data_plate/Y0.npy', allow_pickle=True)
eigenvectors = np.load('data_plate/eigvec.npy', allow_pickle=True)
eigenvalues = np.load('data_plate/eigval.npy', allow_pickle=True)

w0 = eigenvectors[np.arange(0, len(eigenvalues), DOF), query_mode - 1]
w0 = w0.reshape((ny + 1, nx + 1))
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_wireframe(X0, Y0, w0)
ax.set_title("Buckling Mode - {y} at critical load of (N) : {x}".format(y = query_mode, x = eigenvalues[query_mode - 1]))
ax.set_axis_off()
ax.set_box_aspect(aspect = (1, 1, 1))
plt.show()
