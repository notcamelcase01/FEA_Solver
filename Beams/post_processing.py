import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = np.linspace(-150, 150, 300)
y = np.linspace(-150, 150, 300)
X,Y = np.meshgrid(x,y)
data = np.exp(-(X/80.)**2-(Y/80.)**2)

R = np.sqrt(X**2+Y**2)
flag =np.logical_not( (R<110) * (R>10) )
data[flag] = np.nan

palette = plt.cm.jet
palette.set_bad(alpha = 0.0)

im = plt.imshow(data)

plt.colorbar(im)
plt.savefig(__file__+".png")
plt.show()