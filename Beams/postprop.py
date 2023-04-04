import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('bmh')
def epsilon_2(x, A, S, l, c):
    return A - S  * np.sqrt(c) / np.sinh(l/(2 * np.sqrt(c)))  * np.cosh((2 * x - l)/(2*np.sqrt(c)))

def epsilon_1(x, A, K, l, c):
    return A + (K - A) / np.cosh(l/2/np.sqrt(c)) * np.cosh((2 * x - l)/2/np.sqrt(c))
plt.rc('legend',fontsize='medium') # using a named size


fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlabel("Non-dimensional Distance")
ax.set_ylabel("Strain")
ax.set_ylim((1, 2.5))
plt.suptitle("A = 2 & K = 1")
line, = ax.plot([], [], lw = 3)

def animate(i):
    l = 2 + i/1.5
    x = np.linspace(0, l, 100)
    y = epsilon_1(x, 2, 1, l, 1)
    line.set_data(x, y)
    ax.set_xlim((0, l))
    return line,


anim = FuncAnimation(fig, animate,
                     frames=200, interval=1)

anim.save('continuousSineWave.mp4',
          writer='ffmpeg', fps=30)

