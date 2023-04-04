import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
def epsilon_2(x, A, S, l, c):
    return A - S  * np.sqrt(c) / np.sinh(l/(2 * np.sqrt(c)))  * np.cosh((2 * x - l)/(2*np.sqrt(c)))

def epsilon_1(x, A, K, l, c):
    A = 0
    return A + (K - A) / np.cosh(l/2/np.sqrt(c)) * np.cosh((2 * x - l)/2/np.sqrt(c))
plt.rc('legend',fontsize='medium') # using a named size


fig, (ax, ay) = plt.subplots(1, 2, figsize=(16, 8))
aa = np.linspace(0, 2, 100)
ax.plot(aa, epsilon_1(aa, 2, 1, 2, 1), label=r"$\mathbf{\frac{l}{\sqrt{c}}}$ = 2")
ax.set_xlabel("Non-dimensional Distance")
ax.set_ylabel("Strain")

aa = np.linspace(0, 20, 100)
ay.plot(aa, epsilon_1(aa, 2, 1, 20, 1), label=r"$\mathbf{\frac{l}{\sqrt{c}}}$ = 20")
ay.set_xlabel("Non-dimensional Distance")
ay.set_ylabel("Strain")
ax.legend(fontsize="23")
ay.legend(fontsize="23")
plt.suptitle("A = 2 & K = 1")
#ax.set_ylim((0, 1.25))
ay.set_ylim((0, 2.5))

plt.tight_layout()
plt.show()
