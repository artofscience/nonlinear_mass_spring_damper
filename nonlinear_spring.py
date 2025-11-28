"""
Comparing springs with varying amount of non-linearity (stiffening).
"""

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from msd import MSD

msd = MSD()  # setup mass-spring-damper system

time = [0.0, 100]  # time span
y0 = [0.0, 1.0]  # perturb system with nonzero initial velocity

for power in [1, 3, 5, 7]:
    msd.fs = lambda x: x ** power
    sol = solve_ivp(msd, time, y0, atol=1e-10, rtol=1e-10)  # solve system
    plt.plot(sol.t, sol.y[0], label=f"Power = {power}")  # plot position vs. time

plt.legend()
plt.show()  # show plot
