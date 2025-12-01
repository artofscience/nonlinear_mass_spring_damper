from math import pi

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from msd import MSD

class MSDM(MSD):
    """
    Inertial force based on rate of change of momentum F = m * a +  dm/dt * v
    """
    def __init__(self, m=lambda x: 1.0, dmdt = lambda x: 0.0, fs=lambda x: 1.0 * x, fd=lambda v: 0.1 * v, fe=lambda t: 0.0):
        super().__init__(m, fs, fd, fe)
        self.dmdt = dmdt

    def __call__(self, t, y):
        out = super().__call__(t, y)
        out[1] -= self.dmdt(y[1]) * y[1] / self.m(y[0])
        return out

if __name__ == '__main__':

    alpha = 0.1
    m = lambda x: 1e-6 + alpha * x
    dmdt = lambda v: alpha * v

    s = 10000  # speed of frequency sweep
    w = lambda t: t / s  # frequency as function of time in Hertz (cycle/s)

    fe = lambda t: 10 + 0.1 * np.sin(2*pi * w(t) * t)  # external harmonic force
    fd = lambda v: 0.01 * v

    msd = MSDM(m=m, dmdt=dmdt, fe=fe, fd=fd)  # setup mass-spring-damper system

    time = [0.0, 2000]  # time span
    y0 = [10.0, 0.0]  # perturb system with nonzero initial velocity

    sol = solve_ivp(msd, time, y0, atol=1e-10, rtol=1e-10)  # solve system
    plt.plot(sol.t, sol.y[0])  # plot position vs. time

    plt.show()  # show plot
