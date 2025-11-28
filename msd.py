from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class MSD:
    def __init__(self, m=lambda x: 1.0, fs=lambda x: 1.0 * x, fd=lambda v: 0.1 * v, fe=lambda t: 0.0):
        self.m = m
        self.fs = fs
        self.fd = fd
        self.fe = fe

    def __call__(self, t, y):
        return [y[1], (self.fe(t) - self.fd(y[1]) - self.fs(y[0])) / self.m(y[0])]


if __name__ == '__main__':
    msd = MSD()  # setup mass-spring-damper system

    time = [0.0, 100]  # time span
    y0 = [0.0, 1.0]  # perturb system with nonzero initial velocity

    sol = solve_ivp(msd, time, y0, atol=1e-10, rtol=1e-10)  # solve system
    plt.plot(sol.t, sol.y[0])  # plot position vs. time

    plt.show()  # show plot
