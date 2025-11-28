"""
Consider the 2nd order ODE:

m[x] * a + d[v] + e[x] = f[t],

where '[.,.]' denotes 'function of'.

or, equivalently, the set of two 1st order ODE:

v = dx/dt
a = ( f[t] - d[v] - e[x] ) / m[x]

with

position x,
velocity v,
acceleration a,

elastic force e[x], e.g. e[x] = kl * x.
Or e[x] = kl * x + knl * x**3, representing a stiffening spring.

damping force d[v], e.g. d[v] = cl * v.
Or d[v] = cl * v + cnl * v**2, representing nonlinear damping, for example from a fluidic resistor.

with kl, knl, cl, cnl some coefficients.

harmonic load f[t] = f0 + fh * sin(2*pi * w[t] * t),

with dead load f0, harmonic load magnitude fh and imposed frequency w[t].
E.g. to apply a frequency sweep, w[t] = s * t,
with s some proportionality constant driving the frequency as function of time.

mass m[x] = m0 + p * x, with p some displacement-based density of units kg/m.
E.g. for a 1D bellow, p ~ rho * A, with fluid density rho and bellow area A.
Note: one has to assure positive mass, e.g. via m0 > p * x for any x.
Dead load f0 can be used to apply preload, as to indirectly ensure x > 0.
"""


from math import pi

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from msd import MSD

m0 = 1e-3  # bellow mass
r = 0.01  # bellow radius
A = pi * r ** 2  # bellow area
rho = 1000  # fluid density
m = lambda x: m0 + rho * A * x  # mass of bellow and fluid

kl = 40  # bellow linear spring constant
knl = 0.5 # bellow nonlinear spring constant
fs = lambda x: kl * x #+ knl * x**3  # elastic force of bellow

cl = 0.01  # bellow linear damping coefficient
cnl = 0.01 # bellow nonlinear damping coefficient
fd = lambda v: cl * v #+ cnl * v**2  # dissipative force of bellow

s = 10000  # speed of frequency sweep
w = lambda t: t / s  # frequency as function of time in Hertz (cycle/s)

f0 = 100  # dead load for preloading (ensuring x remains positive)
fh = 1.0  # magnitude of harmonic force
radps = 2 * pi  # conversion rate Hz to rad/s
fe = lambda t: f0 + fh * np.sin(radps * w(t) * t)  # external harmonic force

msd = MSD(m, fs, fd, fe)  # setup mass-spring-damper

t0 = 5000  # begin time
tend = 8000  # end time

x0 = msd.fe(t0) / msd.fs(1.0)  # initial position (deflection from dead load)
v0 = 0.0  # initial velocity

sol = solve_ivp(msd, [t0, tend], [x0, v0], atol=1e-6, rtol=1e-6)  # solve MSD for position and velocity

plt.plot(sol.t, sol.y[0])  # plot position vs. time
plt.show()  # show plot
