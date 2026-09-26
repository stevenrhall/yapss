"""

The minimum time to climb: how fast an F-4 can reach 65 600 ft.

This is the example whose model is *expensive*. Every evaluation looks up the thrust in a
radial-basis interpolator and the atmosphere and aerodynamic coefficients in cubic splines, so
the dynamics cost far more than the plumbing around them -- which is what a compiled or
black-box model looks like, and which is why the released version of this problem uses central
differences rather than tracing.

The tables, the splines and the interpolator are the released version's, built the same way and
in the same order, so any difference in the answer is the API's and not a slip in transcribing
the physics.

"""

# The aerodynamic coefficients are conventionally capitalized.
# ruff: noqa: N806

__all__ = ["main", "plot_solution", "setup"]

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, RBFInterpolator

import yapss
from yapss.math import cos, sin

# -- the model: the thrust table, the atmosphere, and the aerodynamic coefficients ---------------

# mach number array, and altitude array in thousands of feet
mach_data = np.array((0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8))
h_data = np.array((0, 5, 10, 15, 20, 25, 30, 40, 50, 70), dtype=float)

# normalize so that each array has range [0,1]
h_data /= 70.0
mach_data /= 1.8

# Thrust data. Note that there are zero entries where the data is unknown or
# undefined. Thrust is in thousands of lbf.
# fmt:off
thrust_data = np.array(
    [[24.2,    0,    0,    0,    0,    0,    0,    0,    0,    0],
     [28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7,    0,    0,    0],
     [28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2,  7.3,  4.4,    0],
     [30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3,  8.1,  4.9,    0],
     [34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1,  9.4,  5.6,  1.1],
     [37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2,  6.8,  1.4],
     [36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4,  8.3,  1.7],
     [   0, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0,  2.2],
     [   0,    0,    0, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9,  2.9],
     [   0,    0,    0,    0,    0, 34.6, 31.1, 21.7, 13.3,  3.1]],
)
# fmt: on

# convert to lbf
thrust_data *= 1000

# Find non-empty entries in thrust table, and form argument and value arrays for the
# radial basis function interpolator
thrust_table = []
mh = []
for j, mj in enumerate(mach_data):
    for k, hk in enumerate(h_data):
        thrust_data_point = thrust_data[j][k]
        if thrust_data_point != 0:
            thrust_table.append(thrust_data_point)
            mh.append([mj, hk])

thrust_rbf_interpolator = RBFInterpolator(
    mh,
    thrust_table,
    smoothing=0,
    kernel="cubic",
)


def thrust_function(mach: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Determine the thrust available at the given mach numbers and altitudes.

    Parameters
    ----------
    mach : array_like
        The Mach numbers.
    h : array_like
        The altitudes (ft).

    Returns
    -------
    numpy.ndarray
        The thrust at each Mach number and altitude (lbf).
    """
    shape = mach.shape
    length = 1
    for i in shape:
        length *= i
    mach = mach.reshape([length])
    h = h.reshape([length])
    thrust = thrust_rbf_interpolator(np.stack([mach / 1.8, h / 70000], -1))
    return np.array(thrust.reshape(shape), dtype=float)


# make splines of atmospheric data, using the U.S. 1976 Standard Atmosphere in US
# customary units. Data from: http://www.pdas.com/atmosTable1US.html

# fmt: off
atmosphere_data = np.array(
    #  h     rho       c
    # --  --------  ------
    [[ 0, 2.377E-3, 1116.5],
     [ 5, 2.048E-3, 1097.1],
     [10, 1.756E-3, 1077.4],
     [15, 1.496E-3, 1057.4],
     [20, 1.267E-3, 1036.9],
     [25, 1.066E-3, 1016.1],
     [30, 8.907E-4,  994.8],
     [35, 7.382E-4,  973.1],
     [40, 5.873E-4,  968.1],
     [45, 4.623E-4,  968.1],
     [50, 3.639E-4,  968.1],
     [55, 2.865E-4,  968.1],
     [60, 2.256E-4,  968.1],
     [65, 1.777E-4,  968.1],
     [70, 1.392E-4,  970.9],
     [75, 1.091E-4,  974.3],
     [80, 8.571E-5,  977.6],
     [85, 6.743E-5,  981.0],
     [90, 5.315E-5,  984.3]],
)
# fmt: on

atmosphere_data[:, 0] *= 1000

get_rho = CubicSpline(atmosphere_data[:, 0], atmosphere_data[:, 1])
get_c = CubicSpline(atmosphere_data[:, 0], atmosphere_data[:, 2])

# cubic splines of areodynamic parameters. In order to get desired results, some
# spline points are doubled, effectively forcing the slope at those points to be zero.
# Plots of the resulting spline functions show that the desired result is obtained.
eps = 1e-5

# lift curve slope (CLalpha)
mach_cla = [0, 0.4, 0.8, 0.84 - eps, 0.84, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
cla = [3.44, 3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44]
get_cla = CubicSpline(mach_cla, cla)

# baseline drag coefficient (CD0)
mach_cd0 = [0, 0.4, 0.8, 0.86 - eps, 0.86, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
cd0 = [0.013, 0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035]
get_cd0 = CubicSpline(mach_cd0, cd0)

# eta
mach_eta = [
    0,
    0.4,
    0.8 - eps,
    0.8,
    0.9,
    1.0,
    1.0 + eps,
    1.2 - eps,
    1.2,
    1.4,
    1.6,
    1.6 + eps,
    1.8 - eps,
    1.8,
]
eta_data = [
    0.54,
    0.54,
    0.54,
    0.54,
    0.75 - 0.01,
    0.79,
    0.79 - eps / 10,
    0.78 + eps / 10,
    0.78,
    0.89,
    0.93,
    0.93,
    0.93,
    0.93,
]
get_eta = CubicSpline(mach_eta, eta_data)


S = 530.0
"""Aerodynamic reference area (ft^2)."""
g0 = 32.174
"""Gravitational acceleration (ft/s^2)."""
Isp = 1600.0
"""Specific impulse (s)."""


class State(yapss.State):
    """Where the aircraft is, how fast it is going, and what it weighs."""

    h = yapss.scalar()
    """Altitude."""
    v = yapss.scalar()
    """Speed."""
    gamma = yapss.scalar()
    """Flight path angle."""
    mass = yapss.scalar()
    """Mass."""


class Control(yapss.Control):
    """How the aircraft is flown."""

    alpha = yapss.scalar()
    """Angle of attack."""


class Climb(yapss.Phase):
    """The whole climb."""

    state: State
    control: Control
    time: yapss.Independent


class Phases(yapss.Phases):
    """One phase: the climb."""

    climb: Climb


# Names for the types the annotations below use.
ClimbArg = yapss.ContinuousArg[State, Control]
ClimbOut = yapss.ContinuousOut[State]
MinimumTimeToClimbProblem = yapss.Problem[Phases]


def setup() -> MinimumTimeToClimbProblem:
    """Set up the minimum time to climb problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Bryson minimum time to climb", phases=Phases)
    ph = problem.phases.climb

    @ph.register.continuous
    def continuous(arg: ClimbArg, out: ClimbOut) -> None:
        """Compute the aircraft's dynamics, looking the model up in tables."""
        h, v = arg.state.h, arg.state.v
        gamma, mass = arg.state.gamma, arg.state.mass
        alpha = arg.control.alpha

        rho = get_rho(h)
        mach = v / get_c(h)
        CD0 = get_cd0(mach)
        Clalpha = get_cla(mach)
        eta = get_eta(mach)
        thrust = thrust_function(mach, h)

        CD = CD0 + eta * Clalpha * alpha**2
        CL = Clalpha * alpha
        q = 0.5 * rho * v**2
        D = q * S * CD
        L = q * S * CL

        out.dynamics.h = v * sin(gamma)
        out.dynamics.v = (thrust * cos(alpha) - D) / mass - g0 * sin(gamma)
        out.dynamics.gamma = (thrust * sin(alpha) + L - mass * g0 * cos(gamma)) / (mass * v)
        out.dynamics.mass = -thrust / (g0 * Isp)

    @problem.register.objective
    def objective(arg: yapss.DiscreteArg) -> Any:
        """Return the time taken to climb."""
        return arg[ph].final.time

    h0, v0, gamma_0, m0 = 0.0, 424.260, 0.0, 42000.0 / g0
    hf, vf, gamma_f = 65600.0, 968.148, 0.0
    m_min, m_max = 10.0, 45000.0 / g0

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (100.0, 800.0)
    ph.state.h.initial = (h0, h0)
    ph.state.v.initial = (v0, v0)
    ph.state.gamma.initial = (gamma_0, gamma_0)
    ph.state.mass.initial = (m0, m0)
    ph.state.h.bounds = (0.0, 69000.0)
    ph.state.v.bounds = (1.0, 2000.0)
    ph.state.gamma.bounds = (-40 * pi / 180, 40 * pi / 180)
    ph.state.mass.bounds = (m_min, m_max)
    ph.state.h.final = (hf, hf)
    ph.state.v.final = (vf, vf)
    ph.state.gamma.final = (gamma_f, gamma_f)
    ph.state.mass.final = (m_min, m_max)
    ph.control.alpha.bounds = (-pi / 4, pi / 4)

    ph.time.guess = (0.0, 300.0)
    ph.state.h.guess = (h0, hf)
    ph.state.v.guess = (v0, vf)
    ph.state.gamma.guess = (gamma_0, gamma_f)
    ph.state.mass.guess = (m0, m0)
    ph.control.alpha.guess = (0.0, 0.0)

    ph.state.h.scale = 30000.0
    ph.state.v.scale = 1000.0
    ph.state.gamma.scale = 3.0
    ph.state.mass.scale = 500.0
    ph.control.alpha.scale = 0.2
    ph.time.scale = 200.0
    problem.objective.scale = 200.0

    ph.mesh = yapss.Mesh.uniform(segments=15, points=15)
    problem.derivatives.method = "central-difference"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 1000
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: MinimumTimeToClimbProblem, solution: yapss.Solution) -> None:
    r"""Plot the climb: the trajectory, the four states, the control, and the Hamiltonian.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.climb]
    t = ps.time

    # the trajectory, in the plane the climb is really flown in
    plt.figure()
    plt.plot(ps.state.v, ps.state.h / 1000.0, linewidth=3)
    plt.xlabel(r"Velocity, $v$ (ft/s)")
    plt.ylabel(r"Altitude, $h$ (1000 ft)")
    plt.xlim(0, 1800)
    plt.ylim(-0.3, 65)
    plt.grid()
    plt.tight_layout()

    panels = (
        (r"Altitude, $h$ (1000 ft)", ps.state.h / 1000.0),
        (r"Velocity, $v$ (ft/s)", ps.state.v),
        (r"Flight path angle, $\gamma$ (deg)", ps.state.gamma * 180 / pi),
        (r"Mass, $m$ (slug)", ps.state.mass),
        (r"Angle of attack, $\alpha$ (deg)", ps.control.alpha * 180 / pi),
        (r"Hamiltonian, $\mathcal{H}$", ps.hamiltonian),
    )
    for ylabel, quantity in panels:
        plt.figure()
        plt.plot(t, quantity)
        plt.xlabel(r"Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.xlim(t[0], t[-1])
        plt.grid()
        plt.tight_layout()


def main() -> None:
    """Solve the minimum time to climb problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"time to climb = {solution.objective:.4f} s")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
