"""

The minimum time to climb: how fast an F-4 can reach 65 600 ft.

This is the example whose model is *expensive*. Every evaluation looks up the thrust in a
radial-basis interpolator and the atmosphere and aerodynamic coefficients in cubic splines, so
the dynamics cost far more than the plumbing around them -- which is what a compiled or
black-box model looks like, and which is why the released version of this problem uses central
differences rather than tracing.

The tables, the splines and the interpolator are imported from the released version, so any
difference in the answer is the API's and not a slip in transcribing the physics.

"""

# The aerodynamic coefficients are conventionally capitalized.
# ruff: noqa: N806

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt
from numpy import pi

from yapss import _next as yapss
from yapss.examples.minimum_time_to_climb import (
    get_c,
    get_cd0,
    get_cla,
    get_eta,
    get_rho,
    thrust_function,
)
from yapss.math import cos, sin

S = 530.0
"""Aerodynamic reference area (ft^2)."""
g0 = 32.174
"""Gravitational acceleration (ft/s^2)."""
Isp = 1600.0
"""Specific impulse (s)."""


class Aircraft(yapss.Vector):
    """Where the aircraft is, how fast it is going, and what it weighs."""

    h = yapss.field(units="ft", latex="h", doc="altitude")
    v = yapss.field(units="ft/s", latex="v", doc="speed")
    gamma = yapss.field(units="rad", latex=r"\gamma", doc="flight path angle")
    mass = yapss.field(units="slug", latex="m", doc="mass")


class AngleOfAttack(yapss.Vector):
    """How the aircraft is flown."""

    alpha = yapss.field(units="rad", latex=r"\alpha", doc="angle of attack")


class Phases(yapss.Phases):
    """One phase: the climb."""

    climb = yapss.phase(state=Aircraft, control=AngleOfAttack)


def setup() -> yapss.Problem:
    """Set up the minimum time to climb problem.

    Returns
    -------
    yapss._next.Problem
        The problem.
    """
    problem = yapss.Problem("Bryson minimum time to climb", phases=Phases)
    ph = problem.phases.climb

    @ph.register.continuous
    def climb(arg, out):
        """Compute the aircraft's dynamics, looking the model up in tables."""
        h, v, gamma, mass = arg.state
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
        return out

    @problem.register.objective
    def minimum_time(arg):
        """Return the time taken to climb."""
        return arg[ph].final_time

    h0, v0, gamma_0, m0 = 0.0, 424.260, 0.0, 42000.0 / g0
    hf, vf, gamma_f = 65600.0, 968.148, 0.0
    m_min, m_max = 10.0, 45000.0 / g0

    ph.time.initial = 0.0
    ph.time.final = (100.0, 800.0)
    ph.state.initial.h = h0
    ph.state.initial.v = v0
    ph.state.initial.gamma = gamma_0
    ph.state.initial.mass = m0
    ph.state.bounds.h = (0.0, 69000.0)
    ph.state.bounds.v = (1.0, 2000.0)
    ph.state.bounds.gamma = (-40 * pi / 180, 40 * pi / 180)
    ph.state.bounds.mass = (m_min, m_max)
    ph.state.final.h = hf
    ph.state.final.v = vf
    ph.state.final.gamma = gamma_f
    ph.state.final.mass = (m_min, m_max)
    ph.control.bounds.alpha = (-pi / 4, pi / 4)

    ph.time.guess = (0.0, 300.0)
    ph.state.guess.h = (h0, hf)
    ph.state.guess.v = (v0, vf)
    ph.state.guess.gamma = (gamma_0, gamma_f)
    ph.state.guess.mass = m0
    ph.control.guess.alpha = 0.0

    ph.state.scale.h = ph.state.defect_scale.h = 30000.0
    ph.state.scale.v = ph.state.defect_scale.v = 1000.0
    ph.state.scale.gamma = ph.state.defect_scale.gamma = 3.0
    ph.state.scale.mass = ph.state.defect_scale.mass = 500.0
    ph.control.scale.alpha = 0.2
    ph.time.scale = 200.0
    problem.objective.scale = 200.0

    ph.mesh = yapss.Mesh.uniform(segments=15, points=15)
    problem.derivatives.method = "central-difference"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 1000
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the climb and the angle of attack that flies it.

    Parameters
    ----------
    problem : yapss._next.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._next.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.climb]
    plt.figure()
    plt.plot(ps.time, ps.state.h / 1000.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (1000 ft)")

    plt.figure()
    plt.plot(ps.time, ps.control.alpha * 180 / pi)
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\alpha$ (deg)")


def main() -> None:
    """Solve the minimum time to climb problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"time to climb = {solution.objective:.4f} s")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
