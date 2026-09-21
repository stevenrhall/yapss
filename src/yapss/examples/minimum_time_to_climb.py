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

import yapss
from yapss._legacy.examples.minimum_time_to_climb import (
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


class Phases(yapss.Phases):
    """One phase: the climb."""

    climb = yapss.phase(state=State, control=Control)


def setup() -> yapss.Problem:
    """Set up the minimum time to climb problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Bryson minimum time to climb", phases=Phases)
    ph = problem.phases.climb

    @ph.register.continuous
    def continuous(arg, out):
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
        return out

    @problem.register.objective
    def objective(arg):
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

    ph.state.h.scale = ph.state.h.defect_scale = 30000.0
    ph.state.v.scale = ph.state.v.defect_scale = 1000.0
    ph.state.gamma.scale = ph.state.gamma.defect_scale = 3.0
    ph.state.mass.scale = ph.state.mass.defect_scale = 500.0
    ph.control.alpha.scale = 0.2
    ph.time.scale = 200.0
    problem.objective.scale = 200.0

    ph.mesh = yapss.Mesh.uniform(segments=15, points=15)
    problem.derivatives.method = "central-difference"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 1000
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
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
