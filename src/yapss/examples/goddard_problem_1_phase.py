"""

The Goddard rocket problem in one phase.

A rocket rises vertically against drag and gravity, burning fuel to reach the greatest altitude
it can. The answer is bang-singular-bang: full thrust, then an arc along which a switching
function vanishes and the thrust takes an interior value, then coasting.

One phase cannot represent that middle arc, and watching it fail to is the reason this example
is worth having beside `goddard_problem_3_phase.py`. The transcription never asks the thrust to
be smooth, so across the singular region the solution chatters -- banging between zero and full
from one collocation point to the next, whose average is close to the singular thrust and whose
altitude is close to the right one. The minimizer is minimizing the problem it was given, and
the problem it was given does not say the arc is an arc. Stating the three arcs as three phases
is what says it, which is what the other example does.

"""

__all__ = ["main", "plot_solution", "setup"]

from typing import Any

import matplotlib.pyplot as plt

import yapss
from yapss.math import exp

Tm = 193.044
"""Maximum thrust (lbf)."""
g = 32.174
"""Gravitational acceleration (ft/s^2)."""
sigma = 5.49153484923381010e-05
"""Drag coefficient."""
c = 1580.9425279876559
"""Exhaust velocity (ft/s)."""
h0 = 23800.0
"""Density scale height (ft)."""
m0, mf = 3.0, 1.0
"""Initial and final mass (slug)."""
h_max, v_max = 30_000.0, 15_000.0
"""How far the trajectory is allowed to reach."""
tf_min, tf_max = 20.0, 100.0
"""How long the flight may last."""


class State(yapss.State):
    """Where the rocket is, how fast it is going, and what it weighs."""

    h = yapss.scalar()
    """Altitude."""
    v = yapss.scalar()
    """Velocity."""
    m = yapss.scalar()
    """Mass."""


class Control(yapss.Control):
    """The engine setting."""

    thrust = yapss.scalar()
    """Thrust."""


class Flight(yapss.Phase):
    """The whole flight."""

    state: State
    control: Control
    time: yapss.Independent


class Phases(yapss.Phases):
    """One phase: the whole flight, whatever shape the thrust programme turns out to have."""

    flight: Flight


def setup() -> yapss.Problem[Phases]:
    """Set up the one-phase Goddard rocket problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Goddard rocket, one phase", phases=Phases)
    ph = problem.phases.flight

    @ph.register.continuous
    def continuous(
        arg: yapss.ContinuousArg[State, Control], out: yapss.ContinuousOut[State]
    ) -> None:
        """Compute the rocket's dynamics."""
        h, v, m = arg.state.h, arg.state.v, arg.state.m
        thrust = arg.control.thrust
        out.dynamics.h = v
        out.dynamics.v = (thrust - sigma * v**2 * exp(-h / h0)) / m - g
        out.dynamics.m = -thrust / c

    @problem.register.objective
    def objective(arg: yapss.EndpointArg) -> Any:
        """Return the altitude reached, which is to be made as large as possible."""
        return arg[ph].final.h

    problem.objective.sense = "maximize"

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (tf_min, tf_max)
    ph.state.h.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.m.initial = (m0, m0)
    ph.state.h.bounds = (0, h_max)
    ph.state.v.bounds = (0, v_max)
    ph.state.m.bounds = (mf, m0)
    ph.state.m.final = (mf, mf)
    ph.control.thrust.bounds = (0, Tm)

    ph.time.guess = (0.0, tf_max)
    ph.state.h.guess = (0.0, h_max)
    ph.state.v.guess = (0.0, 0.0)
    ph.state.m.guess = (m0, mf)
    ph.control.thrust.guess = (Tm, 0.0)

    ph.state.h.scale = ph.state.h.defect_scale = 18_000.0
    ph.state.v.scale = ph.state.v.defect_scale = 800.0
    ph.state.m.scale = ph.state.m.defect_scale = 3.0
    ph.time.scale = 30.0

    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem[Phases], solution: yapss.Solution) -> None:
    """Plot the trajectory, the thrust programme, and the Hamiltonian.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.flight]
    panels = (
        ("Thrust, $T$ (lbf)", ps.control.thrust),
        ("Altitude, $h$ (ft)", ps.state.h),
        ("Velocity, $v$ (ft/s)", ps.state.v),
        ("Mass, $m$ (slug)", ps.state.m),
        (r"Hamiltonian, $\mathcal{H}$ (ft/s)", ps.hamiltonian),
    )
    for ylabel, quantity in panels:
        plt.figure()
        plt.plot(ps.time, quantity)
        plt.xlabel("Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.xlim((ps.time[0], ps.time[-1]))
        plt.grid()
        plt.tight_layout()


def main() -> None:
    """Solve the one-phase Goddard rocket problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"\nmaximum altitude = {solution.objective:.3f} ft")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
