"""

The Goddard rocket problem, in three phases, the middle one a singular arc.

A rocket rises vertically against drag and gravity, burning fuel to reach the greatest
altitude it can. The optimal thrust programme is bang-singular-bang: full thrust, then a
singular arc along which a switching function vanishes, then coasting with the engine off.
Each arc is a phase, and the phases are joined by continuity constraints on time and state.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt

from yapss import _next as yapss
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


class Rocket(yapss.Vector):
    """Where the rocket is, how fast it is going, and what it weighs."""

    h = yapss.field(units="ft", latex="h", doc="altitude")
    v = yapss.field(units="ft/s", latex="v", doc="velocity")
    m = yapss.field(units="slug", latex="m", doc="mass")


class Thrust(yapss.Vector):
    """The engine setting."""

    thrust = yapss.field(units="lbf", latex="T", doc="thrust")


class SingularArc(yapss.Vector):
    """The condition that holds along the singular arc."""

    switching = yapss.field(doc="singular-arc switching function")


class Linkage(yapss.Vector):
    """Continuity of time and state where one phase meets the next."""

    boost_singular = yapss.field(size=4, doc="boost to singular: time, then the three states")
    singular_coast = yapss.field(size=4, doc="singular to coast: time, then the three states")


class Phases(yapss.Phases):
    """The three arcs of the flight."""

    boost = yapss.phase(state=Rocket, control=Thrust)
    singular = yapss.phase(state=Rocket, control=Thrust, path=SingularArc)
    coast = yapss.phase(state=Rocket, control=Thrust)


def drag(h, v):
    """Return the drag on the rocket at altitude `h` and speed `v`."""
    return sigma * v**2 * exp(-h / h0)


def setup() -> yapss.Problem:
    """Set up the Goddard rocket problem.

    Returns
    -------
    yapss._next.Problem
        The problem.
    """
    problem = yapss.Problem("Goddard rocket with singular arc", phases=Phases, discrete=Linkage)
    phases = problem.phases
    boost, singular, coast = phases.boost, phases.singular, phases.coast

    def rocket(arg, xdot):
        """Fill in the rocket's dynamics, which are the same in every phase."""
        h, v, m = arg.state
        thrust = arg.control.thrust
        xdot.h = v
        xdot.v = (thrust - drag(h, v)) / m - g
        xdot.m = -thrust / c

    @boost.register.continuous
    def powered(arg, out):
        """Compute the dynamics of a phase with no path constraint."""
        rocket(arg, out.dynamics)
        return out

    coast.register.continuous(powered)

    @singular.register.continuous
    def singular_arc(arg, out):
        """Compute the dynamics, and the switching function that must vanish."""
        rocket(arg, out.dynamics)
        h, v, m = arg.state
        out.path.switching = m * g - (1 + v / c) * drag(h, v)
        return out

    @problem.register.objective
    def final_altitude(arg):
        """Return the altitude reached, which is to be made as large as possible."""
        return arg[coast].final_state.h

    @problem.register.discrete
    def linkage(arg, out):
        """Require time and state to be continuous where the phases meet."""
        b, s, e = arg[boost], arg[singular], arg[coast]
        out.discrete.boost_singular = [
            s.initial_time - b.final_time,
            *(s.initial_state[:] - b.final_state[:]),
        ]
        out.discrete.singular_coast = [
            e.initial_time - s.final_time,
            *(e.initial_state[:] - s.final_state[:]),
        ]
        return out

    problem.objective.sense = "maximize"

    for ph in phases:
        ph.state.bounds.h = (0, 20_000)
        ph.state.bounds.v = (0, 10_000)
        ph.state.bounds.m = (mf, m0)

    boost.time.initial = 0.0
    boost.state.initial.h = 0.0
    boost.state.initial.v = 0.0
    boost.state.initial.m = m0
    coast.state.final.v = (0, None)
    coast.state.final.m = mf

    boost.control.bounds.thrust = Tm
    singular.control.bounds.thrust = (0.01 * Tm, 0.99 * Tm)
    coast.control.bounds.thrust = 0.0

    singular.path.bounds.switching = 0.0
    problem.discrete.bounds.boost_singular = 0.0
    problem.discrete.bounds.singular_coast = 0.0

    for ph in phases:
        k = ph.index
        ph.time.guess = (15.0 * k, 15.0 * (k + 1))
        ph.state.guess.h = (6000 * k, 6000 * (k + 1))
        ph.state.guess.v = 500.0
        ph.state.guess.m = (3 - 2 / 3 * k, 3 - 2 / 3 * (k + 1))
        ph.control.guess.thrust = Tm * (2 - k) / 2

    problem.method = "lgl"
    problem.derivatives.method = "auto"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 500
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the altitude, thrust, and Hamiltonian of each phase.

    Parameters
    ----------
    problem : yapss._next.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._next.Solution
        The solution to plot.
    """
    panels = (
        ("h", "Altitude (ft)", lambda ps: ps.state.h),
        ("T", "Thrust (lbf)", lambda ps: ps.control.thrust),
        ("H", "Hamiltonian", lambda ps: ps.hamiltonian),
    )
    for figure, (_, ylabel, quantity) in enumerate(panels, start=1):
        plt.figure(figure)
        for ph in problem.phases:
            ps = solution[ph]
            plt.plot(ps.time, quantity(ps), label=ph.name)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.legend()


def main() -> None:
    """Solve the Goddard rocket problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final altitude = {solution.objective:.1f} ft")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
