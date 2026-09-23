"""

The Goddard rocket problem, in three phases, the middle one a singular arc.

A rocket rises vertically against drag and gravity, burning fuel to reach the greatest
altitude it can. The optimal thrust programme is bang-singular-bang: full thrust, then a
singular arc along which a switching function vanishes, then coasting with the engine off.
Each arc is a phase, and the phases are joined by continuity constraints on time and state.

"""

__all__ = ["main", "plot_solution", "setup"]

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


class SingularArc(yapss.Path):
    """The condition that holds along the singular arc."""

    switching = yapss.scalar()
    """Singular-arc switching function."""


class Discrete(yapss.Discrete):
    """Continuity of time and state where one phase meets the next.

    One field per quantity rather than one block per joint, because the quantities are of
    different sizes -- an altitude, a speed, a mass, a time -- and a scale factor is one
    number per field. A block of the four could not be scaled at all without giving the four
    numbers positionally, which is the counting this API exists to remove. This problem
    converges without scaling them, so none is set; the point is that it could be.
    """

    boost_singular_h = yapss.scalar()
    """Altitude, boost to singular."""
    boost_singular_v = yapss.scalar()
    """Speed, boost to singular."""
    boost_singular_m = yapss.scalar()
    """Mass, boost to singular."""
    boost_singular_time = yapss.scalar()
    """Time, boost to singular."""
    singular_coast_h = yapss.scalar()
    """Altitude, singular to coast."""
    singular_coast_v = yapss.scalar()
    """Speed, singular to coast."""
    singular_coast_m = yapss.scalar()
    """Mass, singular to coast."""
    singular_coast_time = yapss.scalar()
    """Time, singular to coast."""


class Arc(yapss.Phase):
    """A thrust arc, the shape boost and coast share."""

    state: State
    control: Control
    time: yapss.Independent


class Singular(yapss.Phase):
    """The singular arc: a thrust arc, plus the constraint that keeps it singular."""

    state: State
    control: Control
    path: SingularArc
    time: yapss.Independent


class Phases(yapss.Phases):
    """The three arcs of the flight."""

    boost: Arc
    singular: Singular
    coast: Arc


def drag(h, v):
    """Return the drag on the rocket at altitude `h` and speed `v`."""
    return sigma * v**2 * exp(-h / h0)


def setup() -> yapss.Problem:
    """Set up the Goddard rocket problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Goddard rocket with singular arc", phases=Phases, discrete=Discrete)
    phases = problem.phases
    boost, singular, coast = phases.boost, phases.singular, phases.coast

    def rocket(arg, xdot):
        """Fill in the rocket's dynamics, which are the same in every phase."""
        h, v, m = arg.state.h, arg.state.v, arg.state.m
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
        h, v, m = arg.state.h, arg.state.v, arg.state.m
        out.path.switching = m * g - (1 + v / c) * drag(h, v)
        return out

    @problem.register.objective
    def objective(arg):
        """Return the altitude reached, which is to be made as large as possible."""
        return arg[coast].final.h

    @problem.register.discrete
    def discrete(arg, out):
        """Require time and state to be continuous where the phases meet."""
        b, s, e = arg[boost], arg[singular], arg[coast]
        out.discrete.boost_singular_h = s.initial.h - b.final.h
        out.discrete.boost_singular_v = s.initial.v - b.final.v
        out.discrete.boost_singular_m = s.initial.m - b.final.m
        out.discrete.boost_singular_time = s.initial.time - b.final.time
        out.discrete.singular_coast_h = e.initial.h - s.final.h
        out.discrete.singular_coast_v = e.initial.v - s.final.v
        out.discrete.singular_coast_m = e.initial.m - s.final.m
        out.discrete.singular_coast_time = e.initial.time - s.final.time
        return out

    problem.objective.sense = "maximize"

    for ph in phases:
        ph.state.h.bounds = (0, 20_000)
        ph.state.v.bounds = (0, 10_000)
        ph.state.m.bounds = (mf, m0)

    boost.time.initial = (0.0, 0.0)
    boost.state.h.initial = (0.0, 0.0)
    boost.state.v.initial = (0.0, 0.0)
    boost.state.m.initial = (m0, m0)
    coast.state.v.final = (0, None)
    coast.state.m.final = (mf, mf)

    boost.control.thrust.bounds = (Tm, Tm)
    singular.control.thrust.bounds = (0.01 * Tm, 0.99 * Tm)
    coast.control.thrust.bounds = (0.0, 0.0)

    singular.path.switching.bounds = (0.0, 0.0)
    for name in Discrete._fields:
        getattr(problem.discrete, name).bounds = (0.0, 0.0)

    # Over the handles rather than over `phases`: iterating the container yields a phase of
    # unknown shape, which has no `time` -- the name of the independent variable is the shape's.
    # A tuple of the handles keeps both shapes, and both of them name it `time`.
    for ph in (boost, singular, coast):
        k = ph.index
        ph.time.guess = (15.0 * k, 15.0 * (k + 1))
        ph.state.h.guess = (6000 * k, 6000 * (k + 1))
        ph.state.v.guess = (500.0, 500.0)
        ph.state.m.guess = (3 - 2 / 3 * k, 3 - 2 / 3 * (k + 1))
        ph.control.thrust.guess = (Tm * (2 - k) / 2, Tm * (2 - k) / 2)

    problem.method = "lgl"
    problem.derivatives.method = "auto"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 500
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the thrust, the state histories and the Hamiltonian of every phase.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    panels = (
        ("Thrust, $T$ (lbf)", lambda ps: ps.control.thrust),
        ("Altitude, $h$ (ft)", lambda ps: ps.state.h),
        ("Velocity, $v$ (ft/s)", lambda ps: ps.state.v),
        ("Mass, $m$ (slug)", lambda ps: ps.state.m),
        (r"Hamiltonian, $\mathcal{H}$", lambda ps: ps.hamiltonian),
    )
    for ylabel, quantity in panels:
        plt.figure()
        for ph in problem.phases:
            ps = solution[ph]
            plt.plot(ps.time, quantity(ps), label=ph.name)
        plt.xlabel("Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.tight_layout()


def main() -> None:
    """Solve the Goddard rocket problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final altitude = {solution.objective:.1f} ft")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
