"""

The brachistochrone problem, written as briefly as the API allows.

This is the shortest complete statement of a problem: the variables are named in two classes,
the phase is declared, the dynamics and the objective are registered, and the bounds and the
guess are set. `brachistochrone.py` is the same problem with docstrings and commentary.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt
from numpy import pi

import yapss
from yapss.math import cos, sin

G0 = 32.174
"""Acceleration of gravity, ft/s^2."""


class State(yapss.Vector):
    """Where the bead is and how fast it is going."""

    x = yapss.field()

    y = yapss.field()

    v = yapss.field()


class Control(yapss.Vector):
    """The slope of the path."""

    u = yapss.field()


class Phases(yapss.Phases):
    """One phase: the bead slides."""

    slide = yapss.phase(state=State, control=Control)


def setup() -> yapss.Problem:
    """Set up the brachistochrone problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Brachistochrone", phases=Phases)
    ph = problem.phases.slide

    @ph.register.continuous
    def continuous(arg, out):
        """Compute the bead's dynamics."""
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = G0 * sin(u)
        return out

    @problem.register.objective
    def objective(arg):
        """Return the time taken, which is the objective."""
        return arg[ph].final.time

    ph.time.initial = (0.0, 0.0)
    ph.state.initial.x = (0.0, 0.0)
    ph.state.initial.y = (0.0, 0.0)
    ph.state.initial.v = (0.0, 0.0)
    ph.state.final.x = (1.0, 1.0)
    ph.state.bounds.x = (0, 10)
    ph.state.bounds.y = (0, 10)
    ph.state.bounds.v = (0, 10)
    ph.control.bounds.u = (-pi / 2, pi / 2)

    ph.time.guess = (0.0, 1.0)
    ph.state.guess.x = (0, 1)
    ph.state.guess.y = (0, 1)
    ph.state.guess.v = (0, 5)

    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the path the bead takes.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.slide]
    plt.figure()
    plt.plot(ps.state.x, ps.state.y, linewidth=2)
    plt.xlabel("Horizontal position, $x(t)$")
    plt.ylabel("Vertical position, $y(t)$")
    plt.grid()
    plt.xlim((0.0, 1.0))
    plt.ylim((0.8, -0.1))
    plt.axis("scaled")
    plt.tight_layout()


def main() -> None:
    """Solve the brachistochrone problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"\nObjective = {solution.objective}")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
