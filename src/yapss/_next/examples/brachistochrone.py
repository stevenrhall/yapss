"""

The brachistochrone problem: the shape of the fastest slide between two points.

A bead slides without friction from the origin to x = 1 under gravity. The control is the
slope angle of the path, and the objective is the time taken.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt
from numpy import pi

from yapss import _next as yapss
from yapss.math import cos, sin

g0 = 32.174


class Slide(yapss.Vector):
    """Where the bead is and how fast it is going."""

    x = yapss.field(units="ft", latex="x", doc="horizontal position")
    y = yapss.field(units="ft", latex="y", doc="vertical drop")
    v = yapss.field(units="ft/s", latex="v", doc="speed")


class Angle(yapss.Vector):
    """The slope of the path."""

    u = yapss.field(units="rad", latex=r"\theta", doc="path angle")


class Phases(yapss.Phases):
    """One phase: the bead slides."""

    slide = yapss.phase(state=Slide, control=Angle)


def setup() -> yapss.Problem:
    """Set up the brachistochrone problem.

    Returns
    -------
    yapss._next.Problem
        The problem.
    """
    problem = yapss.Problem("Brachistochrone", phases=Phases)
    ph = problem.phases.slide

    @ph.register.continuous
    def slide(arg, out):
        """Compute the bead's dynamics."""
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = g0 * sin(u)
        return out

    @problem.register.objective
    def minimum_time(arg):
        """Return the time taken, which is the objective."""
        return arg[ph].final.time

    ph.time.initial = 0.0
    ph.state.initial.x = 0.0
    ph.state.initial.y = 0.0
    ph.state.initial.v = 0.0
    ph.state.final.x = 1.0
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
    """Plot the path the bead takes and the angle along it.

    Parameters
    ----------
    problem : yapss._next.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._next.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.slide]
    plt.figure()
    plt.plot(ps.state.x, -ps.state.y)
    plt.xlabel("x (ft)")
    plt.ylabel("-y (ft)")
    plt.title("Brachistochrone")

    plt.figure()
    plt.plot(ps.time, ps.control.u)
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\theta$ (rad)")


def main() -> None:
    """Solve the brachistochrone problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final time = {solution.objective:.6f} s")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
