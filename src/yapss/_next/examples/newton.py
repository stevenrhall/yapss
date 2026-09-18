"""

Newton's minimal resistance problem: the nosecone shape of least pressure drag.

The phase runs over the radius, not over time, so the phase names its independent variable
``r`` and that is what it is called everywhere afterwards -- in setup, in the callback, and in
the solution. The released version of this example calls it ``time`` and apologises in a
comment.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt
import numpy as np

from yapss import _next as yapss


class Profile(yapss.Vector):
    """The shape of the nosecone."""

    y = yapss.field(latex="y", doc="height of the profile")
    yp = yapss.field(latex="y'", doc="slope of the profile")


class Curvature(yapss.Vector):
    """How the slope is allowed to change."""

    u = yapss.field(latex="u", doc="second derivative of the profile")


class Resistance(yapss.Vector):
    """What is being minimized."""

    drag = yapss.field(latex="D", doc="pressure drag on the nosecone")


class Phases(yapss.Phases):
    """One phase, run over the radius rather than over time."""

    nose = yapss.phase(
        state=Profile,
        control=Curvature,
        integral=Resistance,
        r=yapss.field(latex="r", doc="radius"),
    )


def setup(y_max: float = 1.0) -> yapss.Problem:
    """Set up Newton's minimal resistance problem.

    Parameters
    ----------
    y_max : float, default 1.0
        The greatest height of the nosecone.

    Returns
    -------
    yapss._next.Problem
        The problem.
    """
    problem = yapss.Problem("Newton's minimal resistance problem", phases=Phases)
    ph = problem.phases.nose

    @ph.register.continuous
    def nose(arg, out):
        """Compute the profile's dynamics and the drag integrand."""
        yp, u, r = arg.state.yp, arg.control.u, arg.r
        out.dynamics.y = yp
        out.dynamics.yp = u
        out.integrand.drag = 8 * r / (1 + yp**2)
        return out

    @problem.register.objective
    def least_drag(arg):
        """Return the drag, which is the objective."""
        return arg[ph].integral.drag

    ph.r.initial = 0.0
    ph.r.final = 1.0
    ph.state.bounds.y = (0.0, y_max)
    ph.state.bounds.yp = (-np.inf, 0.0)
    ph.control.bounds.u = (-np.inf, 0.0)

    ph.r.guess = (0.0, 1.0)
    ph.state.guess.y = (y_max, 0.0)
    ph.state.guess.yp = (-y_max, -y_max)
    ph.control.guess.u = (0.0, 0.0)

    problem.derivatives.order = "second"
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the nosecone profile, reflected about its axis.

    Parameters
    ----------
    problem : yapss._next.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._next.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.nose]
    r = np.concatenate((-ps.r[::-1], ps.r))
    y = np.concatenate((ps.state.y[::-1], ps.state.y))
    plt.figure()
    plt.plot(r, y, "r", linewidth=2)
    plt.xlabel("Radius, $r/R$")
    plt.ylabel("Height, $y/R$")
    plt.title("Newton's minimal resistance nosecone")
    plt.axis("equal")


def main() -> None:
    """Solve Newton's minimal resistance problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
