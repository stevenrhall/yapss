"""

Newton's minimal resistance problem: the nosecone shape of least pressure drag.

The phase runs over the radius, not over time, so the phase names its independent variable
``r`` and that is what it is called everywhere afterwards -- in setup, in the callback, and in
the solution. The released version of this example calls it ``time`` and apologises in a
comment.

There are two formulations, and `setup2` is the one to prefer. In Newton's model a gas
particle strikes the surface once and leaves, so drag falls as the local slope steepens --
which means an unconstrained shape would be driven to a sawtooth of arbitrarily steep facets
and no drag at all. That is outside the model rather than a real answer, since such a shape
would have the particles colliding again, so the profile is required to be convex. The true
optimum is therefore flat out to some radius and sloping beyond it, with a corner where the
two meet.

`setup` fixes the flat part at zero radius and asks one phase to represent that corner with a
polynomial. It does not merely tolerate the resulting oscillation, it is rewarded for it: the
wiggles steepen the local slope, which lowers the integrand. Convexity is imposed at the
collocation points, and between them the polynomial is free -- so the solve satisfies
``u <= 0`` at every node while the slope it returns still rises across six of its ninety
intervals, which no convex profile can do. `setup2` optimizes the radius of the flat tip
instead, removing the corner; its slope is monotone throughout and it reaches a lower drag.

"""

__all__ = ["main", "plot_solution", "setup", "setup2"]

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import yapss


class State(yapss.State):
    """The shape of the nosecone."""

    y = yapss.scalar()
    """Height of the profile."""
    yp = yapss.scalar()
    """Slope of the profile."""


class Control(yapss.Control):
    """How the slope is allowed to change."""

    u = yapss.scalar()
    """Second derivative of the profile."""


class Integral(yapss.Integral):
    """What is being minimized."""

    drag = yapss.scalar()
    """Pressure drag on the nosecone."""


class Nose(yapss.Phase):
    """The nosecone's profile, run over the radius ``r`` rather than time."""

    state: State
    control: Control
    integral: Integral
    r: yapss.Independent


class Phases(yapss.Phases):
    """One phase, run over the radius rather than over time."""

    nose: Nose


def setup(y_max: float = 1.0) -> yapss.Problem:
    """Set up Newton's minimal resistance problem.

    Parameters
    ----------
    y_max : float, default 1.0
        The greatest height of the nosecone.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Newton's minimal resistance problem", phases=Phases)
    ph = problem.phases.nose

    @ph.register.continuous
    def continuous(arg, out):
        """Compute the profile's dynamics and the drag integrand."""
        yp, u, r = arg.state.yp, arg.control.u, arg.r
        out.dynamics.y = yp
        out.dynamics.yp = u
        out.integrand.drag = 8 * r / (1 + yp**2)
        return out

    @problem.register.objective
    def objective(arg):
        """Return the drag, which is the objective."""
        return arg[ph].integral.drag

    ph.r.initial = (0.0, 0.0)
    ph.r.final = (1.0, 1.0)
    ph.state.y.bounds = (0.0, y_max)
    ph.state.yp.bounds = (None, 0.0)
    ph.control.u.bounds = (None, 0.0)

    ph.r.guess = (0.0, 1.0)
    ph.state.y.guess = (y_max, 0.0)
    ph.state.yp.guess = (-y_max, -y_max)
    ph.control.u.guess = (0.0, 0.0)

    problem.derivatives.order = "second"
    problem.ipopt_options.print_level = 3
    return problem


def setup2(y_max: float = 1.0) -> yapss.Problem:
    """Set up the alternate formulation of Newton's minimal resistance problem.

    The radius of the flat tip becomes a variable: the phase starts at a free ``r`` and the
    disc's own drag, ``4 r0**2``, is added to the objective. With the corner gone, the curve
    the phase has to represent is smooth, and the solution comes out with the flat tip at
    r0 = 0.351 and a slope of -0.9998 where it meets the curve, against the classical value of
    exactly -1. See the module docstring for why the first formulation goes wrong.

    Parameters
    ----------
    y_max : float, default 1.0
        The greatest height of the nosecone.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = setup(y_max)
    ph = problem.phases.nose

    @problem.register.objective(replace=True)
    def objective(arg):
        """Return the drag of the curve plus the drag of the flat tip."""
        return arg[ph].integral.drag + 4 * arg[ph].initial.r**2

    ph.r.initial = (0.0, 1.0)
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution, **kwargs: Any) -> None:
    """Plot one nosecone profile, reflected about its axis.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    **kwargs
        Passed to `matplotlib.pyplot.plot`, for a label or a style.
    """
    ps = solution[problem.phases.nose]
    r = np.concatenate((-ps.r[::-1], ps.r))
    y = np.concatenate((ps.state.y[::-1], ps.state.y))
    plt.plot(r, y, linewidth=2, **kwargs)
    plt.xlabel("Radius, $r/R$")
    plt.ylabel("Height, $y/R$")
    plt.axis("equal")
    plt.grid()


def main() -> None:
    """Solve both formulations, and the second one at three aspect ratios.

    Three figures, in the order the documentation presents them: the shape the first
    formulation returns, the shape the second one returns, and the second one at three
    heights. The first two are the same problem, and comparing them is the point -- see the
    module docstring for why the first goes wrong.
    """
    problem = setup()
    solution = problem.solve()
    print(f"drag, one polynomial through the corner = {solution.objective:.6f}")
    plt.figure()
    plot_solution(problem, solution, color="r")
    plt.title("First formulation")

    problem2 = setup2()
    solution2 = problem2.solve()
    r0 = solution2[problem2.phases.nose].initial.r
    print(f"drag, with the flat tip free           = {solution2.objective:.6f}")
    print(f"radius of the flat tip, r0 = {r0:.6f}")
    plt.figure()
    plot_solution(problem2, solution2, color="r")
    plt.title("Second formulation")

    plt.figure()
    for y_max in (0.5, 1.0, 2.0):
        aspect = setup2(y_max)
        aspect.ipopt_options.print_level = 0
        aspect_solution = aspect.solve()
        print(f"drag at y_max/R = {y_max}: {aspect_solution.objective:.6f}")
        plot_solution(aspect, aspect_solution, label=f"$y_{{max}}/R = {y_max}$")
    plt.legend()
    plt.title("Optimal nosecones for three aspect ratios")

    plt.show()


if __name__ == "__main__":
    main()
