"""

Minimizing the Rosenbrock function, which is not an optimal control problem at all.

The Rosenbrock function has a narrow curved valley with its minimum at (1, 1), and is the
standard test of whether an optimizer can follow such a valley. There is no trajectory here, so
there are no phases: the problem is two parameters and an objective.

It is in the corpus as the smallest thing YAPSS will solve, and as the shortest illustration
that a problem need not have a phase to be a problem.

"""

__all__ = ["main", "plot_rosenbrock", "setup"]

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import yapss

MINIMUM = (1.0, 1.0)
"""Where the function is smallest, which the solver should find."""


class Parameter(yapss.Vector):
    """The point to be chosen."""

    x = yapss.field()
    """Horizontal coordinate."""
    y = yapss.field()
    """Vertical coordinate."""


class Phases(yapss.Phases):
    """None. Nothing here evolves in time, so there is no phase."""


def rosenbrock(x: Any, y: Any) -> Any:
    """Return the Rosenbrock function at `(x, y)`.

    Parameters
    ----------
    x, y : array_like
        Where to evaluate the function.

    Returns
    -------
    array_like
        The value of the function there.
    """
    return 100 * (y - x**2) ** 2 + (1 - x) ** 2


def setup() -> yapss.Problem:
    """Set up the Rosenbrock minimization problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Rosenbrock", phases=Phases, parameter=Parameter)

    @problem.register.objective
    def objective(arg):
        """Return the Rosenbrock function at the chosen point."""
        return rosenbrock(arg.parameter.x, arg.parameter.y)

    problem.parameter.guess.x = -1.2
    problem.parameter.guess.y = 1.0

    problem.ipopt_options.print_level = 5
    problem.ipopt_options.tol = 1e-10
    return problem


def plot_rosenbrock(solution: yapss.Solution | None = None) -> None:
    """Contour the function, marking its minimum and, if given, the point found.

    Parameters
    ----------
    solution : yapss.Solution, optional
        The solution to mark.
    """
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    f = rosenbrock(*np.meshgrid(x, y))

    plt.figure()
    contours = plt.contour(
        x, y, f, [1, 3, 10, 30, 100, 300, 1000, 3000], colors="black", linewidths=0.5
    )
    plt.clabel(contours, inline=True, fontsize=8)
    plt.plot(*MINIMUM, ".r", markersize=10)
    if solution is not None:
        plt.plot(solution.parameter.x, solution.parameter.y, "+b", markersize=12)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Rosenbrock function")
    plt.xticks(range(-2, 3))
    plt.yticks(range(-1, 4))
    plt.tight_layout()


def main() -> None:
    """Minimize the Rosenbrock function and plot where the minimum was found."""
    problem = setup()
    solution = problem.solve()
    print(f"\nminimum at x = {solution.parameter.x:.9f}, y = {solution.parameter.y:.9f}")
    print(f"f(x, y) = {solution.objective:.3e}")
    plot_rosenbrock(solution)
    plt.show()


if __name__ == "__main__":
    main()
