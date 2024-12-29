"""

YAPSS solution to the minimization of the Rosenbrock function.

"""

__all__ = ["main", "plot_rosenbrock", "setup"]

import numpy as np

# third party imports
from matplotlib import pyplot as plt

# package imports
from yapss import ObjectiveArg, Problem


def setup() -> Problem:
    """Set up the problem to minimize the Rosenbrock problem.

    Returns
    -------
    problem : Problem
        The Rosenbrock function minimization problem.
    """
    # problem has no dynamic phases, and 2 parameters
    problem = Problem(name="Rosenbrock", nx=[], ns=2)

    def objective(arg: ObjectiveArg) -> None:
        """Rosenbrock objective callback function."""
        x0, x1 = arg.parameter
        arg.objective = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2

    problem.functions.objective = objective

    # guess
    problem.guess.parameter = -1.2, 1.0

    # yapss options
    problem.derivatives.order = "second"
    problem.derivatives.method = "auto"

    # ipopt options
    problem.ipopt_options.tol = 1e-20
    problem.ipopt_options.print_level = 5

    return problem


def plot_rosenbrock() -> None:
    """Make a contour plot of the Rosenbrock function."""
    x0 = np.linspace(-2, 2, 400)
    x1 = np.linspace(-1, 3, 400)
    x0_grid, x1_grid = np.meshgrid(x0, x1)
    f = 100 * (x1_grid - x0_grid**2) ** 2 + (1 - x0_grid) ** 2

    plt.figure(1)
    levels = [1, 3, 10, 30, 100, 300, 1000, 3000]
    cp = plt.contour(x0, x1, f, levels, colors="black", linewidths=0.5)
    plt.clabel(cp, inline=1, fontsize=8)
    plt.plot(1, 1, ".r", markersize=10)
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")
    plt.title("Rosenbrock function")
    plt.xticks(range(-2, 3))
    plt.yticks(range(-1, 4))
    plt.tight_layout()


def main() -> None:
    """Demonstrate the solution to the Rosenbrock function minimization problem."""
    ocp = setup()
    ocp.derivatives.method = "auto"
    ocp.derivatives.order = "first"
    ocp.ipopt_options.print_level = 5
    solution = ocp.solve()
    x_opt = solution.parameter
    plt.figure()
    print(f"\nThe optimal solution is at the point x = ({x_opt[0]}, {x_opt[1]})")
    plot_rosenbrock()
    plt.show()


if __name__ == "__main__":
    main()
