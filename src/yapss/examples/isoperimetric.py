"""

YAPSS solution of the isoperimetric problem.

"""

__all__ = ["main", "plot_solution", "setup"]

# standard library imports
import math

# package imports
import numpy as np

# third party imports
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from yapss import ContinuousArg, DiscreteArg, ObjectiveArg, Problem, Solution


def setup() -> Problem:
    """Set up the isoperimetric optimization problem.

    Returns
    -------
    Problem
        The isoperimetric optimization problem.
    """
    # problem has 1 phase, with 2 states, 2 controls, 1 path constraint, and 3
    # integrals. There are four constraints: 2 to constrain the curved to be closed,
    # and 2 to constrain the centroid of the curve to be at the origin.

    ocp = Problem(name="Isoperimetric Problem", nx=[2], nu=[2], nq=[3], nh=[1], nd=4)

    def objective(arg: ObjectiveArg) -> None:
        """Objective callback function."""
        arg.objective = arg.phase[0].integral[0]

    def continuous(arg: ContinuousArg) -> None:
        """Continuous callback function."""
        x, y = arg.phase[0].state
        ux, uy = arg.phase[0].control
        arg.phase[0].dynamics[:] = ux, uy
        arg.phase[0].path[0] = ux**2 + uy**2
        arg.phase[0].integrand[0] = (y * ux - x * uy) / 2
        arg.phase[0].integrand[1] = x
        arg.phase[0].integrand[2] = y

    def discrete(arg: DiscreteArg) -> None:
        """Discrete callback function."""
        arg.discrete[:2] = arg.phase[0].final_state - arg.phase[0].initial_state
        arg.discrete[2:4] = arg.phase[0].integral[1:]

    ocp.functions.objective = objective
    ocp.functions.continuous = continuous
    ocp.functions.discrete = discrete

    # bounds
    bounds = ocp.bounds.phase[0]
    bounds.path.lower[0] = bounds.path.upper[0] = 1
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.state.lower[:] = -10
    bounds.state.upper[:] = +10
    bounds.control.lower[:] = -2
    bounds.control.upper[:] = +2
    ocp.bounds.discrete.lower = ocp.bounds.discrete.upper = [0, 0, 0, 0]
    ocp.bounds.discrete.lower[2:] = -1e-4
    ocp.bounds.discrete.upper[2:] = +1e-4

    # guess
    guess = ocp.guess.phase[0]
    guess.time = [0.0, 0.25, 0.5, 0.75, 1.0]
    guess.state = [
        [1.0, 0.0, -1.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 1.0, 0.0],
    ]

    # mesh
    m, n = 3, 15
    ocp.mesh.phase[0].collocation_points = m * (n,)
    ocp.mesh.phase[0].fraction = m * (1.0 / m,)

    # set objective scale to maximize objective
    ocp.scale.objective = -1

    # yapss and ipopt options
    ocp.derivatives.method = "auto"
    ocp.derivatives.order = "second"
    ocp.ipopt_options.tol = 1e-20
    ocp.ipopt_options.print_level = 3

    return ocp


def plot_solution(solution: Solution) -> None:
    """Plot the solution to the isoperimetric problem.

    The collocation points are shown as dots, and the curve is interpolated between
    points with a cubic spline.

    Parameters
    ----------
    solution: Solution
        The solution to the isoperimetric problem.
    """
    plt.figure(1)
    plt.clf()
    x, y = solution.phase[0].state
    s = solution.phase[0].time
    sp = np.linspace(0, 1, 500)
    xp = interp1d(s, x, kind="cubic")(sp)
    yp = interp1d(s, y, kind="cubic")(sp)
    plt.plot(xp, yp)
    plt.plot(x, y, ".", markersize=10)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis("square")
    plt.tight_layout()


def main() -> None:
    """Demonstrate the solution to the isoperimetric problem."""
    problem = setup()
    solution = problem.solve()

    # print the solution
    area = solution.objective
    area_ideal = 1 / (4 * math.pi)
    print(f"\n\nMaximum area = {area} (Should be 1 / (4 pi) = {area_ideal})")
    print(f"Relative error in solution = {abs(area - area_ideal) / area_ideal}")

    # plot the solution
    plot_solution(solution)
    plt.show()


if __name__ == "__main__":
    main()
