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
    # integrals. There are 2 discrete constraints, to constrain the curve to be closed;
    # the centroid is placed at the origin by bounds on the second and third integrals.

    ocp = Problem(name="Isoperimetric Problem", nx=[2], nu=[2], nq=[3], nh=[1], nd=2)

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

    ocp.functions.objective = objective
    ocp.sense = "maximize"
    ocp.functions.continuous = continuous
    ocp.functions.discrete = discrete

    # bounds
    bounds = ocp.bounds.phase[0]
    bounds.path.lower[0] = bounds.path.upper[0] = 1
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    ocp.bounds.discrete.lower = ocp.bounds.discrete.upper = [0, 0]

    # centroid of the curve at the origin
    bounds.integral.lower[1:] = 0
    bounds.integral.upper[1:] = 0

    # guess
    guess = ocp.guess.phase[0]
    # A square of perimeter 1, matching the path constraint. (The diamond used
    # previously had perimeter 4*sqrt(2), so the guess violated the constraint it was
    # meant to start from.)
    guess.time = [0.0, 0.25, 0.5, 0.75, 1.0]
    guess.state = np.array([[1.0, 1.0, -1.0, -1.0, 1.0], [1.0, -1.0, -1.0, 1.0, 1.0]]) / 8

    # mesh
    m, n = 3, 12
    ocp.mesh.phase[0].collocation_points = m * (n,)
    ocp.mesh.phase[0].fraction = m * (1.0 / m,)

    # yapss and ipopt options
    ocp.derivatives.method = "auto"
    ocp.derivatives.order = "second"
    # A tighter tolerance than one would normally use, to show how accurate the
    # pseudospectral method is here with a modest number of collocation points. The
    # relative error against the closed-form answer, printed by `main()`, is the real
    # check -- it holds whichever way Ipopt happens to terminate.
    ocp.ipopt_options.tol = 1e-14
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
    t = solution.phase[0].time
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

    # plot the Hamiltonian
    plt.figure(2)
    plt.clf()
    hamiltonian = solution.phase[0].hamiltonian
    plt.plot(t, hamiltonian)


def main() -> None:
    """Demonstrate the solution to the isoperimetric problem."""
    problem = setup()
    solution = problem.solve()

    # print the solution
    area = solution.objective
    area_ideal = 1 / (4 * math.pi)
    print(f"\n\nMaximum area = {area} (Should be 1 / (4 pi) = {area_ideal})")
    print(f"Relative error in solution = {abs(area - area_ideal) / area_ideal}")

    print(solution.discrete_multiplier)

    # plot the solution
    plot_solution(solution)
    plt.show()


if __name__ == "__main__":
    main()
