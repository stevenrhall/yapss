"""

Minimal YAPSS solution of the brachistochrone optimal control problem.

"""

__all__ = ["main", "plot_solution", "setup"]

# third party imports
import matplotlib.pyplot as plt

from yapss import ContinuousArg, ObjectiveArg, Problem, Solution

# package imports
from yapss.math import cos, sin


def setup() -> Problem:
    """Set up the brachistochrone optimal control problem.

    Returns
    -------
    Problem
        The brachistochrone optimal control problem.
    """
    # Initialize optimal control problem
    problem = Problem(name="Brachistochrone", nx=[3], nu=[1])

    g0 = 32.174

    def objective(arg: ObjectiveArg) -> None:
        """Objective callback function. Objective is to minimize final time."""
        arg.objective = arg.phase[0].final_time

    # continuous function
    def continuous(arg: ContinuousArg) -> None:
        """Continuous callback function."""
        x, y, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics = v * cos(u), v * sin(u), g0 * sin(u)

    problem.functions.objective = objective
    problem.functions.continuous = continuous

    # bounds
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.initial_state.lower[:] = bounds.initial_state.upper[:] = 0.0
    bounds.final_state.lower[0] = bounds.final_state.upper[0] = 1.0
    bounds.state.lower[:] = 0.0
    bounds.state.upper[:] = 10.0

    # guess
    phase = problem.guess.phase[0]
    phase.time = [0.0, 1.0]
    phase.state = [[0.0, 1.0], [0.0, 1.0], [0.0, 10.0]]
    phase.control = [[0.0, 0.0]]
    problem.derivatives.method = "auto"

    # ipopt options
    problem.ipopt_options.print_level = 3

    return problem


def plot_solution(solution: Solution) -> None:
    """Plot the solution to the brachistochrone optimal control problem.

    Parameters
    ----------
    solution : Solution
        The solution to the brachistochrone optimal control problem.
    """
    # extract solution
    x, y, _ = solution.phase[0].state

    # plot
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Horizontal position, $x(t)$")
    plt.ylabel("Vertical position, $y(t)$")
    plt.axis("equal")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.8, -0.1])
    plt.tight_layout()
    plt.grid()


def main() -> None:
    """Demonstrate the solution to the brachistochrone optimal control problem."""
    ocp = setup()
    solution = ocp.solve()
    print(f"\nObjective = {solution.objective}")
    plot_solution(solution)
    plt.show()


if __name__ == "__main__":
    main()
