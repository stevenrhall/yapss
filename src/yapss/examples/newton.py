"""

YAPSS solution of Newton's minimal resistance problem.

"""

__all__ = ["main", "plot_solution", "setup", "setup2"]

# third party imports
import matplotlib.pyplot as plt

# package imports
import numpy as np

from yapss import (
    ContinuousArg,
    ContinuousHessianArg,
    ContinuousJacobianArg,
    ObjectiveArg,
    ObjectiveGradientArg,
    ObjectiveHessianArg,
    Problem,
    Solution,
)


def objective(arg: ObjectiveArg) -> None:
    """Objective function for Newton's minimal resistance problem."""
    arg.objective = arg.phase[0].integral[0]


def continuous(arg: ContinuousArg) -> None:
    """Newton's minimal resistance problem dynamics and cost integrand."""
    _, yp = arg.phase[0].state
    (u,) = arg.phase[0].control
    r = arg.phase[0].time
    arg.phase[0].dynamics[:] = yp, u
    arg.phase[0].integrand[:] = (8 * r / (1 + yp**2),)


# Gradient, Jacobian, and Hessian functions are not required except when
# derivatives.method = "user" option is selected.


def objective_gradient(arg: ObjectiveGradientArg) -> None:
    """Calculate gradient of the objective function for Newton's minimal resistance problem.

    Needed only if the ``derivatives.method = "user"`` option is selected.
    """
    arg.gradient[0, "q", 0] = 1


def objective_hessian(_: ObjectiveHessianArg) -> None:
    """Calculate Hessian of the objective function for Newton's minimal resistance problem.

    Needed only if the ``derivatives.method = "user"`` option is selected.
    """


def continuous_jacobian(arg: ContinuousJacobianArg) -> None:
    """Calculate Jacobian of the dynamics and cost integrand for minimal resistance problem.

    Needed only if the ``derivatives.method = "user"`` option is selected.
    """
    _, yp = arg.phase[0].state
    r = arg.phase[0].time
    jacobian = arg.phase[0].jacobian
    jacobian[("f", 0), ("x", 1)] = 1
    jacobian[("f", 1), ("u", 0)] = 1
    jacobian[("g", 0), ("t", 0)] = 8 / (1 + yp**2)
    jacobian[("g", 0), ("x", 1)] = -16 * yp * r / (1 + yp**2) ** 2


def continuous_hessian(arg: ContinuousHessianArg) -> None:
    """Calculate Hessian of the dynamics and cost integrand for Newton's minimal resistance problem.

    Needed only if the ``derivatives.method = "user"`` option is selected.
    """
    _, yp = arg.phase[0].state
    r = arg.phase[0].time
    hessian = arg.phase[0].hessian
    hessian[("g", 0), ("x", 1), ("x", 1)] = 16 * r * (3 * yp**2 - 1) / (1 + yp**2) ** 3
    hessian[("g", 0), ("x", 1), ("t", 0)] = -16 * yp / (1 + yp**2) ** 2


# end derivative functions


def setup(y_max: float = 1.0) -> Problem:
    """Set up Newton's minimal resistance problem as an optimal control problem.

    Parameters
    ----------
    y_max : float, optional
        Maximum height of the nosecone, by default 1.0.

    Returns
    -------
    Problem
        Newton's minimal resistance problem as an optimal control problem.
    """
    ocp = Problem(
        name="Newton's minimal resistance problem",
        nx=[2],
        nu=[1],
        nq=[1],
    )

    # functions
    ocp.functions.objective = objective
    ocp.functions.objective_gradient = objective_gradient
    ocp.functions.objective_hessian = objective_hessian
    ocp.functions.continuous = continuous
    ocp.functions.continuous_jacobian = continuous_jacobian
    ocp.functions.continuous_hessian = continuous_hessian

    # bounds
    bounds = ocp.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.state.lower[0] = 0
    bounds.state.upper = y_max, 0
    bounds.control.upper = (0,)

    # guess
    phase = ocp.guess.phase[0]
    phase.time = [0.0, 1.0]
    phase.state = [[y_max, 0.0], [-y_max, -y_max]]
    phase.control = [[0.0, 0.0]]

    # solver settings
    ocp.derivatives.order = "second"

    # we use the "user" option to demonstrate how to provide derivatives
    ocp.derivatives.method = "user"

    # ipopt options
    ocp.ipopt_options.print_level = 3

    return ocp


def objective2(arg: ObjectiveArg) -> None:
    """Improved objective function for Newton's minimal resistance problem."""
    arg.objective = arg.phase[0].integral[0] + 4 * arg.phase[0].initial_time ** 2


def objective_gradient2(arg: ObjectiveGradientArg) -> None:
    """Calculate gradient of improved objective function for Newton's minimal resistance problem.

    Needed only if the derivatives.method = "user" option is selected.
    """
    arg.gradient[0, "q", 0] = 1
    arg.gradient[0, "t0", 0] = 8 * arg.phase[0].initial_time


def objective_hessian2(arg: ObjectiveHessianArg) -> None:
    """Calculate Hessian of improved objective function for Newton's minimal resistance problem.

    Needed only if the derivatives.method = "user" option is selected.
    """
    arg.hessian[(0, "t0", 0), (0, "t0", 0)] = 8


def setup2(y_max: float = 1.0) -> Problem:
    """Set up alternate formulation of Newton's minimal resistance problem.

    In this alternate formulation, the radius of the flat portion of the nosecone is a
    parameter to be optimized, and the curve to be optimized starts at this radius. This
    formulation improves the solution considerably.

    Parameters
    ----------
    y_max : float, optional
        Maximum height of the nosecone, by default 1.0.

    Returns
    -------
    Problem
        Newton's minimal resistance problem as an optimal control problem.
    """
    ocp = setup(y_max)
    ocp.functions.objective = objective2
    ocp.functions.objective_gradient = objective_gradient2
    ocp.functions.objective_hessian = objective_hessian2
    ocp.bounds.phase[0].initial_time.upper = 0.7
    return ocp


def plot_solution(solution: Solution) -> None:
    """Plot the solution to Newton's minimal resistance problem.

    Parameters
    ----------
    solution : Solution
        The solution to the Newton's minimal resistance problem.
    """
    # plot style information
    linewidth = 2
    plt.rc("font", size=14)
    plt.rc("font", family="sans-serif")

    # extract information from solution
    r = solution.phase[0].time
    y, _ = solution.phase[0].state
    r = np.concatenate((-r[-1::-1], r))
    y = np.concatenate((y[-1::-1], y))

    # plot
    plt.plot(r, y, "r", linewidth=linewidth)
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-0.1, 2.1])
    plt.xlabel("Radius, $r/R$")
    plt.ylabel("Height, $y/R$")
    plt.tight_layout()


def _truncate_float(number: float, decimals: int = 5) -> float:
    """Truncate a float to a specified number of decimal places."""
    factor = 10**decimals
    return float(int(number * factor) / factor)


def main() -> None:
    """Demonstrate the solution to Newton's minimal resistance problem."""
    # solve problem using the first formulation
    problem = setup(y_max=1)
    solution1a = problem.solve()
    plt.figure(1, figsize=(6.4, 4))
    plot_solution(solution1a)
    plt.axis((-1.2, 1.2, -0.1, 1.1))
    plt.axis("equal")
    plt.title("Solution with $y_{max} = 1$, first formulation")
    plt.xticks(np.linspace(-1, 1, 5))
    plt.tight_layout()

    # solve problem using the second formulation
    problem = setup2(y_max=1)
    solution1b = problem.solve()
    plt.figure(2, figsize=(6.4, 4))
    plot_solution(solution1b)
    plt.axis((-1.2, 1.2, -0.1, 1.1))
    plt.axis("equal")
    plt.title("Solution with $y_{max} = 1$, second formulation")
    plt.xticks(np.linspace(-1, 1, 5))
    plt.tight_layout()

    # solve problem for different values of y_max
    plt.figure(3)
    objective_list: list[float] = []
    y_max_list = [0.5, 1, 2]
    for y_max in y_max_list:
        problem = setup2(y_max=y_max)
        solution = problem.solve()
        objective_list.append(solution.objective)
        plot_solution(solution)
    plt.title("Solution for different values of $y_{max}$")
    plt.tight_layout()

    # print objective values for different values of y_max, in a table
    print("\nObjective values for different values of y_max:\n")
    print("y_max | Objective (C_D)")
    print("------+----------------")
    for y_max, objective_value in zip(y_max_list, objective_list):
        print(f"{y_max:5.2f} |  {_truncate_float(objective_value):8.5f}...")

    plt.show()


if __name__ == "__main__":
    main()
