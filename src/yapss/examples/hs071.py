"""

YAPSS solution of the HS071 constrained function minimization problem.

"""

__all__ = ["main", "print_solution", "setup"]

import numpy as np

# third party imports
from numpy.typing import NDArray

# package imports
from yapss import DiscreteArg, ObjectiveArg, Problem, Solution


def setup() -> Problem:
    """Set up the problem statement for Hock and Schittkowski Problem 71 (HS071)."""
    # parameter optimization problem with 4 parameters and 2 constraints
    ocp = Problem(name="HS071", nx=[], ns=4, nd=2)

    def objective(arg: ObjectiveArg) -> None:
        """HS071 objective callback function."""
        x = arg.parameter
        arg.objective = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def discrete(arg: DiscreteArg) -> None:
        """HS071 discrete constraint callback function."""
        x = arg.parameter
        arg.discrete[:] = (
            x[0] * x[1] * x[2] * x[3],
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3],
        )

    ocp.functions.objective = objective
    ocp.functions.discrete = discrete

    # bounds
    ocp.bounds.discrete.lower = 25.0, 40.0
    ocp.bounds.discrete.upper[1] = 40.0
    ocp.bounds.parameter.lower = 4 * [1.0]
    ocp.bounds.parameter.upper = 4 * [5.0]

    # guess
    ocp.guess.parameter = 1.0, 5.0, 5.0, 1.0

    # yapss options
    ocp.derivatives.order = "second"
    ocp.derivatives.method = "auto"
    ocp.ipopt_options.print_level = 5

    return ocp


def print_solution(solution: Solution) -> None:
    """Print the solution of the HS071 constrained function minimization problem.

    Parameters
    ----------
    solution : Solution
        The solution of the HS071 function minimization problem.
    """

    def print_variable(name: str, values: NDArray[np.float64]) -> None:
        for i, _value in enumerate(values):
            print(f"{name}[{i}] = {_value:1.6e}")

    x = solution.parameter
    print()
    print("Solution of the primal variables, x")
    print_variable("x", x)
    print("\nSolution of the bound multipliers, z_L and z_U")
    nlp_info = solution.nlp_info
    print_variable("z_L", nlp_info.mult_x_L)
    print_variable("z_U", nlp_info.mult_x_U)
    print("\nSolution of the constraint multipliers, lambda")
    print_variable("lambda", solution.discrete_multiplier)
    print("\nObjective value")
    print(f"f(x*) = {solution.objective:1.6e}")


def main() -> None:
    """Demonstrate the solution to the HS071 constrained function minimization problem."""
    problem = setup()
    solution = problem.solve()
    print_solution(solution)


if __name__ == "__main__":
    main()
