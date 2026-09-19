"""

Hock and Schittkowski problem 71, which has no phases at all.

A problem with no phases is an ordinary nonlinear program: parameters to choose, constraints
relating them, an objective to minimize. Nothing about it is a special case -- the objective
and discrete callbacks are the ones every problem has, and there are simply no phases for
`arg[ph]` to reach.

It is here because it is the shortest complete statement of that, and because a library that
refused it would be refusing arithmetic it can already do.

"""

__all__ = ["main", "print_solution", "setup"]

import yapss

OPTIMUM = 17.01401714
"""The known objective value, used as the installation smoke test."""


class Design(yapss.Vector):
    """The four variables to be chosen."""

    x = yapss.field(size=4, doc="design variables")


class Constraints(yapss.Vector):
    """The two constraints relating them."""

    product = yapss.field(doc="the product of all four, at least 25")
    sum_of_squares = yapss.field(doc="the sum of their squares, exactly 40")


class Phases(yapss.Phases):
    """None. The problem has no trajectory, so it has no phases."""


def setup() -> yapss.Problem:
    """Set up the HS071 problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("HS071", phases=Phases, parameter=Design, discrete=Constraints)

    @problem.register.objective
    def cost(arg):
        """Return the objective."""
        x = arg.parameter.x
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    @problem.register.discrete
    def constraints(arg, out):
        """Compute the two constraints."""
        x = arg.parameter.x
        out.discrete.product = x[0] * x[1] * x[2] * x[3]
        out.discrete.sum_of_squares = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        return out

    problem.parameter.bounds.x[:] = (1.0, 5.0)
    problem.parameter.guess.x[:] = [1.0, 5.0, 5.0, 1.0]
    problem.discrete.bounds.product = (25.0, None)
    problem.discrete.bounds.sum_of_squares = (40.0, 40.0)

    problem.ipopt_options.print_level = 3
    return problem


def print_solution(solution: yapss.Solution) -> None:
    """Print the design variables, the constraints and the objective.

    Parameters
    ----------
    solution : yapss.Solution
        The solution to print.
    """
    for i, value in enumerate(solution.parameter.x):
        print(f"x[{i}] = {value:1.6e}")
    print(f"\nproduct         = {solution.discrete.product:1.6e}")
    print(f"sum of squares  = {solution.discrete.sum_of_squares:1.6e}")
    print(f"\nf(x*) = {solution.objective:1.6e}")


def main() -> None:
    """Solve HS071 and print the solution."""
    problem = setup()
    solution = problem.solve()
    print_solution(solution)

    if not solution.converged:
        msg = "YAPSS did not converge to an optimal solution."
        raise RuntimeError(msg)
    if abs(solution.objective - OPTIMUM) > 1e-6 * OPTIMUM:
        msg = "YAPSS returned an unexpected objective value."
        raise RuntimeError(msg)
    print("\nYAPSS solution is correct.")


if __name__ == "__main__":
    main()
