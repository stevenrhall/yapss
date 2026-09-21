"""

Hock and Schittkowski problem 71, which has no phases at all.

A problem with no phases is an ordinary nonlinear program: parameters to choose, constraints
relating them, an objective to minimize. Nothing about it is a special case -- the objective
and discrete callbacks are the ones every problem has, and there are simply no phases for
`arg[ph]` to reach.

It is here because it is the shortest complete statement of that, and because a library that
refused it would be refusing arithmetic it can already do.

Its derivatives are written by hand rather than traced, because with no phases there is no
continuous callback and so they are four short functions. How to write them is explained
where it is taught, in `yapss.examples.brachistochrone_user_derivatives`; this is only the
smallest problem where the whole of that surface fits on one screen.

"""

__all__ = ["main", "print_solution", "setup"]

import yapss

OPTIMUM = 17.01401714
"""The known objective value, used as the installation smoke test."""


class Parameter(yapss.Parameter):
    """The four variables to be chosen."""

    x = yapss.vector(4)
    """Design variables."""


class Discrete(yapss.Discrete):
    """The two constraints relating them."""

    product = yapss.scalar()
    """The product of all four, at least 25."""
    sum_of_squares = yapss.scalar()
    """The sum of their squares, exactly 40."""


class Phases(yapss.Phases):
    """None. The problem has no trajectory, so it has no phases."""


def setup() -> yapss.Problem:
    """Set up the HS071 problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("HS071", phases=Phases, parameter=Parameter, discrete=Discrete)

    @problem.register.objective
    def objective(arg):
        """Return the objective."""
        x = arg.parameter.x
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    @problem.register.discrete
    def discrete(arg, out):
        """Compute the two constraints."""
        x = arg.parameter.x
        out.discrete.product = x[0] * x[1] * x[2] * x[3]
        out.discrete.sum_of_squares = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        return out

    @problem.register.objective_gradient
    def objective_gradient(arg, gradient):
        """Compute the gradient of the objective."""
        x, dx = arg.parameter.x, gradient.parameter.x
        gradient[dx[0]] = x[3] * (2 * x[0] + x[1] + x[2])
        gradient[dx[1]] = x[0] * x[3]
        gradient[dx[2]] = x[0] * x[3] + 1.0
        gradient[dx[3]] = x[0] * (x[0] + x[1] + x[2])
        return gradient

    @problem.register.objective_hessian
    def objective_hessian(arg, hessian):
        """Compute the second derivatives of the objective.

        Each unordered pair is written once: ``hessian[dx[0], dx[3]]`` and
        ``hessian[dx[3], dx[0]]`` name the same derivative, and writing both is refused
        rather than summed. Pairs left out are zero everywhere, which is most of them.
        """
        x, dx = arg.parameter.x, hessian.parameter.x
        hessian[dx[0], dx[0]] = 2 * x[3]
        hessian[dx[0], dx[1]] = x[3]
        hessian[dx[0], dx[2]] = x[3]
        hessian[dx[0], dx[3]] = 2 * x[0] + x[1] + x[2]
        hessian[dx[1], dx[3]] = x[0]
        hessian[dx[2], dx[3]] = x[0]
        return hessian

    @problem.register.discrete_jacobian
    def discrete_jacobian(arg, jacobian):
        """Compute the first derivatives of the two constraints."""
        x, dx = arg.parameter.x, jacobian.parameter.x
        for i in range(4):
            others = [x[j] for j in range(4) if j != i]
            jacobian.discrete.product[dx[i]] = others[0] * others[1] * others[2]
            jacobian.discrete.sum_of_squares[dx[i]] = 2 * x[i]
        return jacobian

    @problem.register.discrete_hessian
    def discrete_hessian(arg, hessian):
        """Compute the second derivatives of the two constraints.

        The product is bilinear in every pair, so each off-diagonal entry is the product of
        the two variables not named; its diagonal is zero. The sum of squares is the other
        way round: diagonal only, and constant.
        """
        x, dx = arg.parameter.x, hessian.parameter.x
        for i in range(4):
            hessian.discrete.sum_of_squares[dx[i], dx[i]] = 2.0
            for j in range(i + 1, 4):
                rest = [k for k in range(4) if k not in (i, j)]
                hessian.discrete.product[dx[i], dx[j]] = x[rest[0]] * x[rest[1]]
        return hessian

    problem.derivatives.method = "user"

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
