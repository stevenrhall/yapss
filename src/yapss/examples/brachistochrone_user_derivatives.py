"""

The brachistochrone problem, with its derivatives supplied by hand.

The same problem as `yapss.examples.brachistochrone`, so the two can be read side by
side: everything but the four derivative callbacks is the same, and the answer is the same.

A derivative is reached by the names of the things it relates, read in the order it is
spoken. ``jacobian.dynamics.x.v`` is the derivative of the dynamics of ``x`` with respect to
``v``, and ``hessian.dynamics.x.v.u`` chains one more name for the second derivative. The
variable is named on its own, because a phase has one differentiation namespace and that is
what the Jacobian's columns are.

Only the entries that are not structurally zero are written. That is the one place in this
API where leaving a name unassigned means something, and it means "this derivative is zero
everywhere"; a derivative that is zero only at this point is written as ``0.0``.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt
from numpy import pi

import yapss
from yapss.math import cos, sin

from .brachistochrone import Control, Phases, State, g0, plot_solution

__all__ += ["Control", "Phases", "State"]


def setup() -> yapss.Problem:
    """Set up the brachistochrone problem with derivatives supplied by hand.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Brachistochrone (user derivatives)", phases=Phases)
    ph = problem.phases.slide

    # ------------------------------------------------------------------ values

    @ph.register.continuous
    def continuous(arg, out):
        """Compute the bead's dynamics."""
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = g0 * sin(u)
        return out

    @problem.register.objective
    def objective(arg):
        """Return the time taken, which is the objective."""
        return arg[ph].final.time

        # ------------------------------------------------------------- derivatives

    @ph.register.continuous_jacobian
    def slide_jacobian(arg, jacobian):
        """Compute the first derivatives of the dynamics."""
        v, u = arg.state.v, arg.control.u
        jacobian.dynamics.x.v = cos(u)
        jacobian.dynamics.x.u = -v * sin(u)
        jacobian.dynamics.y.v = sin(u)
        jacobian.dynamics.y.u = v * cos(u)
        jacobian.dynamics.v.u = g0 * cos(u)
        return jacobian

    @ph.register.continuous_hessian
    def slide_hessian(arg, hessian):
        """Compute the second derivatives of the dynamics.

        Each unordered pair is written once. ``hessian.dynamics.x.v.u`` and
        ``hessian.dynamics.x.u.v`` name the same derivative, and writing both is refused
        rather than summed.
        """
        v, u = arg.state.v, arg.control.u
        hessian.dynamics.x.v.u = -sin(u)
        hessian.dynamics.x.u.u = -v * cos(u)
        hessian.dynamics.y.v.u = cos(u)
        hessian.dynamics.y.u.u = -v * sin(u)
        hessian.dynamics.v.u.u = -g0 * sin(u)
        return hessian

    @problem.register.objective_gradient
    def minimum_time_gradient(_arg, gradient):
        """Compute the gradient of the objective, which is one in the final time."""
        gradient[gradient.phases[ph].final.time] = 1.0
        return gradient

    @problem.register.objective_hessian
    def minimum_time_hessian(_arg, hessian):
        """Compute the Hessian of the objective, which is zero everywhere.

        Registering it is what says so. An entry not written is structurally zero, and a
        callback that writes none says that of every entry -- which is not the same as
        leaving the callback out, because that would be indistinguishable from forgetting
        it, and a wrong Hessian costs iterations rather than raising.
        """
        return hessian

        # ------------------------------------------------------------------- setup

    ph.time.initial = (0.0, 0.0)
    ph.state.x.initial = (0.0, 0.0)
    ph.state.y.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.x.final = (1.0, 1.0)
    ph.state.x.bounds = (0, 10)
    ph.state.y.bounds = (0, 10)
    ph.state.v.bounds = (0, 10)
    ph.control.u.bounds = (-pi / 2, pi / 2)

    ph.time.guess = (0.0, 1.0)
    ph.state.x.guess = (0, 1)
    ph.state.y.guess = (0, 1)
    ph.state.v.guess = (0, 5)

    problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    problem.ipopt_options.print_level = 3
    return problem


def main() -> None:
    """Solve the brachistochrone problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final time = {solution .objective :.6f} s")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
