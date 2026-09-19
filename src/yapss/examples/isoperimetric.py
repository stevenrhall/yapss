"""

The isoperimetric problem: the closed curve of given perimeter that encloses the most area.

The answer is a circle, and the area it encloses is 1/(4*pi) for a perimeter of 1. What makes
the problem worth having in the corpus is the machinery it needs: the independent variable is
arc length rather than time, the curve is held at unit speed by a path constraint, the area is
an integral, two further integrals place the centroid at the origin, and two discrete
constraints require the curve to close.

Nothing in the API knows that the independent variable is a length. A phase runs over a
variable it names -- here `s` -- and `time` is simply the name YAPSS uses where a problem does
not say otherwise.

"""

__all__ = ["main", "plot_solution", "setup"]

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import yapss

PERIMETER = 1.0
"""The length of the curve, which the unit-speed constraint and the arc-length span fix."""
AREA = 1 / (4 * math.pi)
"""The largest area a closed curve of unit perimeter can enclose."""


class Curve(yapss.Vector):
    """A point on the curve."""

    x = yapss.field(latex="x", doc="horizontal position")
    y = yapss.field(latex="y", doc="vertical position")


class Tangent(yapss.Vector):
    """The direction the curve is going, which is the control."""

    tx = yapss.field(latex="t_x", doc="horizontal component of the tangent")
    ty = yapss.field(latex="t_y", doc="vertical component of the tangent")


class UnitSpeed(yapss.Vector):
    """The constraint that makes the independent variable arc length."""

    speed_squared = yapss.field(doc="squared speed along the curve, which must be one")


class Moments(yapss.Vector):
    """What is accumulated along the curve."""

    area = yapss.field(doc="area enclosed, by the shoelace formula")
    x_moment = yapss.field(doc="first moment about the y axis")
    y_moment = yapss.field(doc="first moment about the x axis")


class Closure(yapss.Vector):
    """What it means for the curve to close."""

    closure_x = yapss.field(doc="horizontal gap between the ends")
    closure_y = yapss.field(doc="vertical gap between the ends")


class Phases(yapss.Phases):
    """One phase, running over arc length rather than time."""

    curve = yapss.phase(
        state=Curve,
        control=Tangent,
        path=UnitSpeed,
        integral=Moments,
        s=yapss.field(latex="s", doc="arc length"),
    )


def setup() -> yapss.Problem:
    """Set up the isoperimetric problem.

    Returns
    -------
    yapss._api.Problem
        The problem.
    """
    problem = yapss.Problem("Isoperimetric problem", phases=Phases, discrete=Closure)
    ph = problem.phases.curve

    @ph.register.continuous
    def trace(arg, out):
        """Move along the curve, accumulating the area and the moments."""
        x, y = arg.state.x, arg.state.y
        tx, ty = arg.control.tx, arg.control.ty
        out.dynamics.x = tx
        out.dynamics.y = ty
        out.path.speed_squared = tx**2 + ty**2
        out.integrand.area = (y * tx - x * ty) / 2
        out.integrand.x_moment = x
        out.integrand.y_moment = y
        return out

    @problem.register.objective
    def enclosed_area(arg):
        """Return the area enclosed, which is to be made as large as possible."""
        return arg[ph].integral.area

    @problem.register.discrete
    def closed(arg, out):
        """Require the curve to return to where it started."""
        end = arg[ph]
        out.discrete.closure_x = end.final.x - end.initial.x
        out.discrete.closure_y = end.final.y - end.initial.y
        return out

    problem.objective.sense = "maximize"

    # Arc length runs from 0 to the perimeter, and unit speed is what makes it arc length.
    ph.s.initial = (0.0, 0.0)
    ph.s.final = (PERIMETER, PERIMETER)
    ph.path.bounds.speed_squared = (1.0, 1.0)

    # The centroid at the origin, which fixes the circle's position rather than its shape.
    ph.integral.bounds.x_moment = (0.0, 0.0)
    ph.integral.bounds.y_moment = (0.0, 0.0)

    problem.discrete.bounds.closure_x = (0.0, 0.0)
    problem.discrete.bounds.closure_y = (0.0, 0.0)

    # A square of the right perimeter, so the guess satisfies the constraint it starts from.
    side = PERIMETER / 4
    corners = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * PERIMETER
    ph.s.guess = (0.0, PERIMETER)
    ph.state.guess.x = yapss.interp(corners, side * np.array([0.5, 0.5, -0.5, -0.5, 0.5]))
    ph.state.guess.y = yapss.interp(corners, side * np.array([0.5, -0.5, -0.5, 0.5, 0.5]))

    ph.mesh = yapss.Mesh.uniform(segments=3, points=12)

    # A tighter tolerance than one would normally use, to show how accurate the pseudospectral
    # method is here with a modest number of collocation points. The relative error against the
    # closed-form answer, which `main` prints, is the real check.
    problem.ipopt_options.tol = 1e-14
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the curve found and the Hamiltonian along it.

    The collocation points are shown as dots, with a cubic spline through them.

    Parameters
    ----------
    problem : yapss._api.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._api.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.curve]
    s = ps.s
    fine = np.linspace(s[0], s[-1], 500)
    x = interp1d(s, ps.state.x, kind="cubic")(fine)
    y = interp1d(s, ps.state.y, kind="cubic")(fine)

    plt.figure()
    plt.plot(x, y)
    plt.plot(ps.state.x, ps.state.y, ".", markersize=10)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis("square")
    plt.tight_layout()

    plt.figure()
    plt.plot(s, ps.hamiltonian)
    plt.xlabel("Arc length, $s$")
    plt.ylabel(r"Hamiltonian, $\mathcal{H}$")
    plt.grid()
    plt.tight_layout()


def main() -> None:
    """Solve the isoperimetric problem and plot the curve it finds."""
    problem = setup()
    solution = problem.solve()
    area = solution.objective
    print(f"\nmaximum area = {area} (should be 1 / (4 pi) = {AREA})")
    print(f"relative error = {abs(area - AREA) / AREA:.3e}")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
