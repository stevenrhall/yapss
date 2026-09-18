"""

The orbit raising problem: reach the largest orbit a low-thrust vehicle can in a fixed time.

The vehicle's mass falls as it burns, so its thrust acceleration depends explicitly on the
time -- which makes this the one ported example whose continuous callback reads the phase's
independent variable.

"""

__all__ = ["main", "plot_solution", "setup"]

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from yapss import _next as yapss
from yapss.math import sqrt

m_0, r_0, mu = 1.0, 1.0, 1.0
thrust, m_dot = 0.1405, 0.0749
t_0, t_f = 0.0, 3.32
v_r_0, v_r_f = 0.0, 0.0
theta_0, v_theta_0 = 0.0, 1.0

r_min, r_max = 1.0, 10.0
v_min, v_max = -10.0, 10.0
u_min, u_max = -1.1, 1.1


class Orbit(yapss.Vector):
    """Where the vehicle is and how fast it is going, in polar coordinates."""

    r = yapss.field(latex="r", doc="radius")
    theta = yapss.field(units="rad", latex=r"\theta", doc="polar angle")
    v_r = yapss.field(latex="v_r", doc="radial velocity")
    v_theta = yapss.field(latex=r"v_\theta", doc="tangential velocity")


class Steering(yapss.Vector):
    """The direction the thrust points, as a unit vector in polar coordinates."""

    u_r = yapss.field(latex="u_r", doc="radial component of the thrust direction")
    u_theta = yapss.field(latex=r"u_\theta", doc="tangential component")


class Limits(yapss.Vector):
    """The steering vector must have unit magnitude."""

    unit_thrust = yapss.field(doc="squared magnitude of the thrust direction")


class Target(yapss.Vector):
    """The orbit that must be reached."""

    circular = yapss.field(doc="the final orbit must be circular")


class Phases(yapss.Phases):
    """One phase: the vehicle thrusts continuously."""

    raise_ = yapss.phase(state=Orbit, control=Steering, path=Limits)


def setup() -> yapss.Problem:
    """Set up the orbit raising problem.

    Returns
    -------
    yapss._next.Problem
        The problem.
    """
    problem = yapss.Problem("Orbit raising", phases=Phases, discrete=Target)
    ph = problem.phases.raise_

    @ph.register.continuous
    def raising(arg, out):
        """Compute the vehicle's dynamics and the magnitude of its steering vector."""
        r, v_r, v_theta = arg.state.r, arg.state.v_r, arg.state.v_theta
        u_r, u_theta = arg.control.u_r, arg.control.u_theta
        # the mass falls as the vehicle burns, so the acceleration depends on the time itself
        a = thrust / (m_0 - m_dot * arg.time)
        out.dynamics.r = v_r
        out.dynamics.theta = v_theta / r
        out.dynamics.v_r = v_theta**2 / r - mu / r**2 + a * u_r
        out.dynamics.v_theta = -(v_r * v_theta) / r + a * u_theta
        out.path.unit_thrust = u_r**2 + u_theta**2
        return out

    @problem.register.objective
    def largest_orbit(arg):
        """Return the final radius, which is to be made as large as possible."""
        return arg[ph].final.r

    @problem.register.discrete
    def circular(arg, out):
        """Require the final orbit to be circular."""
        final = arg[ph].final
        out.discrete.circular = final.v_theta - sqrt(mu / final.r)
        return out

    problem.objective.sense = "maximize"

    ph.time.initial = t_0
    ph.time.final = t_f
    ph.state.initial.r = r_0
    ph.state.initial.theta = theta_0
    ph.state.initial.v_r = v_r_0
    ph.state.initial.v_theta = v_theta_0
    ph.state.final.v_r = v_r_f
    ph.state.bounds.r = (r_min, r_max)
    ph.state.bounds.theta = (-pi, pi)
    ph.state.bounds.v_r = (v_min, v_max)
    ph.state.bounds.v_theta = (v_min, v_max)
    ph.control.bounds.u_r = (u_min, u_max)
    ph.control.bounds.u_theta = (u_min, u_max)
    ph.path.bounds.unit_thrust = (-np.inf, 1.0)
    problem.discrete.bounds.circular = 0.0

    ph.time.guess = (t_0, t_f)
    ph.state.guess.r = (r_0, 1.5 * r_0)
    ph.state.guess.theta = (theta_0, pi)
    ph.state.guess.v_r = (v_r_0, v_r_f)
    ph.state.guess.v_theta = (v_theta_0, 0.5 * v_theta_0)
    ph.control.guess.u_r = (0.0, 1.0)
    ph.control.guess.u_theta = (1.0, 0.0)

    problem.method = "lgl"
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the trajectory and the steering angle.

    Parameters
    ----------
    problem : yapss._next.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._next.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.raise_]
    plt.figure()
    plt.polar(ps.state.theta, ps.state.r)
    plt.title("Orbit raising trajectory")

    plt.figure()
    plt.plot(ps.time, np.arctan2(ps.control.u_r, ps.control.u_theta) * 180 / pi)
    plt.xlabel("Time")
    plt.ylabel("Steering angle (deg)")


def main() -> None:
    """Solve the orbit raising problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
