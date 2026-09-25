"""

The orbit raising problem: reach the largest orbit a low-thrust vehicle can in a fixed time.

The vehicle's mass falls as it burns, so its thrust acceleration depends explicitly on the
time -- which makes this the one ported example whose continuous callback reads the phase's
independent variable.

"""

__all__ = ["main", "plot_solution", "setup"]

from math import pi
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import yapss
from yapss.math import sqrt

m_0, r_0, mu = 1.0, 1.0, 1.0
thrust, m_dot = 0.1405, 0.0749
t_0, t_f = 0.0, 3.32
v_r_0, v_r_f = 0.0, 0.0
theta_0, v_theta_0 = 0.0, 1.0

r_min, r_max = 1.0, 10.0
v_min, v_max = -10.0, 10.0
u_min, u_max = -1.1, 1.1


class State(yapss.State):
    """Where the vehicle is and how fast it is going, in polar coordinates."""

    r = yapss.scalar()
    """Radius."""
    theta = yapss.scalar()
    """Polar angle."""
    v_r = yapss.scalar()
    """Radial velocity."""
    v_theta = yapss.scalar()
    """Tangential velocity."""


class Control(yapss.Control):
    """The direction the thrust points, as a unit vector in polar coordinates."""

    u_r = yapss.scalar()
    """Radial component of the thrust direction."""
    u_theta = yapss.scalar()
    """Tangential component."""


class Path(yapss.Path):
    """The steering vector must have unit magnitude."""

    unit_thrust = yapss.scalar()
    """Squared magnitude of the thrust direction."""


class Discrete(yapss.Discrete):
    """The orbit that must be reached."""

    circular = yapss.scalar()
    """The final orbit must be circular."""


class Transfer(yapss.Phase):
    """The transfer, with the vehicle thrusting throughout."""

    state: State
    control: Control
    path: Path
    time: yapss.Independent


class Phases(yapss.Phases):
    """One phase: the vehicle thrusts continuously."""

    raise_: Transfer


def setup() -> yapss.Problem[Phases, Discrete]:
    """Set up the orbit raising problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Orbit raising", phases=Phases, discrete=Discrete)
    ph = problem.phases.raise_

    @ph.register.continuous
    def continuous(
        arg: yapss.ContinuousArg[State, Control], out: yapss.ContinuousOut[State, Path]
    ) -> None:
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

    @problem.register.objective
    def objective(arg: yapss.EndpointArg) -> Any:
        """Return the final radius, which is to be made as large as possible."""
        return arg[ph].final.r

    @problem.register.discrete
    def discrete(arg: yapss.EndpointArg, out: yapss.DiscreteOut[Discrete]) -> None:
        """Require the final orbit to be circular."""
        final = arg[ph].final
        out.discrete.circular = final.v_theta - sqrt(mu / final.r)

    problem.objective.sense = "maximize"

    ph.time.initial = (t_0, t_0)
    ph.time.final = (t_f, t_f)
    ph.state.r.initial = (r_0, r_0)
    ph.state.theta.initial = (theta_0, theta_0)
    ph.state.v_r.initial = (v_r_0, v_r_0)
    ph.state.v_theta.initial = (v_theta_0, v_theta_0)
    ph.state.v_r.final = (v_r_f, v_r_f)
    ph.state.r.bounds = (r_min, r_max)
    ph.state.theta.bounds = (-pi, pi)
    ph.state.v_r.bounds = (v_min, v_max)
    ph.state.v_theta.bounds = (v_min, v_max)
    ph.control.u_r.bounds = (u_min, u_max)
    ph.control.u_theta.bounds = (u_min, u_max)
    ph.path.unit_thrust.bounds = (None, 1.0)
    problem.discrete.circular.bounds = (0.0, 0.0)

    ph.time.guess = (t_0, t_f)
    ph.state.r.guess = (r_0, 1.5 * r_0)
    ph.state.theta.guess = (theta_0, pi)
    ph.state.v_r.guess = (v_r_0, v_r_f)
    ph.state.v_theta.guess = (v_theta_0, 0.5 * v_theta_0)
    ph.control.u_r.guess = (0.0, 1.0)
    ph.control.u_theta.guess = (1.0, 0.0)

    problem.spectral_method = "lgl"
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem[Phases, Discrete], solution: yapss.Solution) -> None:
    """Plot the states, the controls, the steering angle, the orbit, and the Hamiltonian.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.raise_]
    time = ps.time

    plt.figure()
    state = ps.state
    for values, label in (
        (state.r, "$r(t)$"),
        (state.v_r, "$v_r(t)$"),
        (state.v_theta, r"$v_\theta(t)$"),
    ):
        plt.plot(time, values, label=label)
    plt.xlabel("Time")
    plt.ylabel("States")
    plt.xlim(time[0], time[-1])
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(time, ps.control.u_r, label="$u_r(t)$")
    plt.plot(time, ps.control.u_theta, label=r"$u_\theta(t)$")
    plt.xlabel("Time")
    plt.ylabel("Controls")
    plt.xlim(time[0], time[-1])
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(time, np.arctan2(ps.control.u_r, ps.control.u_theta) * 180 / pi)
    plt.xlabel("Time")
    plt.ylabel("Steering angle (deg)")
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.polar(ps.state.theta, ps.state.r)
    plt.title("Orbit raising trajectory")
    plt.tight_layout()

    plt.figure()
    plt.plot(time, ps.hamiltonian)
    plt.xlabel("Time")
    plt.ylabel(r"Hamiltonian, $\mathcal{H}$")
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()


def main() -> None:
    """Solve the orbit raising problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
