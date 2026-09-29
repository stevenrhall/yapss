"""

The brachistochrone problem: the shape of the fastest slide between two points.

A bead slides without friction from the origin to x = 1 under gravity. The control is the
slope angle of the path, and the objective is the time taken.

"""

__all__ = ["main", "plot_solution", "setup"]

# standard library imports
from typing import Any

# third-party imports
import matplotlib.pyplot as plt

# package imports
import yapss
from yapss.math import cos, pi, sin

g0 = 32.174


class State(yapss.State):
    """Define the state vector of the brachistochrone problem."""

    x = yapss.scalar()
    """Horizontal position."""
    y = yapss.scalar()
    """Vertical drop."""
    v = yapss.scalar()
    """Speed."""


class Control(yapss.Control):
    """Define the control vector of the brachistochrone problem."""

    u = yapss.scalar()
    """Path angle."""


class Phase(yapss.Phase):
    """The bead's descent, built from the state and control above."""

    state: State
    control: Control


class Phases(yapss.Phases):
    """Define the only phase of the problem."""

    phase: Phase


class Brachistochrone(yapss.Problem):
    """The brachistochrone: the fastest slide between two points."""

    phases: Phases


# Names for the types the annotations below use.
PhaseArg = yapss.ContinuousArg[State, Control]
PhaseOut = yapss.ContinuousOut[State]


def setup() -> Brachistochrone:
    """Set up the brachistochrone problem.

    Returns
    -------
    Brachistochrone
        The problem of finding the shape of a wire along which a bead slides without
        friction from the origin to x = 1 in minimum time.
    """
    problem = Brachistochrone("Brachistochrone")
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(arg: PhaseArg, out: PhaseOut) -> None:
        """Compute the bead's dynamics."""
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = g0 * sin(u)

    @problem.register.objective
    def objective(arg: yapss.DiscreteArg) -> Any:
        """Return the time taken, which is the objective."""
        return arg[ph].final_time

    # set boundary conditions
    ph.time.initial = (0.0, 0.0)
    ph.state.x.initial = (0.0, 0.0)
    ph.state.y.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.x.final = (1.0, 1.0)
    ph.state.x.bounds = (0, 10)
    ph.state.y.bounds = (0, 10)
    ph.state.v.bounds = (0, 10)
    ph.control.u.bounds = (-pi / 2, pi / 2)

    # set guess for solution
    ph.time.guess = (0.0, 1.0)
    ph.state.x.guess = (0, 1)
    ph.state.y.guess = (0, 1)
    ph.state.v.guess = (0, 5)

    # set ipopt options
    problem.ipopt_options.print_level = 3

    return problem


def plot_solution(problem: Brachistochrone, solution: yapss.Solution) -> None:
    """Plot the trajectory, the states, the control, the costate, and the Hamiltonian.

    Parameters
    ----------
    problem : Brachistochrone
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution.phases[problem.phases.phase]
    time = ps.time

    plt.figure()
    plt.plot(ps.state.x, ps.state.y, linewidth=2)
    plt.xlabel("Horizontal position, $x(t)$ (ft)")
    plt.ylabel("Vertical position, $y(t)$ (ft)")
    plt.xlim((0.0, 1.0))
    plt.ylim((0.8, -0.1))
    plt.axis("scaled")
    plt.grid()
    plt.tight_layout()

    plt.figure()
    for values, label in ((ps.state.x, "$x(t)$"), (ps.state.y, "$y(t)$"), (ps.state.v, "$v(t)$")):
        plt.plot(time, values, linewidth=2, label=label)
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel("States")
    plt.xlim((time[0], time[-1]))
    plt.legend(framealpha=1.0)
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(time, ps.control.u, linewidth=2)
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel(r"Control, $\theta(t)$ (rad)")
    plt.xlim((time[0], time[-1]))
    plt.grid()
    plt.tight_layout()

    plt.figure()
    costate = ps.costate
    for values, label in (
        (costate.x, r"$p_x(t)$"),
        (costate.y, r"$p_y(t)$"),
        (costate.v, r"$p_v(t)$"),
    ):
        plt.plot(time, values, linewidth=2, label=label)
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel("Costates")
    plt.xlim((time[0], time[-1]))
    plt.legend(framealpha=1.0)
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(time, ps.hamiltonian, linewidth=2)
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel(r"Hamiltonian, $\mathcal{H}$")
    plt.xlim((time[0], time[-1]))
    plt.ylim((-1.1, -0.9))
    plt.grid()
    plt.tight_layout()


def main() -> None:
    """Solve the brachistochrone problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final time = {solution.objective:.6f} s")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
