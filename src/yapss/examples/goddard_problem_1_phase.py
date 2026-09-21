"""

The Goddard rocket problem in one phase, with derivatives supplied by hand.

A rocket rises vertically against drag and gravity, burning fuel to reach the greatest altitude
it can. The answer is bang-singular-bang: full thrust, then an arc along which a switching
function vanishes and the thrust takes an interior value, then coasting.

One phase cannot represent that middle arc, and watching it fail to is the reason this example
is worth having beside `goddard_problem_3_phase.py`. The transcription never asks the thrust to
be smooth, so across the singular region the solution chatters -- banging between zero and full
from one collocation point to the next, whose average is close to the singular thrust and whose
altitude is close to the right one. The minimizer is minimizing the problem it was given, and
the problem it was given does not say the arc is an arc. Stating the three arcs as three phases
is what says it, which is what the other example does.

It is also where the derivative callbacks are written out on a problem with real dynamics. The
entries are named rather than numbered -- ``jacobian.dynamics.v.h`` is the derivative of the
rate of change of speed with respect to altitude -- so the sparsity structure is the set of
names written, and an entry omitted is structurally zero. Compare `brachistochrone.py` with
`brachistochrone_user_derivatives.py` for the same pairing on a smaller problem.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt

import yapss
from yapss.math import exp

Tm = 193.044
"""Maximum thrust (lbf)."""
g = 32.174
"""Gravitational acceleration (ft/s^2)."""
sigma = 5.49153484923381010e-05
"""Drag coefficient."""
c = 1580.9425279876559
"""Exhaust velocity (ft/s)."""
h0 = 23800.0
"""Density scale height (ft)."""
m0, mf = 3.0, 1.0
"""Initial and final mass (slug)."""
h_max, v_max = 30_000.0, 15_000.0
"""How far the trajectory is allowed to reach."""
tf_min, tf_max = 20.0, 100.0
"""How long the flight may last."""


class State(yapss.State):
    """Where the rocket is, how fast it is going, and what it weighs."""

    h = yapss.scalar()
    """Altitude."""
    v = yapss.scalar()
    """Velocity."""
    m = yapss.scalar()
    """Mass."""


class Control(yapss.Control):
    """The engine setting."""

    thrust = yapss.scalar()
    """Thrust."""


class Flight(yapss.Phase):
    """The whole flight."""

    state: State
    control: Control


class Phases(yapss.Phases):
    """One phase: the whole flight, whatever shape the thrust programme turns out to have."""

    flight: Flight


def setup() -> yapss.Problem:
    """Set up the one-phase Goddard rocket problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem("Goddard rocket, one phase", phases=Phases)
    ph = problem.phases.flight

    @ph.register.continuous
    def continuous(arg, out):
        """Compute the rocket's dynamics."""
        h, v, m = arg.state.h, arg.state.v, arg.state.m
        thrust = arg.control.thrust
        out.dynamics.h = v
        out.dynamics.v = (thrust - sigma * v**2 * exp(-h / h0)) / m - g
        out.dynamics.m = -thrust / c
        return out

    @problem.register.objective
    def objective(arg):
        """Return the altitude reached, which is to be made as large as possible."""
        return arg[ph].final.h

    # ------------------------------------------------------------- derivatives

    @ph.register.continuous_jacobian
    def rocket_jacobian(arg, jacobian):
        """Compute the first derivatives of the dynamics."""
        h, v, m = arg.state.h, arg.state.v, arg.state.m
        thrust = arg.control.thrust
        drag_over_v2 = sigma * exp(-h / h0)
        drag_over_v = drag_over_v2 * v
        drag = drag_over_v * v

        jacobian.dynamics.h.v = 1.0
        jacobian.dynamics.v.h = drag / (h0 * m)
        jacobian.dynamics.v.v = -2 * drag_over_v / m
        jacobian.dynamics.v.m = -(thrust - drag) / m**2
        jacobian.dynamics.v.thrust = 1 / m
        jacobian.dynamics.m.thrust = -1 / c
        return jacobian

    @ph.register.continuous_hessian
    def rocket_hessian(arg, hessian):
        """Compute the second derivatives of the dynamics.

        Only the speed's rate of change is nonlinear, so it is the only output with any
        entries. Each unordered pair is written once: ``hessian.dynamics.v.h.m`` and
        ``hessian.dynamics.v.m.h`` name the same derivative, and writing both is refused
        rather than summed.
        """
        h, v, m = arg.state.h, arg.state.v, arg.state.m
        thrust = arg.control.thrust
        drag_over_v2 = sigma * exp(-h / h0)
        drag_over_v = drag_over_v2 * v
        drag = drag_over_v * v

        hessian.dynamics.v.h.h = -drag / (h0**2 * m)
        hessian.dynamics.v.h.v = 2 * drag_over_v / (h0 * m)
        hessian.dynamics.v.h.m = -drag / (h0 * m**2)
        hessian.dynamics.v.v.v = -2 * drag_over_v2 / m
        hessian.dynamics.v.v.m = 2 * drag_over_v / m**2
        hessian.dynamics.v.m.m = 2 * (thrust - drag) / m**3
        hessian.dynamics.v.m.thrust = -1 / m**2
        return hessian

    @problem.register.objective_gradient
    def final_altitude_gradient(_arg, gradient):
        """Compute the gradient of the objective, which is one in the final altitude."""
        gradient[gradient.phases[ph].final.h] = 1.0
        return gradient

    @problem.register.objective_hessian
    def final_altitude_hessian(_arg, hessian):
        """Compute the Hessian of the objective, which is zero everywhere.

        Registering it is what says so: an entry not written is structurally zero, and a
        callback that writes none says that of every entry. Leaving the callback out would be
        indistinguishable from forgetting it.
        """
        return hessian

    # ------------------------------------------------------------------- setup

    problem.objective.sense = "maximize"
    problem.derivatives.method = "user"

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (tf_min, tf_max)
    ph.state.h.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.m.initial = (m0, m0)
    ph.state.h.bounds = (0, h_max)
    ph.state.v.bounds = (0, v_max)
    ph.state.m.bounds = (mf, m0)
    ph.state.m.final = (mf, mf)
    ph.control.thrust.bounds = (0, Tm)

    ph.time.guess = (0.0, tf_max)
    ph.state.h.guess = (0.0, h_max)
    ph.state.v.guess = (0.0, 0.0)
    ph.state.m.guess = (m0, mf)
    ph.control.thrust.guess = (Tm, 0.0)

    ph.state.h.scale = ph.state.h.defect_scale = 18_000.0
    ph.state.v.scale = ph.state.v.defect_scale = 800.0
    ph.state.m.scale = ph.state.m.defect_scale = 3.0
    ph.time.scale = 30.0

    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the trajectory, the thrust programme, and the Hamiltonian.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.flight]
    panels = (
        ("Thrust, $T$ (lbf)", ps.control.thrust),
        ("Altitude, $h$ (ft)", ps.state.h),
        ("Velocity, $v$ (ft/s)", ps.state.v),
        ("Mass, $m$ (slug)", ps.state.m),
        (r"Hamiltonian, $\mathcal{H}$ (ft/s)", ps.hamiltonian),
    )
    for ylabel, quantity in panels:
        plt.figure()
        plt.plot(ps.time, quantity)
        plt.xlabel("Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.xlim((ps.time[0], ps.time[-1]))
        plt.grid()
        plt.tight_layout()


def main() -> None:
    """Solve the one-phase Goddard rocket problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"\nmaximum altitude = {solution.objective:.3f} ft")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
