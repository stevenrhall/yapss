"""

Dynamic soaring: extracting energy from a wind that grows with altitude.

An albatross, or a sailplane, can fly a closed circuit indefinitely without thrust if the wind
speed increases with height: the bird climbs into the faster air heading upwind and dives back
through the slower air heading downwind, gaining more energy from the shear than drag takes
away. The question this problem answers is how strong the shear has to be.

So the wind gradient is not given. It is a parameter of the problem, `beta`, and it is the
objective: the smallest gradient for which a closed circuit exists. The circuit is closed by
three discrete constraints requiring the speed, the flight path angle and the heading to return
to their initial values -- the heading after one full turn, which is why its constraint is 360
degrees rather than zero.

This is the one example where a parameter appears in a phase's dynamics, and it is also the
largest mesh in the corpus, because the lift coefficient runs up against its load-factor limit
and the derivatives are discontinuous where it does.

"""

__all__ = ["main", "plot_solution", "setup"]

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import yapss
from yapss.math import cos, sin

w0 = 0.0
"""Wind speed at zero altitude (ft/s)."""
g0 = 32.2
"""Gravitational acceleration (ft/s^2)."""
cd0 = 0.00873
"""Zero-lift drag coefficient."""
rho0 = 0.002378
"""Air density (slug/ft^3)."""
mass = 5.6
"""Mass of the vehicle (slug)."""
area = 45.09703
"""Reference area (ft^2)."""
k = 0.045
"""Induced drag factor."""
cl_max = 1.5
"""Largest lift coefficient the wing will give."""
load_factor_max = 5.0
"""The structural limit, which the path constraint enforces."""


class State(yapss.State):
    """Where the vehicle is and how it is moving."""

    x = yapss.scalar()
    """East position."""
    y = yapss.scalar()
    """North position."""
    h = yapss.scalar()
    """Altitude."""
    v = yapss.scalar()
    """Airspeed."""
    gamma = yapss.scalar()
    """Flight path angle."""
    psi = yapss.scalar()
    """Heading angle."""


class Control(yapss.Control):
    """How the vehicle is being flown."""

    cl = yapss.scalar()
    """Lift coefficient."""
    phi = yapss.scalar()
    """Bank angle."""


class Path(yapss.Path):
    """What the airframe will take."""

    load_factor = yapss.scalar()
    """Load factor, in gravities."""


class Parameter(yapss.Parameter):
    """The wind profile, which is what the problem is solving for."""

    beta = yapss.scalar()
    """Wind gradient with altitude."""


class Discrete(yapss.Discrete):
    """What it means for the flight to be a repeatable circuit."""

    v_periodic = yapss.scalar()
    """Change in airspeed over the circuit."""
    gamma_periodic = yapss.scalar()
    """Change in flight path angle."""
    psi_periodic = yapss.scalar()
    """Change in heading, one full turn."""


class Phases(yapss.Phases):
    """One phase: one circuit of the loop."""

    loop = yapss.phase(state=State, control=Control, path=Path)


def setup() -> yapss.Problem:
    """Set up the dynamic soaring problem.

    Returns
    -------
    yapss.Problem
        The problem.
    """
    problem = yapss.Problem(
        "Dynamic soaring", phases=Phases, parameter=Parameter, discrete=Discrete
    )
    ph = problem.phases.loop

    @ph.register.continuous
    def continuous(arg, out):
        """Compute the flight dynamics in a wind that grows with altitude."""
        h, v = arg.state.h, arg.state.v
        gamma, psi = arg.state.gamma, arg.state.psi
        cl, phi = arg.control.cl, arg.control.phi
        beta = arg.parameter.beta

        weight = mass * g0
        pressure = rho0 * v**2 / 2
        lift = pressure * area * cl
        drag = pressure * area * (cd0 + k * cl**2)

        cos_gamma, sin_gamma = cos(gamma), sin(gamma)
        cos_psi, sin_psi = cos(psi), sin(psi)

        h_dot = v * sin_gamma
        # the wind the vehicle feels changes as it climbs, and that is the whole mechanism
        wind_dot = beta * h_dot

        out.dynamics.x = v * cos_gamma * sin_psi + beta * h + w0
        out.dynamics.y = v * cos_gamma * cos_psi
        out.dynamics.h = h_dot
        out.dynamics.v = -drag / mass - g0 * sin_gamma - wind_dot * cos_gamma * sin_psi
        out.dynamics.gamma = (
            lift * cos(phi) - weight * cos_gamma + mass * wind_dot * sin_gamma * sin_psi
        ) / (mass * v)
        out.dynamics.psi = (lift * sin(phi) - mass * wind_dot * cos_psi) / (mass * v * cos_gamma)
        out.path.load_factor = (0.5 * rho0 * area / weight) * cl * v**2
        return out

    @problem.register.objective
    def objective(arg):
        """Return the wind gradient, which is what is to be made as small as possible."""
        return arg.parameter.beta

    @problem.register.discrete
    def discrete(arg, out):
        """Require the flight to come back to the state it started in, one turn later."""
        end = arg[ph]
        out.discrete.v_periodic = end.final.v - end.initial.v
        out.discrete.gamma_periodic = end.final.gamma - end.initial.gamma
        out.discrete.psi_periodic = end.final.psi - end.initial.psi
        return out

    # ------------------------------------------------------------------- setup

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (10.0, 30.0)

    # the circuit starts and ends at the origin
    for end in (ph.state.initial, ph.state.final):
        end.x = (0.0, 0.0)
        end.y = (0.0, 0.0)
        end.h = (0.0, 0.0)

    ph.state.bounds.x = (-1500, 1500)
    ph.state.bounds.y = (-1000, 1000)
    ph.state.bounds.h = (0, 1000)
    ph.state.bounds.v = (10, 350)
    ph.state.bounds.gamma = (np.radians(-75), np.radians(75))
    ph.state.bounds.psi = (np.radians(-225), np.radians(225))

    ph.control.bounds.cl = (0, cl_max)
    ph.control.bounds.phi = (np.radians(-75), np.radians(75))
    ph.path.bounds.load_factor = (-2, load_factor_max)

    problem.discrete.bounds.v_periodic = (0.0, 0.0)
    problem.discrete.bounds.gamma_periodic = (0.0, 0.0)
    problem.discrete.bounds.psi_periodic = (np.radians(360), np.radians(360))

    # A circuit that is roughly the right shape and size, so the solver starts from a closed
    # loop rather than having to find one.
    tf = 24.0
    t = np.linspace(0.0, tf, num=50)
    turn = 2 * np.pi * t / tf
    x = 600 * (np.cos(turn) - 1)
    ph.time.guess = (0.0, tf)
    ph.state.guess.x = yapss.interp(t, x)
    ph.state.guess.y = yapss.interp(t, -200 * np.sin(turn))
    ph.state.guess.h = yapss.interp(t, -0.7 * x)
    ph.state.guess.v = (150.0, 150.0)
    ph.state.guess.gamma = (0.0, 0.0)
    ph.state.guess.psi = yapss.interp(t, np.radians(t / tf * 360))
    ph.control.guess.cl = (0.5, 0.5)
    ph.control.guess.phi = (np.radians(45), np.radians(45))
    problem.parameter.guess.beta = 0.08

    # Scaling, which this problem needs: the states run over four orders of magnitude.
    problem.objective.scale = 0.1
    problem.parameter.scale.beta = 0.1
    for name in Discrete._fields:
        setattr(problem.discrete.scale, name, 200.0)
    for name, value in (("x", 1000.0), ("y", 1000.0), ("h", 1000.0), ("v", 200.0)):
        setattr(ph.state.scale, name, value)
        setattr(ph.state.defect_scale, name, value)
    ph.state.scale.gamma = ph.state.defect_scale.gamma = 1.0
    ph.state.scale.psi = ph.state.defect_scale.psi = 6.0
    ph.path.scale.load_factor = 7.0
    ph.time.scale = 30.0

    # A dense mesh, to capture where the lift coefficient meets its limit and the derivatives
    # of the solution are discontinuous.
    ph.mesh = yapss.Mesh.uniform(segments=50, points=6)
    problem.method = "lgl"

    problem.ipopt_options.max_iter = 500
    problem.ipopt_options.print_level = 3
    return problem


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    """Plot the circuit in three dimensions, and the quantities along it.

    Parameters
    ----------
    problem : yapss.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    ps = solution[problem.phases.loop]
    t = ps.time
    x, y, h = ps.state.x, ps.state.y, ps.state.h

    plt.figure()
    ax: Axes3D = plt.axes(projection=Axes3D.name)
    ax.plot3D(x, y, h)
    ax.plot3D(0 * x - 1200, y, h, "r--")
    ax.plot3D(x, 0 * y + 500, h, "r--")
    ax.plot3D(x, y, 0 * h - 100, "r--")
    ax.set_xlim([-1200, 0])
    ax.set_ylim([-600, 500])
    ax.set_zlim([-100, 1000])
    ax.set_xlabel(r"$x$ (ft)")
    ax.set_ylabel(r"$y$ (ft)")
    ax.set_zlabel(r"$h$ (ft)")
    plt.tight_layout()

    panels = (
        (r"Velocity, $v$ (ft/s)", ps.state.v),
        (r"Flight path angle, $\gamma$ (deg)", np.rad2deg(ps.state.gamma)),
        (r"Heading angle, $\psi$ (deg)", np.rad2deg(ps.state.psi)),
    )
    for ylabel, quantity in panels:
        plt.figure()
        plt.plot(t, quantity)
        plt.xlabel(r"Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.grid()
        plt.tight_layout()

    # the lift coefficient, against the load-factor limit that bounds it
    plt.figure()
    limit = load_factor_max * (mass * g0) / (0.5 * rho0 * area * ps.state.v**2)
    plt.plot(t, limit, "r--")
    plt.plot(t, ps.control.cl)
    plt.ylim((0, 1))
    legend = plt.legend(["Load factor limit", "Lift coefficient, $C_{L}$"])
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0)
    plt.xlabel(r"Time, $t$ (s)")
    plt.ylabel(r"Lift coefficient, $C_L$")
    plt.grid()
    plt.tight_layout()

    for ylabel, quantity in (
        (r"Bank angle, $\phi$ (deg)", np.rad2deg(ps.control.phi)),
        (r"Hamiltonian, $\mathcal{H}$", ps.hamiltonian),
    ):
        plt.figure()
        plt.plot(t, quantity)
        plt.xlabel(r"Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.grid()
        plt.tight_layout()


def main() -> None:
    """Solve the dynamic soaring problem and plot the circuit it finds."""
    problem = setup()
    solution = problem.solve()
    print(f"\nsmallest wind gradient = {solution.objective:.9f} 1/s")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
