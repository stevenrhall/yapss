"""

The Delta III ascent problem: put as much mass as possible into a target orbit.

A four-stage launch vehicle rises from Cape Canaveral into a geosynchronous transfer orbit.
Each stage is a phase, with its own thrust and mass flow, and the phases are joined by
continuity of position and velocity -- but not of mass, which jumps when a stage is dropped.
The trajectory ends on five of the six classical orbital elements.

The physics -- the vector helpers, the orbital-element conversions, and the constants -- is
imported from the released version of this example, so that any difference in the answer is
attributable to the API and not to a slip in transcribing the mathematics.

"""

# The orbital elements are conventionally capitalized.
# ruff: noqa: N806

__all__ = ["main", "plot_solution", "setup"]

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np

import yapss
from yapss._legacy.examples.delta_iii_ascent import (
    CD,
    I1,
    I2,
    T1,
    T2,
    Is,
    Omega_f,
    R_e,
    S,
    Ts,
    a_f,
    cross,
    dot,
    e_f,
    g0,
    h0,
    i_f,
    mag,
    mf_0,
    mf_1,
    mf_2,
    mi_0,
    mi_1,
    mi_2,
    mi_3,
    mu,
    oe_to_rv,
    omega_e,
    omega_f,
    pi_p,
    psi_l,
    rho0,
    t0,
    t1,
    t2,
    t3,
    t4_max,
)
from yapss.math import arccos, cos, exp, pi, sin, sqrt

m_total = mi_0
"""Lift-off mass (kg), which also scales the objective."""
length_scale = R_e
velocity_scale = sqrt(mu / R_e)
time_scale = length_scale / velocity_scale
r_max, v_max, ten = 2 * R_e, 10_000.0, 10.0

EDGES = (t0, t1, t2, t3, t4_max)
"""The time at which each stage begins, and the latest the last one may end."""
LAST = 3
"""The index of the final stage."""
INITIAL_MASS = (mi_0, mi_1, mi_2, mi_3)
FINAL_MASS = (mf_0, mf_1, mf_2, pi_p)
THRUST = (6 * Ts + T1, 3 * Ts + T1, T1, T2)
MASS_FLOW = (
    -(6 * Ts / (g0 * Is) + T1 / (g0 * I1)),
    -(3 * Ts / (g0 * Is) + T1 / (g0 * I1)),
    -T1 / (g0 * I1),
    -T2 / (g0 * I2),
)


class Vehicle(yapss.Vector):
    """Where the vehicle is, how fast it is going, and what it weighs."""

    r = yapss.field(size=3, units="m", latex=r"\mathbf{r}", doc="position")
    v = yapss.field(size=3, units="m/s", latex=r"\mathbf{v}", doc="velocity")
    m = yapss.field(units="kg", latex="m", doc="mass")


class Steering(yapss.Vector):
    """The direction the thrust points, as a unit vector."""

    u = yapss.field(size=3, latex=r"\mathbf{u}", doc="thrust direction")


class Limits(yapss.Vector):
    """What must hold at every instant of the flight."""

    unit_thrust = yapss.field(doc="the steering vector must have unit magnitude")
    radius = yapss.field(units="m", doc="the vehicle must stay above the ground")


class Constraints(yapss.Vector):
    """Continuity where the stages meet, and the orbit that must be reached.

    Position and velocity are separate groups, rather than one block of six, because they are
    scaled differently; the same reason separates the semi-major axis from the angles.
    """

    stage_0_1_position = yapss.field(size=3, units="m")
    stage_0_1_velocity = yapss.field(size=3, units="m/s")
    stage_1_2_position = yapss.field(size=3, units="m")
    stage_1_2_velocity = yapss.field(size=3, units="m/s")
    stage_2_3_position = yapss.field(size=3, units="m")
    stage_2_3_velocity = yapss.field(size=3, units="m/s")
    semi_major_axis = yapss.field(units="m", latex="a")
    eccentricity = yapss.field(latex="e")
    inclination = yapss.field(units="deg", latex="i")
    raan = yapss.field(units="deg", latex=r"\Omega", doc="right ascension of ascending node")
    argument_of_perigee = yapss.field(units="deg", latex=r"\omega")


class Phases(yapss.Phases):
    """One phase per stage."""

    stage_0 = yapss.phase(state=Vehicle, control=Steering, path=Limits)
    stage_1 = yapss.phase(state=Vehicle, control=Steering, path=Limits)
    stage_2 = yapss.phase(state=Vehicle, control=Steering, path=Limits)
    stage_3 = yapss.phase(state=Vehicle, control=Steering, path=Limits)


def make_dynamics(thrust, mass_flow):
    """Return the continuous callback of a stage with the given thrust and mass flow."""

    def dynamics(arg, out):
        """Compute the vehicle's dynamics and the constraints that hold along the way.

        The arithmetic is grouped exactly as the released version of this example groups it, so
        that the two solve the same problem to the last bit and can be compared.
        """
        r_vec, v_vec, m = arg.state.r, arg.state.v, arg.state.m
        u_vec = arg.control.u

        r = (r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2) ** 0.5
        rho = rho0 * exp(-(r - R_e) / h0)
        omega_cross_r = cross([0, 0, omega_e], r_vec)
        relative = [v_vec[i] - omega_cross_r[i] for i in range(3)]
        q_factor = 0.5 * rho * mag(relative) * CD * S
        drag = [-q_factor * relative[i] for i in range(3)]

        mu_over_r3 = mu / r**3
        thrust_over_m = thrust / m
        one_over_m = 1 / m

        out.dynamics.r = v_vec
        out.dynamics.v = [
            -mu_over_r3 * r_vec[i] + thrust_over_m * u_vec[i] + one_over_m * drag[i]
            for i in range(3)
        ]
        out.dynamics.m = mass_flow
        out.path.unit_thrust = mag(u_vec)
        out.path.radius = mag(r_vec)
        return out

    return dynamics


def orbital_elements(r_vec, v_vec):
    """Return five of the six classical orbital elements of the given state."""
    r, v = mag(r_vec), mag(v_vec)
    h_vec = cross(r_vec, v_vec)
    n_vec = cross([0, 0, 1], h_vec)
    h, n = mag(h_vec), mag(n_vec)
    e_vec = tuple(
        ((v**2 - mu / r) * r_vec[i] - dot(r_vec, v_vec) * v_vec[i]) / mu for i in range(3)
    )
    e = mag(e_vec)
    return (
        1 / (2 / r - v**2 / mu),
        e,
        arccos(h_vec[2] / h) * 180 / pi,
        360 - arccos(n_vec[0] / n) * 180 / pi,
        arccos(dot(n_vec, e_vec) / (n * e)) * 180 / pi,
    )


def setup() -> yapss.Problem:
    """Set up the Delta III ascent problem.

    Returns
    -------
    yapss._api.Problem
        The problem.
    """
    problem = yapss.Problem("Delta III ascent", phases=Phases, discrete=Constraints)
    stages = list(problem.phases)

    for stage, thrust, mass_flow in zip(stages, THRUST, MASS_FLOW, strict=True):
        stage.register.continuous(make_dynamics(thrust, mass_flow))

    @problem.register.objective
    def final_mass(arg):
        """Return the mass delivered to orbit, which is to be made as large as possible."""
        return arg[stages[LAST]].final.m

    @problem.register.discrete
    def constraints(arg, out):
        """Join the stages, and require the final state to be on the target orbit."""
        for index, (before, after) in enumerate(pairwise(stages)):
            first, second = arg[before].final, arg[after].initial
            setattr(out.discrete, f"stage_{index}_{index + 1}_position", second.r - first.r)
            setattr(out.discrete, f"stage_{index}_{index + 1}_velocity", second.v - first.v)
        final = arg[stages[LAST]].final
        a, e, i, Omega, omega = orbital_elements(final.r, final.v)
        out.discrete.semi_major_axis = a
        out.discrete.eccentricity = e
        out.discrete.inclination = i
        out.discrete.raan = Omega
        out.discrete.argument_of_perigee = omega
        return out

    problem.objective.sense = "maximize"
    problem.objective.scale = m_total

    _set_bounds(problem, stages)
    _set_scales(problem, stages)
    _set_guess(stages)

    for stage in stages:
        stage.mesh = yapss.Mesh.uniform(segments=5, points=5)
    problem.method = "lgl"
    problem.derivatives.method = "auto"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 1000
    problem.ipopt_options.print_level = 3
    return problem


def _set_bounds(problem, stages):
    """Set the bounds on every stage, and the bounds the constraints must meet."""
    launch = [R_e * cos(psi_l), 0.0, R_e * sin(psi_l)]
    launch_velocity = [0.0, R_e * omega_e * cos(psi_l), 0.0]

    for index, stage in enumerate(stages):
        stage.state.bounds.r[:] = (-r_max, r_max)
        stage.state.bounds.v[:] = (-v_max, v_max)
        stage.state.initial.r[:] = (-r_max, r_max)
        stage.state.initial.v[:] = (-v_max, v_max)
        stage.state.final.r[:] = (-r_max, r_max)
        stage.state.final.v[:] = (-v_max, v_max)
        stage.state.bounds.m = (FINAL_MASS[index] - ten, INITIAL_MASS[index] + ten)
        # The last stage may not deliver less than the payload itself, so its final mass has
        # no leeway below.
        floor = pi_p if index == LAST else FINAL_MASS[index] - ten
        stage.state.final.m = (floor, INITIAL_MASS[index] + ten)
        stage.control.bounds.u[:] = (-1.1, 1.1)
        stage.path.bounds.unit_thrust = (1.0, 1.0)
        stage.path.bounds.radius = (R_e, None)
        stage.time.initial = (EDGES[index], EDGES[index])
        edge = EDGES[index + 1]
        stage.time.final = (t3, t4_max) if index == LAST else (edge, edge)

    # fixed component by component: one bound per row, each with its ends together
    stages[0].state.initial.r[:] = [(x, x) for x in launch]
    stages[0].state.initial.v[:] = [(x, x) for x in launch_velocity]
    stages[0].state.initial.m = (INITIAL_MASS[0], INITIAL_MASS[0])
    for index, stage in enumerate(stages[1:], start=1):
        mass = INITIAL_MASS[index]
        stage.state.initial.m = (mass, mass)

    bounds = problem.discrete.bounds
    bounds.stage_0_1_position[:] = (0.0, 0.0)
    bounds.stage_0_1_velocity[:] = (0.0, 0.0)
    bounds.stage_1_2_position[:] = (0.0, 0.0)
    bounds.stage_1_2_velocity[:] = (0.0, 0.0)
    bounds.stage_2_3_position[:] = (0.0, 0.0)
    bounds.stage_2_3_velocity[:] = (0.0, 0.0)
    problem.discrete.bounds.semi_major_axis = (a_f, a_f)
    problem.discrete.bounds.eccentricity = (e_f, e_f)
    problem.discrete.bounds.inclination = (i_f, i_f)
    problem.discrete.bounds.raan = (Omega_f, Omega_f)
    problem.discrete.bounds.argument_of_perigee = (omega_f, omega_f)


def _set_scales(problem, stages):
    """Condition the problem: say how large each quantity typically is."""
    for stage in stages:
        stage.state.scale.r[:] = length_scale
        stage.state.scale.v[:] = velocity_scale
        stage.state.scale.m = m_total
        stage.state.defect_scale.r[:] = length_scale
        stage.state.defect_scale.v[:] = velocity_scale
        stage.state.defect_scale.m = m_total
        stage.path.scale.unit_thrust = 1.0
        stage.path.scale.radius = length_scale
        stage.time.scale = time_scale
    scale = problem.discrete.scale
    scale.stage_0_1_position[:] = length_scale
    scale.stage_0_1_velocity[:] = velocity_scale
    scale.stage_1_2_position[:] = length_scale
    scale.stage_1_2_velocity[:] = velocity_scale
    scale.stage_2_3_position[:] = length_scale
    scale.stage_2_3_velocity[:] = velocity_scale
    problem.discrete.scale.semi_major_axis = length_scale


def _set_guess(stages):
    """Guess a continuous climb from the launch site to the target orbit."""
    final_position, final_velocity = oe_to_rv(a_f, e_f, i_f, Omega_f, omega_f, 0.0, mu)
    final_position = np.asarray(final_position, dtype=float)
    final_velocity = np.asarray(final_velocity, dtype=float)
    initial_position = np.array([R_e * cos(psi_l), 0.0, R_e * sin(psi_l)])
    initial_velocity = np.array([0.0, R_e * omega_e * cos(psi_l), 0.0])

    initial_radius, final_radius = np.linalg.norm(initial_position), np.linalg.norm(final_position)
    initial_latitude = np.arcsin(initial_position[2] / initial_radius)
    final_latitude = np.arcsin(final_position[2] / final_radius)
    initial_longitude = np.arctan2(initial_position[1], initial_position[0])
    final_longitude = np.arctan2(final_position[1], final_position[0])
    turn = np.arctan2(
        np.sin(final_longitude - initial_longitude), np.cos(final_longitude - initial_longitude)
    )

    for index, stage in enumerate(stages):
        start, end = EDGES[index], EDGES[index + 1]
        time = np.linspace(start, end, 9)
        fraction = time / t4_max
        latitude = initial_latitude + fraction * (final_latitude - initial_latitude)
        longitude = initial_longitude + fraction * turn
        radius = initial_radius + fraction * (final_radius - initial_radius)
        position = np.vstack(
            (
                radius * np.cos(latitude) * np.cos(longitude),
                radius * np.cos(latitude) * np.sin(longitude),
                radius * np.sin(latitude),
            )
        )
        velocity = (
            initial_velocity[:, None] + fraction * (final_velocity - initial_velocity)[:, None]
        )
        stage.time.guess = (start, end)
        stage.state.guess.r[:] = yapss.interp(time, position)
        stage.state.guess.v[:] = yapss.interp(time, velocity)
        stage.state.guess.m = yapss.interp(
            time, np.linspace(INITIAL_MASS[index], FINAL_MASS[index], len(time))
        )
        stage.control.guess.u[:] = yapss.interp(time, np.tile([[0.0], [1.0], [0.0]], (1, 9)))


def plot_solution(problem: yapss.Problem, solution: yapss.Solution) -> None:
    r"""Plot the ascent: altitude, position, velocity, mass, steering, and the Hamiltonian.

    Every quantity spans four phases, so each panel is a loop over them. The mass is the one
    that jumps, at each stage separation.

    Parameters
    ----------
    problem : yapss._api.Problem
        The problem that was solved, which carries the phase handles.
    solution : yapss._api.Solution
        The solution to plot.
    """
    stages = list(problem.phases)
    color = ("darkblue", "maroon", "darkorange")
    tf = solution[stages[LAST]].final.time

    def panel(series, ylabel, ylim=None, legend=None, colors=(0,)):
        """Plot one or more series over every stage."""
        plt.figure()
        for stage in stages:
            ps = solution[stage]
            for index, values in enumerate(series(ps)):
                plt.plot(ps.time, values, color[colors[index % len(colors)]])
        if legend:
            plt.legend(legend)
        plt.xlim(0, tf)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(r"Time, $t$ (s)")
        plt.ylabel(ylabel)
        plt.grid()
        plt.tight_layout()

    def magnitude(vector):
        """Return the Euclidean norm of a block field's three rows."""
        return np.sqrt(sum(vector[i] ** 2 for i in range(3)))

    panel(
        lambda ps: [(magnitude(ps.state.r) - R_e) / 1000],
        r"Altitude, $h$ (km)",
        ylim=[0, 250],
    )
    panel(
        lambda ps: [ps.state.r[i] / 1e6 for i in range(3)],
        "Position vector (1000 km)",
        ylim=(0, 6),
        legend=[r"$r_{1}(t)$", r"$r_{2}(t)$", r"$r_{3}(t)$"],
        colors=(0, 1, 2),
    )
    panel(
        lambda ps: [magnitude(ps.state.v)],
        r"Magnitude of inertial velocity, $v(t)$ (m/s)",
        ylim=[0, 12000],
    )
    panel(
        lambda ps: [ps.state.v[i] for i in range(3)],
        "Inertial velocity vector (m/s)",
        legend=[r"$v_{1}(t)$", r"$v_{2}(t)$", r"$v_{3}(t)$"],
        colors=(0, 1, 2),
    )
    panel(lambda ps: [ps.state.m / 1000], r"Vehicle mass, $m$ (1000 kg)", ylim=[0, 300])
    panel(
        lambda ps: [ps.control.u[i] for i in range(3)],
        r"Components of thrust direction, $u(t)$",
        ylim=[-0.8, 1.1],
        legend=[r"$u_{1}(t)$", r"$u_{2}(t)$", r"$u_{3}(t)$"],
        colors=(0, 1, 2),
    )
    panel(lambda ps: [ps.hamiltonian], r"Hamiltonian, $\lambda^T f$ (kg/s)")


def main() -> None:
    """Solve the Delta III ascent problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final mass = {solution.objective:.2f} kg")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
