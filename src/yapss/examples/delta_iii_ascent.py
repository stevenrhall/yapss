"""

The Delta III ascent problem: put as much mass as possible into a target orbit.

A four-stage launch vehicle rises from Cape Canaveral into a geosynchronous transfer orbit.
Each stage is a phase, with its own thrust and mass flow, and the phases are joined by
continuity of position and velocity -- but not of mass, which jumps when a stage is dropped.
The trajectory ends on five of the six classical orbital elements.

The problem is originally due to Benson:

    David Benson. A Gauss pseudospectral transcription for optimal control. PhD thesis,
    Massachusetts Institute of Technology, 2005. https://hdl.handle.net/1721.1/28919.

The atmosphere is the one part of the model that `setup` lets the caller choose, to show what
to do with a function that ``"auto"`` cannot trace. Benson's atmosphere is an exponential,
which ``"auto"`` traces like everything else. The same exponential can be hidden from it behind
`yapss.math.external`, with its derivatives supplied or left to be differenced, and the answer
does not change. The last choice is the case the wrapper is for: a standard atmosphere from a
library, which has no formula to trace.

"""

# The orbital elements are conventionally capitalized.
# ruff: noqa: N803, N806

__all__ = ["main", "plot_solution", "setup"]

# standard library imports
from collections.abc import Callable, Sequence
from typing import Any

# third-party imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# package imports
import yapss
from yapss.math import arccos, arcsin, arctan2, cos, exp, external, pi, sin, sqrt

# the Earth, the atmosphere and the launch site

mu = 3.986012e14
"""Earth's gravitational parameter (m^3/s^2)."""
R_e = 6378145.0
"""Earth's radius (m)."""
g0 = 9.80665
"""Sea-level gravity (m/s^2)."""
h0 = 7200.0
"""Density scale height of the atmosphere (m)."""
rho0 = 1.225
"""Sea-level air density (kg/m^3)."""
omega_e = 7.29211585e-5
"""Earth's rotation rate (rad/s)."""
CD = 0.5
"""Drag coefficient."""
S = 4 * pi
"""Aerodynamic reference area (m^2)."""
psi_l = 28.5 * pi / 180.0
"""Latitude of the launch site (rad)."""

# the vehicle: nine solid boosters, a first stage, a second stage, and the payload

pi_s, pi_1, pi_2, pi_p = 19290.0, 104380.0, 19300.0, 4164.0
"""Total mass of one booster, the first stage, the second stage, and the payload (kg)."""
rho_s, rho_1, rho_2 = 17010.0, 95550.0, 16820.0
"""Propellant mass of one booster, the first stage, and the second stage (kg)."""
phi_s, phi_1 = pi_s - rho_s, pi_1 - rho_1
"""Dry mass of one booster and of the first stage (kg)."""
Ts, T1, T2 = 628500.0, 1083100.0, 110094.0
"""Thrust of one booster, the first stage, and the second stage (N)."""
tau_s, tau_1, tau_2 = 75.2, 261.0, 700.0
"""Burn time of a booster, the first stage, and the second stage (s)."""
# specific impulse of a booster, the first stage, and the second stage (s)
Is = Ts * tau_s / (rho_s * g0)
I1 = T1 * tau_1 / (rho_1 * g0)
I2 = T2 * tau_2 / (rho_2 * g0)

t0, t1, t2, t3 = 0.0, 75.2, 150.4, 261.0
"""The time at which each stage begins (s)."""
t4_max = t3 + tau_2
"""The latest the flight may end (s)."""

# Stage 0 burns six boosters and the first stage; stage 1 drops the six spent boosters and
# burns the other three; stage 2 drops those and burns the first stage alone; stage 3 drops the
# first stage and burns the second.
mi_0 = 9 * pi_s + pi_1 + pi_2 + pi_p
mf_0 = mi_0 - 6 * rho_s - tau_s / tau_1 * rho_1
mi_1 = mf_0 - 6 * phi_s
mf_1 = mi_1 - 3 * rho_s - tau_s / tau_1 * rho_1
mi_2 = mf_1 - 3 * phi_s
mf_2 = mi_2 - (1 - 2 * tau_s / tau_1) * rho_1
mi_3 = mf_2 - phi_1

a_f, e_f, i_f, Omega_f, omega_f = 24361140, 0.7308, 28.5, 269.8, 130.5
"""The target orbit: semi-major axis (m), eccentricity, inclination, right ascension of the
ascending node, and argument of perigee (degrees)."""

m_total = mi_0
"""Lift-off mass (kg), which also scales the objective."""
length_scale = R_e
velocity_scale = sqrt(mu / R_e)
time_scale = length_scale / velocity_scale
altitude_scale = 32_000.0
"""The scale of the radius path constraint: its range, not its size.

The radius varies by a few hundred kilometres over the ascent, a hundredth of its value.
Scaled by the radius, a violation of that constraint of several kilometres looks negligible to
Ipopt, and its early iterates pass below the ground, where the exponential atmosphere grows
without bound. Scaled by the altitude, the iterates stay above it, and the solve takes about a
third of the iterations.
"""
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


class State(yapss.State):
    """Vehicle position, velocity, and mass."""

    r = yapss.vector(3)
    """Position."""
    v = yapss.vector(3)
    """Velocity."""
    m = yapss.scalar()
    """Mass."""


class Control(yapss.Control):
    """Thrust direction, path constrained to be a unit vector."""

    u = yapss.vector(3)
    """Thrust vector direction."""


class Path(yapss.Path):
    """Path constraints."""

    unit_thrust = yapss.scalar()
    """The thrust vector direction is constrained to be a unit vector."""
    radius = yapss.scalar()
    """The vehicle is constrained to be above the surface of the Earth."""


class Discrete(yapss.Discrete):
    """Continuity where the stages meet, and the orbit that must be reached.

    Position and velocity are separate groups, rather than one block of six, because they
    are scaled differently; the same reason separates the semi-major axis from the angles.
    """

    stage_0_1_position = yapss.vector(3)
    """Position continuity between stages 0 and 1."""
    stage_0_1_velocity = yapss.vector(3)
    """Velocity continuity between stages 0 and 1."""
    stage_1_2_position = yapss.vector(3)
    """Position continuity between stages 1 and 2."""
    stage_1_2_velocity = yapss.vector(3)
    """Velocity continuity between stages 1 and 2."""
    stage_2_3_position = yapss.vector(3)
    """Position continuity between stages 2 and 3."""
    stage_2_3_velocity = yapss.vector(3)
    """Velocity continuity between stages 2 and 3."""
    semi_major_axis = yapss.scalar()
    """Semi-major axis for the target orbit."""
    eccentricity = yapss.scalar()
    """Eccentricity for the target orbit."""
    inclination = yapss.scalar()
    """Inclination for the target orbit."""
    raan = yapss.scalar()
    """Right ascension of ascending node for the target orbit."""
    argument_of_perigee = yapss.scalar()
    """Argument of perigee for the target orbit."""


class Stage(yapss.Phase):
    """One stage's burn; the four stages share this shape."""

    state: State
    control: Control
    path: Path


class Phases(yapss.Phases):
    """One phase per stage."""

    stage_0: Stage
    stage_1: Stage
    stage_2: Stage
    stage_3: Stage


class DeltaIII(yapss.Problem):
    """The Delta III launch to orbit, in four stages."""

    phases: Phases
    discrete: Discrete


# Names for the types the annotations below use.
StageArg = yapss.ContinuousArg[State, Control]
StageOut = yapss.ContinuousOut[State, Path]


Vector3 = NDArray[Any] | Sequence[Any]
"""A 3-vector: a vector field's value, or a list of its components."""

Continuous = Callable[[StageArg, StageOut], None]
"""The type of a stage's continuous callback."""

Density = Callable[[Any], Any]
"""The type of an atmosphere: the air density (kg/m^3) at an altitude (m)."""

ATMOSPHERES = ("exponential", "supplied", "differenced", "icao")
"""The atmospheres `setup` offers."""


def exponential_density(h: Any) -> Any:
    """Return the density of the exponential atmosphere at altitude `h`.

    Written with `yapss.math.exp`, so ``"auto"`` traces it with the rest of the model.
    """
    return rho0 * exp(-h / h0)


def numpy_density(h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the same density, written with NumPy: a function ``"auto"`` cannot trace."""
    return rho0 * np.exp(-h / h0)


def numpy_density_slope(h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the first derivative of `numpy_density`."""
    return -numpy_density(h) / h0


def numpy_density_curvature(h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the second derivative of `numpy_density`."""
    return numpy_density(h) / h0**2


def icao_density() -> Density:
    """Return the density of the ICAO standard atmosphere, from the ``ambiance`` package.

    ``ambiance`` covers altitudes up to 80 km, and the vehicle climbs well above that, so the
    density is continued upward from its value at 80 km with the scale height of the
    exponential atmosphere. Drag is negligible there.

    Returns
    -------
    callable
        The density, wrapped for use in a callback.

    Raises
    ------
    ImportError
        If ``ambiance`` is not installed. YAPSS does not install it.
    """
    try:
        from ambiance import Atmosphere  # noqa: PLC0415  -- optional, and this is its one use
    except ImportError as error:
        msg = (
            'The "icao" atmosphere uses the ambiance package, which is not installed and which '
            'YAPSS does not install. Install it with "pip install ambiance".'
        )
        raise ImportError(msg) from error

    top = 80_000.0
    density_at_top = float(Atmosphere(top).density[0])

    @external(scale=h0, vectorized=True)
    def density(h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the ICAO standard density at altitudes `h`, continued above 80 km."""
        rho = np.empty_like(h)
        above = h >= top
        rho[above] = density_at_top * np.exp(-(h[above] - top) / h0)
        if not above.all():  # ambiance refuses an empty array
            # below the model's floor the density is held at its value there
            rho[~above] = Atmosphere(np.maximum(h[~above], -5000.0)).density
        return rho

    return density


def make_density(atmosphere: str) -> Density:
    """Return the density function of the named atmosphere.

    Parameters
    ----------
    atmosphere : str
        One of `ATMOSPHERES`. See `setup`.

    Returns
    -------
    callable
        The density (kg/m^3) at an altitude (m), for use in a callback.

    Raises
    ------
    ValueError
        If `atmosphere` is not one of `ATMOSPHERES`.
    ImportError
        If `atmosphere` is ``"icao"`` and ``ambiance`` is not installed.
    """
    if atmosphere == "exponential":
        return exponential_density
    if atmosphere == "supplied":
        return external(
            numpy_density,
            jacobian=numpy_density_slope,
            hessian=numpy_density_curvature,
            scale=h0,
            vectorized=True,
        )
    if atmosphere == "differenced":
        return external(numpy_density, scale=h0, vectorized=True)
    if atmosphere == "icao":
        return icao_density()
    msg = f"atmosphere must be one of {ATMOSPHERES}; got {atmosphere!r}"
    raise ValueError(msg)


def make_dynamics(thrust: float, mass_flow: float, density: Density) -> Continuous:
    """Return the continuous callback of a stage with the given thrust and mass flow."""

    def continuous(arg: StageArg, out: StageOut) -> None:
        """Compute the vehicle's dynamics and the constraints that hold along the way."""
        r_vec, v_vec, m = arg.state.r, arg.state.v, arg.state.m
        u_vec = arg.control.u

        r = (r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2) ** 0.5
        rho = density(r - R_e)
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

    return continuous


def cross(x1: Vector3, x2: Vector3) -> list[Any]:
    """Return the cross product of two 3-vectors."""
    x3: list[Any] = [0, 0, 0]
    x3[0] = x1[1] * x2[2] - x1[2] * x2[1]
    x3[1] = x1[2] * x2[0] - x1[0] * x2[2]
    x3[2] = x1[0] * x2[1] - x1[1] * x2[0]
    return x3


def mag(x: Vector3) -> Any:
    """Return the magnitude of a vector, kept away from zero so its derivative exists."""
    return (sum(xi**2 for xi in x) + 1e-100) ** 0.5


def dot(x1: Vector3, x2: Vector3) -> Any:
    """Return the dot product of two 3-vectors."""
    return sum(x1[i] * x2[i] for i in range(3))


def oe_to_rv(  # noqa: PLR0913, PLR0917 -- the six elements
    a: float, e: float, i: float, Omega: float, omega: float, nu: float, mu_: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert classical orbital elements to inertial position and velocity.

    Parameters
    ----------
    a : float
        Semi-major axis.
    e : float
        Eccentricity.
    i : float
        Inclination (degrees).
    Omega : float
        Right ascension of the ascending node (degrees).
    omega : float
        Argument of periapsis (degrees).
    nu : float
        True anomaly (rad).
    mu_ : float
        Gravitational parameter.

    Returns
    -------
    tuple of numpy.ndarray
        The inertial position and velocity.
    """
    p = a * (1 - e**2)
    r = p / (1 + e * cos(nu))
    r_vec = np.array([r * cos(nu), r * sin(nu), 0])
    v_vec = sqrt(mu_ / p) * np.array([-sin(nu), e + cos(nu), 0])
    deg_to_rad = pi / 180
    c_O = cos(deg_to_rad * Omega)
    s_O = sin(deg_to_rad * Omega)
    c_o = cos(deg_to_rad * omega)
    s_o = sin(deg_to_rad * omega)
    c_i = cos(deg_to_rad * i)
    s_i = sin(deg_to_rad * i)
    R = np.array(
        [
            [c_O * c_o - s_O * s_o * c_i, -c_O * s_o - s_O * c_o * c_i, +s_O * s_i],
            [s_O * c_o + c_O * s_o * c_i, -s_O * s_o + c_O * c_o * c_i, -c_O * s_i],
            [s_o * s_i, c_o * s_i, c_i],
        ],
    )
    return R @ r_vec, R @ v_vec


def orbital_elements(r_vec: Vector3, v_vec: Vector3) -> tuple[Any, ...]:
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


def setup(atmosphere: str = "exponential") -> DeltaIII:
    """Set up the Delta III ascent problem.

    Parameters
    ----------
    atmosphere : str, default "exponential"
        How the air density is computed:

        - ``"exponential"``: an exponential atmosphere, traced by ``"auto"``.
        - ``"supplied"``: the same exponential, hidden from ``"auto"`` behind
          `yapss.math.external`, with its first and second derivatives supplied.
        - ``"differenced"``: the same again, with its derivatives left to be differenced.
        - ``"icao"``: the ICAO standard atmosphere of the ``ambiance`` package, which is not
          installed with YAPSS.

        The first three are the same model and give the same answer.

    Returns
    -------
    DeltaIII
        The problem.

    Raises
    ------
    ValueError
        If `atmosphere` is not one of those.
    ImportError
        If `atmosphere` is ``"icao"`` and ``ambiance`` is not installed.
    """
    density = make_density(atmosphere)
    problem = DeltaIII("Delta III ascent")
    phases = problem.phases
    stages = [phases.stage_0, phases.stage_1, phases.stage_2, phases.stage_3]

    for stage, thrust, mass_flow in zip(stages, THRUST, MASS_FLOW, strict=True):
        stage.register.continuous(make_dynamics(thrust, mass_flow, density))

    @problem.register.objective
    def objective(arg: yapss.DiscreteArg) -> Any:
        """Return the mass delivered to orbit, which is to be made as large as possible."""
        return arg[stages[LAST]].final_state.m

    @problem.register.discrete
    def discrete(arg: yapss.DiscreteArg, out: yapss.DiscreteOut[Discrete]) -> None:
        """Join the stages, and require the final state to be on the target orbit."""
        s0, s1, s2, s3 = (arg[stage] for stage in stages)
        out.discrete.stage_0_1_position = s1.initial_state.r - s0.final_state.r
        out.discrete.stage_0_1_velocity = s1.initial_state.v - s0.final_state.v
        out.discrete.stage_1_2_position = s2.initial_state.r - s1.final_state.r
        out.discrete.stage_1_2_velocity = s2.initial_state.v - s1.final_state.v
        out.discrete.stage_2_3_position = s3.initial_state.r - s2.final_state.r
        out.discrete.stage_2_3_velocity = s3.initial_state.v - s2.final_state.v
        final = s3.final_state
        a, e, i, Omega, omega = orbital_elements(final.r, final.v)
        out.discrete.semi_major_axis = a
        out.discrete.eccentricity = e
        out.discrete.inclination = i
        out.discrete.raan = Omega
        out.discrete.argument_of_perigee = omega

    problem.objective.sense = "maximize"
    problem.objective.scale = m_total

    _set_bounds(problem, stages)
    _set_scales(problem, stages)
    _set_guess(stages)

    for stage in stages:
        stage.mesh = yapss.Mesh.uniform(segments=5, points=5)
    problem.spectral_method = "lg"
    problem.derivatives.method = "auto"
    problem.derivatives.order = "second"
    problem.ipopt_options.max_iter = 1000
    problem.ipopt_options.print_level = 3
    return problem


def _set_bounds(problem: DeltaIII, stages: list[Stage]) -> None:
    """Set the bounds on every stage, and the bounds the constraints must meet."""
    launch = [R_e * cos(psi_l), 0.0, R_e * sin(psi_l)]
    launch_velocity = [0.0, R_e * omega_e * cos(psi_l), 0.0]

    for index, stage in enumerate(stages):
        stage.state.r.bounds[:] = (-r_max, r_max)
        stage.state.v.bounds[:] = (-v_max, v_max)
        stage.state.r.initial[:] = (-r_max, r_max)
        stage.state.v.initial[:] = (-v_max, v_max)
        stage.state.r.final[:] = (-r_max, r_max)
        stage.state.v.final[:] = (-v_max, v_max)
        stage.state.m.bounds = (FINAL_MASS[index] - ten, INITIAL_MASS[index] + ten)
        # The last stage may not deliver less than the payload itself, so its final mass has
        # no leeway below.
        floor = pi_p if index == LAST else FINAL_MASS[index] - ten
        stage.state.m.final = (floor, INITIAL_MASS[index] + ten)
        stage.control.u.bounds[:] = (-1.1, 1.1)
        stage.path.unit_thrust.bounds = (1.0, 1.0)
        stage.path.radius.bounds = (R_e, None)
        stage.time.initial = (EDGES[index], EDGES[index])
        edge = EDGES[index + 1]
        stage.time.final = (t3, t4_max) if index == LAST else (edge, edge)

    # fixed component by component: one bound per row, each with its ends together
    stages[0].state.r.initial[:] = [(x, x) for x in launch]
    stages[0].state.v.initial[:] = [(x, x) for x in launch_velocity]
    stages[0].state.m.initial = (INITIAL_MASS[0], INITIAL_MASS[0])
    for index, stage in enumerate(stages[1:], start=1):
        mass = INITIAL_MASS[index]
        stage.state.m.initial = (mass, mass)

    discrete = problem.discrete
    discrete.stage_0_1_position.bounds[:] = (0.0, 0.0)
    discrete.stage_0_1_velocity.bounds[:] = (0.0, 0.0)
    discrete.stage_1_2_position.bounds[:] = (0.0, 0.0)
    discrete.stage_1_2_velocity.bounds[:] = (0.0, 0.0)
    discrete.stage_2_3_position.bounds[:] = (0.0, 0.0)
    discrete.stage_2_3_velocity.bounds[:] = (0.0, 0.0)
    problem.discrete.semi_major_axis.bounds = (a_f, a_f)
    problem.discrete.eccentricity.bounds = (e_f, e_f)
    problem.discrete.inclination.bounds = (i_f, i_f)
    problem.discrete.raan.bounds = (Omega_f, Omega_f)
    problem.discrete.argument_of_perigee.bounds = (omega_f, omega_f)


def _set_scales(problem: DeltaIII, stages: list[Stage]) -> None:
    """Condition the problem: say how large each quantity typically is."""
    for stage in stages:
        stage.state.r.scale[:] = length_scale
        stage.state.v.scale[:] = velocity_scale
        stage.state.m.scale = m_total
        stage.path.unit_thrust.scale = 1.0
        stage.path.radius.scale = altitude_scale
        stage.time.scale = time_scale
    discrete = problem.discrete
    discrete.stage_0_1_position.scale[:] = length_scale
    discrete.stage_0_1_velocity.scale[:] = velocity_scale
    discrete.stage_1_2_position.scale[:] = length_scale
    discrete.stage_1_2_velocity.scale[:] = velocity_scale
    discrete.stage_2_3_position.scale[:] = length_scale
    discrete.stage_2_3_velocity.scale[:] = velocity_scale
    problem.discrete.semi_major_axis.scale = length_scale


def _set_guess(stages: list[Stage]) -> None:
    """Guess a continuous climb from the launch site to the target orbit."""
    final_position, final_velocity = oe_to_rv(a_f, e_f, i_f, Omega_f, omega_f, 0.0, mu)
    final_position = np.asarray(final_position, dtype=float)
    final_velocity = np.asarray(final_velocity, dtype=float)
    initial_position = np.array([R_e * cos(psi_l), 0.0, R_e * sin(psi_l)])
    initial_velocity = np.array([0.0, R_e * omega_e * cos(psi_l), 0.0])

    initial_radius, final_radius = np.linalg.norm(initial_position), np.linalg.norm(final_position)
    initial_latitude = arcsin(initial_position[2] / initial_radius)
    final_latitude = arcsin(final_position[2] / final_radius)
    initial_longitude = arctan2(initial_position[1], initial_position[0])
    final_longitude = arctan2(final_position[1], final_position[0])
    turn = arctan2(
        sin(final_longitude - initial_longitude), cos(final_longitude - initial_longitude)
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
                radius * cos(latitude) * cos(longitude),
                radius * cos(latitude) * sin(longitude),
                radius * sin(latitude),
            )
        )
        velocity = (
            initial_velocity[:, None] + fraction * (final_velocity - initial_velocity)[:, None]
        )
        stage.time.guess = (start, end)
        stage.state.r.guess[:] = yapss.interp(time, position)
        stage.state.v.guess[:] = yapss.interp(time, velocity)
        stage.state.m.guess = yapss.interp(
            time, np.linspace(INITIAL_MASS[index], FINAL_MASS[index], len(time))
        )
        stage.control.u.guess[:] = yapss.interp(time, np.tile([[0.0], [1.0], [0.0]], (1, 9)))


def plot_solution(problem: DeltaIII, solution: yapss.Solution) -> None:
    r"""Plot the ascent: altitude, position, velocity, mass, steering, and the Hamiltonian.

    Every quantity spans four phases, so each figure is a loop over them. The mass is the one
    that jumps, at each stage separation.

    Parameters
    ----------
    problem : DeltaIII
        The problem that was solved, which carries the phase handles.
    solution : yapss.Solution
        The solution to plot.
    """
    stages = [solution.phases[ph] for ph in problem.phases]
    color = ("darkblue", "maroon", "darkorange")

    altitude = plt.figure()
    for ps in stages:
        plt.plot(ps.time, (mag(ps.state.r) - R_e) / 1000, color[0])
    plt.ylabel(r"Altitude, $h$ (km)")
    plt.ylim(0, 250)

    position = plt.figure()
    for ps in stages:
        for i in range(3):
            plt.plot(ps.time, ps.state.r[i] / 1e6, color[i])
    plt.ylabel("Position vector (1000 km)")
    plt.ylim(0, 6)
    plt.legend([r"$r_{1}(t)$", r"$r_{2}(t)$", r"$r_{3}(t)$"])

    speed = plt.figure()
    for ps in stages:
        plt.plot(ps.time, mag(ps.state.v), color[0])
    plt.ylabel(r"Magnitude of inertial velocity, $v(t)$ (m/s)")
    plt.ylim(0, 12000)

    velocity = plt.figure()
    for ps in stages:
        for i in range(3):
            plt.plot(ps.time, ps.state.v[i], color[i])
    plt.ylabel("Inertial velocity vector (m/s)")
    plt.legend([r"$v_{1}(t)$", r"$v_{2}(t)$", r"$v_{3}(t)$"])

    mass = plt.figure()
    for ps in stages:
        plt.plot(ps.time, ps.state.m / 1000)
    plt.ylabel(r"Vehicle mass, $m$ (1000 kg)")
    plt.ylim(0, 300)

    steering = plt.figure()
    for ps in stages:
        for i in range(3):
            plt.plot(ps.time, ps.control.u[i], color[i])
    plt.ylabel(r"Components of thrust direction, $u(t)$")
    plt.ylim(-0.8, 1.1)
    plt.legend([r"$u_{1}(t)$", r"$u_{2}(t)$", r"$u_{3}(t)$"])

    hamiltonian = plt.figure()
    for ps in stages:
        plt.plot(ps.time, ps.hamiltonian, color[0])
    plt.ylabel(r"Hamiltonian, $\lambda^T f$ (kg/s)")

    for figure in (altitude, position, speed, velocity, mass, steering, hamiltonian):
        axes = figure.gca()
        axes.set_xlim(0, 1000)
        axes.set_xlabel(r"Time, $t$ (s)")
        axes.grid()
        figure.tight_layout()


def main() -> None:
    """Solve the Delta III ascent problem and plot the solution."""
    problem = setup()
    solution = problem.solve()
    print(f"final mass = {solution.objective:.2f} kg")
    plot_solution(problem, solution)
    plt.show()


if __name__ == "__main__":
    main()
