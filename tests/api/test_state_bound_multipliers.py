"""The multipliers of a state's bounds and of an integral's bounds.

A state's general bound has a density, ``ps.multiplier.state``, read as if the state had no
initial or final bound; each end has a multiplier per row, ``ps.multiplier.initial_state`` and
``final_state``, read as if it had no general bound. At an end the NLP bounds the state by the
tighter of the two, so one multiplier is there, and it belongs to whichever side is active.

What the multipliers mean is checked the way every multiplier here is: against the objective's
sensitivity to the bound, from a central difference of two solves. The problem is the
brachistochrone with a floor, ``y <= b`` (``y`` positive down), which the bead reaches and
rides to the end: a state bound active over an interval that includes the final point.
"""

import pytest

import yapss
from yapss.math import cos, sin

METHODS = ("lgl", "lgr", "lg")
DB = 1e-5


class State(yapss.State):
    x = yapss.scalar()
    y = yapss.scalar()
    v = yapss.scalar()


class Control(yapss.Control):
    u = yapss.scalar()


class Integral(yapss.Integral):
    effort = yapss.scalar()


class Phase(yapss.Phase):
    state: State
    control: Control
    integral: Integral


class Phases(yapss.Phases):
    phase: Phase


class Floor(yapss.Problem):
    phases: Phases


def floor(method, b=0.45, xf=1.0, effort=None):
    """The brachistochrone with a floor at `b`, ending at `xf`, its effort bounded above."""
    problem = Floor("floor")
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(arg, out):
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = 32.174 * sin(u)
        out.integrand.effort = u**2

    @problem.register.objective
    def objective(arg):
        return arg[ph].final_time

    ph.time.initial = (0.0, 0.0)
    for field in (ph.state.x, ph.state.y, ph.state.v):
        field.initial = (0.0, 0.0)
    ph.state.x.final = (xf, xf)
    ph.state.y.bounds = (None, b)
    ph.control.u.bounds = (-2.0, 2.0)
    ph.integral.effort.bounds = (None, 1e3 if effort is None else effort)
    ph.time.guess = (0.0, 1.0)
    ph.state.x.guess = (0.0, 1.0)
    ph.state.y.guess = (0.0, 0.3)
    ph.state.v.guess = (0.0, 5.0)
    ph.mesh = yapss.Mesh.uniform(segments=10, points=10)
    problem.spectral_method = method
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.tol = 1e-12
    return problem


def sensitivity(method, **bound):
    """Return dJ/d(bound) by a central difference, for the one keyword given."""
    ((name, value),) = bound.items()
    up = floor(method, **{name: value + DB}).solve().objective
    down = floor(method, **{name: value - DB}).solve().objective
    return (up - down) / (2 * DB)


@pytest.mark.parametrize(
    "method",
    [
        "lgl",
        pytest.param(
            "lgr",
            marks=pytest.mark.xfail(
                strict=True,
                reason="the NLP bounds the state at the final point, which LGR does not "
                "collocate; its multiplier has no weight in the quadrature",
            ),
        ),
        pytest.param(
            "lg",
            marks=pytest.mark.xfail(
                strict=True,
                reason="the NLP bounds the state at the segment ends, which LG does not "
                "collocate; their multipliers have no weight in the quadrature",
            ),
        ),
    ],
)
def test_the_density_s_integral_is_the_sensitivity_to_the_bound(method):
    ps = floor(method).solve().phases.phase
    integral = ps.weights @ ps.multiplier.state.y
    assert -integral == pytest.approx(sensitivity(method, b=0.45), rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_the_general_bound_s_multipliers_sum_to_the_sensitivity(method):
    """Every stored point the bound acts on, collocated or not, read from the solver's record."""
    solution = floor(method).solve()
    ps = solution.phases.phase
    nlp = solution.nlp
    rows = ps.nlp.index.variable.state.y
    total = (nlp.mult_x_U - nlp.mult_x_L)[rows][1:].sum()  # the first point is y(0) = 0
    assert -total == pytest.approx(sensitivity(method, b=0.45), rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_a_fixed_end_value_s_multiplier_is_its_sensitivity(method):
    """x(tf) = xf has no general bound, so its end multiplier is the whole of what is there."""
    ps = floor(method).solve().phases.phase
    assert -ps.multiplier.final_state.x == pytest.approx(sensitivity(method, xf=1.0), rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_an_end_bound_takes_none_of_an_active_general_bound(method):
    """y ends on the floor, which is its general bound; y has no final bound of its own."""
    ps = floor(method).solve().phases.phase
    assert ps.multiplier.final_state.y == 0.0
    assert ps.multiplier.initial_state.y != 0.0  # y(0) = 0 is held by its initial bound


@pytest.mark.parametrize("method", METHODS)
def test_the_general_bound_takes_none_of_an_active_end_bound(method):
    """y(0) = 0 is held by the initial bound, not the floor, so the density does not see it."""
    ps = floor(method).solve().phases.phase
    first = ps.multiplier.state.y[0]
    assert abs(first) < 1e-3 * abs(ps.multiplier.initial_state.y) or not ps.collocated[0]


@pytest.mark.parametrize("method", METHODS)
def test_an_integral_bound_s_multiplier_is_its_sensitivity(method):
    """The effort bounded below its free value, so the bound is active."""
    ps = floor(method, effort=0.2).solve().phases.phase
    assert ps.integral.effort == pytest.approx(0.2, rel=1e-7)  # within Ipopt's bound relaxation
    expected = sensitivity(method, effort=0.2)
    assert -ps.multiplier.integral_bound.effort == pytest.approx(expected, rel=1e-5)
