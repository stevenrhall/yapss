"""End-to-end solves of Newton's minimal resistance problem through the redesigned API.

The phase runs over a radius, which is the phase's `time`: every phase's independent variable
is called `time`, whatever it measures.
"""

import numpy as np
import pytest

from yapss.examples.newton import Phases, setup, setup2

RELEASED = 1.5033524160103926
"""What the same problem gives through the released API."""

RELEASED2 = 1.4992639203593585
"""What `setup2`, the alternate formulation, gives through the released API."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_agrees_with_the_released_api(problem):
    # Not to the digits RELEASED was recorded with: `setup` leaves one polynomial to fit a
    # corner (see the example's docstring), and where Ipopt stops on that is roundoff. CI's
    # platforms spread over 1.50335226-1.50335256, about 1e-7 either side of the macOS value.
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-6)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-6)


def test_the_alternate_formulation_agrees_with_the_released_api():
    """`setup2` optimizes the radius of the flat tip rather than fixing it at zero."""
    problem = setup2()
    problem.ipopt_options.print_level = 0
    assert problem.solve().objective == pytest.approx(RELEASED2, rel=1e-12)


def test_the_alternate_formulation_frees_the_initial_radius():
    """Its whole point: r0 becomes a variable, and the optimum puts it well away from zero."""
    problem = setup2()
    problem.ipopt_options.print_level = 0
    ps = problem.solve().phases[problem.phases.phase]
    assert ps.initial.time > 0.3
    assert ps.final.time == pytest.approx(1.0)


def test_the_radius_is_the_phase_s_time(problem):
    solution = problem.solve()
    ps = solution.phases[problem.phases.phase]
    assert ps.time.shape == ps.state.y.shape
    assert ps.time[0] == pytest.approx(0.0)
    assert ps.final.time == pytest.approx(1.0)
    assert ps.initial.time == pytest.approx(0.0)
    assert ps.duration == pytest.approx(1.0)


def test_a_misspelled_endpoint_name_is_answered(problem):
    ps = problem.solve().phases[problem.phases.phase]
    with pytest.raises(AttributeError, match="Did you mean 'time'"):
        _ = ps.final.tim


def test_the_callback_reads_the_independent_variable_by_name(problem):
    seen = {}

    def spy(arg, out):
        seen["r"] = np.shape(arg.time)
        yp = arg.state.yp
        out.dynamics.y = yp
        out.dynamics.yp = arg.control.u
        out.integrand.drag = 8 * arg.time / (1 + yp**2)

    problem.phases.phase.register.continuous(spy)
    problem.derivatives.method = "central-difference"
    problem.solve()
    assert seen["r"][0] > 1


def test_the_declaration_names_what_it_holds():
    phases = Phases()
    assert [phase.name for phase in phases] == ["phase"]
    assert phases.phase._independent == "time"
