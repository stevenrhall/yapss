"""End-to-end solves of Newton's minimal resistance problem through the redesigned API.

This is the example that names its independent variable: the phase runs over a radius, so it
is `r` in setup, in the callback, and in the solution, and there is no `time` anywhere.
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
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-8)


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
    ps = problem.solve()[problem.phases.nose]
    assert ps.initial.r > 0.3
    assert ps.final.r == pytest.approx(1.0)


def test_the_independent_variable_is_named_r(problem):
    solution = problem.solve()
    ps = solution[problem.phases.nose]
    assert ps.r.shape == ps.state.y.shape
    assert ps.r[0] == pytest.approx(0.0)
    assert ps.final.r == pytest.approx(1.0)
    assert ps.initial.r == pytest.approx(0.0)
    assert ps.duration == pytest.approx(1.0)


def test_there_is_no_time_anywhere(problem):
    # naming the independent variable replaces the name rather than adding to it
    with pytest.raises(AttributeError, match="phase 'nose' has no setting 'time'"):
        _ = problem.phases.nose.time
    ps = problem.solve()[problem.phases.nose]
    with pytest.raises(AttributeError, match="phase solution has no 'time'"):
        _ = ps.time
    with pytest.raises(AttributeError, match="has no 'time'"):
        _ = ps.final.time
    with pytest.raises(AttributeError, match="Did you mean 'r'"):
        _ = ps.final.rr


def test_the_callback_reads_the_independent_variable_by_name(problem):
    seen = {}

    def spy(arg, out):
        seen["r"] = np.shape(arg.r)
        yp = arg.state.yp
        out.dynamics.y = yp
        out.dynamics.yp = arg.control.u
        out.integrand.drag = 8 * arg.r / (1 + yp**2)
        return out

    problem.phases.nose.register.continuous(spy, replace=True)
    problem.derivatives.method = "central-difference"
    problem.solve()
    assert seen["r"][0] > 1


def test_the_declaration_names_what_it_holds():
    phases = Phases()
    assert [phase.name for phase in phases] == ["nose"]
    assert phases.nose._independent == "r"
