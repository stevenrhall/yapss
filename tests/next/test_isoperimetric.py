"""The isoperimetric problem, whose independent variable is an arc length.

Two things make it worth having. It is the only problem in the corpus with more than one
integral, and two of the three are bounded rather than read; and its phase runs over `s` rather
than `time`, so the name appears on the phase, in the callback argument and on the solution --
the API has no notion that the independent variable is ever a time.

The answer is known in closed form, which makes this the accuracy check of the corpus: the
pseudospectral method should reach 1/(4*pi) to near machine precision on 36 collocation points.
"""

import math

import numpy as np
import pytest

from yapss.examples.isoperimetric import AREA, PERIMETER, setup


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_finds_the_area_of_the_circle(problem):
    assert problem.solve().objective == pytest.approx(AREA, rel=1e-12)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    """The example's `tol` of 1e-14 is for exact derivatives, so the others are given their own.

    A central-difference gradient carries an error near the square root of machine epsilon, so
    Ipopt cannot satisfy a 1e-14 KKT test with one and stops on a too-small search direction.
    That is the tolerance being unreachable rather than the answer being wrong, so the test
    asks for a tolerance the method can meet and still checks the area it arrives at.
    """
    problem.derivatives.method = method
    if method != "auto":
        problem.ipopt_options.tol = 1e-10
    assert problem.solve().objective == pytest.approx(AREA, rel=1e-6)


def test_the_curve_is_a_circle_of_the_right_radius(problem):
    """Every point the same distance from the origin, which the moments put there."""
    ps = problem.solve()[problem.phases.curve]
    radius = np.hypot(ps.state.x, ps.state.y)
    assert radius.std() < 1e-9
    assert radius.mean() == pytest.approx(PERIMETER / (2 * math.pi), rel=1e-9)


def test_the_curve_closes_and_the_centroid_is_at_the_origin(problem):
    solution = problem.solve()
    assert solution.discrete.closure_x == pytest.approx(0.0, abs=1e-9)
    assert solution.discrete.closure_y == pytest.approx(0.0, abs=1e-9)
    ps = solution[problem.phases.curve]
    assert ps.integral.x_moment == pytest.approx(0.0, abs=1e-9)
    assert ps.integral.y_moment == pytest.approx(0.0, abs=1e-9)


def test_the_independent_variable_is_named_s(problem):
    """`time` is only the default name, and nothing here uses it."""
    ph = problem.phases.curve
    assert hasattr(ph, "s")
    assert not hasattr(ph, "time")
    ps = problem.solve()[ph]
    assert ps.s[0] == pytest.approx(0.0)
    assert ps.s[-1] == pytest.approx(PERIMETER)


def test_the_speed_is_one_all_along(problem):
    """Which is what makes the independent variable an arc length rather than a parameter."""
    ps = problem.solve()[problem.phases.curve]
    np.testing.assert_allclose(ps.path.speed_squared, 1.0, atol=1e-9)
