"""The Rosenbrock function, which is a problem with no phases and no constraints.

It is the smallest thing YAPSS will solve: two parameters and an objective. hs071 covers the
phase-less case with constraints; this one has none, so the discrete vector is empty too, and
the whole NLP is two columns and no rows.
"""

import pytest

from yapss.examples.rosenbrock import MINIMUM, Parameter, Phases, rosenbrock, setup


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_the_minimum_is_found(problem):
    solution = problem.solve()
    assert solution.parameter.x == pytest.approx(MINIMUM[0], abs=1e-8)
    assert solution.parameter.y == pytest.approx(MINIMUM[1], abs=1e-8)
    assert solution.objective == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(0.0, abs=1e-12)


def test_there_are_no_phases_and_no_constraints(problem):
    assert list(problem.phases) == []
    assert Phases._declared == {}
    assert Parameter._fields == ("x", "y")
    assert problem.solve().discrete._fields == ()


def test_the_function_is_the_one_the_objective_uses():
    """`rosenbrock` is shared by the callback and the contour plot, so one cannot drift."""
    assert rosenbrock(*MINIMUM) == 0.0
    assert rosenbrock(0.0, 0.0) == 1.0
