"""A phase's duration bounds reach the solver, and the solution reports their multiplier."""

import pytest

from yapss.examples.brachistochrone_minimal import setup


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_an_inactive_duration_bound_leaves_the_solution_alone(problem):
    free = problem.solve()
    problem.phases.phase.duration.bounds = (0.0, 10.0)
    bounded = problem.solve()
    assert bounded.objective == pytest.approx(free.objective, rel=1e-8)
    assert bounded.phases.phase.multiplier.duration == pytest.approx(0.0, abs=1e-6)


def test_an_active_lower_bound_holds_the_duration_there(problem):
    """The minimum-time slide is shorter than 0.5, so a lower bound of 0.5 is active."""
    problem.phases.phase.duration.bounds = (0.5, None)
    solution = problem.solve()
    ps = solution.phases.phase
    assert ps.duration == pytest.approx(0.5, rel=1e-6)
    assert solution.objective == pytest.approx(0.5, rel=1e-6)
    # the objective is the duration itself, so its sensitivity to the bound is one
    assert abs(ps.multiplier.duration) == pytest.approx(1.0, rel=1e-6)


def test_the_bound_is_read_when_the_problem_is_solved(problem):
    """A snapshot: changing the bound after a solve does not alter that solve's record."""
    problem.phases.phase.duration.bounds = (0.5, None)
    first = problem.solve()
    problem.phases.phase.duration.bounds = (0.0, None)
    assert first.phases.phase.duration == pytest.approx(0.5, rel=1e-6)
    assert problem.solve().phases.phase.duration < 0.5
