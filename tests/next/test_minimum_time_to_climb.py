"""End-to-end solve of the minimum time to climb through the redesigned API.

This is the problem whose model is expensive -- a radial-basis interpolator for thrust and
cubic splines for the atmosphere, looked up on every evaluation. It is the realistic shape of
the case central differences exist for, where the dynamics cost far more than the API around
them, and it is the counterweight to Delta III, whose model is cheap Python arithmetic.
"""

import pytest

from yapss._next.examples.minimum_time_to_climb import Aircraft, setup

TIME_TO_CLIMB = 320.458760680413
"""What the same problem gives through the released API, to the last bit."""


@pytest.fixture(scope="module")
def solution():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem, problem.solve()


def test_it_solves_to_the_released_answer(solution):
    _, result = solution
    assert result.converged
    assert result.objective == pytest.approx(TIME_TO_CLIMB, rel=1e-10)


def test_it_agrees_with_the_same_problem_in_the_released_api(solution):
    from yapss.examples.minimum_time_to_climb import setup as legacy_setup

    legacy = legacy_setup()
    legacy.ipopt_options.print_level = 0
    _, result = solution
    assert result.objective == pytest.approx(legacy.solve().objective, rel=1e-10)


def test_the_terminal_conditions_are_met(solution):
    problem, result = solution
    final = result[problem.phases.climb].final
    assert final.h == pytest.approx(65600.0, rel=1e-6)
    assert final.v == pytest.approx(968.148, rel=1e-6)
    assert final.gamma == pytest.approx(0.0, abs=1e-6)


def test_the_solution_is_reached_by_name(solution):
    problem, result = solution
    ps = result[problem.phases.climb]
    assert ps.state.h.shape == ps.time.shape
    assert ps.control.alpha.shape == ps.time.shape
    assert Aircraft._fields == ("h", "v", "gamma", "mass")
