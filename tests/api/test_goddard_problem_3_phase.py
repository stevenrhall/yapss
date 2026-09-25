"""End-to-end solves of the three-phase Goddard rocket problem through the redesigned API."""

import numpy as np
import pytest

from yapss._api.fields import aspects_of
from yapss.examples.goddard_problem_3_phase import Discrete, Phases, mf, setup

FINAL_ALTITUDE = 18550.871863824515
"""What the same problem gives through the released API."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_solves_to_the_released_answer(problem):
    solution = problem.solve()
    assert solution.converged
    assert solution.objective == pytest.approx(FINAL_ALTITUDE, rel=1e-10)


def test_it_agrees_with_the_same_problem_in_the_released_api(problem):
    from yapss._legacy.examples.goddard_problem_3_phase import setup as legacy_setup

    legacy = legacy_setup()
    legacy.ipopt_options.print_level = 0
    assert problem.solve().objective == pytest.approx(legacy.solve().objective, rel=1e-10)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(FINAL_ALTITUDE, rel=1e-8)


@pytest.mark.parametrize("method", ["lgl", "lgr", "lg"])
def test_every_spectral_method_agrees(problem, method):
    problem.spectral_method = method
    assert problem.solve().objective == pytest.approx(FINAL_ALTITUDE, rel=1e-8)


def test_the_phases_are_joined(problem):
    """The linkage groups are the continuity conditions, and they are met exactly."""
    solution = problem.solve()
    boost, singular, coast = (
        problem.phases.boost,
        problem.phases.singular,
        problem.phases.coast,
    )
    for name in Discrete._fields:
        assert getattr(solution.discrete, name) == pytest.approx(0.0, abs=1e-8)
    assert solution[singular].initial.time == pytest.approx(solution[boost].final.time)
    assert solution[coast].initial.time == pytest.approx(solution[singular].final.time)


def test_the_path_constraint_holds_on_the_singular_arc(problem):
    solution = problem.solve()
    switching = solution[problem.phases.singular].path.switching
    assert np.abs(switching).max() == pytest.approx(0.0, abs=1e-6)


def test_the_terminal_conditions_are_met(problem):
    solution = problem.solve()
    assert solution[problem.phases.coast].final.m == pytest.approx(mf)
    assert solution[problem.phases.coast].final.h == pytest.approx(solution.objective, rel=1e-12)


def test_the_discrete_vector_is_read_by_name_and_by_position(problem):
    solution = problem.solve()
    assert len(solution.discrete) == 8
    assert solution.discrete[:].shape == (8,)
    assert solution.discrete[0] == solution.discrete.boost_singular_h
    assert np.isfinite(solution.multiplier.discrete.singular_coast_time)


def test_a_phase_without_a_path_declares_none(problem):
    solution = problem.solve()
    assert len(solution[problem.phases.boost].path) == 0


def test_one_callback_serves_two_phases(problem):
    """`powered` is registered on boost by decorator and on coast by call."""
    assert problem.phases.boost._continuous is problem.phases.coast._continuous
    assert problem.phases.singular._continuous is not problem.phases.boost._continuous


def test_registering_a_second_callback_replaces(problem):
    """And leaves the other phases alone, which is what sharing one callback rests on."""
    shared = problem.phases.coast._continuous

    def other(arg, out):
        pass

    problem.phases.boost.register.continuous(other)
    assert problem.phases.boost._continuous is other
    assert problem.phases.coast._continuous is shared


def test_an_unbounded_discrete_group_is_refused():
    problem = setup()
    aspects_of(problem.discrete).bounds._values.pop("singular_coast_time")
    with pytest.raises(ValueError, match="'singular_coast_time' has no bound"):
        problem.validate()


def test_the_declarations_name_what_they_hold():
    assert Discrete._fields[0] == "boost_singular_h"
    assert len(Discrete._fields) == 8
    assert Discrete._nrows == 8
    assert [phase.name for phase in Phases()] == ["boost", "singular", "coast"]
