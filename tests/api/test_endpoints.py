"""What an endpoint callback and a solution are given at each end of a phase.

The state at an end and the time there are two names, one kind of thing each: the state is a
vector of the phase's state fields, whose positions address its rows, and the time is a number.
"""

import numpy as np
import pytest

from yapss.examples.brachistochrone import setup


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_each_end_has_its_state_and_its_time(problem):
    ps = problem.solve().phases[problem.phases.phase]
    assert ps.final_state.x == pytest.approx(1.0)
    assert ps.final_time == pytest.approx(np.sqrt(np.pi / 32.174), rel=1e-6)
    assert ps.initial_state.x == pytest.approx(0.0, abs=1e-8)


def test_the_time_is_not_a_state(problem):
    ps = problem.solve().phases[problem.phases.phase]
    with pytest.raises(AttributeError, match="has no field 'time'"):
        _ = ps.final_state.time
    with pytest.raises(AttributeError, match="Did you mean 'final_time'"):
        _ = ps.final_tim


def test_positions_address_the_state_rows(problem):
    end = problem.solve().phases[problem.phases.phase].final_state
    assert len(end) == 3
    assert end[:].shape == (3,)
    assert end[0] == end.x


def test_a_misspelled_endpoint_field_is_answered_by_the_state(problem):
    end = problem.solve().phases[problem.phases.phase].final_state
    with pytest.raises(AttributeError, match="Did you mean 'x'"):
        _ = end.xx


def test_the_two_ends_and_the_duration_agree(problem):
    ps = problem.solve().phases[problem.phases.phase]
    assert ps.duration == pytest.approx(ps.final_time - ps.initial_time)
    assert ps.initial_time == pytest.approx(0.0)
