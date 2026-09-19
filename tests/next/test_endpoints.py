"""What an endpoint callback and a solution are given at each end of a phase.

The state at an end and the independent variable there are one namespace, because that is what
they are: the phase's variables at a point. Positions address the state's rows; the independent
variable is a scalar in the same namespace, reached by name.
"""

import numpy as np
import pytest

from yapss.examples.brachistochrone import setup


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_an_endpoint_holds_the_state_and_the_independent_variable(problem):
    end = problem.solve()[problem.phases.slide].final
    assert end.x == pytest.approx(1.0)
    assert end.time == pytest.approx(np.sqrt(np.pi / 32.174), rel=1e-6)


def test_positions_address_the_state_rows(problem):
    end = problem.solve()[problem.phases.slide].final
    assert len(end) == 3
    assert end[:].shape == (3,)
    assert end[0] == end.x


def test_a_misspelled_endpoint_field_is_answered_by_the_state(problem):
    end = problem.solve()[problem.phases.slide].final
    with pytest.raises(AttributeError, match="Did you mean 'x'"):
        _ = end.xx


def test_the_two_ends_and_the_duration_agree(problem):
    ps = problem.solve()[problem.phases.slide]
    assert ps.duration == pytest.approx(ps.final.time - ps.initial.time)
    assert ps.initial.time == pytest.approx(0.0)
