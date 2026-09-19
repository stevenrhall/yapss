"""The shortest complete statement of a problem, which must agree with the long one.

`brachistochrone_minimal` and `brachistochrone` are the same problem written twice: one with
units, labels and commentary, one with none of it. They should give the same answer to the last
bit, and this is what says so -- if they diverge, one of them has been edited and the other has
not.
"""

import pytest

from yapss.examples.brachistochrone import setup as setup_full
from yapss.examples.brachistochrone_minimal import setup

RELEASED = 0.312480130713672
"""The time the same problem gives through the released API."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_agrees_with_the_released_api(problem):
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-10)


def test_it_agrees_with_the_annotated_version(problem):
    full = setup_full()
    full.ipopt_options.print_level = 0
    assert problem.solve().objective == pytest.approx(full.solve().objective, rel=1e-12)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-7)


def test_the_bead_arrives_where_it_was_sent(problem):
    ps = problem.solve()[problem.phases.slide]
    assert ps.final.x == pytest.approx(1.0, abs=1e-8)
    assert ps.initial.x == pytest.approx(0.0, abs=1e-8)
    assert ps.initial.v == pytest.approx(0.0, abs=1e-8)
