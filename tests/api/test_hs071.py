"""A problem with no phases, which is an ordinary nonlinear program.

HS071 is here for the degenerate case rather than for the mathematics. Spec 1.1 says zero is a
count -- no phases, no states, no fields, no rows -- because nothing about the transcription
changes shape at the bottom of any of those ranges, so nothing in the interface refuses them.
Two exceptions have been retired since: `Phases` refused to declare none, and `phases=`
was a required keyword, so a problem with no phases had to hand over an empty class.
"""

import numpy as np
import pytest

import yapss
from yapss.examples.hs071 import Discrete, Parameter, main, setup

RELEASED = 17.014017140224134
"""What the same problem gives through the released API, to the last bit."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_a_problem_may_have_no_phases(problem):
    """The case the spec's 'zero is a count' principle covers, and the one that was refused."""
    assert list(problem.phases) == []
    assert list(yapss.Phases()) == []


def test_it_agrees_with_the_released_api(problem):
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-14)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-10)


def test_the_solution_reads_under_its_own_names(problem):
    solution = problem.solve()
    assert solution.parameter.x.shape == (4,)
    np.testing.assert_allclose(
        solution.parameter.x, [1.0, 4.742999644, 3.821149979, 1.379408293], atol=1e-8
    )
    assert solution.discrete.product == pytest.approx(25.0)
    assert solution.discrete.sum_of_squares == pytest.approx(40.0)
    assert solution.multiplier.discrete.product != 0.0


def test_there_are_no_phases_to_reach(problem):
    solution = problem.solve()
    assert list(problem.phases) == []
    with pytest.raises(KeyError, match="has no phase 'slide'. The problem declared no phases"):
        _ = solution["slide"]


def test_the_declarations_are_what_they_look_like():
    assert Parameter._fields == ("x",)
    assert Parameter._nrows == 4
    assert Discrete._fields == ("product", "sum_of_squares")


def test_main_runs(capsys):
    """`main` is the documented install smoke test, and raises if the answer moves.

    Called directly rather than through `runpy`, as `tests/examples/test_example_main.py` does:
    re-executing an already-imported module warns, and the module guard is not what is at stake.
    """
    main()
    assert "YAPSS solution is correct." in capsys.readouterr().out
