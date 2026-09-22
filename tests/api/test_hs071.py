"""A problem with no phases, which is an ordinary nonlinear program.

HS071 is here for the degenerate case rather than for the mathematics. Spec 1.1 says zero is a
count -- no phases, no states, no fields, no rows -- because nothing about the transcription
changes shape at the bottom of any of those ranges, so nothing in the interface refuses them.
Until this example there was one exception, and it was an error message rather than a
limitation: `Phases` refused to declare none.

It is also the only problem in the corpus whose `"user"` derivatives need no continuous
callbacks at all, since there is no phase to have any.
"""

import numpy as np
import pytest

import yapss
from yapss.examples.hs071 import Discrete, Parameter, Phases, main, setup

RELEASED = 17.014017140224134
"""What the same problem gives through the released API, to the last bit."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_a_phases_class_may_declare_none():
    """The case the spec's 'zero is a count' principle covers, and the one that was refused."""
    assert list(Phases()) == []
    assert Phases._declared == {}


def test_it_agrees_with_the_released_api(problem):
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-14)


@pytest.mark.parametrize(
    "method", ["user", "auto", "central-difference", "central-difference-full"]
)
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


def test_user_derivatives_need_no_continuous_callbacks():
    """With no phases there is no continuous callback, so `"user"` wants only the endpoint four.

    The example itself is written that way, so this states what `setup()` already does rather
    than building a second copy of it.
    """
    problem = setup()
    problem.ipopt_options.print_level = 0
    assert problem.derivatives.method == "user"
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-10)


def test_the_hand_written_derivatives_are_the_ones_used():
    """Break one entry of the objective gradient and the solve must not reach the answer.

    Without this the example would pass whether or not its callbacks were reached, since a
    wrong derivative costs iterations rather than raising.
    """
    problem = setup()
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.max_iter = 5

    def wrong(arg, gradient):
        dx = gradient.parameter.x
        for i in range(4):
            gradient[dx[i]] = 1.0
        return gradient

    problem.register.objective_gradient(wrong, replace=True)
    with pytest.warns(yapss.IpoptConvergenceWarning):
        assert problem.solve().objective != pytest.approx(RELEASED, rel=1e-6)


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
