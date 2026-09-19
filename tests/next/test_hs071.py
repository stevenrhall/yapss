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

from yapss import _next as yapss
from yapss._next.examples.hs071 import Constraints, Design, Phases, main, setup

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
    assert solution.discrete_multiplier.product != 0.0


def test_there_are_no_phases_to_reach(problem):
    solution = problem.solve()
    assert list(problem.phases) == []
    with pytest.raises(KeyError, match="takes a phase handle"):
        _ = solution["slide"]


def test_user_derivatives_need_no_continuous_callbacks():
    """With no phases there is no continuous callback, so `"user"` wants only the endpoint pair."""
    problem = setup()
    problem.ipopt_options.print_level = 0

    @problem.register.objective_gradient
    def gradient(arg, gradient):
        x = arg.parameter.x
        gradient.x[0] = x[3] * (2 * x[0] + x[1] + x[2])
        gradient.x[1] = x[0] * x[3]
        gradient.x[2] = x[0] * x[3] + 1.0
        gradient.x[3] = x[0] * (x[0] + x[1] + x[2])
        return gradient

    @problem.register.objective_hessian
    def hessian(arg, hessian):
        x = arg.parameter.x
        hessian.x[0].x[0] = 2 * x[3]
        hessian.x[0].x[1] = x[3]
        hessian.x[0].x[2] = x[3]
        hessian.x[0].x[3] = 2 * x[0] + x[1] + x[2]
        hessian.x[1].x[3] = x[0]
        hessian.x[2].x[3] = x[0]
        return hessian

    @problem.register.discrete_jacobian
    def jacobian(arg, jacobian):
        x = arg.parameter.x
        for i in range(4):
            jacobian.discrete.product.x[i] = np.prod([x[j] for j in range(4) if j != i])
            jacobian.discrete.sum_of_squares.x[i] = 2 * x[i]
        return jacobian

    @problem.register.discrete_hessian
    def discrete_hessian(arg, hessian):
        x = arg.parameter.x
        for i in range(4):
            hessian.discrete.sum_of_squares.x[i].x[i] = 2.0
            for j in range(i + 1, 4):
                rest = [k for k in range(4) if k not in (i, j)]
                hessian.discrete.product.x[i].x[j] = x[rest[0]] * x[rest[1]]
        return hessian

    problem.derivatives.method = "user"
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-10)


def test_the_declarations_are_what_they_look_like():
    assert Design._fields == ("x",)
    assert Design._nrows == 4
    assert Constraints._fields == ("product", "sum_of_squares")


def test_main_runs(capsys):
    """`main` is the documented install smoke test, and raises if the answer moves.

    Called directly rather than through `runpy`, as `tests/examples/test_example_main.py` does:
    re-executing an already-imported module warns, and the module guard is not what is at stake.
    """
    main()
    assert "YAPSS solution is correct." in capsys.readouterr().out
