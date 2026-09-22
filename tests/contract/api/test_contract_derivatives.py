"""How derivatives are computed, and what choosing badly says.

Three methods and two orders. The choice is a setting on the problem rather than an argument to
`solve`, so a misspelling is refused at the line that makes it rather than at the solve.
"""

from __future__ import annotations

from ._api import problem, raises, solvable


def test_the_method_is_one_of_three() -> None:
    """The message lists them, which is shorter than going to look."""
    p = problem()
    with raises(
        ValueError,
        "derivatives.method must be one of",
        "central-difference-full",
        at="derivatives.method",
    ):
        p.derivatives.method = "fd"


def test_derivatives_by_hand_are_not_offered() -> None:
    """0.3.0 offered ``"user"``, so it is refused with the reason and what to use instead.

    Hand-written derivatives were feasible only for problems small enough that ``"auto"``
    differentiates them instantly; a model that cannot be traced is what central differences
    are for.
    """
    p = problem()
    with raises(
        ValueError,
        "derivatives.method = 'user' is not offered",
        "Use 'auto'",
        "'central-difference'",
        at='derivatives.method = "user"',
    ):
        p.derivatives.method = "user"


def test_the_order_is_first_or_second() -> None:
    """Second is the default: Ipopt converges in fewer iterations with exact Hessians."""
    p = problem()
    with raises(ValueError, "derivatives.order must be one of first, second", at="order"):
        p.derivatives.order = "third"


def test_every_method_reaches_the_same_answer() -> None:
    """The methods differ in cost and in what they need, never in what they compute."""
    answers = [solvable(method).solve().objective for method in ("auto", "central-difference")]
    assert abs(answers[0] - answers[1]) < 1e-5


def test_first_order_solves_too() -> None:
    """Without a Hessian Ipopt uses a quasi-Newton update, which is slower and still right."""
    p = solvable()
    p.derivatives.order = "first"
    assert abs(p.solve().objective - solvable().solve().objective) < 1e-5
