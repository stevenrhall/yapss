"""A problem and a solution each have one place for the caller's own names: ``user``.

Every other name on either is YAPSS's, refused if it is not one YAPSS defined, so that a name
YAPSS adds later cannot collide with one a caller chose. ``user`` is the exception, stated: a
namespace YAPSS never reads, writes to, or adds a name to. The problem's is not a setting, so
no solve records it, and a solution's starts empty.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from yapss._api.settings import settings_of

from ._api import problem, raises, solvable

AREA = "problem"


@pytest.fixture(scope="module")
def solved():
    ocp = solvable()
    ocp.ipopt_options.print_level = 0
    ocp.user.case = "nominal"
    return ocp, ocp.solve()


def test_a_problem_holds_any_name_of_the_caller_s_under_user() -> None:
    """Set, read, and deleted like any attribute, with no check on the name or the value."""
    p = problem()
    assert vars(p.user) == {}
    p.user.mass = 3.0
    p.user.table = {"a": [1, 2]}
    assert p.user.mass == 3.0
    del p.user.mass
    assert vars(p.user) == {"table": {"a": [1, 2]}}


def test_each_problem_has_its_own() -> None:
    """Two problems of one class share nothing."""
    first, second = problem(), problem()
    first.user.tag = 1
    assert not hasattr(second.user, "tag")


def test_user_cannot_be_replaced_whole() -> None:
    """Names are set on it one at a time, as with every object a problem holds."""
    p = problem()
    with raises(
        AttributeError, "user cannot be replaced", "'<problem>.user.<name> = ...'", at="p.user ="
    ):
        p.user = SimpleNamespace()
    with raises(AttributeError, "user cannot be deleted", at="del"):
        del p.user


def test_a_misspelled_user_is_suggested() -> None:
    """The name is one of the problem's, so a near miss finds it."""
    p = problem()
    with raises(AttributeError, "Did you mean 'user'?", at="usr"):
        p.usr.mass = 3.0


def test_the_problem_s_user_is_not_a_setting(solved) -> None:
    """Nothing the caller keeps there is recorded with a solve, or carried to its solution."""
    ocp, solution = solved
    assert "user" not in dir(settings_of(ocp))
    assert "user" not in dir(solution.settings)
    assert vars(solution.user) == {}
    assert ocp.user.case == "nominal"


def test_a_solution_holds_any_name_of_the_caller_s_under_user(solved) -> None:
    """The one place on a solution that takes a new name."""
    _, solution = solved
    solution.user.margin = 0.25
    assert solution.user.margin == 0.25
    del solution.user.margin
    with raises(AttributeError, "solution.margin cannot be assigned", at="solution.margin"):
        solution.margin = 0.25


def test_a_solution_s_user_cannot_be_replaced_whole(solved) -> None:
    """As on the problem."""
    _, solution = solved
    with raises(
        AttributeError,
        "solution.user cannot be replaced",
        "'solution.user.<name> = ...'",
        at="solution.user =",
    ):
        solution.user = SimpleNamespace()


def test_a_solution_pickles_with_what_the_caller_put_there(solved) -> None:
    """Empty or not: a solution pickles if what is under `user` does."""
    _, solution = solved
    assert vars(pickle.loads(pickle.dumps(solution)).user) == {}
    solution.user.note = "kept"
    solution.user.samples = [1.0, 2.0]
    try:
        copy = pickle.loads(pickle.dumps(solution))
    finally:
        del solution.user.note, solution.user.samples
    assert vars(copy.user) == {"note": "kept", "samples": [1.0, 2.0]}
    assert copy.objective == solution.objective


def test_two_solves_give_solutions_that_share_nothing(solved) -> None:
    """Each solution's namespace is its own."""
    ocp, solution = solved
    again = ocp.solve()
    again.user.tag = 2
    assert not hasattr(solution.user, "tag")
