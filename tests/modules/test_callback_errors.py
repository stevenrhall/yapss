"""Tests for the messages YAPSS attaches to errors in and around user callbacks (W5)."""

import math
import re
import warnings

import numpy as np
import pytest

from yapss._legacy.examples import brachistochrone, brachistochrone_minimal
from yapss._private.assembly import over_points

HINT = "Use the functions of yapss.math instead."


def _setup(method):
    problem = brachistochrone_minimal.setup()
    problem.ipopt_options.print_level = 0
    problem.derivatives.method = method
    return problem


def _notes(error):
    return getattr(error, "__notes__", [])


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_one_note_names_the_users_callback(method):
    problem = _setup(method)

    def continuous(arg):
        raise KeyError("user error")

    problem.functions.continuous = continuous
    with pytest.raises(KeyError, match="user error") as info:
        problem.solve()
    code = continuous.__code__
    assert _notes(info.value) == [
        f"Raised in functions.continuous = {continuous.__qualname__} "
        f"({code.co_filename}, line {code.co_firstlineno}).",
    ]


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
def test_a_failure_during_the_solve_is_noted_once():
    """Central difference wraps the callbacks in YAPSS's own functions; only the user's is named."""
    problem = _setup("central-difference")
    inner = problem.functions.continuous
    calls = {"n": 0}

    def continuous(arg):
        calls["n"] += 1
        if calls["n"] > 20:
            raise RuntimeError("fails mid-solve")
        inner(arg)

    problem.functions.continuous = continuous
    with pytest.raises(RuntimeError, match="fails mid-solve") as info:
        problem.solve()
    assert calls["n"] > 20
    (note,) = _notes(info.value)
    assert note.startswith("Raised in functions.continuous = ")


# NumPy < 2.3 warns before raising here; newer NumPy raises directly
@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")
def test_a_float_only_function_on_a_symbol_points_to_yapss_math():
    problem = _setup("auto")
    inner = problem.functions.continuous

    def continuous(arg):
        inner(arg)
        math.sin(arg.phase[0].state[2])

    problem.functions.continuous = continuous
    with pytest.raises(TypeError) as info:
        problem.solve()
    assert HINT in _notes(info.value)[-1]


def test_no_yapss_math_hint_off_the_symbolic_trace():
    problem = _setup("central-difference")

    def continuous(arg):
        raise TypeError("user error")

    problem.functions.continuous = continuous
    with pytest.raises(TypeError) as info:
        problem.solve()
    assert not any(HINT in note for note in _notes(info.value))


def test_no_yapss_math_hint_when_the_message_already_names_it():
    problem = _setup("auto")
    inner = problem.functions.continuous

    def continuous(arg):
        inner(arg)
        if arg.phase[0].state[2] > 0:  # a Python `if` on a symbol: the error names yapss.math
            pass

    problem.functions.continuous = continuous
    with pytest.raises(TypeError, match="yapss.math") as info:
        problem.solve()
    assert not any(HINT in note for note in _notes(info.value))


@pytest.mark.parametrize(
    "value",
    [np.array([1.0]), np.array([1.0, 2.0]), [1.0], np.zeros((1, 1))],
    ids=["length-1 array", "array", "list", "2-D"],
)
def test_an_objective_with_a_dimension_raises_at_the_assignment(value):
    problem = _setup("central-difference")

    def objective(arg):
        arg.objective = value

    problem.functions.objective = objective
    with pytest.raises(TypeError, match=re.escape(f"got a value of shape {np.shape(value)}")):
        problem.solve()


@pytest.mark.parametrize("method", ["auto", "central-difference"])
@pytest.mark.parametrize(
    "make",
    [lambda arg: arg.phase[0].final_time, lambda arg: np.float64(1.0) * arg.phase[0].final_time],
    ids=["input", "numpy scalar times input"],
)
def test_a_scalar_objective_is_accepted(method, make):
    problem = _setup(method)

    def objective(arg):
        arg.objective = make(arg)

    problem.functions.objective = objective
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solution = problem.solve()
    assert solution.converged


@pytest.mark.parametrize(
    ("value", "shape"),
    [(np.ones(1), "(1,)"), (np.ones(11), "(11,)"), (np.ones((10, 1)), "(10, 1)")],
    ids=["too short", "too long", "2-D"],
)
def test_a_wrong_shaped_entry_raises_naming_it(value, shape):
    key = (("f", 2), ("u", 0), ("u", 0))
    message = (
        f"arg.phase[1].hessian[('f', 2), ('u', 0), ('u', 0)] has shape {shape}; a derivative "
        f"entry is a scalar, or has one value per evaluation point (10)."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        over_points(value, (10,), 1, "hessian", key)


@pytest.mark.parametrize("value", [2.0, np.float64(2.0), np.array(2.0), np.full(10, 2.0)])
def test_a_scalar_or_one_value_per_point_is_accepted(value):
    result = over_points(value, (10,), 0, "jacobian", (("f", 0), ("x", 0)))
    np.testing.assert_array_equal(result, np.full(10, 2.0))


def test_a_wrong_shaped_user_jacobian_entry_raises_from_the_solve():
    problem = brachistochrone.setup()
    problem.ipopt_options.print_level = 0
    jacobian = problem.functions.continuous_jacobian

    def continuous_jacobian(arg):
        jacobian(arg)
        arg.phase[0].jacobian[("f", 0), ("x", 2)] = np.array([1.0, 2.0])

    problem.functions.continuous_jacobian = continuous_jacobian
    with pytest.raises(ValueError, match=re.escape("arg.phase[0].jacobian[('f', 0), ('x', 2)]")):
        problem.solve()
