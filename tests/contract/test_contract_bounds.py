"""Contract: `problem.bounds`.

What a user may do
    - Assign a whole bound array as any 1-D sequence of real numbers of the right length:
      list, tuple, range, ndarray of any real dtype, a list of NumPy scalars.
    - Assign one element, or a slice, with a real number or a matching sequence; a scalar
      broadcasts over a slice.
    - Use -inf for a lower bound and +inf for an upper bound, and a finite fixed value
      (lower == upper).
    - Assign a scalar bound (initial_time, final_time, duration) any real number, NumPy
      scalars included.
    - Rely on the defaults (-inf, +inf; duration.lower 0) and on `reset()` restoring them.
    - Rely on every accepted form reaching the NLP bounds as float64.

What a user may get wrong
    - Wrong length, wrong dimension, or a scalar for a whole array: `ValueError` at the
      assignment.
    - A non-real scalar bound (str, bool, None, ndarray): `TypeError` at the assignment.
    - NaN anywhere, +inf as a lower or -inf as an upper bound, a crossing,
      boundary-state bounds outside the state bounds, infeasible time bounds, a negative
      duration bound: `ValueError` from `validate()` (run by `solve()`), naming the bound.
    - A misspelled attribute: `AttributeError` at the assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from yapss._private.bounds import (
    get_nlp_constraint_function_bounds,
    get_nlp_decision_variable_bounds,
)
from yapss.examples import brachistochrone_minimal

from ._contract import (
    SCALAR_FORMS,
    SEQUENCE_FORMS,
    assert_float64_array,
    not_yet,
    problem,
    raises,
    reference,
)

# (path used in messages, accessor) for every kind of array bound
ARRAYS = {
    "bounds.phase[0].state": lambda b: b.phase[0].state,
    "bounds.phase[1].control": lambda b: b.phase[1].control,
    "bounds.phase[0].initial_state": lambda b: b.phase[0].initial_state,
    "bounds.phase[0].final_state": lambda b: b.phase[0].final_state,
    "bounds.phase[1].integral": lambda b: b.phase[1].integral,
    "bounds.phase[0].path": lambda b: b.phase[0].path,
    "bounds.discrete": lambda b: b.discrete,
    "bounds.parameter": lambda b: b.parameter,
}
SCALARS = ["initial_time", "final_time", "duration"]


# ---------------------------------------------------------------- what a user may do


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("form", SEQUENCE_FORMS)
@pytest.mark.parametrize("path", ARRAYS)
def test_whole_array_accepts_any_real_sequence(path, form, side):
    ocp = problem()
    bound = ARRAYS[path](ocp.bounds)
    n = len(getattr(bound, side))
    other = "upper" if side == "lower" else "lower"
    setattr(bound, other, [np.inf if other == "upper" else -np.inf] * n)
    setattr(bound, side, SEQUENCE_FORMS[form](n))
    assert_float64_array(getattr(bound, side), reference(n))
    ocp.bounds.validate()


@pytest.mark.parametrize("form", SCALAR_FORMS)
@pytest.mark.parametrize("path", ARRAYS)
def test_element_accepts_any_real_scalar(path, form):
    ocp = problem()
    bound = ARRAYS[path](ocp.bounds)
    bound.lower[-1] = SCALAR_FORMS[form]
    bound.upper[-1] = SCALAR_FORMS[form]
    assert bound.lower.dtype == np.float64
    assert bound.lower[-1] == bound.upper[-1] == 3.0
    ocp.bounds.validate()


def test_slice_accepts_scalar_broadcast_sequence_and_ndarray():
    ocp = problem()
    state = ocp.bounds.phase[0].state
    state.lower[:] = 0
    assert_float64_array(state.lower, [0.0, 0.0])
    state.upper[:] = [1, 2]
    assert_float64_array(state.upper, [1.0, 2.0])
    state.upper[1:] = np.array([5.0])
    assert_float64_array(state.upper, [1.0, 5.0])
    ocp.bounds.validate()


def test_chained_fixed_value_assignment():
    """The documented idiom `lower[i] = upper[i] = value` fixes a variable."""
    ocp = problem()
    b = ocp.bounds.phase[0]
    b.initial_state.lower[:] = b.initial_state.upper[:] = 0.0
    b.final_state.lower[0] = b.final_state.upper[0] = 1.0
    assert_float64_array(b.initial_state.lower, [0.0, 0.0])
    assert b.final_state.lower[0] == b.final_state.upper[0] == 1.0
    ocp.bounds.validate()


@pytest.mark.parametrize("path", ARRAYS)
def test_infinite_bounds_on_the_correct_side_are_valid(path):
    ocp = problem()
    bound = ARRAYS[path](ocp.bounds)
    bound.lower[:] = -np.inf
    bound.upper[:] = np.inf
    ocp.bounds.validate()


@pytest.mark.parametrize("form", SCALAR_FORMS)
@pytest.mark.parametrize("name", SCALARS)
def test_scalar_bound_accepts_any_real_number(name, form):
    ocp = problem()
    scalar = getattr(ocp.bounds.phase[0], name)
    scalar.upper = SCALAR_FORMS[form]
    scalar.lower = SCALAR_FORMS[form]
    assert type(scalar.upper) is float
    assert scalar.lower == scalar.upper == 3.0


def test_scalar_bounds_accept_infinity_on_the_correct_side():
    ocp = problem()
    b = ocp.bounds.phase[0]
    b.initial_time.lower = -np.inf
    b.final_time.upper = np.inf
    b.duration.upper = np.inf
    ocp.bounds.validate()


def test_defaults_and_reset():
    ocp = problem()
    for path, get in ARRAYS.items():
        bound = get(ocp.bounds)
        assert np.all(bound.lower == -np.inf), path
        assert np.all(bound.upper == np.inf), path
    b = ocp.bounds.phase[0]
    assert (b.initial_time.lower, b.initial_time.upper) == (-np.inf, np.inf)
    assert (b.duration.lower, b.duration.upper) == (0.0, np.inf)

    b.state.lower[:] = 1.0
    b.duration.lower = 2.0
    ocp.bounds.parameter.upper[0] = 3.0
    ocp.bounds.reset()
    assert np.all(b.state.lower == -np.inf)
    assert b.duration.lower == 0.0
    assert np.all(ocp.bounds.parameter.upper == np.inf)


def test_accepted_forms_reach_the_nlp_bounds():
    """Bounds set by elements, slices, NumPy types, and ranges give the same NLP bounds."""
    plain = brachistochrone_minimal.setup()
    varied = brachistochrone_minimal.setup()
    pb, vb = plain.bounds.phase[0], varied.bounds.phase[0]

    pb.control.lower = [-1.5]
    vb.control.lower[0] = np.float32(-1.5)
    pb.state.upper = [10.0, 10.0, 10.0]
    vb.state.upper[:] = np.int64(10)
    pb.final_time.upper = 5.0
    vb.final_time.upper = np.int64(5)

    for getter in (get_nlp_decision_variable_bounds, get_nlp_constraint_function_bounds):
        for a, b in zip(getter(plain), getter(varied), strict=True):
            assert a.dtype == b.dtype == np.float64
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------- what a user may get wrong


@pytest.mark.parametrize(
    "value",
    [[1.0], [1.0, 2.0, 3.0], [[1.0, 2.0]], 1.0],
    ids=["too short", "too long", "2-D", "scalar"],
)
def test_whole_array_of_wrong_shape_raises_at_the_assignment(value):
    ocp = problem()
    with raises(ValueError, "length 2", at="state.lower ="):
        ocp.bounds.phase[0].state.lower = value


def test_complex_values_raise_at_the_assignment():
    ocp = problem()
    with raises(TypeError, at="state.lower ="):
        ocp.bounds.phase[0].state.lower = [1j, 2.0]


@pytest.mark.parametrize("value", ["3.0", True, None], ids=["str", "bool", "None"])
@pytest.mark.parametrize("name", SCALARS)
def test_scalar_bound_of_non_real_type_raises_at_the_assignment(name, value):
    ocp = problem()
    with raises(TypeError, f"bounds.phase[1].{name}.upper", at=".upper ="):
        getattr(ocp.bounds.phase[1], name).upper = value


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("path", ARRAYS)
def test_nan_is_reported_by_validate(path, side):
    ocp = problem()
    values = getattr(ARRAYS[path](ocp.bounds), side)
    values[-1] = np.nan
    with raises(ValueError, f"{path}.{side}[i] is NaN for indices i in [{len(values) - 1}]"):
        ocp.bounds.validate()


def test_none_in_a_sequence_raises_at_the_assignment():
    """`None` used to become NaN here, reported only later by validate()."""
    ocp = problem()
    with raises(TypeError, "bounds.phase[0].state.upper", at="state.upper ="):
        ocp.bounds.phase[0].state.upper = [10.0, None]


@pytest.mark.parametrize("path", ARRAYS)
def test_infinity_on_the_wrong_side_is_reported_by_validate(path):
    ocp = problem()
    bound = ARRAYS[path](ocp.bounds)
    bound.lower[0] = bound.upper[0] = np.inf
    with raises(ValueError, f"{path}.lower[i] is +inf", "must be less than +inf"):
        ocp.bounds.validate()
    bound.lower[0] = bound.upper[0] = -np.inf
    with raises(ValueError, f"{path}.upper[i] is -inf", "must be greater than -inf"):
        ocp.bounds.validate()


@pytest.mark.parametrize("name", SCALARS)
def test_scalar_nan_and_wrong_side_infinity_are_reported_by_validate(name):
    ocp = problem()
    scalar = getattr(ocp.bounds.phase[0], name)
    scalar.lower = np.nan
    with raises(ValueError, f"bounds.phase[0].{name}.lower is NaN"):
        ocp.bounds.validate()
    scalar.lower = np.inf
    scalar.upper = np.inf
    with raises(ValueError, f"bounds.phase[0].{name}.lower is +inf"):
        ocp.bounds.validate()


def test_crossing_is_reported_by_validate():
    ocp = problem()
    state = ocp.bounds.phase[0].state
    state.lower = [0.0, 5.0]
    state.upper = [1.0, 3.0]
    with raises(ValueError, "bounds.phase[0].state.lower[i] is greater than", "[1]"):
        ocp.bounds.validate()


def test_boundary_state_outside_state_bounds_is_reported_by_validate():
    ocp = problem()
    b = ocp.bounds.phase[0]
    b.state.lower = [5.0, 5.0]
    b.initial_state.upper = [0.0, 10.0]
    with raises(
        ValueError, "bounds.phase[0].initial_state and bounds.phase[0].state do not overlap"
    ):
        ocp.bounds.validate()


def test_infeasible_time_bounds_are_reported_by_validate():
    ocp = problem()
    b = ocp.bounds.phase[0]
    b.initial_time.lower = 0.0
    b.final_time.upper = 5.0
    b.duration.lower = 10.0
    with raises(ValueError, "Time bounds are infeasible"):
        ocp.bounds.validate()


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_negative_duration_bound_is_reported_by_validate(side):
    ocp = problem()
    ocp.bounds.phase[0].duration.lower = -2.0
    ocp.bounds.phase[0].duration.upper = -1.0 if side == "upper" else 5.0
    with raises(ValueError, f"bounds.phase[0].duration.{side} cannot be less than zero"):
        ocp.bounds.validate()


def test_validate_runs_before_solve():
    ocp = brachistochrone_minimal.setup()
    ocp.bounds.phase[0].control.upper[0] = np.nan
    with raises(ValueError, "bounds.phase[0].control.upper[i] is NaN"):
        ocp.solve()


@pytest.mark.parametrize(
    ("owner", "typo"),
    [
        (lambda b: b, "phse"),
        (lambda b: b.phase[0], "stat"),
        (lambda b: b.phase[0].state, "lowr"),
        (lambda b: b.phase[0].final_time, "uper"),
    ],
    ids=["Bounds", "PhaseBounds", "ArrayBounds", "ScalarBounds"],
)
def test_misspelled_attribute_raises_at_the_assignment(owner, typo):
    ocp = problem()
    with raises(AttributeError, typo, at="setattr"):
        setattr(owner(ocp.bounds), typo, 1.0)


def test_bound_arrays_cannot_be_deleted():
    ocp = problem()
    with raises(AttributeError, "cannot delete", at="del"):
        del ocp.bounds.phase[0].state.lower


# ----------------------------------------------------------------- not yet met


def test_whole_array_assignment_copies():
    ocp = problem()
    source = np.array([1.0, 2.0])
    ocp.bounds.phase[0].state.lower = source
    source[0] = 99.0
    assert_float64_array(ocp.bounds.phase[0].state.lower, [1.0, 2.0])


def test_assigning_one_bound_from_another_does_not_alias():
    ocp = problem()
    state = ocp.bounds.phase[0].state
    state.lower = state.upper
    state.reset()
    assert np.all(state.lower == -np.inf)
    assert np.all(state.upper == np.inf)


@not_yet("E2 part 2", "NaN written into a bound element raises at the assignment")
def test_nan_element_raises_at_the_assignment():
    ocp = problem()
    with raises(ValueError, "bounds.phase[0].state.lower", at="state.lower[0] ="):
        ocp.bounds.phase[0].state.lower[0] = np.nan


def test_wrong_length_message_names_the_bound():
    ocp = problem()
    with raises(ValueError, "bounds.phase[0].state.lower", "2"):
        ocp.bounds.phase[0].state.lower = [1.0]


# Decided 2026-09-14 (Steve): bounds, guess, and scale accept real numbers and convert
# nothing else. Strings (even numeric ones; `np.inf` covers infinity), bools, and complex
# values raise TypeError naming the attribute.
NOT_REAL = {
    "numeric strings": ["1", "2"],
    "bools": [True, False],
    "complex list": [1j, 2.0],
    "complex ndarray": np.array([1 + 1j, 2.0]),
}


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("form", NOT_REAL)
def test_non_real_whole_array_raises_at_the_assignment(form):
    ocp = problem()
    with raises(TypeError, "bounds.phase[0].state.lower", at="state.lower ="):
        ocp.bounds.phase[0].state.lower = NOT_REAL[form]


@not_yet("E2 part 2", "a non-real value written into a bound element raises at the assignment")
@pytest.mark.parametrize("value", ["1", True], ids=["numeric string", "bool"])
def test_non_real_element_raises_at_the_assignment(value):
    ocp = problem()
    with raises(TypeError, "bounds.phase[0].state.lower", at="state.lower[0] ="):
        ocp.bounds.phase[0].state.lower[0] = value
