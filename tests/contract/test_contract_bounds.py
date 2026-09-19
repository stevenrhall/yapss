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
    - Assign bounds in any order: `lower = upper = x`, or moving a window from [0, 1] to
      [5, 6] one side at a time, even though the bounds cross in between.
    - Take a copy of a bound array (`copy()`, `astype`, indexing with a list or a mask,
      `np.sort`, arithmetic) and get a plain ndarray to use freely; print one and see
      `array(...)`.

What a user may get wrong
    - Wrong length, wrong dimension, or a scalar for a whole array: `ValueError` at the
      assignment.
    - A non-real value (str, bool, complex, None), whole, in a sequence, or written into an
      element or slice: `TypeError` at the assignment, naming the bound.
    - A value wrong on its own -- NaN, +inf as a lower or -inf as an upper bound, a
      negative duration bound -- however it is written (whole array, element, slice, mask,
      view, in-place operator, `fill`, scalar bound): `ValueError` at the assignment,
      naming the bound, with the bound unchanged.
    - Values wrong only together -- a crossing, boundary-state bounds outside the state
      bounds, infeasible time bounds: `ValueError` from `validate()` (run by `solve()`),
      naming the bound.
    - A value wrong on its own written around the checks (`np.copyto`, `np.put`, `flat`,
      `view(np.ndarray)`): `ValueError` from `validate()`. A bool written that way is
      converted by NumPy and not detected.
    - A misspelled attribute: `AttributeError` at the assignment.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from yapss._legacy.examples import brachistochrone_minimal
from yapss._private.bounds import (
    get_nlp_constraint_function_bounds,
    get_nlp_decision_variable_bounds,
)

from ._contract import (
    SCALAR_FORMS,
    SEQUENCE_FORMS,
    assert_float64_array,
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
        for a, b in zip(getter(plain._to_spec()), getter(varied._to_spec()), strict=True):
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


def around_the_checks(array):
    """Return a plain view that writes into `array` without its checks (a deliberate backdoor)."""
    return array.view(np.ndarray)


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("path", ARRAYS)
def test_nan_written_around_the_checks_is_reported_by_validate(path, side):
    ocp = problem()
    values = getattr(ARRAYS[path](ocp.bounds), side)
    around_the_checks(values)[-1] = np.nan
    with raises(ValueError, f"{path}.{side}[i] is NaN for indices i in [{len(values) - 1}]"):
        ocp.bounds.validate()


def test_none_in_a_sequence_raises_at_the_assignment():
    """`None` used to become NaN here, reported only later by validate()."""
    ocp = problem()
    with raises(TypeError, "bounds.phase[0].state.upper", at="state.upper ="):
        ocp.bounds.phase[0].state.upper = [10.0, None]


@pytest.mark.parametrize("path", ARRAYS)
def test_infinity_on_the_wrong_side_written_around_the_checks_is_reported_by_validate(path):
    ocp = problem()
    bound = ARRAYS[path](ocp.bounds)
    around_the_checks(bound.lower)[0] = around_the_checks(bound.upper)[0] = np.inf
    with raises(ValueError, f"{path}.lower[i] is +inf", "must be less than +inf"):
        ocp.bounds.validate()
    around_the_checks(bound.lower)[0] = around_the_checks(bound.upper)[0] = -np.inf
    with raises(ValueError, f"{path}.upper[i] is -inf", "must be greater than -inf"):
        ocp.bounds.validate()


def test_bounds_may_cross_between_assignments():
    """Moving a window one side at a time crosses in between; only validate() judges it."""
    ocp = problem()
    state = ocp.bounds.phase[0].state
    state.lower, state.upper = [0.0, 0.0], [1.0, 1.0]
    state.lower = [5.0, 5.0]  # crosses the old upper bound
    state.upper = [6.0, 6.0]
    b = ocp.bounds.phase[0].final_time
    b.upper = 1.0
    b.lower = b.upper = 5.0  # chained: lower is assigned first, while upper is still 1.0
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


def test_validate_runs_before_solve():
    ocp = brachistochrone_minimal.setup()
    around_the_checks(ocp.bounds.phase[0].control.upper)[0] = np.nan
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
    state.upper = [1.0, 2.0]
    state.lower = state.upper
    state.upper[0] = 5.0
    assert_float64_array(state.lower, [1.0, 2.0])


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


@pytest.mark.parametrize("value", ["1", True], ids=["numeric string", "bool"])
def test_non_real_element_raises_at_the_assignment(value):
    ocp = problem()
    with raises(TypeError, "bounds.phase[0].state.lower", at="state.lower[0] ="):
        ocp.bounds.phase[0].state.lower[0] = value
    assert np.all(ocp.bounds.phase[0].state.lower == -np.inf)


def test_bool_among_floats_raises_at_the_assignment():
    ocp = problem()
    state = ocp.bounds.phase[0].state
    with raises(TypeError, "bounds.phase[0].state.lower", at="state.lower ="):
        state.lower = [1.0, True]
    with raises(TypeError, "bounds.phase[0].state.upper", at="state.upper[:] ="):
        state.upper[:] = [1.0, True]
    assert np.all(state.lower == -np.inf)
    assert np.all(state.upper == np.inf)


# Decided 2026-09-16 (Steve): a value wrong on its own raises where it is written, however
# it is written, and a refused write changes nothing. Values wrong only together (a
# crossing) stay with validate(), so bounds can be assigned in any order.
#
# Each case: the side written, the bad value, and an in-place operation producing it from
# the default (-inf lower, +inf upper).
SINGLE_VALUE_ERRORS = {
    "NaN lower": ("lower", np.nan, lambda a: a.__iadd__(np.inf)),
    "NaN upper": ("upper", np.nan, lambda a: a.__isub__(np.inf)),
    "+inf lower": ("lower", np.inf, lambda a: a.__imul__(-1.0)),
    "-inf upper": ("upper", -np.inf, lambda a: a.__imul__(-1.0)),
}


def _whole(bound, side, v):
    setattr(bound, side, [0.0] * (len(getattr(bound, side)) - 1) + [v])


def _element(bound, side, v):
    getattr(bound, side)[-1] = v


def _slice(bound, side, v):
    getattr(bound, side)[:] = [0.0] * (len(getattr(bound, side)) - 1) + [v]


def _mask(bound, side, v):
    a = getattr(bound, side)
    a[np.isinf(a)] = v


def _view(bound, side, v):
    view = getattr(bound, side)[:]
    view[-1] = v


def _reshaped_view(bound, side, v):
    view = getattr(bound, side).reshape(1, -1)
    view[0, -1] = v


def _fill(bound, side, v):
    getattr(bound, side).fill(v)


# (writer, the fragment of its source line that performs the write)
WRITES = {
    "whole": (_whole, "setattr(bound, side,"),
    "element": (_element, "[-1] = v"),
    "slice": (_slice, "[:] = [0.0]"),
    "mask": (_mask, "a[np.isinf(a)] = v"),
    "view": (_view, "view[-1] = v"),
    "reshaped view": (_reshaped_view, "view[0, -1] = v"),
    "fill": (_fill, ".fill(v)"),
}


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("case", SINGLE_VALUE_ERRORS)
@pytest.mark.parametrize("path", ["bounds.phase[0].state", "bounds.discrete"])
def test_single_value_error_raises_at_the_write(path, case, write):
    ocp = problem()
    bound = ARRAYS[path](ocp.bounds)
    side, value, _ = SINGLE_VALUE_ERRORS[case]
    writer, at = WRITES[write]
    stored = getattr(bound, side)
    before = stored.copy()
    with raises(ValueError, f"{path}.{side}", at=at):
        writer(bound, side, value)
    assert getattr(bound, side) is stored
    np.testing.assert_array_equal(stored, before)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("case", SINGLE_VALUE_ERRORS)
def test_in_place_operator_producing_a_single_value_error_raises(case):
    ocp = problem()
    side, _, operation = SINGLE_VALUE_ERRORS[case]
    stored = getattr(ocp.bounds.phase[0].state, side)
    before = stored.copy()
    with raises(ValueError, f"bounds.phase[0].state.{side}", at="lambda a: a.__i"):
        operation(stored)
    np.testing.assert_array_equal(stored, before)


@pytest.mark.parametrize("case", SINGLE_VALUE_ERRORS)
@pytest.mark.parametrize("name", SCALARS)
def test_scalar_bound_single_value_error_raises_at_the_assignment(name, case):
    ocp = problem()
    scalar = getattr(ocp.bounds.phase[0], name)
    side, value, _ = SINGLE_VALUE_ERRORS[case]
    before = getattr(scalar, side)
    with raises(ValueError, f"bounds.phase[0].{name}.{side}", at="setattr(scalar"):
        setattr(scalar, side, value)
    assert getattr(scalar, side) == before


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_negative_duration_bound_raises_at_the_assignment(side):
    """The duration's lower bound of zero is all that keeps a phase from running backward."""
    ocp = problem()
    duration = ocp.bounds.phase[0].duration
    before = getattr(duration, side)
    with raises(
        ValueError,
        f"bounds.phase[0].duration.{side} cannot be less than zero",
        at="setattr(duration",
    ):
        setattr(duration, side, -1.0)
    assert getattr(duration, side) == before


# Decided 2026-09-16 (Steve): only an array that writes into the problem is checked. A copy
# is the user's own and is a plain ndarray; a bound array prints like one.
COPIES = {
    "copy()": lambda a: a.copy(),
    "astype": lambda a: a.astype(np.float64),
    "np.array": np.array,
    "list index": lambda a: a[[0, 1]],
    "mask index": lambda a: a[a < 0],
    "np.sort": np.sort,
    "arithmetic": lambda a: a + 1.0,
}


@pytest.mark.parametrize("make", COPIES)
def test_a_copy_of_a_bound_is_a_plain_ndarray(make):
    ocp = problem()
    lower = ocp.bounds.phase[0].state.lower
    result = COPIES[make](lower)
    assert type(result) is np.ndarray
    assert not np.shares_memory(result, lower)
    result[0] = np.nan  # the copy is the user's own
    assert np.all(lower == -np.inf)


def test_a_bound_array_prints_like_an_ndarray():
    ocp = problem()
    lower = ocp.bounds.phase[0].state.lower
    assert repr(lower) == "array([-inf, -inf])"
    assert str(lower) == "[-inf -inf]"


# A deep-copied problem keeps its checks: deepcopy is how Solution records the problem it
# solved. Pickling is not part of the contract (decided 2026-09-17).
def test_a_deep_copied_problem_keeps_its_checks():
    ocp = copy.deepcopy(problem())
    with raises(ValueError, "bounds.phase[0].state.lower"):
        ocp.bounds.phase[0].state.lower[0] = np.nan
