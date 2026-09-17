"""Contract: `problem.guess`.

What a user may do
    - Assign a phase's `time` as any strictly increasing 1-D sequence of at least two real
      numbers: list, tuple, range, ndarray of any real dtype, NumPy scalars.
    - Assign `state` and `control` as 2-D arrays of shape (n, len(time)): nested lists or
      tuples, or an ndarray of any real dtype.
    - Read back a `state` or `control` that was assigned, whether or not `time` is set.
    - Read `state` or `control` once `time` is set and get zeros, then write into them by
      element, row, column, or slice; the writes stick. The zeros follow a later change of
      `time` for as long as they have not been written.
    - Assign `integral` and `parameter` as 1-D sequences of the right length, or by element.
    - Rely on a guess copying what it is set from, and on every accepted form reaching the
      NLP starting point.
    - Warm-start with `problem.guess(solution)` (equivalently `guess.from_solution`).
    - Return the whole guess, or one phase's, to its state when the problem was created with
      `guess.reset()` or `guess.phase[p].reset()`: `time`, `state`, and `control` unset,
      `integral` and `parameter` zeros.

What a user may get wrong
    - A `time` that is not strictly increasing, too short, or not 1-D: `ValueError` at the
      assignment.
    - A `state` or `control` that is not 2-D or has the wrong number of rows: `ValueError`
      at the assignment. The wrong number of columns: `ValueError` from `validate()`.
    - Reading an unassigned `state` or `control` before `time`: `ValueError` naming what to
      set.
    - A phase whose `time` was never set: `ValueError` from `validate()`.
    - A `parameter` of the wrong length, or a scalar: `ValueError` at the assignment.
    - A string, a bool, a complex value, or `None`, whole or in a sequence: `TypeError` at
      the assignment, naming the attribute. A numeric string is no different.
    - A non-finite guess value, including in `time`: `ValueError` at the assignment.
    - A misspelled attribute: `AttributeError` at the assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
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

MATRIX_FORMS = {
    "nested lists of int": lambda: [[0, 1, 2], [3, 4, 5]],
    "nested tuples of float": lambda: ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)),
    "ndarray int64": lambda: np.arange(6, dtype=np.int64).reshape(2, 3),
    "ndarray float32": lambda: np.arange(6, dtype=np.float32).reshape(2, 3),
}
MATRIX_REFERENCE = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def timed():
    """Return the contract problem with a three-point time guess in both phases."""
    ocp = problem()
    for phase in ocp.guess.phase:
        phase.time = [0.0, 0.5, 1.0]
    return ocp


# ---------------------------------------------------------------- what a user may do


@pytest.mark.parametrize("form", SEQUENCE_FORMS)
def test_time_accepts_any_increasing_real_sequence(form):
    ocp = timed()
    ocp.guess.phase[0].time = SEQUENCE_FORMS[form](3)
    assert_float64_array(ocp.guess.phase[0].time, reference(3))
    ocp.guess.validate()


@pytest.mark.parametrize("form", MATRIX_FORMS)
def test_state_accepts_any_real_matrix(form):
    ocp = timed()
    ocp.guess.phase[0].state = MATRIX_FORMS[form]()
    assert_float64_array(ocp.guess.phase[0].state, MATRIX_REFERENCE)
    ocp.guess.validate()


@pytest.mark.parametrize("form", ["nested lists of int", "ndarray float32"])
def test_control_accepts_any_real_matrix(form):
    ocp = timed()
    ocp.guess.phase[1].control = MATRIX_FORMS[form]()[:1]
    assert_float64_array(ocp.guess.phase[1].control, MATRIX_REFERENCE[:1])
    ocp.guess.validate()


def test_unset_state_and_control_are_zeros_that_accept_element_row_column_and_slice_writes():
    ocp = timed()
    phase = ocp.guess.phase[0]
    assert_float64_array(phase.state, np.zeros((2, 3)))
    assert_float64_array(phase.control, np.zeros((1, 3)))
    phase.state[0, 1] = np.float32(5)
    phase.state[1] = [7, 8, 9]
    phase.state[:, 0] = 1
    phase.control[0, 1:] = np.array([2.0, 3.0])
    assert_float64_array(phase.state, [[1.0, 5.0, 0.0], [1.0, 8.0, 9.0]])
    assert_float64_array(phase.control, [[0.0, 2.0, 3.0]])
    ocp.guess.validate()


def test_unwritten_default_follows_a_change_of_time():
    ocp = timed()
    phase = ocp.guess.phase[0]
    assert phase.state.shape == (2, 3)
    phase.time = [0.0, 0.25, 0.5, 1.0]
    assert phase.state.shape == (2, 4)
    ocp.guess.validate()


@pytest.mark.parametrize("form", SEQUENCE_FORMS)
def test_parameter_accepts_any_real_sequence(form):
    ocp = timed()
    ocp.guess.parameter = SEQUENCE_FORMS[form](2)
    assert_float64_array(ocp.guess.parameter, reference(2))


@pytest.mark.parametrize("form", SCALAR_FORMS)
def test_parameter_and_integral_elements_accept_any_real_scalar(form):
    ocp = timed()
    ocp.guess.parameter[1] = SCALAR_FORMS[form]
    ocp.guess.phase[0].integral[0] = SCALAR_FORMS[form]
    assert_float64_array(ocp.guess.parameter, [0.0, 3.0])
    assert_float64_array(ocp.guess.phase[0].integral, [3.0])


@pytest.mark.parametrize("form", ["list of int", "ndarray float32", "tuple of float"])
def test_integral_accepts_a_real_sequence(form):
    ocp = timed()
    ocp.guess.phase[0].integral = SEQUENCE_FORMS[form](1)
    assert_float64_array(ocp.guess.phase[0].integral, reference(1))


def test_guess_copies_what_it_is_set_from():
    ocp = timed()
    time = np.array([0.0, 1.0, 2.0])
    state = np.zeros((2, 3))
    parameter = np.array([1.0, 2.0])
    ocp.guess.phase[0].time = time
    ocp.guess.phase[0].state = state
    ocp.guess.parameter = parameter
    time[1] = 99.0
    state[0, 0] = 99.0
    parameter[0] = 99.0
    assert ocp.guess.phase[0].time[1] == 1.0
    assert ocp.guess.phase[0].state[0, 0] == 0.0
    assert ocp.guess.parameter[0] == 1.0


def test_accepted_forms_reach_the_nlp_starting_point():
    """A guess set through NumPy types, tuples, and element writes gives the same z0."""
    plain = brachistochrone_minimal.setup()
    varied = brachistochrone_minimal.setup()
    pg, vg = plain.guess.phase[0], varied.guess.phase[0]
    pg.time = [0.0, 1.0]
    pg.state = [[0.0, 1.0], [0.0, 1.0], [0.0, 5.0]]
    pg.control = [[0.0, 0.5]]
    vg.time = np.array([0, 1], dtype=np.int64)
    vg.state = ((0, 1), (np.float32(0), np.float32(1)), (0.0, 5.0))
    vg.control[0, 1] = np.float32(0.5)

    z0 = []
    for ocp in (plain, varied):
        mesh = Mesh(ocp.mesh.phase)
        mesh.set_matrices(ocp.spectral_method)
        z0.append(make_initial_guess_nlp(ocp, mesh))
    assert z0[0].dtype == z0[1].dtype == np.float64
    np.testing.assert_array_equal(z0[0], z0[1])


def test_warm_start_from_a_solution_copies_it():
    ocp = brachistochrone_minimal.setup()
    ocp.ipopt_options.print_level = 0
    solution = ocp.solve()
    ocp.guess(solution)
    phase = ocp.guess.phase[0]
    np.testing.assert_array_equal(phase.time, solution.phase[0].time)
    np.testing.assert_array_equal(phase.state, solution.phase[0].state)
    phase.state[0, 0] = 99.0
    assert solution.phase[0].state[0, 0] != 99.0
    assert ocp.solve().nlp_info.ipopt_status == 0


# ---------------------------------------------------------- what a user may get wrong


@pytest.mark.parametrize(
    "value",
    [[0.0], [1.0, 0.0], [0.0, 0.0], [[0.0, 1.0]]],
    ids=["too short", "decreasing", "repeated", "2-D"],
)
def test_time_that_is_not_strictly_increasing_1d_raises_at_the_assignment(value):
    ocp = problem()
    with raises(ValueError, "guess.phase[0].time", "strictly increasing", at="time ="):
        ocp.guess.phase[0].time = value


def test_state_that_is_not_2d_raises_at_the_assignment():
    ocp = timed()
    with raises(ValueError, "guess.phase[0].state", "2-dimensional", at="state ="):
        ocp.guess.phase[0].state = [0.0, 1.0, 2.0]


def test_state_with_wrong_number_of_rows_raises_at_the_assignment():
    ocp = timed()
    with raises(ValueError, "state", "guess.phase[0]", "2 rows", at="state ="):
        ocp.guess.phase[0].state = [[0.0, 1.0, 2.0]]


def test_state_with_wrong_number_of_columns_is_reported_by_validate():
    ocp = timed()
    ocp.guess.phase[0].state = [[0.0, 1.0], [2.0, 3.0]]
    with raises(ValueError, "guess.phase[0].state", "(2, 3)"):
        ocp.guess.validate()


@pytest.mark.parametrize("name", ["state", "control"])
def test_reading_before_time_raises_naming_time(name):
    ocp = problem()
    with raises(ValueError, f"guess.phase[0].{name}", "guess.phase[0].time", at="getattr"):
        getattr(ocp.guess.phase[0], name)


def test_unset_time_is_reported_by_validate():
    ocp = problem()
    ocp.guess.phase[0].time = [0.0, 1.0]
    with raises(ValueError, "guess.phase[1].time has not been set"):
        ocp.guess.validate()


@pytest.mark.parametrize(
    "value", [[1.0], [1.0, 2.0, 3.0], 1.0], ids=["too short", "too long", "scalar"]
)
def test_parameter_of_wrong_shape_raises_at_the_assignment(value):
    ocp = problem()
    with raises(ValueError, "guess.parameter", "length 2", at="parameter ="):
        ocp.guess.parameter = value


def test_strings_raise_at_the_assignment():
    """Numeric or not, a string is the wrong type: both raise TypeError, naming the target.

    Non-numeric strings used to raise `ValueError` from NumPy's conversion while numeric
    ones were silently converted, so the two differed for no reason a user could act on.
    """
    ocp = timed()
    with raises(TypeError, "guess.phase[0].state", at="state ="):
        ocp.guess.phase[0].state = [["a", "b", "c"], ["d", "e", "f"]]
    with raises(TypeError, "guess.parameter", at="parameter ="):
        ocp.guess.parameter = ["a", "b"]


@pytest.mark.parametrize(
    ("owner", "typo"),
    [(lambda g: g, "parmeter"), (lambda g: g.phase[0], "stat")],
    ids=["Guess", "PhaseGuess"],
)
def test_misspelled_attribute_raises_at_the_assignment(owner, typo):
    ocp = problem()
    with raises(AttributeError, typo, at="setattr"):
        setattr(owner(ocp.guess), typo, [0.0])


# ----------------------------------------------------------------- not yet met


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_time_raises_at_the_assignment(bad):
    ocp = problem()
    with raises(ValueError, "guess.phase[0].time", at="time ="):
        ocp.guess.phase[0].time = [0.0, bad]


@not_yet("redesign (W4)", "columns are checked at the assignment once time is set")
def test_state_with_wrong_number_of_columns_raises_at_the_assignment():
    ocp = timed()
    with raises(ValueError, "guess.phase[0].state", at="state ="):
        ocp.guess.phase[0].state = [[0.0, 1.0], [2.0, 3.0]]


def test_integral_of_wrong_length_message_names_the_attribute():
    ocp = timed()
    with raises(ValueError, "guess.phase[0].integral", at="integral ="):
        ocp.guess.phase[0].integral = [1.0, 2.0]


@not_yet(
    "redesign (W4 from_solution)", "a warm start from something that is not a Solution says so"
)
def test_warm_start_from_a_non_solution_raises_helpfully():
    ocp = problem()
    with raises(TypeError, "Solution"):
        ocp.guess(None)


@not_yet(
    "redesign (W4 from_solution)",
    "a warm start from a mismatched problem raises before changing the guess",
)
def test_warm_start_from_a_mismatched_solution_changes_nothing():
    source = brachistochrone_minimal.setup()
    source.ipopt_options.print_level = 0
    solution = source.solve()
    target = problem()  # different counts
    target.guess.phase[0].time = [0.0, 1.0]
    with pytest.raises((ValueError, TypeError)):
        target.guess(solution)
    np.testing.assert_array_equal(target.guess.phase[0].time, [0.0, 1.0])


def _fill(ocp):
    """Set every guess value of the contract problem to something other than its default."""
    for p, phase in enumerate(ocp.guess.phase):
        phase.time = [0.0, 1.0, 2.0]
        phase.state = np.ones((ocp.nx[p], 3))
        phase.control = np.ones((ocp.nu[p], 3))
        phase.integral = np.ones(ocp.nq[p])
    ocp.guess.parameter = [1.0, 2.0]


def _assert_phase_is_new(ocp, p):
    phase = ocp.guess.phase[p]
    assert phase.time is None
    for name in ("state", "control"):
        with raises(ValueError, f"guess.phase[{p}].time", at="getattr(phase, name)"):
            getattr(phase, name)
    assert_float64_array(phase.integral, np.zeros(ocp.nq[p]))
    phase.time = [0.0, 1.0]
    assert_float64_array(phase.state, np.zeros((ocp.nx[p], 2)))


def test_guess_reset_is_a_new_guess():
    ocp = problem()
    _fill(ocp)
    ocp.guess.reset()
    with raises(ValueError, "guess.phase[0].time has not been set", at="validate()"):
        ocp.guess.validate()
    assert_float64_array(ocp.guess.parameter, [0.0, 0.0])
    for p in range(ocp.np):
        _assert_phase_is_new(ocp, p)


def test_phase_guess_reset_leaves_the_rest():
    ocp = problem()
    _fill(ocp)
    ocp.guess.phase[0].reset()
    _assert_phase_is_new(ocp, 0)
    assert_float64_array(ocp.guess.phase[1].time, [0.0, 1.0, 2.0])
    assert_float64_array(ocp.guess.phase[1].state, np.ones((1, 3)))
    assert_float64_array(ocp.guess.parameter, [1.0, 2.0])


@pytest.mark.parametrize("name", ["state", "control"])
def test_an_assigned_guess_reads_back_before_time_is_set(name):
    ocp = problem()
    rows = ocp.nx[0] if name == "state" else ocp.nu[0]
    setattr(ocp.guess.phase[0], name, np.arange(3 * rows).reshape(rows, 3))
    assert_float64_array(getattr(ocp.guess.phase[0], name), np.arange(3 * rows).reshape(rows, 3))


# Decided 2026-09-14 (Steve): guess values accept real numbers and convert nothing else,
# and must be finite -- checked at a whole assignment, and by validate() for values
# written into elements.
NOT_REAL = {
    "numeric strings": lambda shape: np.arange(np.prod(shape)).astype(str).reshape(shape).tolist(),
    "bools": lambda shape: (np.arange(np.prod(shape)).reshape(shape) > 0).tolist(),
    "complex ndarray": lambda shape: np.arange(np.prod(shape)).reshape(shape) + 1j,
}
# Each form is increasing along its last axis where that is possible, so that a time guess
# fails on its type rather than on the strictly-increasing rule.
TARGETS = {
    "time": ("guess.phase[0].time", lambda g: g.phase[0], "time", (2,)),
    "state": ("guess.phase[0].state", lambda g: g.phase[0], "state", (2, 3)),
    "control": ("guess.phase[0].control", lambda g: g.phase[0], "control", (1, 3)),
    "integral": ("guess.phase[0].integral", lambda g: g.phase[0], "integral", (1,)),
    "parameter": ("guess.parameter", lambda g: g, "parameter", (2,)),
}


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("form", NOT_REAL)
@pytest.mark.parametrize("target", TARGETS)
def test_non_real_guess_raises_at_the_assignment(target, form):
    ocp = timed()
    path, owner, name, shape = TARGETS[target]
    with raises(TypeError, path, at="setattr"):
        setattr(owner(ocp.guess), name, NOT_REAL[form](shape))


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["NaN", "inf"])
@pytest.mark.parametrize("target", ["state", "control", "integral", "parameter"])
def test_non_finite_guess_raises_at_the_assignment(target, bad):
    ocp = timed()
    path, owner, name, shape = TARGETS[target]
    values = np.zeros(shape)
    values.flat[-1] = bad
    with raises(ValueError, path, at="setattr"):
        setattr(owner(ocp.guess), name, values)


@not_yet(
    "redesign (W4)", "a non-finite value written into a guess element is reported by validate()"
)
@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["NaN", "inf"])
@pytest.mark.parametrize("target", ["state", "control", "integral", "parameter"])
def test_non_finite_guess_element_is_reported_by_validate(target, bad):
    ocp = timed()
    path, owner, name, _ = TARGETS[target]
    getattr(owner(ocp.guess), name).flat[-1] = bad
    with raises(ValueError, path):
        ocp.guess.validate()
