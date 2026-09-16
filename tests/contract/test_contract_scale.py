"""Contract: `problem.scale`.

What a user may do
    - Assign a whole scale array (per phase: state, control, integral, dynamics, path;
      problem: discrete, parameter) as any 1-D sequence of finite positive real numbers of
      the right length.
    - Assign one element, or a slice, with a finite positive real number or a matching
      sequence; a scalar broadcasts over a slice.
    - Assign `scale.phase[p].time` and `scale.objective` any finite positive real number,
      NumPy scalars included.
    - Rely on the defaults (all ones), on a scale copying what it is set from, and on every
      accepted value reaching the NLP scaling.
    - Return every factor to 1.0 with `scale.reset()`, or one phase's with
      `scale.phase[p].reset()`; an array read before the reset sees the ones.

What a user may get wrong
    - A zero, negative, NaN, or infinite factor in a whole assignment, or in `time` or
      `objective`: `ValueError` at the assignment, naming the scale.
    - The same written into an element or slice: `ValueError` from `validate()` (run by
      `solve()`), naming the element.
    - A negative `objective`: the message points to `problem.sense`.
    - Wrong length or a scalar for a whole array: `ValueError` at the assignment.
    - A misspelled attribute: `AttributeError` at the assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from yapss._private.solver import get_nlp_scaling
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

# (name used in whole-assignment messages, name used by validate(), accessor, owner)
ARRAYS = {
    "phase[0].state": (
        "Scale 'state' in phase 0",
        "scale.phase[0].state",
        lambda s: s.phase[0],
        "state",
    ),
    "phase[1].control": (
        "Scale 'control' in phase 1",
        "scale.phase[1].control",
        lambda s: s.phase[1],
        "control",
    ),
    "phase[0].integral": (
        "Scale 'integral' in phase 0",
        "scale.phase[0].integral",
        lambda s: s.phase[0],
        "integral",
    ),
    "phase[0].dynamics": (
        "Scale 'dynamics' in phase 0",
        "scale.phase[0].dynamics",
        lambda s: s.phase[0],
        "dynamics",
    ),
    "phase[1].path": (
        "Scale 'path' in phase 1",
        "scale.phase[1].path",
        lambda s: s.phase[1],
        "path",
    ),
    "discrete": ("Scale 'discrete'", "scale.discrete", lambda s: s, "discrete"),
    "parameter": ("Scale 'parameter'", "scale.parameter", lambda s: s, "parameter"),
}
BAD_FACTORS = {"zero": 0.0, "negative": -1.0, "NaN": np.nan, "inf": np.inf}


def get(ocp, key):
    _, _, owner, name = ARRAYS[key]
    return getattr(owner(ocp.scale), name)


# ---------------------------------------------------------------- what a user may do


@pytest.mark.parametrize("form", SEQUENCE_FORMS)
@pytest.mark.parametrize("key", ARRAYS)
def test_whole_array_accepts_any_positive_real_sequence(key, form):
    ocp = problem()
    _, _, owner, name = ARRAYS[key]
    n = len(get(ocp, key))
    setattr(owner(ocp.scale), name, SEQUENCE_FORMS[form](n))
    assert_float64_array(get(ocp, key), reference(n))
    ocp.scale.validate()


@pytest.mark.parametrize("form", SCALAR_FORMS)
@pytest.mark.parametrize("key", ARRAYS)
def test_element_accepts_any_positive_real_scalar(key, form):
    ocp = problem()
    get(ocp, key)[-1] = SCALAR_FORMS[form]
    assert get(ocp, key).dtype == np.float64
    assert get(ocp, key)[-1] == 3.0
    ocp.scale.validate()


def test_slice_accepts_scalar_broadcast_and_sequence():
    ocp = problem()
    state = ocp.scale.phase[0].state
    state[:] = 3
    assert_float64_array(state, [3.0, 3.0])
    state[1:] = [np.float32(4)]
    assert_float64_array(state, [3.0, 4.0])
    ocp.scale.validate()


@pytest.mark.parametrize("form", SCALAR_FORMS)
def test_time_and_objective_accept_any_positive_real_number(form):
    ocp = problem()
    ocp.scale.phase[1].time = SCALAR_FORMS[form]
    ocp.scale.objective = SCALAR_FORMS[form]
    assert type(ocp.scale.phase[1].time) is float
    assert type(ocp.scale.objective) is float
    assert ocp.scale.phase[1].time == ocp.scale.objective == 3.0
    ocp.scale.validate()


def test_defaults_are_ones():
    ocp = problem()
    for key in ARRAYS:
        assert_float64_array(get(ocp, key), np.ones(len(get(ocp, key))))
    assert ocp.scale.phase[0].time == ocp.scale.objective == 1.0


def test_scale_copies_what_it_is_set_from():
    ocp = problem()
    source = np.array([1.0, 2.0])
    ocp.scale.phase[0].state = source
    source[0] = 99.0
    assert_float64_array(ocp.scale.phase[0].state, [1.0, 2.0])


def test_accepted_values_reach_the_nlp_scaling():
    """Scales set by elements, slices, and NumPy types give the same NLP scaling."""
    plain = brachistochrone_minimal.setup()
    varied = brachistochrone_minimal.setup()
    plain.scale.phase[0].state = [2.0, 2.0, 4.0]
    varied.scale.phase[0].state[:2] = np.int64(2)
    varied.scale.phase[0].state[2] = np.float32(4)
    plain.scale.phase[0].time = 0.5
    varied.scale.phase[0].time = np.float32(0.5)
    plain.scale.objective = 3.0
    varied.scale.objective = np.int64(3)

    objective_plain, z_plain, c_plain = get_nlp_scaling(plain)
    objective_varied, z_varied, c_varied = get_nlp_scaling(varied)
    assert objective_plain == objective_varied
    np.testing.assert_array_equal(z_plain, z_varied)
    np.testing.assert_array_equal(c_plain, c_varied)
    assert 0.25 in z_plain  # a state scale of 4 enters Ipopt as its reciprocal


# ---------------------------------------------------------- what a user may get wrong


@pytest.mark.parametrize("bad", BAD_FACTORS)
@pytest.mark.parametrize("key", ARRAYS)
def test_bad_factor_in_a_whole_assignment_raises_at_the_assignment(key, bad):
    ocp = problem()
    label, _, owner, name = ARRAYS[key]
    n = len(get(ocp, key))
    values = [1.0] * n
    values[-1] = BAD_FACTORS[bad]
    with raises(ValueError, label, "must be finite and positive", at="setattr"):
        setattr(owner(ocp.scale), name, values)
    assert_float64_array(get(ocp, key), np.ones(n))  # the previous value is kept


@pytest.mark.parametrize("bad", BAD_FACTORS)
@pytest.mark.parametrize("key", ARRAYS)
def test_bad_factor_written_into_an_element_is_reported_by_validate(key, bad):
    ocp = problem()
    values = get(ocp, key)
    values[-1] = BAD_FACTORS[bad]
    _, path, _, _ = ARRAYS[key]
    with raises(ValueError, f"{path}[{len(values) - 1}] must be finite and positive"):
        ocp.scale.validate()


@pytest.mark.parametrize("bad", BAD_FACTORS)
def test_bad_time_factor_raises_at_the_assignment(bad):
    ocp = problem()
    with raises(ValueError, "Scale 'time' in phase 0", "must be finite and positive", at="time ="):
        ocp.scale.phase[0].time = BAD_FACTORS[bad]


@pytest.mark.parametrize("bad", BAD_FACTORS)
def test_bad_objective_factor_raises_at_the_assignment_and_points_to_sense(bad):
    ocp = problem()
    with raises(ValueError, "scale.objective", "problem.sense", at="objective ="):
        ocp.scale.objective = BAD_FACTORS[bad]


def test_validate_runs_before_solve():
    ocp = brachistochrone_minimal.setup()
    ocp.scale.phase[0].dynamics[0] = 0.0
    with raises(ValueError, "scale.phase[0].dynamics[0] must be finite and positive"):
        ocp.solve()


@pytest.mark.parametrize(
    "value", [[1.0], [1.0, 2.0, 3.0], 1.0], ids=["too short", "too long", "scalar"]
)
def test_whole_array_of_wrong_shape_raises_at_the_assignment(value):
    ocp = problem()
    with raises(ValueError, "Scale 'state' in phase 0", "length 2", at="state ="):
        ocp.scale.phase[0].state = value


@pytest.mark.parametrize(
    ("owner", "typo"),
    [(lambda s: s, "objectve"), (lambda s: s.phase[0], "stat")],
    ids=["Scale", "ScalePhase"],
)
def test_misspelled_attribute_raises_at_the_assignment(owner, typo):
    ocp = problem()
    with raises(AttributeError, typo, at="setattr"):
        setattr(owner(ocp.scale), typo, 1.0)


# ----------------------------------------------------------------- not yet met


@not_yet("E2 part 2", "a bad factor written into an element raises at the assignment")
def test_bad_element_raises_at_the_assignment():
    ocp = problem()
    with raises(ValueError, "scale.phase[0].state", at="state[0] ="):
        ocp.scale.phase[0].state[0] = 0.0


PHASE_ARRAYS = ("state", "control", "integral", "dynamics", "path")


def _fill(ocp):
    """Set every scale factor of the contract problem to 2.0."""
    for phase in ocp.scale.phase:
        for name in PHASE_ARRAYS:
            setattr(phase, name, 2 * np.ones(len(getattr(phase, name))))
        phase.time = 2.0
    ocp.scale.discrete = [2.0, 2.0]
    ocp.scale.parameter = [2.0, 2.0]
    ocp.scale.objective = 2.0


def _assert_phase_is_ones(phase, value=1.0):
    for name in PHASE_ARRAYS:
        array = getattr(phase, name)
        assert_float64_array(array, value * np.ones(len(array)))
    assert phase.time == value


def test_scale_reset_restores_ones():
    ocp = problem()
    _fill(ocp)
    held = ocp.scale.phase[0].state
    ocp.scale.reset()
    for phase in ocp.scale.phase:
        _assert_phase_is_ones(phase)
    assert_float64_array(ocp.scale.discrete, [1.0, 1.0])
    assert_float64_array(ocp.scale.parameter, [1.0, 1.0])
    assert ocp.scale.objective == 1.0
    assert type(ocp.scale.objective) is float
    assert held is ocp.scale.phase[0].state


def test_scale_phase_reset_leaves_the_rest():
    ocp = problem()
    _fill(ocp)
    ocp.scale.phase[0].reset()
    _assert_phase_is_ones(ocp.scale.phase[0])
    _assert_phase_is_ones(ocp.scale.phase[1], 2.0)
    assert_float64_array(ocp.scale.parameter, [2.0, 2.0])
    assert ocp.scale.objective == 2.0


def test_scale_phase_is_read_only():
    ocp = problem()
    with raises(AttributeError, at="phase ="):
        ocp.scale.phase = ()


# Decided 2026-09-14 (Steve): scale factors accept real numbers and convert nothing else.
NOT_REAL = {
    "numeric strings": ["1", "2"],
    "bools": [True, True],
    "complex ndarray": np.array([1 + 1j, 2.0]),
}


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("form", NOT_REAL)
def test_non_real_whole_array_raises_at_the_assignment(form):
    ocp = problem()
    with raises(TypeError, "Scale 'state' in phase 0", at="state ="):
        ocp.scale.phase[0].state = NOT_REAL[form]


@pytest.mark.parametrize("value", ["2", True, 2 + 0j], ids=["numeric string", "bool", "complex"])
@pytest.mark.parametrize("name", ["time", "objective"])
def test_non_real_time_or_objective_raises_at_the_assignment(name, value):
    ocp = problem()
    owner, label = (
        (ocp.scale.phase[0], "Scale 'time' in phase 0")
        if name == "time"
        else (ocp.scale, "scale.objective")
    )
    with raises(TypeError, label, at="setattr"):
        setattr(owner, name, value)
