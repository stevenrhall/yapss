"""What a scale is, and why it is never a way to change the answer.

A scale says how large a quantity typically is, so that the solver works in numbers near one.
It conditions the problem and changes nothing about what is being solved -- which is the whole
reason a scale must be positive, and why flipping a sign is `problem.objective.sense` instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ._api import not_yet, problem, raises, solvable

# ------------------------------------------------------------------ a scale is a number


def test_a_scale_is_one_positive_number() -> None:
    """Unlike a bound, a scale is a single number: it says how large, not between what."""
    ph = problem().phases.first
    ph.state.x.scale = 1000.0
    assert ph.state.x.scale == 1000.0


def test_a_scale_is_not_a_pair() -> None:
    """The pair form belongs to bounds, and the message says what a scale is instead."""
    ph = problem().phases.first
    with raises(TypeError, "a scale is a positive number", at="x.scale"):
        ph.state.x.scale = (1, 2)


def test_a_scale_may_not_be_zero() -> None:
    """Dividing by it is the whole of what it does."""
    ph = problem().phases.first
    with raises(ValueError, "must be positive", at="x.scale"):
        ph.state.x.scale = 0.0


def test_a_scale_may_not_be_negative() -> None:
    """A negative scale would flip a sign, which is a change of problem, not of conditioning."""
    ph = problem().phases.first
    with raises(
        ValueError,
        "must be positive",
        "To maximize, set <problem>.objective.sense",
        at="x.scale",
    ):
        ph.state.x.scale = -1.0


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_scale_must_be_finite(value: float) -> None:
    """Refused where it is written, rather than by the solver, which names no field."""
    ph = problem().phases.first
    with raises(ValueError, "phases.first.state.x.scale", "must be a finite number", at="x.scale"):
        ph.state.x.scale = value


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_scale_must_be_finite_in_every_row(value: float) -> None:
    """A block field's rows are each a scale, checked as one."""
    ph = problem().phases.first
    with raises(ValueError, "must be a finite number", at="y.scale"):
        ph.state.y.scale[1] = value


def test_the_objective_and_time_scales_must_be_finite_too() -> None:
    """They are set on their own containers, and go through the same check."""
    p = problem()
    with raises(ValueError, "must be a finite number", at="objective.scale"):
        p.objective.scale = math.nan
    with raises(ValueError, "must be a finite number", at="time.scale"):
        p.phases.first.time.scale = math.inf


def test_the_objective_scale_is_positive_too() -> None:
    """The objective's scale is a magnitude; maximizing is said with 'sense'."""
    p = problem()
    with raises(ValueError, "must be positive", at="objective.scale"):
        p.objective.scale = -1.0


# ------------------------------------------ the dynamics are scaled beside the other outputs


def test_the_dynamics_scale_is_yapss_s_choice_until_set() -> None:
    """None, the default, means YAPSS chooses; a number is the user's choice; reading returns
    what was written, so which of them chose is visible."""
    ph = problem().phases.first
    assert ph.dynamics.x.scale is None
    assert list(ph.dynamics.y.scale) == [None, None]
    ph.dynamics.x.scale = 10.0
    ph.dynamics.y.scale[0] = 3.0
    assert (ph.dynamics.x.scale, list(ph.dynamics.y.scale)) == (10.0, [3.0, None])
    ph.dynamics.x.scale = None
    assert ph.dynamics.x.scale is None


def test_the_dynamics_have_a_scale_and_nothing_else() -> None:
    """A defect is an equality, so there is nothing to bound."""
    ph = problem().phases.first
    with raises(AttributeError, "phases.first.dynamics.x has no setting 'bounds'", at="bounds"):
        ph.dynamics.x.bounds = (0.0, 1.0)


def test_only_the_dynamics_scale_takes_none() -> None:
    """It alone has a natural source, the state it belongs to; any other scale is a number."""
    ph = problem().phases.first
    with raises(TypeError, "a scale is a positive number; got None", at="ph.state.x.scale"):
        ph.state.x.scale = None  # type: ignore[assignment]


def test_arithmetic_on_rows_yapss_chooses_is_refused() -> None:
    """There is no number to multiply, and NumPy would carry None on as NaN."""
    ph = problem().phases.first
    with raises(ValueError, "YAPSS chooses the scale", "set the rows first", at="*= 2"):
        ph.dynamics.y.scale[:] *= 2


def test_scaling_does_not_change_the_answer() -> None:
    """The point of the rule above, stated as a solve rather than as a refusal."""
    plain = solvable()
    scaled = solvable()
    ph = scaled.phases.slide
    ph.state.x.scale = 2.0
    ph.state.v.scale = 5.0
    ph.dynamics.v.scale = 3.0
    ph.time.scale = 0.5
    scaled.objective.scale = 0.5
    assert abs(plain.solve().objective - scaled.solve().objective) < 1e-6


def test_a_boolean_is_not_a_scale() -> None:
    """The rule every aspect taking a number shares."""
    ph = problem().phases.first
    with raises(TypeError, "a boolean is not a number", at="x.scale"):
        ph.state.x.scale = True


def test_a_time_scale_is_a_positive_number() -> None:
    """The phase's own extent is scaled like anything else."""
    ph = problem().phases.first
    with raises(ValueError, "must be positive", at="time.scale"):
        ph.time.scale = 0.0


def test_a_path_constraint_has_a_scale() -> None:
    """Constraints are scaled too, which is what keeps a residual near one."""
    ph = problem().phases.first
    ph.path.g.scale = 7.0
    assert ph.path.g.scale == 7.0


def test_a_block_field_is_scaled_row_by_row() -> None:
    """One number per row, because a scale is one number and rows differ in size."""
    ph = problem().phases.first
    ph.state.y.scale[:] = 2.0
    assert list(ph.state.y.scale) == [2.0, 2.0]


def test_a_block_fields_scales_can_be_multiplied_in_place() -> None:
    """Scaling every row up at once is arithmetic on numbers, and reads as it would on an
    array: the rows read back as numbers, and the product is assigned back through [:]."""
    ph = problem().phases.first
    ph.state.y.scale[:] = 2.0
    ph.state.y.scale[:] *= 3
    assert list(ph.state.y.scale) == [6.0, 6.0]
    ph.state.y.scale[:] = ph.state.y.scale[:] / 2
    assert list(ph.state.y.scale) == [3.0, 3.0]


def test_an_array_of_numbers_gives_one_scale_per_row() -> None:
    """A scale is one number, so a 1-D array can only be one per row, as NumPy reads it."""
    ph = problem().phases.first
    ph.state.y.scale[:] = np.array([2.0, 5.0])
    assert list(ph.state.y.scale) == [2.0, 5.0]


def test_what_comes_back_in_through_an_in_place_operator_is_checked() -> None:
    """The product is an assignment like any other, so it cannot store a scale that is not
    positive."""
    ph = problem().phases.first
    with raises(ValueError, "must be positive", at="*= -1"):
        ph.state.y.scale[:] *= -1


def test_a_write_into_scale_rows_read_back_is_refused() -> None:
    """The rows read back are a copy; a write into one would be lost, so it is refused, and the
    message names the assignment that changes the setting."""
    ph = problem().phases.first
    rows = ph.state.y.scale[:]
    with raises(
        ValueError, "a copy", "'<problem>.phases.first.state.y.scale[0] = ...'", at="rows[0]"
    ):
        rows[0] = 5.0


def test_a_bounds_rows_still_read_as_bounds() -> None:
    """Only a number-valued setting reads back as an array: a bound is a pair."""
    ph = problem().phases.first
    ph.state.y.bounds[:] = (0.0, 1.0)
    assert ph.state.y.bounds[:] == ((0.0, 1.0), (0.0, 1.0))


def test_the_objective_scale_message_points_to_sense() -> None:
    """A negative objective scale is almost always an attempt to maximize."""
    p = problem()
    with raises(ValueError, "objective.sense = 'maximize'", at="objective.scale"):
        p.objective.scale = -1.0


@pytest.mark.parametrize("value", ["2", 2 + 0j])
def test_time_and_objective_scales_are_real_numbers(value: object) -> None:
    """A string or a complex number is refused, on the scales set outside any field too."""
    p = problem()
    with raises(TypeError, "a scale is a positive number", at="time.scale"):
        p.phases.first.time.scale = value  # type: ignore[assignment]
    with raises(TypeError, "a scale is a positive number", at="objective.scale"):
        p.objective.scale = value  # type: ignore[assignment]


def test_a_misspelled_scale_setting_is_refused_with_a_suggestion() -> None:
    """The objective's settings are checked like a field's."""
    p = problem()
    with raises(AttributeError, "'scal'", "Did you mean 'scale'", at="objective.scal"):
        p.objective.scal = 2.0  # type: ignore[attr-defined]
