"""What a scale is, and why it is never a way to change the answer.

A scale says how large a quantity typically is, so that the solver works in numbers near one.
It conditions the problem and changes nothing about what is being solved -- which is the whole
reason a scale must be positive, and why flipping a sign is `problem.objective.sense` instead.
"""

from __future__ import annotations

from ._api import problem, raises, solvable

# ------------------------------------------------------------------ a scale is a number


def test_a_scale_is_one_positive_number() -> None:
    """Unlike a bound, a scale is a single number: it says how large, not between what."""
    ph = problem().phases.first
    ph.state.scale.x = 1000.0
    assert ph.state.scale.x == 1000.0


def test_a_scale_is_not_a_pair() -> None:
    """The pair form belongs to bounds, and the message says what a scale is instead."""
    ph = problem().phases.first
    with raises(TypeError, "a scale is a positive number", at="scale.x"):
        ph.state.scale.x = (1, 2)


def test_a_scale_may_not_be_zero() -> None:
    """Dividing by it is the whole of what it does."""
    ph = problem().phases.first
    with raises(ValueError, "must be positive", at="scale.x"):
        ph.state.scale.x = 0.0


def test_a_scale_may_not_be_negative() -> None:
    """A negative scale would flip a sign, which is a change of problem, not of conditioning."""
    ph = problem().phases.first
    with raises(
        ValueError,
        "must be positive",
        "never changes what it means",
        at="scale.x",
    ):
        ph.state.scale.x = -1.0


def test_the_objective_scale_is_positive_too() -> None:
    """The objective's scale is a magnitude; maximizing is said with 'sense'."""
    p = problem()
    with raises(ValueError, "must be positive", at="objective.scale"):
        p.objective.scale = -1.0


# ------------------------------------------------------- a state has two scales, not one


def test_a_state_has_a_scale_and_a_defect_scale() -> None:
    """How large the state is, and how large its defect is, are different questions."""
    ph = problem().phases.first
    ph.state.scale.x = 1000.0
    ph.state.defect_scale.x = 10.0
    assert (ph.state.scale.x, ph.state.defect_scale.x) == (1000.0, 10.0)


def test_scaling_does_not_change_the_answer() -> None:
    """The point of the rule above, stated as a solve rather than as a refusal."""
    plain = solvable()
    scaled = solvable()
    ph = scaled.phases.slide
    ph.state.scale.x = ph.state.defect_scale.x = 2.0
    ph.state.scale.v = ph.state.defect_scale.v = 5.0
    ph.time.scale = 0.5
    scaled.objective.scale = 0.5
    assert abs(plain.solve().objective - scaled.solve().objective) < 1e-6


def test_a_boolean_is_not_a_scale() -> None:
    """The rule every aspect taking a number shares."""
    ph = problem().phases.first
    with raises(TypeError, "a boolean is not a number", at="scale.x"):
        ph.state.scale.x = True


def test_a_time_scale_is_a_positive_number() -> None:
    """The phase's own extent is scaled like anything else."""
    ph = problem().phases.first
    with raises(ValueError, "must be positive", at="time.scale"):
        ph.time.scale = 0.0


def test_a_path_constraint_has_a_scale() -> None:
    """Constraints are scaled too, which is what keeps a residual near one."""
    ph = problem().phases.first
    ph.path.scale.g = 7.0
    assert ph.path.scale.g == 7.0


def test_a_block_field_is_scaled_row_by_row() -> None:
    """One number per row, because a scale is one number and rows differ in size."""
    ph = problem().phases.first
    ph.state.scale.y[:] = 2.0
    assert list(ph.state.scale.y) == [2.0, 2.0]
