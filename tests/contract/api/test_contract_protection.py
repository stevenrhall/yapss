"""What happens to a name that was never declared.

Every container a user touches answers for the names it has and refuses the rest, at the line
where the name is written. This is one rule with one message shape, stated here once rather
than on each page, and the value of it is that a typo is a refusal rather than a setting that
quietly does nothing.
"""

from __future__ import annotations

from ._api import problem, raises, solvable


def test_a_setting_that_does_not_exist_is_refused() -> None:
    """Assigning an unknown name on the problem is a typo, not a new setting."""
    p = problem()
    with raises(AttributeError, "problem has no setting 'nope'", at="p.nope"):
        p.nope = 1


def test_a_phase_setting_that_does_not_exist_is_refused() -> None:
    """The same on a phase, and the message names which phase."""
    p = problem()
    with raises(AttributeError, "phase 'first' has no setting 'nope'", at="ph.nope"):
        ph = p.phases.first
        ph.nope = 1


def test_a_phase_that_was_not_declared_is_refused() -> None:
    """Phases answer for the names the declaration gave them."""
    p = problem()
    with raises(AttributeError, "Phases has no phase 'nope'", at="p.phases.nope"):
        _ = p.phases.nope


def test_a_removed_setting_is_refused_like_any_other() -> None:
    """`ipopt_source` was removed in 0.3.0 and is not a name this API has at all."""
    p = problem()
    with raises(AttributeError, "no setting 'ipopt_source'", at="p.ipopt_source"):
        p.ipopt_source = "x"


def test_a_misspelling_is_refused_where_it_is_written() -> None:
    """Not at the solve: at the assignment, which is the line a user has to change."""
    p = solvable()
    with raises(AttributeError, "no setting", at="p.derivatives.methodd"):
        p.derivatives.methodd = "auto"
