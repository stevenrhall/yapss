"""What a bound is, and how an aspect of a vector is written.

A bound is a pair, and this is the first page on which a *value* is put into a declaration, so
the mechanics of writing one -- a whole field, one row, a slice, a row count that must match --
are stated here rather than repeated on every aspect's page. The guess and the scale pages
state what differs about them, not what they share with this.
"""

from __future__ import annotations

import numpy as np

import yapss

from ._api import State, not_yet, problem, proposed, raises

# ------------------------------------------------------------------- a bound is a pair


def test_a_bound_is_a_pair_not_a_number() -> None:
    """The commonest mistake, and the message hands back the fix already written out."""
    ph = problem().phases.first
    with raises(
        TypeError,
        "a bound is a pair",
        "write (500.0, 500.0)",
        at="bounds.x",
    ):
        ph.state.bounds.x = 500.0


def test_a_bound_of_none_names_the_pair_that_means_unbounded() -> None:
    """`None` is how one side is left open, so the message says how to open both."""
    ph = problem().phases.first
    with raises(TypeError, "write (None, None)", at="bounds.x"):
        ph.state.bounds.x = None


def test_a_bound_takes_exactly_two_values() -> None:
    """Three values is not an interval, and the message counts them."""
    ph = problem().phases.first
    with raises(ValueError, "got 3 values", at="bounds.x"):
        ph.state.bounds.x = (1, 2, 3)


def test_a_bound_may_not_be_inverted() -> None:
    """A lower bound above its upper bound has no solutions, so it is refused where written."""
    ph = problem().phases.first
    with raises(ValueError, "lower 2.0 > upper 1.0", at="bounds.x"):
        ph.state.bounds.x = (2, 1)


def test_each_side_of_a_bound_is_a_number_or_none() -> None:
    """Anything else is a mistake about what a bound is."""
    ph = problem().phases.first
    with raises(TypeError, "a number or None", at="bounds.x"):
        ph.state.bounds.x = (0, "a")


def test_a_boolean_is_not_a_number() -> None:
    """True is 1 to Python and a mistake here, most often a comparison written by accident."""
    ph = problem().phases.first
    with raises(TypeError, "a boolean is not a number", "comparison", at="bounds.x"):
        ph.state.bounds.x = True


def test_one_side_may_be_left_open() -> None:
    """The accepted form, which is what makes the refusals above about shape and not about None."""
    ph = problem().phases.first
    ph.state.bounds.x = (0.0, None)
    assert ph.state.bounds.x == (0.0, np.inf)


def test_leaving_one_side_unchanged_is_not_built_yet() -> None:
    """`...` is designed and not built, and says so rather than doing something else."""
    ph = problem().phases.first
    with raises(TypeError, "not implemented yet", "give both ends", at="bounds.x"):
        ph.state.bounds.x = (0.0, ...)


# -------------------------------------------------------- writing a field that has rows


def test_a_block_field_refuses_a_bare_name() -> None:
    """A field of several rows cannot be written by name alone: it cannot show what was meant."""
    ph = problem().phases.first
    with raises(TypeError, "has 2 rows, so say which", at="bounds.y"):
        ph.state.bounds.y = (0.0, 1.0)


def test_a_block_field_takes_the_same_bound_for_every_row() -> None:
    """`[:]` with one bound gives every row that bound, which is the common case."""
    ph = problem().phases.first
    ph.state.bounds.y[:] = (0.0, 1.0)
    assert list(ph.state.bounds.y) == [(0.0, 1.0), (0.0, 1.0)]


def test_a_block_field_takes_one_bound_per_row() -> None:
    """A sequence of bounds gives the rows their own, in order."""
    ph = problem().phases.first
    ph.state.bounds.y[:] = [(0.0, 1.0), (2.0, 3.0)]
    assert list(ph.state.bounds.y) == [(0.0, 1.0), (2.0, 3.0)]


def test_a_row_may_be_written_by_index() -> None:
    """One row at a time, which is what an index is for."""
    ph = problem().phases.first
    ph.state.bounds.y[1] = (4.0, 5.0)
    assert ph.state.bounds.y[1] == (4.0, 5.0)


def test_the_number_of_rows_must_match() -> None:
    """Three bounds for two rows is a mistake about the declaration, and the message counts both."""
    ph = problem().phases.first
    with raises(ValueError, "covers 2 rows; got 3 values", at="bounds.y"):
        ph.state.bounds.y[:] = [(0.0, 1.0)] * 3


@not_yet(
    "message",
    "indexing a block field by name gives Python's own comparison error, not a YAPSS message",
)
def test_an_index_is_an_integer_or_a_slice() -> None:
    """Rows are addressed by position, and a name is not a position.

    What a user gets today is ``'>=' not supported between instances of 'str' and 'int'``,
    raised by the index check comparing the name against the row count. The message YAPSS
    already has for this is written and unreached.
    """
    ph = problem().phases.first
    with raises(TypeError, "integers or slices", at="bounds.y"):
        ph.state.bounds.y["x"] = (0.0, 1.0)


def test_an_index_out_of_range_names_the_fields() -> None:
    """The message says how many rows there are and what they are called."""
    ph = problem().phases.first
    with raises(IndexError, "has 2 rows; there is no row 5", at="bounds.y"):
        ph.state.bounds.y[5] = (0.0, 1.0)


# ------------------------------------------------------------------ reading a bound back


def test_a_bound_never_written_is_unbounded() -> None:
    """A bound has a default, and the default is no bound at either end.

    This is what separates a bound from a guess: leaving a bound out is a statement, so it
    reads back rather than refusing, while a guess left out has nothing to fall back on.
    """
    ph = problem().phases.first
    assert ph.state.bounds.x == (-np.inf, np.inf)


def test_a_bound_written_reads_back() -> None:
    """What was stored is what is read, converted to floats."""
    ph = problem().phases.first
    ph.state.bounds.x = (0, 1)
    assert ph.state.bounds.x == (0.0, 1.0)


def test_a_misspelled_field_is_refused_with_a_suggestion() -> None:
    """The name is checked against the declaration, and a near miss is named."""
    ph = problem().phases.first
    with raises(AttributeError, "xx", at="bounds.xx"):
        ph.state.bounds.xx = (0.0, 1.0)


# ------------------------------------------------------------- the declaration is not a value


def test_a_declaration_is_not_a_value() -> None:
    """Building one says where the values go instead."""
    with raises(TypeError, "is a declaration, not a value", "phase.state.bounds.", at="State("):
        State(x=(0.0, 1.0))


@proposed("a declaration is sealed after it is written, so a field cannot be replaced on it")
def test_a_field_cannot_be_replaced_on_the_declaration() -> None:
    """Assigning to a declared field on the class silently replaces the field today.

    `State.x = (0.0, 1.0)` is accepted and leaves a declaration whose class attribute is no
    longer a field, which nothing afterwards mentions. Whether this is worth sealing against
    is not decided: it is not something a user does by accident in the way the other mistakes
    on this page are.

    Declared here rather than taken from the shared declarations, because the assignment under
    test succeeds: done to a shared class it would leave every later clause reading a
    declaration this one broke.
    """

    class Local(yapss.Vector):
        x = yapss.field()

    with raises(TypeError, "is a declaration", at="Local.x"):
        Local.x = (0.0, 1.0)


# ---------------------------------------------------- the vector itself, addressed wrongly


def test_a_row_given_a_sequence_of_elements_is_refused() -> None:
    """One row takes one element; a sequence of them is a mistake about which row is meant."""
    ph = problem().phases.first
    with raises(
        TypeError,
        "is one row, so it takes one element",
        at="bounds.y[0]",
    ):
        ph.state.bounds.y[0] = [(0.0, 1.0), (2.0, 3.0)]


def test_a_slice_takes_one_element_or_one_per_row() -> None:
    """Anything else cannot be spread over the rows, and the message says both forms."""
    ph = problem().phases.first
    with raises(
        TypeError,
        "takes one element for all of them or 2 of them",
        at="bounds.y[:]",
    ):
        ph.state.bounds.y[:] = 5


def test_a_vector_cannot_be_read_by_slice() -> None:
    """A vector is a namespace, not a sequence: its fields are reached by name."""
    ph = problem().phases.first
    with raises(TypeError, "cannot be read by slice", "by name", at="ph.state.bounds[0:1]"):
        _ = ph.state.bounds[0:1]


def test_a_vector_cannot_be_written_by_position() -> None:
    """Positions are what this API replaced, so writing one says how to write a name."""
    ph = problem().phases.first
    with raises(
        TypeError,
        "cannot be set by position",
        "for example 'x = ...'",
        at="ph.state.bounds[0]",
    ):
        ph.state.bounds[0] = (0.0, 1.0)


def test_a_vector_index_out_of_range_names_the_fields() -> None:
    """Reading a row by position is allowed, and the message names what the rows are."""
    ph = problem().phases.first
    with raises(IndexError, "out of range for 3 rows", "(x, y)", at="ph.state.bounds[9]"):
        _ = ph.state.bounds[9]


def test_a_vector_index_is_not_a_name() -> None:
    """A name is reached as an attribute; an index is a position."""
    ph = problem().phases.first
    with raises(TypeError, "indices are integers or slices, not str", at="ph.state.bounds["):
        _ = ph.state.bounds["x"]
