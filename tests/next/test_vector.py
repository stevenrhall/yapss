"""Tests of the redesigned API's Vector declaration and its element kinds."""

import math
import re

import numpy as np
import pytest

from yapss._next import Empty, Vector, field
from yapss._next.kinds import Bounds, Guess, ReadOnlyRows, Rows


class Rocket(Vector):
    """Three plain fields."""

    h = field(units="ft", latex="h", doc="altitude")
    v = field(units="ft/s")
    m = field()


class Ascent(Vector):
    """Two block fields and a plain one."""

    r = field(size=3)
    v = field(size=3)
    m = field()


# --- declaration --------------------------------------------------------------------------


def test_fields_are_collected_in_declaration_order():
    assert Rocket._fields == ("h", "v", "m")
    assert len(Rocket._new(Bounds, "x")) == 3


def test_metadata_is_kept_and_the_marker_is_removed():
    assert Rocket._meta["h"].units == "ft"
    assert Rocket._meta["h"].latex == "h"
    assert not hasattr(Rocket, "h")


def test_block_fields_occupy_consecutive_rows():
    assert Ascent._rows[:4] == (("r", 0), ("r", 1), ("r", 2), ("v", 0))
    assert Ascent._nrows == 7
    assert Ascent._offsets == {"r": 0, "v": 3, "m": 6}


def test_a_plain_value_in_a_field_position_is_refused():
    with pytest.raises(TypeError, match=r"A\.r is not a field"):

        class A(Vector):
            r = 3.0


def test_behavior_on_a_declaration_is_refused():
    with pytest.raises(TypeError, match=r"holds no behavior"):

        class A(Vector):
            h = field()

            def go(self):
                """Not allowed."""


@pytest.mark.parametrize("body", ["h: float = field()", "h: float"])
def test_annotated_fields_are_refused(body):
    namespace = {"Vector": Vector, "field": field}
    with pytest.raises(TypeError, match=r"is annotated"):
        exec(f"class A(Vector):\n    {body}", namespace)


def test_inheriting_a_declaration_is_refused():
    with pytest.raises(TypeError, match=r"cannot inherit from the vector declaration"):

        class A(Rocket):
            w = field()


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"size": -1}, ValueError, "must be 0 or more"),
        ({"size": 2.5}, TypeError, "must be an integer"),
        ({"size": True}, TypeError, "must be an integer"),
        ({"units": 3}, TypeError, "must be a string"),
    ],
)
def test_field_arguments_are_checked_at_declaration(kwargs, error, match):
    with pytest.raises(error, match=match):
        field(**kwargs)


def test_a_declaration_cannot_be_instantiated():
    with pytest.raises(TypeError, match=r"Rocket is a declaration, not a value"):
        Rocket(h=1.0, v=2.0, m=3.0)


def test_empty_has_no_fields_and_says_so():
    empty = Empty._new(Rows, "phase 'boost' path")
    with pytest.raises(AttributeError, match=r"has no fields"):
        empty.switching = 1.0


# --- bounds -------------------------------------------------------------------------------


@pytest.fixture
def bounds():
    return Rocket._new(Bounds, "phase 'boost' state bounds")


def test_a_bound_defaults_to_free(bounds):
    assert bounds.h == (-math.inf, math.inf)


@pytest.mark.parametrize(
    ("written", "stored"),
    [
        (None, (-math.inf, math.inf)),
        (3.0, (3.0, 3.0)),
        (3, (3.0, 3.0)),
        ((0, 1), (0.0, 1.0)),
        ((None, 1), (-math.inf, 1.0)),
        ((0, None), (0.0, math.inf)),
        ((2, 2), (2.0, 2.0)),
    ],
)
def test_a_bound_is_stored_normalized(bounds, written, stored):
    bounds.h = written
    assert bounds.h == stored


@pytest.mark.parametrize(
    ("written", "error", "match"),
    [
        ("fast", TypeError, "must be a number, None, or a"),
        ((2, 1), ValueError, r"lower 2\.0 > upper 1\.0"),
        ((0, 1, 2), ValueError, "a bound tuple is"),
        ((2, False), TypeError, "a boolean is not a number"),
        (True, TypeError, "a boolean is not a number"),
        ([0, 1], TypeError, "holds one row, so it takes one value"),
        (..., TypeError, "must be a number, None, or a"),
    ],
)
def test_a_bad_bound_is_refused_at_the_line(bounds, written, error, match):
    with pytest.raises(error, match=match):
        bounds.h = written


def test_a_misspelled_field_suggests_the_right_one(bounds):
    with pytest.raises(AttributeError, match=r"has no field 'hh'\. Did you mean 'h'\?"):
        bounds.hh = 1.0


def test_bounds_cannot_be_set_by_position(bounds):
    with pytest.raises(TypeError, match="cannot be set by position"):
        bounds[0] = 1.0


def test_one_bound_covers_every_row_of_a_block_field():
    bounds = Ascent._new(Bounds, "phase 'ascent' state bounds")
    bounds.r = (-1.0, 1.0)
    assert bounds.r == (-1.0, 1.0)
    assert bounds._elements("r") == ((-1.0, 1.0),) * 3


def test_a_list_gives_a_block_field_one_bound_per_row():
    """A list is per row; a tuple is one value for the whole field."""
    bounds = Ascent._new(Bounds, "phase 'ascent' state bounds")
    bounds.r = [1.0, (0, 2), None]
    assert bounds._elements("r") == ((1.0, 1.0), (0.0, 2.0), (-math.inf, math.inf))


def test_a_per_row_list_must_have_one_value_per_row():
    bounds = Ascent._new(Bounds, "phase 'ascent' state bounds")
    with pytest.raises(ValueError, match=r"'r' has 3 rows; got 2 values"):
        bounds.r = [1.0, 2.0]


def test_a_two_row_block_refuses_a_bare_tuple_as_ambiguous():
    """On a two-row field, `(0, 1)` reads equally as one interval or as two values."""

    class Pair(Vector):
        w = field(size=2)

    bounds = Pair._new(Bounds, "bounds")
    with pytest.raises(TypeError, match="reads two ways"):
        bounds.w = (0, 1)
    bounds.w = [0, 1]
    assert bounds._elements("w") == ((0.0, 0.0), (1.0, 1.0))
    bounds.w = [(0, 1), (0, 1)]
    assert bounds._elements("w") == ((0.0, 1.0), (0.0, 1.0))


# --- guess --------------------------------------------------------------------------------


def test_an_unset_guess_is_zero():
    guess = Rocket._new(Guess, "phase 'boost' state guess")
    assert guess.h == ("constant", 0.0)


def test_a_guess_is_a_constant_or_a_pair():
    guess = Rocket._new(Guess, "phase 'boost' state guess")
    guess.h = 5.0
    guess.v = (0, 100)
    assert guess.h == ("constant", 5.0)
    assert guess.v == ("linear", 0.0, 100.0)


@pytest.mark.parametrize(
    ("written", "error"),
    [(None, TypeError), ("x", TypeError), ((0, "x"), TypeError), ((0, 1, 2), ValueError)],
)
def test_a_bad_guess_is_refused(written, error):
    guess = Rocket._new(Guess, "phase 'boost' state guess")
    with pytest.raises(error):
        guess.h = written


# --- output rows --------------------------------------------------------------------------


@pytest.fixture
def out():
    return Ascent._new(Rows, "dynamics", npoints=4)


def test_every_field_starts_missing(out):
    assert out.missing() == ["r", "v", "m"]


def test_a_block_field_takes_one_row_per_member(out):
    out.r = np.arange(12.0).reshape(3, 4)
    assert out.r.shape == (3, 4)
    assert out.missing() == ["v", "m"]


def test_a_scalar_broadcasts_over_a_block_field(out):
    out.v = 0.0
    assert out.missing() == ["r", "m"]
    assert list(out.v) == [0.0, 0.0, 0.0]


def test_rows_can_be_set_by_index_and_slice(out):
    out[6] = 1.0
    out[0:3] = np.zeros((3, 4))
    out[3:6] = 2.0
    assert out.missing() == []
    assert out[:].shape == (7, 4)


def test_reading_an_unassigned_field_is_an_error(out):
    with pytest.raises(AttributeError, match=r"'v' has not been assigned"):
        out.v


@pytest.mark.parametrize(
    ("statement", "error", "match"),
    [
        ("out.r = np.ones((2, 4))", ValueError, r"'r' needs 3 rows; got 2"),
        ("out.m = np.ones(3)", ValueError, "one value per time point, 4"),
        ("out.m = np.ones((2, 4))", ValueError, r"got an array of shape \(2, 4\)"),
        ("out[0:2] = np.ones(4)", ValueError, "needs 2 values; got 4"),
        ("out[99] = 1.0", IndexError, "out of range for 7 rows"),
        ("out[1.5] = 1.0", TypeError, "indices are integers or slices"),
    ],
)
def test_bad_row_writes_are_refused(out, statement, error, match):
    with pytest.raises(error, match=match):
        exec(statement, {"out": out, "np": np})


def test_an_array_across_a_slice_is_rows_not_one_row(out):
    """A per-point row must be repeated explicitly; an array is always a sequence of rows."""
    row = np.ones(4)
    with pytest.raises(ValueError, match="needs 2 values; got 4"):
        out[0:2] = row
    out[0:2] = (row,) * 2
    assert out[0].shape == (4,)


def test_the_last_write_of_a_field_wins(out):
    out.m = 1.0
    out[6] = 2.0
    assert out.m == 2.0


# --- read-only ----------------------------------------------------------------------------


def test_inputs_refuse_writes():
    state = Rocket._new(ReadOnlyRows, "phase 'boost' state", npoints=4)
    with pytest.raises(AttributeError, match="is read-only"):
        state.h = 1.0


def test_inputs_read_by_name_index_and_slice():
    state = Rocket._new(ReadOnlyRows, "phase 'boost' state", npoints=3)
    for row, value in enumerate([np.zeros(3), np.ones(3), 2 * np.ones(3)]):
        state._values[row] = value
    assert state.v.tolist() == [1.0, 1.0, 1.0]
    assert state[2].tolist() == [2.0, 2.0, 2.0]
    assert state[0:2].shape == (2, 3)
    assert [row.tolist() for row in state] == [[0.0] * 3, [1.0] * 3, [2.0] * 3]


# --- block fields of one row and of none --------------------------------------------------


class Generated(Vector):
    """The shape a declaration built by an algorithm takes: block fields of every size."""

    none = field(size=0)
    one = field(size=1)
    two = field(size=2)
    scalar = field()


def test_a_block_field_keeps_its_leading_axis_at_every_size():
    # This is the whole point of allowing size=1 and size=0: a declaration whose sizes are
    # computed has one rank, so nothing about the callback changes at k == 1 or k == 0.
    rows = np.arange(4 * 4, dtype=float).reshape(4, 4)
    state = Generated._new(ReadOnlyRows, "state", npoints=4)
    state._fill(rows)
    assert state.none.shape == (0, 4)
    assert state.one.shape == (1, 4)
    assert state.two.shape == (2, 4)
    assert state.scalar.shape == (4,)


def test_a_one_row_block_is_not_the_same_declaration_as_a_scalar_field():
    rows = np.arange(4 * 4, dtype=float).reshape(4, 4)
    state = Generated._new(ReadOnlyRows, "state", npoints=4)
    state._fill(rows)
    # Both occupy one flat row; only the scalar drops the axis. Were they the same, 'one'[0]
    # would silently mean the first point rather than the first row.
    assert state.one[0].tolist() == state.one.tolist()[0]
    assert state.scalar[0] == rows[3][0]


def test_sizes_lay_out_flat_rows_in_declaration_order():
    assert Generated._rows == (("one", 0), ("two", 0), ("two", 1), ("scalar", None))
    assert Generated._nrows == 0 + 1 + 2 + 1


def test_an_empty_block_reads_as_empty_and_takes_no_rows():
    values = Generated._new(Rows, "dynamics", npoints=4)
    values.none = []
    values.one = np.zeros(4)
    values.two = np.zeros((2, 4))
    values.scalar = np.zeros(4)
    assert values._is_complete()
    assert values.none.shape == (0, 4)


def test_an_empty_block_needs_no_assignment_to_be_complete():
    values = Generated._new(Rows, "dynamics", npoints=4)
    values.one = np.zeros(4)
    values.two = np.zeros((2, 4))
    values.scalar = np.zeros(4)
    assert values._is_complete()


# --- messages -----------------------------------------------------------------------------


def test_messages_name_the_aspect_they_came_from():
    bounds = Rocket._new(Bounds, "phase 'coast' state bounds")
    with pytest.raises(TypeError, match=re.escape("phase 'coast' state bounds 'h'")):
        bounds.h = "fast"


# --- containers -----------------------------------------------------------------------------


def test_every_declared_setting_actually_exists():
    """A container that names a setting it never installs would fail only when read.

    The names a container declares in `_held` and `_settable` and the ones it installs are
    written in different places, so they can drift apart; this walks every container a problem
    reaches and reads each one.
    """
    from yapss._next import Phases, Problem, phase
    from yapss._next.containers import Container

    class S(Vector):
        x = field()

    class C(Vector):
        u = field()

    class H(Vector):
        g = field()

    class Q(Vector):
        j = field()

    class D(Vector):
        d = field()

    class P(Vector):
        p = field()

    class OnePhase(Phases):
        only = phase(state=S, control=C, path=H, integral=Q)

    problem = Problem("t", phases=OnePhase, discrete=D, parameter=P)
    seen = []
    pending = [problem, *problem.phases]
    while pending:
        container = pending.pop()
        if not isinstance(container, Container):
            continue
        seen.append(container)
        for name in (*container._held, *container._settable):
            value = getattr(container, name)
            if isinstance(value, Container):
                pending.append(value)
    assert len(seen) >= 8
