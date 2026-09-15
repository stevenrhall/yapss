"""The callback output container: whole-row writes, write marks, and read-only reads.

The contract suite checks the rule through solves under every method; these tests check the
container directly -- which rows each write form marks, what reads return, and the NumPy
protocol paths (views, computed arrays, ``out=``, ``np.copyto``) that could otherwise write
without passing through `OutputArray.__setitem__`.
"""

from __future__ import annotations

import numpy as np
import pytest
from casadi import SX

import yapss.math as ym
from yapss._private.outputs import OutputArray, OutputRow
from yapss.math.wrapper import SXW

X = np.linspace(1.0, 2.0, 5)


def output(rows=3, points=5, dtype=np.float64):
    return OutputArray.zeros(
        (rows, points), dtype, label="arg.phase[0].dynamics", count="nx = 3 in phase 0"
    )


def marks(out):
    return out.written.astype(int).tolist()


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("out[:] = (X, 2 * X, 3.0)", [1, 1, 1]),
        ("out[:] = np.vstack([X, X, X])", [1, 1, 1]),
        ("out[0:2] = (X, X)", [1, 1, 0]),
        ("out[::2] = (X, 1.0)", [1, 0, 1]),
        ("out[1:] = 0.0", [0, 1, 1]),
        ("out[0] = X", [1, 0, 0]),
        ("out[-1] = X", [0, 0, 1]),
        ("out[1] = [v**2 for v in X]", [0, 1, 0]),
        ("out[1] = np.bool_(True)", [0, 1, 0]),
        ("out[2] += X", [0, 0, 1]),
        ("out += (X, X, X)", [1, 1, 1]),
        ("row = out[1]; row -= 1.0", [0, 1, 0]),
        ("out[0] = 0.0", [1, 0, 0]),
        ("out[1:2] = X", [0, 1, 0]),
    ],
)
def test_each_whole_row_form_marks_exactly_its_rows(statement, expected):
    out = output()
    exec(statement, {"out": out, "np": np, "X": X})  # noqa: S102
    assert marks(out) == expected


def test_reads_mark_nothing_and_reset_clears_values_and_marks():
    out = output()
    out[:] = (X, X, X)
    out.reset()
    _ = out[0], out[:, 1], np.asarray(out), out.sum(), out.T
    assert marks(out) == [0, 0, 0]
    assert not np.asarray(out).any()


def test_in_place_operators_compute_the_right_values():
    out = output()
    out[0] = X
    out[0] += X
    out[1] = 2.0
    out[1] **= 3
    row = out[2]
    row -= 1.0
    np.testing.assert_array_equal(
        np.asarray(out), np.vstack([2 * X, np.full(5, 8.0), np.full(5, -1.0)])
    )


def test_the_array_is_read_only_and_views_refuse_writes_with_the_rule():
    out = output()
    with pytest.raises(ValueError, match="read-only"):
        np.asarray(out)[0, 0] = 1.0
    column = out[:, 1]
    with pytest.raises(TypeError, match=r"part of arg\.phase\[0\]\.dynamics"):
        column[0] = 1.0
    with pytest.raises(TypeError, match=r"arg\.phase\[0\]\.dynamics\[0\]"):
        out[0][:2] = 1.0


def test_arrays_computed_from_an_output_are_ordinary_data():
    out = output()
    out[:] = (X, X, X)
    doubled = 2 * out
    copied = out.copy()
    concatenated = np.concatenate((out[0], out[1]))
    for array in (doubled, copied, concatenated, out[0] * 2):
        array[0] = -1.0
        assert np.asarray(array).flat[0] == -1.0
    listed = np.concatenate([out[0], out[1]])
    assert listed.size == 10
    picked = out[0][[0, 2]]  # a copy of part of a row: ordinary data
    picked[0] = -1.0
    picked += 1.0
    np.testing.assert_array_equal(picked, [0.0, 2.5])
    plain = np.zeros(5)
    np.copyto(plain, out[0])
    np.add(out[0], 1.0, out=plain)
    np.testing.assert_array_equal(plain, X + 1.0)
    quotient, remainder = np.zeros(5), np.zeros(5)
    np.divmod(out[0], 1.0, out=(quotient, remainder))
    np.testing.assert_array_equal(quotient + remainder, X)
    assert type(doubled) is np.ndarray
    assert type(out[0] * 2) is np.ndarray
    assert isinstance(out[0], OutputRow)
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize(
    "statement",
    [
        "np.copyto(out[0], X)",
        "np.copyto(out, 0.0)",
        "np.multiply(X, 2.0, out=out[0])",
        "np.add(out, 1.0, out=out)",
    ],
)
def test_out_and_copyto_cannot_write_into_an_output(statement):
    out = output()
    with pytest.raises(TypeError, match=r"arg\.phase\[0\]\.dynamics"):
        exec(statement, {"out": out, "np": np, "X": X})  # noqa: S102
    assert marks(out) == [0, 0, 0]


@pytest.mark.parametrize(
    ("statement", "exc", "fragment"),
    [
        ("out[0, 1] = 1.0", TypeError, r"dynamics\[0, 1\]"),
        ("out[[0, 1]] = (X, X)", TypeError, "whole rows"),
        ("out[...] = 0.0", TypeError, r"\[\.\.\.\]"),
        ("out[True] = X", TypeError, "whole rows"),
        ("out[:] = (X, X)", ValueError, "expected 3 rows, got 2"),
        ("out[:] = X", ValueError, "one expression over the points cannot fill 3 rows"),
        ("out[0] = np.array([1.0])", ValueError, "length-1 array"),
        ("out[0] = [1.0]", ValueError, "list with one value"),
        ("out[0] = X[:3]", ValueError, "row of 5 points"),
        ("out[0] = [1.0, 2.0, 3.0]", ValueError, "expected 5 values, one per point, got 3"),
        ("out[:] = np.ones((2, 5))", ValueError, r"expected shape \(3, 5\), got \(2, 5\)"),
        ("column = out[:, 0]; column += 1.0", TypeError, r"part of arg\.phase\[0\]\.dynamics"),
        ("out[3] = X", IndexError, r"range\(3\).*nx = 3 in phase 0"),
    ],
)
def test_refused_writes_name_the_output_and_change_nothing(statement, exc, fragment):
    out = output()
    with pytest.raises(exc, match=fragment):
        exec(statement, {"out": out, "np": np, "X": X})  # noqa: S102
    assert marks(out) == [0, 0, 0]
    assert not np.asarray(out).any()


def test_an_output_with_no_rows_accepts_only_nothing():
    empty = OutputArray.zeros(
        (0, 5), np.float64, label="arg.phase[0].path", count="nh = 0 in phase 0"
    )
    empty[:] = ()
    empty[:] = 0.0
    with pytest.raises(ValueError, match="nh = 0"):
        empty[:] = X
    with pytest.raises(IndexError, match="no rows"):
        empty[0] = X


def test_discrete_outputs_hold_one_value_per_row():
    discrete = OutputArray.zeros((3,), np.float64, label="arg.discrete", count="nd = 3")
    discrete[0] = 1.5
    discrete[1:3] = np.array([2.0, 3.0])
    discrete[2] += 1.0
    assert discrete[0] == 1.5
    assert isinstance(discrete[0], np.floating)
    np.testing.assert_array_equal(np.asarray(discrete), [1.5, 2.0, 4.0])
    assert marks(discrete) == [1, 1, 1]
    discrete.reset()
    discrete[0:1] = np.array([7.0])  # a slice of rows takes one value per row
    discrete[1] = np.float64(8.0)  # a NumPy scalar is one value
    np.testing.assert_array_equal(np.asarray(discrete), [7.0, 8.0, 0.0])
    for array in (np.array([7.0]), np.array([1.0, 2.0])):
        with pytest.raises(
            ValueError,
            match=r"expected one value, got an array of shape .*arg\.discrete\[2\] = value\[0\]",
        ):
            discrete[2] = array
    with pytest.raises(ValueError, match="got a list"):
        discrete[2] = [7.0]
    with pytest.raises(ValueError, match=r"arg\.discrete\[2:3\] = value"):
        discrete[-1] = np.array([7.0])
    assert marks(discrete) == [1, 1, 0]


def test_symbolic_outputs_follow_the_same_rule_at_one_point():
    out = output(points=1, dtype=np.object_)
    t = SXW(SX.sym("t"))
    s = ym.sin(t)
    out[:] = (s, 2 * s, 3.0)
    out[0] += s
    out[1] = np.array([s], dtype=object)
    assert marks(out) == [1, 1, 1]
    assert isinstance(np.asarray(out)[0, 0], SXW)
    with pytest.raises(TypeError, match="whole row"):
        out[0][0] = s
    with pytest.raises(ValueError, match="cannot fill 3 rows"):
        out[:] = np.array([s], dtype=object)
