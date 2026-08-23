"""

Tests for symbolic comparison operators on SXArray.

A user callback written for YAPSS treats each state as a scalar, so gated expressions
like ``(abs(y) <= 1) * x`` -- enforce a constraint only where ``y`` is in range -- are
written with plain operators, not with ``yapss.math`` function calls. Those operators go
straight to :class:`numpy.ndarray`, so ``yapss.math`` cannot intercept them.

Before :class:`SXArray`, numpy's object-dtype loop coerced each elementwise result to
``bool``; :class:`SXW` has no ``__bool__``, so every comparison came back ``True`` and
the mask vanished. The gate then evaluated correctly under the finite-difference
derivative methods and silently not at all under ``"auto"`` -- the constraint was
enforced everywhere instead of over a finite interval.

These tests compare the symbolic result against numpy on floats, which is the reference
the finite-difference methods produce.

"""

import casadi as ca
import numpy as np
import pytest

from yapss.math.wrapper import SXW, SXArray, sx_array

# sample points, chosen to sit either side of the gate edges and on them
Y_VALUES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
X_VALUE = 3.0


@pytest.fixture(name="symbols")
def symbols_fixture():
    """Return symbolic (y, x) and the matching SXArray pair."""
    sy, sx = ca.SX.sym("y"), ca.SX.sym("x")
    return sy, sx, sx_array([SXW(sy)]), sx_array([SXW(sx)])


def evaluate(expression, sy, sx, y_values, x_value=X_VALUE):
    """Evaluate a length-1 symbolic expression at each of `y_values`."""
    scalar = SXW(np.atleast_1d(expression)[0])._value
    function = ca.Function("f", [sy, sx], [scalar])
    return np.array([float(function(y, x_value)) for y in y_values])


OPERATORS = [
    ("lt", lambda a, b: a < b),
    ("le", lambda a, b: a <= b),
    ("gt", lambda a, b: a > b),
    ("ge", lambda a, b: a >= b),
    ("eq", lambda a, b: a == b),
    ("ne", lambda a, b: a != b),
]


@pytest.mark.parametrize(("name", "operator"), OPERATORS, ids=[n for n, _ in OPERATORS])
def test_comparison_matches_numpy(name, operator, symbols):
    """Each comparison operator agrees with the same operator on a float array."""
    sy, sx, y, _ = symbols
    actual = evaluate(operator(y, 0.5), sy, sx, Y_VALUES)
    expected = operator(np.array(Y_VALUES), 0.5).astype(float)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(("name", "operator"), OPERATORS, ids=[n for n, _ in OPERATORS])
def test_comparison_reflected(name, operator, symbols):
    """A scalar on the left dispatches to the reflected operator, not to numpy's."""
    sy, sx, y, _ = symbols
    actual = evaluate(operator(0.5, y), sy, sx, Y_VALUES)
    expected = operator(0.5, np.array(Y_VALUES)).astype(float)
    assert np.array_equal(actual, expected)


def test_gated_constraint_matches_float_path(symbols):
    """``(abs(y) <= 1) * x`` keeps its mask.

    This is the regression test for the defect itself: the gate must select a finite
    interval, not collapse to enforcing the constraint everywhere.
    """
    sy, sx, y, x = symbols
    actual = evaluate((np.abs(y) <= 1) * x, sy, sx, Y_VALUES)
    expected = (np.abs(np.array(Y_VALUES)) <= 1) * X_VALUE
    assert np.array_equal(actual, expected)
    # and specifically: it is not the unmasked constraint
    assert not np.array_equal(actual, np.full(len(Y_VALUES), X_VALUE))


def test_mask_combination(symbols):
    """`&`, `|`, and `~` combine masks, since `and`, `or`, `not` cannot be overloaded."""
    sy, sx, y, x = symbols
    values = np.array(Y_VALUES)

    both = ((y >= -1) & (y <= 1)) * x
    assert np.array_equal(
        evaluate(both, sy, sx, Y_VALUES),
        ((values >= -1) & (values <= 1)) * X_VALUE,
    )

    either = ((y < -1) | (y > 1)) * x
    assert np.array_equal(
        evaluate(either, sy, sx, Y_VALUES),
        ((values < -1) | (values > 1)) * X_VALUE,
    )

    negated = (~(y > 1)) * x
    assert np.array_equal(evaluate(negated, sy, sx, Y_VALUES), (~(values > 1)) * X_VALUE)


def test_comparison_against_array_operand(symbols):
    """Comparing two symbolic arrays elementwise, rather than against a scalar."""
    sy, sx, y, x = symbols
    actual = evaluate((y <= x) * x, sy, sx, Y_VALUES)
    expected = (np.array(Y_VALUES) <= X_VALUE) * X_VALUE
    assert np.array_equal(actual, expected)


def test_subclass_survives_arithmetic(symbols):
    """Ufuncs and arithmetic must preserve SXArray, or the operators stop applying."""
    _, _, y, x = symbols
    for result in (np.abs(y), y + 1.0, 2.0 * y, y * x, np.sin(y)):
        assert isinstance(result, SXArray), f"lost SXArray through {result!r}"
    # and the comparison still works on the result
    assert isinstance(np.abs(y) <= 1, SXArray)


def test_float_path_is_unchanged():
    """Plain float arrays must keep numpy's ordinary bool semantics."""
    values = np.array(Y_VALUES)
    mask = np.abs(values) <= 1
    assert mask.dtype == np.bool_
    assert np.array_equal(mask * X_VALUE, np.where(np.abs(values) <= 1, X_VALUE, 0.0))


def test_sxarray_is_unhashable():
    """__eq__ is overridden, so SXArray must not be hashable -- as for ndarray and SXW."""
    y = sx_array([SXW(ca.SX.sym("y"))])
    with pytest.raises(TypeError):
        hash(y)
