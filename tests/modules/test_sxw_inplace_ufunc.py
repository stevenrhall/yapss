"""

Tests for ``out=`` handling in :meth:`SXW.__array_ufunc__`.

:class:`SXW` inherits its operators from :class:`numpy.lib.mixins.NDArrayOperatorsMixin`,
which spells every augmented assignment as ``ufunc(self, other, out=(self,))``. A user
callback that accumulates a term at a time -- ``d = 0.0`` then ``d += ...`` in a loop --
therefore reaches ``__array_ufunc__`` with an ``SXW`` in ``out``, even though nothing in
the callback mentions ``out``.

That buffer cannot be written: an :class:`SXW` wraps an immutable CasADi value. What
made it a defect is that numpy's dispatch considers ``out`` operands as well as inputs,
so forwarding the untouched ``SXW`` to the ufunc re-entered ``__array_ufunc__`` with
identical arguments. The chain terminated only while SX *handled* the inner call and
ignored ``out``; whenever SX returned NotImplemented instead, numpy fell through to this
override again and the re-dispatch recursed without bound.

SX returns NotImplemented in two circumstances, so the recursion was reachable on both
sides of the CasADi 3.8 boundary:

- CasADi >= 3.8 under ``GlobalOptions.setNumpyMode(1)``, where SX refuses any output
  buffer it does not recognize -- so every ufunc was affected.
- CasADi <= 3.7.2 for a unary ufunc it has no implementation for. ``np.negative``,
  ``np.absolute``, ``np.square`` and ``np.reciprocal`` all recursed there. Those same
  calls raise ``TypeError`` without ``out=``, so no correct program was made wrong --
  but a wrong one failed with ``RecursionError`` instead of a usable diagnostic.

The mode matters, so these tests run under every numpy mode the installed CasADi
offers. On CasADi < 3.8 there is only the legacy behavior and ``setNumpyMode`` does not
exist; the mode-1 cases are skipped there rather than silently passing.

"""

import casadi as ca
import numpy as np
import pytest

from yapss.math.wrapper import SXW

# CasADi < 3.8 has a single (legacy) numpy mode and no way to select one. From 3.8 on,
# 0 is the default (legacy plus a FutureWarning), -1 is legacy silenced, and 1 is the
# casadi-aware support that exposes the recursion this module pins.
HAS_NUMPY_MODE = hasattr(ca.GlobalOptions, "setNumpyMode")
NUMPY_MODES = (0, -1, 1) if HAS_NUMPY_MODE else (None,)


@pytest.fixture(params=NUMPY_MODES)
def numpy_mode(request):
    """Run the test under one CasADi numpy mode, restoring the previous one after."""
    mode = request.param
    if mode is None:
        yield None
        return
    previous = ca.GlobalOptions.getNumpyMode()
    ca.GlobalOptions.setNumpyMode(mode)
    try:
        yield mode
    finally:
        ca.GlobalOptions.setNumpyMode(previous)


def test_augmented_assignment_accumulates(numpy_mode):
    """``d += ...`` builds the expression rather than recursing.

    This is the loop shape a user callback writes when a discrete constraint is a sum
    of terms. Before the fix it raised RecursionError under numpy mode 1.
    """
    a = SXW(ca.SX.sym("a"))
    total = SXW(ca.SX(0.0))
    for _ in range(3):
        total += (a + 3) * (a - 2)

    expected = SXW(ca.SX(0.0))
    for _ in range(3):
        expected = expected + (a + 3) * (a - 2)

    assert str(total) == str(expected)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda x, y: x.__iadd__(y), "(a+b)"),
        (lambda x, y: x.__isub__(y), "(a-b)"),
        (lambda x, y: x.__imul__(y), "(a*b)"),
        (lambda x, y: x.__itruediv__(y), "(a/b)"),
    ],
    ids=["iadd", "isub", "imul", "itruediv"],
)
def test_inplace_operators_return_the_expression(numpy_mode, operation, expected):
    """Every in-place operator routes through ``out=`` and must survive it."""
    x, y = SXW(ca.SX.sym("a")), SXW(ca.SX.sym("b"))
    assert str(operation(x, y)) == f"SXW(SX({expected}))"


def test_explicit_out_does_not_write_through(numpy_mode):
    """An explicit ``out=`` is accepted and ignored, as CasADi has always done.

    Pinned deliberately rather than left to chance: the value comes back through the
    return, and the buffer is untouched. Raising instead is not an option while
    NDArrayOperatorsMixin routes augmented assignment through ``out``, since the two
    are indistinguishable here. Making this case raise is a 0.3.0-style behavior
    change, and this test is the one to flip if that is ever decided.
    """
    a, b = SXW(ca.SX.sym("a")), SXW(ca.SX.sym("b"))
    sink = SXW(ca.SX(0.0))

    result = np.add(a, b, out=sink)

    assert str(result) == "SXW(SX((a+b)))"
    assert str(sink) == "SXW(SX(0))"


def test_symbolic_operands_never_reach_the_ufunc(numpy_mode):
    """A symbolic operation is computed from the casadi table, not by calling the ufunc.

    Forwarding to the ufunc is what recursed: numpy's dispatch considered the SXW in
    ``out`` and re-entered ``__array_ufunc__``. The rewrite never calls the ufunc on a
    symbol at all, so the ufunc must see no call. Asserting on that pins the mechanism,
    not just the symptom, so a refactor that reintroduces the forwarding fails here even
    on a CasADi version where it happens not to recurse.
    """
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return np.add(*args, **kwargs)

    spy.__name__ = "add"
    a, b = SXW(ca.SX.sym("a")), SXW(ca.SX.sym("b"))
    result = SXW.__array_ufunc__(a, spy, "__call__", a, b, out=(a,))

    assert calls == []
    assert str(result) == "SXW(SX((a+b)))"


def test_numeric_operands_forward_to_the_ufunc_with_out(numpy_mode):
    """When nothing symbolic remains, the call is numpy's, ``out`` buffer included.

    The fix keys on an SXW being present in ``out``; an ordinary float buffer must pass
    through untouched and be written normally.
    """
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return np.add(*args, **kwargs)

    spy.__name__ = "add"
    buffer = np.zeros(1)
    SXW.__array_ufunc__(SXW(ca.SX(1.0)), spy, "__call__", 2.0, 3.0, out=(buffer,))

    assert "out" in seen
    assert buffer[0] == 5.0


def test_inplace_on_an_array_writes_the_buffer(numpy_mode):
    """``array += w`` on an object array writes each element in place.

    numpy spells it ``np.add(array, w, out=(array,))``. Before the rewrite the whole
    result vector was written into every element (a 3-vector inside each SXW); now each
    element gets its own scalar and the buffer is the same object it was.
    """
    from yapss.math.wrapper import sx_array

    w = SXW(ca.SX.sym("w"))
    array = sx_array([SXW(ca.SX.sym("x0")), SXW(ca.SX.sym("x1"))])
    before = id(array)
    array += w

    assert id(array) == before
    assert [str(item) for item in array] == ["SXW(SX((x0+w)))", "SXW(SX((x1+w)))"]


@pytest.mark.parametrize(
    "ufunc_name",
    ["negative", "absolute", "square", "reciprocal", "sqrt", "exp", "sin", "floor"],
)
def test_unary_out_changes_nothing(numpy_mode, ufunc_name):
    """A unary ufunc behaves identically with and without ``out=``.

    The invariant that actually matters, and the one that holds across CasADi versions
    and numpy modes: passing ``out=`` may not change the outcome, whether that outcome
    is a value or an exception. Stated this way the test needs no per-version
    expectations -- on casadi 3.7.2 the four ufuncs CasADi cannot evaluate raise
    ``TypeError`` both ways, and on 3.8 they succeed both ways.

    Before the fix this failed on the *pinned* casadi 3.7.2, not merely under mode 1:
    ``np.negative(x, out=x)`` raised RecursionError while ``np.negative(x)`` raised
    TypeError.
    """
    ufunc = getattr(np, ufunc_name)

    def outcome(use_out):
        value = SXW(ca.SX.sym("x"))
        try:
            kwargs = {"out": SXW(ca.SX.sym("x"))} if use_out else {}
            return "value", str(ufunc(value, **kwargs))
        except Exception as exc:  # noqa: BLE001
            return "error", type(exc).__name__

    assert outcome(use_out=True) == outcome(use_out=False)


def test_out_holding_no_sxw_is_left_alone(numpy_mode):
    """A genuine numeric buffer still reaches the ufunc.

    The fix keys on an SXW being present in ``out``, so an ordinary float buffer must
    pass through untouched and be written normally.
    """
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return np.add(*args, **kwargs)

    buffer = np.zeros(1)
    SXW.__array_ufunc__(SXW(ca.SX(1.0)), spy, "__call__", 2.0, 3.0, out=(buffer,))

    assert "out" in seen
    assert buffer[0] == 5.0
