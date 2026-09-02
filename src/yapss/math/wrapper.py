"""

Wrap casadi SX scalars so that they can live inside numpy arrays.

The ``"auto"`` derivative method traces the user's callbacks with casadi symbols. A casadi
``SX`` cannot be a numpy array element directly: it is always a 2-D matrix, it is not
iterable "by design", and it defines ``__array__``, so numpy splices its interior shape into
any array built around it -- ``dtype=object`` does not prevent that. :class:`SXW` wraps one
scalar symbol as an opaque Python object, and :class:`SXArray` is the object-dtype ndarray
that holds them.

The rule that keeps this airtight is that **casadi never sees a numpy array, and numpy never
sees a bare SX**. Every operation on a symbol resolves to one entry of :data:`UFUNCS`, a table
of casadi implementations keyed by numpy ufunc name, applied to one scalar at a time. Three
paths lead there:

* the operator path -- ``w * x``, ``np.sin(w)`` -- through :meth:`SXW.__array_ufunc__`;
* numpy's object-dtype loop, which calls a method of the ufunc's name on each array element
  (``elem.sqrt()``), through :meth:`SXW.__getattr__`;
* the module-level functions in :mod:`yapss.math.functions`, through :func:`apply_ufunc` and
  :func:`reduce_symbolic`.

Anything not in the table raises rather than falling through to casadi's own numpy
interoperability, whose behavior differs between casadi versions and numpy modes.

"""

# future imports
from __future__ import annotations

# standard imports
import operator
from typing import TYPE_CHECKING, Any, ClassVar

# third party imports
import casadi as ca
import numpy as np
from casadi import SX
from numpy.lib.mixins import NDArrayOperatorsMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


class UnsupportedMathFunctionError(TypeError):
    """Raised for a numpy function that YAPSS callbacks cannot support.

    Subclasses :class:`TypeError`, which is what numpy itself raises today when one of these
    is handed a symbolic argument.
    """


class UnsupportedMathFunctionWarning(FutureWarning):
    """An unsupported function was called on real arguments; it will raise from 0.3.0.

    These functions have no symbolic equivalent, so they already fail under the ``"auto"``
    derivative method. On real arguments they still evaluate, which lets a formulation depend
    on the derivative method chosen; from 0.3.0 they raise `UnsupportedMathFunctionError` on
    both paths.
    """


# Functions with no meaning on a symbolic value, with the reason. A symbolic argument raises,
# as it already did before 0.2.2 -- numpy itself raised TypeError -- and a real argument
# warns and evaluates. Rejecting both paths is the goal: a function that works under one
# derivative method and fails under another lets a formulation depend on the derivative
# method, which is the class of defect this module exists to prevent. But the real path
# worked through 0.2.1, so it may not be taken away in a patch release; the warning says so
# and 0.3.0 raises. Use numpy directly if one of these is needed outside a callback.
REJECTED: dict[str, str] = {
    "nextafter": "steps between adjacent floating-point values",
    "signbit": "reads the floating-point sign bit, including the sign of negative zero",
    "spacing": "returns the distance to the adjacent floating-point value",
}


def rejected_message(name: str) -> str:
    """Return the message explaining why `name` is refused in a callback."""
    return (
        f"'{name}' is not supported in YAPSS callback functions because it {REJECTED[name]}, "
        f"which has no symbolic equivalent. Callback functions must give the same result "
        f"under every derivative method. Use 'numpy.{name}' directly if you need it outside "
        f"a callback."
    )


# ====================================================================================
# casadi implementations
# ====================================================================================
#
# Everything a symbol can be asked to do, spelled once. Formulas rather than casadi
# functions where casadi has none (``square``, ``cbrt``, ``trunc``) or where its convention
# differs from numpy's (``mod``). The conformance test checks each against numpy.


def _absolute(x: SX) -> SX:
    return ca.sign(x) * x


def _cbrt(x: SX) -> SX:
    return ca.sign(x) * ca.fabs(x) ** (1 / 3)


def _trunc(x: SX) -> SX:
    return ca.sign(x) * ca.floor(ca.fabs(x))


def _rint(x: SX) -> SX:
    """Round half to even, exactly as numpy does.

    Built from ``floor`` and the fractional part rather than from ``floor(x + 0.5)``: the
    addition rounds when ``x`` is within half an ulp below a half-integer, and every
    ``floor(x + 0.5)`` scheme is wrong there. ``x - floor(x)`` is exact in double
    precision, so the tie test is exact. Doubles of magnitude 2**52 and above are already
    integers (and so is infinity, for this purpose); they pass through unchanged.
    """
    r = ca.floor(x)
    d = x - r
    odd = ca.eq(r - 2 * ca.floor(r / 2), 1)
    rounded = r + ca.if_else(ca.gt(d, 0.5), 1, ca.if_else(ca.lt(d, 0.5), 0, odd))
    return ca.if_else(ca.ge(ca.fabs(x), 2.0**52), x, rounded)


def _round(x: SX, decimals: int = 0) -> SX:
    """Round to `decimals` places the way numpy does: scale, rint, unscale."""
    if decimals == 0:
        return _rint(x)
    if decimals > 0:
        factor = 10.0**decimals
        return _rint(x * factor) / factor
    factor = 10.0 ** (-decimals)
    return _rint(x / factor) * factor


def _mod(x: SX, y: SX) -> SX:
    # floors and takes the sign of the divisor, like numpy's mod and remainder; ca.fmod
    # truncates and takes the sign of the dividend, like numpy's fmod
    return x - y * ca.floor(x / y)


def _floor_divide(x: SX, y: SX) -> SX:
    return ca.floor(x / y)


def _logaddexp(x: SX, y: SX) -> SX:
    """Return log(exp(x) + exp(y)), computed so that large arguments do not overflow."""
    return ca.fmax(x, y) + ca.log1p(ca.exp(-ca.fabs(x - y)))


def _logaddexp2(x: SX, y: SX) -> SX:
    """Return log2(2**x + 2**y), computed so that large arguments do not overflow."""
    return ca.fmax(x, y) + ca.log1p(2 ** -ca.fabs(x - y)) / np.log(2)


def _copysign(x: SX, y: SX) -> SX:
    """Return the magnitude of x with the sign of y.

    Differs from numpy at ``y == -0.0``: numpy reads the sign bit, which a symbolic expression
    cannot represent, so negative zero is treated as positive.
    """
    return ca.if_else(ca.ge(y, 0), ca.fabs(x), -ca.fabs(x))


def _heaviside(x: SX, y: SX) -> SX:
    """Return 0 where x < 0, y where x == 0, and 1 where x > 0."""
    return ca.gt(x, 0) + ca.eq(x, 0) * y


def _logical_xor(x: SX, y: SX) -> SX:
    return ca.logic_and(ca.logic_or(x, y), ca.logic_not(ca.logic_and(x, y)))


def _clip(x: SX, lo: SX, hi: SX) -> SX:
    # numpy's clip is min(max(x, lo), hi), so lo > hi yields hi; same here
    return ca.fmin(ca.fmax(x, lo), hi)


def _where(condition: SX, x: SX, y: SX) -> SX:
    return ca.if_else(condition, x, y)


UNARY: dict[str, Callable[[SX], SX]] = {
    "negative": operator.neg,
    "positive": lambda x: x,
    "absolute": _absolute,
    "fabs": _absolute,
    "sign": ca.sign,
    "sqrt": ca.sqrt,
    "square": lambda x: x * x,
    "cbrt": _cbrt,
    "reciprocal": lambda x: 1 / x,
    "exp": ca.exp,
    "exp2": lambda x: 2**x,
    "expm1": ca.expm1,
    "log": ca.log,
    "log2": lambda x: ca.log(x) / np.log(2),
    "log10": ca.log10,
    "log1p": ca.log1p,
    "sin": ca.sin,
    "cos": ca.cos,
    "tan": ca.tan,
    "arcsin": ca.asin,
    "arccos": ca.acos,
    "arctan": ca.atan,
    "sinh": ca.sinh,
    "cosh": ca.cosh,
    "tanh": ca.tanh,
    "arcsinh": ca.asinh,
    "arccosh": ca.acosh,
    "arctanh": ca.atanh,
    "deg2rad": lambda x: x * (np.pi / 180),
    "radians": lambda x: x * (np.pi / 180),
    "rad2deg": lambda x: x * (180 / np.pi),
    "degrees": lambda x: x * (180 / np.pi),
    "floor": ca.floor,
    "ceil": ca.ceil,
    "trunc": _trunc,
    "rint": _rint,
    "conjugate": lambda x: x,
    "conj": lambda x: x,
    # ``~`` on a mask; numpy spells it invert (bitwise) and logical_not
    "logical_not": ca.logic_not,
    "invert": ca.logic_not,
}

BINARY: dict[str, Callable[[SX, SX], SX]] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "true_divide": operator.truediv,
    "power": ca.power,
    "float_power": ca.power,
    "arctan2": ca.atan2,
    "hypot": ca.hypot,
    # fmax and fmin are casadi's names for maximum and minimum; numpy's NaN-ignoring fmax
    # is deliberately not reproduced -- see yapss.math.functions
    "maximum": ca.fmax,
    "fmax": ca.fmax,
    "minimum": ca.fmin,
    "fmin": ca.fmin,
    "fmod": ca.fmod,
    "mod": _mod,
    "remainder": _mod,
    "floor_divide": _floor_divide,
    "equal": ca.eq,
    "not_equal": ca.ne,
    "less": ca.lt,
    "less_equal": ca.le,
    "greater": ca.gt,
    "greater_equal": ca.ge,
    "logical_and": ca.logic_and,
    "logical_or": ca.logic_or,
    "logical_xor": _logical_xor,
    # ``&``, ``|``, ``^`` on masks
    "bitwise_and": ca.logic_and,
    "bitwise_or": ca.logic_or,
    "bitwise_xor": _logical_xor,
    "copysign": _copysign,
    "heaviside": _heaviside,
    "logaddexp": _logaddexp,
    "logaddexp2": _logaddexp2,
}

TERNARY: dict[str, Callable[[SX, SX, SX], SX]] = {
    "clip": _clip,
    "where": _where,
}

UFUNCS: dict[str, Callable[..., SX]] = {**UNARY, **BINARY, **TERNARY}

# Reductions over every element. numpy's own reduce loops for these compare and then call
# bool() on the result, which a symbol cannot answer; a fold over the casadi binary
# operation is exact. add and multiply are absent on purpose: numpy reduces those through
# __add__ and __mul__, which already dispatch correctly, and they must keep working on the
# plain object arrays numpy hands them.
REDUCTIONS: dict[str, Callable[[SX, SX], SX]] = {
    "max": ca.fmax,
    "amax": ca.fmax,
    "min": ca.fmin,
    "amin": ca.fmin,
    "all": ca.logic_and,
    "any": ca.logic_or,
}


# ====================================================================================
# dispatch
# ====================================================================================


def is_symbolic(value: Any) -> bool:
    """Return whether `value` is an SXW or an array-like holding one.

    A bare casadi value is refused outright: a callback receives its arguments already
    wrapped, so a bare SX can only mean a user built one, and every operation below would
    otherwise hand it to numpy, which is exactly what this module exists to prevent.
    """
    if isinstance(value, SXW):
        return True
    if isinstance(value, (ca.SX, ca.MX, ca.DM)):
        msg = (
            f"A bare casadi {type(value).__name__} reached yapss.math. Callback arguments are "
            f"already wrapped; build expressions from them rather than from casadi symbols."
        )
        raise TypeError(msg)
    if isinstance(value, np.ndarray):
        return value.dtype == object and any(isinstance(item, SXW) for item in value.flat)
    if isinstance(value, (list, tuple)):
        return any(is_symbolic(item) for item in value)
    return False


def _as_object_array(value: Any) -> NDArray[np.object_]:
    """Return `value` as an object-dtype array, an SXW becoming a 0-d array holding it.

    Built by assignment rather than ``np.asarray(value, dtype=object)`` so that numpy never
    inspects the wrapper: it would probe ``__array__`` and the sequence protocol, and the
    only thing that must never happen here is numpy discovering a shape inside a symbol.
    """
    if isinstance(value, SXW):
        array = np.empty((), dtype=object)
        array[()] = value
        return array
    if isinstance(value, np.ndarray):
        return value.astype(object, copy=False)
    return np.asarray(value, dtype=object)


def _raw(value: Any) -> Any:
    """Return the casadi SX inside an SXW, and a Python scalar for a numpy one."""
    if isinstance(value, SXW):
        return value._value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _apply_scalar(name: str, elements: Sequence[Any]) -> Any:
    """Apply the ufunc `name` to one set of scalar operands."""
    if not any(isinstance(item, SXW) for item in elements):
        # a purely numeric element of a mixed array: numpy's own arithmetic
        return getattr(np, name)(*elements)
    if name in REJECTED:
        raise UnsupportedMathFunctionError(rejected_message(name))
    implementation = UFUNCS.get(name)
    if implementation is None:
        msg = (
            f"numpy.{name} has no symbolic implementation in yapss.math, so it cannot be "
            f"applied to a callback argument under the 'auto' derivative method."
        )
        raise UnsupportedMathFunctionError(msg)
    return SXW(implementation(*[_raw(item) for item in elements]))


def _elementwise(name: str, operands: Sequence[Any]) -> Any:
    """Apply the ufunc `name` elementwise over broadcast operands.

    Scalar operands give an SXW; anything with a shape gives an SXArray. Every element is
    computed by :func:`_apply_scalar`, so casadi only ever sees scalars.
    """
    arrays = [_as_object_array(item) for item in operands]
    if all(array.ndim == 0 for array in arrays):
        return _apply_scalar(name, [array[()] for array in arrays])
    broadcast = np.broadcast_arrays(*arrays)
    result = np.empty(broadcast[0].shape, dtype=object)
    for index in np.ndindex(result.shape):
        result[index] = _apply_scalar(name, [array[index] for array in broadcast])
    return result.view(SXArray)


def map_unary(symbolic: Callable[[SX], SX], numeric: Callable[[Any], Any], value: Any) -> Any:
    """Apply `symbolic` to every SXW in `value` and `numeric` to every other element."""
    array = _as_object_array(value)

    def one(item: Any) -> Any:
        return SXW(symbolic(item._value)) if isinstance(item, SXW) else numeric(item)

    if array.ndim == 0:
        return one(array[()])
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(result.shape):
        result[index] = one(array[index])
    return result.view(SXArray)


def apply_ufunc(name: str, *args: Any, **kwargs: Any) -> Any:
    """Apply ``numpy.<name>`` to real arguments, or its casadi implementation to symbolic ones.

    This is what the module-level functions of :mod:`yapss.math` call. On real input it is
    numpy, keyword arguments and all; on symbolic input it is the table, elementwise.
    """
    if not any(is_symbolic(item) for item in args):
        return getattr(np, name)(*args, **kwargs)
    if kwargs:
        msg = (
            f"yapss.math.{name}: keyword arguments ({', '.join(kwargs)}) are not supported "
            f"on a symbolic value."
        )
        raise TypeError(msg)
    return _elementwise(name, args)


def reduce_symbolic(name: str, value: Any, axis: Any = None, **kwargs: Any) -> Any:
    """Reduce `value` with ``numpy.<name>`` if real, or with a casadi fold if symbolic."""
    if not is_symbolic(value):
        return getattr(np, name)(value, axis=axis, **kwargs)
    if axis is not None or kwargs:
        msg = f"yapss.math.{name}: only a full reduction is supported on a symbolic value."
        raise TypeError(msg)
    items = [_raw(item) for item in _as_object_array(value).flat]
    if not items:
        msg = f"zero-size array to reduction operation {name} which has no identity"
        raise ValueError(msg)
    fold = REDUCTIONS[name]
    accumulated = items[0]
    for item in items[1:]:
        accumulated = fold(accumulated, item)
    return SXW(accumulated)


# ====================================================================================
# the wrapper
# ====================================================================================


class SXW(NDArrayOperatorsMixin):
    """One casadi SX scalar, opaque to numpy.

    Arithmetic, comparisons, and mask operators come from
    :class:`numpy.lib.mixins.NDArrayOperatorsMixin`, which routes every one of them through
    :meth:`__array_ufunc__` and so through :data:`UFUNCS`.
    """

    # __eq__ returns a symbolic SXW, not a bool -- the same numpy.ndarray convention that
    # makes arrays unhashable. Set explicitly, like numpy does.
    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(self, value: float | SX | SXW):
        # a one-element array is its element: np.sum(sxarray) is a 0-d SXArray, and a
        # one-integral phase's ``integral`` is a 1-element SXArray
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        while isinstance(value, SXW):
            value = value._value
        if isinstance(value, np.generic):
            value = value.item()
        sx = value if isinstance(value, SX) else SX(value)
        # A wrapper holding a matrix is not a scalar with extra data; it is the symptom of
        # casadi having been handed an array somewhere. Refuse it here, where the operation
        # that produced it is still on the stack, instead of three layers later inside
        # casadi's derivative code.
        if sx.shape != (1, 1):
            msg = (
                f"SXW holds exactly one scalar; got a casadi value of shape {sx.shape}. "
                f"A symbolic array is an SXArray of SXW scalars, never one SXW around a "
                f"casadi matrix."
            )
            raise TypeError(msg)
        self._value: SX = sx

    def __repr__(self) -> str:
        """Return the printable representation of the object."""
        return f"{SXW.__name__}({self._value!r})"

    def __bool__(self) -> bool:
        """Refuse a truth value, as casadi's own SX does.

        A Python ``if``, ``and``, ``or``, ``not``, the builtins ``max``, ``min``, ``sorted``,
        and ``in``, and numpy's ``where``, ``clip``, ``all``, ``any``, and the ``max``/``min``
        reductions all ask for it. Without this method Python answers ``True``, and every one
        of those silently takes one branch under the ``"auto"`` derivative method while the
        finite-difference methods evaluate the real condition. The message names the
        symbolic spellings.
        """
        msg = (
            "The truth value of a symbolic value is undefined. Use yapss.math.where for a "
            "conditional expression, yapss.math.clip, maximum, or minimum for a bound, and "
            "yapss.math.all or any for a mask. A Python `if`, `and`, `or`, or `not`, the "
            "builtins max, min, and sorted, and `in` all take the truth value and so cannot "
            "be used on a callback argument."
        )
        raise TypeError(msg)

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Implement the array ufunc protocol by dispatching into :data:`UFUNCS`."""
        # A reduction of one scalar is that scalar: np.sum(w), np.max(w), np.all(w).
        if method == "reduce" and len(inputs) == 1 and isinstance(inputs[0], SXW):
            return inputs[0]
        if method != "__call__":
            return NotImplemented

        # ``out=`` cannot be honored, and must not be forwarded.
        #
        # NDArrayOperatorsMixin spells every augmented assignment as
        # ``ufunc(self, other, out=(self,))``, so a plain ``d += ...`` in a user callback
        # arrives here with an SXW in ``out``. numpy's dispatch considers ``out`` operands as
        # well as inputs, so a forwarded SXW re-entered this method with byte-identical
        # arguments whenever casadi declined the inner call -- on casadi 3.7.2 for a unary
        # ufunc it lacks, and under ``GlobalOptions.setNumpyMode(1)`` (casadi >= 3.8) for
        # every ufunc -- and recursed without bound. An SXW wraps an immutable casadi value
        # and could never serve as an output buffer; augmented assignment is unaffected,
        # because ``__iadd__`` rebinds the name from the return value.
        out = kwargs.pop("out", None)
        if out is not None and any(isinstance(item, SXW) for item in out):
            out = None
        if not any(is_symbolic(item) for item in inputs):
            # only a dropped ``out`` was symbolic; the operation itself is numpy's
            return ufunc(*inputs, **({"out": out} if out is not None else {}), **kwargs)
        if kwargs:
            msg = (
                f"numpy.{ufunc.__name__}: keyword arguments ({', '.join(kwargs)}) are not "
                f"supported on a symbolic value."
            )
            raise TypeError(msg)
        result = _elementwise(ufunc.__name__, inputs)
        if out is None:
            return result
        # A real ndarray buffer, from ``array += w`` on an object array or SXArray: numpy
        # spells that ``np.add(array, w, out=(array,))``, and here the buffer can be honored.
        (buffer,) = out
        buffer[...] = result
        return buffer

    def __getattr__(self, name: str) -> Callable[..., SXW]:
        """Supply the methods numpy's object-dtype loop calls on array elements.

        ``np.sqrt(array)`` on an object array calls ``element.sqrt()`` on each element; the
        method of that name is looked up here and resolved through :data:`UNARY`, so an
        array element and a scalar reach the same implementation.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name in REJECTED:
            raise UnsupportedMathFunctionError(rejected_message(name))
        implementation = UNARY.get(name)
        if implementation is None:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

        def method(*_args: Any, **_kwargs: Any) -> SXW:
            return SXW(implementation(self._value))

        return method

    # Python-level rounding protocols, which are not ufuncs

    def __floor__(self) -> SXW:
        """Return the floor of the argument."""
        return SXW(ca.floor(self._value))

    def __ceil__(self) -> SXW:
        """Return the ceiling of the argument."""
        return SXW(ca.ceil(self._value))

    def __trunc__(self) -> SXW:
        """Return the argument truncated toward zero."""
        return SXW(_trunc(self._value))

    def __round__(self, ndigits: int | None = None) -> SXW:
        """Round half to even, as the builtin does; refuse `ndigits`.

        ``round(x)`` on a float is half-to-even on the double, which :func:`_rint`
        reproduces. ``round(x, n)`` is not: Python rounds the *exact decimal* value of the
        double (``round(2.675, 2)`` is ``2.67``), which no sequence of double-precision
        operations reproduces, and which ``numpy.round`` does not attempt (it gives
        ``2.68``). A symbol therefore cannot give the finite-difference answer, and the
        call is refused in favor of ``yapss.math.round``, whose two paths agree.
        """
        if ndigits is not None:
            msg = (
                "round(x, ndigits) on a symbolic value is not supported: the builtin rounds "
                "the exact decimal value of a float, which has no symbolic equivalent. Use "
                "yapss.math.round(x, decimals), which follows numpy under every derivative "
                "method."
            )
            raise UnsupportedMathFunctionError(msg)
        return SXW(_rint(self._value))


class SXArray(np.ndarray[Any, np.dtype[Any]]):
    """Object-dtype ndarray of :class:`SXW` that keeps comparisons symbolic.

    numpy's object-dtype comparison loop coerces each elementwise result to ``bool``, which
    :class:`SXW` refuses. Overriding the comparison and mask operators on the array type is
    the only place the operator form can be caught: ``yapss.math`` never sees it.

    Every symbolic array a callback receives is one of these, and every array-valued result
    of an operation on a symbol is returned as one.
    """

    # ensure the subclass wins reflected operations against plain ndarrays
    __array_priority__ = 20.0

    def __array_wrap__(
        self,
        array: NDArray[Any],
        context: Any = None,
        return_scalar: bool = False,  # noqa: FBT001, FBT002  (numpy's signature)
    ) -> Any:
        """Keep array results as SXArray, and give a 0-d result back as its element.

        numpy re-wraps a reduction in the subclass, so without this ``np.sum(sxarray)``
        is a 0-d SXArray rather than the SXW it holds.
        """
        if array.ndim == 0:
            return array[()]
        return array.view(SXArray)

    def _elementwise(self, other: Any, name: str) -> SXArray:
        """Apply the two-argument ufunc `name` elementwise, broadcasting `other`."""
        result = _elementwise(name, (self, other))
        return result if isinstance(result, SXArray) else sx_array([result])

    # comparison operators

    def __eq__(self, other: Any) -> SXArray:  # type: ignore[override]
        """Return the elementwise symbolic equality."""
        return self._elementwise(other, "equal")

    def __ne__(self, other: Any) -> SXArray:  # type: ignore[override]
        """Return the elementwise symbolic inequality."""
        return self._elementwise(other, "not_equal")

    def __lt__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic less-than."""
        return self._elementwise(other, "less")

    def __le__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic less-than-or-equal."""
        return self._elementwise(other, "less_equal")

    def __gt__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic greater-than."""
        return self._elementwise(other, "greater")

    def __ge__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic greater-than-or-equal."""
        return self._elementwise(other, "greater_equal")

    # mask combination: `&`, `|`, and `~`, since `and`, `or`, and `not` cannot be
    # overloaded in Python

    def __and__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical and."""
        return self._elementwise(other, "logical_and")

    def __rand__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical and."""
        return self._elementwise(other, "logical_and")

    def __or__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical or."""
        return self._elementwise(other, "logical_or")

    def __ror__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical or."""
        return self._elementwise(other, "logical_or")

    def __invert__(self) -> SXArray:
        """Return the elementwise symbolic logical not."""
        result = _elementwise("logical_not", (self,))
        return result if isinstance(result, SXArray) else sx_array([result])

    # __eq__ is overridden above, so __hash__ must be restated, as for SXW
    __hash__: ClassVar[None] = None


def sx_array(values: Any) -> SXArray:
    """Build an :class:`SXArray` from a sequence of :class:`SXW` values."""
    return np.array(values, dtype=object).view(SXArray)
