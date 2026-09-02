"""

Module-level functions of ``yapss.math`` that need symbolic dispatch.

A unary ufunc such as ``sin`` needs nothing here: numpy dispatches it to
:meth:`SXW.__array_ufunc__` for a scalar and to the element method for an array, and both
resolve through :data:`yapss.math.wrapper.UFUNCS`. The functions below are the ones numpy
would otherwise get wrong on a symbol -- by coercing to ``bool``, by handing casadi an array,
or by having no symbolic meaning at all.

"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from .wrapper import (
    REDUCTIONS,
    REJECTED,
    UnsupportedMathFunctionError,
    UnsupportedMathFunctionWarning,
    _round,
    apply_ufunc,
    is_symbolic,
    map_unary,
    reduce_symbolic,
    rejected_message,
)

__all__ = [
    "UnsupportedMathFunctionError",
    "UnsupportedMathFunctionWarning",
    "all",
    "amax",
    "amin",
    "any",
    "arctan2",
    "clip",
    "copysign",
    "equal",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "max",
    "maximum",
    "min",
    "minimum",
    "mod",
    "nextafter",
    "not_equal",
    "power",
    "remainder",
    "rint",
    "round",
    "sign",
    "signbit",
    "spacing",
    "where",
]


def _ufunc(name: str) -> Callable[..., Any]:
    """Return ``numpy.<name>`` extended to symbolic arguments."""

    def function(*args: Any, **kwargs: Any) -> Any:
        return apply_ufunc(name, *args, **kwargs)

    function.__name__ = name
    function.__qualname__ = name
    function.__doc__ = (
        f"``numpy.{name}`` on real arguments; its casadi implementation, elementwise, on "
        f"symbolic ones."
    )
    return function


def _reduction(name: str) -> Callable[..., Any]:
    """Return ``numpy.<name>`` extended to a full reduction of symbolic arguments."""

    def function(value: Any, axis: Any = None, **kwargs: Any) -> Any:
        return reduce_symbolic(name, value, axis=axis, **kwargs)

    function.__name__ = name
    function.__qualname__ = name
    function.__doc__ = (
        f"``numpy.{name}`` on real arguments; on symbolic ones, an exact fold of "
        f"``{REDUCTIONS[name].__name__}`` over every element (``axis`` is not supported)."
    )
    return function


def _rejected(name: str) -> Callable[..., Any]:
    """Build a stub that rejects `name`, which is unsupported in a YAPSS callback."""
    numpy_function = getattr(np, name)
    message = rejected_message(name)

    def rejected(*args: Any, **kwargs: Any) -> Any:
        try:
            for arg in args:
                np.array(arg, dtype=np.float64)
        except TypeError:
            # A symbolic argument. This raised before 0.2.2 as well.
            raise UnsupportedMathFunctionError(message) from None
        warnings.warn(
            f"{message} It still evaluates on real arguments, but will raise "
            f"UnsupportedMathFunctionError in 0.3.0.",
            UnsupportedMathFunctionWarning,
            stacklevel=2,
        )
        return numpy_function(*args, **kwargs)

    rejected.__name__ = name
    rejected.__qualname__ = name
    rejected.__doc__ = (
        f"Raise :class:`UnsupportedMathFunctionError` on a symbolic argument, or warn "
        f"with :class:`UnsupportedMathFunctionWarning` and evaluate on a real one; "
        f"{name} {REJECTED[name]}."
    )
    return rejected


# Two-argument functions. numpy's object loop cannot dispatch these to an element method
# the way it does a unary ufunc, and the comparison and logical ones would coerce their
# result to bool.
arctan2 = _ufunc("arctan2")
hypot = _ufunc("hypot")
power = _ufunc("power")
maximum = _ufunc("maximum")
minimum = _ufunc("minimum")
fmod = _ufunc("fmod")
mod = _ufunc("mod")
remainder = _ufunc("remainder")
floor_divide = _ufunc("floor_divide")
copysign = _ufunc("copysign")
heaviside = _ufunc("heaviside")
logaddexp = _ufunc("logaddexp")
logaddexp2 = _ufunc("logaddexp2")
equal = _ufunc("equal")
not_equal = _ufunc("not_equal")
less = _ufunc("less")
less_equal = _ufunc("less_equal")
greater = _ufunc("greater")
greater_equal = _ufunc("greater_equal")
logical_and = _ufunc("logical_and")
logical_or = _ufunc("logical_or")
logical_xor = _ufunc("logical_xor")
logical_not = _ufunc("logical_not")
sign = _ufunc("sign")

# Aliases, deliberately of maximum and minimum rather than of numpy's NaN-ignoring np.fmax
# and np.fmin: the central-difference methods find the sparsity structure by setting one
# variable to NaN and recording which outputs come back NaN
# (finite_difference.get_continuous_jacobian_structure_nan), so a NaN-absorbing fmax would
# swallow the probe and hide a real dependency, yielding a silently incomplete Jacobian.
# The cost is that fmax disagrees with np.fmax on NaN input. Note also that ca.fmax ignores
# NaN, so under "auto" all four return the other operand instead of propagating the NaN; a
# callback should not be producing NaN in the first place.
fmax = maximum
fmin = minimum
float_power = power

# Three-argument functions that numpy evaluates by coercing a comparison to bool, and so
# silently returned their first argument on a symbol before 0.2.3.
clip = _ufunc("clip")
where = _ufunc("where")

# Reductions whose numpy loops coerce to bool. sum and prod need nothing: numpy reduces
# them through __add__ and __mul__.
max = _reduction("max")  # noqa: A001
min = _reduction("min")  # noqa: A001
amax = _reduction("amax")
amin = _reduction("amin")
all = _reduction("all")  # noqa: A001
any = _reduction("any")  # noqa: A001

# Rounding. rint is a ufunc and needs only its table entry; round takes a `decimals`
# parameter, which is not an operand, so it is spelled out. Both round half to even,
# exactly as numpy does -- see wrapper._rint.
rint = _ufunc("rint")


def round(x: Any, decimals: int = 0) -> Any:  # noqa: A001
    """Round half to even, to `decimals` places, exactly as ``numpy.round`` does."""
    if not is_symbolic(x):
        return np.round(x, decimals)
    return map_unary(
        lambda value: _round(value, decimals),
        lambda value: np.round(value, decimals),
        x,
    )


# Functions with no symbolic meaning; see REJECTED in yapss.math.wrapper.
nextafter = _rejected("nextafter")
signbit = _rejected("signbit")
spacing = _rejected("spacing")
