"""

Provides a set of functions that are compatible with both NumPy and CasADi.

"""

from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np

from .wrapper import SXW

__all__ = [
    "UnsupportedMathFunctionError",
    "arctan2",
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
    "maximum",
    "minimum",
    "mod",
    "nextafter",
    "not_equal",
    "power",
    "remainder",
    "rint",
    "sign",
    "signbit",
    "spacing",
]


def vectorized_two_arg_func(
    func: Callable[..., Any],
    casadi_func: Callable[..., Any],
) -> Callable[[Any, Any], Any]:
    """
    Wrap a two-argument function to handle both SXW and numeric inputs.

    Parameters
    ----------
    func : Callable
        A numeric function that takes two arguments (e.g., np.arctan2 or np.maximum).
    casadi_func : Callable
        A CasADi-compatible function for SXW inputs (e.g., ca.arctan2 or ca.fmax).

    Returns
    -------
    Callable
        A vectorized version of the function that applies element-wise and handles mixed types.
    """

    def _elementwise_func(x: Any, y: Any) -> Any:
        if isinstance(x, SXW) or isinstance(y, SXW):
            x_value = x._value if isinstance(x, SXW) else x
            y_value = y._value if isinstance(y, SXW) else y
            return SXW(casadi_func(x_value, y_value))

        return func(x, y)

    vectorized_func = np.vectorize(_elementwise_func)

    def wrapped_func(x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            np.array(x, dtype=np.float64)
            np.array(y, dtype=np.float64)
            return func(x, y, *args, **kwargs)
        except TypeError:
            pass

        result = vectorized_func(x, y, *args, **kwargs)
        return result.item() if result.ndim == 0 else result

    return wrapped_func


def vectorized_one_arg_func(
    func: Callable[..., Any],
    casadi_func: Callable[..., Any],
) -> Callable[[Any], Any]:
    """
    Wrap a one-argument function to handle both SXW and numeric inputs.

    Parameters
    ----------
    func : Callable
        A numeric function that takes one argument (e.g., np.sqrt or np.sin).
    casadi_func : Callable
        A CasADi-compatible function for SXW inputs (e.g., ca.sqrt or ca.sin).

    Returns
    -------
    Callable
        A vectorized version of the function that applies element-wise and handles mixed types.
    """

    def _elementwise_func(x: Any) -> Any:
        if isinstance(x, SXW):
            x_value = x._value
            return SXW(casadi_func(x_value))

        return func(x)

    vectorized_func = np.vectorize(_elementwise_func)

    def wrapped_func(x: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            np.array(x, dtype=np.float64)
            return func(x, *args, **kwargs)
        except TypeError:
            pass
        result = vectorized_func(x, *args, **kwargs)
        return result.item() if result.ndim == 0 else result

    return wrapped_func


arctan2 = vectorized_two_arg_func(np.arctan2, ca.atan2)
maximum = vectorized_two_arg_func(np.maximum, ca.fmax)
minimum = vectorized_two_arg_func(np.minimum, ca.fmin)
power = vectorized_two_arg_func(np.power, ca.power)
hypot = vectorized_two_arg_func(np.hypot, ca.hypot)

sign = vectorized_one_arg_func(np.sign, ca.sign)

# Comparison and logical functions.
#
# These must be dispatched explicitly rather than left to numpy. numpy's object-dtype
# loop coerces each elementwise result to bool; SXW defines no __bool__, so Python's
# default makes every comparison True. That silently discards masks such as
# ``(abs(y) <= 1) * x``, which evaluate correctly under the finite-difference derivative
# methods but not under "auto". CasADi represents all of these exactly, so the two paths
# can and should agree.

equal = vectorized_two_arg_func(np.equal, ca.eq)
not_equal = vectorized_two_arg_func(np.not_equal, ca.ne)
less = vectorized_two_arg_func(np.less, ca.lt)
less_equal = vectorized_two_arg_func(np.less_equal, ca.le)
greater = vectorized_two_arg_func(np.greater, ca.gt)
greater_equal = vectorized_two_arg_func(np.greater_equal, ca.ge)

logical_and = vectorized_two_arg_func(np.logical_and, ca.logic_and)
logical_or = vectorized_two_arg_func(np.logical_or, ca.logic_or)
logical_not = vectorized_one_arg_func(np.logical_not, ca.logic_not)


# Modular arithmetic.
#
# numpy uses two different conventions and casadi supplies only one of them: `fmod`
# truncates, taking the sign of the dividend, while `mod` and `remainder` floor, taking
# the sign of the divisor. Getting these backwards is silent, so the conformance test
# samples all four sign combinations.

fmod = vectorized_two_arg_func(np.fmod, ca.fmod)
mod = vectorized_two_arg_func(np.mod, lambda x, y: x - y * ca.floor(x / y))
remainder = vectorized_two_arg_func(np.remainder, lambda x, y: x - y * ca.floor(x / y))
floor_divide = vectorized_two_arg_func(np.floor_divide, lambda x, y: ca.floor(x / y))

# Aliases of functions that already dispatch correctly.

fmax = maximum
fmin = minimum
float_power = power


def _logaddexp(x: Any, y: Any) -> Any:
    """Return log(exp(x) + exp(y)), computed so that large arguments do not overflow."""
    return ca.fmax(x, y) + ca.log1p(ca.exp(-ca.fabs(x - y)))


def _logaddexp2(x: Any, y: Any) -> Any:
    """Return log2(2**x + 2**y), computed so that large arguments do not overflow."""
    return ca.fmax(x, y) + ca.log1p(2 ** -ca.fabs(x - y)) / np.log(2)


def _copysign(x: Any, y: Any) -> Any:
    """Return the magnitude of x with the sign of y.

    Differs from numpy at ``y == -0.0``: numpy reads the sign bit, which a symbolic
    expression cannot represent, so negative zero is treated as positive.
    """
    return ca.if_else(ca.ge(y, 0), ca.fabs(x), -ca.fabs(x))


def _heaviside(x: Any, y: Any) -> Any:
    """Return 0 where x < 0, y where x == 0, and 1 where x > 0."""
    return ca.gt(x, 0) + ca.eq(x, 0) * y


def _logical_xor(x: Any, y: Any) -> Any:
    """Return the elementwise exclusive or of the truth values of x and y."""
    return ca.logic_and(ca.logic_or(x, y), ca.logic_not(ca.logic_and(x, y)))


logaddexp = vectorized_two_arg_func(np.logaddexp, _logaddexp)
logaddexp2 = vectorized_two_arg_func(np.logaddexp2, _logaddexp2)
copysign = vectorized_two_arg_func(np.copysign, _copysign)
heaviside = vectorized_two_arg_func(np.heaviside, _heaviside)
logical_xor = vectorized_two_arg_func(np.logical_xor, _logical_xor)


class UnsupportedMathFunctionError(TypeError):
    """Raised for a numpy function that YAPSS callbacks cannot support.

    Subclasses :class:`TypeError`, which is what numpy itself raises today when one of
    these is handed a symbolic argument.
    """


# Functions with no meaning on a symbolic value. Each raises unconditionally rather than
# only under automatic differentiation: a function that works under one derivative
# method and fails under another lets a formulation depend on the derivative method,
# which is the class of defect this module exists to prevent. Use numpy directly if one
# of these is needed outside a callback.
_REJECTED = {
    "nextafter": "steps between adjacent floating-point values",
    "rint": (
        "rounds half to even, which casadi cannot reproduce; floor(x + 0.5) rounds half "
        "away from zero and would silently disagree with numpy"
    ),
    "signbit": "reads the floating-point sign bit, including the sign of negative zero",
    "spacing": "returns the distance to the adjacent floating-point value",
}


def _make_rejected(name: str, reason: str) -> Callable[..., Any]:
    """Build a stub that explains why `name` is unsupported in a YAPSS callback."""

    def rejected(*_args: Any, **_kwargs: Any) -> Any:
        msg = (
            f"'{name}' is not supported in YAPSS callback functions because it {reason}, "
            f"which has no symbolic equivalent. Callback functions must give the same "
            f"result under every derivative method. Use 'numpy.{name}' directly if you "
            f"need it outside a callback."
        )
        raise UnsupportedMathFunctionError(msg)

    rejected.__name__ = name
    rejected.__qualname__ = name
    rejected.__doc__ = f"Raise :class:`UnsupportedMathFunctionError`; {name} {reason}."
    return rejected


nextafter = _make_rejected("nextafter", _REJECTED["nextafter"])
rint = _make_rejected("rint", _REJECTED["rint"])
signbit = _make_rejected("signbit", _REJECTED["signbit"])
spacing = _make_rejected("spacing", _REJECTED["spacing"])
