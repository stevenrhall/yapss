"""

The math functions that work under every derivative method.

Each is NumPy's function of the same name on real arguments, and also takes the symbolic ones a
callback is given under ``"auto"``. ``fmax``, ``fmin`` and ``where`` differ from NumPy's: they
return NaN where NumPy's would drop it, since the central-difference methods find the sparsity
structure by setting a variable to NaN. The module provides nothing else; the rest of NumPy is
imported from NumPy.

"""

import typing as _typing

import numpy as _np  # noqa: ICN001

from yapss._backend.exceptions import REMOVED_NAMES as _REMOVED_NAMES
from yapss.math import functions

__all__ = [  # noqa: RUF022
    "abs",
    "absolute",
    "acos",
    "acosh",
    "add",
    "all",
    "amax",
    "amin",
    "any",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "cbrt",
    "ceil",
    "clip",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "invert",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "max",
    "maximum",
    "min",
    "minimum",
    "mod",
    "multiply",
    "negative",
    "not_equal",
    "pi",
    "positive",
    "pow",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "rint",
    "round",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    "where",
]

from numpy import abs  # noqa: A004
from numpy import all  # noqa: A004
from numpy import any  # noqa: A004
from numpy import max  # noqa: A004
from numpy import min  # noqa: A004
from numpy import round  # noqa: A004
from numpy import sum  # noqa: A004
from numpy import (
    absolute,
    add,
    amax,
    amin,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cbrt,
    ceil,
    clip,
    conj,
    conjugate,
    copysign,
    cos,
    cosh,
    deg2rad,
    degrees,
    divide,
    equal,
    exp,
    exp2,
    expm1,
    fabs,
    float_power,
    floor,
    floor_divide,
    fmax,
    fmin,
    fmod,
    greater,
    greater_equal,
    heaviside,
    hypot,
    invert,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    matmul,
    maximum,
    minimum,
    mod,
    multiply,
    negative,
    not_equal,
    pi,
    positive,
    power,
    rad2deg,
    radians,
    reciprocal,
    remainder,
    rint,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    true_divide,
    trunc,
    where,
)

# NumPy 2's names for the same functions, bound here so that they need no NumPy 2
acos = arccos
acosh = arccosh
asin = arcsin
asinh = arcsinh
atan = arctan
atan2 = arctan2
atanh = arctanh
pow = power  # noqa: A001

globals()["atan2"] = functions.arctan2
globals()["arctan2"] = functions.arctan2
globals()["hypot"] = functions.hypot
globals()["maximum"] = functions.maximum
globals()["minimum"] = functions.minimum
globals()["power"] = functions.power
globals()["sign"] = functions.sign
globals()["equal"] = functions.equal
globals()["not_equal"] = functions.not_equal
globals()["less"] = functions.less
globals()["less_equal"] = functions.less_equal
globals()["greater"] = functions.greater
globals()["greater_equal"] = functions.greater_equal
globals()["logical_and"] = functions.logical_and
globals()["logical_or"] = functions.logical_or
globals()["logical_not"] = functions.logical_not
globals()["fmod"] = functions.fmod
globals()["mod"] = functions.mod
globals()["remainder"] = functions.remainder
globals()["floor_divide"] = functions.floor_divide
globals()["fmax"] = functions.fmax
globals()["fmin"] = functions.fmin
globals()["float_power"] = functions.float_power
globals()["logaddexp"] = functions.logaddexp
globals()["logaddexp2"] = functions.logaddexp2
globals()["copysign"] = functions.copysign
globals()["heaviside"] = functions.heaviside
globals()["logical_xor"] = functions.logical_xor
globals()["rint"] = functions.rint
globals()["round"] = functions.round
globals()["clip"] = functions.clip
globals()["where"] = functions.where
globals()["max"] = functions.max
globals()["min"] = functions.min
globals()["amax"] = functions.amax
globals()["amin"] = functions.amin
globals()["all"] = functions.all
globals()["any"] = functions.any
globals()["UnsupportedMathFunctionError"] = functions.UnsupportedMathFunctionError

# hidden from type checkers, as in yapss/__init__.py
if not _typing.TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        # ImportError, not AttributeError: `from yapss.math import X` replaces an AttributeError
        # from here with a generic "cannot import name" and keeps nothing of the message
        if name in _REMOVED_NAMES:
            raise ImportError(_REMOVED_NAMES[name])
        msg = f"module 'yapss.math' has no attribute {name!r}"
        # AttributeError, so that hasattr() stays False for numpy's several hundred names;
        # `from yapss.math import linspace` then shows Python's own "cannot import name"
        if not name.startswith("_") and name in dir(_np):
            msg += (
                f"; yapss.math provides only the math functions it supports in callbacks under "
                f"every derivative method, and {name!r} is not one of them. If needed outside a "
                f'callback, or with a derivative method other than "auto", import it from numpy.'
            )
        raise AttributeError(msg)
