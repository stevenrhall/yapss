"""

Drop-in replacement for numpy, with additional support for the SXW wrapper class.

"""

from __future__ import annotations

import numpy as _np  # noqa: ICN001

from yapss.math import functions

__all__ = [  # noqa: RUF022
    "abs",
    "absolute",
    "add",
    "all",
    "any",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
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
    "divmod",
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
    "frexp",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "invert",
    "lcm",
    "ldexp",
    "left_shift",
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
    "modf",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "pi",
    "positive",
    "pow",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
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

from numpy import *  # noqa: F403
from numpy import abs  # noqa: A004
from numpy import all  # noqa: A004
from numpy import any  # noqa: A004
from numpy import divmod  # noqa: A004
from numpy import max  # noqa: A004
from numpy import min  # noqa: A004
from numpy import round  # noqa: A004
from numpy import sum  # noqa: A004
from numpy import (
    absolute,
    add,
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
    frexp,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    invert,
    lcm,
    ldexp,
    left_shift,
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
    modf,
    multiply,
    negative,
    nextafter,
    not_equal,
    pi,
    positive,
    power,
    rad2deg,
    radians,
    reciprocal,
    remainder,
    right_shift,
    rint,
    sign,
    signbit,
    sin,
    sinh,
    spacing,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    true_divide,
    trunc,
    where,
)

# Dynamically re-export all other attributes from numpy
for _attr in dir(_np):
    if not _attr.startswith("_"):  # Skip private attributes
        globals()[_attr] = getattr(_np, _attr)

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
globals()["nextafter"] = functions.nextafter
globals()["rint"] = functions.rint
globals()["signbit"] = functions.signbit
globals()["spacing"] = functions.spacing
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
globals()["UnsupportedMathFunctionWarning"] = functions.UnsupportedMathFunctionWarning
