"""

Stand-ins for functions CasADi cannot trace, with their coefficients as inputs.

This package depends on CasADi and NumPy and on nothing else in YAPSS, so that it can be used
with plain CasADi expressions. See `core` for how it works.

"""

from .core import External, Function, StandIn, coefficients, external, inputs, tables, tracing
from .steps import EPS, difference_steps

__all__ = [
    "EPS",
    "External",
    "Function",
    "StandIn",
    "coefficients",
    "difference_steps",
    "external",
    "inputs",
    "tables",
    "tracing",
]
