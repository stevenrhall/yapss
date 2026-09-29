"""

The values the problem's four named options take, as types a user's own code can name.

``problem.spectral_method``, ``problem.derivatives.method``, ``problem.derivatives.order`` and
``problem.objective.sense`` are typed with these, so a type checker reports a misspelled value
where it is assigned. A value that reaches the setter through a variable typed only ``str`` is
reported too, since a checker cannot tell which string it holds; annotating the variable with
the alias, ``def setup(method: yapss.SpectralMethod = "lgl")``, is the cure. The run-time check
at the assignment is the same either way.

"""

from typing import Literal, TypeAlias, get_args

__all__ = [
    "DERIVATIVE_METHODS",
    "DERIVATIVE_ORDERS",
    "OBJECTIVE_SENSES",
    "SPECTRAL_METHODS",
    "DerivativeMethod",
    "DerivativeOrder",
    "ObjectiveSense",
    "SpectralMethod",
]

SpectralMethod: TypeAlias = Literal["lgl", "lgr", "lg"]
"""The collocation method, ``problem.spectral_method``."""

DerivativeMethod: TypeAlias = Literal["auto", "central-difference", "central-difference-full"]
"""How derivatives are computed, ``problem.derivatives.method``."""

DerivativeOrder: TypeAlias = Literal["first", "second"]
"""The order of the derivatives Ipopt is given, ``problem.derivatives.order``."""

ObjectiveSense: TypeAlias = Literal["minimize", "maximize"]
"""Whether the objective is minimized or maximized, ``problem.objective.sense``."""

SPECTRAL_METHODS: tuple[str, ...] = get_args(SpectralMethod)
DERIVATIVE_METHODS: tuple[str, ...] = get_args(DerivativeMethod)
DERIVATIVE_ORDERS: tuple[str, ...] = get_args(DerivativeOrder)
OBJECTIVE_SENSES: tuple[str, ...] = get_args(ObjectiveSense)
