"""YAPSS — Yet Another PseudoSpectral Solver."""

# standard library imports
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._api.declare import Phases, phase
from ._api.mesh import Mesh
from ._api.problem import Problem
from ._api.sampled import interp
from ._api.solution import PhaseSolution, Solution
from ._api.vector import Empty, Vector, field
from ._backend.exceptions import (
    REMOVED_NAMES,
    LargeSegmentWarning,
    YapssDeprecationWarning,
    YapssError,
    YapssWarning,
)
from ._backend.ipopt_options import IpoptOptionSettingWarning
from ._backend.ipopt_status import IpoptStatus
from ._backend.solution import IpoptConvergenceWarning

# re-exported so that every warning and error category YAPSS can raise is discoverable
# from the top-level package; yapss.math remains their defining module
from .math.functions import UnsupportedMathFunctionError

__all__ = [
    "Empty",
    "IpoptConvergenceWarning",
    "IpoptOptionSettingWarning",
    "IpoptStatus",
    "LargeSegmentWarning",
    "Mesh",
    "PhaseSolution",
    "Phases",
    "Problem",
    "Solution",
    "UnsupportedMathFunctionError",
    "Vector",
    "YapssDeprecationWarning",
    "YapssError",
    "YapssWarning",
    "__version__",
    "field",
    "interp",
    "phase",
]

try:
    __version__ = version("yapss")
except PackageNotFoundError:
    __version__ = "0.0.0"

# hidden from type checkers: a module __getattr__ makes them accept every attribute name, so
# `yapss.Problm` would no longer be reported as a typo
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in REMOVED_NAMES:
            raise AttributeError(REMOVED_NAMES[name])
        msg = f"module 'yapss' has no attribute {name!r}"
        raise AttributeError(msg)
