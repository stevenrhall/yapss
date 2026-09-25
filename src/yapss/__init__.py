"""YAPSS — Yet Another PseudoSpectral Solver."""

# standard library imports
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._api.args import ContinuousArg, ContinuousOut, DiscreteOut, EndpointArg
from ._api.declare import Independent, Phase, Phases
from ._api.mesh import Mesh
from ._api.old_api import OLD_ROOT_NAMES, old_api_message
from ._api.problem import Problem
from ._api.sampled import interp
from ._api.solution import PhaseSolution, Solution
from ._api.vector import Control, Discrete, Integral, Parameter, Path, State, scalar, vector
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
    "ContinuousArg",
    "ContinuousOut",
    "Control",
    "Discrete",
    "DiscreteOut",
    "EndpointArg",
    "Independent",
    "Integral",
    "IpoptConvergenceWarning",
    "IpoptOptionSettingWarning",
    "IpoptStatus",
    "LargeSegmentWarning",
    "Mesh",
    "Parameter",
    "Path",
    "Phase",
    "PhaseSolution",
    "Phases",
    "Problem",
    "Solution",
    "State",
    "UnsupportedMathFunctionError",
    "YapssDeprecationWarning",
    "YapssError",
    "YapssWarning",
    "__version__",
    "interp",
    "scalar",
    "vector",
]

try:
    __version__ = version("yapss")
except PackageNotFoundError:
    __version__ = "0.0.0"

# hidden from type checkers: a module __getattr__ makes them accept every attribute name, so
# `yapss.Problm` would no longer be reported as a typo
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        # ImportError, not AttributeError: `from yapss import X` replaces an AttributeError
        # from here with a generic "cannot import name" and keeps nothing of the message
        if name in REMOVED_NAMES:
            raise ImportError(REMOVED_NAMES[name])
        if name in OLD_ROOT_NAMES:
            raise ImportError(old_api_message(f"yapss.{name}"))
        msg = f"module 'yapss' has no attribute {name!r}"
        raise AttributeError(msg)
