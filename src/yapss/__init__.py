"""YAPSS — Yet Another PseudoSpectral Solver."""

# standard library imports
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._private.exceptions import (
    REMOVED_NAMES,
    YapssDeprecationWarning,
    YapssError,
    YapssWarning,
)

# module imports
from ._private.input_args import ContinuousArg as ContinuousArg_
from ._private.input_args import ContinuousHessianArg, ContinuousJacobianArg
from ._private.input_args import DiscreteArg as DiscreteArg_
from ._private.input_args import DiscreteHessianArg, DiscreteJacobianArg
from ._private.input_args import ObjectiveArg as ObjectiveArg_
from ._private.input_args import ObjectiveGradientArg, ObjectiveHessianArg
from ._private.ipopt_options import IpoptOptionSettingWarning
from ._private.ipopt_status import IpoptStatus
from ._private.problem import LargeSegmentWarning, Problem
from ._private.setup_check import UnsetOutputWarning
from ._private.solution import IpoptConvergenceWarning, Solution

# re-exported so that every warning and error category YAPSS can raise is discoverable
# from the top-level package; yapss.math remains their defining module
from .math.functions import UnsupportedMathFunctionError

__all__ = [
    "ContinuousArg",
    "ContinuousHessianArg",
    "ContinuousJacobianArg",
    "DiscreteArg",
    "DiscreteHessianArg",
    "DiscreteJacobianArg",
    "IpoptConvergenceWarning",
    "IpoptOptionSettingWarning",
    "IpoptStatus",
    "LargeSegmentWarning",
    "ObjectiveArg",
    "ObjectiveGradientArg",
    "ObjectiveHessianArg",
    "Problem",
    "Solution",
    "UnsetOutputWarning",
    "UnsupportedMathFunctionError",
    "YapssDeprecationWarning",
    "YapssError",
    "YapssWarning",
    "__version__",
]

try:
    __version__ = version("yapss")
except PackageNotFoundError:
    __version__ = "0.0.0"

# The three generic argument types are exported as the classes themselves, so that
# ``isinstance(arg, yapss.ContinuousArg)`` works on every instance a callback receives,
# whatever its element type (a subscripted generic is refused by isinstance, and the
# other six argument types are plain classes). For a type checker they are the float64
# specialization, which is what a callback's annotation means.
if TYPE_CHECKING:
    import numpy as np

    ContinuousArg = ContinuousArg_[np.float64]
    DiscreteArg = DiscreteArg_[np.float64]
    ObjectiveArg = ObjectiveArg_[np.float64]
else:
    ContinuousArg = ContinuousArg_
    DiscreteArg = DiscreteArg_
    ObjectiveArg = ObjectiveArg_

# hidden from type checkers: a module __getattr__ makes them accept every attribute name, so
# `yapss.Problm` would no longer be reported as a typo
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in REMOVED_NAMES:
            raise AttributeError(REMOVED_NAMES[name])
        msg = f"module 'yapss' has no attribute {name!r}"
        raise AttributeError(msg)
