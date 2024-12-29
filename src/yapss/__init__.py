"""YAPSS — Yet Another PseudoSpectral Solver."""

# standard library imports
from importlib.metadata import PackageNotFoundError, version

# package imports
import numpy as np

# module imports
from ._private.input_args import ContinuousArg as ContinuousArg_
from ._private.input_args import ContinuousHessianArg, ContinuousJacobianArg
from ._private.input_args import DiscreteArg as DiscreteArg_
from ._private.input_args import DiscreteHessianArg, DiscreteJacobianArg
from ._private.input_args import ObjectiveArg as ObjectiveArg_
from ._private.input_args import ObjectiveGradientArg, ObjectiveHessianArg
from ._private.problem import Problem
from ._private.solution import Solution

__all__ = [
    "ContinuousArg",
    "ContinuousHessianArg",
    "ContinuousJacobianArg",
    "DiscreteArg",
    "DiscreteHessianArg",
    "DiscreteJacobianArg",
    "ObjectiveArg",
    "ObjectiveGradientArg",
    "ObjectiveHessianArg",
    "Problem",
    "Solution",
    "__version__",
]

try:
    __version__ = version("yapss")
except PackageNotFoundError:
    __version__ = "0.0.0"

ContinuousArg = ContinuousArg_[np.float64]
DiscreteArg = DiscreteArg_[np.float64]
ObjectiveArg = ObjectiveArg_[np.float64]

del np
