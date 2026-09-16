"""YAPSS — Yet Another PseudoSpectral Solver."""

# standard library imports
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._private.exceptions import YapssDeprecationWarning, YapssError, YapssWarning

# module imports
from ._private.input_args import ContinuousArg as ContinuousArg_
from ._private.input_args import ContinuousHessianArg, ContinuousJacobianArg
from ._private.input_args import DiscreteArg as DiscreteArg_
from ._private.input_args import DiscreteHessianArg, DiscreteJacobianArg
from ._private.input_args import ObjectiveArg as ObjectiveArg_
from ._private.input_args import ObjectiveGradientArg, ObjectiveHessianArg
from ._private.problem import Problem
from ._private.setup_check import UnsetOutputWarning
from ._private.solution import IpoptConvergenceWarning, Solution
from ._private.solver import IpoptOptionSettingWarning
from ._private.user import MirroredHessianPairWarning

# re-exported so that every warning and error category YAPSS can raise is discoverable
# from the top-level package; yapss.math remains their defining module
from .math.functions import UnsupportedMathFunctionError, UnsupportedMathFunctionWarning

__all__ = [
    "ContinuousArg",
    "ContinuousHessianArg",
    "ContinuousJacobianArg",
    "DiscreteArg",
    "DiscreteHessianArg",
    "DiscreteJacobianArg",
    "IpoptConvergenceWarning",
    "IpoptOptionSettingWarning",
    "MirroredHessianPairWarning",
    "ObjectiveArg",
    "ObjectiveGradientArg",
    "ObjectiveHessianArg",
    "Problem",
    "Solution",
    "UnsetOutputWarning",
    "UnsupportedMathFunctionError",
    "UnsupportedMathFunctionWarning",
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
