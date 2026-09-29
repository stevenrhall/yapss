"""

The API of 0.3.0, kept as test support.

Everything here was reached as ``yapss.X`` up to 0.3.0. It lives under `tests/` so that the
tests written against it keep exercising the shared back end until they are re-anchored on
the new front end, and so that no copy of the old front end ships in the package. Nothing here
carries a compatibility promise. Import it as::

    from tests.support import legacy as yapss

"""

from typing import TYPE_CHECKING

from yapss._backend.exceptions import LargeSegmentWarning
from yapss._backend.input_args import ContinuousArg as ContinuousArg_
from yapss._backend.input_args import ContinuousHessianArg, ContinuousJacobianArg
from yapss._backend.input_args import DiscreteArg as DiscreteArg_
from yapss._backend.input_args import DiscreteHessianArg, DiscreteJacobianArg
from yapss._backend.input_args import ObjectiveArg as ObjectiveArg_
from yapss._backend.input_args import ObjectiveGradientArg, ObjectiveHessianArg
from yapss._backend.ipopt_options import IpoptOptionSettingWarning
from yapss._backend.ipopt_status import IpoptStatus
from yapss._backend.solution import IpoptConvergenceWarning, Solution
from yapss.math.functions import UnsupportedMathFunctionError

from .problem import Problem

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
    "UnsupportedMathFunctionError",
]

# The three generic argument types are exported as the classes themselves, so that
# ``isinstance(arg, yapss.ContinuousArg)`` works on every instance a callback receives,
# whatever its element type (a subscripted generic is refused by isinstance, and the other
# six argument types are plain classes). For a type checker they are the float64
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
