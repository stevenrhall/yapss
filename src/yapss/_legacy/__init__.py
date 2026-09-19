"""

The API of 0.3.0, under a name of its own.

Everything here was reached as ``yapss.X`` up to 0.3.0, and is reached as
``yapss._legacy.X`` now that the top-level package carries the redesigned API. The module
exists so that the old front end, and the tests and examples written against it, keep working
while the new one takes over the public name -- most of what `tests/modules` checks is really
the shared back end, reached through this front end, and that coverage is worth keeping until
it is re-anchored on the new one.

Nothing here carries a compatibility promise. Import it the way the prototype of the new API
was imported::

    from yapss import _legacy as yapss

"""

from typing import TYPE_CHECKING

from yapss._private.input_args import ContinuousArg as ContinuousArg_
from yapss._private.input_args import ContinuousHessianArg, ContinuousJacobianArg
from yapss._private.input_args import DiscreteArg as DiscreteArg_
from yapss._private.input_args import DiscreteHessianArg, DiscreteJacobianArg
from yapss._private.input_args import ObjectiveArg as ObjectiveArg_
from yapss._private.input_args import ObjectiveGradientArg, ObjectiveHessianArg
from yapss._private.ipopt_options import IpoptOptionSettingWarning
from yapss._private.ipopt_status import IpoptStatus
from yapss._private.problem import LargeSegmentWarning, Problem
from yapss._private.solution import IpoptConvergenceWarning, Solution
from yapss.math.functions import UnsupportedMathFunctionError

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
