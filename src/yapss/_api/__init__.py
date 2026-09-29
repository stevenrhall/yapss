"""

The YAPSS API.

This package holds the front end that `yapss` itself exports: `Problem`, the declarations a
problem is written in, and the solution it returns. Nothing imports from here directly --
``import yapss`` reaches all of it.

"""

from .args import ContinuousArg, ContinuousOut, DiscreteArg, DiscreteOut
from .declare import Phase, Phases
from .mesh import Mesh
from .options import DerivativeMethod, DerivativeOrder, ObjectiveSense, SpectralMethod
from .problem import Problem
from .sampled import interp
from .solution import PhaseSolution, Solution
from .vector import Control, Discrete, Integral, Parameter, Path, State, scalar, vector

__all__ = [
    "ContinuousArg",
    "ContinuousOut",
    "Control",
    "DerivativeMethod",
    "DerivativeOrder",
    "Discrete",
    "DiscreteArg",
    "DiscreteOut",
    "Integral",
    "Mesh",
    "ObjectiveSense",
    "Parameter",
    "Path",
    "Phase",
    "PhaseSolution",
    "Phases",
    "Problem",
    "Solution",
    "SpectralMethod",
    "State",
    "interp",
    "scalar",
    "vector",
]
