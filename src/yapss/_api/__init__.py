"""

The YAPSS API.

This package holds the front end that `yapss` itself exports: `Problem`, the declarations a
problem is written in, and the solution it returns. Nothing imports from here directly --
``import yapss`` reaches all of it -- and the package keeps its own name only until the two
front ends finish changing places. The 0.3.0 API it replaced is `yapss._legacy`.

"""

from .args import ContinuousArg, ContinuousOut, DiscreteOut, EndpointArg
from .declare import Independent, Phase, Phases
from .mesh import Mesh
from .problem import Problem
from .sampled import interp
from .solution import PhaseSolution, Solution
from .vector import Control, Discrete, Integral, Parameter, Path, State, scalar, vector

__all__ = [
    "ContinuousArg",
    "ContinuousOut",
    "Control",
    "Discrete",
    "DiscreteOut",
    "EndpointArg",
    "Independent",
    "Integral",
    "Mesh",
    "Parameter",
    "Path",
    "Phase",
    "PhaseSolution",
    "Phases",
    "Problem",
    "Solution",
    "State",
    "interp",
    "scalar",
    "vector",
]
