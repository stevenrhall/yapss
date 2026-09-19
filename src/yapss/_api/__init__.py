"""

The YAPSS API.

This package holds the front end that `yapss` itself exports: `Problem`, the declarations a
problem is written in, and the solution it returns. Nothing imports from here directly --
``import yapss`` reaches all of it -- and the package keeps its own name only until the two
front ends finish changing places. The 0.3.0 API it replaced is `yapss._legacy`.

"""

from .declare import Phases, phase
from .mesh import Mesh
from .problem import Problem
from .sampled import interp
from .solution import PhaseSolution, Solution
from .vector import Empty, Vector, field

__all__ = [
    "Empty",
    "Mesh",
    "PhaseSolution",
    "Phases",
    "Problem",
    "Solution",
    "Vector",
    "field",
    "interp",
    "phase",
]
