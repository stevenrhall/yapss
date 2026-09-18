"""

Prototype of the redesigned YAPSS API.

This package is private and carries no compatibility promise. It exists so that the redesigned
API can be run and judged against real problems before any of it is published. Import it
under the name the released API will use::

    from yapss import _next as yapss

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
