"""The classes a solution is made of, for annotations and for the documentation.

A solution is read through attributes -- ``solution.multiplier``, ``ps.nlp.index`` -- and is never
built by hand, so these classes are rarely named. They are public for the code that does name
one: a function that takes a phase's multipliers, say, annotated ``PhaseMultiplier``. `Solution`
and `PhaseSolution` are here as well as in `yapss` itself.
"""

from ._api.run import Run, Seconds
from ._api.settings import Callback, Settings, SettingsGroup
from ._api.solution import (
    ConstraintIndex,
    Convergence,
    Jacobian,
    NLPIndex,
    NLPRecord,
    NLPScale,
    PhaseIndex,
    PhaseMultiplier,
    PhaseNLP,
    PhasePoint,
    PhaseSolution,
    ProblemConstraintIndex,
    ProblemMultiplier,
    ProblemVariableIndex,
    Solution,
    VariableIndex,
)

__all__ = [
    "Callback",
    "ConstraintIndex",
    "Convergence",
    "Jacobian",
    "NLPIndex",
    "NLPRecord",
    "NLPScale",
    "PhaseIndex",
    "PhaseMultiplier",
    "PhaseNLP",
    "PhasePoint",
    "PhaseSolution",
    "ProblemConstraintIndex",
    "ProblemMultiplier",
    "ProblemVariableIndex",
    "Run",
    "Seconds",
    "Settings",
    "SettingsGroup",
    "Solution",
    "VariableIndex",
]
