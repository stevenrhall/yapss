"""

The solution, read under the names the problem declared.

Every quantity of a phase is a read-only vector of that phase's own classes, so a state is
``ps.state.h`` wherever it is reached, and a helper written against a callback's endpoint
values also accepts a solution's.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .containers import suggest
from .kinds import ReadOnlyRows

if TYPE_CHECKING:
    from .declare import Phase
    from .spec import PhaseSpec, ProblemSpec
    from .vector import Vector

__all__ = ["PhaseSolution", "Solution"]


def _vector(declaration: type[Vector], rows: Any, label: str) -> Any:
    """Return a read-only vector of `declaration` holding `rows`."""
    vector = declaration._new(ReadOnlyRows, label, None)
    vector._fill(rows)
    return vector


class PhaseSolution:
    """One phase of a solution.

    Attributes
    ----------
    time : numpy.ndarray
        The time points every quantity of the phase is given on.
    state, costate, dynamics : Vector
        Arrays over `time`, named by the phase's state class.
    control : Vector
        Arrays over `time`, named by the phase's control class.
    path : Vector
        Arrays over `time`, named by the phase's path class.
    integral : Vector
        One value per integral.
    initial_state, final_state : Vector
        The state at each end of the phase.
    hamiltonian : numpy.ndarray
        The Hamiltonian over `time`.
    """

    __slots__ = (
        "control",
        "costate",
        "dynamics",
        "final_state",
        "final_time",
        "hamiltonian",
        "initial_state",
        "initial_time",
        "integral",
        "mesh",
        "path",
        "state",
        "time",
    )

    def __init__(self, phase: PhaseSpec, data: Any) -> None:
        label = f"phase '{phase.name}' solution"
        state = np.asarray(data.state)
        values: dict[str, Any] = {
            "time": data.time,
            "state": _vector(phase.state, state, f"{label} state"),
            "costate": _vector(phase.state, data.costate, f"{label} costate"),
            "dynamics": _vector(phase.state, data.dynamics, f"{label} dynamics"),
            "control": _vector(phase.control, data.control, f"{label} control"),
            "path": _vector(phase.path, data.path, f"{label} path"),
            "integral": _vector(phase.integral, data.integral, f"{label} integral"),
            "initial_state": _vector(phase.state, state[:, 0], f"{label} initial state"),
            "final_state": _vector(phase.state, state[:, -1], f"{label} final state"),
            "initial_time": data.time[0],
            "final_time": data.time[-1],
            "hamiltonian": data.hamiltonian,
            "mesh": phase.mesh,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """Refuse an unknown name with a suggestion."""
        if name.startswith("_"):
            raise AttributeError(name)
        msg = f"the phase solution has no '{name}'.{suggest(name, self.__slots__)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: a solution is a record of what was solved."""
        del value
        msg = f"a solution is read-only; '{name}' cannot be assigned"
        raise AttributeError(msg)


class Solution:
    """The result of a solve.

    Attributes
    ----------
    objective : float
        The objective value.
    converged : bool
        Whether Ipopt reported a converged solve.
    status : IpoptStatus
        What Ipopt reported.
    parameter, discrete, discrete_multiplier : Vector
        The problem-level values, named by the classes the problem declared.
    """

    __slots__ = (
        "_phases",
        "_spec",
        "converged",
        "discrete",
        "discrete_multiplier",
        "objective",
        "parameter",
        "status",
    )

    def __init__(self, spec: ProblemSpec, legacy: Any) -> None:
        phases = {
            phase.handle: PhaseSolution(phase, legacy.phase[phase.index]) for phase in spec.phases
        }
        values: dict[str, Any] = {
            "_spec": spec,
            "_phases": phases,
            "objective": legacy.objective,
            "converged": legacy.converged,
            "status": legacy.status,
            "parameter": _vector(spec.parameter, legacy.parameter, "parameter"),
            "discrete": _vector(spec.discrete, legacy.discrete, "discrete"),
            "discrete_multiplier": _vector(
                spec.discrete, legacy.discrete_multiplier, "discrete multiplier"
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __getitem__(self, phase: Phase) -> PhaseSolution:
        """Return the solution for `phase`, which is a phase handle."""
        phases: dict[Phase, PhaseSolution] = object.__getattribute__(self, "_phases")
        try:
            return phases[phase]
        except (KeyError, TypeError):
            msg = (
                f"solution[...] takes a phase handle, such as 'problem.phases.<name>'; "
                f"got {phase!r}"
            )
            raise KeyError(msg) from None

    def __getattr__(self, name: str) -> Any:
        """Refuse an unknown name with a suggestion."""
        if name.startswith("_"):
            raise AttributeError(name)
        names = tuple(n for n in self.__slots__ if not n.startswith("_"))
        msg = f"the solution has no '{name}'.{suggest(name, names)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: a solution is a record of what was solved."""
        del value
        msg = f"a solution is read-only; '{name}' cannot be assigned"
        raise AttributeError(msg)

    def __repr__(self) -> str:
        """Return a short representation naming the objective and status."""
        return f"<Solution objective={self.objective!r} converged={self.converged!r}>"
