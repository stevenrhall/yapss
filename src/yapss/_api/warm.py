"""

Guess a problem from a solution: a warm start (spec 6.1).

`Problem.guess_from_solution` writes the problem's guess aspects from a solution, as ordinary
guesses that can be edited afterwards: each phase's time guess from the span it was solved over,
its states and controls as `yapss.interp` samples on the points the solution reports, and its
integrals and the problem's parameters as the numbers they took.

Everything must match. Phases match by name; within each, every state, control and integral by
name and block size; and the parameters too. A mismatch is either a mistake -- a renamed state,
the wrong solution -- or deliberate growth, and YAPSS cannot tell which, so every mismatch is
reported and nothing is written. The per-phase form pairs one phase of the solution with one of
the problem, which covers subsets and renames, and leaves the rest of the problem as it is.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import aspects_of
from .sampled import interp

if TYPE_CHECKING:
    from .problem import Problem
    from .solution import PhaseSolution, Solution

__all__ = ["guess_from_solution"]

ROLES = ("state", "control", "integral")
"""The vectors of a phase that are guessed, and matched by name and size."""


def guess_from_solution(
    problem: Problem[Any, Any, Any],
    solution: Solution[Any, Any],
    solution_phase: Any = None,
    guess_phase: Any = None,
) -> None:
    """Write `problem`'s guesses from `solution`. See `Problem.guess_from_solution`."""
    if (solution_phase is None) != (guess_phase is None):
        msg = (
            "guess_from_solution pairs one phase with another: give both solution_phase= and "
            "guess_phase=, or neither to match every phase by name"
        )
        raise TypeError(msg)
    mismatches: list[str] = []
    if guess_phase is not None:
        pairs = [(solution[solution_phase], _own_phase(problem, guess_phase))]
        parameters = False
    else:
        pairs, mismatches = _pair_by_name(problem, solution)
        parameters = True
    mismatches += [m for ps, ph in pairs for m in _phase_mismatches(ps, ph)]
    if parameters:
        mismatches += _mismatches(
            "parameters", _shape(solution.parameter), _shape_of(problem.parameter)
        )
    if mismatches:
        _refuse(mismatches)
    for ps, ph in pairs:
        _write_phase(ps, ph)
    if parameters:
        _write_scalars(solution.parameter, problem.parameter)


def _own_phase(problem: Problem[Any, Any, Any], phase: Any) -> Any:
    """Return `problem`'s phase for `phase`, a handle of it or its name."""
    for ph in problem.phases:
        if ph is phase or (isinstance(phase, str) and ph._name == phase):
            return ph
    msg = f"guess_phase= must be a phase of this problem, or its name; got {phase!r}"
    raise KeyError(msg)


def _pair_by_name(
    problem: Problem[Any, Any, Any], solution: Solution[Any, Any]
) -> tuple[list[tuple[PhaseSolution, Any]], list[str]]:
    """Pair every phase of the problem with the solution's phase of the same name."""
    names = object.__getattribute__(solution, "_names")
    own = {ph._name: ph for ph in problem.phases}
    mismatches = [
        f"phase '{n}' is in the solution but not the problem" for n in names if n not in own
    ]
    mismatches += [
        f"phase '{n}' is in the problem but not the solution" for n in own if n not in names
    ]
    pairs = [(solution[name], own[name]) for name in names if name in own]
    return pairs, mismatches


def _phase_mismatches(ps: PhaseSolution, ph: Any) -> list[str]:
    """List every state, control and integral of `ph` that `ps` does not match."""
    found = []
    for role in ROLES:
        found += _mismatches(
            f"phase '{ph._name}' {role}",
            _shape(getattr(ps, role)),
            _shape_of(getattr(ph, role)),
        )
    return found


def _shape(vector: Any) -> dict[str, int | None]:
    """Return each field of a solution's vector with its block size, None for a scalar."""
    meta = type(vector)._meta
    return {name: meta[name].size for name in type(vector)._fields}


def _shape_of(fields: Any) -> dict[str, int | None]:
    """Return each field of a problem's declared vector with its block size."""
    return _shape(aspects_of(fields).guess)


def _mismatches(
    label: str, solved: dict[str, int | None], declared: dict[str, int | None]
) -> list[str]:
    found = []
    for name, size in solved.items():
        if name not in declared:
            found.append(f"{label} '{name}' is in the solution but not the problem")
        elif size != declared[name]:
            found.append(
                f"{label} '{name}' has {_rows(size)} in the solution and "
                f"{_rows(declared[name])} in the problem"
            )
    found += [
        f"{label} '{name}' is in the problem but not the solution"
        for name in declared
        if name not in solved
    ]
    return found


def _rows(size: int | None) -> str:
    return "one row" if size is None else f"{size} rows"


def _refuse(mismatches: list[str]) -> None:
    msg = (
        "the solution does not match the problem, so no guess was written:\n  "
        + "\n  ".join(mismatches)
        + "\nFor a subset or a renamed phase, pair phases with solution_phase= and "
        "guess_phase=; for growth within a phase, guess the new fields with yapss.interp."
    )
    raise ValueError(msg)


def _write_phase(ps: PhaseSolution, ph: Any) -> None:
    """Write one phase's time, state, control and integral guesses from its solution."""
    time = np.asarray(object.__getattribute__(ps, "_points"), dtype=float)
    getattr(ph, ph._independent).guess = (float(time[0]), float(time[-1]))
    for role in ("state", "control"):
        solved = getattr(ps, role)
        for name in type(solved)._fields:
            values = np.asarray(getattr(solved, name), dtype=float)
            settings = getattr(getattr(ph, role), name)
            if type(solved)._meta[name].size is None:
                settings.guess = interp(time, values)
            else:
                settings.guess[:] = interp(time, values)
    _write_scalars(ps.integral, ph.integral)


def _write_scalars(solved: Any, fields: Any) -> None:
    """Write one-number guesses -- integrals, parameters -- from the values they took."""
    for name in type(solved)._fields:
        value = getattr(solved, name)
        settings = getattr(fields, name)
        if type(solved)._meta[name].size is None:
            settings.guess = float(value)
        else:
            settings.guess[:] = [float(v) for v in np.asarray(value, dtype=float)]
