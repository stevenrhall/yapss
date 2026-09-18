"""

The snapshot a solve is run from.

`ProblemSpec` holds everything the transcription needs and nothing else: the declarations, the
bounds, the guess, the mesh, the callbacks, and the settings. It is taken once per solve, so a
problem edited afterwards -- during a continuation sweep, say -- never alters what an earlier
solution was computed from.

It is deliberately *not* expressed in terms of the callback protocol of any one front end: it
carries the user's own per-phase callbacks, and the bridge to the solver builds whatever
wrappers that solver wants. That is what lets the front end and the transcription be developed
apart.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .sampled import coverage_complaint

if TYPE_CHECKING:
    from collections.abc import Callable

    from .declare import Phase
    from .mesh import Mesh
    from .problem import Problem
    from .vector import Vector

__all__ = ["PhaseSpec", "ProblemSpec", "snapshot", "validate_problem"]

Sense = Literal["minimize", "maximize"]
Method = Literal["lgl", "lgr", "lg"]
DerivativeMethod = Literal["auto", "central-difference", "central-difference-full", "user"]
Order = Literal["first", "second"]


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One phase, as a solve sees it."""

    handle: Phase
    name: str
    index: int
    state: type[Vector]
    control: type[Vector]
    path: type[Vector]
    integral: type[Vector]
    continuous: Callable[..., Any]
    state_bounds: dict[str, Any]
    state_initial: dict[str, Any]
    state_final: dict[str, Any]
    state_guess: dict[str, Any]
    control_bounds: dict[str, Any]
    control_guess: dict[str, Any]
    path_bounds: dict[str, Any]
    integral_bounds: dict[str, Any]
    integral_guess: dict[str, Any]
    state_scale: dict[str, Any]
    state_defect_scale: dict[str, Any]
    control_scale: dict[str, Any]
    path_scale: dict[str, Any]
    integral_scale: dict[str, Any]
    time_initial: tuple[float, float]
    time_final: tuple[float, float]
    time_guess: tuple[float, float]
    time_scale: float
    mesh: Mesh


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A whole problem, as a solve sees it."""

    name: str
    phases: tuple[PhaseSpec, ...]
    discrete: type[Vector]
    discrete_bounds: dict[str, Any]
    discrete_function: Callable[..., Any] | None
    parameter: type[Vector]
    parameter_bounds: dict[str, Any]
    parameter_guess: dict[str, Any]
    parameter_scale: dict[str, Any]
    discrete_scale: dict[str, Any]
    objective_function: Callable[..., Any]
    objective_scale: float
    sense: Sense
    method: Method
    derivative_method: DerivativeMethod
    derivative_order: Order
    ipopt_options: dict[str, Any]
    catch_keyboard_interrupt: bool


def _values(vector: Vector) -> dict[str, tuple[Any, ...]]:
    """Return the stored elements of each field, one per row, filling in the kind's default."""
    return {name: vector._elements(name) for name in vector._fields}


def validate_problem(problem: Problem) -> None:
    """Check that a problem is complete. See `Problem.validate`."""
    complaints: list[str] = []
    for phase in problem.phases:
        label = f"phase '{phase.name}'"
        if phase._continuous is None:
            complaints.append(f"{label} has no continuous callback")
        if phase.time.guess is None:
            complaints.append(f"{label} has no time guess; set 'ph.time.guess = (t0, tf)'")
        complaints.extend(_unbounded(phase.path.bounds, f"{label} path"))
        if phase.time.guess is not None:
            for what in ("state", "control"):
                aspect = getattr(phase, what).guess
                complaints.extend(_uncovered(aspect, phase.time.guess, f"{label} {what} guess"))
    if problem._objective_function is None:
        complaints.append("the problem has no objective callback")
    if problem._discrete_class._fields and problem._discrete_function is None:
        complaints.append("discrete constraints are declared but there is no discrete callback")
    complaints.extend(_unbounded(problem.discrete.bounds, "discrete constraint"))
    if complaints:
        msg = "the problem is incomplete:\n  " + "\n  ".join(complaints)
        raise ValueError(msg)


def _uncovered(guess: Vector, time_guess: tuple[float, float], label: str) -> list[str]:
    """Return a complaint for every sampled guess that does not reach far enough into the phase.

    Samples that fall well short of the phase are a mistake -- the wrong units, the wrong
    phase, or a time guess moved away from its samples -- rather than a choice, so they are
    refused. Samples that fall a little short simply hold their end values.
    """
    complaints = []
    for name in guess._fields:
        element = guess._elements(name)[0]
        if element[0] != "sampled":
            continue
        complaint = coverage_complaint(element[1], time_guess, label, name)
        if complaint is not None:
            complaints.append(complaint)
    return complaints


def _unbounded(bounds: Vector, label: str) -> list[str]:
    """Return a complaint for every constraint left with no bound.

    A constraint that is declared and never bounded is evaluated at every iteration and then
    ignored, which is a mistake rather than a choice, so it is refused rather than warned about.
    """
    return [
        f"{label} '{name}' has no bound; a declared constraint must be bounded"
        for name in bounds._fields
        if name not in bounds._values
    ]


def snapshot(problem: Problem) -> ProblemSpec:
    """Return a `ProblemSpec` recording `problem` as it stands.

    Parameters
    ----------
    problem : Problem
        The problem to record.

    Returns
    -------
    ProblemSpec
        The snapshot, which no later edit of `problem` can alter.
    """
    phases = tuple(
        PhaseSpec(
            handle=phase,
            name=phase.name,
            index=phase.index,
            state=phase._declaration.state,
            control=phase._declaration.control,
            path=phase._declaration.path,
            integral=phase._declaration.integral,
            continuous=phase._continuous,
            state_bounds=_values(phase.state.bounds),
            state_initial=_values(phase.state.initial),
            state_final=_values(phase.state.final),
            state_guess=_values(phase.state.guess),
            control_bounds=_values(phase.control.bounds),
            control_guess=_values(phase.control.guess),
            path_bounds=_values(phase.path.bounds),
            integral_bounds=_values(phase.integral.bounds),
            integral_guess=_values(phase.integral.guess),
            state_scale=_values(phase.state.scale),
            state_defect_scale=_values(phase.state.defect_scale),
            control_scale=_values(phase.control.scale),
            path_scale=_values(phase.path.scale),
            integral_scale=_values(phase.integral.scale),
            time_initial=phase.time.initial,
            time_final=phase.time.final,
            time_guess=phase.time.guess,
            time_scale=phase.time.scale,
            mesh=phase.mesh,
        )
        for phase in problem.phases
    )
    objective = problem._objective_function
    if objective is None:  # pragma: no cover - validate() has already refused this
        msg = "the problem has no objective callback"
        raise ValueError(msg)
    return ProblemSpec(
        name=problem.name,
        phases=phases,
        discrete=problem._discrete_class,
        discrete_bounds=_values(problem.discrete.bounds),
        discrete_function=problem._discrete_function,
        parameter=problem._parameter_class,
        parameter_bounds=_values(problem.parameter.bounds),
        parameter_guess=_values(problem.parameter.guess),
        parameter_scale=_values(problem.parameter.scale),
        discrete_scale=_values(problem.discrete.scale),
        objective_function=objective,
        objective_scale=problem.objective.scale,
        sense=problem.objective.sense,
        method=problem.method,
        derivative_method=problem.derivatives.method,
        derivative_order=problem.derivatives.order,
        ipopt_options=dict(problem.ipopt_options.get_options()),
        catch_keyboard_interrupt=problem.catch_keyboard_interrupt,
    )
