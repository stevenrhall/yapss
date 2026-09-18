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
    independent: str
    """The name of the phase's independent variable, which is `time` unless it was renamed."""
    continuous: Callable[..., Any]
    continuous_jacobian: Callable[..., Any] | None
    continuous_hessian: Callable[..., Any] | None
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
    objective_gradient_function: Callable[..., Any] | None
    objective_hessian_function: Callable[..., Any] | None
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
        independent = getattr(phase, phase._independent)
        if independent.guess is None:
            name = phase._independent
            complaints.append(f"{label} has no {name} guess; set 'ph.{name}.guess = (start, end)'")
        complaints.extend(_unbounded(phase.path.bounds, f"{label} path"))
        if independent.guess is not None:
            for what in ("state", "control"):
                aspect = getattr(phase, what).guess
                complaints.extend(_uncovered(aspect, independent.guess, f"{label} {what} guess"))
    if problem._objective_function is None:
        complaints.append("the problem has no objective callback")
    if problem._discrete_class._fields and problem._discrete_function is None:
        complaints.append("discrete constraints are declared but there is no discrete callback")
    complaints.extend(_unbounded(problem.discrete.bounds, "discrete constraint"))
    if problem.derivatives.method == "user":
        complaints.extend(_missing_derivatives(problem))
    if complaints:
        msg = "the problem is incomplete:\n  " + "\n  ".join(complaints)
        raise ValueError(msg)


def _missing_derivatives(problem: Problem) -> list[str]:
    """Return a complaint for every derivative callback the ``"user"`` method needs.

    Every derivative the method needs is registered, *including* the ones that are zero: an
    empty callback says that every entry of it is structurally zero, and leaving it out would
    be indistinguishable from forgetting it. Forgetting it is not caught by the answer, since
    the gradient and the Jacobian still define the same KKT point -- it costs iterations, and
    losing the speed invisibly is the failure worth refusing in the one feature whose purpose
    is speed.
    """
    second = problem.derivatives.order == "second"
    complaints: list[str] = []
    for phase in problem.phases:
        needed = [("continuous_jacobian", "_continuous_jacobian")]
        if second:
            needed.append(("continuous_hessian", "_continuous_hessian"))
        complaints.extend(
            f"phase '{phase.name}' has no {which} callback, which "
            f"'derivatives.method = \"user\"' requires; register it with "
            f"'@ph.register.{which}'"
            for which, attribute in needed
            if getattr(phase, attribute) is None
        )
    needed = [("objective_gradient", "_objective_gradient_function")]
    if second:
        needed.append(("objective_hessian", "_objective_hessian_function"))
    complaints.extend(
        f"the problem has no {which} callback, which "
        f"'derivatives.method = \"user\"' requires; register it with "
        f"'@problem.register.{which}'"
        for which, attribute in needed
        if getattr(problem, attribute) is None
    )
    if problem._discrete_class._fields:
        complaints.append(
            "derivatives supplied by hand do not yet reach the discrete constraints; use "
            "'auto' or a central-difference method for this problem"
        )
    return complaints


def _uncovered(guess: Vector, time_guess: tuple[float, float], label: str) -> list[str]:
    """Return a complaint for every sampled guess that does not reach far enough into the phase.

    Samples that fall well short of the phase are a mistake -- the wrong units, the wrong
    phase, or a time guess moved away from its samples -- rather than a choice, so they are
    refused. Samples that fall a little short simply hold their end values.
    """
    complaints = []
    for name in guess._fields:
        # Every row, not just the first: a block field can be given one sampled guess per row.
        for element in guess._elements(name):
            if element[0] != "sampled":
                continue
            complaint = coverage_complaint(element[1], time_guess, label, name)
            if complaint is not None:
                complaints.append(complaint)
                break
    return complaints


def _unbounded(bounds: Vector, label: str) -> list[str]:
    """Return a complaint for every constraint left with no bound.

    A constraint that is declared and never bounded is evaluated at every iteration and then
    ignored, which is a mistake rather than a choice, so it is refused rather than warned about.
    """
    return [
        f"{label} '{name}' has no bound; a declared constraint must be bounded"
        for name in bounds._fields
        # A field declared with size=0 holds no rows, so there is nothing to bound.
        if name not in bounds._values and bounds._meta[name].rows
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
            independent=phase._independent,
            continuous=phase._continuous,
            continuous_jacobian=phase._continuous_jacobian,
            continuous_hessian=phase._continuous_hessian,
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
            time_initial=getattr(phase, phase._independent).initial,
            time_final=getattr(phase, phase._independent).final,
            time_guess=getattr(phase, phase._independent).guess,
            time_scale=getattr(phase, phase._independent).scale,
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
        objective_gradient_function=problem._objective_gradient_function,
        objective_hessian_function=problem._objective_hessian_function,
        objective_scale=problem.objective.scale,
        sense=problem.objective.sense,
        method=problem.method,
        derivative_method=problem.derivatives.method,
        derivative_order=problem.derivatives.order,
        ipopt_options=dict(problem.ipopt_options.get_options()),
        catch_keyboard_interrupt=problem.catch_keyboard_interrupt,
    )
