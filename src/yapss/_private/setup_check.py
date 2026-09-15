"""Check the user's callbacks, and the NLP's first derivatives, before Ipopt starts.

Two stages run in `solver.solve`, after the derivative setup and before Ipopt is created.

`check_callbacks` calls the objective, discrete, and continuous callbacks once with floats at
the initial guess, and the continuous callback once more on every evaluation point of each
phase but the last, in reverse order. From those two calls it reports, in terms of the user's
code:

- an output row that was never assigned, as an `UnsetOutputWarning` at the callback's ``def``
  line (an unassigned row is zero, which is rarely intended);
- a value that is NaN or infinite;
- a continuous function that is not pointwise -- an output at one point that depends on the
  inputs at other points (``t[0]``, ``len``, ``mean``, ``cumsum``, ...). ``"auto"`` traces the
  function at one point while the numeric methods pass every point, so such a function gives
  different answers under different methods. The check is deliberately reasonable rather than
  complete. It sees only what the guess reveals: nothing about an input that is constant along
  the guess, and not a reduction the subset happens to leave unchanged (``max`` of data that is
  largest at the first point). Differences below its tolerance are accepted; the tolerance is
  set well above the rounding noise of correct code (batched BLAS inside an interpolator was
  measured at 1.4e-12 relative).

It runs after the derivative setup rather than before it so that the setup's own, more
specific errors come first -- under ``"auto"``, a Python ``if`` on a symbol raises a
`TypeError` naming `yapss.math.where` during the trace, where a float call would raise NumPy's
"truth value is ambiguous".

`check_derivatives` evaluates the NLP's objective gradient and constraint Jacobian at the
starting point. Ipopt checks function values for NaN and Inf, but by default not the derivative
matrices, and its initialization factors a linear system built from the constraint Jacobian
before it reads the constraint values; a non-finite Jacobian entry therefore reaches the linear
solver, whose behavior on NaN is undefined (for some sparsity patterns MUMPS crashes the process
with no Python traceback). YAPSS sets Ipopt's ``check_derivatives_for_naninf`` to stop that at
every iterate, but Ipopt's message names only a matrix; this stage names the problem-level
quantities. The values themselves need no NLP-level check: every constraint row is built from
the decision variables, which the initial guess keeps finite, and from the callback outputs,
which `check_callbacks` has checked. The Hessian is not checked: it depends on multipliers that
do not exist yet, and a non-finite Hessian with finite first derivatives is left to Ipopt.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, TypeVar, assert_never, cast

import numpy as np

from .input_args import ContinuousArg, DiscreteArg, ObjectiveArg
from .structure import get_nlp_dv_structure, nlp_constraint_keys, nlp_variable_keys

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    import yapss

    from .input_args import ContinuousFunctionFloat, DiscreteFunctionFloat, ObjectiveFunctionFloat
    from .mesh import Mesh
    from .nlp import NLP
    from .structure import DVStructure
    from .types_ import CFViewName, DVViewName

__all__ = ["UnsetOutputWarning", "check_callbacks", "check_derivatives"]

K = TypeVar("K", bound=Hashable)

_MAX_REPORTED = 12
"""Number of offending quantities listed before the rest are summarized as a count."""

POINTWISE_RTOL = 1e-7
"""Relative tolerance of the pointwise check."""

POINTWISE_ATOL = 1e-10
"""Absolute tolerance of the pointwise check, in units of the output's scale factor."""

OUTPUTS = ("dynamics", "integrand", "path")


class UnsetOutputWarning(UserWarning):
    """A callback did not assign an output row at the initial guess, so the row is zero.

    Private until the YAPSS warning hierarchy exists; filterable as a `UserWarning`.
    """


def _variable_label(group: tuple[int, DVViewName, int]) -> str:
    """Name the problem-level variable behind a (phase, view, component) group."""
    p, view, i = group
    match view:
        case "x" | "xs":
            label = f"phase {p} state[{i}]"
        case "u":
            label = f"phase {p} control[{i}]"
        case "q":
            label = f"phase {p} integral[{i}]"
        case "t0":
            label = f"phase {p} initial time"
        case "tf":
            label = f"phase {p} final time"
        case "s":
            label = f"parameter[{i}]"
        case _:
            assert_never(view)
    return label


def _constraint_label(group: tuple[int, CFViewName, int]) -> str:
    """Name the problem-level quantity behind a (phase, view, component) group."""
    p, view, i = group
    match view:
        case "defect":
            label = f"phase {p} dynamics[{i}]"
        case "lg_defect":
            label = f"phase {p} dynamics[{i}] (end-of-segment quadrature)"
        case "path":
            label = f"phase {p} path[{i}]"
        case "integral":
            label = f"phase {p} integral[{i}] (integrand)"
        case "duration":
            label = f"phase {p} duration"
        case "discrete":
            label = f"discrete[{i}]"
        case _:
            assert_never(view)
    return label


def _labels(keys: NDArray[np.object_], label: Callable[[Any], str]) -> list[str]:
    """Label every entry by its (phase, view, component) group, formatting each group once.

    The keys come from an object array, so they are untyped here; `nlp_variable_keys` and
    `nlp_constraint_keys` guarantee each is ``(phase, view, component, position)``.
    """
    cache: dict[tuple[Any, ...], str] = {}
    labels = []
    for p, view, i, _ in keys:
        group = (p, view, i)
        if group not in cache:
            cache[group] = label(group)
        labels.append(cache[group])
    return labels


def _describe(values: NDArray[np.float64]) -> str:
    """Say whether the non-finite values in a group are NaN, infinite, or both."""
    has_nan = bool(np.any(np.isnan(values)))
    has_inf = bool(np.any(np.isinf(values)))
    if has_nan and has_inf:
        return "NaN or infinite"
    return "NaN" if has_nan else "infinite"


def _group(
    keys: Sequence[K],
    values: NDArray[np.float64],
) -> list[tuple[K, int, int, str]]:
    """Group entries by key: (key, non-finite count, total count, description).

    Only keys with at least one non-finite entry are returned, in first-seen order.
    """
    totals: dict[K, int] = defaultdict(int)
    bad: dict[K, list[float]] = defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        totals[key] += 1
        if not np.isfinite(value):
            bad[key].append(float(value))
    return [
        (key, len(bad_values), totals[key], _describe(np.array(bad_values)))
        for key, bad_values in bad.items()
    ]


def _entries(count: int, total: int) -> str:
    """Phrase a count of non-finite entries out of a total."""
    return "its one entry" if total == 1 else f"{count} of {total} entries"


def _points(count: int, total: int) -> str:
    """Phrase a count of non-finite values out of the points of a row."""
    return "its one point" if total == 1 else f"{count} of {total} points"


def _callback_location(function: Callable[..., Any]) -> str:
    """Name a callback and, when it has one, the file and line of its ``def``."""
    code = getattr(function, "__code__", None)
    name = getattr(function, "__qualname__", repr(function))
    return name if code is None else f"{name} ({code.co_filename}, line {code.co_firstlineno})"


def _warn_unset(function: Callable[..., Any], attribute: str, output: str) -> None:
    """Warn that ``functions.<attribute>`` never assigned ``output``, at its ``def`` line."""
    message = (
        f"functions.{attribute} never assigned {output} at the initial guess; an output that "
        f"is not assigned is zero. Assign it, or assign {output} = 0.0 if zero is intended."
    )
    code = getattr(function, "__code__", None)
    if code is None:
        # a callable object: point at the user's solve() call instead
        # (this function <- check_callbacks <- solver.solve <- Problem.solve <- user)
        warnings.warn(message, UnsetOutputWarning, stacklevel=5)
    else:
        warnings.warn_explicit(message, UnsetOutputWarning, code.co_filename, code.co_firstlineno)


def _list(lines: list[str]) -> str:
    shown = lines[:_MAX_REPORTED]
    if len(lines) > _MAX_REPORTED:
        shown.append(f"... and {len(lines) - _MAX_REPORTED} more")
    return "\n".join(f"  - {line}" for line in shown)


def check_callbacks(problem: yapss.Problem, mesh: Mesh, z0: NDArray[np.float64]) -> None:
    """Warn about unassigned output rows; raise for non-finite or non-pointwise outputs.

    Parameters
    ----------
    problem : yapss.Problem
    mesh : Mesh
        The mesh, for the mesh time of each evaluation point.
    z0 : NDArray
        The NLP starting point constructed from the initial guess.

    Raises
    ------
    ValueError
        If an output is NaN or infinite at the initial guess, or the continuous function is
        not pointwise. The message lists every finding.
    """
    functions = problem.functions
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    dv.z[:] = z0
    not_finite: list[str] = []

    objective_function = cast("ObjectiveFunctionFloat", functions.objective)
    objective_arg: ObjectiveArg[np.float64] = ObjectiveArg(problem, dv, np.float64)
    objective_function(objective_arg)
    if not objective_arg._objective_written:
        _warn_unset(objective_function, "objective", "arg.objective")
    objective = np.asarray(objective_arg.objective, dtype=np.float64)
    if not np.all(np.isfinite(objective)):
        not_finite.append(f"the objective is {_describe(objective)}")

    if problem.nd > 0 and functions.discrete is not None:
        discrete_function = cast("DiscreteFunctionFloat", functions.discrete)
        discrete_arg: DiscreteArg[np.float64] = DiscreteArg(problem, dv, np.float64)
        discrete_function(discrete_arg)
        discrete = discrete_arg.discrete
        for row in np.flatnonzero(~discrete.written):
            _warn_unset(discrete_function, "discrete", f"arg.discrete[{row}]")
        constraints = discrete.view(np.ndarray)
        not_finite.extend(
            f"discrete[{row}] is {_describe(constraints[row : row + 1])}"
            for row in np.flatnonzero(~np.isfinite(constraints))
        )

    pointwise: list[str] = []
    failure: Exception | None = None
    if problem.np > 0 and functions.continuous is not None:
        continuous_function = cast("ContinuousFunctionFloat", functions.continuous)
        base: ContinuousArg[np.float64] = ContinuousArg(problem, dv, np.float64, tau_u=mesh.tau_u)
        base._sync(z0)
        continuous_function(base)
        for p, phase in enumerate(base.phase):
            for name in OUTPUTS:
                output = getattr(phase, name)
                values = output.view(np.ndarray)
                for i in range(output.shape[0]):
                    if not output.written[i]:
                        _warn_unset(
                            continuous_function, "continuous", f"arg.phase[{p}].{name}[{i}]"
                        )
                    bad = ~np.isfinite(values[i])
                    if bad.any():
                        not_finite.append(
                            f"phase {p} {name}[{i}] is {_describe(values[i][bad])} at "
                            f"{_points(int(bad.sum()), bad.size)}"
                        )
        pointwise, failure = _pointwise_findings(problem, mesh, z0, base, continuous_function)

    sections: list[str] = []
    if not_finite:
        sections.append(
            "The problem functions are not finite at the initial guess, so Ipopt cannot "
            f"start:\n\n{_list(not_finite)}\n\n"
            "Counts are over the points where each function is evaluated. A common cause is a "
            "function evaluated outside its domain there, such as the square root or logarithm "
            "of a negative number, or a division by zero. Check the initial guess, or define "
            "the function at those points."
        )
    if pointwise:
        sections.append(
            f"The continuous callback {_callback_location(continuous_function)} is not "
            "pointwise: evaluated on all points of each phase but the last, in reverse order, "
            f"its outputs changed:\n\n{_list(pointwise)}\n\n"
            "Each output at a point may depend only on the inputs at that point (time, state, "
            "control) and on the parameters. Common causes are t[0] or other indexing across "
            'points, len, mean, sum, cumsum, and diff. "auto" evaluates the function at one '
            "point while the numeric methods pass every point, so a function that is not "
            "pointwise gives different answers under different methods."
        )
    if sections:
        raise ValueError("\n\n".join(sections)) from failure


def _pointwise_findings(
    problem: yapss.Problem,
    mesh: Mesh,
    z0: NDArray[np.float64],
    base: ContinuousArg[np.float64],
    continuous: Callable[[ContinuousArg[np.float64]], None],
) -> tuple[list[str], Exception | None]:
    """Compare the continuous outputs with a call on all points but the last, reversed."""
    # every evaluation point but the last, in reverse order: the count changes (len, sum,
    # mean), the first point moves (t[0], cumsum, diff), and the subset is not symmetric, so
    # even the mean of a linear guess on symmetric points changes (step 5, measurement m9)
    nodes = [np.arange(len(tau) - 1)[::-1] for tau in mesh.tau_u]
    reversed_arg: ContinuousArg[np.float64] = ContinuousArg(
        problem,
        get_nlp_dv_structure(problem, np.float64),
        np.float64,
        tau_u=mesh.tau_u,
        nodes=nodes,
    )
    reversed_arg._sync(z0)
    try:
        continuous(reversed_arg)
    except Exception as exc:  # noqa: BLE001 -- the base call succeeded; report this as a finding
        return [f"the call raised {type(exc).__name__}: {exc}"], exc

    findings: list[str] = []
    for p, (whole_phase, part_phase) in enumerate(zip(base.phase, reversed_arg.phase, strict=True)):
        selected = nodes[p]
        scale = problem.scale.phase[p]
        magnitude = {
            "dynamics": scale.dynamics,
            "integrand": scale.integral / scale.time,
            "path": scale.path,
        }
        for name in OUTPUTS:
            whole = getattr(whole_phase, name)
            part = getattr(part_phase, name).view(np.ndarray)
            for i in np.flatnonzero(whole.written):
                a = whole.view(np.ndarray)[i, selected]
                b = part[i]
                tolerance = POINTWISE_RTOL * np.maximum(np.abs(a), np.abs(b))
                tolerance += POINTWISE_ATOL * float(magnitude[name][i])
                with np.errstate(invalid="ignore"):
                    same = (a == b) | (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tolerance)
                if not same.all():
                    k = int(np.flatnonzero(~same)[0])
                    findings.append(
                        f"phase {p} {name}[{i}] at point {int(selected[k])}: {a[k]:.10g} when "
                        f"evaluated at every point, {b[k]:.10g} on the points reversed"
                    )
    return findings, None


def check_derivatives(problem: yapss.Problem, nlp: NLP, z0: NDArray[np.float64]) -> None:
    """Raise if the NLP objective gradient or constraint Jacobian is not finite at z0.

    Parameters
    ----------
    problem : yapss.Problem
        The problem being solved.
    nlp : NLP
        The transcribed nonlinear program.
    z0 : NDArray
        The NLP starting point constructed from the initial guess.

    Raises
    ------
    ValueError
        If any derivative is NaN or infinite. The message lists the problem-level quantities
        involved.
    """
    gradient = np.array(nlp.gradient(z0), dtype=np.float64)
    jacobian = np.array(nlp.jacobian(z0), dtype=np.float64)
    if np.all(np.isfinite(gradient)) and np.all(np.isfinite(jacobian)):
        return

    variable_labels = _labels(nlp_variable_keys(problem), _variable_label)
    constraint_labels = _labels(nlp_constraint_keys(problem), _constraint_label)
    lines: list[str] = []
    rows, cols = nlp.jacobianstructure()
    pairs = [(constraint_labels[r], variable_labels[c]) for r, c in zip(rows, cols, strict=True)]
    for (row_label, column_label), count, total, what in _group(pairs, jacobian):
        lines.append(
            f"the derivative of {row_label} with respect to {column_label} is {what} "
            f"in {_entries(count, total)}"
        )
    for label, count, total, what in _group(variable_labels, gradient):
        lines.append(
            f"the derivative of the objective with respect to {label} is {what} in "
            f"{_entries(count, total)}"
        )
    msg = (
        "The derivatives of the problem functions are not finite at the initial guess, so "
        f"Ipopt cannot start:\n\n{_list(lines)}\n\n"
        "Counts are over the entries of the constraint Jacobian and the objective gradient in "
        "the nonlinear program built from the initial guess. A common cause is a function "
        "whose derivative is infinite at a point of the guess, such as the square root of "
        "zero. Check the initial guess, or define the function's derivative at those points."
    )
    raise ValueError(msg)
