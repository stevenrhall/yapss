"""Refuse a starting point at which the NLP functions or first derivatives are not finite.

Ipopt checks the objective and constraint values for NaN and Inf, but by default not the
derivative matrices, and its initialization factors a linear system built from the
constraint Jacobian before it ever reads the constraint values: the least-squares
estimate of the starting multipliers requests only the Jacobians and the objective
gradient. A non-finite Jacobian entry therefore reaches the linear solver, whose behavior
on NaN is undefined; for some sparsity patterns MUMPS crashes the process with no Python
traceback. YAPSS sets Ipopt's ``check_derivatives_for_naninf`` by default to stop that at
every iterate, but Ipopt's message names only a matrix. This module evaluates the NLP once
at the starting point, before Ipopt is created, and reports each non-finite entry by the
problem-level quantity it belongs to.

The Hessian is not checked here: it depends on multipliers that do not exist yet, and a
non-finite Hessian with finite first derivatives is left to Ipopt's own check.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, TypeVar, assert_never

import numpy as np

from .structure import nlp_constraint_keys, nlp_variable_keys

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    import yapss

    from .nlp import NLP
    from .types_ import CFViewName, DVViewName

__all__ = ["check_initial_point"]

K = TypeVar("K", bound=Hashable)

_MAX_REPORTED = 12
"""Number of offending quantities listed before the rest are summarized as a count."""


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


def check_initial_point(problem: yapss.Problem, nlp: NLP, z0: NDArray[np.float64]) -> None:
    """Raise if the NLP objective, constraints, or first derivatives are not finite at z0.

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
        If any value is NaN or infinite. The message lists the problem-level quantities
        involved.
    """
    objective = float(nlp.objective(z0))
    constraints = np.array(nlp.constraints(z0), dtype=np.float64)
    gradient = np.array(nlp.gradient(z0), dtype=np.float64)
    jacobian = np.array(nlp.jacobian(z0), dtype=np.float64)

    if (
        np.isfinite(objective)
        and np.all(np.isfinite(constraints))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(jacobian))
    ):
        return

    variable_labels = _labels(nlp_variable_keys(problem), _variable_label)
    constraint_labels = _labels(nlp_constraint_keys(problem), _constraint_label)
    lines: list[str] = []

    if not np.isfinite(objective):
        lines.append(f"the objective is {_describe(np.array([objective]))}")

    bad_constraint_labels: set[str] = set()
    for label, count, total, what in _group(constraint_labels, constraints):
        bad_constraint_labels.add(label)
        lines.append(f"{label} is {what} in {_entries(count, total)}")

    # The derivative of a value that is already non-finite is expected to be non-finite
    # too, so report derivatives only for values that are themselves finite: those are
    # the informative ones, such as sqrt evaluated exactly at zero.
    rows, cols = nlp.jacobianstructure()
    pairs = [(constraint_labels[r], variable_labels[c]) for r, c in zip(rows, cols, strict=True)]
    for (row_label, column_label), count, total, what in _group(pairs, jacobian):
        if row_label not in bad_constraint_labels:
            lines.append(
                f"the derivative of {row_label} with respect to {column_label} is {what} "
                f"in {_entries(count, total)}"
            )

    if np.isfinite(objective):
        for label, count, total, what in _group(variable_labels, gradient):
            lines.append(
                f"the derivative of the objective with respect to {label} is {what} in "
                f"{_entries(count, total)}"
            )

    shown = lines[:_MAX_REPORTED]
    if len(lines) > _MAX_REPORTED:
        shown.append(f"... and {len(lines) - _MAX_REPORTED} more")
    detail = "\n".join(f"  - {line}" for line in shown)
    msg = (
        "The problem functions are not finite at the initial guess, so Ipopt cannot "
        f"start:\n\n{detail}\n\n"
        "Counts are over the entries of the constraint vector and its Jacobian in the "
        "nonlinear program built from the initial guess. A common cause is a function "
        "evaluated outside its domain there, such as the square root or logarithm of a "
        "negative number, or a division by zero. Check the initial guess, or define the "
        "function at those points."
    )
    raise ValueError(msg)
