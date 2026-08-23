"""

Collect user-defined functions and deduce their derivatives structures.

"""

# standard library
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np

# package imports
from .input_args import (
    ContinuousHessianArg,
    ContinuousJacobianArg,
    DiscreteHessianArg,
    DiscreteJacobianArg,
    ObjectiveGradientArg,
    ObjectiveHessianArg,
    ProblemFunctions,
)
from .structure import DVStructure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard library imports
    from collections.abc import Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss


class MirroredHessianPairWarning(FutureWarning):
    """A user-defined Hessian callback set both orders of one variable pair.

    Each unordered pair of variables is one second partial derivative and must be
    supplied exactly once, in either order. Both orders currently assemble as two
    entries that are summed; from 0.3.0 this raises `ValueError`.
    """


def _warn_mirrored_pairs(
    entries: dict[Any, Any],
    what: str,
    *,
    context: bool,
) -> None:
    """Warn when a Hessian structure sets both orders of one variable pair.

    The two orders assemble as mirrored coordinates that the structure fold sums. That
    is the right answer for a user who split one derivative across the two keys and
    the wrong answer (doubled) for a user who supplied both triangles of a symmetric
    Hessian, and nothing downstream can tell which was meant -- so the ambiguity is
    refused, by a warning in 0.2.x and an error from 0.3.0. The behavior in 0.2.x is
    unchanged so that the first kind of user is not broken by a patch release.

    Parameters
    ----------
    entries : dict
        The user's Hessian dictionary, keyed by ``(context, key1, key2)`` when
        ``context`` is true (continuous and discrete Hessians) or ``(key1, key2)``
        for the objective Hessian.
    what : str
        Which Hessian, for the message.
    context : bool
        Whether the keys carry a leading function-index element.
    """
    seen: dict[Any, tuple[Any, Any]] = {}
    for key, value in entries.items():
        head, key1, key2 = key if context else (None, *key)
        if key1 == key2:
            continue
        canonical = (head, tuple(sorted((key1, key2))))
        if canonical not in seen:
            seen[canonical] = (key, value)
            continue
        first, first_value = seen[canonical]
        try:
            same_values = bool(np.allclose(first_value, value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            same_values = False
        if same_values:
            diagnosis = (
                "The two entries have equal values, which looks like both triangles of a "
                "symmetric Hessian: the entries are summed, doubling the term. Remove one."
            )
        else:
            diagnosis = (
                "The two entries are summed. If that is intended, combine them into a single "
                "entry; if not, remove one."
            )
        msg = (
            f"The {what} sets both {first} and {key}, which are the same second derivative. "
            f"{diagnosis} Supplying both orders of a pair will raise ValueError in 0.3.0."
        )
        # stacklevel: this helper <- make_user_functions <- solver.solve <- Problem.solve
        # <- the user's call
        warnings.warn(msg, MirroredHessianPairWarning, stacklevel=5)


def make_user_functions(
    problem: yapss.Problem,
    z0: NDArray[np.float64],
    tau_u: Sequence[NDArray[np.float64]],
) -> ProblemFunctions:
    """
    Assemble a ProblemFunctions object with the required functions and derivative structures.

    Parameters
    ----------
    problem : yapss.Problem
        The problem instance containing user-defined functions and problem settings.
    z0 : np.ndarray
        Initial guess for the decision variables.

    Returns
    -------
    ProblemFunctions
        An object containing the user-defined functions and their derivative structures.

    Raises
    ------
    ValueError
        If any required user-defined function is not provided.
    """
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    dv.z[:] = z0

    # objective gradient
    objective_gradient_arg = ObjectiveGradientArg(problem, dv)
    if problem.functions.objective_gradient is not None:
        problem.functions.objective_gradient(objective_gradient_arg)
    else:
        msg = "'functions.objective_gradient' function is required for 'user' method."
        raise ValueError(msg)
    objective_gradient_structure = tuple(objective_gradient_arg.gradient)

    # discrete Jacobian
    discrete_jacobian_arg = DiscreteJacobianArg(problem, dv)
    if problem.nd > 0:
        if problem.functions.discrete_jacobian is not None:
            problem.functions.discrete_jacobian(discrete_jacobian_arg)
        else:
            msg = "'functions.discrete_jacobian' function is required for 'user' method."
            raise ValueError(msg)
        discrete_jacobian_structure = tuple(discrete_jacobian_arg.jacobian.keys())
    else:
        discrete_jacobian_structure = None

    # continuous Jacobian
    continuous_jacobian_arg = ContinuousJacobianArg(
        problem,
        dv=dv,
        dtype=np.float64,
        tau_u=tau_u,
    )
    continuous_jacobian_arg._sync(z0)
    if problem.functions.continuous_jacobian is not None:
        problem.functions.continuous_jacobian(continuous_jacobian_arg)
    else:
        msg = "'functions.continuous_jacobian' function is required for 'user' method."
        raise ValueError(msg)
    # Extract the jacobian structure from the result
    cjs = [tuple(continuous_jacobian_arg.phase[p].jacobian.keys()) for p in range(problem.np)]
    continuous_jacobian_structure = tuple(cjs)

    objective_hessian_structure = None
    discrete_hessian_structure = None
    continuous_hessian_structure = None

    if problem.derivatives.order == "second":
        # objective hessian
        if problem.functions.objective_hessian is not None:
            objective_hessian_arg = ObjectiveHessianArg(problem, dv)
        else:
            msg = (
                "'functions.objective_hessian' function is required for 'user' method when "
                "'derivatives.order' option is set to 'second'."
            )
            raise ValueError(msg)
        problem.functions.objective_hessian(objective_hessian_arg)
        _warn_mirrored_pairs(objective_hessian_arg.hessian, "objective Hessian", context=False)
        objective_hessian_structure = tuple(objective_hessian_arg.hessian)

        # discrete hessian
        if problem.nd > 0:
            if problem.functions.discrete_hessian is not None:
                discrete_hessian_arg = DiscreteHessianArg(problem, dv)
            else:
                msg = (
                    "'functions.discrete_hessian' function is required for 'user' method when "
                    "'derivatives.order' option is set to 'second'."
                )
                raise ValueError(msg)
            problem.functions.discrete_hessian(discrete_hessian_arg)
            _warn_mirrored_pairs(discrete_hessian_arg.hessian, "discrete Hessian", context=True)
            discrete_hessian_structure = tuple(discrete_hessian_arg.hessian)
        else:
            discrete_hessian_structure = None

        # continuous hessian
        continuous_hessian_arg = ContinuousHessianArg(
            problem,
            dv=dv,
            dtype=np.float64,
            tau_u=tau_u,
        )
        continuous_hessian_arg._sync(z0)
        if problem.functions.continuous_hessian is not None:
            problem.functions.continuous_hessian(continuous_hessian_arg)
        else:
            msg = (
                "'functions.continuous_hessian' function is required for 'user' method when "
                "'derivatives.order' option is set to 'second'."
            )
            raise ValueError(msg)
        for p in range(problem.np):
            _warn_mirrored_pairs(
                continuous_hessian_arg.phase[p].hessian,
                f"continuous Hessian of phase {p}",
                context=True,
            )
        chs = [tuple(continuous_hessian_arg.phase[p].hessian) for p in range(problem.np)]
        continuous_hessian_structure = tuple(chs)

    return ProblemFunctions(
        objective=problem.functions.objective,
        objective_gradient=problem.functions.objective_gradient,
        objective_gradient_structure=objective_gradient_structure,
        objective_hessian=(
            None if problem.derivatives.order != "second" else problem.functions.objective_hessian
        ),
        objective_hessian_structure=objective_hessian_structure,
        # discrete
        discrete=problem.functions.discrete,
        discrete_jacobian=problem.functions.discrete_jacobian,
        discrete_jacobian_structure=discrete_jacobian_structure,
        discrete_hessian=(
            None if problem.derivatives.order != "second" else problem.functions.discrete_hessian
        ),
        discrete_hessian_structure=discrete_hessian_structure,
        # continuous
        continuous=problem.functions.continuous,
        continuous_jacobian=problem.functions.continuous_jacobian,
        continuous_jacobian_structure=continuous_jacobian_structure,
        continuous_hessian=(
            None if problem.derivatives.order != "second" else problem.functions.continuous_hessian
        ),
        continuous_hessian_structure=continuous_hessian_structure,
    )
