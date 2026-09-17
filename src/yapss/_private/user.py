"""

Collect user-defined functions and deduce their derivatives structures.

"""

# standard library
from __future__ import annotations

from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np

# package imports
from . import derivative_keys
from .input_args import (
    ContinuousStore,
    DiscreteHessianArg,
    DiscreteJacobianArg,
    ObjectiveGradientArg,
    ObjectiveHessianArg,
    ProblemFunctions,
    call_callback,
)
from .structure import DVStructure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard library imports
    from collections.abc import Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .types_ import CHS, CJS


def _refuse_mirrored_pairs(
    entries: dict[Any, Any],
    what: str,
    *,
    context: bool,
) -> None:
    """Raise when a Hessian structure sets both orders of one variable pair.

    The two orders would assemble as mirrored coordinates that the structure fold sums. That
    is the right answer for a user who split one derivative across the two keys and the
    wrong answer (doubled) for a user who supplied both triangles of a symmetric Hessian, and
    nothing downstream can tell which was meant, so the ambiguity is refused. It warned with
    `MirroredHessianPairWarning` through 0.2.x and raises from 0.3.0.

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

    Raises
    ------
    ValueError
        If two keys name the same unordered pair of variables.
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
                "symmetric Hessian. Remove one."
            )
        else:
            diagnosis = (
                "If the two entries are parts of one derivative, add them into a single "
                "entry; if not, remove one."
            )
        msg = (
            f"The {what} sets both {first} and {key}, which are the same second derivative. "
            f"Each unordered pair of variables must be set once, in either order. {diagnosis}"
        )
        raise ValueError(msg)


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
        If any required user-defined function is not provided, or a derivative key has
        the right type but a wrong value.
    TypeError
        If a part of a derivative key has the wrong type.
    """
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    dv.z[:] = z0

    # objective gradient
    objective_gradient_arg = ObjectiveGradientArg(problem, dv)
    if problem.functions.objective_gradient is not None:
        call_callback(problem.functions.objective_gradient, objective_gradient_arg)
    else:
        msg = "'functions.objective_gradient' function is required for 'user' method."
        raise ValueError(msg)
    objective_gradient_structure = derivative_keys.objective_gradient_structure(
        problem,
        objective_gradient_arg.gradient,
    )

    # discrete Jacobian
    discrete_jacobian_arg = DiscreteJacobianArg(problem, dv)
    if problem.nd > 0:
        if problem.functions.discrete_jacobian is not None:
            call_callback(problem.functions.discrete_jacobian, discrete_jacobian_arg)
        else:
            msg = "'functions.discrete_jacobian' function is required for 'user' method."
            raise ValueError(msg)
        discrete_jacobian_structure = derivative_keys.discrete_jacobian_structure(
            problem,
            discrete_jacobian_arg.jacobian,
        )
    else:
        discrete_jacobian_structure = None

    # continuous Jacobian: required only when there are phases, as Problem.validate()
    # promises; a parameter-only problem has no continuous function to differentiate.
    continuous_jacobian_structure: CJS = ()
    if problem.np > 0:
        continuous_jacobian_store = ContinuousStore(
            problem,
            dv=dv,
            dtype=np.float64,
            tau_u=tau_u,
        )
        continuous_jacobian_store._sync(z0)
        continuous_jacobian_arg = continuous_jacobian_store.jacobian_arg
        if problem.functions.continuous_jacobian is not None:
            call_callback(problem.functions.continuous_jacobian, continuous_jacobian_arg)
        else:
            msg = "'functions.continuous_jacobian' function is required for 'user' method."
            raise ValueError(msg)
        # Extract the jacobian structure from the result
        continuous_jacobian_structure = tuple(
            derivative_keys.continuous_jacobian_structure(
                problem,
                p,
                continuous_jacobian_arg.phase[p].jacobian,
            )
            for p in range(problem.np)
        )

    objective_hessian_structure = None
    discrete_hessian_structure = None
    continuous_hessian_structure: CHS | None = None

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
        call_callback(problem.functions.objective_hessian, objective_hessian_arg)
        objective_hessian_structure = derivative_keys.objective_hessian_structure(
            problem,
            objective_hessian_arg.hessian,
        )
        _refuse_mirrored_pairs(
            dict(
                zip(
                    objective_hessian_structure, objective_hessian_arg.hessian.values(), strict=True
                )
            ),
            "objective Hessian",
            context=False,
        )

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
            call_callback(problem.functions.discrete_hessian, discrete_hessian_arg)
            discrete_hessian_structure = derivative_keys.discrete_hessian_structure(
                problem,
                discrete_hessian_arg.hessian,
            )
            _refuse_mirrored_pairs(
                dict(
                    zip(
                        discrete_hessian_structure,
                        discrete_hessian_arg.hessian.values(),
                        strict=True,
                    ),
                ),
                "discrete Hessian",
                context=True,
            )
        else:
            discrete_hessian_structure = None

        # continuous hessian, again only when there are phases
        continuous_hessian_structure = ()
        if problem.np > 0:
            continuous_hessian_store = ContinuousStore(
                problem,
                dv=dv,
                dtype=np.float64,
                tau_u=tau_u,
            )
            continuous_hessian_store._sync(z0)
            continuous_hessian_arg = continuous_hessian_store.hessian_arg
            if problem.functions.continuous_hessian is not None:
                call_callback(problem.functions.continuous_hessian, continuous_hessian_arg)
            else:
                msg = (
                    "'functions.continuous_hessian' function is required for 'user' method "
                    "when 'derivatives.order' option is set to 'second'."
                )
                raise ValueError(msg)
            chs = []
            for p in range(problem.np):
                entries = continuous_hessian_arg.phase[p].hessian
                chs_phase = derivative_keys.continuous_hessian_structure(problem, p, entries)
                _refuse_mirrored_pairs(
                    dict(zip(chs_phase, entries.values(), strict=True)),
                    f"continuous Hessian of phase {p}",
                    context=True,
                )
                chs.append(chs_phase)
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
