"""

Collect user-defined functions and deduce their derivatives structures.

"""

# standard library
from __future__ import annotations

from typing import TYPE_CHECKING

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
    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss


def make_user_functions(problem: yapss.Problem, z0: NDArray[np.float64]) -> ProblemFunctions:
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
    continuous_jacobian_arg = ContinuousJacobianArg(problem, dv=dv, dtype=np.float64)
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
            discrete_hessian_structure = tuple(discrete_hessian_arg.hessian)
        else:
            discrete_hessian_structure = None

        # continuous hessian
        continuous_hessian_arg = ContinuousHessianArg(problem, dv=dv, dtype=np.float64)
        if problem.functions.continuous_hessian is not None:
            problem.functions.continuous_hessian(continuous_hessian_arg)
        else:
            msg = (
                "'functions.continuous_hessian' function is required for 'user' method when "
                "'derivatives.order' option is set to 'second'."
            )
            raise ValueError(msg)
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
